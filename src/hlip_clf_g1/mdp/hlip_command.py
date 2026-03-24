"""HLIP command term for CLF-based walking.

Generates reference trajectories for the full body (CoM, pelvis orientation,
swing foot position/orientation, upper body joints) using the Hybrid Linear
Inverted Pendulum capture-point controller and Bezier swing curves.

Maintains a CLF (Control Lyapunov Function) value V and its derivative
Vdot that can be used by reward functions to encourage convergence to the
desired orbit.

Ported from planc/tasks/.../hlip_cmd.py to work with the mjlab framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  wrap_to_pi,
  yaw_quat,
)

from hlip_clf_g1.mdp.clf import CLF
from hlip_clf_g1.mdp.ref_gen import HLIP, bezier_deg, calculate_cur_swing_foot_pos

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# =====================================================================
# Helpers
# =====================================================================

GRAVITY = 9.81


def _euler_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """Extract wrapped Euler XYZ from quaternion. Shape ``(..., 3)``."""
  ex, ey, ez = euler_xyz_from_quat(quat)
  return torch.stack([wrap_to_pi(ex), wrap_to_pi(ey), wrap_to_pi(ez)], dim=-1)


def _to_local_frame(vec: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
  """Rotate a world-frame vector into the yaw-only local frame."""
  return quat_apply(yaw_quat(quat_inv(root_quat)), vec)


def _euler_rates_to_omega(
  eul: torch.Tensor, eul_rates: torch.Tensor
) -> torch.Tensor:
  """Convert ZYX Euler-angle rates to body-frame angular velocity.

  Args:
    eul: Euler angles (roll, pitch, yaw). Shape ``(..., 3)``.
    eul_rates: Euler rates. Shape ``(..., 3)``.

  Returns:
    Angular velocity. Shape ``(..., 3)``.
  """
  _, theta, psi = eul.unbind(-1)
  c_th = torch.cos(theta)
  s_th = torch.sin(theta)
  c_ps = torch.cos(psi)
  s_ps = torch.sin(psi)
  zeros = torch.zeros_like(theta)
  ones = torch.ones_like(theta)

  M = torch.stack(
    [
      torch.stack([c_th * c_ps, s_ps, zeros], dim=-1),
      torch.stack([-c_th * s_ps, c_ps, zeros], dim=-1),
      torch.stack([s_th, zeros, ones], dim=-1),
    ],
    dim=-2,
  )
  return torch.einsum("...ij,...j->...i", M, eul_rates)


# =====================================================================
# HLIP Command Term
# =====================================================================


class HLIPCommandTerm(CommandTerm):
  """HLIP + CLF walking command term.

  Generates a per-env reference trajectory for:
    - CoM position/velocity (3 + 3)
    - Pelvis Euler angles / angular velocity (3 + 3)
    - Swing foot position/velocity in stance-local frame (3 + 3)
    - Swing foot Euler angles / angular velocity (3 + 3)
    - Upper body joint positions/velocities (N + N)

  Also computes CLF V and Vdot at each step. Rewards access these via
  ``self.v`` and ``self.vdot``.

  The actual state is extracted from the robot sensor data and stored in
  ``self.y_act`` / ``self.dy_act`` for use by observations and rewards.
  """

  cfg: "HLIPCommandCfg"

  def __init__(self, cfg: "HLIPCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    # HLIP parameters.
    self.T_ds = cfg.T_ds
    self.z0 = cfg.z0
    self.y_nom = cfg.y_nom
    self.T_min = getattr(cfg, "T_min", getattr(cfg, "gait_half_period", 0.4))
    self.T_max = getattr(cfg, "T_max", getattr(cfg, "gait_half_period", 0.4))
    self.T = torch.empty(self.num_envs, device=self.device).uniform_(self.T_min, self.T_max)

    self.z_sw_max_range = getattr(cfg, "z_sw_max_range", None)
    self.z_sw_max_envs = torch.empty(self.num_envs, device=self.device)
    if self.z_sw_max_range is not None:
      self.z_sw_max_envs.uniform_(self.z_sw_max_range[0], self.z_sw_max_range[1])
    else:
      self.z_sw_max_envs.fill_(cfg.z_sw_max)

    # Resolve foot body indices.
    self._foot_body_ids, _ = self.robot.find_bodies(cfg.foot_body_name)
    assert len(self._foot_body_ids) == 2, (
      f"Expected 2 foot bodies, got {len(self._foot_body_ids)}"
    )

    # Resolve upper body joint indices.
    self._upper_body_joint_ids, self._upper_body_joint_names = (
      self.robot.find_joints(cfg.upper_body_joint_names)
    )
    n_upper = len(self._upper_body_joint_ids)

    # Total number of tracked outputs: 12 (CoM + pelvis + swing + swing_ori) + upper.
    self._num_outputs = 12 + n_upper
    self._yaw_idx = cfg.yaw_idx  # Indices of yaw-like outputs for wrapping.

    # Reference and actual state buffers.
    self.y_out = torch.zeros(
      self.num_envs, self._num_outputs, device=self.device
    )
    self.dy_out = torch.zeros(
      self.num_envs, self._num_outputs, device=self.device
    )
    self.y_act = torch.zeros(
      self.num_envs, self._num_outputs, device=self.device
    )
    self.dy_act = torch.zeros(
      self.num_envs, self._num_outputs, device=self.device
    )

    # Nominal CoM height.
    self.com_z = torch.ones(self.num_envs, device=self.device) * self.z0

    # HLIP controller.
    self.hlip_controller = HLIP(
      grav=GRAVITY,
      z0=self.z0,
      T_ds=self.T_ds,
      T=self.T_min,
      y_nom=self.y_nom,
      device=self.device,
    )

    # CLF.
    step_dt = env.step_dt
    self.clf = CLF(
      n_outputs=self._num_outputs,
      sim_dt=step_dt,
      batch_size=self.num_envs,
      Q_weights=np.array(cfg.Q_weights),
      R_weights=np.array(cfg.R_weights),
      device=self.device,
    )

    # CLF value and derivative.
    self.v = torch.zeros(self.num_envs, device=self.device)
    self.vdot = torch.zeros(self.num_envs, device=self.device)
    self.v_buffer = torch.zeros(self.num_envs, 100, device=self.device)
    self.vdot_buffer = torch.zeros(self.num_envs, 100, device=self.device)

    # Desired velocity from an external source or internally sampled.
    self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)

    # Per-env gait phase tracking.
    self.gait_time = torch.zeros(self.num_envs, device=self.device)
    T_swing = self.T - self.T_ds
    self.full_gait_period = 2.0 * T_swing

    # Per-env stance/swing indices (0=left, 1=right).
    self.stance_idx = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.swing_idx = torch.ones(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Per-env phase variables.
    self.tp = torch.zeros(self.num_envs, device=self.device)  # Normalised gait phase.
    self.phase_var = torch.zeros(self.num_envs, device=self.device)  # Phase within half-gait.
    self.cur_swing_time = torch.zeros(self.num_envs, device=self.device)

    # Stance foot pose at transition (per-env).
    self.stance_foot_pos_0 = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self.stance_foot_ori_quat_0 = torch.zeros(
      self.num_envs, 4, device=self.device
    )
    self.stance_foot_ori_quat_0[:, 0] = 1.0  # Identity quaternion.
    self.stance_foot_ori_0 = torch.zeros(
      self.num_envs, 3, device=self.device
    )

    # Swing foot position in stance-local frame at the start of each half-gait.
    self.swing_foot_pos_0 = torch.zeros(
      self.num_envs, 3, device=self.device
    )

    # Actual stance foot state (updated each tick).
    self.stance_foot_pos = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self.stance_foot_vel = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self.stance_foot_ang_vel = torch.zeros(
      self.num_envs, 3, device=self.device
    )

    # Foot target for observation.
    self.foot_target = torch.zeros(self.num_envs, 3, device=self.device)
    self.ai_foot_target = torch.zeros(self.num_envs, 3, device=self.device)
    self._has_ai_foot_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # Standing / heading control.
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Previous stance idx for transition detection.
    self._prev_stance_idx = self.stance_idx.clone()
    self._swing_elapsed = torch.zeros(self.num_envs, device=self.device)
    self._prev_foot_contact = torch.zeros(
      self.num_envs, 2, dtype=torch.bool, device=self.device
    )

    # Touchdown-based stance switching.
    self._touchdown_enabled = bool(getattr(cfg, "touchdown_switch_enabled", True))
    self._touchdown_sensor_name = str(getattr(cfg, "touchdown_sensor_name", "feet_ground_contact"))
    self._touchdown_min_phase = float(getattr(cfg, "touchdown_min_phase", 0.35))
    self._touchdown_timeout_phase = float(getattr(cfg, "touchdown_timeout_phase", 1.20))

    # Deferred stance foot initialization flag.  During _resample_command the
    # body positions are stale (sim.forward() hasn't run yet), so we defer
    # the stance foot pose read to the first _update_command call.
    self._pending_stance_init = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Metrics.
    self.metrics["v_mean"] = torch.zeros(self.num_envs, device=self.device)

    # MPC configuration/state.
    self._mpc_enabled = bool(getattr(cfg, "mpc_enabled", False))
    self._mpc_horizon = max(int(getattr(cfg, "mpc_horizon", 3)), 1)
    self._mpc_t_candidates = max(int(getattr(cfg, "mpc_t_candidates", 7)), 1)
    self._mpc_t_nom = 0.5 * (self.T_min + self.T_max)
    self._mpc_w_vel = float(getattr(cfg, "mpc_w_vel", 1.0))
    self._mpc_w_time = float(getattr(cfg, "mpc_w_time", 0.2))
    self._mpc_w_foot = float(getattr(cfg, "mpc_w_foot", 0.05))

    x_min, x_max = getattr(cfg, "mpc_foot_target_range_x", (-0.4, 0.8))
    self._mpc_x_min = float(x_min)
    self._mpc_x_max = float(x_max)
    y_min, y_max = self.cfg.foot_target_range_y
    self._mpc_abs_y_min = float(getattr(cfg, "mpc_abs_y_min", y_min))
    self._mpc_abs_y_max = float(getattr(cfg, "mpc_abs_y_max", y_max))
    self._mpc_signed_y_min = float(getattr(cfg, "mpc_signed_y_min", 0.02))
    self._mpc_max_step_length = float(getattr(cfg, "mpc_max_step_length", 0.9))

    fallback_scales = tuple(getattr(cfg, "mpc_fallback_scales", (1.0, 0.75, 0.5, 0.25, 0.0)))
    self._mpc_fallback_scales = tuple(float(s) for s in fallback_scales)

    if self._mpc_t_candidates == 1:
      self._mpc_t_grid = torch.tensor([self._mpc_t_nom], device=self.device)
    else:
      self._mpc_t_grid = torch.linspace(
        self.T_min,
        self.T_max,
        self._mpc_t_candidates,
        device=self.device,
      )

    self._mpc_plan_t = torch.full(
      (self.num_envs, self._mpc_horizon), self._mpc_t_nom, device=self.device
    )
    self._mpc_plan_foothold = torch.zeros(
      self.num_envs, self._mpc_horizon, 3, device=self.device
    )
    self._mpc_active_foot_target = torch.zeros(self.num_envs, 3, device=self.device)
    self._mpc_has_plan = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._mpc_last_cost = torch.zeros(self.num_envs, device=self.device)
    self._mpc_feasible_steps = torch.zeros(self.num_envs, device=self.device)
    self._mpc_fallback_used = torch.zeros(self.num_envs, device=self.device)

    # Heightmap layout for edge detection.
    self._hm_num_x = 0
    self._hm_num_y = 0
    self._hm_edge_threshold = float(getattr(cfg, "mpc_edge_height_threshold", -1.0))
    self._hm_max_stance_height_delta = float(
      getattr(cfg, "mpc_max_stance_height_delta", -1.0)
    )
    if "heightmap" in self._env.scene.sensors:
      hm_pattern = self._env.scene.sensors["heightmap"].cfg.pattern
      hm_res = float(hm_pattern.resolution)
      size_x, size_y = hm_pattern.size
      self._hm_num_x = len(np.arange(-size_x / 2, size_x / 2 + hm_res * 0.5, hm_res))
      self._hm_num_y = len(np.arange(-size_y / 2, size_y / 2 + hm_res * 0.5, hm_res))
      if self._hm_edge_threshold <= 0.0:
        self._hm_edge_threshold = hm_res * math.tan(math.radians(20.0))

  # ------------------------------------------------------------------
  # Properties
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Foot target (kept for observation compatibility)."""
    return self.foot_target

  def set_ai_touchdown_target(self, target_local: torch.Tensor) -> None:
    """Set AI-predicted touchdown target in stance-local frame.

    Args:
      target_local: Tensor of shape (num_envs, 3) in stance-local coordinates.
    """
    if target_local.shape[0] != self.num_envs or target_local.shape[-1] != 3:
      raise ValueError(
        f"Expected AI touchdown tensor with shape ({self.num_envs}, 3), got {tuple(target_local.shape)}."
      )
    self.ai_foot_target.copy_(target_local)
    self._has_ai_foot_target[:] = True

  def _compute_non_edge_mask(
    self,
    hit_pos_w: torch.Tensor,
    distances: torch.Tensor,
    stance_z_w: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Classify steppable (non-edge) heightmap cells.

    Reuses the same edge rule as visualisation: a cell is non-edge when all
    four cardinal neighbors are valid and height differences are below the
    configured threshold. Optionally enforces a maximum absolute height delta
    to the stance foot.
    """
    valid_hits = distances >= 0.0
    if self._hm_num_x <= 0 or self._hm_num_y <= 0:
      return valid_hits

    num_envs, num_rays, _ = hit_pos_w.shape
    if num_rays != self._hm_num_x * self._hm_num_y:
      return valid_hits

    z = hit_pos_w[:, :, 2].reshape(num_envs, self._hm_num_y, self._hm_num_x)
    d = distances.reshape(num_envs, self._hm_num_y, self._hm_num_x)

    z_pad = torch.full(
      (num_envs, self._hm_num_y + 2, self._hm_num_x + 2),
      float("inf"),
      device=self.device,
      dtype=z.dtype,
    )
    d_pad = torch.full(
      (num_envs, self._hm_num_y + 2, self._hm_num_x + 2),
      -1.0,
      device=self.device,
      dtype=d.dtype,
    )
    z_pad[:, 1:-1, 1:-1] = z
    d_pad[:, 1:-1, 1:-1] = d

    z_up = z_pad[:, :-2, 1:-1]
    z_down = z_pad[:, 2:, 1:-1]
    z_left = z_pad[:, 1:-1, :-2]
    z_right = z_pad[:, 1:-1, 2:]

    d_up = d_pad[:, :-2, 1:-1]
    d_down = d_pad[:, 2:, 1:-1]
    d_left = d_pad[:, 1:-1, :-2]
    d_right = d_pad[:, 1:-1, 2:]

    th = self._hm_edge_threshold
    c_up = (d_up >= 0.0) & (torch.abs(z - z_up) < th)
    c_down = (d_down >= 0.0) & (torch.abs(z - z_down) < th)
    c_left = (d_left >= 0.0) & (torch.abs(z - z_left) < th)
    c_right = (d_right >= 0.0) & (torch.abs(z - z_right) < th)

    is_non_edge = c_up & c_down & c_left & c_right & (d >= 0.0)

    if stance_z_w is not None and self._hm_max_stance_height_delta > 0.0:
      stance_z = stance_z_w.view(num_envs, 1, 1)
      stance_delta_ok = torch.abs(z - stance_z) <= self._hm_max_stance_height_delta
      is_non_edge = is_non_edge & stance_delta_ok

    return is_non_edge.reshape(num_envs, num_rays)

  def _get_heightmap_candidates(
    self,
    env_ids: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return stance-local heightmap hits and non-edge validity mask."""
    if "heightmap" not in self._env.scene.sensors:
      return None

    hm_data = self._env.scene.sensors["heightmap"].data
    if hm_data.hit_pos_w is None or hm_data.distances is None:
      return None

    hit_pos_w = hm_data.hit_pos_w[env_ids]
    distances = hm_data.distances[env_ids]
    stance_z = self.stance_foot_pos_0[env_ids, 2]
    non_edge = self._compute_non_edge_mask(hit_pos_w, distances, stance_z)
    valid_hits = non_edge & (distances >= 0.0)

    n, num_rays, _ = hit_pos_w.shape
    rel_w = hit_pos_w - self.stance_foot_pos_0[env_ids].unsqueeze(1)
    rel_w_flat = rel_w.reshape(-1, 3)
    stance_quat = self.stance_foot_ori_quat_0[env_ids].repeat_interleave(num_rays, dim=0)
    rel_local = _to_local_frame(rel_w_flat, stance_quat).reshape(n, num_rays, 3)

    return rel_local, valid_hits

  def _solve_discrete_mpc(
    self,
    env_ids: torch.Tensor,
    cmd_xy: torch.Tensor,
    local_hits: torch.Tensor,
    valid_hits: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve a discrete MPC preview over steppable heightmap candidates."""
    n = len(env_ids)
    horizon = self._mpc_horizon
    num_t = len(self._mpc_t_grid)
    span = max(self.T_max - self.T_min, 1e-6)

    plan_t = torch.full((n, horizon), self._mpc_t_nom, device=self.device)
    plan_foot = torch.zeros(n, horizon, 3, device=self.device)
    total_cost = torch.zeros(n, device=self.device)
    all_feasible = torch.ones(n, dtype=torch.bool, device=self.device)
    feasible_steps = torch.zeros(n, device=self.device)

    # Candidate footholds are stored as absolute points in the current
    # stance-local frame. Each horizon step is scored on the displacement from
    # the previously selected foothold to enforce chained, non-overlapping plans.
    x_abs = local_hits[:, :, 0].unsqueeze(-1)
    y_abs = local_hits[:, :, 1].unsqueeze(-1)
    valid = valid_hits.unsqueeze(-1).expand(-1, -1, num_t)
    t_grid = self._mpc_t_grid.view(1, 1, num_t)
    prev_foot_abs = torch.zeros(n, 3, device=self.device)

    for k in range(horizon):
      stance_k = torch.remainder(self.stance_idx[env_ids] + k, 2)
      side_sign = torch.where(
        stance_k == 0,
        torch.full((n,), -1.0, device=self.device),
        torch.full((n,), 1.0, device=self.device),
      )
      side_offset = side_sign * self.y_nom

      cmd_x = cmd_xy[:, 0].view(n, 1, 1)
      cmd_y = cmd_xy[:, 1].view(n, 1, 1)
      side_offset_e = side_offset.view(n, 1, 1)
      side_sign_e = side_sign.view(n, 1, 1)

      prev_x = prev_foot_abs[:, 0].view(n, 1, 1)
      prev_y = prev_foot_abs[:, 1].view(n, 1, 1)

      dx = x_abs - prev_x
      dy = y_abs - prev_y

      x_ok = (dx >= self._mpc_x_min) & (dx <= self._mpc_x_max)
      dy_abs = torch.abs(dy)
      y_ok = (dy_abs >= self._mpc_abs_y_min) & (dy_abs <= self._mpc_abs_y_max)
      step_len_ok = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=1e-8)) <= self._mpc_max_step_length

      vx_pred = dx / t_grid
      vy_pred = (dy - side_offset_e) / t_grid
      vel_err = (vx_pred - cmd_x) ** 2 + (vy_pred - cmd_y) ** 2

      desired_x = cmd_x * t_grid
      desired_y = side_offset_e + cmd_y * t_grid
      foot_err = (dx - desired_x) ** 2 + (dy - desired_y) ** 2

      time_err = ((t_grid - self._mpc_t_nom) / span) ** 2
      cost = (
        self._mpc_w_vel * vel_err
        + self._mpc_w_time * time_err
        + self._mpc_w_foot * foot_err
      )

      signed_y_ok = (dy * side_sign_e) >= self._mpc_signed_y_min
      feasible = valid & x_ok & y_ok & step_len_ok & signed_y_ok
      inf_cost = torch.full_like(cost, float("inf"))
      cost = torch.where(feasible, cost, inf_cost)

      flat_cost = cost.reshape(n, -1)
      best_cost, best_flat_idx = torch.min(flat_cost, dim=1)
      feasible_k = torch.isfinite(best_cost)
      feasible_steps += feasible_k.float()
      all_feasible &= feasible_k

      ray_idx = torch.div(best_flat_idx, num_t, rounding_mode="floor")
      t_idx = best_flat_idx % num_t
      best_t = self._mpc_t_grid[t_idx]
      best_foot = torch.gather(
        local_hits,
        1,
        ray_idx.view(n, 1, 1).expand(-1, 1, 3),
      ).squeeze(1)

      fallback_foot = prev_foot_abs.clone()
      fallback_foot[:, 0] += cmd_xy[:, 0] * self._mpc_t_nom
      fallback_foot[:, 1] += (side_offset + cmd_xy[:, 1]) * self._mpc_t_nom
      fallback_foot[:, 2] = prev_foot_abs[:, 2]

      plan_t[:, k] = torch.where(
        feasible_k,
        best_t,
        torch.full_like(best_t, self._mpc_t_nom),
      )
      plan_foot[:, k] = torch.where(
        feasible_k.unsqueeze(-1),
        best_foot,
        fallback_foot,
      )
      prev_foot_abs = plan_foot[:, k]
      total_cost += torch.where(feasible_k, best_cost, torch.zeros_like(best_cost))

    total_cost = torch.where(
      all_feasible,
      total_cost,
      torch.full_like(total_cost, float("inf")),
    )
    return plan_t, plan_foot, total_cost, all_feasible, feasible_steps

  def _plan_mpc_transition(self, env_ids: torch.Tensor) -> None:
    """Plan MPC preview and commit the first step in receding-horizon style."""
    if len(env_ids) == 0:
      return

    candidates = self._get_heightmap_candidates(env_ids)
    n = len(env_ids)
    if candidates is None:
      fallback_t = torch.full((n, self._mpc_horizon), self._mpc_t_nom, device=self.device)
      fallback_foot = torch.zeros(n, self._mpc_horizon, 3, device=self.device)
      prev_foot = torch.zeros(n, 3, device=self.device)
      for k in range(self._mpc_horizon):
        side = torch.where(
          torch.remainder(self.stance_idx[env_ids] + k, 2) == 0,
          torch.full((n,), -self.y_nom, device=self.device),
          torch.full((n,), self.y_nom, device=self.device),
        )
        prev_foot[:, 1] += side
        fallback_foot[:, k] = prev_foot
      self._mpc_plan_t[env_ids] = fallback_t
      self._mpc_plan_foothold[env_ids] = fallback_foot
      self._mpc_active_foot_target[env_ids] = fallback_foot[:, 0]
      self._mpc_last_cost[env_ids] = 0.0
      self._mpc_feasible_steps[env_ids] = 0.0
      self._mpc_fallback_used[env_ids] = 1.0
      self._mpc_has_plan[env_ids] = True
      self.vel_command[env_ids] = 0.0
      self.T[env_ids] = fallback_t[:, 0]
      self.full_gait_period[env_ids] = 2.0 * (self.T[env_ids] - self.T_ds)
      return

    local_hits, valid_hits = candidates
    base_cmd = self.vel_command[env_ids, :2].clone()

    best_t, best_foot, best_cost, best_feasible, feasible_steps = self._solve_discrete_mpc(
      env_ids, base_cmd, local_hits, valid_hits
    )
    best_cmd = base_cmd.clone()
    fallback_used = torch.zeros(n, dtype=torch.bool, device=self.device)

    unresolved = ~best_feasible
    if torch.any(unresolved):
      base_speed = torch.norm(base_cmd, dim=1, keepdim=True)
      for scale in self._mpc_fallback_scales:
        ids = unresolved.nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
          break

        speed = base_speed[ids] * scale
        option_cmds = [
          base_cmd[ids] * scale,
          torch.cat([speed, torch.zeros_like(speed)], dim=1),
          torch.cat([-speed, torch.zeros_like(speed)], dim=1),
          torch.cat([torch.zeros_like(speed), speed], dim=1),
          torch.cat([torch.zeros_like(speed), -speed], dim=1),
        ]

        for cmd_option in option_cmds:
          opt_t, opt_foot, opt_cost, opt_feasible, opt_feas_steps = self._solve_discrete_mpc(
            env_ids[ids], cmd_option, local_hits[ids], valid_hits[ids]
          )
          feasible_local = opt_feasible.nonzero(as_tuple=False).flatten()
          if len(feasible_local) == 0:
            continue

          feasible_global = ids[feasible_local]
          better = opt_cost[feasible_local] < best_cost[feasible_global]
          if torch.any(better):
            chosen_local = feasible_local[better]
            chosen_global = feasible_global[better]
            best_t[chosen_global] = opt_t[chosen_local]
            best_foot[chosen_global] = opt_foot[chosen_local]
            best_cost[chosen_global] = opt_cost[chosen_local]
            best_feasible[chosen_global] = True
            feasible_steps[chosen_global] = opt_feas_steps[chosen_local]
            best_cmd[chosen_global] = cmd_option[chosen_local]
            fallback_used[chosen_global] = True

        unresolved = ~best_feasible

    final_unresolved = (~best_feasible).nonzero(as_tuple=False).flatten()
    if len(final_unresolved) > 0:
      zero_cmd = torch.zeros(len(final_unresolved), 2, device=self.device)
      zero_t, zero_foot, zero_cost, _, zero_feas_steps = self._solve_discrete_mpc(
        env_ids[final_unresolved],
        zero_cmd,
        local_hits[final_unresolved],
        valid_hits[final_unresolved],
      )
      best_t[final_unresolved] = zero_t
      best_foot[final_unresolved] = zero_foot
      best_cost[final_unresolved] = torch.where(
        torch.isfinite(zero_cost), zero_cost, torch.zeros_like(zero_cost)
      )
      feasible_steps[final_unresolved] = zero_feas_steps
      best_cmd[final_unresolved] = 0.0
      fallback_used[final_unresolved] = True
      # Full stop fallback when no non-edge sequence is feasible.
      self.vel_command[env_ids[final_unresolved], 2] = 0.0

    self._mpc_plan_t[env_ids] = best_t
    self._mpc_plan_foothold[env_ids] = best_foot
    self._mpc_active_foot_target[env_ids] = best_foot[:, 0]
    self._mpc_last_cost[env_ids] = torch.where(
      torch.isfinite(best_cost), best_cost, torch.zeros_like(best_cost)
    )
    self._mpc_feasible_steps[env_ids] = feasible_steps
    self._mpc_fallback_used[env_ids] = fallback_used.float()
    self._mpc_has_plan[env_ids] = True
    self.vel_command[env_ids, :2] = best_cmd
    self.T[env_ids] = torch.clamp(best_t[:, 0], self.T_min, self.T_max)
    self.full_gait_period[env_ids] = 2.0 * (self.T[env_ids] - self.T_ds)

  # ------------------------------------------------------------------
  # Phase tracking
  # ------------------------------------------------------------------

  def _get_foot_contact_state(self) -> torch.Tensor:
    """Return binary left/right foot contacts as ``(num_envs, 2)`` bool."""
    if self._touchdown_sensor_name not in self._env.scene.sensors:
      return torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

    sensor = self._env.scene[self._touchdown_sensor_name]
    found = getattr(sensor.data, "found", None)
    if found is None:
      return torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
    return found > 0

  def _update_stance_swing_idx(self, dt: float) -> None:
    """Update stance/swing indices from touchdown events.

    Primary trigger is swing-foot touchdown from contact sensing. A timeout
    fallback forces transition if touchdown is missing for too long.
    """
    self._swing_elapsed += dt

    T_swing = torch.clamp(self.T - self.T_ds, min=1e-6)
    phase_var = self._swing_elapsed / T_swing

    trans_ids = torch.empty(0, dtype=torch.long, device=self.device)
    if self._touchdown_enabled:
      foot_contact = self._get_foot_contact_state()
      swing_contact = torch.gather(
        foot_contact,
        1,
        self.swing_idx.unsqueeze(1),
      ).squeeze(1)
      prev_swing_contact = torch.gather(
        self._prev_foot_contact,
        1,
        self.swing_idx.unsqueeze(1),
      ).squeeze(1)

      touchdown = swing_contact & (~prev_swing_contact)
      touchdown_ready = phase_var >= self._touchdown_min_phase
      touchdown_ids = (touchdown & touchdown_ready).nonzero(as_tuple=False).flatten()

      timeout_ids = (phase_var >= self._touchdown_timeout_phase).nonzero(as_tuple=False).flatten()
      if len(touchdown_ids) > 0 and len(timeout_ids) > 0:
        trans_ids = torch.unique(torch.cat([touchdown_ids, timeout_ids], dim=0))
      elif len(touchdown_ids) > 0:
        trans_ids = touchdown_ids
      else:
        trans_ids = timeout_ids

      self._prev_foot_contact = foot_contact.clone()
    else:
      # Fallback to fixed-time half-step switching.
      timeout_ids = (phase_var >= 1.0).nonzero(as_tuple=False).flatten()
      trans_ids = timeout_ids

    new_stance = self.stance_idx.clone()
    if len(trans_ids) > 0:
      new_stance[trans_ids] = 1 - self.stance_idx[trans_ids]
    new_swing = 1 - new_stance

    if len(trans_ids) > 0:
      # Keep swing-height randomization independent of step-time planning.
      if self.z_sw_max_range is not None:
        self.z_sw_max_envs[trans_ids] = torch.empty(len(trans_ids), device=self.device).uniform_(self.z_sw_max_range[0], self.z_sw_max_range[1])

      # Record stance foot pose at transition.
      foot_pos_w = self.robot.data.body_link_pos_w[
        :, self._foot_body_ids, :
      ]  # (B, 2, 3)
      foot_quat_w = self.robot.data.body_link_quat_w[
        :, self._foot_body_ids, :
      ]  # (B, 2, 4)

      # Gather stance foot for transitioning envs.
      ns_tid = new_stance[trans_ids]  # Indices of new stance foot.
      self.stance_foot_pos_0[trans_ids] = foot_pos_w[
        trans_ids,
        ns_tid,
        :,
      ]
      stance_quat = foot_quat_w[trans_ids, ns_tid, :]
      self.stance_foot_ori_quat_0[trans_ids] = stance_quat
      self.stance_foot_ori_0[trans_ids] = _euler_from_quat(stance_quat)

      # Record swing foot position in stance-local frame at transition.
      nsw_tid = new_swing[trans_ids]
      swing_pos_w = foot_pos_w[trans_ids, nsw_tid, :]
      self.swing_foot_pos_0[trans_ids] = _to_local_frame(
        swing_pos_w - self.stance_foot_pos_0[trans_ids],
        self.stance_foot_ori_quat_0[trans_ids],
      )

      # Use the post-transition stance/swing assignment for planning.
      self.stance_idx[trans_ids] = new_stance[trans_ids]
      self.swing_idx[trans_ids] = new_swing[trans_ids]
      self._swing_elapsed[trans_ids] = 0.0

      if self._mpc_enabled:
        self._plan_mpc_transition(trans_ids)
      else:
        self.T[trans_ids] = torch.empty(len(trans_ids), device=self.device).uniform_(
          self.T_min,
          self.T_max,
        )
        self.full_gait_period[trans_ids] = 2.0 * (self.T[trans_ids] - self.T_ds)

    self.stance_idx = new_stance
    self.swing_idx = new_swing
    self._prev_stance_idx = new_stance.clone()

    T_swing = torch.clamp(self.T - self.T_ds, min=1e-6)
    self.phase_var = torch.clamp(self._swing_elapsed / T_swing, min=0.0, max=1.0)
    self.cur_swing_time = torch.minimum(self._swing_elapsed, T_swing)

    # Keep tp in [0,1) while encoding current stance half-cycle.
    phase_norm = torch.clamp(self.phase_var, max=0.999)
    self.tp = 0.5 * self.stance_idx.float() + 0.5 * phase_norm
    self.gait_time = self.cur_swing_time + self.stance_idx.float() * T_swing

  # ------------------------------------------------------------------
  # Reference trajectory generation
  # ------------------------------------------------------------------

  def _generate_orientation_ref(
    self, base_velocity: torch.Tensor, N: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate pelvis and foot orientation references.

    Returns:
      (pelvis_euler, pelvis_eul_dot, foot_eul, foot_eul_dot), each ``(N, 3)``.
    """
    pelvis_euler = torch.zeros(N, 3, device=self.device)

    roll_main_amp = 0.0
    roll_asym_amp = -0.05

    pelvis_euler[:, 0] = (
      roll_main_amp * torch.sin(4 * math.pi * self.tp)
      + roll_asym_amp * torch.sin(2 * math.pi * self.tp)
    )

    # Lateral and turning biases.
    bias_lat = torch.clamp(
      torch.atan(base_velocity[:, 1] / GRAVITY), -0.15, 0.15
    )
    bias_yaw = torch.clamp(
      torch.atan(
        base_velocity[:, 0] * base_velocity[:, 2] / GRAVITY
      ),
      -0.2,
      0.2,
    )
    pelvis_euler[:, 0] += bias_lat + bias_yaw

    pitch_amp = 0.02
    pelvis_euler[:, 1] = (
      self.cfg.pelv_pitch_ref
      + torch.sin(2 * math.pi * self.tp) * pitch_amp
    )

    yaw_amp = 0.0
    default_yaw = yaw_amp * torch.sin(2 * math.pi * self.tp)
    pelvis_euler[:, 2] = (
      default_yaw
      + self.stance_foot_ori_0[:, 2]
      + base_velocity[:, 2] * self.cur_swing_time
    )

    # Euler rates.
    dtp_dt = 1.0 / self.full_gait_period
    pelvis_eul_dot = torch.zeros(N, 3, device=self.device)
    pelvis_eul_dot[:, 0] = (
      roll_main_amp * 4 * math.pi * torch.cos(4 * math.pi * self.tp) * dtp_dt
      + roll_asym_amp * 2 * math.pi * torch.cos(2 * math.pi * self.tp) * dtp_dt
    )
    pelvis_eul_dot[:, 1] = (
      2 * math.pi * torch.cos(2 * math.pi * self.tp) * pitch_amp * dtp_dt
    )
    pelvis_eul_dot[:, 2] = (
      base_velocity[:, 2]
      + yaw_amp * 2 * math.pi * torch.cos(2 * math.pi * self.tp) * dtp_dt
    )

    # Foot orientation: match pelvis yaw.
    foot_eul = torch.zeros(N, 3, device=self.device)
    foot_eul[:, 2] = pelvis_euler[:, 2]
    foot_eul_dot = torch.zeros(N, 3, device=self.device)
    foot_eul_dot[:, 2] = pelvis_eul_dot[:, 2]

    return pelvis_euler, pelvis_eul_dot, foot_eul, foot_eul_dot

  def _generate_upper_body_ref(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate upper body joint position/velocity references.

    Returns:
      (ref, ref_dot), each ``(num_envs, n_upper)``.
    """
    forward_vel = self.vel_command[:, 0]
    N = forward_vel.shape[0]
    phase = 2 * math.pi * self.tp

    sh_pitch0, sh_roll0, sh_yaw0 = self.cfg.shoulder_ref
    elb0 = self.cfg.elbow_ref
    waist_yaw0 = self.cfg.waist_yaw_ref

    sh_pitch_amp = sh_pitch0 * forward_vel
    sh_roll_amp = sh_roll0 * torch.ones_like(forward_vel)
    sh_yaw_amp = sh_yaw0 * torch.ones_like(forward_vel)
    elb_amp = elb0 * forward_vel
    waist_amp = waist_yaw0 * torch.ones_like(forward_vel)

    amp = torch.stack(
      [
        waist_amp,
        sh_pitch_amp, sh_pitch_amp,   # L/R shoulder pitch
        sh_roll_amp, sh_roll_amp,     # L/R shoulder roll
        sh_yaw_amp, sh_yaw_amp,       # L/R shoulder yaw
        elb_amp, elb_amp,             # L/R elbow
      ],
      dim=1,
    ).to(self.device)

    sign = torch.tensor(
      [1, 1, -1, 1, -1, 1, -1, 1, -1],
      device=self.device,
      dtype=torch.float32,
    )

    offset = torch.tensor(
      [
        math.pi,                       # waist_yaw
        math.pi / 2, math.pi / 2,     # L/R shoulder pitch
        math.pi / 2, math.pi / 2,     # L/R shoulder roll
        0.0, 0.0,                      # L/R shoulder yaw
        math.pi / 2, math.pi / 2,     # L/R elbow
      ],
      device=self.device,
      dtype=torch.float32,
    )

    joint_offset = self.robot.data.default_joint_pos[
      :, self._upper_body_joint_ids
    ]

    offset_expanded = offset.unsqueeze(0).expand(N, -1)
    ref = amp * sign * torch.sin(phase.unsqueeze(-1) + offset_expanded) + joint_offset

    dphase_dt = 2 * math.pi / self.full_gait_period
    ref_dot = amp * sign * torch.cos(phase.unsqueeze(-1) + offset_expanded) * dphase_dt.unsqueeze(-1)

    return ref, ref_dot

  def _generate_reference_trajectory(self) -> None:
    """Compute the full reference trajectory for the current phase."""
    base_velocity = self.vel_command
    N = base_velocity.shape[0]

    T_tensor = self.T

    Xdes, Ux, Ydes, Uy = self.hlip_controller.compute_orbit(
      T=T_tensor, cmd=base_velocity
    )

    assert self.hlip_controller.x_init is not None
    assert self.hlip_controller.y_init is not None
    x0 = self.hlip_controller.x_init

    # Gather per-env stance side for y-orbit selection.
    y0 = torch.gather(
      self.hlip_controller.y_init, 1, self.stance_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
    ).squeeze(1)
    Uy_sel = torch.gather(Uy, 1, self.stance_idx.unsqueeze(1)).squeeze(1)

    com_x, com_xd = self.hlip_controller._compute_desire_com_trajectory(
      cur_time=self.cur_swing_time, Xdesire=x0
    )
    com_y, com_yd = self.hlip_controller._compute_desire_com_trajectory(
      cur_time=self.cur_swing_time, Xdesire=y0
    )

    com_pos_des = torch.stack(
      [com_x, com_y, self.com_z], dim=-1
    )
    com_vel_des = torch.stack(
      [com_xd, com_yd, torch.zeros(N, device=self.device)], dim=-1
    )

    # Raw foot target.
    foot_target = torch.stack(
      [Ux, Uy_sel, torch.zeros(N, device=self.device)], dim=-1
    )

    # Yaw-rate adjustment.
    delta_psi = base_velocity[:, 2] * self.cur_swing_time
    q_delta_yaw = quat_from_euler_xyz(
      torch.zeros_like(delta_psi),
      torch.zeros_like(delta_psi),
      delta_psi,
    )
    foot_target_adj = quat_apply(q_delta_yaw, foot_target)
    com_pos_des = quat_apply(q_delta_yaw, com_pos_des)
    com_vel_des = quat_apply(q_delta_yaw, com_vel_des)

    # Clip foot target lateral range.
    foot_target_clipped = foot_target_adj.clone()
    foot_target_clipped[:, 1] = (
      torch.clamp(
        torch.abs(foot_target_adj[:, 1]),
        min=self.cfg.foot_target_range_y[0],
        max=self.cfg.foot_target_range_y[1],
      )
      * torch.sign(Uy_sel)
    )

    # MPC override: use edge-constrained planned foothold from transition planner.
    if self._mpc_enabled:
      mpc_ids = self._mpc_has_plan.nonzero(as_tuple=False).flatten()
      if len(mpc_ids) > 0:
        foot_target_clipped[mpc_ids] = self._mpc_active_foot_target[mpc_ids]
      foot_target_adj = foot_target_clipped

    # Swing foot trajectory via Bezier.
    horizontal_cp = torch.tensor(
      [0.0, 0.0, 1.0, 1.0, 1.0], device=self.device
    ).unsqueeze(0).expand(N, -1)
    T_tensor_sw = self.T

    bht = bezier_deg(0, self.phase_var, T_tensor_sw, horizontal_cp, 4)

    z_sw_max = torch.full((N,), self.cfg.z_sw_max, device=self.device)

    z_sw_neg = torch.full((N,), self.cfg.z_sw_min, device=self.device)
    mpc_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    if self._mpc_enabled:
      mpc_mask = self._mpc_has_plan.clone()
      z_sw_neg[mpc_mask] = foot_target_clipped[mpc_mask, 2]

    need_height_lookup = (~mpc_mask).nonzero(as_tuple=False).flatten()
    if len(need_height_lookup) > 0:
      try:
        heightmap_data = self._env.scene.sensors["heightmap"].data
        hit_pos_w = heightmap_data.hit_pos_w  # (N, num_rays, 3)

        local_target = torch.cat(
          [
            foot_target_clipped[:, :2],
            torch.zeros(N, 1, device=self.device),
          ],
          dim=1,
        )
        world_target_pos = self.stance_foot_pos_0 + quat_apply(
          yaw_quat(self.stance_foot_ori_quat_0), local_target
        )

        diffs = (
          hit_pos_w[need_height_lookup, :, :2]
          - world_target_pos[need_height_lookup].unsqueeze(1)[:, :, :2]
        )
        dist_sq = (diffs ** 2).sum(dim=-1)
        closest_indices = dist_sq.argmin(dim=-1)

        closest_z_w = torch.gather(
          hit_pos_w[need_height_lookup, :, 2],
          1,
          closest_indices.unsqueeze(1),
        ).squeeze(1)
        z_sw_neg[need_height_lookup] = (
          closest_z_w - self.stance_foot_pos_0[need_height_lookup, 2]
        )
      except KeyError:
        pass

    # Start Bezier from the actual swing foot position (stance-local frame)
    # recorded at the beginning of this half-gait.
    z_init = self.swing_foot_pos_0[:, 2]
    step_x_init = self.swing_foot_pos_0[:, 0]
    step_y_init = self.swing_foot_pos_0[:, 1]

    # dynamically adjust z_sw_max to be relative to the highest point between start and end
    z_sw_max_updated = torch.max(z_init, z_sw_neg) + self.z_sw_max_envs

    self.foot_target = torch.cat([foot_target_clipped[:, :2], z_sw_neg.unsqueeze(1)], dim=1)

    foot_pos, sw_z = calculate_cur_swing_foot_pos(
      bht,
      z_init,
      z_sw_max_updated,
      self.phase_var,
      step_x_init,
      step_y_init,
      T_tensor_sw,
      z_sw_neg,
      foot_target_adj[:, 0],
      foot_target_adj[:, 1],
    )

    dbht = bezier_deg(1, self.phase_var, T_tensor_sw, horizontal_cp, 4)
    foot_vel = torch.zeros(N, 3, device=self.device)
    foot_vel[:, 0] = -dbht * step_x_init + dbht * foot_target_adj[:, 0]
    foot_vel[:, 1] = -dbht * step_y_init + dbht * foot_target_adj[:, 1]
    foot_vel[:, 2] = sw_z.squeeze(-1)

    # Upper body.
    upper_pos, upper_vel = self._generate_upper_body_ref()

    # Pelvis and foot orientation.
    pelv_eul, pelv_eul_dot, foot_eul, foot_eul_dot = (
      self._generate_orientation_ref(base_velocity, N)
    )
    omega_ref = _euler_rates_to_omega(pelv_eul, pelv_eul_dot)
    omega_foot_ref = _euler_rates_to_omega(foot_eul, foot_eul_dot)

    # Assemble reference.
    self.y_out = torch.cat(
      [com_pos_des, pelv_eul, foot_pos, foot_eul, upper_pos], dim=-1
    )
    self.dy_out = torch.cat(
      [com_vel_des, omega_ref, foot_vel, omega_foot_ref, upper_vel], dim=-1
    )

  # ------------------------------------------------------------------
  # Actual state extraction
  # ------------------------------------------------------------------

  def _get_actual_state(self) -> None:
    """Extract actual state from robot sensor data in stance-foot-local frame."""
    data = self.robot.data
    root_quat = data.root_link_quat_w

    foot_pos_w = data.body_link_pos_w[:, self._foot_body_ids, :]
    foot_quat_w = data.body_link_quat_w[:, self._foot_body_ids, :]

    # Current stance foot position (for holonomic constraint rewards).
    self.stance_foot_pos = torch.gather(
      foot_pos_w,
      1,
      self.stance_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3),
    ).squeeze(1)

    # Swing foot in stance-local frame.
    swing_pos_w = torch.gather(
      foot_pos_w,
      1,
      self.swing_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3),
    ).squeeze(1)
    swing2stance_local = _to_local_frame(
      swing_pos_w - self.stance_foot_pos_0, self.stance_foot_ori_quat_0
    )

    # CoM in stance-local frame.
    com_w = data.root_com_pos_w
    com2stance_local = _to_local_frame(
      com_w - self.stance_foot_pos_0, self.stance_foot_ori_quat_0
    )

    # Pelvis orientation.
    pelvis_ori = _euler_from_quat(root_quat)

    # Swing foot orientation.
    swing_quat_w = torch.gather(
      foot_quat_w,
      1,
      self.swing_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 4),
    ).squeeze(1)
    swing_foot_ori = _euler_from_quat(swing_quat_w)

    # Velocities.
    com_vel_w = data.root_com_lin_vel_w
    com_vel_local = _to_local_frame(com_vel_w, self.stance_foot_ori_quat_0)

    pelvis_omega_local = data.root_link_ang_vel_b

    foot_lin_vel_w = data.body_link_lin_vel_w[:, self._foot_body_ids, :]
    foot_ang_vel_w = data.body_link_ang_vel_w[:, self._foot_body_ids, :]

    # Stance foot velocity (for holonomic constraint).
    self.stance_foot_vel = torch.gather(
      foot_lin_vel_w,
      1,
      self.stance_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3),
    ).squeeze(1)
    self.stance_foot_ang_vel = torch.gather(
      foot_ang_vel_w,
      1,
      self.stance_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3),
    ).squeeze(1)

    # Swing foot velocities in local frame.
    swing_lin_vel_w = torch.gather(
      foot_lin_vel_w,
      1,
      self.swing_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3),
    ).squeeze(1)
    swing2stance_vel = _to_local_frame(
      swing_lin_vel_w, self.stance_foot_ori_quat_0
    )

    swing_ang_vel_w = torch.gather(
      foot_ang_vel_w,
      1,
      self.swing_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3),
    ).squeeze(1)
    foot_ang_vel_local_swing = quat_apply(
      quat_inv(swing_quat_w), swing_ang_vel_w
    )

    # Upper body.
    upper_body_pos = data.joint_pos[:, self._upper_body_joint_ids]
    upper_body_vel = data.joint_vel[:, self._upper_body_joint_ids]

    # Assemble.
    self.y_act = torch.cat(
      [
        com2stance_local,
        pelvis_ori,
        swing2stance_local,
        swing_foot_ori,
        upper_body_pos,
      ],
      dim=-1,
    )
    self.dy_act = torch.cat(
      [
        com_vel_local,
        pelvis_omega_local,
        swing2stance_vel,
        foot_ang_vel_local_swing,
        upper_body_vel,
      ],
      dim=-1,
    )

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  def _update_metrics(self) -> None:
    self.metrics["v_mean"] = self.v
    if self._mpc_enabled:
      self.metrics["mpc_selected_cost"] = self._mpc_last_cost
      self.metrics["mpc_feasible_steps"] = self._mpc_feasible_steps
      self.metrics["mpc_time_deviation"] = torch.abs(self.T - self._mpc_t_nom)
      self.metrics["mpc_fallback_used"] = self._mpc_fallback_used

  def _reset_mpc_state(self, env_ids: torch.Tensor) -> None:
    """Clear MPC planner state for environments being reset/resampled.

    Command manager resets terminated/time-out environments through
    ``_resample_command``. Keeping this logic in one place guarantees we do
    not carry stale plan buffers across episode boundaries.
    """
    if (not self._mpc_enabled) or len(env_ids) == 0:
      return

    self._mpc_has_plan[env_ids] = False
    self._mpc_last_cost[env_ids] = 0.0
    self._mpc_feasible_steps[env_ids] = 0.0
    self._mpc_fallback_used[env_ids] = 0.0

    self._mpc_active_foot_target[env_ids] = 0.0
    self._mpc_active_foot_target[env_ids, 1] = -self.y_nom
    self._mpc_active_foot_target[env_ids, 2] = self.cfg.z_sw_min

    self._mpc_plan_t[env_ids] = self._mpc_t_nom
    self._mpc_plan_foothold[env_ids] = 0.0
    self._mpc_plan_foothold[env_ids, :, 1] = -self.y_nom
    self._mpc_plan_foothold[env_ids, :, 2] = self.cfg.z_sw_min

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = torch.empty(n, device=self.device)

    # Sample velocity command.
    self.vel_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

    # Zero small commands.
    cmd_mag = (
      torch.norm(self.vel_command[env_ids, :2], dim=1)
      + torch.abs(self.vel_command[env_ids, 2])
    )
    self.vel_command[env_ids] *= (cmd_mag > 0.1).unsqueeze(1)

    # Standing fraction.
    self.is_standing_env[env_ids] = (
      r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    )
    standing = env_ids[self.is_standing_env[env_ids]]
    if len(standing) > 0:
      self.vel_command[standing] = 0.0

    # Reset gait phase and initialise step time.
    if self._mpc_enabled:
      self.T[env_ids] = self._mpc_t_nom
    else:
      self.T[env_ids] = r.uniform_(self.T_min, self.T_max)
    self.full_gait_period[env_ids] = 2.0 * (self.T[env_ids] - self.T_ds)
    if self.z_sw_max_range is not None:
      self.z_sw_max_envs[env_ids] = r.uniform_(self.z_sw_max_range[0], self.z_sw_max_range[1])
    self.gait_time[env_ids] = 0.0
    self._swing_elapsed[env_ids] = 0.0
    self._prev_foot_contact[env_ids] = False

    # Start with left foot as stance (stance_idx=0).
    self.stance_idx[env_ids] = 0
    self.swing_idx[env_ids] = 1
    self._prev_stance_idx[env_ids] = 0

    self._reset_mpc_state(env_ids)

    # NOTE: Body-link positions are stale here (sim.forward() hasn't run
    # after the reset event that randomised the root pose).  We defer the
    # stance foot pose read to the first _update_command call via a flag.
    self._pending_stance_init[env_ids] = True
    self._has_ai_foot_target[env_ids] = False
    self.ai_foot_target[env_ids] = 0.0

    # Reset phase variables.
    self.tp[env_ids] = 0.0
    self.phase_var[env_ids] = 0.0
    self.cur_swing_time[env_ids] = 0.0

    # Reset CLF buffers.
    self.clf.reset(env_ids)
    self.v[env_ids] = 0.0
    self.vdot[env_ids] = 0.0
    self.v_buffer[env_ids] = 0.0
    self.vdot_buffer[env_ids] = 0.0

  def _update_command(self) -> None:
    dt = self._env.step_dt

    # Deferred stance foot initialisation: body positions are now fresh
    # (sim.forward() has run since the last reset event).
    init_ids = self._pending_stance_init.nonzero(as_tuple=False).flatten()
    if len(init_ids) > 0:
      # Defensive clear: envs waiting for post-reset stance init must never
      # carry a valid MPC plan from a previous episode.
      self._reset_mpc_state(init_ids)

      foot_pos_w = self.robot.data.body_link_pos_w[
        :, self._foot_body_ids, :
      ]
      foot_quat_w = self.robot.data.body_link_quat_w[
        :, self._foot_body_ids, :
      ]
      self.stance_foot_pos_0[init_ids] = foot_pos_w[init_ids, 0, :]
      self.stance_foot_ori_quat_0[init_ids] = foot_quat_w[init_ids, 0, :]
      self.stance_foot_ori_0[init_ids] = _euler_from_quat(
        foot_quat_w[init_ids, 0, :]
      )
      # Record initial swing foot position in stance-local frame.
      swing_pos_w = foot_pos_w[init_ids, 1, :]  # swing_idx=1 at init
      self.swing_foot_pos_0[init_ids] = _to_local_frame(
        swing_pos_w - self.stance_foot_pos_0[init_ids],
        self.stance_foot_ori_quat_0[init_ids],
      )
      self._pending_stance_init[init_ids] = False

    if self._mpc_enabled:
      plan_ids = (
        (~self._pending_stance_init) & (~self._mpc_has_plan)
      ).nonzero(as_tuple=False).flatten()
      if len(plan_ids) > 0:
        self._plan_mpc_transition(plan_ids)

    # Standing envs zero velocity.
    standing = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command[standing] = 0.0

    # Update phase.
    self._update_stance_swing_idx(dt)

    # Generate reference and extract actual state.
    self._generate_reference_trajectory()
    self._get_actual_state()

    # Compute CLF V and Vdot.
    vdot, vcur = self.clf.compute_vdot(
      self.y_act, self.y_out, self.dy_act, self.dy_out, self._yaw_idx
    )
    self.vdot = vdot
    self.v = vcur

    # Update rolling buffers.
    if torch.sum(self.v_buffer) == 0:
      self.v_buffer[:] = self.v.unsqueeze(1)
      self.vdot_buffer[:] = self.vdot.unsqueeze(1)
    else:
      self.v_buffer = torch.cat(
        [self.v_buffer[:, 1:], self.v.unsqueeze(-1)], dim=-1
      )
      self.vdot_buffer = torch.cat(
        [self.vdot_buffer[:, 1:], self.vdot.unsqueeze(-1)], dim=-1
      )

  # ------------------------------------------------------------------
  # Visualisation
  # ------------------------------------------------------------------

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw swing foot target, MPC preview, and velocity arrow."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    foot_pos = self.robot.data.body_link_pos_w[batch, self._foot_body_ids, :]
    stance_pos = foot_pos[self.stance_idx[batch].item()].cpu().numpy()
    swing_pos = foot_pos[self.swing_idx[batch].item()].cpu().numpy()

    # Green = stance foot.
    visualizer.add_sphere(
      stance_pos, radius=0.03, color=(0.2, 0.8, 0.2, 0.8),
      label="stance_foot",
    )

    # Red = foot placement target (from HLIP capture-point planner).
    # foot_target is in stance-local frame; rotate by stance yaw to world.
    z_val = self.foot_target[batch, 2].item() if self.foot_target.shape[-1] >= 3 else 0.0
    target_local = torch.tensor(
      [self.foot_target[batch, 0].item(), self.foot_target[batch, 1].item(), z_val],
      device=self.device,
    )
    target_world = (
      quat_apply(
        yaw_quat(self.stance_foot_ori_quat_0[batch].unsqueeze(0)),
        target_local.unsqueeze(0),
      ).squeeze(0)
      + self.stance_foot_pos_0[batch]
    )
    target_np = target_world.cpu().numpy()
    visualizer.add_sphere(
      target_np,
      radius=0.03,
      color=(0.8, 0.2, 0.2, 0.8),
      label="foot_target",
    )

    # Magenta = AI-predicted touchdown target (in stance-local frame -> world).
    if bool(self._has_ai_foot_target[batch].item()):
      ai_local = self.ai_foot_target[batch]
      ai_world = (
        quat_apply(
          yaw_quat(self.stance_foot_ori_quat_0[batch].unsqueeze(0)),
          ai_local.unsqueeze(0),
        ).squeeze(0)
        + self.stance_foot_pos_0[batch]
      )
      visualizer.add_sphere(
        ai_world.cpu().numpy(),
        radius=0.028,
        color=(0.9, 0.1, 0.9, 0.95),
        label="ai_touchdown_target",
      )

    # MPC preview: render the full planned horizon in world frame.
    if self._mpc_enabled and bool(self._mpc_has_plan[batch].item()):
      plan_local = self._mpc_plan_foothold[batch]  # (H, 3), stance-local
      horizon = plan_local.shape[0]
      stance_quat_h = self.stance_foot_ori_quat_0[batch].unsqueeze(0).expand(horizon, -1)
      stance_pos_h = self.stance_foot_pos_0[batch].unsqueeze(0).expand(horizon, -1)
      plan_world = quat_apply(yaw_quat(stance_quat_h), plan_local) + stance_pos_h
      plan_np = plan_world.cpu().numpy()

      step_colors = (
        (1.0, 0.2, 0.2, 0.95),
        (1.0, 0.55, 0.15, 0.95),
        (1.0, 0.85, 0.15, 0.95),
        (0.45, 0.95, 0.25, 0.95),
        (0.2, 0.9, 0.9, 0.95),
        (0.35, 0.65, 1.0, 0.95),
      )
      plan_vis_np = plan_np.copy()

      for k in range(horizon):
        color = step_colors[k % len(step_colors)]
        radius = max(0.028 - 0.0025 * k, 0.016)
        step_pos = plan_vis_np[k].copy()
        # Lift each preview marker slightly so overlapping steps stay visible.
        step_pos[2] += 0.02 * (k + 1)
        plan_vis_np[k] = step_pos
        visualizer.add_sphere(
          step_pos,
          radius=radius,
          color=color,
          label=f"mpc_step_{k + 1}",
        )

        # Draw a vertical indicator to make each step index visible even when
        # multiple footholds share the same x-y location.
        ground_pos = plan_np[k].copy()
        ground_pos[2] += 0.003
        visualizer.add_arrow(
          ground_pos,
          step_pos,
          color=color,
          width=0.004,
          label=f"mpc_step_lift_{k + 1}",
        )

      # Draw arrows between consecutive planned touchdowns for readability.
      chain_points = np.vstack([stance_pos.reshape(1, 3), plan_vis_np])
      for k in range(chain_points.shape[0] - 1):
        p0 = chain_points[k].copy()
        p1 = chain_points[k + 1].copy()
        p0[2] += 0.03
        p1[2] += 0.03
        visualizer.add_arrow(
          p0,
          p1,
          color=(1.0, 0.65, 0.15, 0.75),
          width=0.007,
          label=f"mpc_chain_{k}",
        )

    # Blue = current swing foot.
    visualizer.add_sphere(
      swing_pos, radius=0.025, color=(0.2, 0.2, 0.8, 0.8),
      label="swing_foot",
    )

    # Yellow = reference swing foot position (y_out[6:9] in stance-local frame → world).
    ref_swing_local = self.y_out[batch, 6:9]  # (3,) in stance-foot-local
    stance_quat = self.stance_foot_ori_quat_0[batch]  # (4,)
    stance_pos_0 = self.stance_foot_pos_0[batch]  # (3,)
    # Rotate from local to world: use yaw-only stance quaternion.
    ref_swing_world = quat_apply(
      yaw_quat(stance_quat.unsqueeze(0)), ref_swing_local.unsqueeze(0)
    ).squeeze(0) + stance_pos_0
    ref_swing_np = ref_swing_world.cpu().numpy()
    visualizer.add_sphere(
      ref_swing_np, radius=0.025, color=(0.9, 0.9, 0.1, 0.9),
      label="ref_swing_pos",
    )

    # Cyan arrow = velocity command direction (originating from robot root).
    root_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    root_quat_t = self.robot.data.root_link_quat_w[batch]
    vel_cmd = self.vel_command[batch]  # (3,) = (vx, vy, yaw_rate)
    # Build a local 3D velocity vector and rotate to world frame.
    vel_local = torch.tensor(
      [vel_cmd[0].item(), vel_cmd[1].item(), 0.0],
      device=self.device,
    )
    vel_world = quat_apply(
      yaw_quat(root_quat_t.unsqueeze(0)), vel_local.unsqueeze(0)
    ).squeeze(0).cpu().numpy()
    arrow_start = root_pos.copy()
    arrow_start[2] = stance_pos[2] + 0.05  # Slightly above ground.
    arrow_end = arrow_start + vel_world
    if np.linalg.norm(vel_world) > 0.01:
      visualizer.add_arrow(
        arrow_start, arrow_end,
        color=(0.1, 0.9, 0.9, 0.9), width=0.012,
        label="vel_cmd",
      )

    # Heightmap rendering with edge detection
    if "heightmap" in self._env.scene.sensors:
      hm_data = self._env.scene.sensors["heightmap"].data
      if hm_data.hit_pos_w is not None and hm_data.distances is not None:
        hit_pos_b = hm_data.hit_pos_w[batch]
        dists_b = hm_data.distances[batch]
        non_edge_b = self._compute_non_edge_mask(
          hit_pos_b.unsqueeze(0),
          dists_b.unsqueeze(0),
          self.stance_foot_pos_0[batch, 2].unsqueeze(0),
        ).squeeze(0)

        if self._hm_num_x > 0 and self._hm_num_y > 0:
          hit_grid = hit_pos_b.cpu().numpy().reshape(self._hm_num_y, self._hm_num_x, 3)
          dist_grid = dists_b.cpu().numpy().reshape(self._hm_num_y, self._hm_num_x)
          non_edge_grid = non_edge_b.cpu().numpy().reshape(self._hm_num_y, self._hm_num_x)

          for i in range(self._hm_num_y):
            for j in range(self._hm_num_x):
              if dist_grid[i, j] >= 0:
                h_pos = hit_grid[i, j]
                col = (0.0, 1.0, 0.0, 0.8) if non_edge_grid[i, j] else (1.0, 0.0, 0.0, 0.8)
                visualizer.add_sphere(h_pos, radius=0.02, color=col)

# =====================================================================
# Configuration
# =====================================================================


# Default Q weights: 42 entries (21 output pairs × pos/vel).
# Joint order matches planc regex expansion:
#   waist_yaw, L_shoulder_pitch, R_shoulder_pitch, L_shoulder_roll,
#   R_shoulder_roll, L_shoulder_yaw, R_shoulder_yaw, L_elbow, R_elbow
HLIP_DEFAULT_Q_WEIGHTS = [
  25.0, 200.0,      # com_x
  300.0, 50.0,      # com_y
  400.0, 10.0,      # com_z
  420.0, 20.0,      # pelvis_roll
  200.0, 10.0,      # pelvis_pitch
  300.0, 10.0,      # pelvis_yaw
  1500.0, 125.0,    # swing_x
  1700.0, 125.0,    # swing_y
  3500.0, 100.0,    # swing_z
  30.0, 1.0,        # swing_ori_roll
  10.0, 1.0,        # swing_ori_pitch
  400.0, 10.0,      # swing_ori_yaw
  500.0, 10.0,      # waist_yaw
  40.0, 1.0,        # left shoulder pitch
  40.0, 1.0,        # right shoulder pitch
  100.0, 1.0,       # left shoulder roll
  100.0, 1.0,       # right shoulder roll
  50.0, 1.0,        # left shoulder yaw
  50.0, 1.0,        # right shoulder yaw
  30.0, 1.0,        # left elbow
  30.0, 1.0,        # right elbow
]

# Default R weights: 21 entries (one per output).
# Upper body order matches planc regex expansion.
HLIP_DEFAULT_R_WEIGHTS = [
  0.1, 0.1, 0.1,         # CoM
  0.05, 0.05, 0.05,      # Pelvis
  0.05, 0.05, 0.05,      # Swing foot linear
  0.02, 0.02, 0.02,      # Swing foot orientation
  0.1,                    # Waist yaw
  0.01, 0.01,             # L/R shoulder pitch
  0.01, 0.01,             # L/R shoulder roll
  0.01, 0.01,             # L/R shoulder yaw
  0.01, 0.01,             # L/R elbow
]


@dataclass(kw_only=True)
class HLIPCommandCfg(CommandTermCfg):
  """Configuration for the HLIP + CLF walking command generator."""

  entity_name: str
  """Name of the robot entity in the scene."""

  foot_body_name: str = r".*_ankle_roll_link"
  """Regex pattern matching the two foot bodies (left first in sorted order)."""

  upper_body_joint_names: list[str] | str = field(default_factory=lambda: [
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
  ])
  """Joint names for upper body reference tracking.

  Order matches planc regex expansion:
    waist_yaw, L/R shoulder_pitch, L/R shoulder_roll,
    L/R shoulder_yaw, L/R elbow.
  """

  # HLIP dynamics.
  T_ds: float = 0.0
  """Double support duration [s]. Set to 0 for pure SSP."""

  z0: float = 0.67
  """Nominal CoM height for LIP dynamics [m]."""

  y_nom: float = 0.25
  """Nominal lateral foot separation [m]."""

  T_min: float = 0.4
  """Minimum half of the full gait period [s]. Full stride = 2 * T_min."""

  T_max: float = 0.4
  """Maximum half of the full gait period [s]. Full stride = 2 * T_max."""

  # Swing trajectory.
  z_sw_max: float = 0.1
  """Maximum swing foot height [m]."""

  z_sw_max_range: tuple[float, float] | None = None
  """Range to sample maximum swing foot height from [m]. If None, use z_sw_max."""

  z_sw_min: float = 0.0
  """Swing foot landing height [m]."""

  foot_target_range_y: tuple[float, float] = (0.1, 0.5)
  """Lateral foot target clipping range [m]."""

  # MPC foothold/time planning.
  mpc_enabled: bool = False
  """Enable edge-constrained MPC foothold and step-time planning."""

  mpc_horizon: int = 3
  """Preview horizon (number of planned steps)."""

  mpc_t_candidates: int = 7
  """Number of discrete step-time candidates in [T_min, T_max]."""

  mpc_w_vel: float = 1.0
  """Weight for linear velocity tracking in MPC cost."""

  mpc_w_time: float = 0.2
  """Weight for nominal step-time preference in MPC cost."""

  mpc_w_foot: float = 0.05
  """Weight for foothold displacement consistency in MPC cost."""

  mpc_foot_target_range_x: tuple[float, float] = (-0.4, 0.8)
  """Allowed fore-aft foothold range [m] in stance-local frame."""

  mpc_abs_y_min: float = 0.08
  """Minimum absolute lateral foothold distance [m]."""

  mpc_abs_y_max: float = 0.6
  """Maximum absolute lateral foothold distance [m]."""

  mpc_signed_y_min: float = 0.02
  """Minimum signed lateral displacement towards the swing side [m]."""

  mpc_max_step_length: float = 0.95
  """Maximum step length in the horizontal plane [m]."""

  mpc_edge_height_threshold: float = -1.0
  """Height-difference edge threshold; <0 derives from heightmap resolution."""

  mpc_max_stance_height_delta: float = -1.0
  """Max |z_hit - z_stance| [m] for steppability; <=0 disables this filter."""

  mpc_fallback_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
  """Velocity scaling factors used when searching fallback feasible plans."""

  touchdown_switch_enabled: bool = True
  """Switch stance when swing foot touchdown is detected."""

  touchdown_sensor_name: str = "feet_ground_contact"
  """Contact sensor used to detect swing-foot touchdown events."""

  touchdown_min_phase: float = 0.35
  """Earliest normalized swing phase at which touchdown can trigger switch."""

  touchdown_timeout_phase: float = 1.20
  """Fallback forced switch threshold when touchdown is not detected."""

  # Pelvis orientation.
  pelv_pitch_ref: float = 0.0
  """Pelvis pitch reference offset [rad]."""

  # Upper body.
  shoulder_ref: tuple[float, float, float] = (0.16, 0.0, 0.0)
  """Shoulder (pitch, roll, yaw) amplitude scalars."""

  elbow_ref: float = 0.1
  """Elbow swing amplitude scalar."""

  waist_yaw_ref: float = 0.0
  """Waist yaw oscillation amplitude."""

  # CLF weights.
  Q_weights: list[float] = field(default_factory=lambda: HLIP_DEFAULT_Q_WEIGHTS)
  """CLF Q weight diagonal (length = 2 * num_outputs)."""

  R_weights: list[float] = field(default_factory=lambda: HLIP_DEFAULT_R_WEIGHTS)
  """CLF R weight diagonal (length = num_outputs)."""

  yaw_idx: list[int] = field(default_factory=lambda: [5, 11])
  """Indices of yaw outputs for angle wrapping in CLF computation."""

  # Velocity command.
  rel_standing_envs: float = 0.05
  """Fraction of environments with zero-velocity command."""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: "ManagerBasedRlEnv") -> HLIPCommandTerm:
    return HLIPCommandTerm(self, env)
