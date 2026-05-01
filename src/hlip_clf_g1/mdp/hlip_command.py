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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  transform_points,
  wrap_to_pi,
  yaw_quat,
)

from hlip_clf_g1.mdp.clf import CLF
from hlip_clf_g1.mdp.ref_gen import HLIP, bezier_deg, calculate_cur_swing_foot_pos

if TYPE_CHECKING:
  import viser

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
    self._fixed_velocity_command_enabled = bool(
      getattr(cfg, "fixed_velocity_command_enabled", False)
    )
    self._fixed_lin_vel_x = float(getattr(cfg, "fixed_lin_vel_x", 0.0))
    self._fixed_lin_vel_y = float(getattr(cfg, "fixed_lin_vel_y", 0.0))
    self._resample_velocity_on_step = bool(
      getattr(cfg, "resample_velocity_on_step", False)
    )
    self._resample_velocity_on_step_probability = min(
      max(float(getattr(cfg, "resample_velocity_on_step_probability", 1.0)), 0.0),
      1.0,
    )
    self._yaw_hold_enabled = bool(getattr(cfg, "yaw_hold_enabled", False))
    self._yaw_hold_target = float(getattr(cfg, "yaw_hold_target", 0.0))
    self._yaw_hold_kp = float(getattr(cfg, "yaw_hold_kp", 0.0))
    self._yaw_hold_max_ang_vel = max(
      float(getattr(cfg, "yaw_hold_max_ang_vel", 0.8)),
      0.0,
    )

    # Optional interactive manual command control from the viewer GUI.
    self._manual_cmd_enabled: "viser.GuiCheckboxHandle | None" = None
    self._manual_cmd_sliders: list["viser.GuiSliderHandle"] = []
    self._manual_cmd_get_env_idx: Callable[[], int] | None = None

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
    self._swing_contact_elapsed = torch.zeros(self.num_envs, device=self.device)
    self._swing_no_contact_elapsed = torch.zeros(self.num_envs, device=self.device)
    self._mpc_contact_recompute_cooldown = torch.zeros(
      self.num_envs, device=self.device
    )

    # Post-transition recovery lockout: blocks _recover_stance_from_contact
    # for a short period after any stance transition (normal or recovery)
    # to prevent immediate flip-back loops.  Unlike _mpc_contact_recompute_cooldown
    # this does NOT block MPC replanning.
    self._recovery_lockout = torch.zeros(self.num_envs, device=self.device)

    # Per-tick guard: prevents _update_stance_swing_idx from flipping envs
    # that already transitioned via _recover_stance_from_contact this tick.
    self._recovery_this_tick = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Touchdown-based stance switching.
    self._touchdown_enabled = bool(getattr(cfg, "touchdown_switch_enabled", True))
    self._touchdown_sensor_name = str(getattr(cfg, "touchdown_sensor_name", "feet_ground_contact"))
    self._touchdown_min_phase = float(getattr(cfg, "touchdown_min_phase", 0.60))
    self._touchdown_timeout_phase = float(getattr(cfg, "touchdown_timeout_phase", 1000))
    self._touchdown_contact_grace_s = max(
      float(getattr(cfg, "touchdown_contact_grace_period_s", 0.05)),
      0.0,
    )
    self._touchdown_min_force_n = max(
      float(getattr(cfg, "touchdown_min_force_n", 75.0)),
      0.0,
    )
    self._touchdown_debounce_s = max(
      float(getattr(cfg, "touchdown_debounce_s", 0.04)),
      0.0,
    )
    self._mpc_contact_recompute_grace_s = max(
      float(getattr(cfg, "mpc_contact_recompute_grace_period_s", 0.3)),
      0.0,
    )
    self._phase_end_stance_flip_only = bool(
      getattr(cfg, "phase_end_stance_flip_only", False)
    )
    self._mpc_stance_recovery_min_elapsed_s = max(
      float(getattr(cfg, "mpc_stance_recovery_min_elapsed_s", 0.15)),
      0.0,
    )
    self._mpc_contact_recovery_enabled = bool(
      getattr(cfg, "mpc_contact_recovery_enabled", True)
    )
    self._mpc_replan_wait_for_stance_contact = bool(
      getattr(cfg, "mpc_replan_wait_for_stance_contact", True)
    )

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
    self._mpc_edge_distance_cost = max(
      float(getattr(cfg, "mpc_edge_distance_cost", 10.0)),
      0.0,
    )

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
    self._mpc_replan_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._mpc_last_cost = torch.zeros(self.num_envs, device=self.device)
    self._mpc_feasible_steps = torch.zeros(self.num_envs, device=self.device)
    self._mpc_fallback_used = torch.zeros(self.num_envs, device=self.device)

    # Heightmap layout for edge detection.
    self._hm_num_x = 0
    self._hm_num_y = 0
    self._hm_edge_threshold = float(getattr(cfg, "mpc_edge_height_threshold", -1.0))
    self._hm_edge_clearance_distance = max(
      float(getattr(cfg, "mpc_edge_clearance_distance", 0.0)),
      0.0,
    )
    self._hm_max_stance_height_delta = float(
      getattr(cfg, "mpc_max_stance_height_delta", -1.0)
    )
    if "heightmap" in self._env.scene.sensors:
      hm_pattern = self._env.scene.sensors["heightmap"].cfg.pattern
      hm_res = float(hm_pattern.resolution)
      size_x, size_y = hm_pattern.size
      # Derive grid shape from metric extents to avoid floating-step drift.
      self._hm_num_x = int(round(float(size_x) / hm_res)) + 1
      self._hm_num_y = int(round(float(size_y) / hm_res)) + 1
      if self._hm_edge_threshold <= 0.0:
        self._hm_edge_threshold = hm_res * math.tan(math.radians(20.0))

  # ------------------------------------------------------------------
  # Properties
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Foot target (kept for observation compatibility)."""
    return self.foot_target

  def _manual_control_enabled(self) -> bool:
    if self._manual_cmd_enabled is not None:
      return bool(self._manual_cmd_enabled.value)
    return bool(self.cfg.manual_control)

  def _manual_control_target_env_idx(self) -> int:
    if self._manual_cmd_get_env_idx is None:
      return 0
    env_idx = int(self._manual_cmd_get_env_idx())
    return max(0, min(env_idx, self.num_envs - 1))

  def _manual_control_command_values(self) -> tuple[float, float, float]:
    if len(self._manual_cmd_sliders) == 3:
      return (
        float(self._manual_cmd_sliders[0].value),
        float(self._manual_cmd_sliders[1].value),
        float(self._manual_cmd_sliders[2].value),
      )
    return (0.0, 0.0, 0.0)

  def _apply_manual_control(self, env_ids: torch.Tensor | None = None) -> None:
    if not self._manual_control_enabled():
      return

    env_idx = self._manual_control_target_env_idx()
    env_tensor = torch.tensor([env_idx], device=self.device, dtype=torch.long)
    if env_ids is not None:
      if not bool(torch.any(env_ids == env_idx)):
        return

    vx, vy, wz = self._manual_control_command_values()
    self.vel_command[env_tensor, 0] = vx
    self.vel_command[env_tensor, 1] = vy
    self.vel_command[env_tensor, 2] = wz
    self.is_standing_env[env_tensor] = False

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create manual velocity-command sliders."""
    from viser import Icon

    ranges = self.cfg.ranges
    axis_specs = (
      ("lin_vel_x", max(abs(ranges.lin_vel_x[0]), abs(ranges.lin_vel_x[1]))),
      ("lin_vel_y", max(abs(ranges.lin_vel_y[0]), abs(ranges.lin_vel_y[1]))),
      ("ang_vel_z", max(abs(ranges.ang_vel_z[0]), abs(ranges.ang_vel_z[1]))),
    )
    sliders: list["viser.GuiSliderHandle"] = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox(
        "Manual Control",
        initial_value=bool(self.cfg.manual_control),
      )

      for label, max_abs in axis_specs:
        slider = server.gui.add_slider(
          label,
          min=-max_abs,
          max=max_abs,
          step=0.01,
          initial_value=0.0,
        )
        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in sliders:
          slider.value = 0.0

      # Optional runtime waist PD tuning controls.
      try:
        upper_body_lock = self._env.action_manager.get_term("upper_body_pd_lock")
      except Exception:
        upper_body_lock = None

      if upper_body_lock is not None and hasattr(upper_body_lock, "get_waist_pd_gains"):
        waist_gains = upper_body_lock.get_waist_pd_gains()
        if waist_gains:
          pd_sliders: dict[str, tuple["viser.GuiSliderHandle", "viser.GuiSliderHandle"]] = {}
          default_waist_gains = dict(waist_gains)

          for joint_name, (kp0, kd0) in sorted(waist_gains.items()):
            label = joint_name.replace("_joint", "")
            kp_max = max(100.0, 3.0 * float(kp0))
            kd_max = max(5.0, 3.0 * float(kd0))

            kp_slider = server.gui.add_slider(
              f"{label} kp",
              min=0.0,
              max=kp_max,
              step=max(0.1, kp_max / 500.0),
              initial_value=float(kp0),
            )
            kd_slider = server.gui.add_slider(
              f"{label} kd",
              min=0.0,
              max=kd_max,
              step=max(0.01, kd_max / 500.0),
              initial_value=float(kd0),
            )
            pd_sliders[joint_name] = (kp_slider, kd_slider)

            @kp_slider.on_update
            def _(_ev, _joint=joint_name, _kp=kp_slider, _kd=kd_slider) -> None:
              upper_body_lock.set_waist_pd_gains(
                _joint,
                kp=float(_kp.value),
                kd=float(_kd.value),
              )

            @kd_slider.on_update
            def _(_ev, _joint=joint_name, _kp=kp_slider, _kd=kd_slider) -> None:
              upper_body_lock.set_waist_pd_gains(
                _joint,
                kp=float(_kp.value),
                kd=float(_kd.value),
              )

          reset_waist_pd = server.gui.add_button("Reset Waist PD")

          @reset_waist_pd.on_click
          def _(_) -> None:
            for joint_name, (kp_def, kd_def) in default_waist_gains.items():
              kp_slider, kd_slider = pd_sliders[joint_name]
              kp_slider.value = float(kp_def)
              kd_slider.value = float(kd_def)
              upper_body_lock.set_waist_pd_gains(
                joint_name,
                kp=float(kp_def),
                kd=float(kd_def),
              )

    self._manual_cmd_enabled = enabled
    self._manual_cmd_sliders = sliders
    self._manual_cmd_get_env_idx = get_env_idx

  def _compute_steppability_masks(
    self,
    hit_pos_w: torch.Tensor,
    distances: torch.Tensor,
    stance_z_w: torch.Tensor | None = None,
    sensor_pos_w: torch.Tensor | None = None,
    sensor_quat_w: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify steppable, near-edge, and height-blocked heightmap cells.

    A cell is edge-safe when all four cardinal neighbors are valid and
    height differences are below the
    configured threshold. Cells with world height below zero are marked as
    hard unsteppable, then a clearance margin is applied from blocked cells.
    Cells that pass edge checks but fail clearance are marked as near-edge.
    """
    del stance_z_w

    valid_hits = distances >= 0.0
    inf_dist = torch.full_like(distances, float("inf"))
    if self._hm_num_x <= 0 or self._hm_num_y <= 0:
      return valid_hits, torch.zeros_like(valid_hits), torch.zeros_like(valid_hits), inf_dist

    num_envs, num_rays, _ = hit_pos_w.shape
    if num_rays != self._hm_num_x * self._hm_num_y:
      return valid_hits, torch.zeros_like(valid_hits), torch.zeros_like(valid_hits), inf_dist

    hm_sensor = self._env.scene.sensors.get("heightmap", None)
    if hm_sensor is None:
      return valid_hits, torch.zeros_like(valid_hits), torch.zeros_like(valid_hits), inf_dist

    pattern = hm_sensor.cfg.pattern
    resolution = float(pattern.resolution)

    def _classify_from_grids(
      z_grid: torch.Tensor,
      d_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      z_pad = torch.full(
        (num_envs, self._hm_num_y + 2, self._hm_num_x + 2),
        float("inf"),
        device=self.device,
        dtype=z_grid.dtype,
      )
      d_pad = torch.full(
        (num_envs, self._hm_num_y + 2, self._hm_num_x + 2),
        -1.0,
        device=self.device,
        dtype=d_grid.dtype,
      )
      z_pad[:, 1:-1, 1:-1] = z_grid
      d_pad[:, 1:-1, 1:-1] = d_grid

      z_up = z_pad[:, :-2, 1:-1]
      z_down = z_pad[:, 2:, 1:-1]
      z_left = z_pad[:, 1:-1, :-2]
      z_right = z_pad[:, 1:-1, 2:]

      d_up = d_pad[:, :-2, 1:-1]
      d_down = d_pad[:, 2:, 1:-1]
      d_left = d_pad[:, 1:-1, :-2]
      d_right = d_pad[:, 1:-1, 2:]

      th = self._hm_edge_threshold
      c_up = (d_up >= 0.0) & (torch.abs(z_grid - z_up) <= th)
      c_down = (d_down >= 0.0) & (torch.abs(z_grid - z_down) <= th)
      c_left = (d_left >= 0.0) & (torch.abs(z_grid - z_left) <= th)
      c_right = (d_right >= 0.0) & (torch.abs(z_grid - z_right) <= th)

      valid_grid = d_grid >= 0.0
      is_non_edge_grid = c_up & c_down & c_left & c_right & valid_grid

      # Absolute world-z steppability rule: below-ground hits are unsteppable.
      height_blocked_grid = valid_grid & (z_grid < 0.0)

      edge_distance_grid = torch.full_like(z_grid, float("inf"))
      clearance_cells = 0
      if self._hm_edge_clearance_distance > 0.0:
        clearance_cells = max(
          int(math.ceil(self._hm_edge_clearance_distance / max(resolution, 1.0e-6))),
          1,
        )
        edge_core = (valid_grid & (~is_non_edge_grid) & (~height_blocked_grid)).unsqueeze(1).to(dtype=torch.float32)
        if torch.any(edge_core > 0.5):
          edge_distance_cells = torch.full_like(z_grid, float("inf"))
          edge_distance_cells[edge_core.squeeze(1) > 0.5] = 0.0
          for radius in range(1, clearance_cells + 1):
            neighborhood = F.max_pool2d(
              edge_core,
              kernel_size=2 * radius + 1,
              stride=1,
              padding=radius,
            ).squeeze(1) > 0.5
            edge_distance_cells = torch.where(
              neighborhood & (edge_distance_cells > float(radius)),
              torch.full_like(edge_distance_cells, float(radius)),
              edge_distance_cells,
            )
          edge_distance_grid = edge_distance_cells * resolution

      near_edge_grid = torch.zeros_like(is_non_edge_grid)
      if clearance_cells > 0:
        near_edge_grid = (
          is_non_edge_grid
          & (edge_distance_grid < self._hm_edge_clearance_distance)
        )

      # Height-blocked cells are hard unsteppable regardless of edge class.
      near_edge_grid = near_edge_grid & (~height_blocked_grid)
      steppable_grid = is_non_edge_grid & (~height_blocked_grid)
      return steppable_grid, near_edge_grid, height_blocked_grid, edge_distance_grid

    if sensor_pos_w is None or sensor_quat_w is None:
      hm_data = hm_sensor.data
      if (
        hm_data.pos_w is not None
        and hm_data.quat_w is not None
        and hm_data.pos_w.shape[0] == num_envs
        and hm_data.quat_w.shape[0] == num_envs
      ):
        sensor_pos_w = hm_data.pos_w
        sensor_quat_w = hm_data.quat_w
      else:
        # Fall back to legacy reshape behavior when sensor pose batch is ambiguous.
        z = hit_pos_w[:, :, 2].reshape(num_envs, self._hm_num_y, self._hm_num_x)
        d = distances.reshape(num_envs, self._hm_num_y, self._hm_num_x)
        steppable, near_edge, height_blocked, edge_dist = _classify_from_grids(z, d)
        return (
          steppable.reshape(num_envs, num_rays),
          near_edge.reshape(num_envs, num_rays),
          height_blocked.reshape(num_envs, num_rays),
          edge_dist.reshape(num_envs, num_rays),
        )

    assert sensor_pos_w is not None
    assert sensor_quat_w is not None

    size_x, size_y = pattern.size

    # Convert hit positions to sensor-local XY to recover per-ray grid indices.
    points_rel = hit_pos_w - sensor_pos_w.unsqueeze(1)
    points_local = transform_points(points_rel, quat=quat_inv(sensor_quat_w))
    gx = (points_local[..., 0] + 0.5 * float(size_x)) / resolution
    gy = (points_local[..., 1] + 0.5 * float(size_y)) / resolution
    ix = torch.floor(gx + 0.5).to(torch.long)
    iy = torch.floor(gy + 0.5).to(torch.long)
    in_bounds = (
      (ix >= 0) & (ix < self._hm_num_x) &
      (iy >= 0) & (iy < self._hm_num_y)
    )

    # Guard against quantization collisions/holes at finer resolutions:
    # if scatter is not bijective, use the deterministic ray-order reshape path.
    num_cells = self._hm_num_x * self._hm_num_y
    flat_idx = iy * self._hm_num_x + ix
    flat_idx_safe = flat_idx.clamp(min=0, max=max(num_cells - 1, 0))
    counts = torch.zeros(
      (num_envs, num_cells),
      device=self.device,
      dtype=torch.long,
    )
    counts.scatter_add_(1, flat_idx_safe, in_bounds.to(dtype=torch.long))
    one_to_one = torch.all(in_bounds, dim=1) & torch.all(counts == 1, dim=1)
    if not bool(torch.all(one_to_one)):
      z = hit_pos_w[:, :, 2].reshape(num_envs, self._hm_num_y, self._hm_num_x)
      d = distances.reshape(num_envs, self._hm_num_y, self._hm_num_x)
      steppable, near_edge, height_blocked, edge_dist = _classify_from_grids(z, d)
      return (
        steppable.reshape(num_envs, num_rays),
        near_edge.reshape(num_envs, num_rays),
        height_blocked.reshape(num_envs, num_rays),
        edge_dist.reshape(num_envs, num_rays),
      )

    # Scatter ray data into a canonical [num_y, num_x] grid.
    z_grid = torch.full(
      (num_envs, self._hm_num_y, self._hm_num_x),
      float("inf"),
      device=self.device,
      dtype=hit_pos_w.dtype,
    )
    d_grid = torch.full(
      (num_envs, self._hm_num_y, self._hm_num_x),
      -1.0,
      device=self.device,
      dtype=distances.dtype,
    )
    batch_ids = torch.arange(num_envs, device=self.device, dtype=torch.long).unsqueeze(1).expand(num_envs, num_rays)
    valid_scatter = in_bounds & valid_hits
    z_world = hit_pos_w[..., 2]
    z_grid[batch_ids[valid_scatter], iy[valid_scatter], ix[valid_scatter]] = z_world[valid_scatter]
    d_grid[batch_ids[valid_scatter], iy[valid_scatter], ix[valid_scatter]] = distances[valid_scatter]
    steppable_grid, near_edge_grid, height_blocked_grid, edge_dist_grid = _classify_from_grids(z_grid, d_grid)

    ix_safe = ix.clamp(min=0, max=self._hm_num_x - 1)
    iy_safe = iy.clamp(min=0, max=self._hm_num_y - 1)
    steppable = steppable_grid[batch_ids, iy_safe, ix_safe]
    near_edge = near_edge_grid[batch_ids, iy_safe, ix_safe]
    height_blocked = height_blocked_grid[batch_ids, iy_safe, ix_safe]
    edge_distance = edge_dist_grid[batch_ids, iy_safe, ix_safe]
    mask_valid = in_bounds & valid_hits
    edge_distance = torch.where(
      mask_valid,
      edge_distance,
      torch.full_like(edge_distance, float("inf")),
    )
    return steppable & mask_valid, near_edge & mask_valid, height_blocked & mask_valid, edge_distance

  def _compute_non_edge_mask(
    self,
    hit_pos_w: torch.Tensor,
    distances: torch.Tensor,
    stance_z_w: torch.Tensor | None = None,
    sensor_pos_w: torch.Tensor | None = None,
    sensor_quat_w: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Backward-compatible steppability mask used by existing MPC code paths."""
    steppable, _, _, _ = self._compute_steppability_masks(
      hit_pos_w,
      distances,
      stance_z_w,
      sensor_pos_w,
      sensor_quat_w,
    )
    return steppable

  def _classify_camera_rays_by_heightmap(
    self,
    env_idx: int,
  ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Classify camera rays as steppable, near-edge, or height-blocked."""
    if "head_camera_rays" not in self._env.scene.sensors:
      return None, None, None
    if "heightmap" not in self._env.scene.sensors:
      return None, None, None

    ray_sensor = self._env.scene.sensors["head_camera_rays"]
    ray_data = ray_sensor.data
    hm_sensor = self._env.scene.sensors["heightmap"]
    hm_data = hm_sensor.data

    if ray_data.hit_pos_w is None or ray_data.distances is None:
      return None, None, None
    if (
      hm_data.hit_pos_w is None
      or hm_data.distances is None
      or hm_data.pos_w is None
      or hm_data.quat_w is None
    ):
      return None, None, None

    hit_pos_w = ray_data.hit_pos_w[env_idx : env_idx + 1]
    ray_dist = ray_data.distances[env_idx : env_idx + 1]
    hm_hit = hm_data.hit_pos_w[env_idx : env_idx + 1]
    hm_dist = hm_data.distances[env_idx : env_idx + 1]
    hm_pos = hm_data.pos_w[env_idx : env_idx + 1]
    hm_quat = hm_data.quat_w[env_idx : env_idx + 1]
    stance_z = self.stance_foot_pos_0[env_idx, 2].view(1)

    steppable_cells, near_edge_cells, height_blocked_cells, _ = self._compute_steppability_masks(
      hm_hit,
      hm_dist,
      stance_z,
      sensor_pos_w=hm_pos,
      sensor_quat_w=hm_quat,
    )

    num_hm_rays = int(steppable_cells.shape[1])
    hm_pattern = hm_sensor.cfg.pattern
    size_x, size_y = hm_pattern.size
    resolution = float(hm_pattern.resolution)

    num_x = self._hm_num_x
    num_y = self._hm_num_y
    if num_x <= 0 or num_y <= 0 or num_x * num_y != num_hm_rays:
      num_x = int(round(size_x / resolution)) + 1
      num_y = int(round(size_y / resolution)) + 1
    if num_x * num_y != num_hm_rays:
      return None, None, None

    num_rays = int(ray_dist.shape[1])
    ray_idx_lut = torch.full((1, num_y, num_x), -1, device=self.device, dtype=torch.long)
    hm_points_rel = hm_hit - hm_pos.unsqueeze(1)
    hm_points_local = transform_points(hm_points_rel, quat=quat_inv(hm_quat))
    hm_gx = (hm_points_local[..., 0] + 0.5 * float(size_x)) / resolution
    hm_gy = (hm_points_local[..., 1] + 0.5 * float(size_y)) / resolution
    hm_ix = torch.floor(hm_gx + 0.5).to(torch.long)
    hm_iy = torch.floor(hm_gy + 0.5).to(torch.long)
    hm_in_bounds = (hm_ix >= 0) & (hm_ix < num_x) & (hm_iy >= 0) & (hm_iy < num_y)

    hm_batch = torch.zeros((1, num_hm_rays), device=self.device, dtype=torch.long)
    hm_ray_ids = torch.arange(num_hm_rays, device=self.device, dtype=torch.long).unsqueeze(0)
    ray_idx_lut[hm_batch[hm_in_bounds], hm_iy[hm_in_bounds], hm_ix[hm_in_bounds]] = hm_ray_ids[hm_in_bounds]

    ray_points_rel = hit_pos_w - hm_pos.unsqueeze(1)
    ray_points_local = transform_points(ray_points_rel, quat=quat_inv(hm_quat))
    gx = (ray_points_local[..., 0] + 0.5 * float(size_x)) / resolution
    gy = (ray_points_local[..., 1] + 0.5 * float(size_y)) / resolution
    ix = torch.floor(gx + 0.5).to(torch.long)
    iy = torch.floor(gy + 0.5).to(torch.long)

    in_bounds = (ix >= 0) & (ix < num_x) & (iy >= 0) & (iy < num_y)
    ix_safe = ix.clamp(min=0, max=num_x - 1)
    iy_safe = iy.clamp(min=0, max=num_y - 1)
    batch = torch.zeros((1, num_rays), device=self.device, dtype=torch.long)
    flat_idx = ray_idx_lut[batch, iy_safe, ix_safe]
    lut_valid = flat_idx >= 0
    flat_idx_safe = flat_idx.clamp(min=0, max=num_hm_rays - 1)

    steppable_flat = torch.gather(steppable_cells, dim=1, index=flat_idx_safe)
    near_edge_flat = torch.gather(near_edge_cells, dim=1, index=flat_idx_safe)
    height_blocked_flat = torch.gather(height_blocked_cells, dim=1, index=flat_idx_safe)
    valid_mask = in_bounds & lut_valid & (ray_dist >= 0.0)

    steppable_flat = steppable_flat & valid_mask
    near_edge_flat = near_edge_flat & valid_mask
    height_blocked_flat = height_blocked_flat & valid_mask
    return steppable_flat.squeeze(0), near_edge_flat.squeeze(0), height_blocked_flat.squeeze(0)

  def _get_heightmap_candidates(
    self,
    env_ids: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return stance-local heightmap hits, validity, and edge penalties."""
    if "heightmap" not in self._env.scene.sensors:
      return None

    hm_data = self._env.scene.sensors["heightmap"].data
    if hm_data.hit_pos_w is None or hm_data.distances is None:
      return None

    hit_pos_w = hm_data.hit_pos_w[env_ids]
    distances = hm_data.distances[env_ids]
    sensor_pos_w = hm_data.pos_w[env_ids]
    sensor_quat_w = hm_data.quat_w[env_ids]
    stance_z = self.stance_foot_pos_0[env_ids, 2]
    _, _, height_blocked, edge_distance = self._compute_steppability_masks(
      hit_pos_w,
      distances,
      stance_z,
      sensor_pos_w=sensor_pos_w,
      sensor_quat_w=sensor_quat_w,
    )
    valid_hits = (distances >= 0.0) & (~height_blocked)
    edge_penalty = torch.zeros_like(distances)
    if self._hm_edge_clearance_distance > 0.0 and self._mpc_edge_distance_cost > 0.0:
      clearance = self._hm_edge_clearance_distance
      penetration = torch.clamp(clearance - edge_distance, min=0.0)
      edge_penalty = self._mpc_edge_distance_cost * (penetration / clearance) ** 2
      edge_penalty = torch.where(valid_hits, edge_penalty, torch.zeros_like(edge_penalty))

    n, num_rays, _ = hit_pos_w.shape
    rel_w = hit_pos_w - self.stance_foot_pos_0[env_ids].unsqueeze(1)
    rel_w_flat = rel_w.reshape(-1, 3)
    stance_quat = self.stance_foot_ori_quat_0[env_ids].repeat_interleave(num_rays, dim=0)
    rel_local = _to_local_frame(rel_w_flat, stance_quat).reshape(n, num_rays, 3)

    return rel_local, valid_hits, edge_penalty

  def _solve_discrete_mpc(
    self,
    env_ids: torch.Tensor,
    cmd_xy: torch.Tensor,
    local_hits: torch.Tensor,
    valid_hits: torch.Tensor,
    edge_penalty: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve a discrete MPC preview by minimizing velocity error over feasible candidates."""
    n = len(env_ids)
    horizon = self._mpc_horizon
    num_t = len(self._mpc_t_grid)
    num_rays = local_hits.shape[1]
    num_cands = num_rays * num_t
    
    # -----------------------------------------------------------------
    # ITERATIVE STOCHASTIC ROLLOUT
    # Evaluate S independent step sequences.
    # Sample 0 is strictly greedy. Samples 1..(S-1) add Gumbel noise.
    # We iterate over S to save VRAM compared to expanding all tensors.
    # -----------------------------------------------------------------
    S = 16  # Iterative loop allows large S with very low VRAM usage
    temperature = 0.1  # Exploration noise scaling
    
    # Pre-flatten candidates to simplify broadcasting
    cand_x = local_hits[:, :, 0].unsqueeze(-1).expand(n, num_rays, num_t).reshape(n, num_cands)
    cand_y = local_hits[:, :, 1].unsqueeze(-1).expand(n, num_rays, num_t).reshape(n, num_cands)
    cand_z = local_hits[:, :, 2].unsqueeze(-1).expand(n, num_rays, num_t).reshape(n, num_cands)
    cand_valid = valid_hits.unsqueeze(-1).expand(n, num_rays, num_t).reshape(n, num_cands)
    cand_penalty = edge_penalty.unsqueeze(-1).expand(n, num_rays, num_t).reshape(n, num_cands)
    cand_t = self._mpc_t_grid.view(1, num_t).expand(n, num_rays, num_t).reshape(n, num_cands)

    best_plan_t = torch.full((n, horizon), self._mpc_t_nom, device=self.device)
    best_plan_foot = torch.zeros(n, horizon, 3, device=self.device)
    best_total_cost = torch.full((n,), float("inf"), device=self.device)
    best_all_feasible = torch.zeros(n, dtype=torch.bool, device=self.device)
    best_feasible_steps = torch.zeros(n, device=self.device)

    for s in range(S):
      plan_t_s = torch.full((n, horizon), self._mpc_t_nom, device=self.device)
      plan_foot_s = torch.zeros(n, horizon, 3, device=self.device)
      total_cost_s = torch.zeros(n, device=self.device)
      all_feasible_s = torch.ones(n, dtype=torch.bool, device=self.device)
      feasible_steps_s = torch.zeros(n, device=self.device)
      
      prev_foot_abs = torch.zeros(n, 3, device=self.device)

      for k in range(horizon):
        stance_k = torch.remainder(self.stance_idx[env_ids] + k, 2)
        side_sign = torch.where(
          stance_k == 0,
          torch.full((n,), -1.0, device=self.device),
          torch.full((n,), 1.0, device=self.device),
        )
        side_offset = side_sign * self.y_nom
        
        cmd_x = cmd_xy[:, 0].view(n, 1)
        cmd_y = cmd_xy[:, 1].view(n, 1)
        side_offset_e = side_offset.view(n, 1)
        side_sign_e = side_sign.view(n, 1)

        dx = cand_x - prev_foot_abs[:, 0].unsqueeze(-1)
        dy = cand_y - prev_foot_abs[:, 1].unsqueeze(-1)
        dt = cand_t

        x_ok = (dx >= self._mpc_x_min) & (dx <= self._mpc_x_max)
        dy_abs = torch.abs(dy)
        y_ok = (dy_abs >= self._mpc_abs_y_min) & (dy_abs <= self._mpc_abs_y_max)
        step_len_ok = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=1e-8)) <= self._mpc_max_step_length
        signed_y_ok = (dy * side_sign_e) >= self._mpc_signed_y_min
        
        feasible = cand_valid & x_ok & y_ok & step_len_ok & signed_y_ok

        vx_pred = dx / dt
        vy_pred = (dy - side_offset_e) / dt
        vel_err = (vx_pred - cmd_x) ** 2 + (vy_pred - cmd_y) ** 2
        
        cost = vel_err + cand_penalty
        cost = torch.where(feasible, cost, torch.full_like(cost, float("inf")))

        if s == 0:
            noisy_cost = cost
        else:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(cost) + 1e-10))
            noisy_cost = cost + temperature * gumbel_noise

        _, chosen_idx = torch.min(noisy_cost, dim=-1)  # (n,)
        
        chosen_t = torch.gather(cand_t, 1, chosen_idx.unsqueeze(-1)).squeeze(-1)
        chosen_x = torch.gather(cand_x, 1, chosen_idx.unsqueeze(-1)).squeeze(-1)
        chosen_y = torch.gather(cand_y, 1, chosen_idx.unsqueeze(-1)).squeeze(-1)
        chosen_z = torch.gather(cand_z, 1, chosen_idx.unsqueeze(-1)).squeeze(-1)
        chosen_foot = torch.stack([chosen_x, chosen_y, chosen_z], dim=-1)
        
        true_cost = torch.gather(cost, 1, chosen_idx.unsqueeze(-1)).squeeze(-1)
        feasible_k = torch.isfinite(true_cost)
        
        feasible_steps_s += feasible_k.float()
        all_feasible_s &= feasible_k

        # Fallback logic
        fallback_foot = prev_foot_abs.clone()
        fallback_foot[:, 0] += cmd_xy[:, 0] * self._mpc_t_nom
        fallback_dy = (side_offset + cmd_xy[:, 1]) * self._mpc_t_nom
        fallback_dy_abs = torch.clamp(
          torch.abs(fallback_dy),
          min=self._mpc_abs_y_min,
          max=self._mpc_abs_y_max,
        )
        fallback_dy_sign = torch.sign(fallback_dy)
        fallback_dy_sign = torch.where(
          torch.abs(fallback_dy_sign) > 0.0,
          fallback_dy_sign,
          side_sign,
        )
        fallback_foot[:, 1] += fallback_dy_abs * fallback_dy_sign
        
        fb_xy = fallback_foot[:, :2]  # (n, 2)
        hit_xy = local_hits[:, :, :2]  # (n, num_rays, 2)
        dist_sq = ((hit_xy - fb_xy.unsqueeze(1)) ** 2).sum(dim=-1)
        dist_sq = torch.where(valid_hits, dist_sq, torch.full_like(dist_sq, float("inf")))
        nearest_idx = dist_sq.argmin(dim=-1)
        nearest_z = torch.gather(local_hits[:, :, 2], 1, nearest_idx.unsqueeze(-1)).squeeze(-1)
        nearest_valid = torch.gather(valid_hits, 1, nearest_idx.unsqueeze(-1)).squeeze(-1)
        fallback_foot[:, 2] = torch.where(nearest_valid, nearest_z, torch.zeros_like(nearest_z))

        plan_t_s[:, k] = torch.where(
          feasible_k,
          chosen_t,
          torch.full_like(chosen_t, self._mpc_t_nom),
        )
        plan_foot_s[:, k, :] = torch.where(
          feasible_k.unsqueeze(-1),
          chosen_foot,
          fallback_foot,
        )
        prev_foot_abs = plan_foot_s[:, k, :]
        total_cost_s += torch.where(feasible_k, true_cost, torch.zeros_like(true_cost))

      # Heavily penalize samples that resulted in fallbacks
      infeasibility_penalty = 1e6 * (horizon - feasible_steps_s)
      total_cost_s += infeasibility_penalty
      
      # Update best tracking
      better_mask = total_cost_s < best_total_cost
      
      best_total_cost = torch.where(better_mask, total_cost_s, best_total_cost)
      best_all_feasible = torch.where(better_mask, all_feasible_s, best_all_feasible)
      best_feasible_steps = torch.where(better_mask, feasible_steps_s, best_feasible_steps)
      
      best_plan_t = torch.where(better_mask.unsqueeze(-1), plan_t_s, best_plan_t)
      best_plan_foot = torch.where(better_mask.unsqueeze(-1).unsqueeze(-1), plan_foot_s, best_plan_foot)

    return best_plan_t, best_plan_foot, best_total_cost, best_all_feasible, best_feasible_steps

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
        side_sign = torch.where(
          torch.remainder(self.stance_idx[env_ids] + k, 2) == 0,
          torch.full((n,), -1.0, device=self.device),
          torch.full((n,), 1.0, device=self.device),
        )
        nominal_dy = side_sign * self.y_nom
        dy_abs = torch.clamp(
          torch.abs(nominal_dy),
          min=self._mpc_abs_y_min,
          max=self._mpc_abs_y_max,
        )
        prev_foot[:, 1] += dy_abs * side_sign
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

    local_hits, valid_hits, edge_penalty = candidates
    base_cmd = self.vel_command[env_ids, :2].clone()

    best_t, best_foot, best_cost, best_feasible, feasible_steps = self._solve_discrete_mpc(
      env_ids,
      base_cmd,
      local_hits,
      valid_hits,
      edge_penalty,
    )
    best_cmd = base_cmd
    fallback_used = ~best_feasible

    self._mpc_plan_t[env_ids] = best_t
    self._mpc_plan_foothold[env_ids] = best_foot
    self._mpc_active_foot_target[env_ids] = best_foot[:, 0]
    self._mpc_last_cost[env_ids] = torch.where(
      torch.isfinite(best_cost), best_cost, torch.zeros_like(best_cost)
    )
    self._mpc_feasible_steps[env_ids] = feasible_steps
    self._mpc_fallback_used[env_ids] = fallback_used.float()
    self._mpc_has_plan[env_ids] = True
    self._mpc_replan_pending[env_ids] = False
    self.vel_command[env_ids, :2] = best_cmd
    self.T[env_ids] = torch.clamp(best_t[:, 0], self.T_min, self.T_max)
    self.full_gait_period[env_ids] = 2.0 * (self.T[env_ids] - self.T_ds)

  # ------------------------------------------------------------------
  # Phase tracking
  # ------------------------------------------------------------------

  def _get_foot_contact_state(self) -> torch.Tensor:
    """Return binary left/right foot contacts as ``(num_envs, 2)`` bool.

    Contact requires both a positive ``found`` flag *and* a net contact
    force magnitude above ``_touchdown_min_force_n``.  This filters out
    toe-grazing contacts that register as found but carry negligible
    ground reaction force.
    """
    if self._touchdown_sensor_name not in self._env.scene.sensors:
      return torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

    sensor = self._env.scene[self._touchdown_sensor_name]
    found = getattr(sensor.data, "found", None)
    if found is None:
      return torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

    has_contact = found > 0

    # Optionally gate on force magnitude to reject toe-grazing contacts.
    if self._touchdown_min_force_n > 0.0:
      force = getattr(sensor.data, "force", None)
      if force is not None:
        # force shape: (num_envs, num_feet, 3) — net force in global frame.
        force_mag = torch.linalg.norm(force, dim=-1)  # (num_envs, num_feet)
        has_contact = has_contact & (force_mag >= self._touchdown_min_force_n)

    return has_contact

  def _get_stance_contact_state(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Return stance-foot contact state for the provided environments."""
    if len(env_ids) == 0:
      return torch.zeros(0, dtype=torch.bool, device=self.device)

    foot_contact = self._get_foot_contact_state()
    return torch.gather(
      foot_contact[env_ids],
      1,
      self.stance_idx[env_ids].unsqueeze(1),
    ).squeeze(1)

  def _start_contact_recompute_cooldown(self, env_ids: torch.Tensor) -> None:
    """Start post-contact-replan cooldown for the provided environments."""
    if len(env_ids) == 0 or self._mpc_contact_recompute_grace_s <= 0.0:
      return
    self._mpc_contact_recompute_cooldown[env_ids] = self._mpc_contact_recompute_grace_s

  def _recover_stance_from_contact(
    self,
    foot_contact: torch.Tensor | None = None,
  ) -> None:
    """Recover stance assignment when MPC-assumed stance loses contact.

    If the currently assumed stance foot is not in contact while the other
    foot is, we switch stance to the contacting foot and trigger MPC replanning.

    Sets ``_recovery_this_tick`` for recovered envs so that
    ``_update_stance_swing_idx`` will not attempt a second transition on the
    same tick.
    """
    if (not self._mpc_enabled) or (not self._mpc_contact_recovery_enabled):
      return

    active_mask = ~self._pending_stance_init
    if not torch.any(active_mask):
      return

    if foot_contact is None:
      foot_contact = self._get_foot_contact_state()
    stance_contact = torch.gather(
      foot_contact,
      1,
      self.stance_idx.unsqueeze(1),
    ).squeeze(1)
    swing_contact = torch.gather(
      foot_contact,
      1,
      self.swing_idx.unsqueeze(1),
    ).squeeze(1)
    swing_contact_ready = self._swing_contact_elapsed >= self._touchdown_contact_grace_s

    T_swing = torch.clamp(self.T - self.T_ds, min=1e-6)
    phase_var = torch.clamp(self._swing_elapsed / T_swing, min=0.0, max=1.0)

    # Avoid recovering too early in the half-step to reduce contact jitter flips.
    past_min_elapsed = self._swing_elapsed >= self._mpc_stance_recovery_min_elapsed_s
    phase_ready = phase_var >= self._touchdown_min_phase
    # Require sustained swing-foot contact to reduce jitter-driven flips.
    recover_ids = (
      active_mask
      & past_min_elapsed
      & phase_ready
      & (~stance_contact)
      & swing_contact
      & swing_contact_ready
      & (self._mpc_contact_recompute_cooldown <= 0.0)
      & (self._recovery_lockout <= 0.0)
    ).nonzero(as_tuple=False).flatten()
    if len(recover_ids) == 0:
      return

    foot_pos_w = self.robot.data.body_link_pos_w[:, self._foot_body_ids, :]
    foot_quat_w = self.robot.data.body_link_quat_w[:, self._foot_body_ids, :]

    new_stance = self.swing_idx[recover_ids]
    new_swing = 1 - new_stance

    self.stance_idx[recover_ids] = new_stance
    self.swing_idx[recover_ids] = new_swing
    self._prev_stance_idx[recover_ids] = new_stance
    self._swing_elapsed[recover_ids] = 0.0
    self._swing_contact_elapsed[recover_ids] = 0.0

    # Mark these envs so _update_stance_swing_idx skips them this tick.
    self._recovery_this_tick[recover_ids] = True

    self.stance_foot_pos_0[recover_ids] = foot_pos_w[recover_ids, new_stance, :]
    stance_quat = foot_quat_w[recover_ids, new_stance, :]
    self.stance_foot_ori_quat_0[recover_ids] = stance_quat
    self.stance_foot_ori_0[recover_ids] = _euler_from_quat(stance_quat)

    swing_pos_w = foot_pos_w[recover_ids, new_swing, :]
    self.swing_foot_pos_0[recover_ids] = _to_local_frame(
      swing_pos_w - self.stance_foot_pos_0[recover_ids],
      self.stance_foot_ori_quat_0[recover_ids],
    )

    self._prev_foot_contact[recover_ids] = foot_contact[recover_ids]
    self._mpc_has_plan[recover_ids] = False
    self._mpc_replan_pending[recover_ids] = True

    self._plan_mpc_transition(recover_ids)
    self._start_contact_recompute_cooldown(recover_ids)

  def _update_stance_swing_idx(
    self,
    dt: float,
    foot_contact: torch.Tensor | None = None,
  ) -> None:
    """Update stance/swing indices from touchdown events.

    Primary trigger is swing-foot touchdown from contact sensing. A timeout
    fallback forces transition if touchdown is missing for too long.

    Envs that were already transitioned by ``_recover_stance_from_contact``
    this tick (marked via ``_recovery_this_tick``) are excluded from further
    transitions to prevent same-tick double stance flips.
    """
    self._swing_elapsed += dt

    T_swing = torch.clamp(self.T - self.T_ds, min=1e-6)
    phase_var = self._swing_elapsed / T_swing

    # Build a mask of envs eligible for normal transition (exclude recovered).
    eligible_mask = ~self._recovery_this_tick

    trans_ids = torch.empty(0, dtype=torch.long, device=self.device)
    if self._phase_end_stance_flip_only:
      # Teacher mode: switch strictly at half-step end.
      timeout_ids = (eligible_mask & (phase_var >= 1.0)).nonzero(as_tuple=False).flatten()
      trans_ids = timeout_ids
    elif self._touchdown_enabled:
      if foot_contact is None:
        foot_contact = self._get_foot_contact_state()
      swing_contact = torch.gather(
        foot_contact,
        1,
        self.swing_idx.unsqueeze(1),
      ).squeeze(1)
      touchdown = swing_contact & (
        self._swing_contact_elapsed >= self._touchdown_contact_grace_s
      )
      touchdown_ready = phase_var >= self._touchdown_min_phase
      # Switch stance when the swing foot has sustained force above the
      # threshold for the required grace period and minimum phase has elapsed.
      trans_ids = (eligible_mask & touchdown & touchdown_ready).nonzero(as_tuple=False).flatten()

      self._prev_foot_contact = foot_contact.clone()
    else:
      # Fallback to fixed-time half-step switching.
      timeout_ids = (eligible_mask & (phase_var >= 1.0)).nonzero(as_tuple=False).flatten()
      trans_ids = timeout_ids

    if len(trans_ids) > 0:
      # Compute new stance/swing only for transitioning envs.
      new_stance_tid = 1 - self.stance_idx[trans_ids]
      new_swing_tid = 1 - new_stance_tid

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
      self.stance_foot_pos_0[trans_ids] = foot_pos_w[
        trans_ids,
        new_stance_tid,
        :,
      ]
      stance_quat = foot_quat_w[trans_ids, new_stance_tid, :]
      self.stance_foot_ori_quat_0[trans_ids] = stance_quat
      self.stance_foot_ori_0[trans_ids] = _euler_from_quat(stance_quat)

      # Record swing foot position in stance-local frame at transition.
      swing_pos_w = foot_pos_w[trans_ids, new_swing_tid, :]
      self.swing_foot_pos_0[trans_ids] = _to_local_frame(
        swing_pos_w - self.stance_foot_pos_0[trans_ids],
        self.stance_foot_ori_quat_0[trans_ids],
      )

      # Commit the post-transition stance/swing assignment.
      self.stance_idx[trans_ids] = new_stance_tid
      self.swing_idx[trans_ids] = new_swing_tid
      self._prev_stance_idx[trans_ids] = new_stance_tid
      self._swing_elapsed[trans_ids] = 0.0
      self._swing_contact_elapsed[trans_ids] = 0.0
      self._swing_no_contact_elapsed[trans_ids] = 0.0

      # Block recovery from flipping stance back for a cooldown period after
      # every normal transition (touchdown or timeout).  Uses a dedicated
      # lockout that does NOT interfere with MPC replanning.
      self._recovery_lockout[trans_ids] = self._mpc_contact_recompute_grace_s

      if self._mpc_enabled:
        # At every stance switch, invalidate the previous horizon and
        # immediately replan.  The stance foot pose was already recorded
        # above, so we have an accurate anchor for the new plan.  Deferring
        # replanning until stance contact is confirmed can leave a window
        # where _mpc_has_plan is False, causing the reference trajectory to
        # fall back to raw HLIP targets and produce a discontinuous "pop."
        self._mpc_has_plan[trans_ids] = False
        self._mpc_replan_pending[trans_ids] = False
        self._plan_mpc_transition(trans_ids)
      else:
        if self._resample_velocity_on_step:
          sample_ids = trans_ids
          if self._manual_control_enabled():
            manual_idx = self._manual_control_target_env_idx()
            sample_ids = trans_ids[trans_ids != manual_idx]
          if self._resample_velocity_on_step_probability < 1.0:
            sample_mask = (
              torch.rand(len(sample_ids), device=self.device)
              < self._resample_velocity_on_step_probability
            )
            sample_ids = sample_ids[sample_mask]
          self._sample_velocity_command(sample_ids)

        self.T[trans_ids] = torch.empty(len(trans_ids), device=self.device).uniform_(
          self.T_min,
          self.T_max,
        )
        self.full_gait_period[trans_ids] = 2.0 * (self.T[trans_ids] - self.T_ds)

    # Update phase variables for ALL envs (no global stance_idx overwrite).
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
    wr_roll0, wr_pitch0, wr_yaw0 = self.cfg.wrist_ref
    waist_yaw0 = self.cfg.waist_yaw_ref
    waist_pitch0 = self.cfg.waist_pitch_ref
    waist_roll0 = self.cfg.waist_roll_ref

    sh_pitch_amp = sh_pitch0 * forward_vel
    sh_roll_amp = sh_roll0 * torch.ones_like(forward_vel)
    sh_yaw_amp = sh_yaw0 * torch.ones_like(forward_vel)
    elb_amp = elb0 * forward_vel
    wr_roll_amp = wr_roll0 * torch.ones_like(forward_vel)
    wr_pitch_amp = wr_pitch0 * torch.ones_like(forward_vel)
    wr_yaw_amp = wr_yaw0 * torch.ones_like(forward_vel)
    waist_yaw_amp = waist_yaw0 * torch.ones_like(forward_vel)
    waist_pitch_amp = torch.zeros_like(forward_vel)
    waist_roll_amp = waist_roll0 * torch.ones_like(forward_vel)

    amp = torch.stack(
      [
        waist_yaw_amp,
        waist_pitch_amp,
        waist_roll_amp,
        sh_pitch_amp, sh_pitch_amp,   # L/R shoulder pitch
        sh_roll_amp, sh_roll_amp,     # L/R shoulder roll
        sh_yaw_amp, sh_yaw_amp,       # L/R shoulder yaw
        elb_amp, elb_amp,             # L/R elbow
        wr_roll_amp, wr_roll_amp,     # L/R wrist roll
        wr_pitch_amp, wr_pitch_amp,   # L/R wrist pitch
        wr_yaw_amp, wr_yaw_amp,       # L/R wrist yaw
      ],
      dim=1,
    ).to(self.device)

    sign = torch.tensor(
      [1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
      device=self.device,
      dtype=torch.float32,
    )

    offset = torch.tensor(
      [
        math.pi,                       # waist_yaw
        math.pi / 2,                   # waist_pitch
        math.pi / 2,                   # waist_roll
        math.pi / 2, math.pi / 2,     # L/R shoulder pitch
        math.pi / 2, math.pi / 2,     # L/R shoulder roll
        0.0, 0.0,                      # L/R shoulder yaw
        math.pi / 2, math.pi / 2,     # L/R elbow
        math.pi / 2, math.pi / 2,     # L/R wrist roll
        math.pi / 2, math.pi / 2,     # L/R wrist pitch
        0.0, 0.0,                      # L/R wrist yaw
      ],
      device=self.device,
      dtype=torch.float32,
    )

    joint_offset = self.robot.data.default_joint_pos[
      :, self._upper_body_joint_ids
    ]
    if joint_offset.shape[1] != amp.shape[1]:
      raise ValueError(
        "Upper-body reference dimension mismatch: "
        f"{joint_offset.shape[1]} joints selected, but {amp.shape[1]} "
        "reference terms configured."
      )

    offset_expanded = offset.unsqueeze(0).expand(N, -1)
    ref = amp * sign * torch.sin(phase.unsqueeze(-1) + offset_expanded) + joint_offset

    # Dynamic upright waist reference: learned by tracking, not hard-locked.
    # Keep a forward lean offset on pitch while compensating base roll/pitch.
    base_euler = _euler_from_quat(self.robot.data.root_link_quat_w)
    if "waist_pitch_joint" in self._upper_body_joint_names:
      waist_pitch_idx = self._upper_body_joint_names.index("waist_pitch_joint")
      ref[:, waist_pitch_idx] += (
        waist_pitch0
        - self.cfg.waist_upright_pitch_gain * base_euler[:, 1]
      )
    if "waist_roll_joint" in self._upper_body_joint_names:
      waist_roll_idx = self._upper_body_joint_names.index("waist_roll_joint")
      ref[:, waist_roll_idx] += -self.cfg.waist_upright_roll_gain * base_euler[:, 0]

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

    # Keep raw HLIP target in MPC mode; per-step lateral constraints are enforced
    # by the MPC feasibility checks at stance transitions.
    foot_target_ref = foot_target_adj.clone()
    if self._mpc_enabled:
      mpc_ids = self._mpc_has_plan.nonzero(as_tuple=False).flatten()
      if len(mpc_ids) > 0:
        foot_target_ref[mpc_ids] = self._mpc_active_foot_target[mpc_ids]
    else:
      foot_target_ref[:, 1] = (
        torch.clamp(
          torch.abs(foot_target_adj[:, 1]),
          min=self.cfg.foot_target_range_y[0],
          max=self.cfg.foot_target_range_y[1],
        )
        * torch.sign(Uy_sel)
      )

    foot_target_adj = foot_target_ref

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
      z_sw_neg[mpc_mask] = foot_target_ref[mpc_mask, 2]

    need_height_lookup = (~mpc_mask).nonzero(as_tuple=False).flatten()
    if len(need_height_lookup) > 0:
      try:
        heightmap_data = self._env.scene.sensors["heightmap"].data
        hit_pos_w = heightmap_data.hit_pos_w  # (N, num_rays, 3)

        local_target = torch.cat(
          [
            foot_target_ref[:, :2],
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

    self.foot_target = torch.cat([foot_target_ref[:, :2], z_sw_neg.unsqueeze(1)], dim=1)

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
    self._mpc_replan_pending[env_ids] = False
    self._mpc_last_cost[env_ids] = 0.0
    self._mpc_feasible_steps[env_ids] = 0.0
    self._mpc_fallback_used[env_ids] = 0.0
    self._mpc_contact_recompute_cooldown[env_ids] = 0.0

    self._mpc_active_foot_target[env_ids] = 0.0
    self._mpc_active_foot_target[env_ids, 1] = -self.y_nom
    self._mpc_active_foot_target[env_ids, 2] = self.cfg.z_sw_min

    self._mpc_plan_t[env_ids] = self._mpc_t_nom
    self._mpc_plan_foothold[env_ids] = 0.0
    self._mpc_plan_foothold[env_ids, :, 1] = -self.y_nom
    self._mpc_plan_foothold[env_ids, :, 2] = self.cfg.z_sw_min

  def _finalize_pending_stance_init(self) -> None:
    """Resolve deferred post-reset stance/swing initialization.

    After reset events mutate the root pose, body-link data is only valid after
    ``sim.forward()``. This helper finalizes stance/swing anchors once fresh
    kinematics are available.
    """
    init_ids = self._pending_stance_init.nonzero(as_tuple=False).flatten()
    if len(init_ids) == 0:
      return

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

  def prime_for_observation(self) -> None:
    """Prime command/reference state for reset-time observation computation.

    ``env.reset()`` computes observations before ``command_manager.compute`` is
    called. To avoid returning all-zero HLIP command-derived observations on the
    very first spawn, this method initializes stance anchors, optionally builds
    an MPC plan, and generates reference/actual trajectories without advancing
    gait phase.
    """
    if not bool(torch.any(self._pending_stance_init)):
      return

    self._finalize_pending_stance_init()

    if self._mpc_enabled:
      plan_ready_mask = (
        (~self._pending_stance_init)
        & (~self._mpc_has_plan)
        & (~self._mpc_replan_pending)
      )
      if self._phase_end_stance_flip_only:
        # Reset-time priming should be considered the start of a half-step.
        plan_ready_mask = plan_ready_mask & (self._swing_elapsed <= 1.0e-6)

      plan_ids = plan_ready_mask.nonzero(as_tuple=False).flatten()
      if len(plan_ids) > 0:
        self._plan_mpc_transition(plan_ids)

      if not self._phase_end_stance_flip_only:
        pending_ids = (
          (~self._pending_stance_init)
          & (~self._mpc_has_plan)
          & self._mpc_replan_pending
        ).nonzero(as_tuple=False).flatten()
        if len(pending_ids) > 0:
          stance_contact = self._get_stance_contact_state(pending_ids)
          ready_ids = pending_ids[stance_contact]
          if len(ready_ids) > 0:
            self._plan_mpc_transition(ready_ids)

    self._generate_reference_trajectory()
    self._get_actual_state()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Keep the viewer-selected env under manual control; resample others normally.
    sample_env_ids = env_ids
    if self._manual_control_enabled():
      manual_idx = self._manual_control_target_env_idx()
      sample_env_ids = env_ids[env_ids != manual_idx]

    self._sample_velocity_command(sample_env_ids)

    self._apply_manual_control(env_ids)

    # Reset gait phase and initialise step time.
    if self._mpc_enabled:
      self.T[env_ids] = self._mpc_t_nom
    else:
      self.T[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
        self.T_min,
        self.T_max,
      )
    self.full_gait_period[env_ids] = 2.0 * (self.T[env_ids] - self.T_ds)
    if self.z_sw_max_range is not None:
      self.z_sw_max_envs[env_ids] = torch.empty(
        len(env_ids), device=self.device
      ).uniform_(self.z_sw_max_range[0], self.z_sw_max_range[1])
    self.gait_time[env_ids] = 0.0
    self._swing_elapsed[env_ids] = 0.0
    self._swing_contact_elapsed[env_ids] = 0.0
    self._swing_no_contact_elapsed[env_ids] = 0.0
    self._prev_foot_contact[env_ids] = False
    self._recovery_this_tick[env_ids] = False
    self._recovery_lockout[env_ids] = 0.0

    # Start with left foot as stance (stance_idx=0).
    self.stance_idx[env_ids] = 0
    self.swing_idx[env_ids] = 1
    self._prev_stance_idx[env_ids] = 0

    self._reset_mpc_state(env_ids)

    # NOTE: Body-link positions are stale here (sim.forward() hasn't run
    # after the reset event that randomised the root pose).  We defer the
    # stance foot pose read to the first _update_command call via a flag.
    self._pending_stance_init[env_ids] = True

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

  def _sample_velocity_command(self, env_ids: torch.Tensor) -> None:
    """Sample velocity commands for the selected environments."""
    n = len(env_ids)
    if n == 0:
      return

    r = torch.empty(n, device=self.device)

    if self._fixed_velocity_command_enabled:
      # Keep the translational command deterministic for corridor tasks.
      self.vel_command[env_ids, 0] = self._fixed_lin_vel_x
      self.vel_command[env_ids, 1] = self._fixed_lin_vel_y
      self.vel_command[env_ids, 2] = 0.0
    else:
      self.vel_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
      self.vel_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
      self.vel_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

      cmd_mag = (
        torch.norm(self.vel_command[env_ids, :2], dim=1)
        + torch.abs(self.vel_command[env_ids, 2])
      )
      self.vel_command[env_ids] *= (cmd_mag > 0.1).unsqueeze(1)

    self.is_standing_env[env_ids] = (
      r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    )
    standing = env_ids[self.is_standing_env[env_ids]]
    if len(standing) > 0:
      self.vel_command[standing] = 0.0

  def _apply_fixed_linear_command(
    self,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Apply deterministic linear command for selected environments."""
    if not self._fixed_velocity_command_enabled:
      return

    if env_ids is None:
      env_tensor = torch.arange(self.num_envs, device=self.device)
    else:
      env_tensor = env_ids

    if self._manual_control_enabled():
      manual_idx = self._manual_control_target_env_idx()
      env_tensor = env_tensor[env_tensor != manual_idx]

    if len(env_tensor) == 0:
      return

    self.vel_command[env_tensor, 0] = self._fixed_lin_vel_x
    self.vel_command[env_tensor, 1] = self._fixed_lin_vel_y

  def _apply_yaw_hold_control(
    self,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Set yaw-rate command using proportional control to hold yaw."""
    if not self._yaw_hold_enabled:
      return

    if env_ids is None:
      env_tensor = torch.arange(self.num_envs, device=self.device)
    else:
      env_tensor = env_ids

    if self._manual_control_enabled():
      manual_idx = self._manual_control_target_env_idx()
      env_tensor = env_tensor[env_tensor != manual_idx]

    if len(env_tensor) == 0:
      return

    root_quat = self.robot.data.root_link_quat_w[env_tensor]
    yaw = _euler_from_quat(root_quat)[:, 2]
    yaw_err = wrap_to_pi(self._yaw_hold_target - yaw)
    yaw_rate_cmd = torch.clamp(
      self._yaw_hold_kp * yaw_err,
      min=-self._yaw_hold_max_ang_vel,
      max=self._yaw_hold_max_ang_vel,
    )
    self.vel_command[env_tensor, 2] = yaw_rate_cmd

  def _update_command(self) -> None:
    dt = self._env.step_dt

    self._mpc_contact_recompute_cooldown = torch.clamp(
      self._mpc_contact_recompute_cooldown - dt,
      min=0.0,
    )
    self._recovery_lockout = torch.clamp(
      self._recovery_lockout - dt,
      min=0.0,
    )

    # Clear per-tick recovery guard from previous step.
    self._recovery_this_tick.fill_(False)

    # Deferred stance foot initialisation: body positions are now fresh
    # (sim.forward() has run since the last reset event).
    self._finalize_pending_stance_init()

    # Read contact state once per tick and share across all subsystems.
    foot_contact = self._get_foot_contact_state()
    swing_contact = torch.gather(
      foot_contact,
      1,
      self.swing_idx.unsqueeze(1),
    ).squeeze(1)

    # Debounced swing-contact elapsed: tolerate brief contact dropouts
    # (e.g. physics bounces) without resetting the accumulated timer.
    # Only reset after the foot has been off the ground for longer than
    # the debounce window.
    self._swing_no_contact_elapsed = torch.where(
      swing_contact,
      torch.zeros_like(self._swing_no_contact_elapsed),
      self._swing_no_contact_elapsed + dt,
    )
    debounce_expired = self._swing_no_contact_elapsed > self._touchdown_debounce_s
    self._swing_contact_elapsed = torch.where(
      swing_contact,
      self._swing_contact_elapsed + dt,
      torch.where(
        debounce_expired,
        torch.zeros_like(self._swing_contact_elapsed),
        self._swing_contact_elapsed,  # Keep accumulated time during debounce window.
      ),
    )

    # --- Phase 1: Contact-based stance recovery (MPC only) ---
    # Must run BEFORE _update_stance_swing_idx so recovered envs are
    # excluded from a second transition this tick.
    if self._mpc_enabled and not self._phase_end_stance_flip_only:
      self._recover_stance_from_contact(foot_contact)

    # --- Phase 2: Apply command overrides ---
    self._apply_manual_control()

    # Standing envs zero velocity.
    standing = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command[standing] = 0.0

    # Keep manual control dominant over standing/random logic.
    self._apply_manual_control()

    # Optional deterministic command overrides for corridor-style tasks.
    self._apply_fixed_linear_command()
    self._apply_yaw_hold_control()

    # --- Phase 3: Stance transitions (touchdown / timeout) ---
    # Envs already recovered this tick are excluded via _recovery_this_tick.
    self._update_stance_swing_idx(dt, foot_contact)

    # --- Phase 4: Resolve pending MPC plans ---
    # This runs AFTER transitions so that newly-transitioned envs that
    # deferred planning (waiting for stance contact) get a chance to plan.
    if self._mpc_enabled:
      plan_ready_mask = (
        (~self._pending_stance_init)
        & (~self._mpc_has_plan)
        & (~self._mpc_replan_pending)
      )
      if self._phase_end_stance_flip_only:
        # Teacher mode: never replan in mid-swing. Only allow plan creation
        # right after a stance transition/reset when swing time is near zero.
        start_of_half_step = self._swing_elapsed <= max(float(dt), 1.0e-6)
        plan_ready_mask = plan_ready_mask & start_of_half_step

      plan_ids = plan_ready_mask.nonzero(as_tuple=False).flatten()
      if len(plan_ids) > 0:
        self._plan_mpc_transition(plan_ids)

      if not self._phase_end_stance_flip_only:
        pending_ids = (
          (~self._pending_stance_init)
          & (~self._mpc_has_plan)
          & self._mpc_replan_pending
        ).nonzero(as_tuple=False).flatten()
        if len(pending_ids) > 0:
          stance_contact = self._get_stance_contact_state(pending_ids)
          cooldown_ready = self._mpc_contact_recompute_cooldown[pending_ids] <= 0.0
          ready_ids = pending_ids[stance_contact & cooldown_ready]
          if len(ready_ids) > 0:
            self._plan_mpc_transition(ready_ids)
            self._start_contact_recompute_cooldown(ready_ids)

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

    # Camera rays coloured by MPC steppability classification.
    if "head_camera_rays" in self._env.scene.sensors:
      ray_sensor = self._env.scene.sensors["head_camera_rays"]
      ray_data = ray_sensor.data
      if (
        ray_data.hit_pos_w is not None
        and ray_data.distances is not None
        and ray_data.pos_w is not None
        and ray_data.quat_w is not None
      ):
        steppable_flat, near_edge_flat, height_blocked_flat = self._classify_camera_rays_by_heightmap(batch)

        frame_pos = ray_data.pos_w[batch]
        frame_quat = ray_data.quat_w[batch]
        cam_offset = torch.zeros(3, device=self.device, dtype=frame_pos.dtype)
        pattern = getattr(ray_sensor.cfg, "pattern", None)
        if pattern is not None and hasattr(pattern, "pos"):
          cam_offset = torch.tensor(pattern.pos, device=self.device, dtype=frame_pos.dtype)

        ray_origin = quat_apply(
          frame_quat.unsqueeze(0), cam_offset.unsqueeze(0)
        ).squeeze(0) + frame_pos
        ray_origin_np = ray_origin.cpu().numpy()

        ray_hits = ray_data.hit_pos_w[batch]
        ray_dist = ray_data.distances[batch]
        num_rays = int(ray_dist.shape[0])
        for i in range(num_rays):
          if float(ray_dist[i].item()) < 0.0:
            continue

          if height_blocked_flat is not None and bool(height_blocked_flat[i].item()):
            color = (0.2, 0.45, 1.0, 0.85)
          elif near_edge_flat is not None and bool(near_edge_flat[i].item()):
            color = (1.0, 0.55, 0.0, 0.8)
          elif steppable_flat is not None and bool(steppable_flat[i].item()):
            color = (0.2, 1.0, 0.2, 0.65)
          elif steppable_flat is not None:
            color = (1.0, 0.2, 0.2, 0.65)
          else:
            color = (0.8, 0.8, 0.8, 0.45)

          visualizer.add_arrow(
            ray_origin_np,
            ray_hits[i].cpu().numpy(),
            color=color,
            width=0.004,
            label=f"camera_ray_{i}",
          )

    # Heightmap rendering with edge detection
    if "heightmap" in self._env.scene.sensors:
      hm_data = self._env.scene.sensors["heightmap"].data
      if hm_data.hit_pos_w is not None and hm_data.distances is not None:
        hit_pos_b = hm_data.hit_pos_w[batch]
        dists_b = hm_data.distances[batch]
        steppable_b, near_edge_b, height_blocked_b, _ = self._compute_steppability_masks(
          hit_pos_b.unsqueeze(0),
          dists_b.unsqueeze(0),
          self.stance_foot_pos_0[batch, 2].unsqueeze(0),
          sensor_pos_w=hm_data.pos_w[batch].unsqueeze(0),
          sensor_quat_w=hm_data.quat_w[batch].unsqueeze(0),
        )
        steppable_b = steppable_b.squeeze(0)
        near_edge_b = near_edge_b.squeeze(0)
        height_blocked_b = height_blocked_b.squeeze(0)

        if self._hm_num_x > 0 and self._hm_num_y > 0:
          hit_grid = hit_pos_b.cpu().numpy().reshape(self._hm_num_y, self._hm_num_x, 3)
          dist_grid = dists_b.cpu().numpy().reshape(self._hm_num_y, self._hm_num_x)
          steppable_grid = steppable_b.cpu().numpy().reshape(self._hm_num_y, self._hm_num_x)
          near_edge_grid = near_edge_b.cpu().numpy().reshape(self._hm_num_y, self._hm_num_x)
          height_blocked_grid = height_blocked_b.cpu().numpy().reshape(self._hm_num_y, self._hm_num_x)

          for i in range(self._hm_num_y):
            for j in range(self._hm_num_x):
              if dist_grid[i, j] >= 0:
                h_pos = hit_grid[i, j]
                if height_blocked_grid[i, j]:
                  col = (0.2, 0.45, 1.0, 0.9)
                elif near_edge_grid[i, j]:
                  col = (1.0, 0.55, 0.0, 0.85)
                elif steppable_grid[i, j]:
                  col = (0.0, 1.0, 0.0, 0.8)
                else:
                  col = (1.0, 0.0, 0.0, 0.8)
                visualizer.add_sphere(h_pos, radius=0.02, color=col)

# =====================================================================
# Configuration
# =====================================================================


# Default Q weights: 58 entries (29 output pairs × pos/vel).
# Joint order matches planc regex expansion:
#   waist_yaw, waist_pitch, waist_roll,
#   L_shoulder_pitch, R_shoulder_pitch, L_shoulder_roll,
#   R_shoulder_roll, L_shoulder_yaw, R_shoulder_yaw, L_elbow, R_elbow,
#   L_wrist_roll, R_wrist_roll, L_wrist_pitch, R_wrist_pitch,
#   L_wrist_yaw, R_wrist_yaw
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
  500.0, 10.0,      # waist_pitch
  500.0, 10.0,      # waist_roll
  40.0, 1.0,        # left shoulder pitch
  40.0, 1.0,        # right shoulder pitch
  100.0, 1.0,       # left shoulder roll
  100.0, 1.0,       # right shoulder roll
  50.0, 1.0,        # left shoulder yaw
  50.0, 1.0,        # right shoulder yaw
  30.0, 1.0,        # left elbow
  30.0, 1.0,        # right elbow
  20.0, 1.0,        # left wrist roll
  20.0, 1.0,        # right wrist roll
  20.0, 1.0,        # left wrist pitch
  20.0, 1.0,        # right wrist pitch
  20.0, 1.0,        # left wrist yaw
  20.0, 1.0,        # right wrist yaw
]

# Default R weights: 29 entries (one per output).
# Upper body order matches planc regex expansion.
HLIP_DEFAULT_R_WEIGHTS = [
  0.1, 0.1, 0.1,         # CoM
  0.05, 0.05, 0.05,      # Pelvis
  0.05, 0.05, 0.05,      # Swing foot linear
  0.02, 0.02, 0.02,      # Swing foot orientation
  0.1,                    # Waist yaw
  0.1,                    # Waist pitch
  0.1,                    # Waist roll
  0.01, 0.01,             # L/R shoulder pitch
  0.01, 0.01,             # L/R shoulder roll
  0.01, 0.01,             # L/R shoulder yaw
  0.01, 0.01,             # L/R elbow
  0.01, 0.01,             # L/R wrist roll
  0.01, 0.01,             # L/R wrist pitch
  0.01, 0.01,             # L/R wrist yaw
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
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
  ])
  """Joint names for upper body reference tracking.

  Order matches planc regex expansion:
    waist_yaw, waist_pitch, waist_roll,
    L/R shoulder_pitch, L/R shoulder_roll,
    L/R shoulder_yaw, L/R elbow,
    L/R wrist_roll, L/R wrist_pitch, L/R wrist_yaw.
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
  """Enable edge-aware MPC foothold and step-time planning."""

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

  mpc_edge_clearance_distance: float = 0.0
  """Edge-cost clearance radius [m]; footholds farther than this have zero edge cost."""

  mpc_edge_distance_cost: float = 10.0
  """Quadratic edge-proximity cost scale for MPC foothold selection."""

  mpc_max_stance_height_delta: float = -1.0
  """Max |z_hit - z_stance| [m] as a hard unsteppable filter; <=0 disables this."""

  mpc_non_steppable_cost: float = 1.0e6
  """Deprecated: retained for backwards compatibility with older configs."""

  mpc_fallback_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
  """Velocity scaling factors used when searching fallback feasible plans."""

  touchdown_switch_enabled: bool = True
  """Switch stance when swing foot touchdown is detected."""

  touchdown_sensor_name: str = "feet_ground_contact"
  """Contact sensor used to detect swing-foot touchdown events."""

  touchdown_min_phase: float = 0.60
  """Earliest normalized swing phase at which touchdown can trigger switch."""

  touchdown_contact_grace_period_s: float = 0.10
  """Required continuous swing-foot contact duration [s] before switching stance."""

  touchdown_min_force_n: float = 75.0
  """Minimum net contact force magnitude [N] for a foot to count as in contact.

  Toe-grazing contacts below this threshold will not accumulate toward the
  grace period and will not trigger stance transitions.  Set to 0 to disable."""

  touchdown_debounce_s: float = 0.04
  """Contact dropout tolerance [s]. Brief no-contact gaps shorter than this
  will not reset the accumulated contact timer. Prevents physics-bounce
  jitter from restarting the grace period."""

  touchdown_timeout_phase: float = 1.20
  """Fallback forced switch threshold when touchdown is not detected."""

  phase_end_stance_flip_only: bool = False
  """When true, switch stance only at phase end and skip contact-based recovery/gating."""

  mpc_stance_recovery_min_elapsed_s: float = 0.25
  """Minimum elapsed half-step time [s] before stance-contact recovery can trigger."""

  mpc_contact_recovery_enabled: bool = True
  """Allow immediate stance recovery + replan when stance contact is lost."""

  mpc_contact_recompute_grace_period_s: float = 0.3
  """Cooldown [s] after contact-triggered MPC replan before another can trigger."""

  mpc_replan_wait_for_stance_contact: bool = True
  """Defer stance-switch replanning until the new stance foot is in contact."""

  # Pelvis orientation.
  pelv_pitch_ref: float = 0.0
  """Pelvis pitch reference offset [rad]."""

  # Upper body.
  shoulder_ref: tuple[float, float, float] = (0.16, 0.0, 0.0)
  """Shoulder (pitch, roll, yaw) amplitude scalars."""

  elbow_ref: float = 0.1
  """Elbow swing amplitude scalar."""

  wrist_ref: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Wrist (roll, pitch, yaw) amplitude scalars."""

  waist_yaw_ref: float = 0.0
  """Waist yaw oscillation amplitude."""

  waist_pitch_ref: float = 0.0
  """Waist pitch reference offset [rad] applied as a constant lean."""

  waist_roll_ref: float = 0.0
  """Waist roll oscillation amplitude."""

  waist_upright_pitch_gain: float = 1.0
  """Gain for pitch compensation to keep torso upright while leaning forward."""

  waist_upright_roll_gain: float = 1.0
  """Gain for roll compensation to keep torso upright."""

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

  manual_control: bool = False
  """Enable viewer sliders for manual velocity command control."""

  fixed_velocity_command_enabled: bool = False
  """Use deterministic linear velocity command instead of resampling ranges."""

  fixed_lin_vel_x: float = 0.0
  """Fixed commanded forward velocity [m/s] when deterministic mode is enabled."""

  fixed_lin_vel_y: float = 0.0
  """Fixed commanded lateral velocity [m/s] when deterministic mode is enabled."""

  resample_velocity_on_step: bool = False
  """Resample random velocity commands at every non-MPC stance transition."""

  resample_velocity_on_step_probability: float = 1.0
  """Probability of resampling velocity at each non-MPC stance transition."""

  yaw_hold_enabled: bool = False
  """Enable yaw-hold proportional control for commanded yaw rate."""

  yaw_hold_target: float = 0.0
  """Target world yaw angle [rad] for yaw-hold control."""

  yaw_hold_kp: float = 1.5
  """Proportional gain mapping yaw error to commanded yaw rate."""

  yaw_hold_max_ang_vel: float = 0.8
  """Absolute limit [rad/s] for yaw-hold commanded yaw rate."""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: "ManagerBasedRlEnv") -> HLIPCommandTerm:
    return HLIPCommandTerm(self, env)
