"""LIP walking command generator with swing foot trajectory.

Generates velocity commands and plans footsteps using the Linear Inverted
Pendulum (LIP) capture point heuristic. Instead of tracking a CoM trajectory,
the agent tracks a velocity command and follows a smooth swing foot trajectory
defined by sin/cos parametric curves.

The swing trajectory interpolates from the swing foot start position to the
planned footstep target:
  xy(phase) = start + (target - start) * (1 - cos(pi * phase)) / 2
  z(phase)  = swing_height * sin(pi * phase)
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
  matrix_from_quat,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Gravity constant.
GRAVITY = 9.81


class LIPCommand(CommandTerm):
  """Command term: velocity commands + LIP-based swing foot trajectories.

  At each reset a desired walking velocity is sampled. The LIP capture-point
  heuristic plans footstep locations. A smooth swing foot trajectory is
  generated using sin/cos parametric curves from the swing-foot start
  position to the target footstep.

  Command tensor (10 dims):
    [0:3]  velocity command (vx, vy, yaw_rate) - body frame
    [3:6]  swing foot trajectory reference (x, y, z) - body frame
    [6:8]  target footstep relative to CoM (x, y) - body frame
    [8]    step phase  (0 -> 1)
    [9]    support foot indicator  (-1 = left, +1 = right)
  """

  cfg: LIPCommandCfg

  def __init__(self, cfg: LIPCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # LIP capture-point parameter (omega = sqrt(g / z_c)).
    self.omega = math.sqrt(GRAVITY / cfg.com_height)

    # Resolve foot site indices for reading actual foot positions.
    site_ids, site_names = self.robot.find_sites(cfg.feet_site_names)
    # Indices into robot.data.site_pos_w for left and right foot.
    self.left_foot_site_id = site_ids[0]
    self.right_foot_site_id = site_ids[1]

    # Command buffer: 10-dim.
    self._command = torch.zeros(self.num_envs, 10, device=self.device)

    # Desired walking velocity (body frame): [vx, vy, yaw_rate].
    self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)

    # Heading control.
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    # Step tracking.
    self.support_foot_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.support_foot_side = torch.ones(self.num_envs, device=self.device)
    self.step_time = torch.zeros(self.num_envs, device=self.device)

    # Footstep planning.
    self.next_foot_pos_w = torch.zeros(self.num_envs, 2, device=self.device)

    # Swing trajectory start position (world frame, xyz).
    self.swing_start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    # Ground height estimated from the support foot z.
    self.ground_height_w = torch.zeros(self.num_envs, device=self.device)

    # Metrics.
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  # ------------------------------------------------------------------
  # Swing trajectory
  # ------------------------------------------------------------------

  def _compute_swing_trajectory_w(self, phase: torch.Tensor) -> torch.Tensor:
    """Reference swing foot position in **world** frame via sin/cos curves.

    xy(phase) = start + (target - start) * (1 - cos(pi * phase)) / 2
    z(phase)  = swing_height * sin(pi * phase)

    Args:
      phase: Step phase in [0, 1]. Shape ``(num_envs,)``.

    Returns:
      Reference position in world frame. Shape ``(num_envs, 3)``.
    """
    alpha = (1.0 - torch.cos(math.pi * phase)) / 2.0  # (N,)

    ref_xy = self.swing_start_pos_w[:, :2] + alpha.unsqueeze(-1) * (
      self.next_foot_pos_w - self.swing_start_pos_w[:, :2]
    )

    ref_z = self.ground_height_w + self.cfg.swing_height * torch.sin(math.pi * phase)

    return torch.cat([ref_xy, ref_z.unsqueeze(-1)], dim=-1)

  # ------------------------------------------------------------------
  # Footstep planning (capture-point heuristic)
  # ------------------------------------------------------------------

  def _plan_next_footstep(self, env_ids: torch.Tensor) -> None:
    """Plan next footstep using the instantaneous capture point.

    footstep = support + capture_point + vel_offset + lateral_offset

    The capture point uses the **actual** robot velocity (not LIP
    predictions), making the planner reactive to disturbances.
    """
    n = len(env_ids)
    if n == 0:
      return

    root_pos_xy = self.robot.data.root_link_pos_w[env_ids, :2]

    heading = self.robot.data.heading_w[env_ids]
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    # World-frame horizontal velocity (unaffected by pitch/roll).
    vel_w = self.robot.data.root_link_lin_vel_w[env_ids, :2]

    # Capture point relative to support foot (world).
    com_rel = root_pos_xy - self.support_foot_pos_w[env_ids]
    capture_pt = com_rel + vel_w / self.omega

    # Desired-velocity offset (body -> world, then * T/2).
    vel_cmd_xy = self.vel_command[env_ids, :2]
    vel_offset = (
      torch.stack(
        [
          vel_cmd_xy[:, 0] * cos_h - vel_cmd_xy[:, 1] * sin_h,
          vel_cmd_xy[:, 0] * sin_h + vel_cmd_xy[:, 1] * cos_h,
        ],
        dim=-1,
      )
      * self.cfg.step_duration
      * 0.5
    )

    # Lateral offset for the *next* foot.
    # next_side: -1 = left foot, +1 = right foot.
    # In body frame +Y is left, so left foot needs +Y and right foot needs -Y.
    next_side = -self.support_foot_side[env_ids]
    lateral_b = torch.zeros(n, 2, device=self.device)
    lateral_b[:, 1] = -next_side * self.cfg.foot_separation / 2.0
    lateral_w = torch.stack(
      [
        lateral_b[:, 0] * cos_h - lateral_b[:, 1] * sin_h,
        lateral_b[:, 0] * sin_h + lateral_b[:, 1] * cos_h,
      ],
      dim=-1,
    )

    self.next_foot_pos_w[env_ids] = (
      self.support_foot_pos_w[env_ids] + capture_pt + vel_offset + lateral_w
    )

  # ------------------------------------------------------------------
  # Step transitions
  # ------------------------------------------------------------------

  def _get_actual_foot_pos(self, env_ids: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    """Read actual foot positions from robot site data.

    Returns:
      left_xy, right_xy, left_z, right_z  all indexed by env_ids.
    """
    left_xy = self.robot.data.site_pos_w[env_ids, self.left_foot_site_id, :2]
    right_xy = self.robot.data.site_pos_w[env_ids, self.right_foot_site_id, :2]
    left_z = self.robot.data.site_pos_w[env_ids, self.left_foot_site_id, 2]
    right_z = self.robot.data.site_pos_w[env_ids, self.right_foot_site_id, 2]
    return left_xy, right_xy, left_z, right_z

  def _transition_step(self, env_ids: torch.Tensor) -> None:
    """Swap support/swing foot and start a new step.

    Uses the actual robot foot site positions instead of planned positions.
    """
    if len(env_ids) == 0:
      return

    left_xy, right_xy, left_z, right_z = self._get_actual_foot_pos(env_ids)

    # Before flipping: old support foot becomes new swing-foot start.
    old_is_right = (self.support_foot_side[env_ids] > 0).unsqueeze(-1)
    self.swing_start_pos_w[env_ids, :2] = torch.where(
      old_is_right, right_xy, left_xy
    )
    old_is_right_1d = old_is_right.squeeze(-1)
    self.swing_start_pos_w[env_ids, 2] = torch.where(
      old_is_right_1d, right_z, left_z
    )

    # Flip support side.
    self.support_foot_side[env_ids] = -self.support_foot_side[env_ids]

    # New support foot = actual position of the foot that was swinging.
    new_is_right = (self.support_foot_side[env_ids] > 0).unsqueeze(-1)
    self.support_foot_pos_w[env_ids] = torch.where(
      new_is_right, right_xy, left_xy
    )
    new_is_right_1d = new_is_right.squeeze(-1)
    self.ground_height_w[env_ids] = torch.where(
      new_is_right_1d, right_z, left_z
    )

    # Reset step timer.
    self.step_time[env_ids] = 0.0

    # Plan next footstep.
    self._plan_next_footstep(env_ids)

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt

    # World-frame horizontal velocity -> yaw-only body frame.
    vel_w = self.robot.data.root_link_lin_vel_w[:, :2]
    heading = self.robot.data.heading_w
    cos_h = torch.cos(-heading)
    sin_h = torch.sin(-heading)
    actual_vel_yaw = torch.stack(
      [
        vel_w[:, 0] * cos_h - vel_w[:, 1] * sin_h,
        vel_w[:, 0] * sin_h + vel_w[:, 1] * cos_h,
      ],
      dim=-1,
    )
    self.metrics["error_vel_xy"] += (
      torch.norm(self.vel_command[:, :2] - actual_vel_yaw, dim=-1)
      / max_command_step
    )
    actual_ang_vel_b = self.robot.data.root_link_ang_vel_b
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command[:, 2] - actual_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = torch.empty(n, device=self.device)

    # Sample desired walking velocity.
    self.vel_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

    # Zero out very small commands.
    cmd_magnitude = torch.norm(self.vel_command[env_ids, :2], dim=1) + torch.abs(
      self.vel_command[env_ids, 2]
    )
    self.vel_command[env_ids] *= (cmd_magnitude > 0.1).unsqueeze(1)

    # Standing envs.
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    standing_ids = env_ids[self.is_standing_env[env_ids]]
    if len(standing_ids) > 0:
      self.vel_command[standing_ids] = 0.0

    # Heading control.
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

    # ---- initialise support foot from root position + foot separation ----
    # After a reset, site positions may be stale (no FK yet), so we compute
    # foot positions from the reset root state and the known foot separation.
    self.support_foot_side[env_ids] = torch.where(
      r.uniform_(0.0, 1.0) > 0.5,
      torch.ones(n, device=self.device),
      -torch.ones(n, device=self.device),
    )

    root_pos_xy = self.robot.data.root_link_pos_w[env_ids, :2]
    root_z = self.robot.data.root_link_pos_w[env_ids, 2]
    heading = self.robot.data.heading_w[env_ids]
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    # Left foot is at +Y, right foot at -Y in body frame.
    half_sep = self.cfg.foot_separation / 2.0
    # Body-frame offsets: left = (0, +half_sep), right = (0, -half_sep).
    left_xy = root_pos_xy + torch.stack(
      [-sin_h * half_sep, cos_h * half_sep], dim=-1
    )
    right_xy = root_pos_xy + torch.stack(
      [sin_h * half_sep, -cos_h * half_sep], dim=-1
    )
    # Assume feet are on the ground at root_z - com_height.
    foot_z = root_z - self.cfg.com_height

    # Support foot = position of the chosen support foot.
    is_right = (self.support_foot_side[env_ids] > 0).unsqueeze(-1)
    self.support_foot_pos_w[env_ids] = torch.where(is_right, right_xy, left_xy)
    self.ground_height_w[env_ids] = foot_z

    # Swing foot start = position of the opposite foot.
    self.swing_start_pos_w[env_ids, :2] = torch.where(is_right, left_xy, right_xy)
    self.swing_start_pos_w[env_ids, 2] = foot_z

    self.step_time[env_ids] = 0.0

    # Plan first footstep.
    self._plan_next_footstep(env_ids)

    # Reset the command tensor so observations see clean values immediately.
    self._command[env_ids, 0:3] = self.vel_command[env_ids]
    # Swing ref at start position (phase=0 → at swing start, z=ground).
    # Convert swing start to body frame for the command tensor.
    swing_start_rel_w = self.swing_start_pos_w[env_ids] - self.robot.data.root_link_pos_w[env_ids, :3]
    neg_heading = -self.robot.data.heading_w[env_ids]
    cos_nh = torch.cos(neg_heading)
    sin_nh = torch.sin(neg_heading)
    self._command[env_ids, 3] = swing_start_rel_w[:, 0] * cos_nh - swing_start_rel_w[:, 1] * sin_nh
    self._command[env_ids, 4] = swing_start_rel_w[:, 0] * sin_nh + swing_start_rel_w[:, 1] * cos_nh
    self._command[env_ids, 5] = swing_start_rel_w[:, 2]
    # Target footstep in body frame.
    next_rel_w = self.next_foot_pos_w[env_ids] - root_pos_xy
    self._command[env_ids, 6] = next_rel_w[:, 0] * cos_nh - next_rel_w[:, 1] * sin_nh
    self._command[env_ids, 7] = next_rel_w[:, 0] * sin_nh + next_rel_w[:, 1] * cos_nh
    self._command[env_ids, 8] = 0.0  # phase = 0
    self._command[env_ids, 9] = self.support_foot_side[env_ids]

  def _update_command(self) -> None:
    dt = self._env.step_dt

    # Heading control.
    if self.cfg.heading_command:
      heading_error = wrap_to_pi(
        self.heading_target - self.robot.data.heading_w
      )
      heading_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command[heading_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * heading_error[heading_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )

    # Standing envs -> zero velocity.
    standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command[standing_ids] = 0.0

    # Advance step timer.
    self.step_time += dt

    # Step transitions.
    step_done = self.step_time >= self.cfg.step_duration
    step_done_ids = step_done.nonzero(as_tuple=False).flatten()
    if len(step_done_ids) > 0:
      self._transition_step(step_done_ids)

    # Phase in [0, 1].
    phase = self.step_time / self.cfg.step_duration

    # --- Continuously update support foot and re-plan footstep ---
    # Read actual stance foot position so the planner reacts to disturbances.
    all_ids = torch.arange(self.num_envs, device=self.device)
    left_xy, right_xy, left_z, right_z = self._get_actual_foot_pos(all_ids)
    is_right_sup = (self.support_foot_side > 0).unsqueeze(-1)
    self.support_foot_pos_w[:] = torch.where(is_right_sup, right_xy, left_xy)
    is_right_1d = is_right_sup.squeeze(-1)
    self.ground_height_w[:] = torch.where(is_right_1d, right_z, left_z)

    # Re-plan the touchdown target every tick using current CoM & velocity.
    self._plan_next_footstep(all_ids)

    # Swing trajectory reference (world -> body frame).
    swing_ref_w = self._compute_swing_trajectory_w(phase)

    root_pos = self.robot.data.root_link_pos_w[:, :3]
    swing_ref_rel_w = swing_ref_w - root_pos  # relative to root, world frame

    heading = self.robot.data.heading_w
    cos_h = torch.cos(-heading)
    sin_h = torch.sin(-heading)
    swing_ref_b = torch.stack(
      [
        swing_ref_rel_w[:, 0] * cos_h - swing_ref_rel_w[:, 1] * sin_h,
        swing_ref_rel_w[:, 0] * sin_h + swing_ref_rel_w[:, 1] * cos_h,
        swing_ref_rel_w[:, 2],
      ],
      dim=-1,
    )

    # Target footstep relative to CoM, body frame.
    root_pos_xy = self.robot.data.root_link_pos_w[:, :2]
    next_foot_rel_w = self.next_foot_pos_w - root_pos_xy
    next_foot_b = torch.stack(
      [
        next_foot_rel_w[:, 0] * cos_h - next_foot_rel_w[:, 1] * sin_h,
        next_foot_rel_w[:, 0] * sin_h + next_foot_rel_w[:, 1] * cos_h,
      ],
      dim=-1,
    )

    # Assemble command tensor (10 dims).
    self._command[:, 0:3] = self.vel_command       # desired velocity (body)
    self._command[:, 3:6] = swing_ref_b            # swing trajectory ref (body)
    self._command[:, 6:8] = next_foot_b            # target footstep (body)
    self._command[:, 8] = phase                    # step phase
    self._command[:, 9] = self.support_foot_side   # support foot indicator

  # ------------------------------------------------------------------
  # Visualisation
  # ------------------------------------------------------------------

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw swing trajectory curve, footstep targets, velocity arrows."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    base_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    if np.linalg.norm(base_pos) < 1e-6:
      return

    ground_z = self.ground_height_w[batch].item()

    # Support foot (green).
    sp = self.support_foot_pos_w[batch].cpu().numpy()
    visualizer.add_sphere(
      np.array([sp[0], sp[1], ground_z]), radius=0.03, color=(0.2, 0.8, 0.2, 0.8)
    )

    # Target footstep (red).
    nf = self.next_foot_pos_w[batch].cpu().numpy()
    visualizer.add_sphere(
      np.array([nf[0], nf[1], ground_z]), radius=0.03, color=(0.8, 0.2, 0.2, 0.8)
    )

    # Swing trajectory curve (yellow dots).
    start_xy = self.swing_start_pos_w[batch, :2].cpu().numpy()
    target_xy = nf
    num_pts = 12
    for i in range(num_pts):
      p = i / (num_pts - 1)
      alpha = (1.0 - math.cos(math.pi * p)) / 2.0
      rx = start_xy[0] + alpha * (target_xy[0] - start_xy[0])
      ry = start_xy[1] + alpha * (target_xy[1] - start_xy[1])
      rz = ground_z + self.cfg.swing_height * math.sin(math.pi * p)
      visualizer.add_sphere(
        np.array([rx, ry, rz]), radius=0.01, color=(0.8, 0.8, 0.2, 0.5)
      )

    # Current swing reference (blue).
    phase_now = self.step_time[batch].item() / self.cfg.step_duration
    alpha_now = (1.0 - math.cos(math.pi * phase_now)) / 2.0
    crx = start_xy[0] + alpha_now * (target_xy[0] - start_xy[0])
    cry = start_xy[1] + alpha_now * (target_xy[1] - start_xy[1])
    crz = ground_z + self.cfg.swing_height * math.sin(math.pi * phase_now)
    visualizer.add_sphere(
      np.array([crx, cry, crz]), radius=0.025, color=(0.2, 0.2, 0.8, 0.8)
    )

    # Velocity command arrow (yaw-only rotation, unaffected by pitch/roll).
    cmd = self.vel_command[batch].cpu().numpy()
    h = self.robot.data.heading_w[batch].item()
    cos_h_v = math.cos(h)
    sin_h_v = math.sin(h)

    def yaw_to_world(vec):
      rx = vec[0] * cos_h_v - vec[1] * sin_h_v
      ry = vec[0] * sin_h_v + vec[1] * cos_h_v
      return base_pos + np.array([rx, ry, vec[2]])

    arrow_from = yaw_to_world(np.array([0, 0, 0.3]))
    arrow_to = yaw_to_world(np.array([cmd[0], cmd[1], 0.3]))
    visualizer.add_arrow(
      arrow_from, arrow_to, color=(0.6, 0.6, 0.2, 0.7), width=0.012
    )


# ====================================================================
# Configuration dataclass
# ====================================================================


@dataclass(kw_only=True)
class LIPCommandCfg(CommandTermCfg):
  """Configuration for the LIP walking command generator."""

  entity_name: str
  """Name of the robot entity in the scene."""

  feet_site_names: tuple[str, str] = ("left_foot", "right_foot")
  """Site names for the left and right foot (order matters: left first)."""

  com_height: float = 0.75
  """Nominal CoM height used for the capture-point calculation [m]."""

  step_duration: float = 0.4
  """Duration of each walking step [s]."""

  foot_separation: float = 0.2
  """Lateral distance between feet in default stance [m]."""

  swing_height: float = 0.08
  """Peak height of the swing foot arc [m]."""

  heading_command: bool = False
  """Derive yaw-rate from heading error feedback."""

  heading_control_stiffness: float = 0.5
  """P-gain for heading -> yaw-rate conversion."""

  rel_standing_envs: float = 0.05
  """Fraction of environments with zero-velocity command."""

  rel_heading_envs: float = 0.25
  """Fraction of environments using heading-based yaw commands."""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    heading: tuple[float, float] | None = None

  ranges: Ranges = field(default_factory=Ranges)

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> LIPCommand:
    return LIPCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "heading_command=True but ranges.heading is set to None."
      )
