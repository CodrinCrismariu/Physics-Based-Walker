"""Reward functions for the LIP walking task.

Rewards that encourage the agent to:
  - Track a velocity command (like the velocity task).
  - Follow the planned swing foot trajectory (sin/cos arc) closely.
  - Maintain proper gait timing, foot contacts, and regularisation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.tasks.lip.mdp.lip_command import LIPCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# =====================================================================
# Velocity tracking rewards
# =====================================================================


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking the commanded linear velocity (xy).

  Uses world-frame horizontal velocity rotated into a yaw-only body frame
  so that pitch/roll of the torso do not affect the velocity comparison.

  exp(-||v_cmd_xy - v_actual_xy||^2 / std^2)
  """
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  vel_cmd = command_term.vel_command

  # World-frame horizontal velocity -> yaw-only body frame.
  vel_w = asset.data.root_link_lin_vel_w[:, :2]
  heading = asset.data.heading_w
  cos_h = torch.cos(-heading)
  sin_h = torch.sin(-heading)
  actual_vel_yaw = torch.stack(
    [
      vel_w[:, 0] * cos_h - vel_w[:, 1] * sin_h,
      vel_w[:, 0] * sin_h + vel_w[:, 1] * cos_h,
    ],
    dim=-1,
  )

  xy_error = torch.sum(torch.square(vel_cmd[:, :2] - actual_vel_yaw), dim=1)
  z_error = torch.square(asset.data.root_link_lin_vel_b[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking the commanded yaw rate."""
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  vel_cmd = command_term.vel_command
  actual_ang_vel = asset.data.root_link_ang_vel_b

  z_error = torch.square(vel_cmd[:, 2] - actual_ang_vel[:, 2])
  xy_error = torch.sum(torch.square(actual_ang_vel[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


# =====================================================================
# Swing trajectory tracking reward
# =====================================================================


def track_swing_trajectory(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.03,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the swing foot for following the sin/cos reference trajectory.

  Measures the 3-D distance between the actual swing foot position and the
  reference point on the planned arc (computed by the command term).
  The reward is: exp(-||swing_foot - ref||^2 / std^2).

  Active for the entire step (all phases), so the agent is rewarded for
  tracking the *curve*, not just the final landing position.
  """
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  # Reference swing foot position in world frame.
  phase = command_term.command[:, 8]
  swing_ref_w = command_term._compute_swing_trajectory_w(phase)

  # Determine which foot is the swing foot.
  support_side = command_term.command[:, 9]  # -1=left, +1=right
  # swing index: support=+1(right) -> swing=left(idx 0)
  #              support=-1(left)  -> swing=right(idx 1)
  swing_idx = ((1 - support_side) / 2).long()

  # Actual swing foot position in world frame (xyz).
  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # (B, 2, 3)
  swing_foot_pos = torch.gather(
    foot_pos, 1, swing_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3)
  ).squeeze(1)  # (B, 3)

  error = torch.sum(torch.square(swing_foot_pos - swing_ref_w), dim=-1)
  return torch.exp(-error / std**2)


def track_swing_trajectory_xy(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking only the xy component of the swing trajectory.

  Useful for separate weighting of horizontal vs vertical tracking.
  """
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  phase = command_term.command[:, 8]
  swing_ref_w = command_term._compute_swing_trajectory_w(phase)

  support_side = command_term.command[:, 9]
  swing_idx = ((1 - support_side) / 2).long()

  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (B, 2, 2)
  swing_foot_xy = torch.gather(
    foot_pos, 1, swing_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
  ).squeeze(1)  # (B, 2)

  error = torch.sum(torch.square(swing_foot_xy - swing_ref_w[:, :2]), dim=-1)
  return torch.exp(-error / std**2)


def track_swing_trajectory_z(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.02,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking only the z (height) component of the swing trajectory.

  Encourages proper foot clearance following the sin arc.
  """
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  phase = command_term.command[:, 8]
  swing_ref_w = command_term._compute_swing_trajectory_w(phase)

  support_side = command_term.command[:, 9]
  swing_idx = ((1 - support_side) / 2).long()

  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (B, 2)
  swing_foot_z = torch.gather(foot_z, 1, swing_idx.unsqueeze(1)).squeeze(1)

  error = torch.square(swing_foot_z - swing_ref_w[:, 2])
  return torch.exp(-error / std**2)


# =====================================================================
# Footstep placement reward
# =====================================================================


def track_footstep_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  std: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward placing the swing foot near the planned landing position.

  Active only during the final portion of the step (phase > 0.7) to avoid
  conflicting with mid-swing trajectory tracking.
  """
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (B, 2, 2)

  phase = command_term.command[:, 8]
  support_side = command_term.command[:, 9]
  swing_idx = ((1 - support_side) / 2).long()

  swing_foot_pos = torch.gather(
    foot_pos, 1, swing_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
  ).squeeze(1)  # (B, 2)

  des_foot_pos = command_term.next_foot_pos_w  # (B, 2)
  error = torch.sum(torch.square(swing_foot_pos - des_foot_pos), dim=-1)
  reward = torch.exp(-error / std**2)

  late_phase = (phase > 0.7).float()
  return reward * late_phase


# =====================================================================
# Gait / contact rewards
# =====================================================================


class support_foot_contact:
  """Reward keeping the support foot in contact with the ground."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    command_term: LIPCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene[sensor_name]

    support_side = command_term.command[:, 9]

    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()

    # support foot index: -1 -> 0 (left), +1 -> 1 (right)
    support_idx = ((1 + support_side) / 2).long()
    support_contact = torch.gather(
      in_contact, 1, support_idx.unsqueeze(1)
    ).squeeze(1)

    return support_contact


def step_frequency(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """Reward contact transitions occurring near the planned step boundary."""
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  contact_sensor: ContactSensor = env.scene[sensor_name]

  phase = command_term.command[:, 8]

  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  any_first_contact = torch.any(first_contact, dim=-1).float()

  return any_first_contact * torch.exp(-10.0 * torch.square(phase - 0.9))


def feet_air_time(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  threshold: float = 0.4,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward balanced single-stance air/contact times (from velocity task)."""
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data

  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
  )[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)

  vel_cmd = command_term.vel_command
  total_cmd = torch.norm(vel_cmd[:, :2], dim=1) + torch.abs(vel_cmd[:, 2])
  reward *= (total_cmd > command_threshold).float()
  return reward


def feet_slip(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise foot sliding (xy velocity while in contact)."""
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  vel_cmd = command_term.vel_command
  total_cmd = torch.norm(vel_cmd[:, :2], dim=1) + torch.abs(vel_cmd[:, 2])
  active = (total_cmd > command_threshold).float()

  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()

  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_sq = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact, dim=1)

  return vel_sq * active


# =====================================================================
# Swing foot clearance (cost)
# =====================================================================


def swing_foot_clearance(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_height: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise swing foot height deviating from target during mid-swing.

  Weighted by xy velocity so static feet aren't penalised.
  """
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]

  support_side = command_term.command[:, 9]
  phase = command_term.command[:, 8]

  swing_idx = ((1 - support_side) / 2).long()

  swing_foot_z = torch.gather(foot_z, 1, swing_idx.unsqueeze(1)).squeeze(1)
  swing_foot_vel = torch.gather(
    foot_vel_xy, 1, swing_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
  ).squeeze(1)
  vel_norm = torch.norm(swing_foot_vel, dim=-1)

  delta = torch.abs(swing_foot_z - target_height)
  cost = delta * vel_norm

  mid_swing = ((phase > 0.2) & (phase < 0.7)).float()
  return cost * mid_swing


# =====================================================================
# Regularisation rewards (shared with velocity task)
# =====================================================================


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalise self-collisions."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise excessive body angular velocities (xy only)."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalise whole-body angular momentum."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def upright_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for keeping the base upright.

  Returns the z-component of the projected gravity in body frame,
  which is ~1.0 when perfectly upright and ~0.0 when horizontal.
  projected_gravity_b is normalised to unit length, so z \in [-1, 1].
  We clamp to [0, 1] so a flipped robot gets zero reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # projected_gravity_b[:, 2] ≈ -1 when upright (gravity points -z in world).
  # We want reward ≈ 1 when upright, 0 when fallen.
  uprightness = -asset.data.projected_gravity_b[:, 2]
  return torch.clamp(torch.square(uprightness), min=0.0)