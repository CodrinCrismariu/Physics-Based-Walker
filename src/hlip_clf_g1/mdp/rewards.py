"""Reward functions for the HLIP + CLF walking task.

CLF-specific rewards that encourage the agent to:
  - Converge to the desired HLIP orbit (CLF rewards).
  - Track reference trajectories for CoM, pelvis, swing foot, upper body.
  - Maintain holonomic constraints on the stance foot.
  - Keep proper gait phase contact timing.

Standard locomotion rewards (orientation, joint limits, action rate, etc.)
are imported from the base environment MDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# =====================================================================
# CLF rewards
# =====================================================================


def clf_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_eta_err: float = 0.15,
  eps: float = 1e-6,
) -> torch.Tensor:
  """CLF value reward: r = exp(-V / V_max).

  Encourages minimisation of the CLF value V. Higher reward when the
  tracking error (measured by V) is small.
  """
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  v = cmd.v
  max_clf = cmd.clf.lambda_max * max_eta_err**2 + eps
  return torch.exp(-torch.clamp(v, max=5.0 * max_clf) / max_clf)


def clf_decreasing_condition(
  env: ManagerBasedRlEnv,
  command_name: str,
  alpha: float = 1.0,
  eta_max: float = 0.15,
  eta_dot_max: float = 0.5,
  eps: float = 1e-6,
) -> torch.Tensor:
  """Penalty for violating the CLF decrease condition.

  Returns a normalised violation in [0, 1]:
    penalty = clamp((Vdot + alpha*V) / max_violation)

  Only penalises when Vdot + alpha*V > 0 (i.e., V is not decreasing
  fast enough).
  """
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  v = cmd.v
  vdot = cmd.vdot

  lambda_max = cmd.clf.lambda_max
  norm_P = cmd.clf.norm_P

  max_violation = (
    2.0 * norm_P * eta_max * eta_dot_max
    + alpha * lambda_max * eta_max**2
    + eps
  )
  violation = torch.clamp(vdot + alpha * v, min=0.0)
  return torch.clamp(violation / max_violation, min=0.0, max=1.0)


def vdot_tanh(
  env: ManagerBasedRlEnv,
  command_name: str,
  alpha: float = 1.0,
) -> torch.Tensor:
  """Tanh reward for CLF decay condition satisfaction.

  Returns tanh(-(Vdot + alpha*V)). Positive when condition is satisfied
  (V is decreasing with margin), negative otherwise.
  """
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return torch.tanh(-(cmd.vdot + alpha * cmd.v))


def v_dot_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  eta_max: float = 0.15,
  eta_dot_max: float = 0.5,
  eps: float = 1e-6,
) -> torch.Tensor:
  """Penalise positive Vdot (CLF value increasing)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  norm_P = cmd.clf.norm_P
  max_violation = 2.0 * norm_P * eta_max * eta_dot_max + eps
  return torch.tanh(torch.clamp(cmd.vdot, min=0.0) / max_violation)


# =====================================================================
# Reference tracking rewards
# =====================================================================


def reference_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  term_std: list[float] | Sequence[float],
  term_weight: list[float] | Sequence[float],
) -> torch.Tensor:
  """Per-dimension weighted exponential reference tracking.

  r = sum_i(w_i * exp(-e_i^2 / std_i^2)) / sum(w_i)
  """
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  err = cmd.y_act - cmd.y_out

  weight = torch.as_tensor(term_weight, dtype=err.dtype, device=err.device)
  std = torch.as_tensor(term_std, dtype=err.dtype, device=err.device)

  err_sq_scaled = err**2 / std**2
  reward_per_dim = weight * torch.exp(-err_sq_scaled)
  return reward_per_dim.sum(dim=1) / torch.sum(weight)


def reference_vel_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  term_std: list[float] | Sequence[float],
  term_weight: list[float] | Sequence[float],
) -> torch.Tensor:
  """Per-dimension weighted exponential velocity reference tracking."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  err = cmd.dy_act - cmd.dy_out

  weight = torch.as_tensor(term_weight, dtype=err.dtype, device=err.device)
  std = torch.as_tensor(term_std, dtype=err.dtype, device=err.device)

  err_sq_scaled = err**2 / std**2
  reward_per_dim = weight * torch.exp(-err_sq_scaled)
  return reward_per_dim.sum(dim=1) / torch.sum(weight)


# =====================================================================
# Holonomic constraint rewards
# =====================================================================


def holonomic_constraint(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma_pose: float = 0.2236,  # sqrt(5 * 0.01)
  z_offset: float = 0.036,
) -> torch.Tensor:
  """Stance foot holonomic pose constraint reward.

  Penalises drift of the stance foot from its recorded position at
  the beginning of the stance phase.

  r = exp(-||e_pose||^2 / sigma_pose^2)
  where e_pose = [dx, dy, dz, roll, d_yaw].
  """
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)

  # Planar position error.
  delta_xy = cmd.stance_foot_pos[:, :2] - cmd.stance_foot_pos_0[:, :2]

  # Vertical error.
  delta_z = (cmd.stance_foot_pos[:, 2] - cmd.stance_foot_pos_0[:, 2]).unsqueeze(-1)

  # Orientation: get current stance foot euler.
  from hlip_clf_g1.mdp.hlip_command import _euler_from_quat

  foot_quat_w = cmd.robot.data.body_link_quat_w[:, cmd._foot_body_ids, :]
  stance_quat = torch.gather(
    foot_quat_w,
    1,
    cmd.stance_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 4),
  ).squeeze(1)
  stance_ori = _euler_from_quat(stance_quat)

  roll = stance_ori[:, 0].unsqueeze(-1)
  psi_err = wrap_to_pi(stance_ori[:, 2] - cmd.stance_foot_ori_0[:, 2]).unsqueeze(-1)

  e_pose = torch.cat([delta_xy, delta_z, roll, psi_err], dim=-1)
  return torch.exp(-(e_pose**2).sum(dim=-1) / sigma_pose**2)


def holonomic_constraint_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma_vel: float = 0.3162,  # sqrt(0.1)
) -> torch.Tensor:
  """Stance foot velocity constraint reward.

  Penalises non-zero stance foot velocity (it should be still).

  r = exp(-||[v, omega_z]||^2 / sigma_vel^2)
  """
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)

  v = cmd.stance_foot_vel
  wz = cmd.stance_foot_ang_vel[:, 2].unsqueeze(-1)
  e_vel = torch.cat([v, wz], dim=-1)
  return torch.exp(-(e_vel**2).sum(dim=-1) / sigma_vel**2)


# =====================================================================
# Phase contact reward
# =====================================================================


def phase_contact(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """Reward foot contact matching the expected gait phase.

  Uses the HLIP command term's stance/swing indices to determine which
  foot should be in contact.
  """
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]

  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()  # (B, 2)

  # Stance foot should be in contact, swing foot should not.
  # in_contact[:, 0] = left foot, in_contact[:, 1] = right foot
  # stance_idx: 0 = left stance, 1 = right stance
  stance_contact = torch.gather(
    in_contact, 1, cmd.stance_idx.unsqueeze(1)
  ).squeeze(1)
  swing_no_contact = 1.0 - torch.gather(
    in_contact, 1, cmd.swing_idx.unsqueeze(1)
  ).squeeze(1)

  return stance_contact + swing_no_contact


# =====================================================================
# Ankle / regularisation rewards
# =====================================================================


def ankle_roll_zero(
  env: ManagerBasedRlEnv,
  std: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping ankle roll joints near zero position."""
  asset: Entity = env.scene[asset_cfg.name]
  ankle_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.exp(-torch.sum(ankle_pos**2, dim=-1) / std**2)


def foot_clearance(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_height: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise swing foot height deviation during mid-swing."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  # Swing foot z.
  foot_z = asset.data.body_link_pos_w[:, cmd._foot_body_ids, 2]  # (B, 2)
  swing_z = torch.gather(
    foot_z, 1, cmd.swing_idx.unsqueeze(1)
  ).squeeze(1)

  error = torch.square(swing_z - target_height)

  # Only active during mid-swing.
  mid_swing = ((cmd.phase_var > 0.2) & (cmd.phase_var < 0.7)).float()
  return error * mid_swing


def velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking the commanded linear velocity (xy).

  Uses world-frame horizontal velocity rotated into yaw-only body frame.
  """
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  vel_w = asset.data.root_link_lin_vel_w[:, :2]
  heading = asset.data.heading_w
  cos_h = torch.cos(-heading)
  sin_h = torch.sin(-heading)
  actual_vel = torch.stack(
    [
      vel_w[:, 0] * cos_h - vel_w[:, 1] * sin_h,
      vel_w[:, 0] * sin_h + vel_w[:, 1] * cos_h,
    ],
    dim=-1,
  )

  error = torch.sum(torch.square(cmd.vel_command[:, :2] - actual_vel), dim=1)
  return torch.exp(-error / std**2)


def angular_velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking the commanded yaw rate."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  ang_vel_b = asset.data.root_link_ang_vel_b
  error = torch.square(cmd.vel_command[:, 2] - ang_vel_b[:, 2])
  return torch.exp(-error / std**2)
