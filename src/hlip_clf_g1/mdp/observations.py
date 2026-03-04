"""Observation functions for the HLIP + CLF walking task.

Provides HLIP-specific observations: reference/actual trajectories,
trajectory errors, phase signals, foot velocities, and contact state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# =====================================================================
# Reference / actual trajectory observations
# =====================================================================


def ref_traj(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Reference trajectory positions. Shape (num_envs, n_outputs)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return cmd.y_out


def act_traj(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Actual trajectory positions. Shape (num_envs, n_outputs)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return cmd.y_act


def traj_error(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Trajectory error (ref - actual). Shape (num_envs, n_outputs)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return cmd.y_out - cmd.y_act


def ref_traj_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Reference trajectory velocities. Shape (num_envs, n_outputs)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return cmd.dy_out


def act_traj_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Actual trajectory velocities. Shape (num_envs, n_outputs)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return cmd.dy_act


# =====================================================================
# Phase observations
# =====================================================================


def sin_phase(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """sin(2*pi*tp) phase signal. Shape (num_envs, 1)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return torch.sin(2 * torch.pi * cmd.tp).unsqueeze(-1)


def cos_phase(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """cos(2*pi*tp) phase signal. Shape (num_envs, 1)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return torch.cos(2 * torch.pi * cmd.tp).unsqueeze(-1)


def domain_flag(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Stance side indicator: 0=left, 1=right. Shape (num_envs, 1)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return cmd.stance_idx.float().unsqueeze(-1)


# =====================================================================
# Foot velocity observations
# =====================================================================


def foot_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Foot linear velocities for both feet. Shape (num_envs, 6)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.body_link_lin_vel_w[:, cmd._foot_body_ids, :]  # (B, 2, 3)
  return vel.reshape(vel.shape[0], -1)


def foot_ang_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Foot angular velocities for both feet. Shape (num_envs, 6)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.body_link_ang_vel_w[:, cmd._foot_body_ids, :]  # (B, 2, 3)
  return vel.reshape(vel.shape[0], -1)


# =====================================================================
# Contact observations
# =====================================================================


def contact_state(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Binary foot contact state. Shape (num_envs, 2)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()


# =====================================================================
# Velocity command observation
# =====================================================================


def hlip_velocity_command(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Commanded walking velocity [vx, vy, yaw_rate]. Shape (num_envs, 3)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  return cmd.vel_command


# =====================================================================
# Base height observation
# =====================================================================


def root_quat_w(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Root quaternion (w, x, y, z). Shape (num_envs, 4)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_quat_w


def base_z(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Base height above ground. Shape (num_envs, 1)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)
