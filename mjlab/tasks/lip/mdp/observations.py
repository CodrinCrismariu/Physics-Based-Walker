"""Observation functions for the LIP walking task.

Provides LIP-specific observations: the full command tensor, velocity
command, swing trajectory reference, support foot info, and foot contacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.lip.mdp.lip_command import LIPCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def lip_command(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Full LIP command tensor.

  Returns: (num_envs, 10) containing:
    [0:3]  velocity command (vx, vy, yaw_rate) - body frame
    [3:6]  swing foot trajectory reference (x, y, z) - body frame
    [6:8]  target footstep relative to CoM (x, y) - body frame
    [8]    step phase (0 to 1)
    [9]    support foot indicator (-1=left, +1=right)
  """
  return env.command_manager.get_command(command_name)


def lip_velocity_command(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Desired walking velocity command [vx, vy, yaw_rate].

  Returns: (num_envs, 3)
  """
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  return command_term.vel_command


def support_foot_rel_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Support foot position relative to the robot's root, in body frame.

  Returns: (num_envs, 2)
  """
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  asset: Entity = env.scene[asset_cfg.name]

  root_pos_xy = asset.data.root_link_pos_w[:, :2]
  support_rel_w = command_term.support_foot_pos_w - root_pos_xy

  heading = asset.data.heading_w
  cos_h = torch.cos(-heading)
  sin_h = torch.sin(-heading)
  rel_b_x = support_rel_w[:, 0] * cos_h - support_rel_w[:, 1] * sin_h
  rel_b_y = support_rel_w[:, 0] * sin_h + support_rel_w[:, 1] * cos_h

  return torch.stack([rel_b_x, rel_b_y], dim=-1)


def swing_trajectory_ref(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Current swing foot trajectory reference in body frame (x, y, z).

  This is already packed in command[3:6] but exposed as a separate
  observation for flexibility in observation group composition.

  Returns: (num_envs, 3)
  """
  return env.command_manager.get_command(command_name)[:, 3:6]


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Foot heights (z) from site positions. Shape (num_envs, num_sites)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Binary foot contact. Shape (num_envs, num_feet)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Log-scaled foot contact forces. Shape (num_envs, num_feet*3)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  forces_flat = sensor.data.force.flatten(start_dim=1)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
