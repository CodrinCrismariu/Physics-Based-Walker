"""Curriculum functions for the LIP walking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_lip(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Terrain curriculum based on distance walked, adapted for LIP task."""
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  from mjlab.tasks.lip.mdp.lip_command import LIPCommand
  command_term: LIPCommand = env.command_manager.get_term(command_name)
  vel_command = command_term.vel_command

  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  move_up = distance > terrain_generator.size[0] / 2
  move_down = (
    distance
    < torch.norm(vel_command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)
  return torch.mean(terrain.terrain_levels.float())


def commands_lip(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  """Velocity command curriculum for the LIP task.

  Gradually increases the range of commanded velocities as training progresses.
  """
  del env_ids

  from mjlab.tasks.lip.mdp.lip_command import LIPCommand, LIPCommandCfg
  from typing import cast

  command_term: LIPCommand = env.command_manager.get_term(command_name)
  cfg = cast(LIPCommandCfg, command_term.cfg)

  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]

  return {}
