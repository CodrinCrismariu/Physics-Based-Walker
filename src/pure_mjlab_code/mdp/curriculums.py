"""Curriculum functions for the HLIP + CLF walking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def commands_hlip(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  """Velocity command curriculum for the HLIP task.

  Gradually increases the range of commanded velocities as training progresses.
  """
  del env_ids

  from pure_mjlab_code.mdp.hlip_command import HLIPCommandTerm, HLIPCommandCfg
  from typing import cast

  command_term: HLIPCommandTerm = env.command_manager.get_term(command_name)
  cfg = cast(HLIPCommandCfg, command_term.cfg)

  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]

  return {}


def clf_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  reward_name: str,
  weight_stages: list[dict],
) -> dict[str, torch.Tensor]:
  """CLF reward weight curriculum.

  Gradually increases the weight of CLF-based rewards as the agent
  learns to track the reference trajectory.

  Args:
    command_name: Name of the HLIP command term.
    reward_name: Name of the reward term whose weight is being adjusted.
    weight_stages: List of dicts with 'step' and 'weight' keys.
  """
  del env_ids

  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_cfg = env.reward_manager.get_term_cfg(reward_name)
      reward_cfg.weight = stage["weight"]

  return {}
