"""Reward functions from the LIP walking task needed by HLIP.

Only includes upright_reward and self_collision_cost which are imported
by the HLIP MDP module.
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


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalise self-collisions."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  found = sensor.data.found
  if found.ndim == 1:
    return found.float()
  return found.reshape(found.shape[0], -1).any(dim=1).float()


def upright_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for keeping the base upright.

  Returns the z-component of the projected gravity in body frame,
  which is ~1.0 when perfectly upright and ~0.0 when horizontal.
  projected_gravity_b is normalised to unit length, so z in [-1, 1].
  We clamp to [0, 1] so a flipped robot gets zero reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # projected_gravity_b[:, 2] ≈ -1 when upright (gravity points -z in world).
  # We want reward ≈ 1 when upright, 0 when fallen.
  uprightness = -asset.data.projected_gravity_b[:, 2]
  return torch.clamp(torch.square(uprightness), min=0.0)
