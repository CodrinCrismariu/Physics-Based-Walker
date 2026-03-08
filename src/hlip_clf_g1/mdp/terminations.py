"""Termination conditions for the LIP walking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Terminate on illegal body contacts (e.g. knee hitting ground)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.any(sensor.data.found, dim=-1)

def pelvis_too_low(
  env: ManagerBasedRlEnv,
  minimum_distance: float,
  pelvis_body_name: str,
  foot_body_name: str,
) -> torch.Tensor:
  """Terminate when the pelvis height is too close to the feet."""
  robot = env.scene["robot"]
  
  pelvis_ids, _ = robot.find_bodies(pelvis_body_name)
  foot_ids, _ = robot.find_bodies(foot_body_name)

  pelvis_pos_w = robot.data.body_link_pos_w[:, pelvis_ids, :]  # (num_envs, 1, 3)
  foot_pos_w = robot.data.body_link_pos_w[:, foot_ids, :]  # (num_envs, 2, 3)

  # Check distance between pelvis Z and the maximum foot Z
  pelvis_z = pelvis_pos_w[:, 0, 2]
  max_foot_z = torch.max(foot_pos_w[:, :, 2], dim=-1)[0]
  
  return (pelvis_z - max_foot_z) < minimum_distance
