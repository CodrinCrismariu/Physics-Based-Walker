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


def commanded_but_stationary(
  env: ManagerBasedRlEnv,
  command_name: str,
  stationary_duration_s: float,
  min_command_speed: float,
  max_stationary_speed: float,
) -> torch.Tensor:
  """Terminate if commanded linear speed is non-zero but robot remains near-stationary.

  This condition activates only after ``stationary_duration_s`` from episode start.
  """
  command = env.command_manager.get_command(command_name)
  commanded_speed = torch.linalg.norm(command[:, :2], dim=1)

  robot = env.scene["robot"]
  actual_speed = torch.linalg.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)

  elapsed_s = env.episode_length_buf.float() * env.step_dt
  wait_elapsed = elapsed_s >= stationary_duration_s
  has_nonzero_command = commanded_speed >= min_command_speed
  is_stationary = actual_speed <= max_stationary_speed

  return wait_elapsed & has_nonzero_command & is_stationary


def mpc_foothold_tracking_failure(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_foothold_error: float,
  check_phase_threshold: float = 0.95,
) -> torch.Tensor:
  """Terminate when swing-foot placement deviates too far from MPC foothold target.

  The check is applied near touchdown (end of each half-gait) to evaluate the
  completed step against the active MPC foothold target in stance-local frame.
  """
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)

  # Check once per step near touchdown.
  is_touchdown_window = cmd.phase_var >= check_phase_threshold

  # Only evaluate when MPC has an active plan.
  has_plan = cmd._mpc_has_plan if hasattr(cmd, "_mpc_has_plan") else torch.zeros_like(is_touchdown_window)

  # Both terms are expressed in stance-local coordinates.
  swing_local = cmd.y_act[:, 6:9]
  target_local = cmd._mpc_active_foot_target
  foothold_error = torch.linalg.norm(swing_local - target_local, dim=1)

  return is_touchdown_window & has_plan & (foothold_error > max_foothold_error)
