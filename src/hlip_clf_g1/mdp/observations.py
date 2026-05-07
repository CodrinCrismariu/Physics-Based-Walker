"""Observation functions for the HLIP + CLF walking task.

Provides HLIP-specific observations: reference/actual trajectories,
trajectory errors, phase signals, foot velocities, and contact state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEPTH_CAMERA_EPISODE_NOISE_LEVELS: dict[tuple[int, str], torch.Tensor] = {}


def _prime_hlip_command_if_pending(cmd) -> None:
  """Ensure command-derived observations are initialized on reset.

  During ``env.reset()``, observation computation can occur before
  ``command_manager.compute()``. For HLIP command terms, this helper primes
  reset-time state exactly once so the first observation is not all zeros.
  """
  prime_fn = getattr(cmd, "prime_for_observation", None)
  if callable(prime_fn):
    prime_fn()


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
  _prime_hlip_command_if_pending(cmd)
  return cmd.y_out


def act_traj(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Actual trajectory positions. Shape (num_envs, n_outputs)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  _prime_hlip_command_if_pending(cmd)
  return cmd.y_act


def traj_error(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Trajectory error (ref - actual). Shape (num_envs, n_outputs)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  _prime_hlip_command_if_pending(cmd)
  return cmd.y_out - cmd.y_act


def ref_traj_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Reference trajectory velocities. Shape (num_envs, n_outputs)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  _prime_hlip_command_if_pending(cmd)
  return cmd.dy_out


def act_traj_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Actual trajectory velocities. Shape (num_envs, n_outputs)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  _prime_hlip_command_if_pending(cmd)
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
  _prime_hlip_command_if_pending(cmd)
  return torch.sin(2 * torch.pi * cmd.tp).unsqueeze(-1)


def cos_phase(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """cos(2*pi*tp) phase signal. Shape (num_envs, 1)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  _prime_hlip_command_if_pending(cmd)
  return torch.cos(2 * torch.pi * cmd.tp).unsqueeze(-1)


def domain_flag(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Stance side indicator: 0=left, 1=right. Shape (num_envs, 1)."""
  from hlip_clf_g1.mdp.hlip_command import HLIPCommandTerm

  cmd: HLIPCommandTerm = env.command_manager.get_term(command_name)
  _prime_hlip_command_if_pending(cmd)
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
  _prime_hlip_command_if_pending(cmd)
  return cmd.vel_command


def hlip_velocity_command_linear_only(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Commanded velocity with angular component masked to zero.

  Returns [vx, vy, 0.0] with shape (num_envs, 3), so student observation can
  receive only linear command information while keeping the same tensor layout.
  """
  cmd = hlip_velocity_command(env, command_name)
  cmd_linear_only = cmd.clone()
  cmd_linear_only[:, 2] = 0.0
  return cmd_linear_only


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


def heightmap_data(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Heightmap data. Returns lengths of rays."""
  return env.scene.sensors["heightmap"].data.distances

def rgbd_camera_data(
  env: ManagerBasedRlEnv,
  sensor_name: str = "head_camera",
) -> torch.Tensor:
  """Depth camera data flattened per environment.

  Returns shape (num_envs, H*W), which is compatible with MLP observation
  concatenation. Use ``sensor_name`` to select a specific camera sensor.
  """
  sensor = env.scene.sensors[sensor_name]
  depth = sensor.data.depth
  if depth is None:
    raise ValueError(f"Depth data is not available for sensor '{sensor_name}'.")
  return depth.reshape(depth.shape[0], -1)


def depth_camera_chw_data(
  env: ManagerBasedRlEnv,
  sensor_name: str = "head_camera",
) -> torch.Tensor:
  """Depth camera data in CNN-friendly channel-first image layout.

  Returns shape (num_envs, 1, H, W). This is intended for CNNModel inputs.
  """
  sensor = env.scene.sensors[sensor_name]
  depth = sensor.data.depth
  if depth is None:
    raise ValueError(f"Depth data is not available for sensor '{sensor_name}'.")

  if depth.ndim != 4:
    raise ValueError(
      f"Expected depth tensor with 4 dims [B,H,W,C] or [B,C,H,W], got shape {tuple(depth.shape)}."
    )

  # Camera sensor provides [B, H, W, C] with C=1.
  if depth.shape[-1] == 1:
    return depth.permute(0, 3, 1, 2).contiguous()
  # Accept already channel-first tensors for robustness.
  if depth.shape[1] == 1:
    return depth

  raise ValueError(
    f"Expected a single-channel depth image, got shape {tuple(depth.shape)}."
  )


def _apply_close_depth_bleed(
  depth: torch.Tensor,
  min_depth: float,
  close_depth_bleed_radius: int,
  close_depth_bleed_prob: float,
  close_depth_bleed_max_depth: float,
  close_depth_bleed_level: float | torch.Tensor = 1.0,
) -> torch.Tensor:
  """Randomly propagate close depth pixels into nearby farther pixels.

  ``close_depth_bleed_radius`` is the maximum radius. The bleed grows outward
  one pixel at a time, and each new pixel must be reached through a pixel that
  already propagated. The effective radius shrinks with distance, so the
  closest pixels corrupt the widest neighborhood.
  """
  max_radius = int(close_depth_bleed_radius)
  bleed_prob = torch.as_tensor(
    close_depth_bleed_prob,
    device=depth.device,
    dtype=depth.dtype,
  ).clamp(0.0, 1.0)
  bleed_level = torch.as_tensor(
    close_depth_bleed_level,
    device=depth.device,
    dtype=depth.dtype,
  ).clamp(0.0, 1.0)
  if max_radius <= 0 or not bool(torch.any((bleed_prob * bleed_level) > 0.0).item()):
    return depth

  def _close_weight(local_close_depth: torch.Tensor) -> torch.Tensor:
    if close_depth_bleed_max_depth > min_depth:
      return (
        (close_depth_bleed_max_depth - local_close_depth)
        / (close_depth_bleed_max_depth - min_depth)
      ).clamp(0.0, 1.0)
    return torch.ones_like(depth)

  source_weight = _close_weight(depth)
  source_depth = torch.where(
    source_weight > 0.0,
    depth,
    torch.full_like(depth, float("inf")),
  )
  bled_depth = depth

  for radius in range(1, max_radius + 1):
    local_close_depth = -F.max_pool2d(
      -source_depth,
      kernel_size=3,
      stride=1,
      padding=1,
    )

    close_weight = _close_weight(local_close_depth)
    radius_threshold = (radius - 0.5) / max_radius
    can_reach = (close_weight * bleed_level) >= radius_threshold
    can_bleed = can_reach & (local_close_depth < depth)

    per_pixel_prob = bleed_prob * bleed_level * close_weight
    bleed_mask = (torch.rand_like(depth) < per_pixel_prob) & can_bleed
    candidate_depth = torch.where(can_bleed, local_close_depth, depth)
    bleed_strength = torch.rand_like(depth) * bleed_level * (0.5 + 0.5 * close_weight)
    bled_depth = torch.where(
      bleed_mask,
      torch.lerp(depth, candidate_depth, bleed_strength),
      bled_depth,
    )

    source_depth = torch.where(
      bleed_mask,
      torch.minimum(source_depth, local_close_depth),
      source_depth,
    )

  return bled_depth


def _get_depth_camera_episode_noise_level(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  depth: torch.Tensor,
  randomize_depth_noise_per_episode: bool,
) -> torch.Tensor:
  """Return per-env depth corruption multiplier sampled at episode reset."""
  shape = (depth.shape[0], 1, 1, 1)
  if not randomize_depth_noise_per_episode:
    return torch.ones(shape, device=depth.device, dtype=depth.dtype)

  key = (id(env), sensor_name)
  noise_level = _DEPTH_CAMERA_EPISODE_NOISE_LEVELS.get(key)
  if (
    noise_level is None
    or noise_level.shape != shape
    or noise_level.device != depth.device
    or noise_level.dtype != depth.dtype
  ):
    noise_level = torch.rand(shape, device=depth.device, dtype=depth.dtype)
    _DEPTH_CAMERA_EPISODE_NOISE_LEVELS[key] = noise_level

  return noise_level


def _reset_depth_camera_episode_noise(env_ids: torch.Tensor | slice | None = None) -> None:
  """Resample depth corruption multipliers for reset environments."""
  for noise_level in _DEPTH_CAMERA_EPISODE_NOISE_LEVELS.values():
    if env_ids is None or isinstance(env_ids, slice):
      noise_level.uniform_(0.0, 1.0)
      continue

    reset_ids = env_ids.to(device=noise_level.device, dtype=torch.long)
    reset_ids = reset_ids[(reset_ids >= 0) & (reset_ids < noise_level.shape[0])]
    if reset_ids.numel() > 0:
      noise_level[reset_ids] = torch.rand(
        (reset_ids.numel(), *noise_level.shape[1:]),
        device=noise_level.device,
        dtype=noise_level.dtype,
      )


def depth_camera_sparse_terrain_chw_data(
  env: ManagerBasedRlEnv,
  sensor_name: str = "head_camera",
  min_depth: float = 0.1,
  max_depth: float = 10.0,
  depth_noise_scale: float = 0.1,
  pixel_dropout_prob: float = 0.0,
  close_depth_bleed_radius: int = 0,
  close_depth_bleed_prob: float = 0.0,
  close_depth_bleed_max_depth: float = 2.0,
  randomize_depth_noise_per_episode: bool = False,
) -> torch.Tensor:
  """Depth preprocessing with Gaussian depth noise and optional pixel dropout.

  Keeps depth in metric units after sanitization/clamping and applies
  Gaussian per-pixel noise with standard deviation
  ``depth_noise_scale * depth``. ``pixel_dropout_prob`` turns random pixels into
  max-range returns. The legacy close-depth bleed parameters are retained for
  old configs, but should be left at zero to avoid edge bleeding.
  """
  depth = depth_camera_chw_data(env=env, sensor_name=sensor_name).to(torch.float32)
  episode_noise_level = _get_depth_camera_episode_noise_level(
    env,
    sensor_name=sensor_name,
    depth=depth,
    randomize_depth_noise_per_episode=randomize_depth_noise_per_episode,
  )

  # Replace invalid values and bound the physical sensing range.
  depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=min_depth)
  depth = depth.clamp(min=min_depth, max=max_depth)

  depth = _apply_close_depth_bleed(
    depth,
    min_depth=min_depth,
    close_depth_bleed_radius=close_depth_bleed_radius,
    close_depth_bleed_prob=close_depth_bleed_prob,
    close_depth_bleed_max_depth=close_depth_bleed_max_depth,
    close_depth_bleed_level=episode_noise_level,
  )

  if depth_noise_scale > 0.0:
    noise = (
      torch.randn_like(depth)
      * float(depth_noise_scale)
      * depth
      * episode_noise_level
    )
    depth = depth + noise
    depth = depth.clamp(min=min_depth, max=max_depth)

  if pixel_dropout_prob > 0.0:
    dropout_prob = torch.as_tensor(
      pixel_dropout_prob,
      device=depth.device,
      dtype=depth.dtype,
    ).clamp(0.0, 1.0)
    dropout_mask = torch.rand_like(depth) < dropout_prob
    depth = torch.where(dropout_mask, torch.full_like(depth, max_depth), depth)

  return depth


depth_camera_sparse_terrain_chw_data.reset = _reset_depth_camera_episode_noise
