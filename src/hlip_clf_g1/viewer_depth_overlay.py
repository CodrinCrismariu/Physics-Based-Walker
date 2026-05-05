"""Viser depth-feed hook for displaying processed depth observations."""

from __future__ import annotations

from inspect import signature
from typing import Any, Callable

import numpy as np


def _make_observation_depth_provider(env: Any, sensor: Any) -> Callable[[int], Any] | None:
  """Find a depth observation term that corresponds to a camera sensor."""
  obs_manager = getattr(env.unwrapped, "observation_manager", None)
  if obs_manager is None:
    return None

  sensor_names = {sensor.camera_name, sensor.cfg.name}
  for group_name, term_names in obs_manager.active_terms.items():
    if len(term_names) != 1:
      continue

    term_name = term_names[0]
    term_cfg = obs_manager.get_term_cfg(group_name, term_name)
    params = getattr(term_cfg, "params", None) or {}
    if params.get("sensor_name") not in sensor_names:
      continue

    term_dims = obs_manager.group_obs_term_dim[group_name]
    if len(term_dims) != 1 or len(term_dims[0]) != 3:
      continue

    if "depth" not in f"{group_name}_{term_name}".lower():
      continue

    def _provider(
      env_idx: int,
      group_name: str = group_name,
      term_name: str = term_name,
    ) -> Any:
      obs_buffer = obs_manager.compute(update_history=False)
      group_obs = obs_buffer[group_name]
      if isinstance(group_obs, dict):
        return group_obs[term_name][env_idx]
      return group_obs[env_idx]

    return _provider

  return None


def install_viser_depth_observation_overlay() -> None:
  """Show processed depth observations in Viser camera feeds when available."""
  try:
    from mjlab.sensor import CameraSensor
    from mjlab.viewer.viser.camera_viewer import ViserCameraViewer
    from mjlab.viewer.viser.overlays import ViserCameraOverlays
  except Exception:
    return

  if getattr(ViserCameraOverlays, "_hlip_depth_observation_overlay_installed", False):
    return

  original_init = ViserCameraViewer.__init__
  if "depth_image_provider" not in signature(original_init).parameters:

    def _camera_viewer_init(
      self: Any,
      *args: Any,
      depth_image_provider: Callable[[int], Any] | None = None,
      **kwargs: Any,
    ) -> None:
      original_init(self, *args, **kwargs)
      self._depth_image_provider = depth_image_provider

    ViserCameraViewer.__init__ = _camera_viewer_init

    def _camera_viewer_update(
      self: Any,
      sim_data: Any,
      env_idx: int = 0,
      scene_offset: np.ndarray | None = None,
    ) -> None:
      data = self._camera_sensor.data

      if self._has_rgb and self._rgb_handle is not None and data.rgb is not None:
        rgb_np = data.rgb[env_idx].cpu().numpy()
        if self._needs_upsampling:
          scale = self._display_height // rgb_np.shape[0]
          rgb_np = self._upsample_nearest(rgb_np, scale)
        self._rgb_handle.image = rgb_np

      depth_image = None
      provider = getattr(self, "_depth_image_provider", None)
      if self._has_depth and self._depth_handle is not None:
        if provider is not None:
          depth_image = provider(env_idx)
        elif data.depth is not None:
          depth_image = data.depth[env_idx]

      if depth_image is not None and self._depth_handle is not None:
        if hasattr(depth_image, "detach"):
          depth_np = depth_image.detach().squeeze().cpu().numpy()
        else:
          depth_np = np.asarray(depth_image).squeeze()

        depth_scale = max(self._depth_scale_slider.value, 0.01)
        depth_normalized = np.clip(depth_np / depth_scale, 0.0, 1.0)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        if self._needs_upsampling:
          scale = self._display_height // depth_uint8.shape[0]
          depth_uint8 = self._upsample_nearest(depth_uint8, scale)

        self._depth_handle.image = np.repeat(depth_uint8[:, :, np.newaxis], 3, axis=-1)

      if scene_offset is None:
        scene_offset = np.zeros(3)
      self._update_frustum(sim_data, env_idx, scene_offset)

    ViserCameraViewer.update = _camera_viewer_update

  def _setup_controls(self: Any) -> None:
    camera_sensors = [
      sensor
      for sensor in self.env.unwrapped.scene.sensors.values()
      if isinstance(sensor, CameraSensor)
    ]
    if not camera_sensors:
      self.camera_viewers = []
      return

    self.camera_viewers = [
      ViserCameraViewer(
        self.server,
        sensor,
        self.mj_model,
        depth_image_provider=_make_observation_depth_provider(self.env, sensor),
      )
      for sensor in camera_sensors
    ]

  ViserCameraOverlays.setup_controls = _setup_controls
  ViserCameraOverlays._hlip_depth_observation_overlay_installed = True
