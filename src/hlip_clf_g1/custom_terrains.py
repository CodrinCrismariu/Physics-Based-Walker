"""Custom terrain primitives for HLIP locomotion tasks."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_border


@dataclass(kw_only=True)
class TwoPlatformSteppingCorridorTerrainCfg(SubTerrainCfg):
  """Two end-platforms connected by stepping stones along one axis."""

  platform_height: float = 0.1
  """Top height of platforms and stepping stones, in meters."""

  floor_depth: float = 0.35
  """Depth of the pit floor below zero, in meters."""

  border_width: float = 0.2
  """Border width around each patch, in meters."""

  platform_length_ratio: float = 0.18
  """Platform length along traversal axis as fraction of inner patch length."""

  platform_width_ratio: float = 0.85
  """Platform width as fraction of inner patch width."""

  platform_edge_margin_ratio: float = 0.03
  """Distance from border to each platform as fraction of inner patch length."""

  corridor_width_ratio: float = 0.4
  """Stepping corridor width as fraction of inner patch width."""

  stone_length_range: tuple[float, float] = (0.26, 0.42)
  """Min/max stone length along corridor axis, in meters."""

  stone_width_range: tuple[float, float] = (0.22, 0.32)
  """Min/max stone width across corridor axis, in meters."""

  stone_gap_range: tuple[float, float] = (0.18, 0.26)
  """Min/max gap between consecutive stones along the corridor axis, in meters."""

  stone_height_variation: float = 0.04
  """Maximum top-height perturbation for stones, in meters."""

  stone_size_variation: float = 0.06
  """Maximum random perturbation for stone dimensions, in meters."""

  lateral_displacement_range: float = 0.06
  """Maximum lateral displacement of stones from corridor centerline, in meters."""

  zigzag_offset_range: tuple[float, float] = (0.16, 0.3)
  """Alternating left-right center offset range for consecutive steps, in meters."""

  pair_probability: float = 0.35
  """Probability of placing a side-by-side stone pair at a corridor step."""

  pair_lateral_spacing_range: tuple[float, float] = (0.35, 0.6)
  """Center-to-center spacing range for side-by-side stone pairs, in meters."""

  pair_width_scale_range: tuple[float, float] = (0.75, 0.95)
  """Width scaling range applied to each stone in a side-by-side pair."""

  split_pair_probability: float = 0.5
  """Probability that a pair is split across left and right sides of corridor."""

  split_pair_center_jitter: float = 0.05
  """Center jitter for split pairs so they are not perfectly symmetric."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []

    inner_size_x = max(1.0e-6, self.size[0] - 2.0 * self.border_width)
    inner_size_y = max(1.0e-6, self.size[1] - 2.0 * self.border_width)

    platform_len = float(
      np.clip(self.platform_length_ratio * inner_size_x, 1.0, 0.45 * inner_size_x)
    )
    platform_wid = float(
      np.clip(self.platform_width_ratio * inner_size_y, 1.0, inner_size_y)
    )
    edge_margin = float(
      np.clip(self.platform_edge_margin_ratio * inner_size_x, 0.05, 0.35 * inner_size_x)
    )

    center_y = 0.5 * self.size[1]
    platform_a_x = self.border_width + edge_margin + 0.5 * platform_len
    platform_b_x = self.size[0] - self.border_width - edge_margin - 0.5 * platform_len

    # Maintain a non-degenerate corridor even if parameters are aggressive.
    min_corridor_len = 0.8
    if platform_b_x - platform_a_x < platform_len + min_corridor_len:
      center_x = 0.5 * self.size[0]
      half_sep = 0.5 * (platform_len + min_corridor_len)
      platform_a_x = center_x - half_sep
      platform_b_x = center_x + half_sep

    top_z = self.platform_height
    z_center = (top_z - self.floor_depth) / 2.0
    half_height = (top_z + self.floor_depth) / 2.0

    if self.border_width > 0.0:
      border_geoms = make_border(
        body,
        self.size,
        (inner_size_x, inner_size_y),
        top_z + self.floor_depth,
        (0.5 * self.size[0], 0.5 * self.size[1], z_center),
      )
      for geom in border_geoms:
        geometries.append(TerrainGeometry(geom=geom, color=(0.14, 0.14, 0.14, 1.0)))

    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.size[1] / 2.0, floor_h / 2.0),
      pos=(self.size[0] / 2.0, self.size[1] / 2.0, -self.floor_depth - floor_h / 2.0),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=(0.08, 0.08, 0.08, 1.0)))

    for center_x, color in (
      (platform_a_x, (0.28, 0.70, 0.45, 1.0)),
      (platform_b_x, (0.24, 0.64, 0.42, 1.0)),
    ):
      platform_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(platform_len / 2.0, platform_wid / 2.0, half_height),
        pos=(center_x, center_y, z_center),
      )
      geometries.append(TerrainGeometry(geom=platform_geom, color=color))

    corridor_start_x = platform_a_x + 0.5 * platform_len
    corridor_end_x = platform_b_x - 0.5 * platform_len
    corridor_len = max(0.0, corridor_end_x - corridor_start_x)
    corridor_wid = float(
      np.clip(self.corridor_width_ratio * inner_size_y, 0.45, platform_wid)
    )
    corridor_min_y = center_y - corridor_wid / 2.0
    corridor_max_y = center_y + corridor_wid / 2.0

    gap_low, gap_high = self.stone_gap_range
    gap = gap_low + difficulty * (gap_high - gap_low)

    stone_len_low, stone_len_high = self.stone_length_range
    stone_wid_low, stone_wid_high = self.stone_width_range

    avg_len = stone_len_high - difficulty * (stone_len_high - stone_len_low)
    avg_wid = stone_wid_high - difficulty * (stone_wid_high - stone_wid_low)

    size_var = self.stone_size_variation * difficulty
    # Keep noticeable lateral spread even at low difficulty.
    lateral_var = self.lateral_displacement_range * (0.4 + 0.6 * difficulty)
    height_var = self.stone_height_variation * difficulty

    def _clamp_stone_y(y_pos: float, stone_w: float) -> float:
      return float(
        np.clip(
          y_pos,
          corridor_min_y + stone_w / 2.0,
          corridor_max_y - stone_w / 2.0,
        )
      )

    def _add_stone(stone_x: float, stone_y: float, stone_len: float, stone_wid: float) -> None:
      stone_top_z = max(0.02, top_z + rng.uniform(-height_var, height_var))
      stone_total_h = self.floor_depth + stone_top_z
      stone_center_z = -self.floor_depth + stone_total_h / 2.0

      stone_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(stone_len / 2.0, stone_wid / 2.0, stone_total_h / 2.0),
        pos=(stone_x, stone_y, stone_center_z),
      )
      geometries.append(TerrainGeometry(geom=stone_geom, color=(0.20, 0.76, 0.49, 1.0)))

    cursor_x = corridor_start_x + gap
    stone_count = 0
    step_idx = 0
    start_side = -1.0 if rng.random() < 0.5 else 1.0
    while cursor_x + stone_len_low <= corridor_end_x - gap:
      stone_len = float(np.clip(avg_len + rng.uniform(-size_var, size_var), stone_len_low, stone_len_high))
      stone_wid = float(np.clip(avg_wid + rng.uniform(-size_var, size_var), stone_wid_low, stone_wid_high))

      if cursor_x + stone_len > corridor_end_x - gap:
        break

      stone_x = cursor_x + stone_len / 2.0

      # Explicit zig-zag: alternate center offset left/right from one step to the next.
      step_side = start_side if (step_idx % 2 == 0) else -start_side
      zigzag_offset = float(rng.uniform(*self.zigzag_offset_range))
      base_center_y = center_y + step_side * zigzag_offset
      base_center_y += rng.uniform(-lateral_var, lateral_var)
      base_center_y = _clamp_stone_y(base_center_y, stone_wid)

      use_pair = rng.random() < float(np.clip(self.pair_probability, 0.0, 1.0))
      placed_stones_this_step = 0

      if use_pair:
        pair_spacing = float(rng.uniform(*self.pair_lateral_spacing_range))
        pair_width_scale = float(rng.uniform(*self.pair_width_scale_range))
        pair_wid = float(
          np.clip(
            stone_wid * pair_width_scale,
            0.7 * stone_wid_low,
            stone_wid_high,
          )
        )

        max_pair_shift = (corridor_wid - pair_spacing - pair_wid) / 2.0
        if max_pair_shift > 0.01:
          split_pair = rng.random() < float(np.clip(self.split_pair_probability, 0.0, 1.0))
          if split_pair:
            # One stone on each side of corridor (left + right), with slight jitter.
            center_jitter = min(self.split_pair_center_jitter, max_pair_shift)
            pair_center = center_y + rng.uniform(-center_jitter, center_jitter)
          else:
            # Pair follows zig-zag side, so both stones can sit more left or more right.
            pair_center = float(
              np.clip(
                base_center_y,
                center_y - max_pair_shift,
                center_y + max_pair_shift,
              )
            )

          pair_jitter = min(0.08, 0.2 * max_pair_shift)

          left_y = _clamp_stone_y(
            pair_center - 0.5 * pair_spacing + rng.uniform(-pair_jitter, pair_jitter),
            pair_wid,
          )
          right_y = _clamp_stone_y(
            pair_center + 0.5 * pair_spacing + rng.uniform(-pair_jitter, pair_jitter),
            pair_wid,
          )

          _add_stone(stone_x, left_y, stone_len, pair_wid)
          _add_stone(stone_x, right_y, stone_len, pair_wid)
          placed_stones_this_step = 2

      if placed_stones_this_step == 0:
        stone_y = _clamp_stone_y(base_center_y, stone_wid)
        _add_stone(stone_x, stone_y, stone_len, stone_wid)
        placed_stones_this_step = 1

      cursor_x += stone_len + gap
      stone_count += placed_stones_this_step
      step_idx += 1

    if stone_count == 0 and corridor_len > 0.0:
      fallback_len = float(np.clip(0.5 * corridor_len, stone_len_low, stone_len_high))
      fallback_wid = float(np.clip(avg_wid, stone_wid_low, stone_wid_high))
      fallback_x = 0.5 * (corridor_start_x + corridor_end_x)
      fallback_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(fallback_len / 2.0, fallback_wid / 2.0, half_height),
        pos=(fallback_x, center_y, z_center),
      )
      geometries.append(TerrainGeometry(geom=fallback_geom, color=(0.20, 0.76, 0.49, 1.0)))

    origin = np.array([platform_a_x, center_y, top_z])
    return TerrainOutput(origin=origin, geometries=geometries)
