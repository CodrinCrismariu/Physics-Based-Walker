"""Unitree G1 HLIP + CLF walking environment configurations."""

import re
from dataclasses import replace

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg, GridPatternCfg, ObjRef
from mjlab.terrains import BoxSteppingStonesTerrainCfg, BoxPyramidStairsTerrainCfg, BoxInvertedPyramidStairsTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains import TerrainImporterCfg
from hlip_clf_g1 import mdp
from hlip_clf_g1.hlip_env_cfg import make_hlip_env_cfg


def unitree_g1_hlip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat-terrain HLIP + CLF walking configuration."""
  cfg = make_hlip_env_cfg()

  # ── Scene ──────────────────────────────────────────────────────────
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  site_names = ("left_foot", "right_foot")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  heightmap_cfg = RayCastSensorCfg(
    name="heightmap",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    pattern=GridPatternCfg(size=(1.5, 1), resolution=0.1),
    ray_alignment="yaw",
    max_distance=2.0,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(0, 1, 0, 0.8),
      miss_color=(1, 0, 0, 0.4),
      show_rays=False,
    ),
    include_geom_groups=(0,),
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, heightmap_cfg)

  # ── Viewer ─────────────────────────────────────────────────────────
  cfg.viewer.body_name = "torso_link"

  # ── Action scale ───────────────────────────────────────────────────
  # Only use scales for RL-controlled actuators (legs only).
  rl_body_patterns = (
    r".*_hip_.*",
    r".*_knee_.*",
    r".*_ankle_.*",
  )
  # ── Command ────────────────────────────────────────────────────────
  cfg.commands["hlip"].z_sw_max_range = (0.1, 0.3)
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    k: v for k, v in G1_ACTION_SCALE.items()
    if any(re.fullmatch(p, k) for p in rl_body_patterns)
  }

  # ── Per-robot event config ─────────────────────────────────────────
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # ── Per-robot reward config ────────────────────────────────────────
  # Self-collision reward.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # ── Play mode overrides ───────────────────────────────────────────
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


# ---------------------------------------------------------------------------
# Stepping-stone terrain
# ---------------------------------------------------------------------------

STEPPING_STONE_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "stepping_stones": BoxSteppingStonesTerrainCfg(
      proportion=1.0,
      stone_size_range=(0.4, 0.8),
      stone_distance_range=(0.2, 0.5),
      stone_height=0.2,
      stone_height_variation=0.1,
      floor_depth=2.0,
      platform_width=2.0,
    ),
  },
  add_lights=True,
)


def unitree_g1_hlip_stepping_stone_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 stepping-stone terrain HLIP + CLF walking configuration."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  # ── Terrain ──────────────────────────────────────────────────────────
  assert cfg.scene.terrain is not None
  cfg.scene.terrain = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=replace(STEPPING_STONE_TERRAINS_CFG),
    max_init_terrain_level=5,
  )
  cfg.scene.terrain.terrain_generator.curriculum = False

  # Increase contact limits for complex terrain geometry.
  cfg.sim.nconmax = 120
  cfg.sim.njmax = 600

  # ── Curriculum ───────────────────────────────────────────────────────
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_hlip,
    params={"command_name": "hlip"},
  )

  # ── Play-mode overrides ─────────────────────────────────────────────
  if play:
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0
    cfg.curriculum.pop("terrain_levels", None)

  return cfg

# ---------------------------------------------------------------------------
# Stairs terrain
# ---------------------------------------------------------------------------

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "pyramid_stairs": BoxPyramidStairsTerrainCfg(
      proportion=0.5,
      step_height_range=(0.1, 0.15),
      step_width=0.4,
      platform_width=2.0,
    ),
    "inverted_pyramid_stairs": BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.5,
      step_height_range=(0.1, 0.15),
      step_width=0.4,
      platform_width=2.0,
    ),
  },
  add_lights=True,
)


def unitree_g1_hlip_stairs_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 stairs terrain HLIP + CLF walking configuration."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  # ── Terrain ──────────────────────────────────────────────────────────
  assert cfg.scene.terrain is not None
  cfg.scene.terrain = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=replace(STAIRS_TERRAINS_CFG),
    max_init_terrain_level=5,
  )
  cfg.scene.terrain.terrain_generator.curriculum = True

  # Stairs environment uses adaptive heightmap stepping, disable flat height randomisation
  cfg.commands["hlip"].z_sw_max_range = None

  # Increase contact limits for complex terrain geometry.
  cfg.sim.nconmax = 120
  cfg.sim.njmax = 600

  # ── Curriculum ───────────────────────────────────────────────────────
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_hlip,
    params={"command_name": "hlip"},
  )

  # ── Play-mode overrides ─────────────────────────────────────────────
  if play:
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0
    cfg.curriculum.pop("terrain_levels", None)

  return cfg
