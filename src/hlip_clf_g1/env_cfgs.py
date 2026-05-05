"""Unitree G1 HLIP + CLF walking environment configurations."""

import math
import re
from dataclasses import dataclass, replace

import torch
from mjlab.actuator import (
  BuiltinPositionActuator,
  DelayedActuator,
  DelayedActuatorCfg,
  IdealPdActuator,
  XmlPositionActuator,
)
from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.terrains import BoxSteppingStonesTerrainCfg, BoxPyramidStairsTerrainCfg, BoxInvertedPyramidStairsTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains import TerrainImporterCfg
from hlip_clf_g1 import mdp
from hlip_clf_g1.custom_terrains import TwoPlatformSteppingCorridorTerrainCfg
from hlip_clf_g1.g1_no_hands import get_g1_no_hands_robot_cfg
from hlip_clf_g1.hlip_env_cfg import make_hlip_env_cfg, make_hlip_distillation_env_cfg


HEAD_CAMERA_WIDTH = 32
HEAD_CAMERA_HEIGHT = 24
G1_MOTOR_DELAY_MAX_S = 0.020


class EpisodeRandomDelayedActuator(DelayedActuator):
  """Delayed actuator with a fixed per-episode sampled lag."""

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)

    delay_buffers = tuple(self._delay_buffers.values())
    if not delay_buffers:
      return

    batch_size = delay_buffers[0].batch_size
    device = delay_buffers[0].device
    if env_ids is None:
      num_envs = batch_size
      target_env_ids = None
    elif isinstance(env_ids, slice):
      target_env_ids = torch.arange(batch_size, device=device, dtype=torch.long)[env_ids]
      num_envs = int(target_env_ids.numel())
    else:
      target_env_ids = env_ids.to(device=device, dtype=torch.long)
      num_envs = int(target_env_ids.numel())

    if num_envs <= 0:
      return

    lags = torch.randint(
      int(self.cfg.delay_min_lag),
      int(self.cfg.delay_max_lag) + 1,
      (num_envs,),
      device=device,
      dtype=torch.long,
    )
    self.set_lags(lags, target_env_ids)


@dataclass(kw_only=True)
class EpisodeRandomDelayedActuatorCfg(DelayedActuatorCfg):
  """Delayed actuator config that samples lag on every episode reset."""

  def build(self, entity, target_ids: list[int], target_names: list[str]):
    base_actuator = self.base_cfg.build(entity, target_ids, target_names)
    return EpisodeRandomDelayedActuator(self, base_actuator)


def _apply_g1_motor_delay(
  robot_cfg,
  physics_dt: float,
  max_delay_s: float = G1_MOTOR_DELAY_MAX_S,
) -> None:
  """Wrap G1 position actuators with per-episode motor command delay."""
  max_lag = max(0, int(round(max_delay_s / physics_dt)))

  robot_cfg.articulation.actuators = tuple(
    (
      actuator_cfg
      if isinstance(actuator_cfg, DelayedActuatorCfg)
      else EpisodeRandomDelayedActuatorCfg(
        base_cfg=actuator_cfg,
        delay_target="position",
        delay_min_lag=0,
        delay_max_lag=max_lag,
        delay_hold_prob=1.0,
      )
    )
    for actuator_cfg in robot_cfg.articulation.actuators
  )


@requires_model_fields("actuator_forcerange")
def _g1_effort_limits_with_delayed_actuators(
  env,
  env_ids: torch.Tensor | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Randomize effort limits for plain or delayed position actuators."""
  asset = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  else:
    actuators = asset.actuators[asset_cfg.actuator_ids]
  if not isinstance(actuators, list):
    actuators = [actuators]

  low, high = effort_limit_range
  for actuator in actuators:
    base_actuator = (
      actuator.base_actuator if isinstance(actuator, DelayedActuator) else actuator
    )
    ctrl_ids = base_actuator.global_ctrl_ids
    effort_samples = torch.rand(
      (len(env_ids), len(ctrl_ids)),
      device=env.device,
      dtype=torch.float32,
    ) * (high - low) + low

    if isinstance(base_actuator, (BuiltinPositionActuator, XmlPositionActuator)):
      default_forcerange = env.sim.get_default_field("actuator_forcerange")
      env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
        default_forcerange[ctrl_ids, 0] * effort_samples
      )
      env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
        default_forcerange[ctrl_ids, 1] * effort_samples
      )
    elif isinstance(base_actuator, IdealPdActuator):
      assert base_actuator.default_force_limit is not None
      base_actuator.set_effort_limit(
        env_ids,
        effort_limit=base_actuator.default_force_limit[env_ids] * effort_samples,
      )
    else:
      raise TypeError(
        "G1 effort-limit randomization supports position and ideal-PD actuators, "
        f"got {type(base_actuator).__name__}."
      )


def _apply_g1_joint_pd_gains(robot_cfg) -> None:
  """Override selected G1 actuator PD gains (waist, arms, wrists)."""
  actuator_cfgs = tuple(robot_cfg.articulation.actuators)

  def _sig(names: tuple[str, ...]) -> tuple[frozenset[str], int]:
    return frozenset(names), len(names)

  desired_pd: dict[tuple[frozenset[str], int], tuple[float, float]] = {
    _sig(("waist_yaw_joint",)): (250.0, 5.0),
    _sig(("waist_pitch_joint", "waist_roll_joint")): (250.0, 5.0),
    _sig((".*_elbow_joint", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint")): (80.0, 3.0),
    _sig((".*_wrist_roll_joint",)): (40.0, 1.5),
    _sig((".*_wrist_pitch_joint", ".*_wrist_yaw_joint")): (40.0, 1.5),
  }

  legacy_sig_to_split_targets: dict[tuple[frozenset[str], int], tuple[tuple[str, ...], ...]] = {
    _sig((".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint")): (
      (".*_hip_pitch_joint", ".*_hip_yaw_joint"),
      ("waist_yaw_joint",),
    ),
    _sig((".*_hip_roll_joint", ".*_knee_joint")): (
      (".*_hip_roll_joint",),
      (".*_knee_joint",),
    ),
    _sig((".*_elbow_joint", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_wrist_roll_joint")): (
      (".*_elbow_joint", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"),
      (".*_wrist_roll_joint",),
    ),
  }

  new_actuator_cfgs = []
  for actuator_cfg in actuator_cfgs:
    names = tuple(getattr(actuator_cfg, "target_names_expr", ()))
    sig = _sig(names)

    if sig in legacy_sig_to_split_targets:
      for split_targets in legacy_sig_to_split_targets[sig]:
        split_sig = _sig(split_targets)
        pd_override = desired_pd.get(split_sig)
        if pd_override is None:
          # Preserve original actuator gains for non-overridden groups.
          new_actuator_cfgs.append(
            replace(
              actuator_cfg,
              target_names_expr=split_targets,
            )
          )
        else:
          kp, kd = pd_override
          new_actuator_cfgs.append(
            replace(
              actuator_cfg,
              target_names_expr=split_targets,
              stiffness=kp,
              damping=kd,
            )
          )
      continue

    if sig in desired_pd:
      kp, kd = desired_pd[sig]
      new_actuator_cfgs.append(
        replace(
          actuator_cfg,
          stiffness=kp,
          damping=kd,
        )
      )
      continue

    new_actuator_cfgs.append(actuator_cfg)

  robot_cfg.articulation.actuators = tuple(new_actuator_cfgs)

def _make_head_camera_sensor() -> CameraSensorCfg:
  return CameraSensorCfg(
    name="head_camera",
    parent_body="robot/torso_link",
    pos=(0.15, 0.0, 0.3),#(0.06, 0.0, 0.45),
    quat=(0, -0.1736483, 0, 0.9848077),#(-0.6927357, -0.1405364, 0.1404245, 0.6932876),#(-0.6589899, -0.255809, 0.2556054, 0.6595149),
    width=HEAD_CAMERA_WIDTH,
    height=HEAD_CAMERA_HEIGHT,
    fovy=55.2,
    data_types=("rgb", "depth"),
  )


def _add_head_camera_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  head_camera = _make_head_camera_sensor()
  cfg.scene.sensors = (*cfg.scene.sensors, head_camera)


def _add_head_camera_dr_events(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add startup domain randomization for the head camera extrinsics."""
  camera_asset_cfg = SceneEntityCfg("robot", camera_names="head_camera")
  angle = math.radians(2.5)

  cfg.events["head_camera_pos_dr"] = EventTermCfg(
    mode="reset",
    func=mdp.dr.cam_pos,
    params={
      "asset_cfg": camera_asset_cfg,
      "operation": "add",
      "ranges": {
        0: (-0.025, 0.025),
        1: (-0.025, 0.025),
        2: (-0.025, 0.025),
      },
    },
  )
  cfg.events["head_camera_quat_dr"] = EventTermCfg(
    mode="reset",
    func=mdp.dr.cam_quat,
    params={
      "asset_cfg": camera_asset_cfg,
      "roll_range": (-angle, angle),
      "pitch_range": (-angle, angle),
      "yaw_range": (-angle, angle),
    },
  )


def _make_play_mode_depth_deterministic(cfg: ManagerBasedRlEnvCfg) -> None:
  """Disable depth/image stochasticity in play mode for deterministic behavior."""
  cfg.events.pop("head_camera_pos_dr", None)
  cfg.events.pop("head_camera_quat_dr", None)

  if "head_camera_depth" not in cfg.observations:
    return

  # depth_group = cfg.observations["head_camera_depth"]
  # depth_group.enable_corruption = False

  # depth_term = depth_group.terms.get("depth")
  # if depth_term is not None and depth_term.params is not None:
  #   depth_term.params["depth_noise_scale"] = 0.0
  #   depth_term.params["close_depth_bleed_radius"] = 0
  #   depth_term.params["close_depth_bleed_prob"] = 0.0


def _add_g1_hardware_dr_events(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add conservative G1 motor and joint dynamics randomization."""
  actuator_asset_cfg = SceneEntityCfg("robot")
  joint_asset_cfg = SceneEntityCfg("robot")

  cfg.events["motor_pd_gains"] = EventTermCfg(
    mode="startup",
    func=mdp.dr.pd_gains,
    params={
      "asset_cfg": actuator_asset_cfg,
      "operation": "scale",
      "kp_range": (0.85, 1.15),
      "kd_range": (0.75, 1.25),
    },
  )
  cfg.events["motor_effort_limits"] = EventTermCfg(
    mode="startup",
    func=_g1_effort_limits_with_delayed_actuators,
    params={
      "asset_cfg": actuator_asset_cfg,
      "effort_limit_range": (0.85, 1.1),
    },
  )
  cfg.events["joint_damping"] = EventTermCfg(
    mode="startup",
    func=mdp.dr.joint_damping,
    params={
      "asset_cfg": joint_asset_cfg,
      "operation": "scale",
      "ranges": (0.75, 1.25),
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=mdp.dr.joint_friction,
    params={
      "asset_cfg": joint_asset_cfg,
      "operation": "add",
      "ranges": (0.0, 0.04),
    },
  )


def _configure_generated_terrain(
  cfg: ManagerBasedRlEnvCfg,
  terrain_cfg: TerrainGeneratorCfg,
  max_init_terrain_level: int,
  curriculum: bool,
) -> None:
  assert cfg.scene.terrain is not None
  cfg.scene.terrain = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=replace(terrain_cfg),
    max_init_terrain_level=max_init_terrain_level,
  )
  if cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = curriculum


def _apply_complex_terrain_contact_limits(cfg: ManagerBasedRlEnvCfg) -> None:
  """Increase contact limits for irregular terrain geometry."""
  cfg.sim.nconmax = 120
  cfg.sim.njmax = 600


def _disable_flat_swing_height_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Terrain variants use adaptive stepping and should not randomize flat swing height."""
  cfg.commands["hlip"].z_sw_max_range = None


def _apply_stairs_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_hlip,
    params={"command_name": "hlip"},
  )


def _apply_standard_stairs_velocity_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add velocity command staging for the standard (non-distillation) stairs task."""
  cfg.curriculum["stairs_velocity_commands"] = CurriculumTermCfg(
    func=mdp.commands_hlip,
    params={
      "command_name": "hlip",
      "velocity_stages": [
        {
          "step": 0,
          "lin_vel_x": (0.0, 0.25),
          "lin_vel_y": (0.0, 0.0),
          "ang_vel_z": (0.0, 0.0),
        },
        {
          "step": 10_000,
          "lin_vel_x": (0.0, 0.4),
          "lin_vel_y": (-0.05, 0.05),
          "ang_vel_z": (-0.1, 0.1),
        },
        {
          "step": 30_000,
          "lin_vel_x": (-0.2, 0.5),
          "lin_vel_y": (-0.1, 0.1),
          "ang_vel_z": (-0.2, 0.2),
        },
        {
          "step": 60_000,
          "lin_vel_x": (-0.6, 0.6),
          "lin_vel_y": (-0.2, 0.2),
          "ang_vel_z": (-0.4, 0.4),
        },
      ],
    },
  )


def _apply_stairs_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  if cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("stairs_velocity_commands", None)


def _apply_two_platform_corridor_reset_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep reset jitter on the start platform for corridor terrains."""
  reset_base = cfg.events.get("reset_base")
  if reset_base is None or reset_base.params is None:
    return

  reset_base.params["pose_range"] = {
    "x": (-0.3, 0.3),
    "y": (-0.25, 0.25),
    # Corridor advances along +x, so keep a fixed heading toward platform B.
    "yaw": (0.0, 0.0),
  }


def _apply_two_platform_corridor_command_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use corridor command sampling with yaw-hold feedback."""
  command_cfg = cfg.commands["hlip"]

  # Sample forward velocity command for this terrain/task.
  command_cfg.rel_standing_envs = 0.0
  command_cfg.ranges.lin_vel_x = (0.0, 0.6)
  command_cfg.ranges.lin_vel_y = (0.0, 0.0)
  command_cfg.ranges.ang_vel_z = (0.0, 0.0)
  command_cfg.fixed_velocity_command_enabled = False

  # Keep the torso heading aligned with +x corridor direction.
  command_cfg.yaw_hold_enabled = True
  command_cfg.yaw_hold_target = 0.0
  command_cfg.yaw_hold_kp = 1.8
  command_cfg.yaw_hold_max_ang_vel = 0.8

  # Disable viewer manual sliders for this corridor task.
  command_cfg.manual_control = False


def _apply_distillation_push_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use gentler interval pushes for distillation robustness training."""
  push_event = cfg.events.get("push_robot")
  if push_event is None or push_event.params is None:
    return

  velocity_range = push_event.params.get("velocity_range")
  if not isinstance(velocity_range, dict):
    return

  velocity_range.update(
    {
      "x": (-0.2, 0.2),
      "y": (-0.2, 0.2),
      "roll": (-0.08, 0.08),
      "pitch": (-0.08, 0.08),
      "yaw": (-0.08, 0.08),
    }
  )


def _apply_distillation_task_overrides(
  cfg: ManagerBasedRlEnvCfg,
  play: bool,
) -> None:
  distillation_cfg = make_hlip_distillation_env_cfg()
  cfg.observations = distillation_cfg.observations
  cfg.commands = distillation_cfg.commands
  cfg.terminations = distillation_cfg.terminations
  _apply_distillation_push_overrides(cfg)

  _add_head_camera_sensor(cfg)
  if play:
    cfg.commands["hlip"].manual_control = True
    # cfg.observations["student_vec"].enable_corruption = False
    # _make_play_mode_depth_deterministic(cfg)
  else:
    _add_head_camera_dr_events(cfg)

def unitree_g1_hlip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat-terrain HLIP + CLF walking configuration."""
  cfg = make_hlip_env_cfg()

  # ── Scene ──────────────────────────────────────────────────────────
  robot_cfg = get_g1_no_hands_robot_cfg()
  _apply_g1_joint_pd_gains(robot_cfg)
  _apply_g1_motor_delay(robot_cfg, physics_dt=cfg.sim.mujoco.timestep)
  cfg.scene.entities = {"robot": robot_cfg}

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
    pattern=GridPatternCfg(size=(4, 1.5), resolution=0.025),
    ray_alignment="yaw",
    max_distance=2.0,
    debug_vis=False,
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
  # Keep lower-body joints policy-controlled. Upper-body joints stay locked via
  # zero `joint_pos` scale, while waist pitch/roll are handled by
  # `upright_waist` using the raw `joint_pos` waist action channels.
  rl_body_patterns = (
    r".*_hip_.*",
    r".*_knee_.*",
    r".*_ankle_.*",
  )
  # ── Command ────────────────────────────────────────────────────────
  cfg.commands["hlip"].z_sw_max_range = (0.1, 0.3)
  # Teacher policy uses time-based stance switching and replanning only.
  # cfg.commands["hlip"].touchdown_switch_enabled = False
  # cfg.commands["hlip"].phase_end_stance_flip_only = True
  # cfg.commands["hlip"].mpc_contact_recovery_enabled = False
  # cfg.commands["hlip"].mpc_replan_wait_for_stance_contact = False
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    k: (v if any(re.fullmatch(p, k) for p in rl_body_patterns) else 0.0)
    for k, v in G1_ACTION_SCALE.items()
  }

  # ── Per-robot event config ─────────────────────────────────────────
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  _add_g1_hardware_dr_events(cfg)

  # ── Per-robot reward config ────────────────────────────────────────
  # Self-collision reward.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # ── Play mode overrides ───────────────────────────────────────────
  if play:
    cfg.commands["hlip"].manual_control = True
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=mdp.randomize_terrain,
      mode="reset",
      params={},
    )

  return cfg


def unitree_g1_hlip_random_step_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create flat-terrain HLIP config with random step-time and velocity sampling."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  command_cfg = cfg.commands["hlip"]
  command_cfg.mpc_enabled = False
  command_cfg.resample_velocity_on_step = True
  command_cfg.resample_velocity_on_step_probability = 0.4
  command_cfg.rel_standing_envs = 0.0

  if play:
    command_cfg.manual_control = False

  return cfg


# ---------------------------------------------------------------------------
# Stepping-stone terrain
# ---------------------------------------------------------------------------

SIMPLE_STEPPING_STONE_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(6.0, 6.0),
  border_width=20.0,
  num_rows=8,
  num_cols=8,
  curriculum=False,
  sub_terrains={
    # "pyramid_stairs": BoxPyramidStairsTerrainCfg(
    #   proportion=0.5,
    #   step_height_range=(0.05, 0.2),
    #   step_width=0.4,
    #   platform_width=1,ht
    # ),
    # "inverted_pyramid_stairs": BoxInvertedPyramidStairsTerrainCfg(
    #   proportion=0.5,
    #   step_height_range=(0.05, 0.2),
    #   step_width=0.4,
    #   platform_width=1,
    # ),
    "stepping_stones": BoxSteppingStonesTerrainCfg(
      proportion=1.0,
      stone_size_range=(0.33, 0.36),
      stone_distance_range=(0.18, 0.22),
      stone_height=0.1,
      stone_height_variation=0.05,
      floor_depth=0.35,
      platform_width=1.2,
    ),
  },
  add_lights=True,
)


TWO_PLATFORM_STEPPING_CORRIDOR_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(12.0, 4.0),
  border_width=20.0,
  num_rows=8,
  num_cols=8,
  curriculum=False,
  sub_terrains={
    "two_platform_corridor": TwoPlatformSteppingCorridorTerrainCfg(
      proportion=1.0,
      platform_height=0.2,
      floor_depth=0.1,
      border_width=0.2,
      platform_length_ratio=0.18,
      platform_width_ratio=0.85,
      platform_edge_margin_ratio=0.03,
      corridor_width_ratio=0.4,
      stone_length_range=(0.3, 0.35),#(0.29, 0.35),
      stone_width_range=(0.3, 0.35),#(0.29, 0.35),
      stone_gap_range=(0.0, 0.35),
      stone_height_variation=0.04,
      stone_size_variation=0.04,
      lateral_displacement_range=0.10,
      zigzag_offset_range=(0.1, 0.21),
      pair_probability=0.35,
      pair_lateral_spacing_range=(0.30, 0.55),
      pair_width_scale_range=(0.72, 0.92),
      split_pair_probability=0.7,
      split_pair_center_jitter=0.05,
    ),
  },
  add_lights=True,
)


def unitree_g1_hlip_simple_stepping_stone_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 simple stepping-stone terrain HLIP + CLF walking configuration."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  _configure_generated_terrain(
    cfg,
    terrain_cfg=SIMPLE_STEPPING_STONE_TERRAINS_CFG,
    max_init_terrain_level=0,
    curriculum=False,
  )
  _disable_flat_swing_height_randomization(cfg)
  _apply_complex_terrain_contact_limits(cfg)

  return cfg


def unitree_g1_hlip_two_platform_stepping_corridor_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 elongated two-platform stepping-corridor terrain configuration."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  _configure_generated_terrain(
    cfg,
    terrain_cfg=TWO_PLATFORM_STEPPING_CORRIDOR_TERRAINS_CFG,
    max_init_terrain_level=0,
    curriculum=False,
  )
  _disable_flat_swing_height_randomization(cfg)
  _apply_complex_terrain_contact_limits(cfg)
  _apply_two_platform_corridor_reset_overrides(cfg)
  _apply_two_platform_corridor_command_overrides(cfg)

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
      platform_width=1,
    ),
    "inverted_pyramid_stairs": BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.5,
      step_height_range=(0.1, 0.15),
      step_width=0.4,
      platform_width=1,
    ),
  },
  add_lights=True,
)


def unitree_g1_hlip_stairs_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 stairs terrain HLIP + CLF walking configuration."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  _configure_generated_terrain(
    cfg,
    terrain_cfg=STAIRS_TERRAINS_CFG,
    max_init_terrain_level=5,
    curriculum=True,
  )
  _disable_flat_swing_height_randomization(cfg)
  _apply_complex_terrain_contact_limits(cfg)
  _apply_stairs_curriculum(cfg)
  _apply_standard_stairs_velocity_curriculum(cfg)

  if play:
    _apply_stairs_play_overrides(cfg)

  return cfg

def unitree_g1_hlip_distillation_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 flat-terrain HLIP + CLF distillation configuration."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  _apply_distillation_task_overrides(cfg, play=play)

  return cfg

def unitree_g1_hlip_distillation_stepping_stone_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 stepping-stone HLIP + CLF distillation configuration."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  _apply_distillation_task_overrides(cfg, play=play)
  _configure_generated_terrain(
    cfg,
    terrain_cfg=SIMPLE_STEPPING_STONE_TERRAINS_CFG,
    max_init_terrain_level=0,
    curriculum=False,
  )
  _disable_flat_swing_height_randomization(cfg)
  _apply_complex_terrain_contact_limits(cfg)

  return cfg


def unitree_g1_hlip_distillation_two_platform_stepping_corridor_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 distillation config on elongated two-platform corridor terrain."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  _apply_distillation_task_overrides(cfg, play=play)
  _configure_generated_terrain(
    cfg,
    terrain_cfg=TWO_PLATFORM_STEPPING_CORRIDOR_TERRAINS_CFG,
    max_init_terrain_level=0,
    curriculum=False,
  )
  _disable_flat_swing_height_randomization(cfg)
  _apply_complex_terrain_contact_limits(cfg)
  _apply_two_platform_corridor_reset_overrides(cfg)
  _apply_two_platform_corridor_command_overrides(cfg)

  # MDN distillation student observes only linear command; teacher keeps full
  # [vx, vy, yaw_rate] command in privileged observations.
  cfg.observations["student_vec"].terms["velocity_commands"].func = (  # type: ignore[index]
    mdp.hlip_velocity_command_linear_only
  )

  return cfg

def unitree_g1_distillation_hlip_stairs_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 stairs HLIP + CLF distillation configuration."""
  cfg = unitree_g1_hlip_env_cfg(play=play)

  _apply_distillation_task_overrides(cfg, play=play)
  _configure_generated_terrain(
    cfg,
    terrain_cfg=STAIRS_TERRAINS_CFG,
    max_init_terrain_level=5,
    curriculum=True,
  )
  _disable_flat_swing_height_randomization(cfg)
  _apply_complex_terrain_contact_limits(cfg)
  _apply_stairs_curriculum(cfg)

  if play:
    _apply_stairs_play_overrides(cfg)

  return cfg
