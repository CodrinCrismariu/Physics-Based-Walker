"""Unitree G1 LIP walking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.lip import mdp
from mjlab.tasks.lip.mdp import LIPCommandCfg
from mjlab.tasks.lip.lip_env_cfg import make_lip_env_cfg


def unitree_g1_lip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain LIP walking configuration."""
  cfg = make_lip_env_cfg()

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
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  cfg.viewer.body_name = "torso_link"

  # LIP command configuration for G1.
  lip_cmd = cfg.commands["lip"]
  assert isinstance(lip_cmd, LIPCommandCfg)
  lip_cmd.com_height = 0.75
  lip_cmd.foot_separation = 0.20
  lip_cmd.step_duration = 0.4
  lip_cmd.swing_height = 0.08
  lip_cmd.feet_site_names = ("left_foot", "right_foot")

  # Action scale: lower body (RL-controlled) gets a per-robot scale.
  # Filter G1_ACTION_SCALE to only include RL-controlled actuators.
  import re
  rl_body_patterns = (r".*_hip_.*", r".*_knee_.*", r".*_ankle_.*", r"waist_roll_.*", r"waist_pitch_.*")
  lower_body_scale = {
    k: v for k, v in G1_ACTION_SCALE.items()
    if any(re.fullmatch(p, k) for p in rl_body_patterns)
  }
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = lower_body_scale

  # Per-robot event config.
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Per-robot reward config: set site_names and body_names.
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["track_swing_trajectory"].params["asset_cfg"].site_names = site_names
  cfg.rewards["track_swing_trajectory_z"].params["asset_cfg"].site_names = site_names
  cfg.rewards["track_footstep_position"].params["asset_cfg"].site_names = site_names
  cfg.rewards["swing_foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_slip"].params["asset_cfg"].site_names = site_names

  # Per-robot observation config: set site_names for foot_height.
  cfg.observations["policy"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # Self-collision reward.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg

