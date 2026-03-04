"""Unitree G1 HLIP + CLF walking environment configurations."""

import re

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from pure_mjlab_code import mdp
from pure_mjlab_code.hlip_env_cfg import make_hlip_env_cfg


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
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  # ── Viewer ─────────────────────────────────────────────────────────
  cfg.viewer.body_name = "torso_link"

  # ── Action scale ───────────────────────────────────────────────────
  # Only use scales for RL-controlled actuators (legs only).
  rl_body_patterns = (
    r".*_hip_.*", r".*_knee_.*", r".*_ankle_.*",
  )
  lower_body_scale = {
    k: v for k, v in G1_ACTION_SCALE.items()
    if any(re.fullmatch(p, k) for p in rl_body_patterns)
  }
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = lower_body_scale

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
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
