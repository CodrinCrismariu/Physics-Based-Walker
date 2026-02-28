"""LIP walking task configuration.

This module provides a factory function to create a base LIP walking task
config. The agent learns to:
  - Track a velocity command (like the velocity task).
  - Follow a smooth swing foot trajectory (sin/cos arc) planned by the LIP
    capture-point heuristic.
  - Maintain proper gait timing and regularisation.

Robot-specific configurations call this factory and customise as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.lip import mdp
from mjlab.tasks.lip.mdp import LIPCommandCfg
from mjlab.tasks.lip.mdp.joint_lock_action import JointLockActionCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_lip_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base LIP walking task configuration.

  The LIP task provides:
    - Velocity command + LIP-based footstep planning + swing trajectory
    - Rewards for tracking velocity, swing foot trajectory, landing accuracy
    - Standard locomotion rewards (orientation, joint limits, action rate, etc.)
    - Observations including the LIP command, swing trajectory, contact info
  """

  ##
  # Observations
  ##

  policy_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "lip_command": ObservationTermCfg(
      func=mdp.lip_command,
      params={"command_name": "lip"},
    ),
    "lip_velocity_command": ObservationTermCfg(
      func=mdp.lip_velocity_command,
      params={"command_name": "lip"},
    ),
    "support_foot_rel_pos": ObservationTermCfg(
      func=mdp.support_foot_rel_pos,
      params={"command_name": "lip"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Per-robot.
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    **policy_terms,
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Actions
  ##

  # Lower body (legs + waist roll/pitch): controlled by RL policy.
  # Upper body + waist yaw: kinematically locked to default.
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", "waist_roll_.*", "waist_pitch_.*"),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    ),
    "upper_body_lock": JointLockActionCfg(
      entity_name="robot",
      joint_names=(".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*", "waist_yaw_.*"),
    ),
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "lip": LIPCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 10.0),
      com_height=0.75,         # Override per-robot.
      step_duration=0.4,
      foot_separation=0.2,    # Override per-robot.
      swing_height=0.08,
      heading_command=False,
      heading_control_stiffness=0.5,
      rel_standing_envs=0.05,
      rel_heading_envs=0.0,
      debug_vis=True,
      ranges=LIPCommandCfg.Ranges(
        lin_vel_x=(-0.3, 0.8),
        lin_vel_y=(-0.3, 0.3),
        ang_vel_z=(-0.5, 0.5),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 5.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (-0.15, 0.15),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "body_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=".*_collision"),
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.randomize_encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    # === Velocity tracking ===
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "lip", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "lip", "std": math.sqrt(0.25)},
    ),
    # === Swing trajectory tracking ===
    "track_swing_trajectory": RewardTermCfg(
      func=mdp.track_swing_trajectory,
      weight=2.0,
      params={
        "command_name": "lip",
        "std": 0.03,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Per-robot.
      },
    ),
    "track_swing_trajectory_z": RewardTermCfg(
      func=mdp.track_swing_trajectory_z,
      weight=1.0,
      params={
        "command_name": "lip",
        "std": 0.02,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Per-robot.
      },
    ),
    # === Footstep placement ===
    "track_footstep_position": RewardTermCfg(
      func=mdp.track_footstep_position,
      weight=1.0,
      params={
        "command_name": "lip",
        "sensor_name": "feet_ground_contact",
        "std": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Per-robot.
      },
    ),
    # === Gait rewards ===
    "support_foot_contact": RewardTermCfg(
      func=mdp.support_foot_contact,
      weight=0.5,
      params={
        "command_name": "lip",
        "sensor_name": "feet_ground_contact",
      },
    ),
    "step_frequency": RewardTermCfg(
      func=mdp.step_frequency,
      weight=0.3,
      params={
        "command_name": "lip",
        "sensor_name": "feet_ground_contact",
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.5,
      params={
        "command_name": "lip",
        "sensor_name": "feet_ground_contact",
        "threshold": 0.4,
      },
    ),
    # === Regularisation ===
    "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.0),
    "upright": RewardTermCfg(func=mdp.upright_reward, weight=1.0),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.05,
      params={
        "command_name": "lip",
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Per-robot.
      },
    ),
    "swing_foot_clearance": RewardTermCfg(
      func=mdp.swing_foot_clearance,
      weight=-1.0,
      params={
        "command_name": "lip",
        "target_height": 0.08,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Per-robot.
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,
      params={"sensor_name": "robot/root_angmom"},
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(50.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_lip,
      params={
        "command_name": "lip",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (-0.3, 0.8),
            "lin_vel_y": (-0.3, 0.3),
            "ang_vel_z": (-0.5, 0.5),
          },
          {
            "step": 5000 * 24,
            "lin_vel_x": (-0.5, 1.2),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-0.8, 0.8),
          },
          {
            "step": 10000 * 24,
            "lin_vel_x": (-1.0, 1.5),
            "lin_vel_y": (-0.8, 0.8),
            "ang_vel_z": (-1.0, 1.0),
          },
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="plane",
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
