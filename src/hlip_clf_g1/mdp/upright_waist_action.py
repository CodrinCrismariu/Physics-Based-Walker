"""Action to keep the waist upright.

This action term computes the base roll and pitch, and offsets the default
waist pitch and waist roll to maintain an upright body.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

@dataclass(kw_only=True)
class UprightWaistActionCfg(ActionTermCfg):
  """Configuration for keeping the waist upright."""
  
  pitch_joint_names: tuple[str, ...] | list[str] = ("waist_pitch_.*",)
  roll_joint_names: tuple[str, ...] | list[str] = ("waist_roll_.*",)

  def build(self, env: "ManagerBasedRlEnv") -> "UprightWaistAction":
    return UprightWaistAction(self, env)

class UprightWaistAction(ActionTerm):
  cfg: UprightWaistActionCfg

  def __init__(self, cfg: UprightWaistActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    entity: Entity = self._entity

    pitch_ids, _ = entity.find_joints(cfg.pitch_joint_names)
    roll_ids, _ = entity.find_joints(cfg.roll_joint_names)
    
    self._pitch_ids = torch.tensor(pitch_ids, device=self.device, dtype=torch.long)
    self._roll_ids = torch.tensor(roll_ids, device=self.device, dtype=torch.long)
    
    self._default_pitch = entity.data.default_joint_pos[:, self._pitch_ids].clone()
    self._default_roll = entity.data.default_joint_pos[:, self._roll_ids].clone()

    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    pass

  def apply_actions(self) -> None:
    # Get base orientation
    root_quat = self._entity.data.root_link_quat_w
    roll, pitch, yaw = euler_xyz_from_quat(root_quat)
    
    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)
    
    # We negate the base's pitch and roll to keep the torso upright
    pitch_target = self._default_pitch - pitch.unsqueeze(-1)
    roll_target = self._default_roll - roll.unsqueeze(-1)
    
    # Apply targets
    if len(self._pitch_ids) > 0:
      self._entity.set_joint_position_target(pitch_target, joint_ids=self._pitch_ids)
      self._entity.data.write_joint_position(pitch_target, joint_ids=self._pitch_ids)
      self._entity.data.write_joint_velocity(torch.zeros_like(pitch_target), joint_ids=self._pitch_ids)

    if len(self._roll_ids) > 0:
      self._entity.set_joint_position_target(roll_target, joint_ids=self._roll_ids)
      self._entity.data.write_joint_position(roll_target, joint_ids=self._roll_ids)
      self._entity.data.write_joint_velocity(torch.zeros_like(roll_target), joint_ids=self._roll_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    pass
