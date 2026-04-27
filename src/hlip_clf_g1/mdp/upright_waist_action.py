"""Waist action that keeps torso upright while allowing pitch/roll control.

This term consumes zero new policy dimensions. Instead, it reads the raw
`joint_pos` action channels for `waist_pitch_joint` and `waist_roll_joint`,
then computes waist targets as:
  target_pitch = default_pitch + action_scale_pitch * a_pitch - k_pitch * base_pitch
  target_roll  = default_roll  + action_scale_roll  * a_roll  - k_roll  * base_roll
"""

from __future__ import annotations

import re
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
  """Configuration for upright-compensated waist pitch/roll control."""

  source_action_term: str = "joint_pos"
  """Action term name that provides raw policy actions (default: `joint_pos`)."""

  source_pitch_action_pattern: str = "waist_pitch_joint"
  source_roll_action_pattern: str = "waist_roll_joint"
  """Regex patterns used to find waist channels in source action target names."""

  pitch_joint_names: tuple[str, ...] | list[str] = ("waist_pitch_joint",)
  roll_joint_names: tuple[str, ...] | list[str] = ("waist_roll_joint",)

  action_scale_pitch: float = 0.0
  action_scale_roll: float = 0.0
  """Scale from source raw action to waist target offset [rad]."""

  pitch_offset: float = 0.0
  roll_offset: float = 0.0
  """Constant waist target offsets [rad] around the default pose."""

  upright_pitch_gain: float = 1.0
  upright_roll_gain: float = 1.0
  """Feedback gains against base pitch/roll to keep torso upright."""

  def build(self, env: "ManagerBasedRlEnv") -> "UprightWaistAction":
    return UprightWaistAction(self, env)


class UprightWaistAction(ActionTerm):
  """Apply upright-compensated waist pitch/roll position targets."""

  cfg: UprightWaistActionCfg

  def __init__(self, cfg: UprightWaistActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    entity: Entity = self._entity

    pitch_ids, _ = entity.find_joints(cfg.pitch_joint_names)
    roll_ids, _ = entity.find_joints(cfg.roll_joint_names)
    if len(pitch_ids) == 0 or len(roll_ids) == 0:
      raise ValueError(
        "UprightWaistAction requires both pitch and roll waist joints."
      )

    self._pitch_ids = torch.tensor(pitch_ids, device=self.device, dtype=torch.long)
    self._roll_ids = torch.tensor(roll_ids, device=self.device, dtype=torch.long)

    self._default_pitch = entity.data.default_joint_pos[:, self._pitch_ids].clone()
    self._default_roll = entity.data.default_joint_pos[:, self._roll_ids].clone()

    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    # Lazy-resolved source action channels.
    self._source_term = None
    self._source_pitch_idx: int | None = None
    self._source_roll_idx: int | None = None

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    # No extra action slice is consumed by this term.
    return

  def _resolve_source_indices(self) -> None:
    if self._source_term is not None:
      return

    source_term = self._env.action_manager.get_term(self.cfg.source_action_term)
    if not hasattr(source_term, "target_names") or not hasattr(source_term, "raw_action"):
      raise ValueError(
        f"Action term '{self.cfg.source_action_term}' must expose target_names and raw_action."
      )

    target_names = list(source_term.target_names)

    def _find_idx(pattern: str) -> int:
      for i, name in enumerate(target_names):
        if re.fullmatch(pattern, name):
          return i
      raise ValueError(
        f"Could not find source action channel matching pattern '{pattern}'."
      )

    self._source_term = source_term
    self._source_pitch_idx = _find_idx(self.cfg.source_pitch_action_pattern)
    self._source_roll_idx = _find_idx(self.cfg.source_roll_action_pattern)

  def apply_actions(self) -> None:
    self._resolve_source_indices()
    assert self._source_term is not None
    assert self._source_pitch_idx is not None
    assert self._source_roll_idx is not None

    source_raw = self._source_term.raw_action
    a_pitch = source_raw[:, self._source_pitch_idx].unsqueeze(-1)
    a_roll = source_raw[:, self._source_roll_idx].unsqueeze(-1)

    root_quat = self._entity.data.root_link_quat_w
    roll, pitch, _ = euler_xyz_from_quat(root_quat)
    roll = wrap_to_pi(roll).unsqueeze(-1)
    pitch = wrap_to_pi(pitch).unsqueeze(-1)

    pitch_target = (
      self._default_pitch
      + self.cfg.pitch_offset
      - self.cfg.upright_pitch_gain * pitch
    )
    roll_target = (
      self._default_roll
      + self.cfg.roll_offset
      - self.cfg.upright_roll_gain * roll
    )

    pitch_bias = self._entity.data.encoder_bias[:, self._pitch_ids]
    roll_bias = self._entity.data.encoder_bias[:, self._roll_ids]
    self._entity.set_joint_position_target(
      pitch_target - pitch_bias, joint_ids=self._pitch_ids
    )
    self._entity.set_joint_position_target(
      roll_target - roll_bias, joint_ids=self._roll_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    return
