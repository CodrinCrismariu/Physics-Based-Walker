"""Joint lock action: kinematically lock joints to their default position.

This action term consumes zero policy actions and directly writes qpos/qvel
each step, making the locked joints perfectly rigid regardless of PD gains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class JointLockActionCfg(ActionTermCfg):
  """Configuration for locking joints to their default position."""

  joint_names: tuple[str, ...] | list[str]
  """Joint name patterns to lock (regex patterns)."""

  def build(self, env: "ManagerBasedRlEnv") -> "JointLockAction":
    return JointLockAction(self, env)


class JointLockAction(ActionTerm):
  """Lock joints by writing default qpos and zero qvel every step.

  This term has ``action_dim == 0`` so it does not consume any slice of
  the policy output.  In ``apply_actions`` it writes the default joint
  positions and zero velocities directly into the simulation state.
  """

  cfg: JointLockActionCfg

  def __init__(self, cfg: JointLockActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    entity: Entity = self._entity

    # Resolve joint IDs from patterns.
    joint_ids, joint_names = entity.find_joints(cfg.joint_names)
    self._lock_joint_ids = torch.tensor(
      joint_ids, device=self.device, dtype=torch.long
    )
    self._lock_joint_names = joint_names

    # Store default positions for the locked joints.
    self._default_pos = entity.data.default_joint_pos[
      :, self._lock_joint_ids
    ].clone()

    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    # Nothing to process — no policy actions consumed.
    pass

  def apply_actions(self) -> None:
    # Set PD position target so actuators cooperate (not fight the lock).
    self._entity.set_joint_position_target(
      self._default_pos, joint_ids=self._lock_joint_ids
    )
    # Force qpos/qvel directly to eliminate any drift.
    self._entity.data.write_joint_position(
      self._default_pos, joint_ids=self._lock_joint_ids
    )
    self._entity.data.write_joint_velocity(
      torch.zeros_like(self._default_pos),
      joint_ids=self._lock_joint_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    pass
