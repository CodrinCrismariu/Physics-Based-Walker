from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import Distillation


class DistillationNanGuard(Distillation):
	"""Classic distillation with finite checks around rollout and update."""

	def __init__(
		self,
		*args,
		nan_guard_enabled: bool = True,
		nan_guard_sanitize_rollout_actions: bool = True,
		**kwargs,
	) -> None:
		super().__init__(*args, **kwargs)
		self.nan_guard_enabled = bool(nan_guard_enabled)
		self.nan_guard_sanitize_rollout_actions = bool(nan_guard_sanitize_rollout_actions)
		self._nan_guard_student_action_replacements = 0
		self._nan_guard_teacher_action_replacements = 0

	@staticmethod
	def _tensor_is_finite(tensor: torch.Tensor) -> bool:
		return bool(torch.isfinite(tensor).all().item())

	@classmethod
	def _obs_is_finite(cls, obs) -> bool:
		return all(cls._tensor_is_finite(tensor) for tensor in obs.values())

	@staticmethod
	def _zero_nonfinite(tensor: torch.Tensor) -> torch.Tensor:
		return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

	def _grads_are_finite(self) -> bool:
		for param in self.student.parameters():
			if param.grad is not None and not self._tensor_is_finite(param.grad):
				return False
		return True

	def act(self, obs) -> torch.Tensor:
		actions = self.student(obs, stochastic_output=True).detach()
		privileged_actions = self.teacher(obs).detach()

		if self.nan_guard_enabled and self.nan_guard_sanitize_rollout_actions:
			if not self._tensor_is_finite(actions):
				actions = self._zero_nonfinite(actions)
				self._nan_guard_student_action_replacements += 1
			if not self._tensor_is_finite(privileged_actions):
				privileged_actions = self._zero_nonfinite(privileged_actions)
				self._nan_guard_teacher_action_replacements += 1

		self.transition.actions = actions
		self.transition.privileged_actions = privileged_actions
		self.transition.observations = obs
		return self.transition.actions

	def update(self) -> dict[str, float]:
		self.num_updates += 1
		mean_behavior_loss = 0.0
		mean_grad_norm = 0.0
		grad_step_cnt = 0
		loss = torch.tensor(0.0, device=self.device)
		cnt = 0
		skipped_input_batches = 0
		skipped_output_batches = 0
		skipped_loss_batches = 0
		skipped_grad_steps = 0

		for _epoch in range(self.num_learning_epochs):
			self.student.reset(hidden_state=self.last_hidden_states[0])
			self.teacher.reset(hidden_state=self.last_hidden_states[1])
			self.student.detach_hidden_state()

			for obs, _, privileged_actions, dones in self.storage.generator():
				should_skip = False
				if self.nan_guard_enabled and (
					not self._obs_is_finite(obs) or not self._tensor_is_finite(privileged_actions)
				):
					skipped_input_batches += 1
					should_skip = True

				if not should_skip:
					actions = self.student(obs)
					if self.nan_guard_enabled and not self._tensor_is_finite(actions):
						skipped_output_batches += 1
						should_skip = True

				if not should_skip:
					behavior_loss = self.loss_fn(actions, privileged_actions)
					if self.nan_guard_enabled and not self._tensor_is_finite(behavior_loss):
						skipped_loss_batches += 1
						should_skip = True

				if not should_skip:
					loss = loss + behavior_loss
					mean_behavior_loss += behavior_loss.item()
					cnt += 1

					if cnt % self.gradient_length == 0:
						self.optimizer.zero_grad()
						loss.backward()
						if self.is_multi_gpu:
							self.reduce_parameters()

						if self.max_grad_norm:
							grad_norm = nn.utils.clip_grad_norm_(
								self.student.parameters(),
								self.max_grad_norm,
							)
							grad_is_finite = self._tensor_is_finite(grad_norm) and self._grads_are_finite()
						else:
							grad_norm = torch.tensor(0.0, device=self.device)
							grad_is_finite = self._grads_are_finite()

						if self.nan_guard_enabled and not grad_is_finite:
							skipped_grad_steps += 1
							self.optimizer.zero_grad()
						else:
							self.optimizer.step()
							mean_grad_norm += float(grad_norm.item())
							grad_step_cnt += 1

						self.student.detach_hidden_state()
						loss = torch.tensor(0.0, device=self.device)

				self.student.reset(dones.view(-1))
				self.teacher.reset(dones.view(-1))
				self.student.detach_hidden_state(dones.view(-1))

		mean_behavior_loss /= max(cnt, 1)
		self.storage.clear()
		self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
		self.student.detach_hidden_state()

		skipped_batches = skipped_input_batches + skipped_output_batches + skipped_loss_batches
		loss_dict = {
			"behavior": mean_behavior_loss,
			"nan_guard_finite_batches": float(cnt),
			"nan_guard_skipped_batches": float(skipped_batches),
			"nan_guard_skipped_input_batches": float(skipped_input_batches),
			"nan_guard_skipped_output_batches": float(skipped_output_batches),
			"nan_guard_skipped_loss_batches": float(skipped_loss_batches),
			"nan_guard_skipped_grad_steps": float(skipped_grad_steps),
			"nan_guard_student_action_replacements": float(self._nan_guard_student_action_replacements),
			"nan_guard_teacher_action_replacements": float(self._nan_guard_teacher_action_replacements),
		}
		if grad_step_cnt > 0:
			loss_dict["nan_guard_grad_norm"] = mean_grad_norm / grad_step_cnt

		self._nan_guard_student_action_replacements = 0
		self._nan_guard_teacher_action_replacements = 0
		return loss_dict


class DistillationMDN(Distillation):
	"""MDN distillation variant with action or teacher-distribution losses.

	This class is additive and does not modify upstream Distillation behavior for
	existing tasks. It expects the student model to implement ``mdn_nll`` and
	``mdn_log_prob`` and optionally ``mdn_entropy``.
	"""

	def __init__(
		self,
		*args,
		mdn_loss_type: str = "action_nll",
		mdn_teacher_num_samples: int = 1,
		mdn_teacher_std_scale: float = 1.0,
		mdn_teacher_sample_std_floor: float = 1.0e-6,
		mdn_entropy_coef: float = 0.0,
		**kwargs,
	) -> None:
		super().__init__(*args, **kwargs)
		if mdn_loss_type not in ("action_nll", "teacher_distribution"):
			raise ValueError(
				"mdn_loss_type must be one of {'action_nll', 'teacher_distribution'}."
			)
		if mdn_teacher_num_samples <= 0:
			raise ValueError("mdn_teacher_num_samples must be > 0.")
		if mdn_teacher_std_scale < 0.0:
			raise ValueError("mdn_teacher_std_scale must be >= 0.")
		if mdn_teacher_sample_std_floor < 0.0:
			raise ValueError("mdn_teacher_sample_std_floor must be >= 0.")

		self.mdn_loss_type = mdn_loss_type
		self.mdn_teacher_num_samples = int(mdn_teacher_num_samples)
		self.mdn_teacher_std_scale = float(mdn_teacher_std_scale)
		self.mdn_teacher_sample_std_floor = float(mdn_teacher_sample_std_floor)
		self.mdn_entropy_coef = float(mdn_entropy_coef)

	def _sample_teacher_distribution(
		self,
		obs,
	) -> tuple[torch.Tensor, torch.Tensor]:
		if not getattr(self.teacher, "stochastic", False):
			raise TypeError(
				"MDN teacher-distribution matching requires a stochastic teacher model."
			)

		with torch.no_grad():
			self.teacher(obs, stochastic_output=True)
			teacher_mean = self.teacher.output_mean.detach()
			teacher_std = self.teacher.output_std.detach() * self.mdn_teacher_std_scale
			if self.mdn_teacher_sample_std_floor > 0.0:
				teacher_std = torch.clamp(
					teacher_std,
					min=self.mdn_teacher_sample_std_floor,
				)

			eps = torch.randn(
				(self.mdn_teacher_num_samples, *teacher_mean.shape),
				device=teacher_mean.device,
				dtype=teacher_mean.dtype,
			)
			teacher_actions = (
				teacher_mean.unsqueeze(0) + teacher_std.unsqueeze(0) * eps
			)

		return teacher_actions, teacher_std

	def _compute_behavior_loss(
		self,
		obs,
		privileged_actions: torch.Tensor,
	) -> tuple[torch.Tensor, float | None]:
		if self.mdn_loss_type == "teacher_distribution":
			if not hasattr(self.student, "mdn_log_prob"):
				raise TypeError(
					"MDN teacher-distribution matching requires student.mdn_log_prob(obs, actions, ...)."
				)
			teacher_actions, teacher_std = self._sample_teacher_distribution(obs)
			return (
				-self.student.mdn_log_prob(obs, teacher_actions).mean(),
				teacher_std.mean().item(),
			)

		if not hasattr(self.student, "mdn_nll"):
			raise TypeError(
				"MDN action NLL requires a student model exposing mdn_nll(obs, actions, ...)."
			)
		return self.student.mdn_nll(obs, privileged_actions, reduction="mean"), None

	def update(self) -> dict[str, float]:
		self.num_updates += 1
		mean_behavior_loss = 0.0
		mean_entropy = 0.0
		mean_teacher_std = 0.0
		teacher_std_cnt = 0
		loss = torch.tensor(0.0, device=self.device)
		cnt = 0

		for _epoch in range(self.num_learning_epochs):
			self.student.reset(hidden_state=self.last_hidden_states[0])
			self.teacher.reset(hidden_state=self.last_hidden_states[1])
			self.student.detach_hidden_state()

			for obs, _, privileged_actions, dones in self.storage.generator():
				behavior_loss, teacher_std = self._compute_behavior_loss(obs, privileged_actions)
				if teacher_std is not None:
					mean_teacher_std += teacher_std
					teacher_std_cnt += 1

				total_loss = behavior_loss
				if self.mdn_entropy_coef != 0.0 and hasattr(self.student, "mdn_entropy"):
					entropy = self.student.mdn_entropy(obs).mean()
					mean_entropy += entropy.item()
					total_loss = total_loss - self.mdn_entropy_coef * entropy

				loss = loss + total_loss
				mean_behavior_loss += behavior_loss.item()
				cnt += 1

				if cnt % self.gradient_length == 0:
					self.optimizer.zero_grad()
					loss.backward()
					if self.is_multi_gpu:
						self.reduce_parameters()
					if self.max_grad_norm:
						nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
					self.optimizer.step()
					self.student.detach_hidden_state()
					loss = torch.tensor(0.0, device=self.device)

				self.student.reset(dones.view(-1))
				self.teacher.reset(dones.view(-1))
				self.student.detach_hidden_state(dones.view(-1))

		mean_behavior_loss /= max(cnt, 1)
		self.storage.clear()
		self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
		self.student.detach_hidden_state()

		loss_dict = {"behavior": mean_behavior_loss}
		if self.mdn_loss_type == "teacher_distribution" and teacher_std_cnt > 0:
			loss_dict["teacher_std"] = mean_teacher_std / teacher_std_cnt
		if self.mdn_entropy_coef != 0.0 and cnt > 0:
			loss_dict["mdn_entropy"] = mean_entropy / cnt

		return loss_dict
