from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import Distillation


class DistillationMDN(Distillation):
	"""MDN distillation variant using negative log-likelihood behavior loss.

	This class is additive and does not modify upstream Distillation behavior for
	existing tasks. It expects the student model to implement ``mdn_nll`` and
	optionally ``mdn_entropy``.
	"""

	def __init__(self, *args, mdn_entropy_coef: float = 0.0, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.mdn_entropy_coef = float(mdn_entropy_coef)

	def update(self) -> dict[str, float]:
		self.num_updates += 1
		mean_behavior_loss = 0.0
		mean_entropy = 0.0
		loss = torch.tensor(0.0, device=self.device)
		cnt = 0

		for _epoch in range(self.num_learning_epochs):
			self.student.reset(hidden_state=self.last_hidden_states[0])
			self.teacher.reset(hidden_state=self.last_hidden_states[1])
			self.student.detach_hidden_state()

			for obs, _, privileged_actions, dones in self.storage.generator():
				if not hasattr(self.student, "mdn_nll"):
					raise TypeError(
						"DistillationMDN requires a student model exposing mdn_nll(obs, actions, ...)."
					)

				behavior_loss = self.student.mdn_nll(obs, privileged_actions, reduction="mean")

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
		if self.mdn_entropy_coef != 0.0 and cnt > 0:
			loss_dict["mdn_entropy"] = mean_entropy / cnt

		return loss_dict
