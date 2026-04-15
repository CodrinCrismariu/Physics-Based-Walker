from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories

from hlip_clf_g1.rl.models.cnn_transformer_model import CNNTransformerModel


class CNNTransformerMDNModel(CNNTransformerModel):
  """CNN+Transformer student with a Mixture Density head.

  The model reuses the CNNTransformer latent pipeline and replaces the final
  Gaussian actor parameterization with a k-component diagonal Gaussian mixture.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, dict[str, object]] | dict[str, object],
    cnns=None,
    hidden_dims: tuple[int, ...] | list[int] = (256,),
    activation: str = "elu",
    obs_normalization: bool = False,
    stochastic: bool = True,
    init_noise_std: float = 1.0,
    noise_std_type: str = "scalar",
    state_dependent_std: bool = False,
    transformer_cfg: dict[str, object] | None = None,
    mdn_num_modes: int = 3,
    mdn_min_std: float = 1.0e-3,
    mdn_min_log_std: float = -5.0,
    mdn_max_log_std: float = 2.0,
    mdn_inference_mode: str = "top_mode_mean",
  ) -> None:
    del init_noise_std, noise_std_type, state_dependent_std

    if mdn_num_modes <= 0:
      raise ValueError("mdn_num_modes must be > 0.")
    if mdn_inference_mode not in ("top_mode_mean", "mixture_mean"):
      raise ValueError(
        "mdn_inference_mode must be one of {'top_mode_mean', 'mixture_mean'}."
      )

    self.action_dim = int(output_dim)
    self.mdn_num_modes = int(mdn_num_modes)
    self.mdn_min_std = float(max(mdn_min_std, 1.0e-8))
    self.mdn_min_log_std = float(mdn_min_log_std)
    self.mdn_max_log_std = float(mdn_max_log_std)
    self.mdn_inference_mode = mdn_inference_mode
    self._mdn_stochastic = bool(stochastic)

    mdn_param_dim = self.mdn_num_modes + 2 * self.mdn_num_modes * self.action_dim

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=mdn_param_dim,
      cnn_cfg=cnn_cfg,
      cnns=cnns,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      stochastic=False,
      init_noise_std=1.0,
      noise_std_type="scalar",
      state_dependent_std=False,
      transformer_cfg=transformer_cfg,
    )

    self._last_logits: torch.Tensor | None = None
    self._last_means: torch.Tensor | None = None
    self._last_stds: torch.Tensor | None = None

  def _split_mdn_params(
    self,
    raw_params: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = self.mdn_num_modes
    a = self.action_dim

    logits = raw_params[..., :k]
    means_flat = raw_params[..., k : k + k * a]
    log_stds_flat = raw_params[..., k + k * a :]

    means = means_flat.view(*raw_params.shape[:-1], k, a)
    log_stds = log_stds_flat.view(*raw_params.shape[:-1], k, a)
    log_stds = torch.clamp(log_stds, min=self.mdn_min_log_std, max=self.mdn_max_log_std)
    stds = torch.exp(log_stds).clamp(min=self.mdn_min_std)

    return logits, means, stds

  def _compute_mdn_params_from_latent(
    self,
    latent: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_params = self.mlp(latent)
    return self._split_mdn_params(raw_params)

  def _compute_mdn_params(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
    latent = self.get_latent(obs, masks, hidden_state)
    return self._compute_mdn_params_from_latent(latent)

  def _cache_params(
    self,
    logits: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
  ) -> None:
    self._last_logits = logits.detach()
    self._last_means = means.detach()
    self._last_stds = stds.detach()

  def _deterministic_action(
    self,
    logits: torch.Tensor,
    means: torch.Tensor,
  ) -> torch.Tensor:
    if self.mdn_inference_mode == "mixture_mean":
      probs = torch.softmax(logits, dim=-1)
      return torch.sum(probs.unsqueeze(-1) * means, dim=-2)

    top_idx = torch.argmax(logits, dim=-1)
    gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(*top_idx.shape, 1, self.action_dim)
    return torch.gather(means, dim=-2, index=gather_idx).squeeze(-2)

  def _sample_action(
    self,
    logits: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
  ) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    flat_probs = probs.reshape(-1, self.mdn_num_modes)
    flat_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
    mode_idx = flat_idx.view(*logits.shape[:-1])

    gather_idx = mode_idx.unsqueeze(-1).unsqueeze(-1).expand(*mode_idx.shape, 1, self.action_dim)
    selected_means = torch.gather(means, dim=-2, index=gather_idx).squeeze(-2)
    selected_stds = torch.gather(stds, dim=-2, index=gather_idx).squeeze(-2)

    return selected_means + selected_stds * torch.randn_like(selected_means)

  @staticmethod
  def _gaussian_log_prob(
    actions: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
  ) -> torch.Tensor:
    var = stds * stds
    log_term = torch.log(stds) + 0.5 * math.log(2.0 * math.pi)
    return -torch.sum(((actions - means) ** 2) / (2.0 * var) + log_term, dim=-1)

  def _mixture_log_prob(
    self,
    actions: torch.Tensor,
    logits: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
  ) -> torch.Tensor:
    actions_expanded = actions.unsqueeze(-2)
    comp_log_prob = self._gaussian_log_prob(actions_expanded, means, stds)
    log_weights = torch.log_softmax(logits, dim=-1)
    return torch.logsumexp(log_weights + comp_log_prob, dim=-1)

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    logits, means, stds = self._compute_mdn_params(obs, masks, hidden_state)
    self._cache_params(logits, means, stds)

    if stochastic_output and self._mdn_stochastic:
      return self._sample_action(logits, means, stds)
    return self._deterministic_action(logits, means)

  def mdn_log_prob(
    self,
    obs: TensorDict,
    actions: torch.Tensor,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    logits, means, stds = self._compute_mdn_params(obs, masks, hidden_state)
    self._cache_params(logits, means, stds)
    return self._mixture_log_prob(actions, logits, means, stds)

  def mdn_nll(
    self,
    obs: TensorDict,
    actions: torch.Tensor,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    reduction: str = "mean",
  ) -> torch.Tensor:
    nll = -self.mdn_log_prob(obs, actions, masks, hidden_state)
    if reduction == "none":
      return nll
    if reduction == "sum":
      return torch.sum(nll)
    if reduction == "mean":
      return torch.mean(nll)
    raise ValueError("reduction must be one of {'none', 'sum', 'mean'}.")

  def mdn_entropy(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    logits, means, stds = self._compute_mdn_params(obs, masks, hidden_state)
    del means
    probs = torch.softmax(logits, dim=-1)

    gaussian_const = 0.5 * (1.0 + math.log(2.0 * math.pi))
    component_entropy = torch.sum(torch.log(stds) + gaussian_const, dim=-1)
    expected_component_entropy = torch.sum(probs * component_entropy, dim=-1)

    categorical_entropy = -torch.sum(
      probs * torch.log(torch.clamp(probs, min=1.0e-8)),
      dim=-1,
    )
    return expected_component_entropy + categorical_entropy

  @property
  def output_mean(self) -> torch.Tensor:
    if self._last_logits is None or self._last_means is None:
      return torch.zeros(self.action_dim, device=next(self.parameters()).device)
    return self._deterministic_action(self._last_logits, self._last_means)

  @property
  def output_std(self) -> torch.Tensor:
    if self._last_logits is None or self._last_stds is None:
      return torch.full(
        (self.action_dim,),
        self.mdn_min_std,
        device=next(self.parameters()).device,
      )

    probs = torch.softmax(self._last_logits, dim=-1)
    expected_std = torch.sum(probs.unsqueeze(-1) * self._last_stds, dim=-2)
    return torch.mean(expected_std, dim=0)

  @property
  def output_entropy(self) -> torch.Tensor:
    if self._last_logits is None or self._last_stds is None:
      return torch.zeros(1, device=next(self.parameters()).device)

    probs = torch.softmax(self._last_logits, dim=-1)
    gaussian_const = 0.5 * (1.0 + math.log(2.0 * math.pi))
    component_entropy = torch.sum(torch.log(self._last_stds) + gaussian_const, dim=-1)
    expected_component_entropy = torch.sum(probs * component_entropy, dim=-1)
    categorical_entropy = -torch.sum(
      probs * torch.log(torch.clamp(probs, min=1.0e-8)),
      dim=-1,
    )
    return expected_component_entropy + categorical_entropy

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    if self._last_logits is None or self._last_means is None or self._last_stds is None:
      raise RuntimeError("MDN parameters are not cached; run forward() before get_output_log_prob().")
    return self._mixture_log_prob(outputs, self._last_logits, self._last_means, self._last_stds)
