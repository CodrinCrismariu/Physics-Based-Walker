from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, HiddenState


class CNNTransformerModel(MLPModel):
  """Student model that fuses depth and vector observations with a transformer.

  Pipeline:
    1) Per-image-group CNN encoder.
    2) Token projection (vector token + CNN tokens).
    3) Transformer encoder over tokens.
    4) Mean pooling, then action head inherited from ``MLPModel``.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, dict[str, Any]] | dict[str, Any],
    cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    hidden_dims: tuple[int, ...] | list[int] = (256,),
    activation: str = "elu",
    obs_normalization: bool = False,
    stochastic: bool = False,
    init_noise_std: float = 1.0,
    noise_std_type: str = "scalar",
    state_dependent_std: bool = False,
    transformer_cfg: dict[str, Any] | None = None,
  ) -> None:
    self._get_obs_dim(obs, obs_groups, obs_set)

    if cnns is not None:
      if set(cnns.keys()) != set(self.obs_groups_2d):
        raise ValueError("The 2D observations must be identical for shared CNN encoders.")
      cnn_modules = dict(cnns)
    else:
      if not all(isinstance(v, dict) for v in cnn_cfg.values()):
        cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
      if len(cnn_cfg) != len(self.obs_groups_2d):
        raise ValueError("The number of CNN configs must match the number of 2D observation groups.")

      cnn_modules = {}
      for idx, obs_group in enumerate(self.obs_groups_2d):
        cnn_modules[obs_group] = CNN(
          input_dim=self.obs_dims_2d[idx],
          input_channels=self.obs_channels_2d[idx],
          **cnn_cfg[obs_group],
        )

    self._cnn_output_dims: dict[str, int] = {}
    for obs_group, cnn in cnn_modules.items():
      if cnn.output_channels is not None:
        raise ValueError("CNN outputs must be flattened before transformer token projection.")
      self._cnn_output_dims[obs_group] = int(cnn.output_dim)  # type: ignore[arg-type]

    cfg = self._resolve_transformer_cfg(transformer_cfg)
    self.transformer_latent_dim = int(cfg["d_model"])
    self._transformer_cfg = cfg

    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      stochastic,
      init_noise_std,
      noise_std_type,
      state_dependent_std,
    )

    self.cnns = nn.ModuleDict(cnn_modules)
    self.cnn_token_projections = nn.ModuleDict(
      {
        obs_group: nn.Linear(self._cnn_output_dims[obs_group], self.transformer_latent_dim)
        for obs_group in self.obs_groups_2d
      }
    )

    self.vector_token_projection: nn.Linear | None = None
    if self.obs_dim > 0:
      self.vector_token_projection = nn.Linear(self.obs_dim, self.transformer_latent_dim)

    token_count = len(self.obs_groups_2d) + (1 if self.vector_token_projection is not None else 0)
    if token_count <= 0:
      raise ValueError("CNNTransformerModel requires at least one token source.")

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=self.transformer_latent_dim,
      nhead=int(cfg["nhead"]),
      dim_feedforward=int(cfg["dim_feedforward"]),
      dropout=float(cfg["dropout"]),
      activation=str(cfg["activation"]),
      batch_first=True,
      norm_first=bool(cfg["norm_first"]),
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(cfg["num_layers"]))

    self.use_positional_encoding = bool(cfg["use_positional_encoding"])
    if self.use_positional_encoding:
      self.positional_embedding = nn.Parameter(
        torch.zeros(1, token_count, self.transformer_latent_dim)
      )
      nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)
    else:
      self.register_parameter("positional_embedding", None)

    self.transformer_norm = nn.LayerNorm(self.transformer_latent_dim)

  @staticmethod
  def _resolve_transformer_cfg(transformer_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = {
      "d_model": 256,
      "nhead": 8,
      "num_layers": 2,
      "dim_feedforward": 512,
      "dropout": 0.0,
      "activation": "gelu",
      "norm_first": False,
      "use_positional_encoding": True,
    }
    if transformer_cfg is not None:
      cfg.update(transformer_cfg)

    d_model = int(cfg["d_model"])
    nhead = int(cfg["nhead"])
    if d_model <= 0 or nhead <= 0 or d_model % nhead != 0:
      raise ValueError("transformer_cfg requires d_model > 0, nhead > 0, and d_model % nhead == 0.")

    return cfg

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state

    tokens: list[torch.Tensor] = []

    if self.vector_token_projection is not None:
      obs_list = [obs[obs_group] for obs_group in self.obs_groups]
      latent_1d = torch.cat(obs_list, dim=-1)
      latent_1d = self.obs_normalizer(latent_1d)
      tokens.append(self.vector_token_projection(latent_1d))

    for obs_group in self.obs_groups_2d:
      cnn_latent = self.cnns[obs_group](obs[obs_group])
      tokens.append(self.cnn_token_projections[obs_group](cnn_latent))

    token_tensor = torch.stack(tokens, dim=1)
    if self.positional_embedding is not None:
      token_tensor = token_tensor + self.positional_embedding

    encoded_tokens = self.transformer(token_tensor)
    pooled = encoded_tokens.mean(dim=1)
    return self.transformer_norm(pooled)

  def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
    active_obs_groups = obs_groups[obs_set]
    obs_dim_1d = 0
    obs_groups_1d = []
    obs_dims_2d = []
    obs_channels_2d = []
    obs_groups_2d = []

    for obs_group in active_obs_groups:
      if len(obs[obs_group].shape) == 4:
        obs_groups_2d.append(obs_group)
        obs_dims_2d.append(obs[obs_group].shape[2:4])
        obs_channels_2d.append(obs[obs_group].shape[1])
      elif len(obs[obs_group].shape) == 2:
        obs_groups_1d.append(obs_group)
        obs_dim_1d += obs[obs_group].shape[-1]
      else:
        raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

    if not obs_groups_2d:
      raise ValueError("CNNTransformerModel requires at least one 2D observation group.")

    self.obs_dims_2d = obs_dims_2d
    self.obs_channels_2d = obs_channels_2d
    self.obs_groups_2d = obs_groups_2d
    return obs_groups_1d, obs_dim_1d

  def _get_latent_dim(self) -> int:
    return self.transformer_latent_dim
