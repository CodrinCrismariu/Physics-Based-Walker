from dataclasses import dataclass, field
from typing import Any, Literal

from mjlab.rl.config import RslRlBaseRunnerCfg


@dataclass
class RslRlDistillationModelCfg:
  """Config for distillation student/teacher models compatible with rsl_rl MLPModel."""

  hidden_dims: tuple[int, ...] = (512, 256, 128)
  """The hidden dimensions of the model."""
  activation: str = "elu"
  """The activation function."""
  obs_normalization: bool = False
  """Whether to normalize observations."""
  init_noise_std: float = 1.0
  """Initial standard deviation for stochastic outputs."""
  noise_std_type: Literal["scalar", "log"] = "scalar"
  """How output noise std is parameterized."""
  stochastic: bool = False
  """Whether the model output is stochastic."""
  state_dependent_std: bool = False
  """Whether output std depends on state."""
  class_name: str = "MLPModel"
  """Model class name resolved by rsl_rl."""


@dataclass
class RslRlDistillationCnnModelCfg(RslRlDistillationModelCfg):
  """Config for distillation models compatible with rsl_rl CNNModel."""

  class_name: str = "CNNModel"
  """Model class name resolved by rsl_rl."""
  cnn_cfg: dict[str, dict[str, Any]] | dict[str, Any] = field(
    default_factory=lambda: {
      "output_channels": (24, 32, 48, 64),
      "kernel_size": (5, 3, 3, 3),
      "stride": (2, 2, 1, 1),
      "padding": "zeros",
      "norm": "batch",
      "activation": "elu",
      "max_pool": (False, False, False, False),
      "global_pool": "none",
      "flatten": True,
    }
  )
  """CNN encoder configuration passed to rsl_rl.modules.CNN."""


@dataclass
class RslRlDistillationCnnTransformerModelCfg(RslRlDistillationCnnModelCfg):
  """Config for distillation models compatible with CNNTransformerModel."""

  class_name: str = "hlip_clf_g1.rl.models.cnn_transformer_model:CNNTransformerModel"
  """Model class name resolved by rsl_rl."""
  transformer_cfg: dict[str, Any] | None = None
  """Optional transformer encoder configuration override."""


@dataclass
class RslRlDistillationCnnMdnModelCfg(RslRlDistillationCnnModelCfg):
  """Config for CNN student with a Mixture Density head and no transformer."""

  class_name: str = "hlip_clf_g1.rl.models.cnn_mdn_model:CNNMDNModel"
  """Model class name resolved by rsl_rl."""
  mdn_num_modes: int = 3
  """Number of Gaussian mixture components."""
  mdn_min_std: float = 1.0e-3
  """Minimum standard deviation clamp applied per action dimension."""
  mdn_min_log_std: float = -5.0
  """Lower bound for predicted log standard deviation."""
  mdn_max_log_std: float = 2.0
  """Upper bound for predicted log standard deviation."""
  mdn_inference_mode: Literal["top_mode_mean", "mixture_mean"] = "top_mode_mean"
  """Deterministic action selection mode for inference/update."""


@dataclass
class RslRlDistillationCnnTransformerMdnModelCfg(RslRlDistillationCnnTransformerModelCfg):
  """Config for CNNTransformer student with a Mixture Density head."""

  class_name: str = "hlip_clf_g1.rl.models.cnn_transformer_mdn_model:CNNTransformerMDNModel"
  """Model class name resolved by rsl_rl."""
  mdn_num_modes: int = 3
  """Number of Gaussian mixture components."""
  mdn_min_std: float = 1.0e-3
  """Minimum standard deviation clamp applied per action dimension."""
  mdn_min_log_std: float = -5.0
  """Lower bound for predicted log standard deviation."""
  mdn_max_log_std: float = 2.0
  """Upper bound for predicted log standard deviation."""
  mdn_inference_mode: Literal["top_mode_mean", "mixture_mean"] = "top_mode_mean"
  """Deterministic action selection mode for inference/update."""


@dataclass
class RslRlDistillationAlgorithmCfg:
  """Configuration for the distillation algorithm."""

  class_name: str = "Distillation"
  """The algorithm class name. Default is Distillation."""
  num_learning_epochs: int = 1
  """The number of updates performed with each sample."""
  learning_rate: float = 1e-3
  """The learning rate for the student policy."""
  gradient_length: int = 15
  """The number of environment steps the gradient flows back."""
  max_grad_norm: None | float = None
  """The maximum norm the gradient is clipped to."""
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """The optimizer to use for the student policy."""
  loss_type: Literal["mse", "huber"] = "huber"
  """The loss type to use for the student policy."""
  nan_guard_enabled: bool = True
  """Whether guarded classic distillation skips non-finite losses/gradients."""
  nan_guard_sanitize_rollout_actions: bool = True
  """Whether guarded classic distillation zeros non-finite rollout actions."""
  mdn_loss_type: Literal["action_nll", "teacher_distribution"] = "action_nll"
  """MDN objective: fit stored teacher actions or the teacher action distribution."""
  mdn_teacher_num_samples: int = 1
  """Number of PPO teacher distribution samples used for MDN distribution matching."""
  mdn_teacher_std_scale: float = 1.0
  """Scale applied to PPO teacher std before MDN distribution matching."""
  mdn_teacher_sample_std_floor: float = 1.0e-6
  """Minimum std used when sampling the PPO teacher distribution."""
  mdn_entropy_coef: float = 0.0
  """Optional entropy regularization coefficient for MDN distillation."""


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Configuration of the runner for distillation algorithms."""

  class_name: str = "DistillationRunner"
  """The runner class name. Default is DistillationRunner."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "student": ("student_vec", "head_camera_depth"),
      "teacher": ("teacher",),
    }
  )
  """Observation groups for distillation student and teacher models."""
  student: (
    RslRlDistillationModelCfg
    | RslRlDistillationCnnModelCfg
    | RslRlDistillationCnnMdnModelCfg
    | RslRlDistillationCnnTransformerModelCfg
    | RslRlDistillationCnnTransformerMdnModelCfg
  ) = field(
    default_factory=lambda: RslRlDistillationCnnModelCfg(stochastic=True)
  )
  """The student model configuration."""
  teacher: RslRlDistillationModelCfg = field(
    default_factory=lambda: RslRlDistillationModelCfg(stochastic=False)
  )
  """The teacher model configuration."""
  algorithm: RslRlDistillationAlgorithmCfg = field(
    default_factory=RslRlDistillationAlgorithmCfg
  )
  """The algorithm configuration."""
