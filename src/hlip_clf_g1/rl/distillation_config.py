from mjlab.rl.config import RslRlBaseRunnerCfg, RslRlOnPolicyRunnerCfg
from dataclasses import dataclass, field
from typing import Any, Literal


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
      "output_channels": (16, 32, 64),
      "kernel_size": (5, 3, 3),
      "stride": (2, 2, 1),
      "padding": "zeros",
      "norm": "batch",
      "activation": "elu",
      "max_pool": (False, False, False),
      "global_pool": "avg",
      "flatten": True,
    }
  )
  """CNN encoder configuration passed to rsl_rl.modules.CNN."""


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
  loss_type: Literal["mse", "huber"] = "mse"
  """The loss type to use for the student policy."""


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
  student: RslRlDistillationModelCfg | RslRlDistillationCnnModelCfg = field(
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

@dataclass
class RslRlDistillationFineTuneRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Configuration of the runner for distillation algorithms."""

  class_name: str = "DistillationFineTuneRunner"
  """The runner class name. Default is DistillationRunner."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor": ("student_vec", "head_camera_depth"),
      "critic": ("critic",),
    }
  )