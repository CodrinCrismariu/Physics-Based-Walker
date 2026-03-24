"""RL configuration for Unitree G1 HLIP + CLF walking task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from hlip_clf_g1.rl.distillation_config import (
  RslRlDistillationModelCfg,
  RslRlDistillationCnnModelCfg,
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
  RslRlDistillationFineTuneRunnerCfg,
)


def _depth_cnn_cfg() -> dict[str, object]:
  return {
    "output_channels": (16, 32, 64),
    "kernel_size": (5, 3, 3),
    "stride": (2, 2, 2),        # Changed to 2 to safely reduce spatial grid size
    "padding": "none",
    "norm": "batch",            # Batch norm is fine (fuses to 0 latency at inference)
    "activation": "lrelu", # Changed for faster inference calculation
    "max_pool": (False, False, False),
    "global_pool": "none",        # Removed! Do not use Global Average Pooling.
    "flatten": True,            
  }

def unitree_g1_hlip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 HLIP + CLF walking task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      init_noise_std=1.0,
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
      stochastic=True,
    ),
    critic=RslRlModelCfg(
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
      stochastic=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.008,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_hlip_clf",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )

def unitree_g1_hlip_distillation_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Create RL runner configuration for Unitree G1 HLIP + CLF walking task."""
  return RslRlDistillationRunnerCfg(
    obs_groups={
      "student": ("student_vec", "head_camera_depth", ),
      "teacher": ("teacher",),
    },
    student=RslRlDistillationCnnModelCfg(
      init_noise_std=0.0,
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
      stochastic=True,
      cnn_cfg=_depth_cnn_cfg(),
    ),
    teacher=RslRlDistillationModelCfg(
      init_noise_std=0.0,
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
      stochastic=True,
    ),
    algorithm=RslRlDistillationAlgorithmCfg(
      num_learning_epochs=10,
      learning_rate=5*1.0e-4,
      gradient_length=2,
      max_grad_norm=2.0,
      optimizer="adam",
      loss_type="huber",
    ),
    experiment_name="g1_hlip_clf_distillation",
    save_interval=10,
    num_steps_per_env=120,
    max_iterations=10001,
  )

def unitree_g1_hlip_fine_tune_ppo_runner_cfg() -> RslRlDistillationFineTuneRunnerCfg:
  """Create RL runner configuration for Unitree G1 HLIP + CLF walking task."""
  return RslRlDistillationFineTuneRunnerCfg(
    obs_groups={
      "actor": ("student_vec", "head_camera_depth"),
      "critic": ("critic",),
    },
    actor=RslRlModelCfg(
      init_noise_std=1.0,
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
      stochastic=True,
      class_name="CNNModel",
      cnn_cfg=_depth_cnn_cfg(),
    ),
    critic=RslRlModelCfg(
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
      stochastic=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.008,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_hlip_clf_distillation_fine_tune",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
