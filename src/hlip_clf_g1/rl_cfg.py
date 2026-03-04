"""RL configuration for Unitree G1 HLIP + CLF walking task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


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
