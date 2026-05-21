"""RL configuration for Unitree G1 HLIP + CLF walking task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from hlip_clf_g1.rl.distillation_config import (
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationCnnModelCfg,
  RslRlDistillationCnnMdnModelCfg,
  RslRlDistillationCnnTransformerModelCfg,
  RslRlDistillationCnnTransformerMdnModelCfg,
  RslRlDistillationModelCfg,
  RslRlDistillationRunnerCfg,
)


_DISTILLATION_STUDENT_OBS_GROUPS = (
  "student_vec",
  "head_camera_depth",
)
_DISTILLATION_TEACHER_OBS_GROUPS = ("teacher",)


def _depth_cnn_cfg() -> dict[str, object]:
  # Sparse-terrain depth images benefit from a wider early receptive field and
  # an extra stage while preserving the 2D map (no global pooling).
  return {
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


def _make_student_model_cfg(
  *,
  init_noise_std: float,
  stochastic: bool,
) -> RslRlDistillationCnnModelCfg:
  return RslRlDistillationCnnModelCfg(
    init_noise_std=init_noise_std,
    obs_normalization=True,
    hidden_dims=(512, 256, 128),
    activation="elu",
    stochastic=stochastic,
    cnn_cfg={"head_camera_depth": _depth_cnn_cfg()},
  )


def _make_student_transformer_model_cfg(
  *,
  init_noise_std: float,
  stochastic: bool,
) -> RslRlDistillationCnnTransformerModelCfg:
  return RslRlDistillationCnnTransformerModelCfg(
    obs_normalization=True,
    init_noise_std=init_noise_std,
    hidden_dims=(512, 256, 128),
    activation="elu",
    stochastic=stochastic,
    cnn_cfg={"head_camera_depth": _depth_cnn_cfg()},
  )


def _make_student_cnn_mdn_model_cfg(
  *,
  stochastic: bool,
) -> RslRlDistillationCnnMdnModelCfg:
  return RslRlDistillationCnnMdnModelCfg(
    obs_normalization=True,
    hidden_dims=(512, 256, 128),
    activation="elu",
    stochastic=stochastic,
    cnn_cfg={"head_camera_depth": _depth_cnn_cfg()},
    mdn_num_modes=2,
    mdn_min_std=1.0e-3,
    mdn_min_log_std=-3.0,
    mdn_max_log_std=2.0,
    mdn_inference_mode="top_mode_mean",
  )


def _make_student_mdn_model_cfg(
  *,
  stochastic: bool,
) -> RslRlDistillationCnnTransformerMdnModelCfg:
  return RslRlDistillationCnnTransformerMdnModelCfg(
    class_name="hlip_clf_g1.rl.models.cnn_transformer_mdn_model:CNNTransformerMDNModel",
    obs_normalization=True,
    hidden_dims=(512, 256, 128),
    activation="elu",
    stochastic=stochastic,
    cnn_cfg={"head_camera_depth": _depth_cnn_cfg()},
    mdn_num_modes=2,
    mdn_min_std=1.0e-3,
    mdn_min_log_std=-3.0,
    mdn_max_log_std=2.0,
    mdn_inference_mode="top_mode_mean",
  )


def _make_teacher_model_cfg(
  *,
  init_noise_std: float,
  stochastic: bool,
) -> RslRlDistillationModelCfg:
  return RslRlDistillationModelCfg(
    init_noise_std=init_noise_std,
    obs_normalization=True,
    hidden_dims=(512, 256, 128),
    activation="elu",
    stochastic=stochastic,
  )


def _make_distillation_algorithm_cfg() -> RslRlDistillationAlgorithmCfg:
  return RslRlDistillationAlgorithmCfg(
    num_learning_epochs=10,
    learning_rate=5 * 1.0e-4,
    gradient_length=2,
    max_grad_norm=2.0,
    optimizer="adam",
    loss_type="huber",
  )


def _make_distillation_mdn_algorithm_cfg() -> RslRlDistillationAlgorithmCfg:
  return RslRlDistillationAlgorithmCfg(
    class_name="hlip_clf_g1.rl.distillation_algorithm:DistillationMDN",
    num_learning_epochs=10,
    learning_rate=5 * 1.0e-4,
    gradient_length=2,
    max_grad_norm=2.0,
    optimizer="adam",
    loss_type="huber",
    mdn_loss_type="teacher_distribution",
    mdn_teacher_num_samples=8,
    mdn_teacher_std_scale=0.1,
    mdn_teacher_sample_std_floor=1.0e-6,
    mdn_entropy_coef=0.0,
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
      learning_rate=0.5e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_hlip_clf",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=4000,
  )


def unitree_g1_hlip_random_step_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO runner config for the flat random-step HLIP task."""
  cfg = unitree_g1_hlip_ppo_runner_cfg()
  cfg.experiment_name = "g1_hlip_clf_random_step"
  return cfg


def unitree_g1_hlip_corridor_ppo_from_distillation_mdn_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO config to fine-tune corridor policies from MDN distillation students.

  This runner uses the distillation student observation interface for actor
  (`student_vec` + `head_camera_depth`) so `student_state_dict` weights can be
  loaded directly into the PPO actor.
  """
  return RslRlOnPolicyRunnerCfg(
    obs_groups={
      "actor": _DISTILLATION_STUDENT_OBS_GROUPS,
      "critic": _DISTILLATION_TEACHER_OBS_GROUPS,
    },
    actor=RslRlModelCfg(
      class_name="hlip_clf_g1.rl.models.cnn_transformer_mdn_model:CNNTransformerMDNModel",
      init_noise_std=1.0,
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
      stochastic=True,
      cnn_cfg={"head_camera_depth": _depth_cnn_cfg()},
    ),
    critic=RslRlModelCfg(
      init_noise_std=1.0,
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
      learning_rate=2.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_hlip_clf_corridor_ppo_finetune_from_mdn",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=4000,
  )


def unitree_g1_hlip_distillation_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Create RL runner configuration for Unitree G1 HLIP + CLF walking task."""
  return RslRlDistillationRunnerCfg(
    obs_groups={
      "student": _DISTILLATION_STUDENT_OBS_GROUPS,
      "teacher": _DISTILLATION_TEACHER_OBS_GROUPS,
    },
    student=_make_student_model_cfg(init_noise_std=0.0, stochastic=True),
    teacher=_make_teacher_model_cfg(
      init_noise_std=0.0,
      stochastic=True,
    ),
    algorithm=_make_distillation_algorithm_cfg(),
    experiment_name="g1_hlip_clf_distillation",
    save_interval=10,
    num_steps_per_env=120,
    max_iterations=1000,
  )


def unitree_g1_hlip_distillation_mdn_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Create MDN distillation runner configuration for Unitree G1 HLIP + CLF."""
  return RslRlDistillationRunnerCfg(
    obs_groups={
      "student": _DISTILLATION_STUDENT_OBS_GROUPS,
      "teacher": _DISTILLATION_TEACHER_OBS_GROUPS,
    },
    student=_make_student_mdn_model_cfg(stochastic=True),
    teacher=_make_teacher_model_cfg(
      init_noise_std=0.0,
      stochastic=True,
    ),
    algorithm=_make_distillation_mdn_algorithm_cfg(),
    experiment_name="g1_hlip_clf_distillation_mdn",
    save_interval=100,
    num_steps_per_env=120,
    max_iterations=2500,
  )


def unitree_g1_hlip_distillation_cnn_mdn_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Create no-transformer MDN distillation runner configuration."""
  return RslRlDistillationRunnerCfg(
    obs_groups={
      "student": _DISTILLATION_STUDENT_OBS_GROUPS,
      "teacher": _DISTILLATION_TEACHER_OBS_GROUPS,
    },
    student=_make_student_cnn_mdn_model_cfg(stochastic=True),
    teacher=_make_teacher_model_cfg(
      init_noise_std=0.0,
      stochastic=True,
    ),
    algorithm=_make_distillation_mdn_algorithm_cfg(),
    experiment_name="g1_hlip_clf_distillation_cnn_mdn",
    save_interval=100,
    num_steps_per_env=120,
    max_iterations=2500,
  )


def unitree_g1_hlip_distillation_transformer_mlp_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Create CNN+Transformer distillation config with a standard MLP head."""
  return RslRlDistillationRunnerCfg(
    obs_groups={
      "student": _DISTILLATION_STUDENT_OBS_GROUPS,
      "teacher": _DISTILLATION_TEACHER_OBS_GROUPS,
    },
    student=_make_student_transformer_model_cfg(
      init_noise_std=0.0,
      stochastic=True,
    ),
    teacher=_make_teacher_model_cfg(
      init_noise_std=0.0,
      stochastic=True,
    ),
    algorithm=_make_distillation_algorithm_cfg(),
    experiment_name="g1_hlip_clf_distillation_transformer_mlp",
    save_interval=100,
    num_steps_per_env=120,
    max_iterations=2500,
  )
