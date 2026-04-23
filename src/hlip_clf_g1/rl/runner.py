import os

import torch
import wandb

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from hlip_clf_g1.rl.exporter import attach_onnx_metadata
from rsl_rl.runners import DistillationRunner

class LIPOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  @staticmethod
  def _load_state_dict_by_shape(module, source_state_dict: dict) -> int:
    """Load model weights by key and tensor shape.

    Useful when loading distilled checkpoints whose output heads differ from the
    current actor setup (for example, different MDN mode count).
    """
    target_state_dict = module.state_dict()
    merged_state_dict = dict(target_state_dict)
    loaded_count = 0
    for key, value in source_state_dict.items():
      if key in merged_state_dict and merged_state_dict[key].shape == value.shape:
        merged_state_dict[key] = value
        loaded_count += 1
    module.load_state_dict(merged_state_dict, strict=False)
    return loaded_count

  @staticmethod
  def _map_distillation_checkpoint_for_ppo(
    loaded_dict: dict,
    load_cfg: dict | None,
  ) -> tuple[dict, dict | None]:
    """Map distillation checkpoints to PPO-compatible keys when possible.

    Distillation checkpoints store the policy under ``student_state_dict``.
    PPO expects ``actor_state_dict`` and, by default, also critic/optimizer
    states. For fine-tuning from a distilled student, we default to loading only
    actor weights unless the caller explicitly provides a different load config.
    """
    if "actor_state_dict" in loaded_dict or "student_state_dict" not in loaded_dict:
      return loaded_dict, load_cfg

    remapped = dict(loaded_dict)
    remapped["actor_state_dict"] = loaded_dict["student_state_dict"]

    if load_cfg is None:
      return remapped, {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
      }

    effective_load_cfg = dict(load_cfg)
    if "critic_state_dict" not in remapped and effective_load_cfg.get("critic"):
      effective_load_cfg["critic"] = False
    if "optimizer_state_dict" not in remapped and effective_load_cfg.get("optimizer"):
      effective_load_cfg["optimizer"] = False
    if "iter" not in remapped and effective_load_cfg.get("iteration"):
      effective_load_cfg["iteration"] = False

    return remapped, effective_load_cfg

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoints with PPO/distillation compatibility handling."""
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    loaded_dict, effective_load_cfg = self._map_distillation_checkpoint_for_ppo(
      loaded_dict,
      load_cfg,
    )
    if effective_load_cfg != load_cfg:
      print(
        "[INFO] Remapped PPO load_cfg "
        f"from {load_cfg} to {effective_load_cfg}."
      )

    try:
      load_iteration = self.alg.load(loaded_dict, effective_load_cfg, strict)
    except RuntimeError:
      if not (
        "student_state_dict" in loaded_dict
        and effective_load_cfg is not None
        and effective_load_cfg.get("actor")
      ):
        raise

      loaded_count = self._load_state_dict_by_shape(
        self.alg.actor,
        loaded_dict["actor_state_dict"],
      )
      if loaded_count == 0:
        raise

      print(
        "[WARN] Actor strict-load from distillation checkpoint failed; "
        f"loaded {loaded_count} actor tensors by key/shape and kept the rest initialized."
      )
      load_iteration = False

    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict.get("infos", {})
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  def save(self, path: str, infos=None):
    """Save the model, export ONNX, and attach metadata."""
    super().save(path, infos)

    if self.logger in ["wandb"]:
      policy_path = path.split("model")[0]
      filename = "policy.onnx"
      self.export_policy_to_onnx(path=policy_path, filename=filename)
      attach_onnx_metadata(
        self.env.unwrapped,
        wandb.run.name,  # type: ignore
        path=policy_path,
        filename=filename,
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

class LIPDistilledOnPolicyRunner(DistillationRunner):
  env: RslRlVecEnvWrapper

  def get_inference_policy(self, device: str | None = None):
    """Return inference policy for distilled student."""
    self.alg.eval_mode()
    policy = self.alg.get_policy().to(device)
    return policy

  @staticmethod
  def _map_load_cfg_for_distillation(
    loaded_dict: dict,
    load_cfg: dict | None,
  ) -> dict | None:
    """Translate generic load flags into distillation-specific flags.

    The stock play script passes ``{"actor": True}`` for inference loading.
    Distillation checkpoints store ``student_state_dict``/``teacher_state_dict``
    and expect ``student``/``teacher`` keys in ``load_cfg``.
    """
    if load_cfg is None:
      return None

    # If caller already uses distillation-native keys, keep as-is.
    if any(key in load_cfg for key in ("student", "teacher", "optimizer", "iteration")):
      return load_cfg

    # Compatibility with play.py actor-centric loading.
    if load_cfg.get("actor"):
      if "student_state_dict" in loaded_dict:
        # Distillation checkpoint: inference should use student only.
        return {
          "student": True,
          "teacher": False,
          "optimizer": False,
          "iteration": False,
        }
      if "actor_state_dict" in loaded_dict:
        # PPO checkpoint fallback: teacher can be initialized from actor.
        return {
          "student": False,
          "teacher": True,
          "optimizer": False,
          "iteration": False,
        }

    return load_cfg

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict | None:
    """Load checkpoint with compatibility handling for play-time inference."""
    loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
    effective_load_cfg = self._map_load_cfg_for_distillation(loaded_dict, load_cfg)
    if effective_load_cfg != load_cfg:
      print(
        "[INFO] Remapped distillation load_cfg "
        f"from {load_cfg} to {effective_load_cfg}."
      )

    load_iteration = self.alg.load(loaded_dict, effective_load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]
    return loaded_dict.get("infos")

  def save(self, path: str, infos=None):
    """Save model and training state using the base distillation runner behavior."""
    super().save(path, infos)
    