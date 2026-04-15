import os

import torch
import wandb

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from hlip_clf_g1.rl.exporter import attach_onnx_metadata
from rsl_rl.runners import DistillationRunner

class LIPOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

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
    