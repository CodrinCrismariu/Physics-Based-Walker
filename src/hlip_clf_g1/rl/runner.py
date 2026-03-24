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


class LIPDistillationFineTuneOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict | None:
    """Load checkpoints with distillation->PPO actor compatibility.

    Fine-tune runs use PPO (`actor`/`critic`), while distillation checkpoints
    store `student_state_dict`/`teacher_state_dict`. When a distillation
    checkpoint is provided, initialize PPO actor from `student_state_dict` and
    load actor-only by default.
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    # Keep compatibility with legacy PPO checkpoints.
    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        if key.startswith("actor."):
          actor_state_dict[key.replace("actor.", "mlp.")] = value
        elif key.startswith("actor_obs_normalizer."):
          actor_state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.")] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        if key.startswith("critic."):
          critic_state_dict[key.replace("critic.", "mlp.")] = value
        elif key.startswith("critic_obs_normalizer."):
          critic_state_dict[key.replace("critic_obs_normalizer.", "obs_normalizer.")] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    if "student_state_dict" in loaded_dict and "actor_state_dict" not in loaded_dict:
      loaded_dict["actor_state_dict"] = loaded_dict["student_state_dict"]
      if load_cfg is None:
        # Teacher/optimizer states from distillation are not PPO-compatible.
        load_cfg = {
          "actor": True,
          "critic": False,
          "optimizer": False,
          "iteration": False,
          "rnd": False,
        }
      print(
        "[INFO] Loading distillation checkpoint into PPO fine-tune runner: "
        "mapped student_state_dict -> actor_state_dict."
      )

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict.get("infos")
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

class LIPDistilledOnPolicyRunner(DistillationRunner):
  env: RslRlVecEnvWrapper

  def get_inference_policy(self, device: str | None = None):
    """Return inference policy and stream AI touchdown predictions to command debug-vis."""
    self.alg.eval_mode()
    policy = self.alg.get_policy().to(device)

    class _PolicyWithTouchdownVis:
      def __init__(self, policy_model, env_wrapper):
        self._policy_model = policy_model
        self._env_wrapper = env_wrapper

      def __call__(self, obs):
        actions = self._policy_model(obs)
        if hasattr(self._policy_model, "get_touchdown_pred"):
          try:
            touchdown = self._policy_model.get_touchdown_pred(obs).detach()
            cmd = self._env_wrapper.unwrapped.command_manager.get_term("hlip")
            if hasattr(cmd, "set_ai_touchdown_target"):
              cmd.set_ai_touchdown_target(touchdown)
          except Exception:
            # Visualization should never break play-time control.
            pass
        return actions

      def reset(self):
        reset_fn = getattr(self._policy_model, "reset", None)
        if reset_fn is not None:
          reset_fn()

    return _PolicyWithTouchdownVis(policy, self.env)

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
    