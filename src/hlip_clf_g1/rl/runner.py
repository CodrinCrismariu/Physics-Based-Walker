import os

import wandb

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from hlip_clf_g1.rl.exporter import attach_onnx_metadata


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
