import os

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach HLIP-specific metadata to ONNX model."""
  onnx_path = os.path.join(path, filename)
  metadata = get_base_metadata(env, run_path)
  attach_metadata_to_onnx(onnx_path, metadata)
