"""Entry point wrapper for repository-level custom training script.

This keeps `uv run custom_train` importable from the installed package while
executing the implementation in `scripts/custom_train.py`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> Path:
  repo_root = Path(__file__).resolve().parents[2]
  repo_root_str = str(repo_root)
  if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
  return repo_root


def main() -> None:
  repo_root = _ensure_repo_root_on_path()

  try:
    module = importlib.import_module("scripts.custom_train")
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "Could not import scripts.custom_train. Ensure scripts/__init__.py and "
      "scripts/custom_train.py exist under project root: "
      f"{repo_root}"
    ) from exc

  entry = getattr(module, "main", None)
  if entry is None:
    raise AttributeError("scripts/custom_train.py does not define a main() function.")

  entry()
