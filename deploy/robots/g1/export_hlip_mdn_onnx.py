#!/usr/bin/env python3
"""Export the latest MDN distillation student checkpoint to ONNX.

Usage (from repo root):
    .venv/bin/python3 deploy/robots/g1/export_hlip_mdn_onnx.py

The script automatically finds the most recent wandb run checkpoint
for the g1_hlip_clf_distillation_mdn experiment and exports
    deploy/robots/g1/config/policy/hlip_mdn/v0/exported/policy.onnx

The exported ONNX has two named inputs:
    student_vec        [1, 96]       float32
    head_camera_depth  [1, 1, 24, 32] float32
and one output:
    actions            [1, 29]       float32  (MDN top-mode mean)

Run this script whenever a new MDN checkpoint is ready to deploy.
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import pathlib

import torch
import torch.nn as nn
from tensordict import TensorDict

# ── repo root on sys.path so hlip_clf_g1 is importable ──────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlip_clf_g1.rl.models.cnn_transformer_mdn_model import CNNTransformerMDNModel
from hlip_clf_g1.rl_cfg import _make_student_mdn_model_cfg, _depth_cnn_cfg

# ── Constants matching training config ──────────────────────────────────────
DEPTH_WIDTH  = 32
DEPTH_HEIGHT = 24
DEPTH_MIN    = 0.1
DEPTH_MAX    = 10.0
VEC_DIM      = 96   # student_vec: 3+3+3+29+29+29
IMG_DIM      = (1, DEPTH_HEIGHT, DEPTH_WIDTH)   # [C, H, W]
ACTION_DIM   = 29

WANDB_DIR = REPO_ROOT / "wandb"
EXPORT_DIR = REPO_ROOT / "deploy/robots/g1/config/policy/hlip_mdn/v0/exported"

OBS_GROUPS = {
    "student": ("student_vec", "head_camera_depth"),
}


def find_latest_checkpoint() -> pathlib.Path:
    """Return the path to the highest-numbered model_*.pt in the most recent wandb run."""
    # Latest run is first after sorting by mtime descending.
    run_dirs = sorted(
        WANDB_DIR.glob("run-*/files"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No wandb run directories found under {WANDB_DIR}.")

    for run_dir in run_dirs:
        pts = sorted(run_dir.glob("model_*.pt"),
                     key=lambda p: int(p.stem.split("_")[1]))
        if pts:
            return pts[-1]  # highest iteration

    raise FileNotFoundError("No model_*.pt checkpoints found in any wandb run directory.")


def build_dummy_obs() -> TensorDict:
    """Build a dummy obs TensorDict with the correct shapes for model construction."""
    return TensorDict(
        {
            "student_vec":        torch.zeros(1, VEC_DIM),
            "head_camera_depth":  torch.zeros(1, *IMG_DIM),
        },
        batch_size=[1],
    )


def build_model(student_state_dict: dict) -> CNNTransformerMDNModel:
    """Instantiate and load the CNNTransformerMDNModel student."""
    cfg = _make_student_mdn_model_cfg(stochastic=False)  # deterministic for export

    obs = build_dummy_obs()
    model = CNNTransformerMDNModel(
        obs=obs,
        obs_groups=OBS_GROUPS,
        obs_set="student",
        output_dim=ACTION_DIM,
        cnn_cfg={"head_camera_depth": _depth_cnn_cfg()},
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        obs_normalization=cfg.obs_normalization,
        stochastic=False,
        mdn_num_modes=cfg.mdn_num_modes,
        mdn_min_std=cfg.mdn_min_std,
        mdn_min_log_std=cfg.mdn_min_log_std,
        mdn_max_log_std=cfg.mdn_max_log_std,
        mdn_inference_mode=cfg.mdn_inference_mode,
    )

    # Load only the student weights.
    incompatible = model.load_state_dict(student_state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"[WARN] Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys: {incompatible.unexpected_keys}")

    model.eval()
    return model


class ONNXWrapper(nn.Module):
    """Thin wrapper that accepts two flat tensors and returns the action tensor.

    This makes the ONNX export have clean named I/O without TensorDict.
    """

    def __init__(self, model: CNNTransformerMDNModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        student_vec: torch.Tensor,        # [B, 96]
        head_camera_depth: torch.Tensor,   # [B, 1, 24, 32]
    ) -> torch.Tensor:                     # [B, 29]
        obs = TensorDict(
            {
                "student_vec": student_vec,
                "head_camera_depth": head_camera_depth,
            },
            batch_size=[student_vec.shape[0]],
        )
        return self.model(obs, masks=None, hidden_state=None, stochastic_output=False)


def export(checkpoint_path: pathlib.Path, output_path: pathlib.Path) -> None:
    print(f"[export] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "student_state_dict" not in ckpt:
        raise KeyError(
            f"Expected 'student_state_dict' in checkpoint, got keys: {list(ckpt.keys())}"
        )
    student_sd = ckpt["student_state_dict"]

    model = build_model(student_sd)
    wrapper = ONNXWrapper(model)
    wrapper.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_vec   = torch.zeros(1, VEC_DIM)
    dummy_depth = torch.zeros(1, 1, DEPTH_HEIGHT, DEPTH_WIDTH)

    print(f"[export] Exporting to: {output_path}")

    # Trace the model first — this uses TorchScript and bypasses the slow dynamo path.
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_vec, dummy_depth), strict=False)

    torch.onnx.export(
        traced,
        (dummy_vec, dummy_depth),
        str(output_path),
        export_params=True,
        opset_version=18,        # ORT 1.22 supports opset 18; skip downconversion
        do_constant_folding=True,
        input_names=["student_vec", "head_camera_depth"],
        output_names=["actions"],
        dynamic_axes={
            "student_vec":        {0: "batch"},
            "head_camera_depth":  {0: "batch"},
            "actions":            {0: "batch"},
        },
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"[export] Done — {output_path}  ({size_kb:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MDN student to ONNX.")
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None,
        help="Path to model_*.pt checkpoint. Auto-detected from latest wandb run if omitted.",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(EXPORT_DIR / "policy.onnx"),
        help="Output .onnx path.",
    )
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    out_path  = pathlib.Path(args.output)

    export(ckpt_path, out_path)


if __name__ == "__main__":
    main()
