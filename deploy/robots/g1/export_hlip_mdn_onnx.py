#!/usr/bin/env python3
"""Export an MDN distillation student checkpoint to ONNX.

Usage (from repo root):
    .venv/bin/python3 deploy/robots/g1/export_hlip_mdn_onnx.py \\
        --checkpoint logs/rsl_rl/g1_hlip_clf_distillation_mdn/2026-04-24_00-38-58/model_1370.pt

The exported ONNX has two named inputs:
    student_vec        [1, 480]       float32  (96 raw × 5 history)
    head_camera_depth  [1, 1, 24, 32] float32
and one output:
    actions            [1, 29]        float32  (MDN top-mode mean)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn
from tensordict import TensorDict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlip_clf_g1.rl.models.cnn_transformer_mdn_model import CNNTransformerMDNModel
from hlip_clf_g1.rl_cfg import _make_student_mdn_model_cfg, _depth_cnn_cfg

# ── Constants ────────────────────────────────────────────────────────────────
DEPTH_WIDTH  = 32
DEPTH_HEIGHT = 24
VEC_DIM      = 480   # 96 raw obs * 5 history frames (history_length=5)
ACTION_DIM   = 29

OBS_GROUPS = {"student": ("student_vec", "head_camera_depth")}

EXPORT_DIR = REPO_ROOT / "deploy/robots/g1/config/policy/hlip_mdn/v0/exported"


# ── Model construction ───────────────────────────────────────────────────────

def build_dummy_obs() -> TensorDict:
    return TensorDict(
        {
            "student_vec":       torch.zeros(1, VEC_DIM),
            "head_camera_depth": torch.zeros(1, 1, DEPTH_HEIGHT, DEPTH_WIDTH),
        },
        batch_size=[1],
    )


def build_model(student_state_dict: dict) -> CNNTransformerMDNModel:
    cfg = _make_student_mdn_model_cfg(stochastic=False)
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
    incompatible = model.load_state_dict(student_state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"[WARN] Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys: {incompatible.unexpected_keys}")
    model.eval()
    return model


# ── ONNX wrapper ─────────────────────────────────────────────────────────────
# Key design: we bypass TensorDict entirely inside the traced forward pass.
# Instead we call the model's sub-modules directly with plain tensors so that
# torch.jit.trace only sees native PyTorch ops.

class ONNXWrapper(nn.Module):
    """Wraps CNNTransformerMDNModel for ONNX export.

    Bypasses TensorDict by calling sub-modules directly with plain tensors.
    The forward path replicates CNNTransformerModel.get_latent() and
    CNNTransformerMDNModel.forward() without any Python-level dispatch.
    """

    def __init__(self, model: CNNTransformerMDNModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        student_vec: torch.Tensor,       # [B, 480]
        head_camera_depth: torch.Tensor, # [B, 1, 24, 32]
    ) -> torch.Tensor:                   # [B, 29]
        m = self.model

        # ── 1. Vector token ──────────────────────────────────────────────────
        # obs_normalizer is an EmpiricalNormalization or Identity — both
        # accept a plain tensor.
        vec_normed = m.obs_normalizer(student_vec)
        vec_token = m.vector_token_projection(vec_normed)           # [B, d_model]

        # ── 2. CNN token ─────────────────────────────────────────────────────
        cnn_feat = m.cnns["head_camera_depth"](head_camera_depth)   # [B, cnn_out]
        cnn_token = m.cnn_token_projections["head_camera_depth"](cnn_feat)  # [B, d_model]

        # ── 3. Transformer ───────────────────────────────────────────────────
        tokens = torch.stack([vec_token, cnn_token], dim=1)         # [B, 2, d_model]
        if m.positional_embedding is not None:
            tokens = tokens + m.positional_embedding
        encoded = m.transformer(tokens)                             # [B, 2, d_model]
        pooled = encoded.mean(dim=1)                                # [B, d_model]
        latent = m.transformer_norm(pooled)

        # ── 4. MLP → MDN params ──────────────────────────────────────────────
        raw_params = m.mlp(latent)                                  # [B, k + 2*k*a]
        logits, means, _stds = m._split_mdn_params(raw_params)

        # ── 5. Deterministic action (top-mode mean) ───────────────────────────
        return m._deterministic_action(logits, means)               # [B, 29]


# ── Export ───────────────────────────────────────────────────────────────────

def export(checkpoint_path: pathlib.Path, output_path: pathlib.Path) -> None:
    print(f"[export] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "student_state_dict" not in ckpt:
        raise KeyError(
            f"Expected 'student_state_dict' in checkpoint, got: {list(ckpt.keys())}"
        )

    model = build_model(ckpt["student_state_dict"])
    wrapper = ONNXWrapper(model)
    wrapper.eval()

    dummy_vec   = torch.zeros(1, VEC_DIM)
    dummy_depth = torch.zeros(1, 1, DEPTH_HEIGHT, DEPTH_WIDTH)

    # Sanity-check: one forward pass before export
    with torch.no_grad():
        out = wrapper(dummy_vec, dummy_depth)
    print(f"[export] Forward pass OK — output shape: {out.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] Exporting to: {output_path}")

    torch.onnx.export(
        wrapper,
        (dummy_vec, dummy_depth),
        str(output_path),
        export_params=True,
        opset_version=18,           # ORT 1.22 supports opset 18 natively
        do_constant_folding=True,
        input_names=["student_vec", "head_camera_depth"],
        output_names=["actions"],
        dynamic_axes={
            "student_vec":       {0: "batch"},
            "head_camera_depth": {0: "batch"},
            "actions":           {0: "batch"},
        },
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[export] Done — {output_path.name}  ({size_mb:.2f} MB)")

    # Verify with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(output_path))
        feeds = {
            "student_vec":       np.zeros((1, VEC_DIM),             dtype=np.float32),
            "head_camera_depth": np.zeros((1, 1, DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.float32),
        }
        result = sess.run(["actions"], feeds)[0]
        print(f"[export] ORT verification OK — actions shape: {result.shape}")
    except ImportError:
        print("[export] onnxruntime not available for verification (OK).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MDN student to ONNX.")
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None,
        help="Path to model_*.pt checkpoint.",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(EXPORT_DIR / "policy.onnx"),
        help="Output .onnx path.",
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        parser.error("--checkpoint is required.")

    export(pathlib.Path(args.checkpoint), pathlib.Path(args.output))


if __name__ == "__main__":
    main()
