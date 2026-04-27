"""Export the most-recent CNNTransformerMDN distillation student to ONNX.

Usage
-----
From the repo root (with the project venv active):

  python scripts/export_mdn_to_onnx.py \
      --checkpoint logs/rsl_rl/g1_hlip_clf_distillation_mdn/2026-04-23_22-00-24/model_140.pt \
      --out      deploy/robots/g1/config/policy/hlip_mdn/v0/exported/policy.onnx

The script reconstructs the student model from the saved config, loads the
student_state_dict, then traces the model with a representative dummy input and
exports it as ONNX.

The ONNX graph exposes **two named inputs**:
  • ``student_vec``       – shape [1, VEC_DIM]   (float32)  vector observations
  • ``head_camera_depth`` – shape [1, C, H, W]   (float32)  depth image

and one named output:
  • ``actions``           – shape [1, ACTION_DIM] (float32)

The ``OrtRunner`` C++ class dispatches inputs by name, so the names above must
match the observation group keys in the deploy YAML exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Make sure the src packages are importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── CNN configuration (must match training) ───────────────────────────────────

CNN_CFG = {
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

# ── Observation / action dimensions ───────────────────────────────────────────
#
# student_vec = base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
#               + joint_pos_rel(29) + joint_vel_rel(29) + last_action(29)
#             = 96 dimensions
#
# head_camera_depth = 1 channel × 24 H × 32 W   (depth in metres, float32)
#
# actions (all 29 joints) = 29 dimensions

VEC_DIM        = 96       # student_vec observation vector size
CAM_CHANNELS   = 1        # depth image channels (single-channel float32)
CAM_HEIGHT     = 24       # DepthImage_.height
CAM_WIDTH      = 32       # DepthImage_.width
ACTION_DIM     = 29       # all joints controlled by the distillation student
MDN_NUM_MODES  = 2        # matches mdn_num_modes in training config


def _build_dummy_obs(device: torch.device) -> "TensorDict":  # noqa: F821
    """Return a minimal TensorDict matching the student observation signature."""
    from tensordict import TensorDict

    return TensorDict(
        {
            "student_vec": torch.zeros(1, VEC_DIM, device=device),
            "head_camera_depth": torch.zeros(
                1, CAM_CHANNELS, CAM_HEIGHT, CAM_WIDTH, device=device
            ),
        },
        batch_size=[1],
        device=device,
    )


def _build_obs_groups() -> dict[str, list[str]]:
    return {
        "student": ["student_vec", "head_camera_depth"],
    }


def _build_student_model(device: torch.device):
    """Instantiate CNNTransformerMDNModel with the training hyper-parameters."""
    from tensordict import TensorDict
    from hlip_clf_g1.rl.models.cnn_transformer_mdn_model import CNNTransformerMDNModel

    obs_groups = _build_obs_groups()
    dummy_obs   = _build_dummy_obs(device)

    model = CNNTransformerMDNModel(
        obs          = dummy_obs,
        obs_groups   = obs_groups,
        obs_set      = "student",
        output_dim   = ACTION_DIM,
        cnn_cfg      = {"head_camera_depth": CNN_CFG},
        hidden_dims  = (512, 256, 128),
        activation   = "elu",
        obs_normalization = True,
        stochastic   = False,          # deterministic at inference
        mdn_num_modes   = MDN_NUM_MODES,
        mdn_min_std     = 1.0e-3,
        mdn_min_log_std = -3.0,
        mdn_max_log_std =  2.0,
        mdn_inference_mode = "top_mode_mean",
    )
    return model.to(device).eval()


class _DirectWrapper(torch.nn.Module):
    """Thin wrapper that calls CNNTransformerMDN internals with plain tensors.

    Bypasses TensorDict entirely so torch.jit.trace/torch.onnx.export
    don't get stuck on TensorDict's custom C++ dispatch hooks.

    Assumes:
      obs_groups  = {"student": ["student_vec", "head_camera_depth"]}
      obs_groups_1d  contains only "student_vec"
      obs_groups_2d  contains only "head_camera_depth"
    """

    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(
        self,
        student_vec: torch.Tensor,      # [B, VEC_DIM]
        head_camera_depth: torch.Tensor, # [B, C, H, W]
    ) -> torch.Tensor:
        s = self.student

        # ── 1. Vector token ─────────────────────────────────────────────────
        # obs_normalizer is applied to the raw concatenated 1-D observations.
        latent_1d = s.obs_normalizer(student_vec)
        vec_token  = s.vector_token_projection(latent_1d)   # [B, d_model]

        # ── 2. CNN token ─────────────────────────────────────────────────────
        cnn_key   = s.obs_groups_2d[0]                      # "head_camera_depth"
        cnn_feat  = s.cnns[cnn_key](head_camera_depth)      # [B, cnn_out]
        img_token = s.cnn_token_projections[cnn_key](cnn_feat)  # [B, d_model]

        # ── 3. Transformer ───────────────────────────────────────────────────
        # token order must match training: [vec_token, img_token]
        token_tensor = torch.stack([vec_token, img_token], dim=1)  # [B, 2, d_model]
        if s.positional_embedding is not None:
            token_tensor = token_tensor + s.positional_embedding
        encoded = s.transformer(token_tensor)                      # [B, 2, d_model]
        pooled  = s.transformer_norm(encoded.mean(dim=1))          # [B, d_model]

        # ── 4. MDN head ──────────────────────────────────────────────────────
        raw_params   = s.mlp(pooled)
        logits, means, _ = s._split_mdn_params(raw_params)
        return s._deterministic_action(logits, means)              # [B, ACTION_DIM]


def export(checkpoint: Path, out: Path) -> None:
    device = torch.device("cpu")

    print(f"[export] Loading checkpoint: {checkpoint}")
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)

    if "student_state_dict" not in loaded:
        raise KeyError(
            "'student_state_dict' not found in checkpoint. "
            f"Available keys: {list(loaded.keys())}"
        )

    student = _build_student_model(device)
    missing, unexpected = student.load_state_dict(
        loaded["student_state_dict"], strict=True
    )
    if missing:
        print(f"[WARN] Missing keys  : {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    # Switch BatchNorm layers to eval mode (important for ONNX trace)
    student.eval()

    wrapper = _DirectWrapper(student).eval()

    dummy_vec   = torch.zeros(1, VEC_DIM,    device=device)
    dummy_depth = torch.zeros(1, CAM_CHANNELS, CAM_HEIGHT, CAM_WIDTH, device=device)

    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_vec, dummy_depth), strict=False)

    print(f"[export] Exporting ONNX → {out}")
    # Use dynamic_axes so the ONNX graph records batch=-1.
    # OrtRunner now handles -1 dims correctly (substitutes 1 at runtime).
    torch.onnx.export(
        traced,
        (dummy_vec, dummy_depth),
        str(out),
        input_names=["student_vec", "head_camera_depth"],
        output_names=["actions"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes={
            "student_vec":       {0: "batch"},
            "head_camera_depth": {0: "batch"},
            "actions":           {0: "batch"},
        },
    )

    print("[export] Done.")
    _verify(out, dummy_vec, dummy_depth)




def _verify(onnx_path: Path, dummy_vec, dummy_depth) -> None:
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feeds = {
            "student_vec":        dummy_vec.numpy(),
            "head_camera_depth":  dummy_depth.numpy(),
        }
        out = sess.run(None, feeds)
        print(f"[verify] ONNX output shape: {out[0].shape}  (expected [1, {ACTION_DIM}])")
        assert out[0].shape == (1, ACTION_DIM), "Unexpected output shape!"
        print("[verify] ✓ ONNX model verified successfully.")
    except ImportError:
        print("[verify] onnxruntime not installed — skipping verification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "logs/rsl_rl/g1_hlip_clf_distillation_mdn"
            "/2026-04-23_22-00-24/model_140.pt"
        ),
        help="Path to the distillation .pt checkpoint (student_state_dict).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "deploy/robots/g1/config/policy/hlip_mdn/v0/exported/policy.onnx"
        ),
        help="Output ONNX file path.",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
