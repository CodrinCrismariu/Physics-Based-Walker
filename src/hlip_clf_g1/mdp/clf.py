"""Control Lyapunov Function (CLF) evaluator.

Computes the CLF value V = eta^T P eta and its time derivative Vdot for
trajectory tracking with stability guarantees.

The system is modelled as a stack of double integrators (one per output
channel). The continuous-time Algebraic Riccati Equation (ARE) is solved
once at construction using SciPy, and the resulting P matrix is cached as
a PyTorch tensor for efficient GPU evaluation.

Ported from planc/tasks/.../clf_cmd/clf.py to work with the mjlab
framework (no Isaac Lab dependencies).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_continuous_are


class CLF:
  """Continuous-time CLF evaluator for relative-degree-2 outputs.

  Args:
    n_outputs: Number of tracked output channels.
    sim_dt: Simulation time step (after decimation).
    batch_size: Number of parallel environments.
    Q_weights: Weight array of length ``2 * n_outputs`` (pos/vel pairs).
    R_weights: Weight array of length ``n_outputs``.
    device: Torch device.
  """

  def __init__(
    self,
    n_outputs: int,
    sim_dt: float,
    batch_size: int,
    Q_weights: np.ndarray | list[float] | None = None,
    R_weights: np.ndarray | list[float] | None = None,
    device: torch.device | str | None = None,
  ):
    self.device = torch.device(device) if device is not None else torch.device("cpu")
    self.sim_dt = sim_dt
    self.n_outputs = n_outputs

    n_states = 2 * n_outputs
    n_inputs = n_outputs

    # Build Q diagonal.
    if Q_weights is None:
      Q_diag = np.ones(n_states)
    else:
      Q_diag = np.asarray(Q_weights, dtype=np.float64)
      assert Q_diag.shape[0] == n_states, (
        f"Q_weights must have {n_states} entries, got {Q_diag.shape[0]}"
      )

    # Build R diagonal.
    if R_weights is None:
      R_diag = 0.1 * np.ones(n_inputs)
    else:
      R_diag = np.asarray(R_weights, dtype=np.float64)
      assert R_diag.shape[0] == n_inputs, (
        f"R_weights must have {n_inputs} entries, got {R_diag.shape[0]}"
      )

    Q_np = np.diag(Q_diag)
    R_np = np.diag(R_diag)

    # Build block-diagonal double-integrator system.
    A_blk = np.array([[0.0, 1.0], [0.0, 0.0]])
    B_blk = np.array([[0.0], [1.0]])
    A_full = np.kron(np.eye(n_outputs), A_blk)
    B_full = np.kron(np.eye(n_outputs), B_blk)

    # Solve the continuous-time ARE.
    P_np = solve_continuous_are(A_full, B_full, Q_np, R_np)

    self.P = torch.from_numpy(P_np).to(self.device).to(torch.float32)
    self.lambda_max = torch.linalg.eigvalsh(self.P)[-1]
    self.norm_P = torch.linalg.norm(self.P, ord=2)

    # History buffer for 3-point backward-difference Vdot.
    self.v_buffer = torch.zeros((batch_size, 3), device=self.device)
    self.step_count = 0

  def reset(self, env_ids: torch.Tensor) -> None:
    """Reset CLF buffers for specific environments."""
    self.v_buffer[env_ids] = 0.0

  def compute_v(
    self,
    y_act: torch.Tensor,
    y_nom: torch.Tensor,
    dy_act: torch.Tensor,
    dy_nom: torch.Tensor,
    yaw_idx: list[int] | None = None,
  ) -> torch.Tensor:
    """Evaluate V = eta^T P eta.

    Args:
      y_act: Actual output positions. Shape ``(batch, n_outputs)``.
      y_nom: Reference output positions. Shape ``(batch, n_outputs)``.
      dy_act: Actual output velocities. Shape ``(batch, n_outputs)``.
      dy_nom: Reference output velocities. Shape ``(batch, n_outputs)``.
      yaw_idx: Indices of yaw-like outputs that need angle wrapping.

    Returns:
      V: CLF value. Shape ``(batch,)``.
    """
    y_err = y_act - y_nom
    dy_err = dy_act - dy_nom

    # Wrap yaw errors to [-pi, pi].
    if yaw_idx:
      yaw_err = y_err[:, yaw_idx]
      wrapped = (yaw_err + torch.pi) % (2.0 * torch.pi) - torch.pi
      y_err[:, yaw_idx] = wrapped

    # Interleave position and velocity errors: [pos0, vel0, pos1, vel1, ...].
    batch = y_act.shape[0]
    eta = torch.zeros(batch, 2 * self.n_outputs, device=y_act.device)
    eta[:, 0::2] = y_err
    eta[:, 1::2] = dy_err

    V = torch.einsum("bi,ij,bj->b", eta, self.P, eta)

    # Shift history buffer.
    self.v_buffer[:, 2] = self.v_buffer[:, 1]
    self.v_buffer[:, 1] = self.v_buffer[:, 0]
    self.v_buffer[:, 0] = V.detach()

    self.step_count += 1
    return V

  def compute_vdot(
    self,
    y_act: torch.Tensor,
    y_nom: torch.Tensor,
    dy_act: torch.Tensor,
    dy_nom: torch.Tensor,
    yaw_idx: list[int] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Vdot via finite differences and return (Vdot, V).

    Uses a 3-point backward difference when enough history is available.

    Args:
      y_act, y_nom, dy_act, dy_nom, yaw_idx: Same as ``compute_v``.

    Returns:
      (vdot, v_curr).
    """
    v_curr = self.compute_v(y_act, y_nom, dy_act, dy_nom, yaw_idx)
    dt = self.sim_dt
    B = v_curr.shape[0]

    if self.step_count >= 3:
      V_k = self.v_buffer[:, 0]
      V_k1 = self.v_buffer[:, 1]
      V_k2 = self.v_buffer[:, 2]
      vdot_raw = (3.0 * V_k - 4.0 * V_k1 + V_k2) / (2.0 * dt)
    elif self.step_count == 2:
      vdot_raw = (self.v_buffer[:, 0] - self.v_buffer[:, 1]) / dt
    else:
      vdot_raw = torch.zeros(B, device=self.device)

    return vdot_raw, v_curr
