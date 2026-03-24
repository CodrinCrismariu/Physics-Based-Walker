"""HLIP reference generation utilities.

Provides the Hybrid Linear Inverted Pendulum (HLIP) class for capture-point
based footstep planning, and Bezier curve utilities for smooth swing foot
trajectories.

Ported from planc/tasks/manager_based/robot_rl/mdp/commands/ref_gen.py to
work with the mjlab framework (pure PyTorch, no Isaac Lab dependencies).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# =====================================================================
# Bezier curve utilities
# =====================================================================


def _ncr(n: int, r: int) -> int:
  """Combination formula for Bezier coefficients."""
  return math.comb(n, r)


def bezier_deg(
  order: int,
  tau: torch.Tensor,
  step_dur: torch.Tensor,
  control_points: torch.Tensor,
  degree: int,
) -> torch.Tensor:
  """Compute Bezier curve value (order=0) or its time-derivative (order=1).

  Args:
    order: 0 for position, 1 for velocity.
    tau: Phase in [0, 1]. Shape ``(batch,)``.
    step_dur: Step duration. Shape ``(batch,)``.
    control_points: Shape ``(batch, degree+1)``.
    degree: Polynomial degree.

  Returns:
    Bezier value. Shape ``(batch,)``.
  """
  tau = torch.clamp(tau, 0.0, 1.0)

  if order == 1:
    # First derivative of Bezier curve.
    cp_diff = control_points[:, 1:] - control_points[:, :-1]  # (batch, degree)
    coefs = torch.tensor(
      [_ncr(degree - 1, i) for i in range(degree)],
      dtype=control_points.dtype,
      device=control_points.device,
    )
    i = torch.arange(degree, device=control_points.device)
    tau_pow = tau.unsqueeze(1) ** i.unsqueeze(0)
    one_minus_pow = (1 - tau).unsqueeze(1) ** (degree - 1 - i).unsqueeze(0)
    terms = degree * cp_diff * coefs.unsqueeze(0) * one_minus_pow * tau_pow
    return terms.sum(dim=1) / step_dur
  else:
    # Position on Bezier curve.
    coefs = torch.tensor(
      [_ncr(degree, i) for i in range(degree + 1)],
      dtype=control_points.dtype,
      device=control_points.device,
    )
    i = torch.arange(degree + 1, device=control_points.device)
    tau_pow = tau.unsqueeze(1) ** i.unsqueeze(0)
    one_minus_pow = (1 - tau).unsqueeze(1) ** (degree - i).unsqueeze(0)
    terms = control_points * coefs.unsqueeze(0) * one_minus_pow * tau_pow
    return terms.sum(dim=1)


def calculate_cur_swing_foot_pos(
  bht: torch.Tensor,
  z_init: torch.Tensor,
  z_sw_max: torch.Tensor,
  tau: torch.Tensor,
  step_x_init: torch.Tensor,
  step_y_init: torch.Tensor,
  T_gait: torch.Tensor,
  zsw_neg: torch.Tensor,
  clipped_step_x: torch.Tensor,
  clipped_step_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Batch-friendly swing foot position via Bezier curves.

  Returns:
    p_swing: Swing foot position. Shape ``(batch, 3)``.
    v_swing_z: Swing foot vertical velocity. Shape ``(batch, 1)``.
  """
  degree_v = 6
  control_v = torch.stack(
    [
      z_init,
      z_init + 0.2 * (z_sw_max - z_init),
      z_init + 0.6 * (z_sw_max - z_init),
      z_sw_max,
      zsw_neg + 0.5 * (z_sw_max - zsw_neg),
      zsw_neg + 0.05 * (z_sw_max - zsw_neg),
      zsw_neg,
    ],
    dim=1,
  )

  # Horizontal: linear interpolation.
  p_swing_x = ((1 - bht) * step_x_init + bht * clipped_step_x).unsqueeze(1)
  p_swing_y = ((1 - bht) * step_y_init + bht * clipped_step_y).unsqueeze(1)

  # Vertical: 6th-degree Bezier.
  p_swing_z = bezier_deg(0, tau, T_gait, control_v, degree_v).unsqueeze(1)
  v_swing_z = bezier_deg(1, tau, T_gait, control_v, degree_v).unsqueeze(1)

  return torch.cat([p_swing_x, p_swing_y, p_swing_z], dim=1), v_swing_z


# =====================================================================
# HLIP (Hybrid Linear Inverted Pendulum)
# =====================================================================


class HLIP(torch.nn.Module):
  """Hybrid Linear Inverted Pendulum for capture-point footstep planning.

  Computes desired orbits and CoM trajectories using the LIP dynamics.
  All tensors are created on the same device as the inputs at runtime.
  """

  def __init__(
    self,
    grav: float,
    z0: float,
    T_ds: float,
    T: float,
    y_nom: float,
    device: torch.device | str = "cpu",
  ):
    super().__init__()
    self.grav = grav
    self.z0 = z0
    self.y_nom = y_nom
    self.T_ds = T_ds
    self.T = T

    dev = torch.device(device)
    self.lambda_ = torch.sqrt(
      torch.tensor(grav / z0, device=dev, dtype=torch.float32)
    )

    self.A_ss = torch.tensor(
      [[0.0, 1.0], [grav / z0, 0.0]], device=dev, dtype=torch.float32
    )
    self.A_ds = torch.tensor(
      [[0.0, 1.0], [0.0, 0.0]], device=dev, dtype=torch.float32
    )
    self.B_usw = torch.tensor([-1.0, 0.0], device=dev, dtype=torch.float32)

    self.A_s2s: torch.Tensor | None = None
    self.B_s2s: torch.Tensor | None = None
    self._compute_s2s_matrices()

    # Buffers set by compute_orbit.
    self.x_init: torch.Tensor | None = None
    self.y_init: torch.Tensor | None = None

  def _compute_s2s_matrices(self) -> None:
    exp_ss = torch.matrix_exp(self.A_ss * (self.T - self.T_ds))
    exp_ds = torch.matrix_exp(self.A_ds * self.T_ds)
    self.A_s2s = exp_ss @ exp_ds
    self.B_s2s = exp_ss @ self.B_usw

  def _remap_for_init_stance_state(
    self,
    X_des_p1: Tensor,
    Y_des_p2: Tensor,
    Ux: Tensor,
    Uy: Tensor,
  ) -> tuple[Tensor, Tensor]:
    Y_left = torch.cat(
      [
        (Y_des_p2[:, 1, 0] - Uy[:, 1]).unsqueeze(-1),
        Y_des_p2[:, 1, 1].unsqueeze(-1),
      ],
      dim=-1,
    )
    Y_right = torch.cat(
      [
        (Y_des_p2[:, 0, 0] - Uy[:, 0]).unsqueeze(-1),
        Y_des_p2[:, 0, 1].unsqueeze(-1),
      ],
      dim=-1,
    )
    X0 = torch.cat(
      [
        (X_des_p1[:, 0] - Ux).unsqueeze(-1),
        X_des_p1[:, 1].unsqueeze(-1),
      ],
      dim=-1,
    )
    return X0, torch.cat(
      [Y_left.unsqueeze(1), Y_right.unsqueeze(1)], dim=1
    )

  def _compute_desire_com_trajectory(
    self, cur_time: torch.Tensor, Xdesire: Tensor
  ) -> tuple[Tensor, Tensor]:
    """Desired CoM trajectory relative to stance foot.

    Args:
      cur_time: Current time within the step. Shape ``(batch,)`` or scalar.
      Xdesire: Initial position and velocity. Shape ``(batch, 2)``.

    Returns:
      (position, velocity), each shape ``(batch,)``.
    """
    x0, v0 = Xdesire[:, 0], Xdesire[:, 1]
    lam = self.lambda_
    pos = x0 * torch.cosh(lam * cur_time) + (v0 / lam) * torch.sinh(
      lam * cur_time
    )
    vel = x0 * lam * torch.sinh(lam * cur_time) + v0 * torch.cosh(
      lam * cur_time
    )
    return pos, vel

  def compute_desired_orbit(
    self, vel: Tensor, T: Tensor
  ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the desired HLIP orbit.

    Args:
      vel: Commanded velocity ``(vx, vy)``. Shape ``(batch, 2)``.
      T: Gait half-period. Shape ``(batch,)`` or scalar.

    Returns:
      (X_des_p1, U_des_p1, Y_des_stacked, U_y_stacked).
    """
    device = vel.device
    batch = vel.shape[0]

    # Use analytical formulas for batched exp_ss and exp_ds
    lam = self.lambda_
    T_ss = T - self.T_ds
    
    cosh_t = torch.cosh(lam * T_ss)
    sinh_t = torch.sinh(lam * T_ss)

    A_exp_ss = torch.zeros(batch, 2, 2, device=device)
    A_exp_ss[:, 0, 0] = cosh_t
    A_exp_ss[:, 0, 1] = sinh_t / lam
    A_exp_ss[:, 1, 0] = lam * sinh_t
    A_exp_ss[:, 1, 1] = cosh_t

    if self.T_ds > 0:
      A_exp_ds = torch.zeros(batch, 2, 2, device=device)
      A_exp_ds[:, 0, 0] = 1.0
      A_exp_ds[:, 0, 1] = self.T_ds
      A_exp_ds[:, 1, 1] = 1.0
      A_s2s = torch.bmm(A_exp_ss, A_exp_ds)
    else:
      A_s2s = A_exp_ss

    B_usw = self.B_usw.to(device)
    B_s2s = torch.matmul(A_exp_ss, B_usw)

    U_des_p1 = vel[:, 0] * T

    eye = torch.eye(2, device=device).unsqueeze(0).expand(batch, -1, -1)
    X_des_p1 = torch.linalg.solve(
      eye - A_s2s, B_s2s * U_des_p1.unsqueeze(-1)
    ).squeeze(-1)

    U_left = vel[:, 1] * T - self.y_nom
    U_right = vel[:, 1] * T + self.y_nom

    A_squared = torch.bmm(A_s2s, A_s2s)
    B_term = torch.bmm(A_s2s, B_s2s.unsqueeze(-1)).squeeze(-1)

    Y_left = torch.linalg.solve(
      eye - A_squared,
      B_term * U_left.unsqueeze(-1) + B_s2s * U_right.unsqueeze(-1),
    ).squeeze(-1)
    Y_right = torch.linalg.solve(
      eye - A_squared,
      B_term * U_right.unsqueeze(-1) + B_s2s * U_left.unsqueeze(-1),
    ).squeeze(-1)

    return (
      X_des_p1,
      U_des_p1,
      torch.stack([Y_left, Y_right], dim=1),
      torch.stack([U_left, U_right], dim=1),
    )

  def compute_orbit(self, T: Tensor, cmd: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute desired orbit and cache initial states.

    Args:
      T: Gait half-period. Shape ``(batch,)`` or scalar.
      cmd: Velocity command ``(vx, vy, ...)``. Shape ``(batch, >=2)``.

    Returns:
      (Xdes, Ux, Ydes, Uy).
    """
    Xdes, Ux, Ydes, Uy = self.compute_desired_orbit(cmd[:, :2], T)
    self.x_init, self.y_init = self._remap_for_init_stance_state(
      Xdes, Ydes, Ux, Uy
    )
    return Xdes, Ux, Ydes, Uy
