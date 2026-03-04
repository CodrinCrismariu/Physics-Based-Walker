# HLIP + CLF Walking Task

A full-body bipedal walking controller that combines the **Hybrid Linear Inverted Pendulum (HLIP)** model for footstep planning with a **Control Lyapunov Function (CLF)** for stability guarantees. An RL policy learns to track the HLIP-generated reference trajectories while satisfying the CLF decrease condition.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [HLIP Reference Generation](#hlip-reference-generation)
4. [CLF Stability Framework](#clf-stability-framework)
5. [Observations](#observations)
6. [Actions](#actions)
7. [Rewards](#rewards)
8. [Events & Domain Randomization](#events--domain-randomization)
9. [Terminations](#terminations)
10. [Gait Phase Tracking](#gait-phase-tracking)
11. [Training](#training)
12. [File Structure](#file-structure)

---

## Overview

The HLIP task trains a walking policy for the Unitree G1 humanoid (29 DoF, 21 actuated). The approach is:

1. **Plan** footsteps using the capture-point dynamics of a Linear Inverted Pendulum.
2. **Generate** smooth full-body reference trajectories (CoM, pelvis, swing foot, upper body) each timestep.
3. **Evaluate** tracking quality using a CLF that models the system as a set of coupled double integrators.
4. **Reward** the RL agent for minimising the CLF value $V$ and satisfying the decrease condition $\dot{V} + \alpha V \leq 0$.

The policy only controls the **lower body** (12 joints: 6 per leg). Upper body joints (shoulders, elbows, wrists) and all waist joints (yaw, roll, pitch) are **locked** at their default positions.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        HLIPCommandTerm                          │
│                                                                  │
│  vel_command ──► HLIP Planner ──► Foot Target (capture point)    │
│       │              │                    │                       │
│       │              ▼                    ▼                       │
│       │     CoM ref (LIP dynamics)   Swing Foot ref (Bezier)     │
│       │              │                    │                       │
│       │              ▼                    ▼                       │
│       └──► Orientation ref ──► Upper Body ref ──► y_out, dy_out  │
│                                                       │          │
│  Robot sensors ──────────────────────────────► y_act, dy_act     │
│                                                       │          │
│  CLF(y_out, y_act, dy_out, dy_act) ──────────► V, Vdot          │
└──────────────────────────────────────────────────────────────────┘
        │                    │                      │
        ▼                    ▼                      ▼
   Observations          Rewards              Critic Obs
        │                    │                      │
        ▼                    ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PPO Policy (RSL-RL)                          │
│                                                                  │
│  Actor:  obs ──► [512, 256, 128] ──► 12-dim joint pos actions    │
│  Critic: obs ──► [512, 256, 128] ──► value estimate              │
└──────────────────────────────────────────────────────────────────┘
```

---

## HLIP Reference Generation

The reference trajectory is computed every control step (0.02 s) and contains **21 output channels**:

| Index | Output | Description |
|-------|--------|-------------|
| 0–2 | CoM position | $(x, y, z)$ in stance-foot-local frame |
| 3–5 | Pelvis orientation | $(roll, pitch, yaw)$ Euler angles |
| 6–8 | Swing foot position | $(x, y, z)$ in stance-foot-local frame |
| 9–11 | Swing foot orientation | $(roll, pitch, yaw)$ Euler angles |
| 12 | Waist yaw | Joint position reference |
| 13–20 | Upper body | Shoulder pitch/roll/yaw + elbow (L/R) |

Each output has a corresponding velocity reference, giving a total state vector of 42 dimensions (21 pos + 21 vel).

### Capture-Point Footstep Planning

The HLIP controller uses the LIP dynamics:

$$\ddot{x} = \frac{g}{z_0}(x - u_{sw})$$

where $z_0 = 0.67$ m is the nominal CoM height and $u_{sw}$ is the swing foot landing position. The step-to-step (S2S) map:

$$\mathbf{x}_{k+1} = A_{s2s}\, \mathbf{x}_k + B_{s2s}\, u_k$$

is computed by matrix-exponentiating the continuous dynamics over the gait half-period $T = 0.4$ s. The desired orbit is found by solving the fixed-point equation $\mathbf{x}^* = (I - A_{s2s})^{-1} B_{s2s}\, u^*$ for the commanded velocity.

### CoM Trajectory

Within each step, the CoM follows the LIP closed-form solution:

$$x(t) = x_0 \cosh(\lambda t) + \frac{\dot{x}_0}{\lambda}\sinh(\lambda t), \quad \lambda = \sqrt{g/z_0}$$

### Swing Foot Trajectory

- **Horizontal**: linear interpolation from the start position to the capture-point target, shaped by a 4th-degree Bezier timing curve `[0, 0, 1, 1, 1]`.
- **Vertical**: a 6th-degree Bezier curve that lifts the foot to `z_sw_max = 0.1 m` mid-swing and lands at `z_sw_min = 0.0 m`.

### Orientation References

- **Pelvis roll**: sinusoidal oscillation with lateral bias from $\arctan(v_y / g)$.
- **Pelvis pitch**: small sinusoidal oscillation around `pelv_pitch_ref`.
- **Pelvis yaw**: integrates the yaw-rate command from the stance foot heading.
- **Swing foot**: matches pelvis yaw.

### Upper Body

Sinusoidal arm swing with amplitude proportional to forward velocity. Shoulder pitch and elbow swing in anti-phase between left and right sides.

### Coordinate Frame

All reference positions/velocities are expressed in the **stance-foot-local frame** — a frame anchored at the stance foot position recorded at the moment of the last stance transition, rotated to match only the stance foot's yaw.

---

## CLF Stability Framework

The CLF models the 21 tracked outputs as independent double integrators:

$$A = \text{blkdiag}\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}_{21}, \quad B = \text{blkdiag}\begin{pmatrix} 0 \\ 1 \end{pmatrix}_{21}$$

### ARE Solution

The continuous-time Algebraic Riccati Equation:

$$A^T P + PA - PBR^{-1}B^T P + Q = 0$$

is solved once at initialisation using `scipy.linalg.solve_continuous_are`. The resulting $P \in \mathbb{R}^{42 \times 42}$ is cached as a PyTorch tensor.

### CLF Value

The tracking error is interleaved as $\eta = [e_{pos,0},\ e_{vel,0},\ e_{pos,1},\ e_{vel,1},\ \ldots]$ (42-dim), and:

$$V = \eta^T P\, \eta$$

### CLF Derivative

$\dot{V}$ is computed via **3-point backward finite difference** on the history of $V$ values:

$$\dot{V}_k \approx \frac{3V_k - 4V_{k-1} + V_{k-2}}{2\Delta t}$$

### Q and R Weights

The Q diagonal (42 entries) controls how aggressively each output and its velocity are penalised. Higher weights on swing foot position (1500–3500) vs. CoM (25–400) prioritise accurate foot placement. R weights (21 entries) are generally small (0.01–0.1), making the CLF responsive.

---

## Observations

### Policy Observations

| Term | Dim | Description |
|------|-----|-------------|
| `base_ang_vel` | 3 | IMU angular velocity (body frame), noise ±0.2 |
| `projected_gravity` | 3 | Gravity vector in body frame, noise ±0.05 |
| `joint_pos` | 29 | Joint positions relative to default, noise ±0.01 |
| `joint_vel` | 29 | Joint velocities × 0.05, noise ±1.0 |
| `actions` | 14 | Previous action |
| `sin_phase` | 1 | $\sin(2\pi \cdot t_p)$ |
| `cos_phase` | 1 | $\cos(2\pi \cdot t_p)$ |
| `contact_state` | 2 | Binary foot contact (left, right) |
| `ref_traj` | 21 | Reference positions $y_{out}$ |
| `act_traj` | 21 | Actual positions $y_{act}$ |
| `ref_traj_vel` | 21 | Reference velocities $\dot{y}_{out}$, clipped to ±20 |
| `act_traj_vel` | 21 | Actual velocities $\dot{y}_{act}$, clipped to ±20 |

**Total policy obs**: 166 dimensions.

### Critic Observations

The critic receives privileged information without sensor noise:

| Term | Dim | Description |
|------|-----|-------------|
| `base_lin_vel` | 3 | Root linear velocity (world frame) |
| `base_ang_vel` | 3 | Root angular velocity (body frame) |
| `root_quat` | 4 | Root quaternion $(w, x, y, z)$ |
| `projected_gravity` | 3 | Gravity in body frame |
| `velocity_commands` | 3 | Commanded $(v_x, v_y, \omega_z)$ × 2.0 |
| `joint_pos` | 29 | Joint positions relative to default |
| `joint_vel` | 29 | Joint velocities × 0.05 |
| `actions` | 14 | Previous action |
| `sin_phase` | 1 | Phase sine |
| `cos_phase` | 1 | Phase cosine |
| `contact_state` | 2 | Foot contact state |
| `ref_traj` | 21 | Reference trajectory positions |
| `act_traj` | 21 | Actual trajectory positions |
| `ref_traj_vel` | 21 | Reference velocities, clipped ±20 |
| `act_traj_vel` | 21 | Actual velocities, clipped ±20 |

**Total critic obs**: 176 dimensions.

---

## Actions

### Lower Body (RL-controlled) — 12 DoF

The policy outputs **12-dimensional** joint position targets processed as:

$$q_{target} = q_{default} + \text{scale} \times a$$

where `scale` is a per-joint dictionary derived from `0.25 × effort_limit / stiffness`:

| Joint Group | Scale | Count |
|-------------|-------|-------|
| `hip_pitch`, `hip_yaw` | 0.548 | 4 |
| `hip_roll`, `knee` | 0.351 | 4 |
| `ankle_pitch`, `ankle_roll` | 0.439 | 4 |

### Upper Body (Locked) — 17 DoF

Shoulders, elbows, wrists, and all waist joints (yaw, roll, pitch) are held at their default positions by directly overwriting `qpos` and `qvel` every step (zero action dimension). This matches planc's approach where the 21-DoF G1 model excludes `waist_pitch` and `waist_roll` entirely.

---

## Rewards

### CLF Rewards

| Reward | Weight | Formula |
|--------|--------|---------|
| `clf_reward` | +10.0 | $\exp(-V / V_{max})$ where $V_{max} = \lambda_{max}(P) \cdot 0.2^2$ |
| `clf_decreasing_condition` | −1.0 | $\text{clamp}((\dot{V} + 0.5V) / V_{max\_viol},\ 0,\ 1)$ |

The CLF reward exponentially rewards small tracking error. The decreasing condition penalises when the Lyapunov function is not decaying fast enough ($\alpha = 0.5$).

### Holonomic Constraints

| Reward | Weight | Formula |
|--------|--------|---------|
| `holonomic_constraint` | +4.0 | $\exp(-\|e_{pose}\|^2 / 0.05)$ where $e_{pose} = [dx, dy, dz, roll, \Delta\psi]$ |
| `holonomic_constraint_vel` | +2.0 | $\exp(-\|[v_{stance}, \omega_z]\|^2 / 0.1)$ |

These enforce that the stance foot stays planted — no sliding, no rotation.

### Regularisation

| Reward | Weight | Description |
|--------|--------|-------------|
| `joint_torques_l2` | −1e-5 | Minimise joint torques |
| `joint_pos_limits` | −1.0 | Penalise approaching joint limits |
| `action_rate_l2` | −0.01 | Smooth actions |
| `self_collisions` | −1.0 | Self-collision penalty (G1-specific) |

---

## Events & Domain Randomization

| Event | Mode | Description |
|-------|------|-------------|
| `reset_base` | reset | Randomise base pose: xy ±0.5 m, yaw ±π |
| `reset_robot_joints` | reset | Reset joints to default (no offset) |
| `push_robot` | interval (10–15 s) | Random velocity push: xy ±1.0 m/s, angular ±0.4 rad/s |
| `body_friction` | startup | Randomise friction coefficients [0.3, 1.2] |
| `encoder_bias` | startup | Joint encoder bias ±0.015 rad |
| `base_com` | startup | Randomise torso CoM offset: xy ±0.05 m, z ±0.01 m |

---

## Terminations

| Condition | Description |
|-----------|-------------|
| `time_out` | Episode length exceeds 20 seconds |
| `fell_over` | Body tilts more than 50° from upright |
| `base_height` | CoM drops below 0.25 m |

---

## Gait Phase Tracking

The gait is parameterised by a normalised phase $t_p \in [0, 1)$:

- $t_p \in [0, 0.5)$: **left stance** (left foot planted, right foot swinging)
- $t_p \in [0.5, 1.0)$: **right stance** (right foot planted, left foot swinging)

Full gait period = $2T = 0.8$ s, giving a stride frequency of 1.25 Hz.

At each stance transition, the command term records:
1. Stance foot world position and orientation (anchor for the local frame)
2. Swing foot position in the new local frame (start point for Bezier trajectory)

---

## Training

### Command

```bash
python scripts/train.py Mjlab-HLIP-CLF-Unitree-G1 --env.scene.num_envs 4096
```

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Actor/Critic hidden dims | [512, 256, 128] |
| Activation | ELU |
| Learning rate | 1e-3 (adaptive schedule) |
| Entropy coefficient | 0.008 |
| Steps per env | 24 |
| Mini-batches | 4 |
| Learning epochs | 5 |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Clip param | 0.2 |
| Max iterations | 10,001 |
| Observation normalisation | Actor + Critic |

### Simulation

| Parameter | Value |
|-----------|-------|
| Physics timestep | 0.005 s |
| Decimation | 4 |
| Control dt | 0.02 s (50 Hz) |
| MuJoCo iterations | 10 |
| MuJoCo LS iterations | 20 |

### Play / Evaluate

```bash
python scripts/play.py Mjlab-HLIP-CLF-Unitree-G1 --checkpoint-file=<path_to_model.pt>
```

---

## File Structure

```
mjlab/tasks/hlip/
├── __init__.py                  # Task registration
├── hlip_env_cfg.py              # Factory: make_hlip_env_cfg() — base config
├── README.md                    # This file
│
├── config/
│   └── g1/
│       ├── env_cfgs.py          # G1-specific overrides (sensors, action scales)
│       └── rl_cfg.py            # PPO runner hyperparameters
│
├── mdp/
│   ├── __init__.py              # Re-exports all MDP functions
│   ├── hlip_command.py          # HLIPCommandTerm — reference generation + CLF
│   ├── clf.py                   # CLF class — ARE solver, V/Vdot computation
│   ├── ref_gen.py               # HLIP planner + Bezier swing foot utilities
│   ├── observations.py          # HLIP-specific observations
│   └── rewards.py               # CLF, holonomic, tracking, phase rewards
│
└── rl/                          # (empty / future extensions)
```

### Key Classes

- **`HLIPCommandTerm`** ([hlip_command.py](mdp/hlip_command.py)): The central orchestrator. Every step it advances the gait phase, generates reference trajectories, extracts actual state, and computes CLF V/Vdot.

- **`CLF`** ([clf.py](mdp/clf.py)): Solves the continuous-time ARE once, then efficiently evaluates $V = \eta^T P \eta$ and $\dot{V}$ via finite differences on GPU.

- **`HLIP`** ([ref_gen.py](mdp/ref_gen.py)): Pure-math LIP planner. Computes desired orbits and CoM trajectories from the S2S map.

- **`HLIPCommandCfg`** ([hlip_command.py](mdp/hlip_command.py)): Dataclass holding all tunable parameters — gait timing, swing height, CLF Q/R weights, velocity ranges, orientation references.
