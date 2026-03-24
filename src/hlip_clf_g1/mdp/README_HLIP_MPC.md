# HLIP + MPC in hlip_command.py

This document explains how the walking command generator works in:

- `src/hlip_clf_g1/mdp/hlip_command.py`

It also references supporting pieces in:

- `src/hlip_clf_g1/mdp/ref_gen.py` (HLIP + Bezier utilities)
- `src/hlip_clf_g1/mdp/clf.py` (CLF evaluator)
- `src/hlip_clf_g1/hlip_env_cfg.py` (task-level parameterization)
- `src/hlip_clf_g1/mdp/observations.py` (heightmap observation accessor)

## 1. What this module is responsible for

`HLIPCommandTerm` is the locomotion command core. Every control step, it:

1. Maintains gait phase and stance/swing bookkeeping.
2. Builds a full-body reference trajectory (`y_out`, `dy_out`) from HLIP.
3. Optionally modifies foot placement and step timing with terrain-aware MPC.
4. Extracts actual robot state (`y_act`, `dy_act`) in the same coordinates.
5. Computes CLF value and derivative (`V`, `Vdot`) for rewards and logging.

In practice, it is not just a "command sampler". It is the trajectory planner, phase machine, and CLF observer combined.

## 2. Coordinate conventions and data layout

The module consistently works in a **stance-local yaw frame** for foot and CoM references:

- Origin: stance foot position at the beginning of the current half-step (`stance_foot_pos_0`).
- Orientation: yaw-only rotation from stance foot orientation (`stance_foot_ori_quat_0`).

Why this matters:

- HLIP equations are easier and cleaner in a local frame.
- MPC evaluates candidate footholds in that same local frame.
- Actual state extraction converts world data back to this frame, so CLF compares like-for-like quantities.

Main tracked vectors per environment:

- `y_out`, `dy_out`: reference output position/velocity.
- `y_act`, `dy_act`: measured output position/velocity.
- `foot_target`: current planned landing point in stance-local frame.
- `vel_command`: command velocity `[vx, vy, yaw_rate]`.

Output structure (`_num_outputs = 12 + n_upper`):

1. CoM position (3)
2. Pelvis Euler orientation (3)
3. Swing foot position (3)
4. Swing foot Euler orientation (3)
5. Upper-body joints (`n_upper`)

Velocities follow the same ordering.

## 3. Lifecycle and call flow

### 3.1 Initialization (`__init__`)

Initialization sets up:

- HLIP model (`HLIP` from `ref_gen.py`).
- CLF model (`CLF` from `clf.py`).
- Gait phase buffers (`tp`, `phase_var`, `cur_swing_time`).
- Stance/swing indices and transition state.
- MPC planning buffers and configuration.
- Heightmap grid metadata for edge classification.

### 3.2 Episode reset (`_resample_command`)

On reset/resample, it:

1. Samples velocity command from configured ranges.
2. Zeros tiny commands.
3. Applies standing fraction (`rel_standing_envs`).
4. Resets gait phase and stance state.
5. Clears CLF history and MPC plan state.
6. Defers initial stance-foot read until the first post-reset update (because body links are stale before `sim.forward()`).

### 3.3 Per-step update (`_update_command`)

Per environment step:

1. Finalize deferred stance-foot initialization if needed.
2. If MPC enabled and no plan exists, plan one transition immediately.
3. Force standing env velocity command to zero.
4. Update phase and detect stance transitions.
5. On transitions, refresh stance/swing anchors and (if enabled) run MPC planning.
6. Generate full reference trajectory.
7. Extract actual state.
8. Compute CLF `V` and `Vdot`.

## 4. HLIP reference generation details

HLIP model comes from `ref_gen.py` and follows LIP dynamics:

$$
\ddot{x} = \frac{g}{z_0}x
$$

with analogous lateral dynamics.

### 4.1 Step-to-step model

`HLIP.compute_desired_orbit` builds batched step-to-step dynamics using:

- `A_ss = exp(A * (T - T_ds))`
- optional double-support transition `A_ds`
- input map `B_s2s`

From commanded velocity and gait half-period `T`, it solves linear systems for desired orbit states and foot placements:

- Forward placement: proportional to `vx * T`
- Lateral placement: `vy * T ± y_nom`

The helper `_remap_for_init_stance_state` computes stance-indexed initial conditions (`x_init`, `y_init`) used by online trajectory rollout.

### 4.2 Online CoM and foot target

Inside `_generate_reference_trajectory`:

1. `compute_orbit(T, cmd)` returns desired orbit quantities (`Xdes`, `Ux`, `Ydes`, `Uy`).
2. Current desired CoM (`com_x`, `com_y`) is rolled out with closed-form hyperbolic expressions.
3. Raw step target is formed as `[Ux, Uy_sel, 0]`.
4. Yaw-rate compensation rotates target and CoM refs by `delta_psi = yaw_rate * cur_swing_time`.
5. Lateral clipping enforces `foot_target_range_y`.

When MPC is enabled, the foot target and step time are overridden by the terrain-aware plan (see section 5).

### 4.3 Swing-foot trajectory

Horizontal progression uses a Bezier phase helper `bht` and linear interpolation between:

- start of current swing (`swing_foot_pos_0`)
- planned touchdown target

Vertical motion uses a 6th-degree Bezier in `calculate_cur_swing_foot_pos`.

Landing height `z_sw_neg` is:

- from MPC foothold z if MPC plan exists, otherwise
- from nearest heightmap ray hit to target x-y.

Peak swing height is adapted to clear both start and end terrain height:

- `z_sw_max_updated = max(z_init, z_sw_neg) + z_sw_max_envs`

### 4.4 Pelvis and upper-body references

`_generate_orientation_ref` adds periodic pelvis roll/pitch components plus lateral/turning biases from command velocity. Yaw tracks stance yaw plus commanded turning.

`_generate_upper_body_ref` applies phase-synchronized sinusoidal references to waist and arm joints, scaled by forward command speed.

## 5. MPC foothold and timing planner (math-first view)

MPC is solved at each stance transition in a stance-local frame.
The planner is discrete: it chooses from raycast foothold candidates and a finite set of step times.

### 5.1 Optimization inputs at one planning instant

Per environment, the planner receives:

1. Commanded planar velocity
   - $v_{cmd} = [v_x, v_y]^T$
2. Current stance side and nominal lateral offset
   - side sign $s_h \in \{-1, +1\}$ for each preview step
   - nominal width parameter $y_{nom}$
3. Candidate foothold cloud from heightmap rays
   - $\mathcal{C} = \{p_j\}_{j=1}^{M}$, with $p_j = (x_j, y_j, z_j)$ in stance-local coordinates
   - validity mask for each candidate (non-edge and not invalid hit)
4. Step-time candidate grid
   - $\mathcal{T} = \{T_1, \dots, T_N\} \subset [T_{min}, T_{max}]$
5. Kinematic and terrain limits
   - x-range, y-range, signed y-min, max step length, optional max stance-height delta

With the current sensor settings, the ray grid is approximately $41 \times 21 = 861$ rays, so typically $M \le 861$.

### 5.2 Decision variables and chained step model

For horizon $H$, the decision at each preview step $h$ is:

- foothold $p_h \in \mathcal{C}$
- step time $T_h \in \mathcal{T}$

Let $p_0$ be the previously selected foothold anchor. Then displacement at step $h$ is:

$$
\Delta p_h = p_h - p_{h-1}, \quad
\Delta x_h = x_h - x_{h-1}, \quad
\Delta y_h = y_h - y_{h-1}.
$$

This chained form is important: each preview step is measured from the previously selected preview foothold, not from a fixed world origin.

### 5.3 Velocity-tracking model

The planner uses step-averaged velocity implied by displacement and duration:

$$
\hat{v}_{x,h} = \frac{\Delta x_h}{T_h}, \qquad
\hat{v}_{y,h} = \frac{\Delta y_h - s_h y_{nom}}{T_h}.
$$

Why the $s_h y_{nom}$ term appears in lateral velocity:

- A healthy alternating gait has built-in lateral foot separation.
- Subtracting $s_h y_{nom}$ removes this nominal side-switch bias.
- What remains is the lateral velocity component that should match command $v_y$.

So velocity tracking error is:

$$
e_{vel,h} = (\hat{v}_{x,h} - v_x)^2 + (\hat{v}_{y,h} - v_y)^2.
$$

### 5.4 Feasibility constraints

Each candidate pair $(p_h, T_h)$ must satisfy:

$$
x_{min} \le \Delta x_h \le x_{max}
$$

$$
y_{abs,min} \le |\Delta y_h| \le y_{abs,max}
$$

$$
\Delta y_h \cdot s_h \ge y_{signed,min}
$$

$$
\sqrt{\Delta x_h^2 + \Delta y_h^2} \le L_{max}
$$

and foothold validity from terrain filtering.

Terrain validity uses a local edge detector:

1. hit must be valid
2. four-neighbor heights must be locally smooth
3. optional stance-height-delta bound

If the edge threshold is not configured explicitly, it is derived from ray resolution:

$$
h_{edge} = \text{resolution} \cdot \tan(20^\circ)
$$

### 5.5 Objective and how command velocity is enforced

For each preview step, the scalar stage cost is:

$$
\ell_h =
w_{vel} \, e_{vel,h}
+ w_{time} \left(\frac{T_h - T_{nom}}{T_{max} - T_{min}}\right)^2
+ w_{foot} \, e_{foot,h}.
$$

The foot consistency term uses a command-implied nominal displacement:

$$
\Delta x_h^{ref} = v_x T_h,
\qquad
\Delta y_h^{ref} = s_h y_{nom} + v_y T_h,
$$

$$
e_{foot,h} =
(\Delta x_h - \Delta x_h^{ref})^2
+ (\Delta y_h - \Delta y_h^{ref})^2.
$$

Total horizon score is the sum of stage costs. Infeasible candidates get infinite cost.

Practical interpretation:

- $w_{vel}$: direct velocity tracking pressure
- $w_{time}$: discourages aggressive time jitter away from $T_{nom}$
- $w_{foot}$: keeps geometry compatible with commanded motion pattern

### 5.6 Solution strategy and fallback logic

The implementation performs a stage-wise discrete search over $(p_h, T_h)$ at each preview step, propagating the chosen foothold as the next anchor.
This is a tractable approximation of full combinatorial search.

If no feasible sequence is found under the base command, fallback tries progressively reduced command magnitudes and axis-aligned alternatives, then a zero-command solve.
If terrain candidates are unavailable, the planner falls back to a deterministic safe stepping pattern and commands near stop.

## 6. Mathematical coupling between HLIP and MPC

HLIP and MPC are coupled through two variables:

1. Selected immediate foothold $p_1$
2. Selected immediate half-step time $T_1$

The coupling is:

1. MPC computes $(p_1, T_1)$ from terrain + command.
2. HLIP uses $T_1$ for orbit rollout and phase timing.
3. Swing trajectory tracks $p_1$ (including its landing height).

So HLIP provides dynamic template structure, and MPC projects that structure onto locally feasible terrain contact choices.

## 7. CLF pipeline in this command term

After reference generation and actual-state extraction:

1. CLF error vector is formed from `(y_act - y_out, dy_act - dy_out)`.
2. Yaw channels are wrapped to `[-pi, pi]` for continuity.
3. CLF value is computed as:

$$
V = \eta^T P \eta
$$

4. `Vdot` is estimated by backward finite differences (1st or 3-point depending on history depth).

The CLF matrix `P` is solved once from continuous-time ARE in `clf.py` and reused.

## 8. Current environment-level settings

In `make_hlip_env_cfg` (`hlip_env_cfg.py`), key runtime choices are:

- `mpc_enabled=True`
- `mpc_horizon=4`
- `mpc_t_candidates=7`
- `T_min=0.3`, `T_max=0.8`
- `mpc_foot_target_range_x=(-0.35, 0.85)`
- `mpc_abs_y_min=0.08`, `mpc_abs_y_max=0.55`
- `mpc_signed_y_min=0.02`
- `mpc_max_step_length=0.95`
- `mpc_max_stance_height_delta=0.3`
- `mpc_fallback_scales=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0)`

Heightmap sensor is configured in `env_cfgs.py` with:

- pelvis frame yaw-aligned rays
- grid size `(3.0, 1.5)` and resolution `0.075`
- max distance `2.0`

And observations expose heightmap distances through `mdp.heightmap_data` in actor and critic groups.

## 9. Debug visualization guide

`_debug_vis_impl` draws:

- stance foot (green)
- active target (red)
- current swing foot (blue)
- reference swing foot (yellow)
- velocity command arrow (cyan)
- MPC horizon chain (multicolor markers and arrows)
- heightmap cells (green non-edge, red edge)

Use this view to quickly diagnose:

- impossible footholds (many red cells around desired path)
- frequent fallbacks (plan not following command direction)
- mismatched swing landing height

## 10. Practical tuning notes

1. If fallback usage is consistently high, first loosen geometric constraints:
   - increase `mpc_max_step_length`
   - widen x/y bounds
   - relax `mpc_signed_y_min`
2. If footholds sit too close to edges, reduce `mpc_edge_height_threshold` or tighten stance height delta.
3. If gait becomes jittery, increase `mpc_w_time` to bias toward nominal timing.
4. If command tracking is poor, increase `mpc_w_vel`.
5. If planner overfits to command and ignores terrain shape, increase `mpc_w_foot` and/or tighten constraints.

## 11. Minimal mental model

At each transition:

1. HLIP proposes where/when to step based on velocity command.
2. MPC checks terrain feasibility over a short horizon and picks feasible footholds and step timing.
3. Swing trajectory tracks that target with smooth Bezier motion.
4. CLF compares actual vs reference and feeds reward shaping.

This loop is what makes the policy terrain-aware while preserving HLIP/CLF structure.
