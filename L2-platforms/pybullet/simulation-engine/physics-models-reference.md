# Physics Models Reference

Detailed equations for all 6 `Physics` enum modes in `BaseAviary`.

## `_physics()` — Base PyBullet Physics (Physics.PYB)

Applies thrust and torque forces to PyBullet bodies, then calls `p.stepSimulation()`.

**Per-motor thrust:**
```
force_i = rpm_i^2 * KF        (applied along propeller link z-axis)
```

**Z-axis torque (yaw):**
```
torque_i = rpm_i^2 * KM
z_torque = -torque_0 + torque_1 - torque_2 + torque_3
```

For `DroneModel.RACE`, torques are negated before the alternating-sign sum.

Forces are applied to each propeller link (indices 0-3) in `LINK_FRAME`. The z-torque is applied to the center-of-mass link (index 4) in `LINK_FRAME`.

## `_dynamics()` — Explicit Dynamics (Physics.DYN)

Bypasses PyBullet physics entirely. Uses explicit Euler integration.

**Forces (world frame):**
```
forces = rpm^2 * KF
thrust = [0, 0, sum(forces)]
thrust_world = rotation @ thrust
force_world = thrust_world - [0, 0, GRAVITY]
```

**Torques (body frame):**

For CF2X/RACE (X-config):
```
x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * L/sqrt(2)
y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * L/sqrt(2)
```
Note: CF2X negates x_torque; RACE does not.

For CF2P (+-config):
```
x_torque = (forces[1] - forces[3]) * L
y_torque = (-forces[0] + forces[2]) * L
```

Z-torque same as `_physics()`.

**Euler integration:**
```
torques = torques - cross(rpy_rates, J @ rpy_rates)    # gyroscopic term
rpy_rates_deriv = J_INV @ torques
acceleration = force_world / M

vel += dt * acceleration
rpy_rates += dt * rpy_rates_deriv
pos += dt * vel
quat = integrateQ(quat, rpy_rates, dt)                  # quaternion integration
```

State is set directly via `p.resetBasePositionAndOrientation()` and `p.resetBaseVelocity()`.

## `_groundEffect()` — Ground Effect Model (Physics.PYB_GND)

Based on analytical model from Shi et al., 2019. Applied in addition to `_physics()`.

**Per-propeller ground effect force:**
```
prop_height = max(link_z_position, GND_EFF_H_CLIP)
gnd_effect_i = rpm_i^2 * KF * GND_EFF_COEFF * (PROP_RADIUS / (4 * prop_height))^2
```

Only applied when `|roll| < pi/2` and `|pitch| < pi/2` (drone roughly upright).

**Height clipping threshold:**
```
GND_EFF_H_CLIP = 0.25 * PROP_RADIUS * sqrt((15 * MAX_RPM^2 * KF * GND_EFF_COEFF) / MAX_THRUST)
```

## `_drag()` — Drag Model (Physics.PYB_DRAG)

Based on system identification from Forster, 2015. Applied to the center of mass.

```
drag_factors = -DRAG_COEFF * sum(2 * pi * rpm / 60)
drag = rotation^T @ (drag_factors * velocity)
```

Where `DRAG_COEFF = [drag_coeff_xy, drag_coeff_xy, drag_coeff_z]` from URDF.

The drag force is applied in `LINK_FRAME` on the center-of-mass link (index 4).

## `_downwash()` — Downwash Model (Physics.PYB_DW)

Based on experiments at DSL by SiQi Zhou. Models the downward force exerted by a higher drone on a lower drone.

For each pair of drones where drone `i` is above drone `nth_drone`:
```
delta_z = pos_i[2] - pos_nth[2]     # vertical separation (must be > 0)
delta_xy = ||pos_i[0:2] - pos_nth[0:2]||   # horizontal distance

alpha = DW_COEFF_1 * (PROP_RADIUS / (4 * delta_z))^2
beta = DW_COEFF_2 * delta_z + DW_COEFF_3

downwash_force = -alpha * exp(-0.5 * (delta_xy / beta)^2)   # z-direction only
```

Only computed for drones within 10m horizontal distance. The effect is a Gaussian-shaped downward force strongest directly below the upper drone.

## `_normalizedActionToRPM()` — Action Normalization

Non-linear mapping from [-1, 1] to [0, MAX_RPM]:

```
if action <= 0:
    rpm = (action + 1) * HOVER_RPM           # -1 → 0, 0 → HOVER_RPM
else:
    rpm = HOVER_RPM + (MAX_RPM - HOVER_RPM) * action  # 0 → HOVER_RPM, 1 → MAX_RPM
```

## Physics Mode Composition Table

| Mode | `_physics` | `_dynamics` | `_groundEffect` | `_drag` | `_downwash` | `p.stepSimulation` |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| PYB | x | | | | | x |
| DYN | | x | | | | |
| PYB_GND | x | | x | | | x |
| PYB_DRAG | x | | | x | | x |
| PYB_DW | x | | | | x | x |
| PYB_GND_DRAG_DW | x | | x | x | x | x |

## URDF Parameters Used by Physics

| Parameter | Symbol | CF2X | CF2P | RACE |
|-----------|--------|------|------|------|
| Mass | M | 0.027 kg | 0.027 kg | 0.830 kg |
| Arm length | L | 0.0397 m | 0.0397 m | 0.109 m |
| Thrust coeff | KF | 3.16e-10 | 3.16e-10 | 8.47e-9 |
| Torque coeff | KM | 7.94e-12 | 7.94e-12 | 2.13e-11 |
| Ground effect coeff | GND_EFF_COEFF | 11.36859 | 11.36859 | 11.36859 |
| Propeller radius | PROP_RADIUS | 2.31348e-2 m | 2.31348e-2 m | 12.7e-2 m |
| Drag XY coeff | DRAG_COEFF_XY | 9.1785e-7 | 9.1785e-7 | 9.1785e-7 |
| Drag Z coeff | DRAG_COEFF_Z | 10.311e-7 | 10.311e-7 | 10.311e-7 |
| Downwash coeff 1 | DW_COEFF_1 | 2267.18 | 2267.18 | 2267.18 |
| Downwash coeff 2 | DW_COEFF_2 | 0.16 | 0.16 | 0.16 |
| Downwash coeff 3 | DW_COEFF_3 | -0.11 | -0.11 | -0.11 |
