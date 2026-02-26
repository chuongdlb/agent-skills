# PID Math Reference

Complete PID gains, mixer matrices, PWM conversion, and integral windup limits for DSLPIDControl.

## PID Gains

### Position PID (Force)

| Axis | P | I | D |
|------|---|---|---|
| X | 0.4 | 0.05 | 0.2 |
| Y | 0.4 | 0.05 | 0.2 |
| Z | 1.25 | 0.05 | 0.5 |

```python
P_COEFF_FOR = [0.4, 0.4, 1.25]
I_COEFF_FOR = [0.05, 0.05, 0.05]
D_COEFF_FOR = [0.2, 0.2, 0.5]
```

### Attitude PID (Torque)

| Axis | P | I | D |
|------|---|---|---|
| Roll | 70000 | 0 | 20000 |
| Pitch | 70000 | 0 | 20000 |
| Yaw | 60000 | 500 | 12000 |

```python
P_COEFF_TOR = [70000., 70000., 60000.]
I_COEFF_TOR = [0., 0., 500.]
D_COEFF_TOR = [20000., 20000., 12000.]
```

## Integral Windup Limits

### Position integral
```python
integral_pos_e = clip(integral_pos_e, -2., 2.)       # all axes
integral_pos_e[2] = clip(integral_pos_e[2], -0.15, 0.15)  # tighter Z limit
```

### Attitude integral
```python
integral_rpy_e = clip(integral_rpy_e, -1500., 1500.)  # all axes
integral_rpy_e[0:2] = clip(integral_rpy_e[0:2], -1., 1.)  # tighter roll/pitch
```

### Torque output clipping
```python
target_torques = clip(target_torques, -3200, 3200)
```

## Mixer Matrices

### CF2X (X-configuration)
```python
MIXER_MATRIX = [
    [-0.5, -0.5, -1],    # motor 0 (front-right)
    [-0.5,  0.5,  1],    # motor 1 (rear-right)
    [ 0.5,  0.5, -1],    # motor 2 (rear-left)
    [ 0.5, -0.5,  1]     # motor 3 (front-left)
]
```

### CF2P (+-configuration)
```python
MIXER_MATRIX = [
    [ 0, -1, -1],    # motor 0 (front, +X)
    [+1,  0,  1],    # motor 1 (left, +Y)
    [ 0,  1, -1],    # motor 2 (rear, -X)
    [-1,  0,  1]     # motor 3 (right, -Y)
]
```

The mixer converts `[roll_torque, pitch_torque, yaw_torque]` to per-motor PWM adjustments:
```
pwm_motor = thrust_pwm + MIXER_MATRIX @ [roll_torque, pitch_torque, yaw_torque]
```

## PWM to RPM Conversion

```python
PWM2RPM_SCALE = 0.2685
PWM2RPM_CONST = 4070.3
MIN_PWM = 20000
MAX_PWM = 65535

rpm = PWM2RPM_SCALE * clip(pwm, MIN_PWM, MAX_PWM) + PWM2RPM_CONST
```

**PWM range to RPM range:**
```
MIN_PWM=20000 → RPM = 0.2685 * 20000 + 4070.3 = 9440.3
MAX_PWM=65535 → RPM = 0.2685 * 65535 + 4070.3 = 21666.0
```

## Position Control Equations

**Thrust computation:**
```
target_thrust = P_FOR * (target_pos - cur_pos)
              + I_FOR * integral_pos_e
              + D_FOR * (target_vel - cur_vel)
              + [0, 0, GRAVITY]

scalar_thrust = max(0, dot(target_thrust, cur_rotation_z_axis))
thrust_pwm = (sqrt(scalar_thrust / (4 * KF)) - PWM2RPM_CONST) / PWM2RPM_SCALE
```

**Target rotation from thrust vector:**
```
z_axis = target_thrust / ||target_thrust||
x_c = [cos(target_yaw), sin(target_yaw), 0]
y_axis = normalize(cross(z_axis, x_c))
x_axis = cross(y_axis, z_axis)
R_target = [x_axis; y_axis; z_axis]^T
target_euler = R_target → XYZ Euler angles
```

## Attitude Control Equations

**Rotation error (SO(3) error):**
```
R_e = R_target^T @ R_current - R_current^T @ R_target
rot_e = [R_e[2,1], R_e[0,2], R_e[1,0]]    # vee map
```

**Angular rate error:**
```
rpy_rates_e = target_rpy_rates - (cur_rpy - last_rpy) / dt
```

**Torque computation:**
```
target_torques = -P_TOR * rot_e + D_TOR * rpy_rates_e + I_TOR * integral_rpy_e
target_torques = clip(target_torques, -3200, 3200)
```

## `_one23DInterface()` — Thrust Shortcut

Converts 1D, 2D, or 4D thrust inputs directly to PWM:

```python
def _one23DInterface(self, thrust):
    DIM = len(thrust)
    pwm = clip((sqrt(thrust / (KF * (4/DIM))) - PWM2RPM_CONST) / PWM2RPM_SCALE, MIN_PWM, MAX_PWM)
    if DIM in [1, 4]: return repeat(pwm, 4/DIM)
    if DIM == 2: return hstack([pwm, flip(pwm)])
```

## MRAC Gains

```python
# Pole placement
desired_poles = -linspace(1, 12, 12)
K = place(A, B, desired_poles)

# Lyapunov
Am = A - B @ K
Q = eye(12) * 600
P = solve_lyapunov(Am^T, -Q)

# Adaptive rates
Gamma_x = eye(12) * 5e-3
Gamma_r = eye(4) * 5e-3
```

## CTBRControl Gains

```python
K_P = [3., 3., 8.]       # Position proportional
K_D = [2.5, 2.5, 5.]     # Velocity proportional (derivative)
K_RATES = [5., 5., 1.]   # Body rate gain
```
