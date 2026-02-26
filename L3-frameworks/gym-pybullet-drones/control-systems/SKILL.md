---
name: gpd-control-systems
description: >
  Flight controllers — DSLPIDControl cascaded PID, MRAC adaptive control, CTBRControl for Betaflight, BaseControl interface, controller pipeline.
layer: L3
domain: [drones]
source-project: gym-pybullet-drones
depends-on: [pybullet-simulation-engine]
tags: [pid, control, mrac, flight-controller]
---

# Control Systems

Three controller implementations convert target positions/velocities into motor RPMs. All follow the `BaseControl` interface pattern.

## Controller Hierarchy

```
BaseControl                    # Abstract base: computeControl → (rpm, pos_e, yaw_e)
├── DSLPIDControl              # Cascaded PID (position → attitude → mixer → PWM → RPM)
└── MRAC                       # Model Reference Adaptive Control (LQR + Lyapunov)

CTBRControl                    # Standalone: Collective Thrust + Body Rates for Betaflight
```

## BaseControl Interface

```python
class BaseControl:
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        self.DRONE_MODEL = drone_model
        self.GRAVITY = g * _getURDFParameter('m')
        self.KF = _getURDFParameter('kf')
        self.KM = _getURDFParameter('km')

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel,
                       cur_ang_vel, target_pos, target_rpy=zeros(3),
                       target_vel=zeros(3), target_rpy_rates=zeros(3)):
        """Returns (rpm, pos_error, yaw_error)"""
        raise NotImplementedError

    def computeControlFromState(self, control_timestep, state, target_pos, ...):
        """Convenience: extracts pos/quat/vel/ang_vel from 20D state vector."""
        return self.computeControl(
            cur_pos=state[0:3], cur_quat=state[3:7],
            cur_vel=state[10:13], cur_ang_vel=state[13:16], ...)

    def setPIDCoefficients(self, p_coeff_pos, i_coeff_pos, d_coeff_pos,
                           p_coeff_att, i_coeff_att, d_coeff_att):
        """Override PID gains at runtime."""
```

## DSLPIDControl — Cascaded PID

The primary controller. Two-stage cascade: position PID → attitude PID → motor mixer.

```
target_pos ──→ [Position PID] ──→ target_thrust + target_rpy
                                          │
cur_state ───→ [Attitude PID] ←──────────┘
                     │
                     ▼
              [Motor Mixer] ──→ PWM ──→ RPM
```

### Position PID Stage

```python
pos_e = target_pos - cur_pos
vel_e = target_vel - cur_vel
integral_pos_e += pos_e * dt              # clamped to [-2, 2], z to [-0.15, 0.15]

target_thrust = P_FOR * pos_e + I_FOR * integral_pos_e + D_FOR * vel_e + [0, 0, GRAVITY]
scalar_thrust = max(0, dot(target_thrust, cur_rotation[:,2]))
thrust_pwm = (sqrt(scalar_thrust / (4*KF)) - PWM2RPM_CONST) / PWM2RPM_SCALE
```

Target rotation is computed from desired heading and thrust direction:
```python
target_z = target_thrust / ||target_thrust||
target_x_c = [cos(target_yaw), sin(target_yaw), 0]
target_y = cross(target_z, target_x_c) / ||cross(target_z, target_x_c)||
target_x = cross(target_y, target_z)
target_euler = Rotation.from_matrix([target_x, target_y, target_z]^T).as_euler('XYZ')
```

### Attitude PID Stage

```python
rot_matrix_e = target_rot^T @ cur_rot - cur_rot^T @ target_rot
rot_e = [rot_matrix_e[2,1], rot_matrix_e[0,2], rot_matrix_e[1,0]]
rpy_rates_e = target_rpy_rates - (cur_rpy - last_rpy) / dt
integral_rpy_e -= rot_e * dt              # clamped to [-1500, 1500], xy to [-1, 1]

target_torques = -P_TOR * rot_e + D_TOR * rpy_rates_e + I_TOR * integral_rpy_e
target_torques = clip(target_torques, -3200, 3200)
```

### Motor Mixer

```python
pwm = thrust_pwm + MIXER_MATRIX @ target_torques
pwm = clip(pwm, MIN_PWM, MAX_PWM)
rpm = PWM2RPM_SCALE * pwm + PWM2RPM_CONST
```

See [pid-math-reference.md](pid-math-reference.md) for gains, mixer matrices, and PWM constants.

## MRAC — Model Reference Adaptive Control

Adaptive controller using LQR reference model and Lyapunov-based adaptation law.

```python
class MRAC(BaseControl):
    def __init__(self, drone_model, g=9.8):
        # Linearize dynamics about hover → A, B matrices
        # Place poles or LQR → nominal gain K
        # Reference model: Am = A - B@K, Bm = B
        # Solve Lyapunov: Am^T @ P + P @ Am = -Q
        # Adaptive gains: Gamma_x = 5e-3 * I_12, Gamma_r = 5e-3 * I_4
```

**Control law:**
```
u = Kx^T @ X + Kr^T @ r
e = X_actual - X_model
Kx_dot = -Gamma_x @ X @ e^T @ P @ Bm
Kr_dot = -Gamma_r @ r @ e^T @ P @ Bm
```

Outputs `(thrust, tx, ty, tz)` which are converted to RPM via the same mixer/PWM pipeline as DSLPIDControl.

Supports CF2X, CF2P, and RACE drones.

## CTBRControl — Collective Thrust + Body Rates

Designed for Betaflight SITL integration. Computes thrust and angular rates directly.

```python
class CTBRControl:
    def computeControl(self, ..., cur_pos, cur_quat, cur_vel, target_pos, target_vel, ...):
        G = [0, 0, -9.8]
        K_P = [3., 3., 8.]
        K_D = [2.5, 2.5, 5.]
        K_RATES = [5., 5., 1.]

        tar_acc = K_P * (target_pos - cur_pos) + K_D * (target_vel - cur_vel) - G
        norm_thrust = dot(tar_acc, rotate_vector([0,0,1], cur_quat))

        # Target attitude from desired acceleration
        z_body = normalize(tar_acc)
        x_body = normalize(cross([0,1,0], z_body))
        y_body = normalize(cross(z_body, x_body))
        tar_att = mat2quat([x_body, y_body, z_body]^T)

        # Body rates from quaternion error
        q_error = qmult(qconj(cur_quat), tar_att)
        body_rates = 2 * K_RATES * q_error[1:]
        if q_error[0] < 0: body_rates = -body_rates

        return norm_thrust, *body_rates
```

Note: `computeControlFromState` reorders quaternion from state as `[state[6], state[3], state[4], state[5]]` (w,x,y,z).

Returns 4 values `(thrust, roll_rate, pitch_rate, yaw_rate)` instead of RPMs.

## Usage with CtrlAviary

```python
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=2, gui=True)
ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
obs, info = env.reset()

for i in range(1000):
    action = np.zeros((2, 4))
    for j in range(2):
        rpm, pos_e, yaw_e = ctrl[j].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[j],
            target_pos=np.array([0, 0, 1])
        )
        action[j,:] = rpm
    obs, reward, terminated, truncated, info = env.step(action)
```

## Key Source Files

| File | Purpose |
|------|---------|
| `gym_pybullet_drones/control/BaseControl.py` | Abstract control interface |
| `gym_pybullet_drones/control/DSLPIDControl.py` | Cascaded PID controller |
| `gym_pybullet_drones/control/MRAC.py` | Adaptive controller |
| `gym_pybullet_drones/control/CTBRControl.py` | Betaflight thrust+rates controller |
| `gym_pybullet_drones/envs/CtrlAviary.py` | Control application environment |
