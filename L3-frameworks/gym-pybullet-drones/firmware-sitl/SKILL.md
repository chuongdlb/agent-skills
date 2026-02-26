---
name: gpd-firmware-sitl
description: >
  Firmware-in-the-loop simulation — CFAviary (pycffirmware Crazyflie SITL), BetaAviary (Betaflight UDP SITL), command interfaces, sensor pipelines.
layer: L3
domain: [drones, sim-to-real]
source-project: gym-pybullet-drones
depends-on: [pybullet-simulation-engine]
tags: [firmware, sitl, crazyflie, betaflight]
---

# Firmware SITL

Two firmware-in-the-loop environments integrate real flight controller firmware into the PyBullet simulation: `CFAviary` for Crazyflie (pycffirmware, in-process) and `BetaAviary` for Betaflight (UDP, out-of-process).

## Architecture Overview

```
CFAviary (in-process)                    BetaAviary (out-of-process)
┌──────────────────────┐                ┌──────────────────────┐
│ BaseAviary           │                │ BaseAviary           │
│  └── CFAviary        │                │  └── BetaAviary      │
│       ├── pycffirmware│               │       ├── UDP socket  │
│       ├── Mellinger/PID│              │       ├── FDM packets  │──→ BF SITL
│       ├── Command queue│              │       ├── RC packets   │──→ BF SITL
│       └── Sensor LPF  │              │       └── PWM receive  │←── BF SITL
└──────────────────────┘                └──────────────────────┘
```

## CFAviary — Crazyflie Firmware SITL

Uses `pycffirmware` (Python bindings for CF firmware) running in the same process. Supports Mellinger or PID controllers at 500/1000 Hz.

### Setup

```bash
# Install pycffirmware
git clone https://github.com/utiasDSL/pycffirmware
cd pycffirmware && pip install .
```

### Constructor

```python
CFAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,          # Only 1 supported currently
    physics=Physics.PYB,
    pyb_freq=500,          # Must be multiple of firmware_freq
    ctrl_freq=25,          # High-level command rate
    gui=False,
    verbose=False
)
```

**Key configuration:**
```python
CONTROLLER = 'mellinger'        # or 'pid'
firmware_freq = 500             # 500 for mellinger, 1000 for PID
GYRO_LPF_CUTOFF_FREQ = 80      # Hz
ACCEL_LPF_CUTOFF_FREQ = 30     # Hz
```

### Step Loop

`CFAviary.step(i)` takes a control step index (not an action):

```python
def step(self, i):
    t = i / self.ctrl_freq
    self._process_command_queue(t)

    while self.tick / self.firmware_freq < t + self.ctrl_dt:
        obs, reward, terminated, truncated, info = super().step(self.action)
        # Extract state from obs
        # Estimate angular rates and acceleration
        self._update_state(timestamp, pos, vel, acc, rpy_deg)
        self._update_sensorData(timestamp, body_acc, gyro_deg_s)
        self._updateSetpoint(t)
        self._step_controller()
        # Convert control output to RPMs
        action = PWM2RPM_SCALE * clip(pwms, MIN_PWM, MAX_PWM) + PWM2RPM_CONST
```

### Command Interface

Commands are queued and processed at the control rate:

```python
env.sendFullStateCmd(pos, vel, acc, yaw, rpy_rate, timestep)
env.sendTakeoffCmd(height, duration)
env.sendTakeoffYawCmd(height, duration, yaw)
env.sendTakeoffVelCmd(height, vel, relative)
env.sendLandCmd(height, duration)
env.sendLandYawCmd(height, duration, yaw)
env.sendLandVelCmd(height, vel, relative)
env.sendGotoCmd(pos, yaw, duration_s, relative)
env.sendStopCmd()
env.notifySetpointStop()
```

**`sendFullStateCmd`** overrides the high-level commander and sets the setpoint directly (position, velocity, acceleration, yaw, body rates).

**High-level commands** (takeoff, land, goto) use the firmware's built-in trajectory planner.

### Tumble Detection

```python
if self.state.acc.z < -0.5:
    tumble_counter += 1
if tumble_counter >= 30:
    # Kill motors, set error flag
```

### Usage Example

```python
from gym_pybullet_drones.envs.CFAviary import CFAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

env = CFAviary(drone_model=DroneModel.CF2X, pyb_freq=500, ctrl_freq=25, gui=True)

for i in range(500):
    obs, reward, terminated, truncated, info = env.step(i)
    if i == 25:  # at t=1s
        env.sendFullStateCmd(
            pos=[0, 0, 1], vel=[0,0,0], acc=[0,0,0],
            yaw=0, rpy_rate=[0,0,0], timestep=i/env.ctrl_freq)
```

## BetaAviary — Betaflight SITL

Communicates with external Betaflight SITL executables via UDP sockets. Supports multiple drones.

### Setup

```bash
# Create Betaflight SITL executables
./gym_pybullet_drones/assets/clone_bfs.sh <num_drones>
# This creates betaflight_sitl/bf0/, bf1/, etc.
```

### Constructor

```python
BetaAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=240,
    gui=False,
    udp_ip="127.0.0.1"
)
```

Automatically spawns SITL processes in separate terminals.

### UDP Protocol

**Port assignment per drone:**
```
Drone j:
  PWM in:   9002 + 10*j    (receive motor PWMs from BF)
  State out: 9003 + 10*j   (send FDM packet to BF)
  RC out:    9004 + 10*j   (send RC commands to BF)
```

**Step loop:** `BetaAviary.step(action, i)` takes both a CTBR action and step index:

```python
def step(self, action, i):
    obs = super().step(self.beta_action)  # step physics with last received PWMs

    for j in range(NUM_DRONES):
        # Send FDM state packet (18 doubles)
        fdm_packet = struct.pack('@dddddddddddddddddd',
            t, wx, -wy, -wz,    # ENU→NED conversion
            0, 0, 0,             # linear acceleration
            1, 0, 0, 0,          # orientation quat
            0, 0, 0,             # velocity
            0, 0, 0,             # position
            1.0)                 # pressure

        # Send RC packet (1 double + 16 uint16)
        rc_packet = struct.pack('@dHHHHHHHHHHHHHHHH',
            t, roll, pitch, thro, yaw,
            aux1, ...)           # aux1: arm switch

        # Receive PWM packet (4 floats)
        pwm_data = sock_pwm.recvfrom(16)
        beta_action = struct.unpack('@ffff', pwm_data)
```

### CTBRControl Integration

BetaAviary uses `CTBRControl` to convert position targets to thrust+body rates:

```python
from gym_pybullet_drones.control.CTBRControl import CTBRControl

ctrl = CTBRControl(drone_model=DroneModel.RACE)
action[j,:] = ctrl.computeControlFromState(
    control_timestep=env.CTRL_TIMESTEP,
    state=obs[j],
    target_pos=target["pos"],
    target_vel=target["vel"]
)
```

The CTBR output `(thrust, roll_rate, pitch_rate, yaw_rate)` is converted to Betaflight RC channels via `ctbr2beta()`:
```python
def ctbr2beta(self, thrust, roll, pitch, yaw):
    # Maps to [1000, 2000] RC channel range
    # MAX_RATE = 360 deg/s, MAX_THRUST = 40.9
```

### Arming Sequence

```python
ARM_TIME = 1.0    # seconds before arming
TRAJ_TIME = 1.5   # seconds before sending trajectory commands
# Before ARM_TIME: aux1=1000 (disarmed)
# After ARM_TIME: aux1=1500 (armed)
# After TRAJ_TIME: start sending CTBR actions
```

See [firmware-api-reference.md](firmware-api-reference.md) for data structures and protocol details.

## Key Source Files

| File | Purpose |
|------|---------|
| `gym_pybullet_drones/envs/CFAviary.py` | Crazyflie firmware SITL environment |
| `gym_pybullet_drones/envs/BetaAviary.py` | Betaflight UDP SITL environment |
| `gym_pybullet_drones/examples/cf.py` | CFAviary usage example |
| `gym_pybullet_drones/examples/beta.py` | BetaAviary usage example |
| `gym_pybullet_drones/control/CTBRControl.py` | Thrust+rates controller for BF |
