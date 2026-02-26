---
name: pybullet-simulation-engine
description: >
  Core PyBullet simulation engine — BaseAviary class hierarchy, physics models, step loop, frequency management, state vectors, and drone URDF configuration.
layer: L2
domain: [drones, simulation]
source-project: gym-pybullet-drones
depends-on: [gymnasium-core-api]
tags: [pybullet, physics, aviary, quadrotor]
---

# Simulation Engine

The simulation engine centers on `BaseAviary`, a `gymnasium.Env` subclass that wraps PyBullet for multi-quadrotor simulation. It manages drone loading, physics stepping, state tracking, vision capture, and provides the abstract interface that all environment subclasses implement.

## Class Hierarchy

```
gymnasium.Env
└── BaseAviary                    # Core simulation: physics, state, rendering
    ├── CtrlAviary                # Direct RPM control (non-RL)
    ├── CFAviary                  # Crazyflie firmware-in-the-loop
    ├── BetaAviary                # Betaflight SITL via UDP
    └── BaseRLAviary              # RL wrapper: action/obs types, PID, buffer
        ├── HoverAviary           # Single-agent hover task
        └── MultiHoverAviary      # Multi-agent hover task
```

## Constructor Parameters

```python
BaseAviary(
    drone_model: DroneModel = DroneModel.CF2X,  # CF2X, CF2P, or RACE
    num_drones: int = 1,
    neighbourhood_radius: float = np.inf,       # for adjacency matrix
    initial_xyzs = None,                        # (NUM_DRONES, 3) or None
    initial_rpys = None,                        # (NUM_DRONES, 3) or None
    physics: Physics = Physics.PYB,             # PYB, DYN, PYB_GND, PYB_DRAG, PYB_DW, PYB_GND_DRAG_DW
    pyb_freq: int = 240,                        # PyBullet physics frequency (Hz)
    ctrl_freq: int = 240,                       # Environment control frequency (Hz)
    gui: bool = False,
    record: bool = False,
    obstacles: bool = False,
    user_debug_gui: bool = True,                # RPM sliders + axis viz
    vision_attributes: bool = False,            # allocate RGB/depth/seg arrays
    output_folder: str = 'results'
)
```

## Frequency Management

```
pyb_freq (e.g., 240 Hz)    — PyBullet internal physics timestep
ctrl_freq (e.g., 30 Hz)    — Environment step() frequency
PYB_STEPS_PER_CTRL = pyb_freq / ctrl_freq  (e.g., 8)
```

Each `env.step()` call runs `PYB_STEPS_PER_CTRL` PyBullet physics substeps. The constraint is `pyb_freq % ctrl_freq == 0`.

## Step Loop

```
step(action)
├── _preprocessAction(action) → clipped_action (NUM_DRONES, 4) RPMs
├── for _ in range(PYB_STEPS_PER_CTRL):
│   ├── _updateAndStoreKinematicInformation()  [if multi-step + extended physics]
│   ├── for each drone:
│   │   └── _physics() / _dynamics() / _groundEffect() / _drag() / _downwash()
│   └── p.stepSimulation()  [unless Physics.DYN]
├── _updateAndStoreKinematicInformation()
├── _computeObs() → obs
├── _computeReward() → reward
├── _computeTerminated() → terminated
├── _computeTruncated() → truncated
├── _computeInfo() → info
└── return obs, reward, terminated, truncated, info
```

## 20D State Vector

`_getDroneStateVector(nth_drone)` returns a 20-element array:

| Index | Field | Unit |
|-------|-------|------|
| 0-2 | x, y, z | meters |
| 3-6 | qx, qy, qz, qw | quaternion |
| 7-9 | roll, pitch, yaw | radians |
| 10-12 | vx, vy, vz | m/s |
| 13-15 | wx, wy, wz | rad/s (world frame) |
| 16-19 | rpm0, rpm1, rpm2, rpm3 | RPM |

```python
state = np.hstack([self.pos[i,:], self.quat[i,:], self.rpy[i,:],
                   self.vel[i,:], self.ang_v[i,:], self.last_clipped_action[i,:]])
```

## Physics Modes

| Physics enum | Methods called per substep |
|---|---|
| `PYB` | `_physics()` |
| `DYN` | `_dynamics()` (explicit Euler, no `p.stepSimulation`) |
| `PYB_GND` | `_physics()` + `_groundEffect()` |
| `PYB_DRAG` | `_physics()` + `_drag()` |
| `PYB_DW` | `_physics()` + `_downwash()` |
| `PYB_GND_DRAG_DW` | `_physics()` + `_groundEffect()` + `_drag()` + `_downwash()` |

See [physics-models-reference.md](physics-models-reference.md) for detailed equations.

## Key Computed Constants

```python
self.GRAVITY = G * M                                    # weight (N)
self.HOVER_RPM = sqrt(GRAVITY / (4 * KF))               # RPM to hover
self.MAX_RPM = sqrt((THRUST2WEIGHT_RATIO * GRAVITY) / (4 * KF))
self.MAX_THRUST = 4 * KF * MAX_RPM**2
self.GND_EFF_H_CLIP = 0.25 * PROP_RADIUS * sqrt((15 * MAX_RPM**2 * KF * GND_EFF_COEFF) / MAX_THRUST)
```

## Key APIs

| Method | Description |
|--------|-------------|
| `step(action)` | Advance simulation one control step |
| `reset(seed, options)` | Reset environment to initial state |
| `_getDroneStateVector(i)` | Get 20D state for drone i |
| `_getAdjacencyMatrix()` | NUM_DRONES x NUM_DRONES neighbor matrix |
| `_normalizedActionToRPM(action)` | [-1,1] → [0, MAX_RPM] (nonlinear) |
| `_getDroneImages(i)` | RGB, depth, segmentation from drone i POV |
| `getPyBulletClient()` | Return PyBullet client ID |
| `getDroneIds()` | Return array of PyBullet body IDs |

## 7 Abstract Methods (must implement in subclasses)

```python
def _actionSpace(self):            # → gymnasium.spaces.Box
def _observationSpace(self):       # → gymnasium.spaces.Box
def _computeObs(self):             # → ndarray
def _preprocessAction(self, action): # → (NUM_DRONES, 4) RPMs
def _computeReward(self):          # → float
def _computeTerminated(self):      # → bool
def _computeTruncated(self):       # → bool
def _computeInfo(self):            # → dict
```

## Enums

```python
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType, ActionType, ObservationType

class DroneModel(Enum):
    CF2X = "cf2x"   # Crazyflie 2.0, X configuration
    CF2P = "cf2p"   # Crazyflie 2.0, + configuration
    RACE = "racer"  # Racing drone, X configuration

class Physics(Enum):
    PYB = "pyb"
    DYN = "dyn"
    PYB_GND = "pyb_gnd"
    PYB_DRAG = "pyb_drag"
    PYB_DW = "pyb_dw"
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"
```

## Key Source Files

| File | Purpose |
|------|---------|
| `gym_pybullet_drones/envs/BaseAviary.py` | Core simulation class |
| `gym_pybullet_drones/utils/enums.py` | DroneModel, Physics, ImageType, ActionType, ObservationType |
| `gym_pybullet_drones/assets/cf2x.urdf` | Crazyflie 2.0 X-config URDF |
| `gym_pybullet_drones/assets/cf2p.urdf` | Crazyflie 2.0 +-config URDF |
| `gym_pybullet_drones/assets/racer.urdf` | Racing drone URDF |
