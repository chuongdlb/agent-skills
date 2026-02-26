---
name: gpd-rl-environments
description: >
  RL environment creation — BaseRLAviary, ActionType/ObservationType dispatch, reward design, HoverAviary reference, single vs multi-agent patterns.
layer: L3
domain: [drones, general-rl]
source-project: gym-pybullet-drones
depends-on: [pybullet-simulation-engine, gymnasium-core-api, gymnasium-spaces]
tags: [rl-environment, gymnasium, hover, multi-agent]
---

# RL Environments

`BaseRLAviary` extends `BaseAviary` to provide a complete RL interface with configurable action/observation types, integrated PID controllers, action buffering, and the 4 abstract methods every RL task must implement.

## Architecture

```
BaseAviary
└── BaseRLAviary
    ├── ActionType dispatch (_preprocessAction)
    ├── ObservationType dispatch (_computeObs, _observationSpace)
    ├── Action buffer (deque of size ctrl_freq // 2)
    ├── Integrated DSLPIDControl (for PID/VEL/ONE_D_PID action types)
    ├── HoverAviary          # Single-agent: hover at [0,0,1]
    └── MultiHoverAviary     # Multi-agent: hover at staggered heights
```

## BaseRLAviary Constructor

```python
BaseRLAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    neighbourhood_radius=np.inf,
    initial_xyzs=None,
    initial_rpys=None,
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=240,
    gui=False,
    record=False,
    obs=ObservationType.KIN,    # KIN or RGB
    act=ActionType.RPM          # RPM, PID, VEL, ONE_D_RPM, ONE_D_PID
)
```

Key defaults set by BaseRLAviary:
- `obstacles=True` (always, for RGB landmarks)
- `user_debug_gui=False` (no RPM sliders in RL)
- `vision_attributes=True` if `obs==ObservationType.RGB`
- `ACTION_BUFFER_SIZE = ctrl_freq // 2`

## ActionType — 5 Action Types

| ActionType | Dims per drone | Input range | What it controls |
|---|---|---|---|
| `RPM` | 4 | [-1, 1] | Normalized RPM per motor |
| `PID` | 3 | [-1, 1] | Target XYZ position → PID → RPM |
| `VEL` | 4 | [-1, 1] | Velocity direction (3) + magnitude (1) |
| `ONE_D_RPM` | 1 | [-1, 1] | Single RPM value → all 4 motors |
| `ONE_D_PID` | 1 | [-1, 1] | Vertical offset → PID → RPM |

Action space shape: `(NUM_DRONES, action_dim)` with bounds `[-1, +1]`.

See [action-observation-reference.md](action-observation-reference.md) for the full dispatch table.

## ObservationType — 2 Observation Types

### KIN (Kinematic)
12D state per drone + flattened action buffer:
```
obs_12 = [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
```
Note: extracts from 20D state as `[state[0:3], state[7:10], state[10:13], state[13:16]]` — quaternion is skipped.

Total observation size per drone: `12 + ACTION_BUFFER_SIZE * action_dim`

### RGB (Vision)
RGBA images from each drone's POV: shape `(NUM_DRONES, 48, 64, 4)`.

When RGB is selected, 4 landmark objects are loaded (block, cube, duck, teddy) at cardinal positions.

## 4 Abstract Methods to Implement

Every RL environment must implement these:

```python
def _computeReward(self) -> float:
    """Return scalar reward for the current step."""

def _computeTerminated(self) -> bool:
    """Return True if the episode reached a success condition."""

def _computeTruncated(self) -> bool:
    """Return True if the episode should end early (timeout, out-of-bounds)."""

def _computeInfo(self) -> dict:
    """Return auxiliary info dictionary."""
```

## HoverAviary — Canonical Single-Agent Example

```python
class HoverAviary(BaseRLAviary):
    def __init__(self, ..., ctrl_freq=30, ...):   # Note: ctrl_freq=30 (not 240)
        self.TARGET_POS = np.array([0, 0, 1])
        self.EPISODE_LEN_SEC = 8
        super().__init__(num_drones=1, ...)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        return max(0, 2 - np.linalg.norm(self.TARGET_POS - state[0:3])**4)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        return np.linalg.norm(self.TARGET_POS - state[0:3]) < .0001

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0
            or abs(state[7]) > .4 or abs(state[8]) > .4):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {"answer": 42}
```

**Reward design:** `max(0, 2 - ||error||^4)` — quartic penalty, capped at 0, max reward is 2 per step.

**Truncation conditions:**
- Position too far: |x|>1.5, |y|>1.5, z>2.0
- Too tilted: |roll|>0.4, |pitch|>0.4
- Timeout: `step_counter / PYB_FREQ > EPISODE_LEN_SEC`

## MultiHoverAviary — Multi-Agent Example

```python
class MultiHoverAviary(BaseRLAviary):
    def __init__(self, ..., num_drones=2, ctrl_freq=30, ...):
        self.EPISODE_LEN_SEC = 8
        super().__init__(num_drones=num_drones, ...)
        self.TARGET_POS = self.INIT_XYZS + np.array([[0,0,1/(i+1)] for i in range(num_drones)])

    def _computeReward(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = 0
        for i in range(self.NUM_DRONES):
            ret += max(0, 2 - np.linalg.norm(self.TARGET_POS[i,:] - states[i][0:3])**4)
        return ret
```

Reward is summed across all drones. Targets are staggered heights: drone 0 at +1.0m, drone 1 at +0.5m, etc.

## Gymnasium Registration

```python
# gym_pybullet_drones/__init__.py
register(id='hover-aviary-v0', entry_point='gym_pybullet_drones.envs:HoverAviary')
register(id='multihover-aviary-v0', entry_point='gym_pybullet_drones.envs:MultiHoverAviary')
```

Usage: `env = gym.make('hover-aviary-v0', obs=ObservationType.KIN, act=ActionType.RPM)`

## Key Source Files

| File | Purpose |
|------|---------|
| `gym_pybullet_drones/envs/BaseRLAviary.py` | RL base class with action/obs dispatch |
| `gym_pybullet_drones/envs/HoverAviary.py` | Single-agent hover task |
| `gym_pybullet_drones/envs/MultiHoverAviary.py` | Multi-agent hover task |
| `gym_pybullet_drones/__init__.py` | Gymnasium registration |
| `gym_pybullet_drones/utils/enums.py` | ActionType, ObservationType enums |
