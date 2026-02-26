---
name: gpd-extending-the-codebase
description: >
  How to extend gym-pybullet-drones — new RL environments, control environments, drone models, Gymnasium registration, extension checklists.
layer: L3
domain: [drones]
source-project: gym-pybullet-drones
depends-on: [pybullet-simulation-engine, gpd-rl-environments, gymnasium-custom-environments]
tags: [extending, templates, registration]
---

# Extending the Codebase

Two primary extension paths exist depending on whether the new environment is for RL or for scripted control. Each path requires implementing different abstract methods.

## Extension Path 1: New RL Environment

Extend `BaseRLAviary` and implement 4 methods. The base class handles action/observation dispatch, PID integration, and action buffering.

### Steps

1. Create a new file in `gym_pybullet_drones/envs/`
2. Subclass `BaseRLAviary`
3. Implement the 4 abstract methods
4. Register with Gymnasium in `__init__.py`

### 4 Required Methods

```python
def _computeReward(self) -> float:
    """Scalar reward. Use self._getDroneStateVector(i) for state access."""

def _computeTerminated(self) -> bool:
    """True when success condition is met."""

def _computeTruncated(self) -> bool:
    """True for early termination (timeout, out-of-bounds, unsafe)."""

def _computeInfo(self) -> dict:
    """Auxiliary info dict (can be empty)."""
```

### Example: HoverAviary pattern

```python
class MyEnv(BaseRLAviary):
    def __init__(self, ..., ctrl_freq=30, ...):
        self.TARGET = ...
        self.EPISODE_LEN_SEC = 8
        super().__init__(num_drones=1, ctrl_freq=ctrl_freq, ...)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        return max(0, 2 - np.linalg.norm(self.TARGET - state[0:3])**4)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        return np.linalg.norm(self.TARGET - state[0:3]) < 0.0001

    def _computeTruncated(self):
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {}
```

## Extension Path 2: New Control Environment

Extend `BaseAviary` directly and implement 7 abstract methods. This gives full control over action/observation spaces and preprocessing.

### 7 Required Methods

```python
def _actionSpace(self):           # → gymnasium.spaces.Box
def _observationSpace(self):      # → gymnasium.spaces.Box
def _computeObs(self):            # → ndarray
def _preprocessAction(self, action):  # → (NUM_DRONES, 4) RPMs
def _computeReward(self):         # → float (can return -1 if unused)
def _computeTerminated(self):     # → bool (can return False if unused)
def _computeTruncated(self):      # → bool
def _computeInfo(self):           # → dict
```

### CtrlAviary Pattern

```python
class MyCtrlEnv(BaseAviary):
    def _actionSpace(self):
        lo = np.array([[0]*4 for _ in range(self.NUM_DRONES)])
        hi = np.array([[self.MAX_RPM]*4 for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _observationSpace(self):
        # 20D: pos(3) + quat(4) + rpy(3) + vel(3) + ang_vel(3) + rpm(4)
        lo = np.array([[-np.inf]*16 + [0]*4 for _ in range(self.NUM_DRONES)])
        hi = np.array([[np.inf]*16 + [self.MAX_RPM]*4 for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _computeObs(self):
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    def _preprocessAction(self, action):
        return np.array([np.clip(action[i,:], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    def _computeReward(self):  return -1
    def _computeTerminated(self):  return False
    def _computeTruncated(self):  return False
    def _computeInfo(self):  return {"answer": 42}
```

## Adding a New DroneModel

### Step 1: Add enum value

```python
# gym_pybullet_drones/utils/enums.py
class DroneModel(Enum):
    CF2X = "cf2x"
    CF2P = "cf2p"
    RACE = "racer"
    MY_DRONE = "my_drone"    # <-- add here
```

### Step 2: Create URDF file

Save as `gym_pybullet_drones/assets/my_drone.urdf` (filename must match enum value + ".urdf").

Required URDF structure:
```xml
<?xml version="1.0" ?>
<robot name="my_drone">
  <properties arm="0.05" kf="3.16e-10" km="7.94e-12" thrust2weight="2.5"
    max_speed_kmh="50" gnd_eff_coeff="11.36859" prop_radius="3e-2"
    drag_coeff_xy="9.1785e-7" drag_coeff_z="10.311e-7"
    dw_coeff_1="2267.18" dw_coeff_2=".16" dw_coeff_3="-.11" />
  <link name="base_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="2e-5" ixy="0" ixz="0" iyy="2e-5" iyz="0" izz="3e-5"/>
    </inertial>
    <visual>...</visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><cylinder radius="0.08" length="0.03"/></geometry>
    </collision>
  </link>
  <!-- prop0_link through prop3_link with motor positions -->
  <!-- center_of_mass_link -->
</robot>
```

### Step 3: Update controllers (if needed)

DSLPIDControl only supports CF2X and CF2P. For a new drone model with PID-based actions:
- Add a mixer matrix in `DSLPIDControl.__init__()`, or
- Create a new controller subclass

## Gymnasium Registration

```python
# gym_pybullet_drones/__init__.py
from gymnasium.envs.registration import register

register(
    id='my-env-v0',
    entry_point='gym_pybullet_drones.envs:MyEnv',
)
```

Existing registrations:
```
ctrl-aviary-v0      → CtrlAviary
velocity-aviary-v0  → VelocityAviary
hover-aviary-v0     → HoverAviary
multihover-aviary-v0 → MultiHoverAviary
```

## Adding Obstacles

Override `_addObstacles()` in your environment subclass:

```python
def _addObstacles(self):
    p.loadURDF("cube_no_rotation.urdf",
               [1, 0, 0.5],
               p.getQuaternionFromEuler([0, 0, 0]),
               physicsClientId=self.CLIENT)
```

Built-in PyBullet URDFs available: `plane.urdf`, `samurai.urdf`, `duck_vhacd.urdf`, `cube_small.urdf`, `cube_no_rotation.urdf`, `sphere2.urdf`, `block.urdf`, `teddy_vhacd.urdf`.

## Extension Checklists

### New RL Environment Checklist
- [ ] Create file in `gym_pybullet_drones/envs/`
- [ ] Subclass `BaseRLAviary`
- [ ] Set `EPISODE_LEN_SEC` and any target state
- [ ] Implement `_computeReward()` (shaped reward works best)
- [ ] Implement `_computeTerminated()` (success condition)
- [ ] Implement `_computeTruncated()` (timeout + safety bounds)
- [ ] Implement `_computeInfo()`
- [ ] Register in `__init__.py`
- [ ] Add test in `tests/test_examples.py`

### New Drone Model Checklist
- [ ] Add enum value in `utils/enums.py`
- [ ] Create URDF with all required properties
- [ ] Verify motor positions match X or + config
- [ ] Update DSLPIDControl mixer matrix (if using PID actions)
- [ ] Test with CtrlAviary at minimum

### New Controller Checklist
- [ ] Subclass `BaseControl`
- [ ] Implement `computeControl()` returning `(rpm, pos_e, yaw_e)`
- [ ] Load drone params via `_getURDFParameter()`
- [ ] Implement `reset()` to zero internal state
- [ ] Test with CtrlAviary loop

See [templates-reference.md](templates-reference.md) for copy-paste scaffolds.

## Key Source Files

| File | Purpose |
|------|---------|
| `gym_pybullet_drones/envs/HoverAviary.py` | RL environment template |
| `gym_pybullet_drones/envs/CtrlAviary.py` | Control environment template |
| `gym_pybullet_drones/utils/enums.py` | DroneModel enum (add new drones here) |
| `gym_pybullet_drones/__init__.py` | Gymnasium registration |
| `gym_pybullet_drones/assets/cf2x.urdf` | URDF template |
| `gym_pybullet_drones/control/BaseControl.py` | Controller interface template |
