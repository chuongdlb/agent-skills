---
name: gpd-logging-testing
description: >
  Logger class for data recording and visualization, real-time sync, pytest patterns, PyBullet debug tools, and video recording.
layer: L3
domain: [drones]
source-project: gym-pybullet-drones
depends-on: [pybullet-simulation-engine, K-Dense-AI/matplotlib]
tags: [logging, testing, pytest, visualization]
---

# Logging & Testing

The `Logger` class records kinematic state and control targets during simulation, saves to NPZ/CSV, and generates 10x2 subplot plots. The testing infrastructure uses pytest with headless PyBullet.

## Logger Class

### Initialization

```python
from gym_pybullet_drones.utils.Logger import Logger

logger = Logger(
    logging_freq_hz=int(env.CTRL_FREQ),  # must match env step rate
    num_drones=1,
    output_folder='results',
    duration_sec=0,     # 0 = dynamic allocation, >0 = preallocate arrays
    colab=False         # True saves plot as PNG instead of displaying
)
```

### Logging

```python
logger.log(
    drone=0,                    # drone index
    timestamp=i / env.CTRL_FREQ,  # simulation time (seconds)
    state=obs[0],               # 20D state vector from _getDroneStateVector()
    control=np.zeros(12)        # 12D control targets (optional)
)
```

**Important:** The `state` parameter must be a 20-element array matching the format from `_getDroneStateVector()`.

### State Vector Reordering

Logger internally reorders the 20D input state to a 16D stored format:

```python
# Input (BaseAviary 20D):
#   [0:3]  pos_x, pos_y, pos_z
#   [3:7]  qx, qy, qz, qw          ← dropped
#   [7:10] roll, pitch, yaw
#   [10:13] vx, vy, vz
#   [13:16] wx, wy, wz
#   [16:20] rpm0, rpm1, rpm2, rpm3

# Stored (Logger 16D):
self.states[drone, :, step] = np.hstack([
    state[0:3],     # [0:3]  pos_x, pos_y, pos_z
    state[10:13],   # [3:6]  vx, vy, vz
    state[7:10],    # [6:9]  roll, pitch, yaw
    state[13:20]    # [9:16] wx, wy, wz, rpm0, rpm1, rpm2, rpm3
])
```

### Saving

```python
# NPZ format
logger.save()
# Output: results/save-flight-MM.DD.YYYY_HH.MM.SS.npy

# CSV format
logger.save_as_csv(comment="my_experiment")
# Output: results/save-flight-my_experiment-MM.DD.YYYY_HH.MM.SS/
#   x0.csv, y0.csv, z0.csv, r0.csv, p0.csv, ya0.csv,
#   vx0.csv, vy0.csv, vz0.csv, wx0.csv, wy0.csv, wz0.csv,
#   rr0.csv, pr0.csv, yar0.csv,    (angular rate derivatives)
#   rpm0-0.csv, rpm1-0.csv, rpm2-0.csv, rpm3-0.csv,
#   pwm0-0.csv, pwm1-0.csv, pwm2-0.csv, pwm3-0.csv
```

### Plotting

```python
logger.plot(pwm=False)  # pwm=True converts RPM to PWM for drones > 0
```

Generates a 10x2 subplot figure. See [logger-format-reference.md](logger-format-reference.md) for the full layout.

## Logging from RL Observations

When logging from KIN observations (12D, no quaternion), reconstruct the 20D state:

```python
obs2 = obs.squeeze()
act2 = action.squeeze()
logger.log(
    drone=0,
    timestamp=i / env.CTRL_FREQ,
    state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
    control=np.zeros(12)
)
```

The zeros fill the quaternion slot (indices 3-6) and the action fills the RPM slot (indices 16-19).

## Real-Time Sync

```python
from gym_pybullet_drones.utils.utils import sync

start = time.time()
for i in range(total_steps):
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    sync(i, start, env.CTRL_TIMESTEP)
```

`sync()` pauses execution to match wall-clock time:
```python
def sync(i, start_time, timestep):
    if timestep > 0.04 or i % int(1/(24*timestep)) == 0:
        elapsed = time.time() - start_time
        if elapsed < i * timestep:
            time.sleep(timestep * i - elapsed)
```

For fast environments (timestep < 0.04s), sync only runs every `1/(24*timestep)` steps (targeting 24 FPS visual update).

## PyBullet Debug Tools

### RPM Sliders (user_debug_gui=True)

When `gui=True` and `user_debug_gui=True`, BaseAviary adds interactive sliders:

```python
# 4 RPM sliders: range [0, MAX_RPM], default HOVER_RPM
self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, MAX_RPM, HOVER_RPM)
# Toggle switch: "Use GUI RPM"
self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0)
```

Click "Use GUI RPM" to override the action and use slider values.

### Drone Axis Visualization

When `user_debug_gui=True`, colored axes are drawn on each drone:
- Red: X-axis
- Green: Y-axis
- Blue: Z-axis
- Length: `2 * L` (twice the arm length)

### Video Recording

```python
env = CtrlAviary(record=True, gui=True)   # records .mp4 via PyBullet
env = CtrlAviary(record=True, gui=False)  # saves PNG frames
```

With GUI: PyBullet's built-in video logging → `.mp4`
Without GUI: Frame-by-frame PNG capture at 24 FPS.

### Camera Images

```python
rgb, dep, seg = env._getDroneImages(nth_drone=0, segmentation=True)
# rgb: (48, 64, 4) uint8 RGBA
# dep: (48, 64) float depth
# seg: (48, 64) int segmentation mask
```

## Testing Infrastructure

### Test Files

```
tests/
├── test_build.py       # Import tests
└── test_examples.py    # Example script execution tests
```

### test_build.py — Import Verification

```python
def test_imports():
    import gym_pybullet_drones
    import gym_pybullet_drones.control
    import gym_pybullet_drones.envs
    import gym_pybullet_drones.examples
    import gym_pybullet_drones.utils
```

### test_examples.py — Functional Tests

```python
def test_pid():
    from gym_pybullet_drones.examples.pid import run
    run(gui=False, plot=False, output_folder='tmp')

def test_learn():
    from gym_pybullet_drones.examples.learn import run
    run(gui=False, plot=False, output_folder='tmp', local=False)
    # local=False uses 1e2 timesteps instead of 1e7
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/test_build.py -v
pytest tests/test_examples.py::test_pid -v
```

### Writing New Tests

Pattern: call the example's `run()` function with `gui=False` and `plot=False`:

```python
def test_my_example():
    from gym_pybullet_drones.examples.my_example import run
    run(gui=False, plot=False, output_folder='tmp')
```

## Utility Functions

```python
from gym_pybullet_drones.utils.utils import sync, str2bool

# sync(i, start_time, timestep) — real-time synchronization
# str2bool(val) — parse string as boolean (for argparse)
```

## Key Source Files

| File | Purpose |
|------|---------|
| `gym_pybullet_drones/utils/Logger.py` | Data logging and plotting |
| `gym_pybullet_drones/utils/utils.py` | sync(), str2bool() utilities |
| `tests/test_build.py` | Import verification tests |
| `tests/test_examples.py` | Example execution tests |
