---
name: isaaclab-configclass-and-utilities
description: >
  Provides guidance on @configclass decorator, MISSING sentinel, inheritance patterns, class_type convention, AppLauncher, math utilities, noise models, and buffers.
layer: L3
domain: [robotics, general-rl]
source-project: IsaacLab
depends-on: [isaacsim-simulation-core]
tags: [configclass, utilities, math, quaternion]
---

# IsaacLab Configclass and Utilities

The `@configclass` decorator and utility modules form the foundation of all IsaacLab configuration. Every environment, asset, sensor, and manager config uses `@configclass`.

## Architecture

```
@configclass decorator (wraps dataclasses)
  ├── MISSING sentinel (required fields)
  ├── Mutable default handling (auto deep-copy)
  ├── to_dict / from_dict / replace / copy / validate
  └── Inheritance with class_type convention

AppLauncher
  ├── CLI argument parsing
  ├── SimulationApp creation
  └── Distributed training support

Utils
  ├── math (quaternions, poses, sampling)
  ├── noise (Gaussian, Uniform, Constant models)
  ├── modifiers (scale, clip, bias, DigitalFilter)
  ├── buffers (CircularBuffer, DelayBuffer, TimestampedBuffer)
  ├── datasets (EpisodeData, HDF5DatasetFileHandler)
  ├── io (YAML, TorchScript loading)
  └── warp (ray casting, mesh conversion)
```

## @configclass Decorator

Wraps Python `dataclasses.dataclass` with enhanced features for configuration management.

```python
from isaaclab.utils import configclass
from dataclasses import MISSING

@configclass
class MyConfig:
    # Required field (must be set before use)
    num_envs: int = MISSING
    # Optional with default
    episode_length: int = 2000
    # Mutable defaults handled automatically (no field(default_factory=...) needed)
    features: list = [0.0, 0.0, 0.0]
```

### Added Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_dict()` | `() -> dict` | Convert to nested dictionary |
| `from_dict(data)` | `(dict) -> None` | Update in-place from dictionary |
| `replace(**kwargs)` | `(**Any) -> Self` | Return new instance with modified fields |
| `copy()` | `() -> Self` | Return deep copy |
| `validate()` | `() -> None` | Raise TypeError if any MISSING fields remain |

### MISSING Sentinel

Use `MISSING` for required fields. Calling `validate()` raises `TypeError` listing all unset fields:

```python
@configclass
class RobotCfg:
    prim_path: str = MISSING
    spawn: SpawnerCfg = MISSING
    num_joints: int = 7  # optional

cfg = RobotCfg()
cfg.validate()  # TypeError: ['prim_path', 'spawn']
```

### Inheritance and class_type Convention

Configs use inheritance with `class_type` to specify the implementation class:

```python
@configclass
class ActuatorCfg:
    class_type: type = MISSING
    joint_names_expr: list[str] = MISSING
    effort_limit: float | None = None

@configclass
class ImplicitActuatorCfg(ActuatorCfg):
    class_type: type = ImplicitActuator  # Sets default implementation

@configclass
class DCMotorCfg(IdealPDActuatorCfg):
    class_type: type = DCMotor
    saturation_effort: float = MISSING
```

Override nested configs with `.replace()`:

```python
base_cfg = LocomotionEnvCfg()
custom_cfg = base_cfg.replace(
    scene=base_cfg.scene.replace(num_envs=64),
    rewards=base_cfg.rewards.replace(track_velocity=1.5),
)
```

### Mutable Default Handling

Unlike plain dataclasses, `@configclass` automatically deep-copies mutable defaults:

```python
@configclass
class MyCfg:
    items: list = [1, 2, 3]          # Safe - auto deep-copied
    nested: dict = {"key": "value"}   # Safe - auto deep-copied
    sub_cfg: SubCfg = SubCfg()        # Safe - auto deep-copied
```

## AppLauncher

Entry point for all IsaacLab scripts. Must be called before any other Isaac Sim imports.

```python
import argparse
from isaaclab.app import AppLauncher

# Create parser and add IsaacLab args
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch (must happen before other Isaac imports)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import Isaac modules
from isaaclab.envs import ManagerBasedRLEnv
```

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--headless` | bool | False | Run without GUI |
| `--device` | str | `"cuda:0"` | Compute device |
| `--livestream` | int | -1 | WebRTC streaming mode |
| `--enable_cameras` | bool | False | Enable cameras in headless |
| `--experience` | str | `""` | Kit experience file |
| `--rendering_mode` | str | `"balanced"` | `"performance"`, `"balanced"`, `"quality"` |
| `--verbose` | bool | False | Verbose logging |

### Environment Variables

`HEADLESS`, `LIVESTREAM`, `ENABLE_CAMERAS`, `EXPERIENCE`, `DEVICE_ID`, `LOCAL_RANK`, `GLOBAL_RANK`

## Math Utilities

All quaternion operations use **wxyz** convention. Import from `isaaclab.utils.math`.

### Quaternion Operations

```python
from isaaclab.utils.math import (
    quat_mul, quat_apply, quat_inv, quat_conjugate,
    quat_from_euler_xyz, euler_xyz_from_quat,
    quat_from_angle_axis, axis_angle_from_quat,
    quat_unique, quat_error_magnitude, yaw_quat,
    quat_slerp, quat_box_minus, quat_box_plus,
    random_orientation, random_yaw_orientation,
)

# Compose rotations
q_combined = quat_mul(q1, q2)
# Rotate a vector
v_rotated = quat_apply(quat, vec)
# Euler <-> Quaternion
quat = quat_from_euler_xyz(roll, pitch, yaw)
roll, pitch, yaw = euler_xyz_from_quat(quat)
```

### Frame Transforms

```python
from isaaclab.utils.math import (
    combine_frame_transforms,
    subtract_frame_transforms,
    compute_pose_error,
    apply_delta_pose,
    transform_points,
)

# Compose two transforms: T_02 = T_01 * T_12
t02, q02 = combine_frame_transforms(t01, q01, t12, q12)
# Compute pose error for control
pos_err, rot_err = compute_pose_error(source_pos, source_rot, target_pos, target_rot)
```

### Sampling Functions

```python
from isaaclab.utils.math import (
    sample_uniform, sample_gaussian, sample_log_uniform,
    sample_triangle, sample_cylinder,
    default_orientation, random_orientation, random_yaw_orientation,
)

# Sample random positions
positions = sample_uniform(lower, upper, size=(N, 3), device="cuda:0")
# Sample random rotations
quats = random_orientation(num=N, device="cuda:0")
```

## Noise Models

Apply noise to sensor data or observations.

```python
from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg, NoiseModelCfg

# Configure noise
noise_cfg = NoiseModelCfg(
    noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05),
)

# Noise with additive bias (persistent per-env bias + per-step noise)
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
noise_cfg = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01),
    bias_noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
)
```

| Noise Type | Parameters | Operation Modes |
|------------|------------|-----------------|
| `ConstantNoiseCfg` | `bias: float` | `"add"`, `"scale"`, `"abs"` |
| `UniformNoiseCfg` | `n_min, n_max: float` | `"add"`, `"scale"`, `"abs"` |
| `GaussianNoiseCfg` | `mean, std: float` | `"add"`, `"scale"`, `"abs"` |

## Modifiers

Transform data in observation/action pipelines.

```python
from isaaclab.utils.modifiers import ModifierCfg, DigitalFilterCfg, IntegratorCfg
from isaaclab.utils.modifiers.modifier import scale, clip, bias

# Function-based modifiers
scale_mod = ModifierCfg(func=scale, params={"multiplier": 2.0})
clip_mod = ModifierCfg(func=clip, params={"bounds": (-1.0, 1.0)})
bias_mod = ModifierCfg(func=bias, params={"value": 0.5})

# Class-based modifiers (stateful)
filter_mod = DigitalFilterCfg(A=[1.0, -0.5], B=[0.25, 0.25])
integrator_mod = IntegratorCfg(dt=0.01)
```

## Buffers

```python
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer, TimestampedBuffer

# Circular buffer for history (LIFO access)
buf = CircularBuffer(max_len=10, batch_size=num_envs, device="cuda:0")
buf.append(data)            # Add new data
recent = buf[torch.zeros(num_envs, dtype=torch.long)]  # Most recent

# Delay buffer for actuator/communication delays
delay = DelayBuffer(history_length=5, batch_size=num_envs, device="cuda:0")
delay.set_time_lag(3)       # 3-step delay
delayed_data = delay.compute(current_data)

# Timestamped buffer for sensor data
ts_buf = TimestampedBuffer(data=torch.zeros(N, 3), timestamp=0.0)
```

## Datasets

```python
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

# Record episode data
episode = EpisodeData()
episode.add("actions", action_tensor)
episode.add("observations", obs_tensor)
episode.pre_export()  # Convert lists to tensors

# Save to HDF5
handler = HDF5DatasetFileHandler()
handler.create("demos.hdf5", env_name="MyEnv-v0")
handler.write_episode(episode)
handler.close()
```

## Reference Files

- [configclass-api-reference.md](configclass-api-reference.md) - Decorator internals, method signatures, inheritance rules, AppLauncher CLI flags
- [utilities-catalog.md](utilities-catalog.md) - Categorized reference for math, noise, modifier, buffer, dataset, IO, and warp utilities

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/isaaclab/isaaclab/utils/configclass.py` | @configclass decorator implementation |
| `source/isaaclab/isaaclab/utils/math.py` | Math utilities (quaternions, poses, sampling) |
| `source/isaaclab/isaaclab/utils/noise/` | Noise models and configs |
| `source/isaaclab/isaaclab/utils/modifiers/` | Data modifiers |
| `source/isaaclab/isaaclab/utils/buffers/` | Circular, delay, timestamped buffers |
| `source/isaaclab/isaaclab/utils/datasets/` | Episode data and HDF5 handlers |
| `source/isaaclab/isaaclab/utils/io/` | YAML and TorchScript IO |
| `source/isaaclab/isaaclab/utils/warp/` | Warp ray casting and mesh ops |
| `source/isaaclab/isaaclab/app/app_launcher.py` | AppLauncher entry point |
