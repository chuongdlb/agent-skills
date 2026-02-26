# Utilities Catalog

Categorized reference for all IsaacLab utility modules.

## Math Functions (`isaaclab.utils.math`)

### Quaternion Operations (wxyz convention)

| Function | Signature | Description |
|----------|-----------|-------------|
| `quat_unique(q)` | `(Tensor) -> Tensor` | Standardize to non-negative w |
| `quat_conjugate(q)` | `(Tensor) -> Tensor` | Quaternion conjugate |
| `quat_inv(q)` | `(Tensor) -> Tensor` | Quaternion inverse |
| `quat_mul(q1, q2)` | `(Tensor, Tensor) -> Tensor` | Quaternion multiplication |
| `quat_apply(q, v)` | `(Tensor, Tensor) -> Tensor` | Rotate vector by quaternion |
| `quat_apply_inverse(q, v)` | `(Tensor, Tensor) -> Tensor` | Rotate by inverse quaternion |
| `quat_apply_yaw(q, v)` | `(Tensor, Tensor) -> Tensor` | Rotate by yaw component only |
| `quat_from_euler_xyz(r, p, y)` | `(Tensor, Tensor, Tensor) -> Tensor` | XYZ Euler to quaternion |
| `euler_xyz_from_quat(q)` | `(Tensor) -> (Tensor, Tensor, Tensor)` | Quaternion to XYZ Euler |
| `quat_from_angle_axis(angle, axis)` | `(Tensor, Tensor) -> Tensor` | Angle-axis to quaternion |
| `axis_angle_from_quat(q)` | `(Tensor) -> Tensor` | Quaternion to axis-angle |
| `quat_error_magnitude(q1, q2)` | `(Tensor, Tensor) -> Tensor` | Rotation difference magnitude |
| `yaw_quat(q)` | `(Tensor) -> Tensor` | Extract yaw-only rotation |
| `quat_slerp(q1, q2, tau)` | `(Tensor, Tensor, float) -> Tensor` | Spherical linear interpolation |
| `quat_box_minus(q1, q2)` | `(Tensor, Tensor) -> Tensor` | Tangent-space difference |
| `quat_box_plus(q, delta)` | `(Tensor, Tensor) -> Tensor` | Add tangent-space delta |
| `matrix_from_quat(q)` | `(Tensor) -> Tensor` | Quaternion to 3x3 rotation matrix |
| `quat_from_matrix(m)` | `(Tensor) -> Tensor` | 3x3 rotation matrix to quaternion |
| `convert_quat(q, to)` | `(Tensor, str) -> Tensor` | Convert between wxyz/xyzw |

### Frame Transforms

| Function | Signature | Description |
|----------|-----------|-------------|
| `combine_frame_transforms(t01, q01, t12, q12)` | `(...) -> (Tensor, Tensor)` | Compose T_02 = T_01 * T_12 |
| `subtract_frame_transforms(t01, q01, t02, q02)` | `(...) -> (Tensor, Tensor)` | Compute T_12 = T_01^-1 * T_02 |
| `compute_pose_error(src_pos, src_rot, tgt_pos, tgt_rot)` | `(...) -> (Tensor, Tensor)` | Position and rotation error |
| `apply_delta_pose(pos, rot, delta)` | `(...) -> (Tensor, Tensor)` | Apply pose delta |
| `transform_points(points, pos, quat)` | `(...) -> Tensor` | Transform points by pose |
| `rigid_body_twist_transform(v, w, t, q)` | `(...) -> (Tensor, Tensor)` | Transform twist between frames |

### Pose Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `make_pose(pos, rot)` | `(Tensor, Tensor) -> Tensor` | Create 4x4 pose matrix |
| `unmake_pose(pose)` | `(Tensor) -> (Tensor, Tensor)` | Extract pos, rot from 4x4 |
| `pose_inv(pose)` | `(Tensor) -> Tensor` | Invert 4x4 pose |
| `interpolate_poses(p1, p2, ...)` | `(...) -> Tensor` | Interpolate between poses |
| `is_identity_pose(pos, rot)` | `(Tensor, Tensor) -> bool` | Check if identity |

### Sampling Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `sample_uniform(lo, hi, size, device)` | `(...) -> Tensor` | Uniform distribution |
| `sample_gaussian(mean, std, size, device)` | `(...) -> Tensor` | Gaussian distribution |
| `sample_log_uniform(lo, hi, size, device)` | `(...) -> Tensor` | Log-uniform distribution |
| `sample_triangle(lo, hi, size, device)` | `(...) -> Tensor` | Triangular distribution |
| `sample_cylinder(radius, h_range, size, device)` | `(...) -> Tensor` | Cylinder surface |
| `default_orientation(num, device)` | `(int, str) -> Tensor` | Identity quaternions |
| `random_orientation(num, device)` | `(int, str) -> Tensor` | Random quaternions |
| `random_yaw_orientation(num, device)` | `(int, str) -> Tensor` | Random yaw-only rotations |

### General Transforms

| Function | Signature | Description |
|----------|-----------|-------------|
| `scale_transform(x, lo, hi)` | `(Tensor, Tensor, Tensor) -> Tensor` | Normalize to [-1, 1] |
| `unscale_transform(x, lo, hi)` | `(Tensor, Tensor, Tensor) -> Tensor` | Denormalize from [-1, 1] |
| `saturate(x, lo, hi)` | `(Tensor, Tensor, Tensor) -> Tensor` | Clamp to range |
| `normalize(x)` | `(Tensor) -> Tensor` | Normalize to unit length |
| `wrap_to_pi(angles)` | `(Tensor) -> Tensor` | Wrap angles to [-π, π] |
| `skew_symmetric_matrix(vec)` | `(Tensor) -> Tensor` | 3D vector to skew matrix |

### Camera/Depth Operations

| Function | Description |
|----------|-------------|
| `unproject_depth(depth, intrinsics)` | Depth image to 3D points |
| `project_points(points, intrinsics)` | 3D points to image coords |
| `orthogonalize_perspective_depth(depth, intrinsics)` | Perspective to orthogonal depth |
| `convert_camera_frame_orientation_convention(orient, origin, target)` | Convert between opengl/ros/world |
| `create_rotation_matrix_from_view(eye, lookat, up)` | View rotation matrix |

## Noise Models (`isaaclab.utils.noise`)

### Noise Configs

| Class | Attributes | Description |
|-------|------------|-------------|
| `ConstantNoiseCfg` | `bias: float = 0.0`, `operation: str = "add"` | Constant bias |
| `UniformNoiseCfg` | `n_min, n_max: float`, `operation: str = "add"` | Uniform random noise |
| `GaussianNoiseCfg` | `mean, std: float`, `operation: str = "add"` | Gaussian random noise |

Operation modes: `"add"` (additive), `"scale"` (multiplicative), `"abs"` (absolute)

### Noise Model Classes

| Class | Config | Description |
|-------|--------|-------------|
| `NoiseModel` | `NoiseModelCfg` | Applies noise per step |
| `NoiseModelWithAdditiveBias` | `NoiseModelWithAdditiveBiasCfg` | Persistent bias + per-step noise |

```python
# NoiseModelCfg attributes
noise_cfg: NoiseCfg = MISSING
func: Callable | None = None    # Optional custom noise function

# NoiseModelWithAdditiveBiasCfg additional attributes
bias_noise_cfg: NoiseCfg = MISSING            # Bias sampling distribution
sample_bias_per_component: bool = True         # Independent bias per component
```

## Modifiers (`isaaclab.utils.modifiers`)

### Function-Based

| Function | Params | Description |
|----------|--------|-------------|
| `scale` | `multiplier: float` | Multiply data by multiplier |
| `clip` | `bounds: tuple[float, float]` | Clip to [min, max] |
| `bias` | `value: float` | Add uniform bias |

### Class-Based (Stateful)

| Class | Config Attributes | Description |
|-------|-------------------|-------------|
| `DigitalFilter` | `A: list[float]`, `B: list[float]` | IIR/FIR digital filter |
| `Integrator` | `dt: float` | Numerical integrator |

## Buffers (`isaaclab.utils.buffers`)

### CircularBuffer

```python
CircularBuffer(max_len: int, batch_size: int, device: str)
```

| Property/Method | Description |
|-----------------|-------------|
| `batch_size` | Number of parallel buffers |
| `max_length` | Maximum history length |
| `current_length` | Tensor of current lengths per batch |
| `buffer` | Raw buffer tensor |
| `reset(batch_ids)` | Reset specified buffers |
| `append(data)` | Add data to buffer |
| `__getitem__(key)` | LIFO access (0=most recent) |

### DelayBuffer

```python
DelayBuffer(history_length: int, batch_size: int, device: str)
```

| Property/Method | Description |
|-----------------|-------------|
| `history_length` | Maximum delay length |
| `time_lags` | Current per-batch delay |
| `set_time_lag(lag, batch_ids)` | Set delay in timesteps |
| `reset(batch_ids)` | Reset buffer |
| `compute(data)` | Return delayed data |

### TimestampedBuffer

```python
@dataclass
class TimestampedBuffer:
    data: torch.Tensor = None
    timestamp: float = -1.0
```

## Datasets (`isaaclab.utils.datasets`)

### EpisodeData

| Property/Method | Description |
|-----------------|-------------|
| `data` | Nested dict of episode data |
| `seed` | Random seed for episode |
| `env_id` | Environment ID |
| `success` | Whether episode succeeded |
| `add(key, value)` | Add key-value pair |
| `get_action(idx)` | Get action at index |
| `get_state(idx)` | Get state at index |
| `pre_export()` | Convert lists to tensors |
| `is_empty()` | Check if empty |

### HDF5DatasetFileHandler

| Method | Description |
|--------|-------------|
| `create(path, env_name)` | Create new HDF5 file |
| `open(path, mode)` | Open existing file |
| `write_episode(episode)` | Write episode data |
| `load_episode(name)` | Load episode by name |
| `get_num_episodes()` | Count episodes |
| `get_episode_names()` | List episode keys |
| `flush()` | Flush to disk |
| `close()` | Close file |

## IO Utilities (`isaaclab.utils.io`)

| Function | Description |
|----------|-------------|
| `load_yaml(filename)` | Load YAML file to dict |
| `dump_yaml(filename, data)` | Save dict/object to YAML |
| `load_torchscript_model(path, device)` | Load JIT model in eval mode |

## Warp Operations (`isaaclab.utils.warp`)

| Function | Description |
|----------|-------------|
| `raycast_mesh(starts, dirs, mesh, ...)` | Ray cast against warp mesh |
| `raycast_single_mesh(starts, dirs, mesh_id, ...)` | Batched ray cast |
| `convert_to_warp_mesh(vertices, indices)` | Convert tensors to warp mesh |

```python
# raycast_mesh returns
(ray_hits,                    # (N, 3) hit positions, inf for miss
 ray_distance,                # (N,) optional distance
 ray_normal,                  # (N, 3) optional surface normal
 ray_face_id)                 # (N,) optional face index, -1 for miss
```
