# Controllers API Reference

Full attribute tables for all controller configs, task-space action configs, IK method comparison, and device class APIs.

## DifferentialIKControllerCfg

**File:** `source/isaaclab/isaaclab/controllers/differential_ik_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | DifferentialIKController | Controller class |
| `command_type` | Literal["position", "pose"] | MISSING | Task-space command type |
| `use_relative_mode` | bool | False | Delta vs absolute commands |
| `ik_method` | Literal["pinv", "svd", "trans", "dls"] | MISSING | IK computation method |
| `ik_params` | dict[str, float] \| None | None | Method-specific params |

**Action Dimensions:**
- `command_type="position"`: 3 (x, y, z)
- `command_type="pose"`, `use_relative_mode=True`: 6 (dx, dy, dz, droll, dpitch, dyaw)
- `command_type="pose"`, `use_relative_mode=False`: 7 (x, y, z, qw, qx, qy, qz)

**Controller Methods:**
```python
__init__(cfg, num_envs, device)
set_command(command, ee_pos, ee_quat)         # Set target
compute(ee_pos, ee_quat, jacobian, joint_pos) # Returns joint position targets
reset(env_ids)                                 # Reset controller state
```

## IK Method Comparison

| Method | Formula | Param | Default | Pros | Cons |
|--------|---------|-------|---------|------|------|
| `"pinv"` | J⁺ = J^T(JJ^T)^{-1} | `k_val` | 1.0 | Simple, fast | Unstable near singularity |
| `"svd"` | Adaptive SVD | `min_singular_value` | 1e-5 | Best singularity handling | Slower |
| `"trans"` | Δq = k J^T Δx | `k_val` | 1.0 | Always stable | Slow convergence |
| `"dls"` | J^T(JJ^T + λ²I)^{-1} | `lambda_val` | 0.01 | Good balance | Damping slows near target |

## OperationalSpaceControllerCfg

**File:** `source/isaaclab/isaaclab/controllers/operational_space_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | OperationalSpaceController | Controller class |
| `target_types` | Sequence[str] | MISSING | Target types list |
| `motion_control_axes_task` | Sequence[int] | (1,1,1,1,1,1) | 6D motion mask |
| `contact_wrench_control_axes_task` | Sequence[int] | (0,0,0,0,0,0) | 6D wrench mask |
| `inertial_dynamics_decoupling` | bool | False | Full inverse dynamics |
| `partial_inertial_dynamics_decoupling` | bool | False | Partial decoupling |
| `gravity_compensation` | bool | False | Gravity compensation |
| `impedance_mode` | str | "fixed" | "fixed", "variable", "variable_kp" |
| `motion_stiffness_task` | float \| Sequence | 100.0 | 6D positional gains |
| `motion_damping_ratio_task` | float \| Sequence | 1.0 | 6D damping ratios |
| `motion_stiffness_limits_task` | tuple | (0, 1000) | Gain limits |
| `motion_damping_ratio_limits_task` | tuple | (0, 100) | Damping limits |
| `contact_wrench_stiffness_task` | float \| None | None | Force feedback gains |
| `nullspace_control` | str | "none" | "none" or "position" |
| `nullspace_stiffness` | float | 10.0 | Null-space stiffness |
| `nullspace_damping_ratio` | float | 1.0 | Null-space damping |

**Target Types:**
- `"pose_rel"` — Relative pose target (delta)
- `"pose_abs"` — Absolute pose target
- `"wrench_abs"` — Absolute force/torque target

**Controller Methods:**
```python
__init__(cfg, num_envs, device)
set_command(command, current_ee_pose_b, current_task_frame_pose_b)
compute(jacobian_b, current_ee_pose_b, current_ee_vel_b, current_ee_force_b,
        mass_matrix, gravity, current_joint_pos, current_joint_vel,
        nullspace_joint_pos_target)  # Returns joint efforts
```

## JointImpedanceControllerCfg

**File:** `source/isaaclab/isaaclab/controllers/joint_impedance.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `command_type` | str | "p_abs" | "p_abs" or "p_rel" |
| `impedance_mode` | str | MISSING | "fixed", "variable", "variable_kp" |
| `stiffness` | float \| Sequence | MISSING | Positional gains |
| `damping_ratio` | float \| Sequence \| None | None | Damping ratios |
| `stiffness_limits` | tuple | (0, 300) | Gain limits |
| `damping_ratio_limits` | tuple | (0, 100) | Damping limits |
| `inertial_compensation` | bool | False | Inverse dynamics |
| `gravity_compensation` | bool | False | Gravity comp |
| `dof_pos_offset` | Sequence \| None | None | Position offset |

## RmpFlowControllerCfg

**File:** `source/isaaclab/isaaclab/controllers/rmp_flow.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "rmp_flow" | "rmp_flow" or "rmp_flow_smoothed" |
| `config_file` | str | MISSING | LULA config file |
| `urdf_file` | str | MISSING | Robot URDF |
| `collision_file` | str | MISSING | Collision model |
| `frame_name` | str | MISSING | Target frame in URDF |
| `evaluations_per_frame` | float | MISSING | Substeps per frame |
| `ignore_robot_state_updates` | bool | False | Use internal world model |

## PinkIKControllerCfg

**File:** `source/isaaclab/isaaclab/controllers/pink_ik/pink_ik_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `urdf_path` | str \| None | None | Robot URDF |
| `mesh_path` | str \| None | None | Mesh files |
| `num_hand_joints` | int | 0 | Hand/gripper joints |
| `variable_input_tasks` | list[FrameTask] | MISSING | Controllable tasks |
| `fixed_input_tasks` | list[FrameTask] | MISSING | Constraint tasks |
| `joint_names` | list[str] \| None | None | Controlled joints |
| `all_joint_names` | list[str] \| None | None | All joints |
| `articulation_name` | str | "robot" | USD articulation name |
| `base_link_name` | str | "base_link" | Base link |
| `show_ik_warnings` | bool | True | Show warnings |

## Task-Space Action Configs

### DifferentialInverseKinematicsActionCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | DifferentialInverseKinematicsAction | Action term class |
| `asset_name` | str | — | Scene entity name |
| `joint_names` | list[str] | MISSING | Joints to control |
| `body_name` | str | MISSING | End-effector body |
| `body_offset` | OffsetCfg \| None | None | EE frame offset |
| `scale` | float \| tuple | 1.0 | Action scaling |
| `controller` | DifferentialIKControllerCfg | MISSING | Controller config |

### OperationalSpaceControllerActionCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | OperationalSpaceControllerAction | Action term class |
| `joint_names` | list[str] | MISSING | Joints to control |
| `body_name` | str | MISSING | End-effector body |
| `body_offset` | OffsetCfg \| None | None | EE offset |
| `controller_cfg` | OperationalSpaceControllerCfg | MISSING | Controller config |
| `position_scale` | float | 1.0 | Position scaling |
| `orientation_scale` | float | 1.0 | Orientation scaling |
| `wrench_scale` | float | 1.0 | Wrench scaling |

### PinkInverseKinematicsActionCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | PinkInverseKinematicsAction | Action term class |
| `pink_controlled_joint_names` | list[str] | MISSING | IK joints |
| `hand_joint_names` | list[str] | MISSING | Gripper joints |
| `controller` | PinkIKControllerCfg | MISSING | Controller config |
| `enable_gravity_compensation` | bool | True | Gravity comp |
| `target_eef_link_names` | dict[str, str] | MISSING | Task-link mapping |

### RMPFlowActionCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | RMPFlowAction | Action term class |
| `joint_names` | list[str] | MISSING | Joints to control |
| `body_name` | str | MISSING | End-effector body |
| `body_offset` | OffsetCfg \| None | None | EE offset |
| `scale` | float \| tuple | 1.0 | Action scaling |
| `controller` | RmpFlowControllerCfg | MISSING | Controller config |
| `articulation_prim_expr` | str | MISSING | Articulation prim pattern |
| `use_relative_mode` | bool | False | Delta commands |

## Device APIs

### Common DeviceBase Interface

```python
class DeviceBase:
    def reset(self) -> None: ...
    def add_callback(self, key, func) -> None: ...
    def advance(self) -> Any: ...     # Returns device-specific data
```

### Se3KeyboardCfg

| Attribute | Type | Default |
|-----------|------|---------|
| `gripper_term` | bool | True |
| `pos_sensitivity` | float | 0.4 |
| `rot_sensitivity` | float | 0.8 |

### Se3GamepadCfg

| Attribute | Type | Default |
|-----------|------|---------|
| `gripper_term` | bool | True |
| `dead_zone` | float | 0.01 |
| `pos_sensitivity` | float | 1.0 |
| `rot_sensitivity` | float | 1.6 |

### Se3SpaceMouseCfg

| Attribute | Type | Default |
|-----------|------|---------|
| `gripper_term` | bool | True |
| `pos_sensitivity` | float | 0.4 |
| `rot_sensitivity` | float | 0.8 |

### HaplyDeviceCfg

| Attribute | Type | Default |
|-----------|------|---------|
| `websocket_uri` | str | "ws://localhost:10001" |
| `pos_sensitivity` | float | 1.0 |
| `data_rate` | float | 200.0 |
| `limit_force` | float | 2.0 |

### OpenXRDeviceCfg

| Attribute | Type | Default |
|-----------|------|---------|
| `xr_cfg` | XrCfg \| None | None |

XrCfg: `anchor_pos`, `anchor_rot`, `anchor_prim_path`
