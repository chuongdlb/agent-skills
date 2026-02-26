---
name: isaaclab-controllers-and-teleop
description: >
  Implements task-space controllers (DifferentialIK, OperationalSpace, RmpFlow, JointImpedance) and teleoperation devices for IsaacLab.
layer: L3
domain: [robotics, manipulation]
source-project: IsaacLab
depends-on: [isaacsim-robotics, isaaclab-configclass-and-utilities]
tags: [controllers, ik, teleop, manipulation]
---

# IsaacLab Controllers and Teleoperation

Controllers compute joint commands from task-space targets. Teleoperation devices provide human input for demonstration recording and interactive evaluation.

## Architecture

```
Controllers (isaaclab.controllers)
  ├── DifferentialIKController     → Jacobian-based IK
  ├── OperationalSpaceController   → Force/impedance in task space
  ├── JointImpedanceController     → Joint-space impedance
  ├── RmpFlowController            → LULA-based reactive planning
  └── PinkIKController             → PINK whole-body IK solver

Task-Space Action Terms (isaaclab.envs.mdp.actions)
  ├── DifferentialInverseKinematicsAction
  ├── OperationalSpaceAction
  ├── PINKTaskSpaceAction
  └── RMPFlowTaskSpaceAction

Teleoperation Devices (isaaclab.devices)
  ├── Se3Keyboard      → WASD + rotation keys
  ├── Se3Gamepad       → Controller sticks + buttons
  ├── Se3SpaceMouse    → 6-DOF input device
  ├── HaplyDevice      → Haptic feedback device
  └── OpenXRDevice     → VR hand tracking
```

## DifferentialIK Controller

Most common controller for manipulation. Computes joint position targets from task-space pose commands via Jacobian.

```python
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

cfg = DifferentialIKControllerCfg(
    command_type="pose",       # "position" (3D) or "pose" (6/7D)
    use_relative_mode=True,    # Delta commands vs absolute targets
    ik_method="dls",           # IK solver method
    ik_params={"lambda_val": 0.01},
)
controller = DifferentialIKController(cfg, num_envs=N, device="cuda:0")

# In control loop:
controller.set_command(command, ee_pos=ee_pos, ee_quat=ee_quat)
joint_targets = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
```

### IK Methods

| Method | Key | Param | Default | Best For |
|--------|-----|-------|---------|----------|
| Pseudo-inverse | `"pinv"` | `k_val` | 1.0 | General use |
| SVD | `"svd"` | `min_singular_value` | 1e-5 | Singularity handling |
| Transpose | `"trans"` | `k_val` | 1.0 | Stability (slower convergence) |
| Damped least squares | `"dls"` | `lambda_val` | 0.01 | Near singularities (recommended) |

### DifferentialIKControllerCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `command_type` | str | MISSING | `"position"` or `"pose"` |
| `use_relative_mode` | bool | False | Delta vs absolute commands |
| `ik_method` | str | MISSING | `"pinv"`, `"svd"`, `"trans"`, `"dls"` |
| `ik_params` | dict \| None | None | Method-specific parameters |

Action dimensions: position=3, relative pose=6, absolute pose=7 (pos + quat)

## OperationalSpace Controller

Task-space impedance/force control with optional null-space control for redundant manipulators.

```python
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

cfg = OperationalSpaceControllerCfg(
    target_types=["pose_abs"],
    impedance_mode="fixed",
    motion_stiffness_task=100.0,
    motion_damping_ratio_task=1.0,
    motion_control_axes_task=(1, 1, 1, 1, 1, 1),
    nullspace_control="position",
)
controller = OperationalSpaceController(cfg, num_envs=N, device="cuda:0")
```

### OperationalSpaceControllerCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_types` | list[str] | MISSING | `"pose_rel"`, `"pose_abs"`, `"wrench_abs"` |
| `impedance_mode` | str | `"fixed"` | `"fixed"`, `"variable"`, `"variable_kp"` |
| `motion_control_axes_task` | tuple | (1,1,1,1,1,1) | 6D motion mask |
| `contact_wrench_control_axes_task` | tuple | (0,0,0,0,0,0) | 6D wrench mask |
| `motion_stiffness_task` | float | 100.0 | Positional gains |
| `motion_damping_ratio_task` | float | 1.0 | Damping ratios |
| `inertial_dynamics_decoupling` | bool | False | Inverse dynamics |
| `gravity_compensation` | bool | False | Gravity comp |
| `nullspace_control` | str | `"none"` | `"none"` or `"position"` |

## JointImpedance Controller

Joint-space impedance control with configurable stiffness and damping.

### JointImpedanceControllerCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `command_type` | str | `"p_abs"` | `"p_abs"` or `"p_rel"` |
| `impedance_mode` | str | MISSING | `"fixed"`, `"variable"`, `"variable_kp"` |
| `stiffness` | float \| list | MISSING | Positional gains |
| `damping_ratio` | float \| list | None | Damping ratios |
| `inertial_compensation` | bool | False | Inverse dynamics |
| `gravity_compensation` | bool | False | Gravity comp |

## RmpFlow Controller

Real-time reactive motion planner using LULA's RMPflow algorithm.

### RmpFlowControllerCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `"rmp_flow"` | `"rmp_flow"` or `"rmp_flow_smoothed"` |
| `config_file` | str | MISSING | LULA config file path |
| `urdf_file` | str | MISSING | Robot URDF path |
| `collision_file` | str | MISSING | Collision model path |
| `frame_name` | str | MISSING | Target EE frame name |
| `evaluations_per_frame` | float | MISSING | Substeps per frame |

## PinkIK Controller

Whole-body IK using the PINK library for complex kinematic tasks.

### PinkIKControllerCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `urdf_path` | str \| None | None | Robot URDF |
| `variable_input_tasks` | list | MISSING | Controllable frame tasks |
| `fixed_input_tasks` | list | MISSING | Constraint tasks |
| `joint_names` | list[str] \| None | None | Controlled joints |
| `base_link_name` | str | `"base_link"` | Base link |

## Task-Space Action Terms

Wrap controllers as action terms for use in manager-based environments.

### DifferentialInverseKinematicsAction

```python
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg

actions_cfg = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.107),
    ),
    controller=DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    ),
)
```

### Other Task-Space Actions

| Action Term | Controller Used | Description |
|-------------|-----------------|-------------|
| `OperationalSpaceControllerActionCfg` | OperationalSpaceController | Force/impedance control |
| `PinkInverseKinematicsActionCfg` | PinkIKController | Whole-body IK |
| `RMPFlowActionCfg` | RmpFlowController | Reactive motion planning |

## Teleoperation Devices

### Se3Keyboard

```python
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

device = Se3Keyboard(Se3KeyboardCfg(
    pos_sensitivity=0.4,
    rot_sensitivity=0.8,
))
```

Key bindings: W/S (X), A/D (Y), Q/E (Z), Z/X (roll), T/G (pitch), C/V (yaw), K (gripper)

### Se3Gamepad

```python
from isaaclab.devices import Se3Gamepad, Se3GamepadCfg

device = Se3Gamepad(Se3GamepadCfg(
    pos_sensitivity=1.0,
    rot_sensitivity=1.6,
    dead_zone=0.01,
))
```

### Se3SpaceMouse

```python
from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg

device = Se3SpaceMouse(Se3SpaceMouseCfg(
    pos_sensitivity=0.4,
    rot_sensitivity=0.8,
))
```

### HaplyDevice

Haptic feedback device via WebSocket connection:

```python
from isaaclab.devices import HaplyDevice, HaplyDeviceCfg

device = HaplyDevice(HaplyDeviceCfg(
    websocket_uri="ws://localhost:10001",
    pos_sensitivity=1.0,
    data_rate=200.0,
    limit_force=2.0,
))
```

### OpenXRDevice

VR hand tracking via OpenXR:

```python
from isaaclab.devices import OpenXRDevice, OpenXRDeviceCfg

device = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=XrCfg()))
```

### Device API Pattern

All devices follow the same interface:

```python
device.reset()
# In control loop:
raw_data = device.advance()
# raw_data contains delta pose and button states
```

## Teleop Integration

```bash
# Run teleoperation with keyboard
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Lift-Franka-IK-Abs-Gripper-State-v0 \
    --teleop_device keyboard \
    --num_envs 1 \
    --sensitivity 1.0
```

## Reference Files

- [controllers-api-reference.md](controllers-api-reference.md) - Full attribute tables for all controller configs, task-space action configs, and device APIs

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/isaaclab/isaaclab/controllers/differential_ik.py` | DifferentialIK controller |
| `source/isaaclab/isaaclab/controllers/operational_space.py` | OperationalSpace controller |
| `source/isaaclab/isaaclab/controllers/joint_impedance.py` | JointImpedance controller |
| `source/isaaclab/isaaclab/controllers/rmp_flow.py` | RmpFlow controller |
| `source/isaaclab/isaaclab/controllers/pink_ik/` | PinkIK controller |
| `source/isaaclab/isaaclab/devices/keyboard/` | Keyboard device |
| `source/isaaclab/isaaclab/devices/gamepad/` | Gamepad device |
| `source/isaaclab/isaaclab/devices/spacemouse/` | SpaceMouse device |
| `source/isaaclab/isaaclab/devices/haply/` | Haply haptic device |
| `source/isaaclab/isaaclab/devices/openxr/` | OpenXR VR device |
| `source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py` | Task-space action terms |
| `scripts/environments/teleoperation/` | Teleop scripts |
| `scripts/tutorials/05_controllers/` | Controller tutorials |
