---
name: isaacsim-robotics
description: >
  Develops robot configurations, manipulators, wheeled robots, grippers, motion planning, and control systems in Isaac Sim.
layer: L2
domain: [robotics, manipulation, locomotion]
source-project: IsaacSim
depends-on: [isaacsim-simulation-core, isaacsim-asset-pipeline]
tags: [robots, manipulators, motion-planning, controllers]
---

# Isaac Sim Robotics

The robotics stack spans three extension groups:
- `isaacsim.robot.*` - Robot types, grippers, schema
- `isaacsim.robot_motion.*` - Motion generation, RMPflow, LULA
- `isaacsim.robot_setup.*` - Configuration tools (wizard, gain tuner, assembler)

## Robot Class Hierarchy

```
SingleArticulation (isaacsim.core.prims)
  └── Robot (isaacsim.core.api)
        ├── SingleManipulator (isaacsim.robot.manipulators)
        │     Has: end_effector (SingleRigidPrim), gripper (Gripper)
        └── WheeledRobot (isaacsim.robot.wheeled_robots)
              Has: wheel_dof_names, wheel_dof_indices
```

## Adding a Robot to the Scene

```python
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
import numpy as np

world = World(stage_units_in_meters=1.0)

# Method 1: Direct USD loading
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

assets_root = get_assets_root_path()
add_reference_to_stage(
    usd_path=f"{assets_root}/Isaac/Robots/UniversalRobots/ur10/ur10.usd",
    prim_path="/World/ur10"
)
robot = world.scene.add(SingleArticulation(prim_path="/World/ur10", name="ur10"))

world.reset()
print(f"DOFs: {robot.num_dof}, Names: {robot.dof_names}")
```

## SingleManipulator

A manipulator with a single end-effector and optional gripper:

```python
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper

# Configure gripper
gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.04, 0.04]),
    joint_closed_positions=np.array([0.0, 0.0]),
    action_deltas=np.array([0.04, 0.04]),
)

# Create manipulator with gripper
manipulator = world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="franka",
        end_effector_prim_name="panda_rightfinger",
        gripper=gripper,
    )
)

world.reset()
manipulator.gripper.open()
```

**Key properties:**
- `end_effector` → SingleRigidPrim
- `gripper` → Gripper (ParallelGripper or SurfaceGripper)

## WheeledRobot

For differential-drive and other wheeled platforms:

```python
from isaacsim.robot.wheeled_robots import WheeledRobot

carter = world.scene.add(
    WheeledRobot(
        prim_path="/World/Carter",
        name="carter",
        wheel_dof_names=["left_wheel", "right_wheel"],
    )
)

world.reset()
# Applies velocity control to wheel joints
from isaacsim.core.utils.types import ArticulationAction
carter.apply_wheel_actions(
    ArticulationAction(joint_velocities=np.array([5.0, 5.0]))
)
```

**Key methods:**
- `get_wheel_positions()` / `set_wheel_positions()`
- `get_wheel_velocities()` / `set_wheel_velocities()`
- `apply_wheel_actions(ArticulationAction)` - Applies actions to wheels only

## Gripper Types

### ParallelGripper

Two-finger parallel gripper with mimic joint support:

```python
from isaacsim.robot.manipulators.grippers import ParallelGripper

gripper = ParallelGripper(
    end_effector_prim_path="/World/Robot/ee_link",
    joint_prim_names=["finger_joint1", "finger_joint2"],
    joint_opened_positions=np.array([0.04, 0.04]),
    joint_closed_positions=np.array([0.0, 0.0]),
    action_deltas=np.array([0.04, 0.04]),
)
```

### SurfaceGripper

Suction-cup style gripper using physics contact:

```python
from isaacsim.robot.manipulators.grippers import SurfaceGripper

gripper = SurfaceGripper(
    end_effector_prim_path="/World/Robot/ee_link",
    surface_gripper_path="/World/Robot/ee_link/SurfaceGripper",
)
gripper.open()
gripper.close()
is_holding = gripper.is_closed()
```

## Controller Pattern

All controllers inherit from `BaseController` and return `ArticulationAction`:

```python
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.types import ArticulationAction

class MyController(BaseController):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, target_position, **kwargs) -> ArticulationAction:
        # Compute joint commands
        return ArticulationAction(joint_positions=computed_positions)

    def reset(self):
        pass
```

### Applying Control

```python
# Via ArticulationController (PD control)
controller = robot.get_articulation_controller()
controller.set_gains(kps=np.array([1e4]*7), kds=np.array([1e2]*7))
controller.apply_action(ArticulationAction(joint_positions=target))

# Via higher-level controller
pick_place = PickPlaceController(name="pp", cspace_controller=rmpflow,
                                  gripper=gripper)
action = pick_place.forward(picking_position, placing_position, ...)
robot.apply_action(action)
```

### PickPlaceController

10-phase state machine for pick-and-place tasks:

1. Move above pick position
2. Lower to object
3. Wait for settling
4. Close gripper
5. Lift object
6. Move to place XY
7. Move to place Z
8. Open gripper
9. Move up
10. Return to start

```python
from isaacsim.robot.manipulators.controllers import PickPlaceController

controller = PickPlaceController(
    name="pick_place",
    cspace_controller=rmpflow_controller,
    gripper=manipulator.gripper,
    end_effector_initial_height=0.4,
    events_dt=[0.008] * 10,
)
```

## Motion Planning

### RMPflow (Real-time Reactive)

```python
from isaacsim.robot_motion.motion_generation import (
    RmpFlow, ArticulationMotionPolicy, interface_config_loader
)

# Load config for specific robot
rmpflow_config = interface_config_loader.load_supported_motion_policy_config(
    "Franka", "RMPflow"
)
rmpflow = RmpFlow(**rmpflow_config)

# Wrap as articulation motion policy
policy = ArticulationMotionPolicy(robot, rmpflow)

# Set target and compute
rmpflow.set_end_effector_target(target_position, target_orientation)
action = policy.get_next_articulation_action()
robot.apply_action(action)
```

### Supported Robots with RMPflow

Franka, FR3, UR3/3e/5/5e/10/10e/16e, Rizon4, Cobotta Pro 900/1300, Kawasaki RS007L/RS007N/RS013N/RS025N/RS080N, FestoCobot, Techman TM12, Kuka KR210, Fanuc CRX10IAL.

Config files: `source/extensions/isaacsim.robot_motion.motion_generation/motion_policy_configs/`

### LULA Motion Generation

Lower-level motion generation via the Lula library (`isaacsim.robot_motion.lula`). Provides C++ bindings for trajectory optimization and kinematics.

## Task Pattern

Tasks encapsulate complete robot scenarios:

```python
from isaacsim.core.api.tasks import PickPlace, FollowTarget

# PickPlace task (abstract - use robot-specific subclass)
class MyPickPlace(PickPlace):
    def set_robot(self):
        return my_robot_instance  # Return configured robot

# FollowTarget task
class MyFollowTarget(FollowTarget):
    def set_robot(self):
        return my_robot_instance

world.add_task(MyPickPlace())
world.reset()
```

Built-in abstract tasks in `isaacsim.core.api.tasks`:
- `PickPlace` - Pick and place with configurable cube
- `FollowTarget` - End-effector target tracking
- `Stacking` - Stack objects

## Robot Setup Tools

| Tool | Extension | Purpose |
|------|-----------|---------|
| Robot Wizard | `isaacsim.robot_setup.wizard` | Guided robot import and configuration |
| Gain Tuner | `isaacsim.robot_setup.gain_tuner` | Interactive PD gain tuning with plots |
| Grasp Editor | `isaacsim.robot_setup.grasp_editor` | Author grasp poses for grippers |
| Assembler | `isaacsim.robot_setup.assembler` | Combine articulations into one robot |
| XRDF Editor | `isaacsim.robot_setup.xrdf_editor` | Edit extended robot definition files |

## Reference Files

- [robot-types.md](robot-types.md) - Pre-configured robot catalog with asset paths
- [controllers-and-tasks.md](controllers-and-tasks.md) - Controller and task API details

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/extensions/isaacsim.robot.manipulators/` | Manipulators, grippers, controllers |
| `source/extensions/isaacsim.robot.wheeled_robots/` | Wheeled robot class |
| `source/extensions/isaacsim.robot.surface_gripper/` | Surface gripper C++ plugin |
| `source/extensions/isaacsim.robot_motion.motion_generation/` | RMPflow, motion policies |
| `source/extensions/isaacsim.robot_motion.lula/` | LULA motion library |
| `source/standalone_examples/api/isaacsim.robot.manipulators/` | Manipulator examples |
| `source/standalone_examples/api/isaacsim.robot.policy.examples/` | Policy examples |
