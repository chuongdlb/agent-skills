# Controllers and Tasks Reference

Complete API reference for robot controllers, articulation actions, and task patterns.

## ArticulationAction

The standard command type for joint-level robot control:

```python
from isaacsim.core.utils.types import ArticulationAction

action = ArticulationAction(
    joint_positions=np.array([...]),     # Desired joint positions (radians)
    joint_velocities=np.array([...]),    # Desired joint velocities (rad/s)
    joint_efforts=np.array([...]),       # Desired joint torques (Nm)
    joint_indices=np.array([...]),       # Optional: indices of joints to control
)
```

Fields can be `None` (unused). The ArticulationController interprets which fields are set to determine the control mode:
- Only `joint_positions` set → position control
- Only `joint_velocities` set → velocity control
- Only `joint_efforts` set → effort/torque control
- Mixed → per-DOF mode must be configured

## BaseController Interface

**File:** `source/extensions/isaacsim.core.api/python/impl/controllers/base_controller.py`

```python
from abc import ABC, abstractmethod
from isaacsim.core.utils.types import ArticulationAction

class BaseController(ABC):
    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArticulationAction:
        """Compute control action. Subclasses implement specific control logic."""
        raise NotImplementedError

    def reset(self):
        """Reset controller state between episodes."""
        pass

    @property
    def name(self) -> str:
        return self._name
```

## ArticulationController

Low-level PD controller for joint control:

```python
controller = robot.get_articulation_controller()

# Configure gains
controller.set_gains(
    kps=np.array([1e4, 1e4, 1e4, 1e4, 1e3, 1e3, 1e3]),  # Position gains
    kds=np.array([1e2, 1e2, 1e2, 1e2, 1e1, 1e1, 1e1]),  # Velocity gains
    save_to_usd=False,  # Persist to USD stage
)

# Apply control
controller.apply_action(ArticulationAction(
    joint_positions=target_positions
))

# Switch control modes
controller.switch_control_mode("position")  # All DOFs to position control
controller.switch_control_mode("velocity")  # All DOFs to velocity control
controller.switch_control_mode("effort")    # All DOFs to effort control

# Per-DOF mode
controller.switch_dof_control_mode(dof_index=6, mode="effort")

# Query joint limits
lower_limits, upper_limits = controller.get_joint_limits()

# Query/set max efforts
max_efforts = controller.get_max_efforts()
controller.set_max_efforts(max_efforts * 0.5)
```

## Built-in Controllers

### PickPlaceController

**File:** `source/extensions/isaacsim.robot.manipulators/python/controllers/pick_place_controller.py`

```python
from isaacsim.robot.manipulators.controllers import PickPlaceController

controller = PickPlaceController(
    name="pick_place",
    cspace_controller=rmpflow,       # Motion planner (BaseController)
    gripper=manipulator.gripper,      # Gripper instance
    end_effector_initial_height=0.4,  # Height for approach/retreat
    events_dt=[0.008] * 10,          # Duration per phase (10 phases)
)

# In step loop:
action = controller.forward(
    picking_position=np.array([0.5, 0.0, 0.02]),
    placing_position=np.array([0.5, 0.3, 0.02]),
    current_joint_positions=robot.get_joint_positions(),
    end_effector_offset=np.array([0, 0, 0.16]),
)
robot.apply_action(action)

# Check state
event = controller.get_current_event()  # Current phase (0-9)
paused = controller.is_paused()         # Waiting state
controller.reset()                       # Reset state machine
```

**10 phases:** approach above → lower → wait → close gripper → lift → move XY → move Z → open gripper → retreat up → return.

### StackingController

Similar to PickPlaceController but for stacking multiple objects at a target location.

## BaseTask Abstract Methods

**File:** `source/extensions/isaacsim.core.api/python/impl/tasks/base_task.py`

```python
class BaseTask:
    def __init__(self, name: str, offset=None):
        self._name = name
        self._offset = offset  # Task frame offset

    # MUST override:
    def set_up_scene(self, scene: Scene):
        """Add objects to the scene. Called during first world.reset()."""
        self._scene = scene

    # SHOULD override:
    def get_observations(self) -> dict:
        """Return task observations for the current state."""
        return {}

    def pre_step(self, time_step_index: int, simulation_time: float):
        """Called before each physics step."""
        pass

    def post_reset(self):
        """Called after each world.reset(). Reset internal task state."""
        pass

    def is_done(self) -> bool:
        """Return True when task is complete."""
        return False

    def calculate_metrics(self) -> dict:
        """Return task metrics."""
        return {}

    def get_task_objects(self) -> dict:
        """Return dict of {name: object} for task objects."""
        return {}
```

## Built-in Task Classes

### PickPlace Task

**File:** `source/extensions/isaacsim.core.api/python/impl/tasks/pick_place.py`

```python
from isaacsim.core.api.tasks import PickPlace

class MyPickPlace(PickPlace):
    def set_robot(self):
        """Must return a configured Robot/SingleManipulator instance."""
        return SingleManipulator(
            prim_path="/World/Robot",
            name="my_robot",
            end_effector_prim_name="ee_link",
            gripper=my_gripper,
        )

# Constructor params
task = MyPickPlace(
    name="my_pick_place",
    cube_initial_position=np.array([0.5, 0.0, 0.05]),
    cube_initial_orientation=None,
    target_position=np.array([0.5, 0.3, 0.05]),
    cube_size=0.05,
    offset=None,
)
```

Creates a DynamicCuboid at `cube_initial_position` and sets up the pick-place scenario.

### FollowTarget Task

**File:** `source/extensions/isaacsim.core.api/python/impl/tasks/follow_target.py`

```python
from isaacsim.core.api.tasks import FollowTarget

class MyFollowTarget(FollowTarget):
    def set_robot(self):
        return my_robot_instance

task = MyFollowTarget(
    name="follow",
    target_prim_path=None,         # Auto-created if None
    target_name="target",
    target_position=np.array([0.5, 0, 0.5]),
    target_orientation=None,
    offset=None,
)
```

Creates a visual cube as a target that the robot's end-effector should track.

## Complete Robot Control Example

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.controllers import PickPlaceController
from isaacsim.robot_motion.motion_generation import (
    RmpFlow, ArticulationMotionPolicy, interface_config_loader
)
import numpy as np

world = World(stage_units_in_meters=1.0)

# 1. Add robot
gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.04, 0.04]),
    joint_closed_positions=np.array([0.0, 0.0]),
    action_deltas=np.array([0.04, 0.04]),
)

robot = world.scene.add(SingleManipulator(
    prim_path="/World/Franka",
    name="franka",
    end_effector_prim_name="panda_rightfinger",
    gripper=gripper,
))

world.scene.add_default_ground_plane()

# 2. Setup controller
rmpflow_config = interface_config_loader.load_supported_motion_policy_config(
    "Franka", "RMPflow"
)
rmpflow = RmpFlow(**rmpflow_config)
controller = PickPlaceController(
    name="pp", cspace_controller=rmpflow, gripper=gripper
)

# 3. Run
world.reset()
for i in range(1000):
    world.step(render=True)
    action = controller.forward(
        picking_position=np.array([0.5, 0.0, 0.02]),
        placing_position=np.array([0.5, 0.3, 0.02]),
        current_joint_positions=robot.get_joint_positions(),
    )
    robot.apply_action(action)

simulation_app.close()
```
