# Robot Types Reference

Pre-configured robots available in Isaac Sim with asset paths, DOF information, and supported motion planners.

## Asset Root

All robot assets are under `{assets_root}/Isaac/Robots/` where `assets_root` is obtained via:
```python
from isaacsim.storage.native import get_assets_root_path
assets_root = get_assets_root_path()
```

## Manipulators

### Universal Robots

| Robot | USD Path | Motion Planners |
|-------|----------|-----------------|
| UR3 | `Isaac/Robots/UniversalRobots/ur3/ur3.usd` | RMPflow |
| UR3e | `Isaac/Robots/UniversalRobots/ur3e/ur3e.usd` | RMPflow |
| UR5 | `Isaac/Robots/UniversalRobots/ur5/ur5.usd` | RMPflow |
| UR5e | `Isaac/Robots/UniversalRobots/ur5e/ur5e.usd` | RMPflow |
| UR10 | `Isaac/Robots/UniversalRobots/ur10/ur10.usd` | RMPflow |
| UR10e | `Isaac/Robots/UniversalRobots/ur10e/ur10e.usd` | RMPflow |
| UR16e | `Isaac/Robots/UniversalRobots/ur16e/ur16e.usd` | RMPflow |

### Franka Emika

| Robot | USD Path | Motion Planners |
|-------|----------|-----------------|
| Franka | `Isaac/Robots/Franka/franka_alt_fingers.usd` | RMPflow (3 variants) |
| FR3 | `Isaac/Robots/Franka/fr3.usd` | RMPflow |

Franka RMPflow variants: standard, NoFeedback, Cortex.

### Denso

| Robot | USD Path | Motion Planners |
|-------|----------|-----------------|
| Cobotta Pro 900 | `Isaac/Robots/Denso/cobotta_pro_900.usd` | RMPflow |
| Cobotta Pro 1300 | `Isaac/Robots/Denso/cobotta_pro_1300.usd` | RMPflow |

### Flexiv

| Robot | USD Path | Motion Planners |
|-------|----------|-----------------|
| Rizon4 | `Isaac/Robots/Flexiv/rizon4.usd` | RMPflow |

### Kawasaki

| Robot | USD Path | Motion Planners |
|-------|----------|-----------------|
| RS007L | `Isaac/Robots/Kawasaki/RS007L.usd` | RMPflow |
| RS007N | `Isaac/Robots/Kawasaki/RS007N.usd` | RMPflow |
| RS013N | `Isaac/Robots/Kawasaki/RS013N.usd` | RMPflow |
| RS025N | `Isaac/Robots/Kawasaki/RS025N.usd` | RMPflow |
| RS080N | `Isaac/Robots/Kawasaki/RS080N.usd` | RMPflow |

### Other Manipulators

| Robot | USD Path | Motion Planners |
|-------|----------|-----------------|
| FestoCobot | `Isaac/Robots/Festo/FestoCobot.usd` | RMPflow |
| Techman TM12 | `Isaac/Robots/Techman/TM12.usd` | RMPflow |
| Kuka KR210 | `Isaac/Robots/Kuka/KR210.usd` | RMPflow |
| Fanuc CRX10IAL | `Isaac/Robots/Fanuc/CRX10IAL.usd` | RMPflow |

## Wheeled Robots

| Robot | Class | Key Features |
|-------|-------|-------------|
| Carter v1 | WheeledRobot | Differential drive, Lidar |
| Carter v2 (Nova Carter) | WheeledRobot | Stereo cameras, RTX Lidar |

## Quadrupeds and Humanoids (Policy-Based)

Available in `isaacsim.robot.policy.examples`:

| Robot | Policy | Description |
|-------|--------|-------------|
| ANYmal | Locomotion | Quadruped from ANYbotics |
| Spot | Locomotion | Boston Dynamics quadruped |
| H1 | Locomotion | Unitree humanoid |
| Franka (Drawer) | FrankaOpenDrawerPolicy | Cabinet drawer opening |

## RMPflow Configuration

RMPflow configs are stored in:
```
source/extensions/isaacsim.robot_motion.motion_generation/motion_policy_configs/
```

The policy map (`policy_map.json`) maps robot names to their configuration files.

### Loading a Robot Config

```python
from isaacsim.robot_motion.motion_generation import interface_config_loader

# List available robots
robots = interface_config_loader.get_supported_robots()

# Load RMPflow config for a robot
config = interface_config_loader.load_supported_motion_policy_config(
    "UR10",     # Robot name (from policy_map.json)
    "RMPflow"   # Policy type
)
```

## Creating Custom Robot Configurations

To add a new robot with RMPflow support:

1. Create URDF/USD of the robot
2. Generate a LULA robot description YAML
3. Create RMPflow configuration YAML
4. Add entry to `policy_map.json`

## Gripper Variants

Many robot USDs include gripper variants accessible via USD variant sets. For example, the UR10 has `Short_Suction` and other gripper options selectable through prim variants.

## Standalone Examples

| Example | Path |
|---------|------|
| Franka pick-up | `source/standalone_examples/api/isaacsim.robot.manipulators/franka_pick_up.py` |
| UR10 RMPflow | `source/standalone_examples/api/isaacsim.robot.manipulators/ur10/` |
| Cobotta examples | `source/standalone_examples/api/isaacsim.robot.manipulators/cobotta/` |
| Getting started | `source/standalone_examples/tutorials/getting_started_robot.py` |
| Policy robots | `source/standalone_examples/api/isaacsim.robot.policy.examples/` |
