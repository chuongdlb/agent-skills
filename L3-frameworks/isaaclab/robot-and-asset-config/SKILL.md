---
name: isaaclab-robot-and-asset-config
description: >
  Configures robots, rigid objects, deformable objects, and articulations — ArticulationCfg, actuator models, spawner configs, physics properties, robot catalog.
layer: L3
domain: [robotics, manipulation, locomotion]
source-project: IsaacLab
depends-on: [isaacsim-asset-pipeline, isaacsim-robotics, isaaclab-configclass-and-utilities]
tags: [robots, actuators, articulation, assets]
---

# IsaacLab Robot and Asset Configuration

Assets are the core simulated entities in IsaacLab. Every robot, object, and interactable element in a scene is configured through the asset system.

## Asset Type Hierarchy

```
AssetBaseCfg
  ├── ArticulationCfg          # Robots with joints and actuators
  ├── RigidObjectCfg            # Single rigid bodies (boxes, balls, tools)
  ├── DeformableObjectCfg       # Soft/deformable bodies (cloth, soft objects)
  └── SurfaceGripperCfg         # Suction/surface grippers

RigidObjectCollectionCfg         # Collection of multiple RigidObjectCfg instances
```

All asset cfgs inherit from `AssetBaseCfg` which provides `prim_path`, `spawn`, `init_state`, `collision_group`, and `debug_vis`.

## ArticulationCfg Anatomy

`ArticulationCfg` is the primary configuration for robots. It extends `AssetBaseCfg` with joint states and actuators.

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

FRANKA_PANDA_CFG = ArticulationCfg(
    # Prim path with environment regex placeholder
    prim_path="{ENV_REGEX_NS}/Robot",

    # Spawner: how to load the asset into the scene
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),

    # Initial state: root pose and joint positions
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),                        # (x, y, z)
        rot=(1.0, 0.0, 0.0, 0.0),                   # (w, x, y, z) quaternion
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,            # regex matching
        },
        joint_vel={".*": 0.0},
    ),

    # Actuators: dict mapping group names to actuator configs
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],   # regex joint selection
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

### Key ArticulationCfg Fields

| Field | Type | Description |
|-------|------|-------------|
| `prim_path` | `str` | USD prim path or expression with `{ENV_REGEX_NS}` |
| `spawn` | `SpawnerCfg` | How to load the asset (UsdFileCfg, UrdfFileCfg, etc.) |
| `init_state` | `InitialStateCfg` | Root pose + joint positions/velocities (regex keys) |
| `actuators` | `dict[str, ActuatorBaseCfg]` | Named actuator groups with joint assignments |
| `soft_joint_pos_limit_factor` | `float` | Scale factor for soft joint limits (default 1.0) |
| `articulation_root_prim_path` | `str \| None` | Relative path to articulation root under prim_path |

## Actuator Model Hierarchy

```
ActuatorBaseCfg
  └── ImplicitActuatorCfg          # Physics engine PD (simplest, fastest)
  └── IdealPDActuatorCfg           # Explicit PD with torque clipping
        ├── DCMotorCfg             # + torque saturation curve
        │     ├── ActuatorNetLSTMCfg   # Neural network (LSTM)
        │     └── ActuatorNetMLPCfg    # Neural network (MLP)
        └── DelayedPDActuatorCfg   # + command delay (min/max steps)
              └── RemotizedPDActuatorCfg  # + joint-angle lookup table
```

### Actuator Selection Guide

| Actuator | Use Case | Key Params |
|----------|----------|------------|
| `ImplicitActuatorCfg` | Default for most robots. PD handled by physics engine. | `stiffness`, `damping` |
| `IdealPDActuatorCfg` | When you need explicit torque computation and clipping. | `stiffness`, `damping`, `effort_limit` |
| `DCMotorCfg` | Realistic DC motor with torque saturation. | `saturation_effort` |
| `DelayedPDActuatorCfg` | Simulating communication/actuator delays. | `min_delay`, `max_delay` |
| `RemotizedPDActuatorCfg` | Cable-driven joints with nonlinear transmission. | `joint_parameter_lookup` |
| `ActuatorNetLSTMCfg` | Learned actuator dynamics (e.g., ANYmal). | `network_file` |
| `ActuatorNetMLPCfg` | Learned actuator dynamics (e.g., Go1). | `network_file`, `pos_scale`, `vel_scale`, `torque_scale` |

### ActuatorBaseCfg Common Attributes

All actuator configs inherit these fields:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `joint_names_expr` | `list[str]` | MISSING | Regex expressions matching joint names |
| `effort_limit` | `dict\|float\|None` | None | Torque limit for actuator model clipping |
| `velocity_limit` | `dict\|float\|None` | None | Velocity limit for actuator model |
| `effort_limit_sim` | `dict\|float\|None` | None | Torque limit applied by physics solver |
| `velocity_limit_sim` | `dict\|float\|None` | None | Velocity limit applied by physics solver |
| `stiffness` | `dict\|float\|None` | MISSING | P-gain (position stiffness) |
| `damping` | `dict\|float\|None` | MISSING | D-gain (velocity damping) |
| `armature` | `dict\|float\|None` | None | Added joint-space inertia for stability |
| `friction` | `dict\|float\|None` | None | Static joint friction coefficient |
| `dynamic_friction` | `dict\|float\|None` | None | Dynamic joint friction coefficient |
| `viscous_friction` | `dict\|float\|None` | None | Viscous joint friction coefficient |

Use `None` for any limit/property to inherit the value from the USD file. Use `dict` with regex keys for per-joint values:

```python
stiffness={
    ".*_hip_.*": 150.0,
    ".*_knee": 200.0,
},
```

## Spawner Types

| Spawner | Import | Description |
|---------|--------|-------------|
| `UsdFileCfg` | `sim_utils.UsdFileCfg` | Load from USD file (most common) |
| `UrdfFileCfg` | `sim_utils.UrdfFileCfg` | Convert and load from URDF |
| `MjcfFileCfg` | `sim_utils.MjcfFileCfg` | Convert and load from MuJoCo XML |
| `GroundPlaneCfg` | `sim_utils.GroundPlaneCfg` | Standard ground plane |
| `SphereCfg` | `sim_utils.SphereCfg` | Primitive sphere shape |
| `CuboidCfg` | `sim_utils.CuboidCfg` | Primitive box shape |
| `CylinderCfg` | `sim_utils.CylinderCfg` | Primitive cylinder shape |
| `CapsuleCfg` | `sim_utils.CapsuleCfg` | Primitive capsule shape |
| `ConeCfg` | `sim_utils.ConeCfg` | Primitive cone shape |

### UsdFileCfg Inline Properties

UsdFileCfg (and all file spawners) accept inline physics property overrides:

```python
spawn=sim_utils.UsdFileCfg(
    usd_path="path/to/robot.usd",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
        fix_root_link=False,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        contact_offset=0.005,
        rest_offset=0.0,
    ),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0),
    variants={"Gripper": "Robotiq_2F_85"},
    scale=(1.0, 1.0, 1.0),
)
```

## Physics Property Configs

| Config | Key Attributes |
|--------|----------------|
| `ArticulationRootPropertiesCfg` | `enabled_self_collisions`, `solver_position_iteration_count`, `solver_velocity_iteration_count`, `fix_root_link` |
| `RigidBodyPropertiesCfg` | `disable_gravity`, `max_depenetration_velocity`, `max_linear_velocity`, `max_angular_velocity`, `retain_accelerations` |
| `CollisionPropertiesCfg` | `collision_enabled`, `contact_offset`, `rest_offset` |
| `MassPropertiesCfg` | `mass`, `density` |
| `JointDrivePropertiesCfg` | `drive_type` ("force" or "acceleration") |
| `FixedTendonPropertiesCfg` | `stiffness`, `damping`, `limit_stiffness` |
| `DeformableBodyPropertiesCfg` | `self_collision`, `solver_position_iteration_count`, `vertex_velocity_damping` |

## Pre-Built Robot Catalog (Summary)

| Robot | Config Variable | Category | Actuator |
|-------|----------------|----------|----------|
| Franka Panda | `FRANKA_PANDA_CFG` | Manipulator | Implicit |
| Franka High PD | `FRANKA_PANDA_HIGH_PD_CFG` | Manipulator | Implicit |
| UR10 | `UR10_CFG` | Manipulator | Implicit |
| Unitree A1 | `UNITREE_A1_CFG` | Quadruped | DCMotor |
| Unitree Go1 | `UNITREE_GO1_CFG` | Quadruped | ActuatorNetMLP |
| Unitree Go2 | `UNITREE_GO2_CFG` | Quadruped | DCMotor |
| Unitree H1 | `H1_CFG` | Humanoid | Implicit |
| Unitree G1 | `G1_CFG` | Humanoid | Implicit |
| ANYmal B/C/D | `ANYMAL_B_CFG` etc. | Quadruped | ActuatorNetLSTM |
| Spot | `SPOT_CFG` | Quadruped | Delayed+Remotized PD |
| Allegro Hand | `ALLEGRO_HAND_CFG` | Hand | Implicit |
| Shadow Hand | `SHADOW_HAND_CFG` | Hand | Implicit |

Full catalog with all variants in [robot-catalog.md](robot-catalog.md).

## .replace() Customization Pattern

Use `.replace()` to create modified copies of pre-built configs:

```python
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

# Create a variant with different stiffness
my_franka = FRANKA_PANDA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=400.0,    # stiffer for IK control
            damping=80.0,
        ),
        # ... other actuator groups
    },
)
```

For shallow modifications, use `.copy()` and mutate:

```python
my_franka = FRANKA_PANDA_CFG.copy()
my_franka.spawn.rigid_props.disable_gravity = True
my_franka.actuators["panda_shoulder"].stiffness = 400.0
my_franka.actuators["panda_shoulder"].damping = 80.0
```

## Regex Matching for Joint/Body Names

Joint names in `joint_names_expr`, `joint_pos`, `stiffness`, `damping`, and other dict-keyed fields use Python regex:

| Pattern | Matches |
|---------|---------|
| `".*"` | All joints |
| `"panda_joint[1-4]"` | panda_joint1 through panda_joint4 |
| `"panda_joint[5-7]"` | panda_joint5 through panda_joint7 |
| `"panda_finger_joint.*"` | All finger joints |
| `".*_hip_joint"` | All hip joints (e.g., FL_hip_joint, FR_hip_joint) |
| `".*_hip_.*"` | All joints containing "_hip_" |
| `"F[L,R]_thigh_joint"` | FL_thigh_joint and FR_thigh_joint |
| `".*HAA"` | All hip abduction/adduction joints |
| `"^(?!thumb_joint_0).*"` | All joints except thumb_joint_0 |
| `"robot0_(FF\|MF\|RF)J(3\|2)"` | Specific finger joints with alternation |

## {ENV_REGEX_NS} Placeholder

The `{ENV_REGEX_NS}` placeholder in `prim_path` is replaced at runtime with the environment namespace regex, typically expanding to `/World/envs/env_.*/`. This enables multi-environment cloning:

```python
prim_path="{ENV_REGEX_NS}/Robot"
# Expands to: /World/envs/env_.*/Robot
```

## Reference Files

- [asset-configuration-api.md](asset-configuration-api.md) - Full attribute tables for all asset, actuator, spawner, and physics property configs
- [robot-catalog.md](robot-catalog.md) - Complete catalog of every pre-built robot in isaaclab_assets with config names, imports, types, actuators, and variants

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/isaaclab/isaaclab/assets/asset_base_cfg.py` | AssetBaseCfg base class |
| `source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py` | ArticulationCfg |
| `source/isaaclab/isaaclab/assets/rigid_object/rigid_object_cfg.py` | RigidObjectCfg |
| `source/isaaclab/isaaclab/assets/deformable_object/deformable_object_cfg.py` | DeformableObjectCfg |
| `source/isaaclab/isaaclab/assets/rigid_object_collection/rigid_object_collection_cfg.py` | RigidObjectCollectionCfg |
| `source/isaaclab/isaaclab/assets/surface_gripper/surface_gripper_cfg.py` | SurfaceGripperCfg |
| `source/isaaclab/isaaclab/actuators/actuator_base_cfg.py` | ActuatorBaseCfg |
| `source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py` | PD-based actuator configs |
| `source/isaaclab/isaaclab/actuators/actuator_net_cfg.py` | Neural network actuator configs |
| `source/isaaclab/isaaclab/sim/spawners/spawner_cfg.py` | SpawnerCfg, RigidObjectSpawnerCfg |
| `source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py` | UsdFileCfg, UrdfFileCfg, MjcfFileCfg, GroundPlaneCfg |
| `source/isaaclab/isaaclab/sim/spawners/shapes/shapes_cfg.py` | Shape spawner configs |
| `source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py` | Physics property configs |
| `source/isaaclab_assets/isaaclab_assets/robots/` | All pre-built robot configurations |
