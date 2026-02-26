# Asset Configuration API Reference

Complete attribute tables for all asset, actuator, spawner, and physics property configuration classes.

## Asset Configuration Classes

### AssetBaseCfg

Base class for all asset configurations. Source: `source/isaaclab/isaaclab/assets/asset_base_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type[AssetBase]` | `None` | Associated asset class (None = spawn only, no interaction) |
| `prim_path` | `str` | MISSING | USD prim path or expression (supports `{ENV_REGEX_NS}`) |
| `spawn` | `SpawnerCfg \| None` | `None` | Spawner config (None = asset already in scene) |
| `init_state` | `InitialStateCfg` | `InitialStateCfg()` | Initial state of the asset |
| `collision_group` | `Literal[0, -1]` | `0` | `0`: local (same env), `-1`: global (all envs) |
| `debug_vis` | `bool` | `False` | Enable debug visualization |

### AssetBaseCfg.InitialStateCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `pos` | `tuple[float, float, float]` | `(0.0, 0.0, 0.0)` | Root position in world frame |
| `rot` | `tuple[float, float, float, float]` | `(1.0, 0.0, 0.0, 0.0)` | Root quaternion (w, x, y, z) in world frame |

### ArticulationCfg

Extends `AssetBaseCfg`. Source: `source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `Articulation` | Always `Articulation` |
| `articulation_root_prim_path` | `str \| None` | `None` | Relative path to articulation root under prim_path (must start with `/`) |
| `init_state` | `ArticulationCfg.InitialStateCfg` | `InitialStateCfg()` | Initial state including joint positions |
| `soft_joint_pos_limit_factor` | `float` | `1.0` | Scale factor for soft joint position limits |
| `actuators` | `dict[str, ActuatorBaseCfg]` | MISSING | Actuator groups mapped by name |
| `actuator_value_resolution_debug_print` | `bool` | `False` | Print debug info when cfg differs from USD |

### ArticulationCfg.InitialStateCfg

Extends `AssetBaseCfg.InitialStateCfg`.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `pos` | `tuple[float, float, float]` | `(0.0, 0.0, 0.0)` | Root position |
| `rot` | `tuple[float, float, float, float]` | `(1.0, 0.0, 0.0, 0.0)` | Root quaternion (w, x, y, z) |
| `lin_vel` | `tuple[float, float, float]` | `(0.0, 0.0, 0.0)` | Linear velocity of root |
| `ang_vel` | `tuple[float, float, float]` | `(0.0, 0.0, 0.0)` | Angular velocity of root |
| `joint_pos` | `dict[str, float]` | `{".*": 0.0}` | Joint positions (regex keys) |
| `joint_vel` | `dict[str, float]` | `{".*": 0.0}` | Joint velocities (regex keys) |

### RigidObjectCfg

Extends `AssetBaseCfg`. Source: `source/isaaclab/isaaclab/assets/rigid_object/rigid_object_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `RigidObject` | Always `RigidObject` |
| `init_state` | `RigidObjectCfg.InitialStateCfg` | `InitialStateCfg()` | Initial state with velocity |

### RigidObjectCfg.InitialStateCfg

Extends `AssetBaseCfg.InitialStateCfg`.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `pos` | `tuple[float, float, float]` | `(0.0, 0.0, 0.0)` | Root position |
| `rot` | `tuple[float, float, float, float]` | `(1.0, 0.0, 0.0, 0.0)` | Root quaternion (w, x, y, z) |
| `lin_vel` | `tuple[float, float, float]` | `(0.0, 0.0, 0.0)` | Linear velocity |
| `ang_vel` | `tuple[float, float, float]` | `(0.0, 0.0, 0.0)` | Angular velocity |

### DeformableObjectCfg

Extends `AssetBaseCfg`. Source: `source/isaaclab/isaaclab/assets/deformable_object/deformable_object_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `DeformableObject` | Always `DeformableObject` |
| `visualizer_cfg` | `VisualizationMarkersCfg` | `DEFORMABLE_TARGET_MARKER_CFG` | Debug visualization markers |

### RigidObjectCollectionCfg

Not a subclass of `AssetBaseCfg`. Source: `source/isaaclab/isaaclab/assets/rigid_object_collection/rigid_object_collection_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `RigidObjectCollection` | Always `RigidObjectCollection` |
| `rigid_objects` | `dict[str, RigidObjectCfg]` | MISSING | Named dict of rigid object configs |

### SurfaceGripperCfg

Extends `AssetBaseCfg`. Source: `source/isaaclab/isaaclab/assets/surface_gripper/surface_gripper_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `SurfaceGripper` | Always `SurfaceGripper` |
| `prim_path` | `str` | MISSING | Expression to find grippers in stage |
| `max_grip_distance` | `float \| None` | `None` | Maximum grip distance |
| `coaxial_force_limit` | `float \| None` | `None` | Coaxial (normal) force limit |
| `shear_force_limit` | `float \| None` | `None` | Shear (tangential) force limit |
| `retry_interval` | `float \| None` | `None` | Time spent trying to grasp |

---

## Actuator Configuration Classes

### ActuatorBaseCfg

Base class for all actuators. Source: `source/isaaclab/isaaclab/actuators/actuator_base_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | MISSING | Associated actuator class |
| `joint_names_expr` | `list[str]` | MISSING | Regex expressions matching joint names |
| `effort_limit` | `dict[str,float] \| float \| None` | `None` | Torque limit for actuator model clipping (explicit actuators only) |
| `velocity_limit` | `dict[str,float] \| float \| None` | `None` | Velocity limit for actuator model (explicit actuators only) |
| `effort_limit_sim` | `dict[str,float] \| float \| None` | `None` | Torque limit for physics solver. None = USD value (implicit) or 1e9 (explicit) |
| `velocity_limit_sim` | `dict[str,float] \| float \| None` | `None` | Velocity limit for physics solver. None = USD value |
| `stiffness` | `dict[str,float] \| float \| None` | MISSING | P-gain. Implicit: set in physics engine. Explicit: used by actuator model. None = USD value |
| `damping` | `dict[str,float] \| float \| None` | MISSING | D-gain. Implicit: set in physics engine. Explicit: used by actuator model. None = USD value |
| `armature` | `dict[str,float] \| float \| None` | `None` | Joint-space inertia addition for stability. None = USD value |
| `friction` | `dict[str,float] \| float \| None` | `None` | Static friction coefficient. None = USD value |
| `dynamic_friction` | `dict[str,float] \| float \| None` | `None` | Dynamic friction coefficient. None = USD value |
| `viscous_friction` | `dict[str,float] \| float \| None` | `None` | Viscous friction coefficient. None = USD value |

### ImplicitActuatorCfg

Extends `ActuatorBaseCfg`. Source: `source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py`

PD control handled by the physics engine (continuous-time PD). No additional attributes beyond base.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `ImplicitActuator` | Physics-engine PD |

### IdealPDActuatorCfg

Extends `ActuatorBaseCfg`. Source: `source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py`

Explicit PD computation with torque clipping. No additional attributes beyond base.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `IdealPDActuator` | Explicit PD with clipping |

### DCMotorCfg

Extends `IdealPDActuatorCfg`. Source: `source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `DCMotor` | DC motor model |
| `saturation_effort` | `float` | MISSING | Peak motor force/torque (N-m) for saturation curve |

### DelayedPDActuatorCfg

Extends `IdealPDActuatorCfg`. Source: `source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `DelayedPDActuator` | PD with delay |
| `min_delay` | `int` | `0` | Minimum delay in physics time-steps |
| `max_delay` | `int` | `0` | Maximum delay in physics time-steps |

### RemotizedPDActuatorCfg

Extends `DelayedPDActuatorCfg`. Source: `source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `RemotizedPDActuator` | Cable-driven with lookup |
| `joint_parameter_lookup` | `list[list[float]]` | MISSING | Shape (N, 3): [joint_angle_rad, transmission_ratio, output_torque_Nm] |

### ActuatorNetLSTMCfg

Extends `DCMotorCfg`. Source: `source/isaaclab/isaaclab/actuators/actuator_net_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `ActuatorNetLSTM` | LSTM actuator network |
| `stiffness` | | `None` | Not used (overridden) |
| `damping` | | `None` | Not used (overridden) |
| `network_file` | `str` | MISSING | Path to TorchScript network weights file |

### ActuatorNetMLPCfg

Extends `DCMotorCfg`. Source: `source/isaaclab/isaaclab/actuators/actuator_net_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | `type` | `ActuatorNetMLP` | MLP actuator network |
| `stiffness` | | `None` | Not used (overridden) |
| `damping` | | `None` | Not used (overridden) |
| `network_file` | `str` | MISSING | Path to TorchScript network weights file |
| `pos_scale` | `float` | MISSING | Scaling for joint position error input |
| `vel_scale` | `float` | MISSING | Scaling for joint velocity input |
| `torque_scale` | `float` | MISSING | Scaling for torque output |
| `input_order` | `Literal["pos_vel", "vel_pos"]` | MISSING | Order of network inputs |
| `input_idx` | `Iterable[int]` | MISSING | Indices of history buffer passed as input (0 = current) |

---

## Spawner Configuration Classes

### SpawnerCfg

Base spawner class. Source: `source/isaaclab/isaaclab/sim/spawners/spawner_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable[..., Usd.Prim]` | MISSING | Function to spawn the asset |
| `visible` | `bool` | `True` | Whether spawned asset is visible |
| `semantic_tags` | `list[tuple[str,str]] \| None` | `None` | Replicator semantic tags |
| `copy_from_source` | `bool` | `True` | Copy (True) or inherit (False) from source prim |

### RigidObjectSpawnerCfg

Extends `SpawnerCfg`. Source: `source/isaaclab/isaaclab/sim/spawners/spawner_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `mass_props` | `MassPropertiesCfg \| None` | `None` | Mass properties |
| `rigid_props` | `RigidBodyPropertiesCfg \| None` | `None` | Rigid body properties |
| `collision_props` | `CollisionPropertiesCfg \| None` | `None` | Collision properties |
| `activate_contact_sensors` | `bool` | `False` | Add PhysxContactReporter to all rigid bodies |

### DeformableObjectSpawnerCfg

Extends `SpawnerCfg`. Source: `source/isaaclab/isaaclab/sim/spawners/spawner_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `mass_props` | `MassPropertiesCfg \| None` | `None` | Mass properties |
| `deformable_props` | `DeformableBodyPropertiesCfg \| None` | `None` | Deformable body properties |

### FileCfg

Extends both `RigidObjectSpawnerCfg` and `DeformableObjectSpawnerCfg`. Source: `source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `scale` | `tuple[float,float,float] \| None` | `None` | Scale of the asset |
| `articulation_props` | `ArticulationRootPropertiesCfg \| None` | `None` | Articulation root properties |
| `fixed_tendons_props` | `FixedTendonPropertiesCfg \| None` | `None` | Fixed tendon properties |
| `spatial_tendons_props` | `SpatialTendonPropertiesCfg \| None` | `None` | Spatial tendon properties |
| `joint_drive_props` | `JointDrivePropertiesCfg \| None` | `None` | Joint drive properties (prefer actuator cfg instead) |
| `visual_material_path` | `str` | `"material"` | Path for visual material |
| `visual_material` | `VisualMaterialCfg \| None` | `None` | Visual material override |

### UsdFileCfg

Extends `FileCfg`. Source: `source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | `spawn_from_usd` | USD spawner function |
| `usd_path` | `str` | MISSING | Path to USD file |
| `variants` | `object \| dict[str,str] \| None` | `None` | Variant selections (e.g., `{"Gripper": "Robotiq_2F_85"}`) |

### UrdfFileCfg

Extends `FileCfg` and `UrdfConverterCfg`. Source: `source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | `spawn_from_urdf` | URDF spawner function |
| Inherits all `UrdfConverterCfg` attributes for URDF-to-USD conversion. |

### MjcfFileCfg

Extends `FileCfg` and `MjcfConverterCfg`. Source: `source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | `spawn_from_mjcf` | MJCF spawner function |
| Inherits all `MjcfConverterCfg` attributes for MJCF-to-USD conversion. |

### GroundPlaneCfg

Extends `SpawnerCfg`. Source: `source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | `spawn_ground_plane` | Ground plane spawner |
| `usd_path` | `str` | Isaac Nucleus grid world | Path to ground plane USD |
| `color` | `tuple[float,float,float] \| None` | `(0.0, 0.0, 0.0)` | Ground plane color (None = unchanged) |
| `size` | `tuple[float,float]` | `(100.0, 100.0)` | Ground plane size in meters |
| `physics_material` | `RigidBodyMaterialCfg` | `RigidBodyMaterialCfg()` | Physics material |

### Shape Spawners

All extend `ShapeCfg` which extends `RigidObjectSpawnerCfg`. Source: `source/isaaclab/isaaclab/sim/spawners/shapes/shapes_cfg.py`

#### ShapeCfg Base Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `visual_material_path` | `str` | `"material"` | Path for visual material |
| `visual_material` | `VisualMaterialCfg \| None` | `None` | Visual material |
| `physics_material_path` | `str` | `"material"` | Path for physics material |
| `physics_material` | `PhysicsMaterialCfg \| None` | `None` | Physics material |

#### SphereCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | `float` | MISSING | Radius in meters |

#### CuboidCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `tuple[float,float,float]` | MISSING | Size (x, y, z) |

#### CylinderCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | `float` | MISSING | Radius in meters |
| `height` | `float` | MISSING | Height in meters |
| `axis` | `Literal["X","Y","Z"]` | `"Z"` | Cylinder axis |

#### CapsuleCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | `float` | MISSING | Radius in meters |
| `height` | `float` | MISSING | Height in meters |
| `axis` | `Literal["X","Y","Z"]` | `"Z"` | Capsule axis |

#### ConeCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | `float` | MISSING | Radius in meters |
| `height` | `float` | MISSING | Height in meters |
| `axis` | `Literal["X","Y","Z"]` | `"Z"` | Cone axis |

---

## Physics Property Configuration Classes

All source: `source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py`

All attributes default to `None` (not modified) unless otherwise noted.

### ArticulationRootPropertiesCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `articulation_enabled` | `bool \| None` | `None` | Enable/disable articulation |
| `enabled_self_collisions` | `bool \| None` | `None` | Enable/disable self-collisions |
| `solver_position_iteration_count` | `int \| None` | `None` | Solver position iterations |
| `solver_velocity_iteration_count` | `int \| None` | `None` | Solver velocity iterations |
| `sleep_threshold` | `float \| None` | `None` | Kinetic energy sleep threshold |
| `stabilization_threshold` | `float \| None` | `None` | Kinetic energy stabilization threshold |
| `fix_root_link` | `bool \| None` | `None` | Fix root link to world (creates FixedJoint if needed) |

### RigidBodyPropertiesCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `rigid_body_enabled` | `bool \| None` | `None` | Enable/disable rigid body |
| `kinematic_enabled` | `bool \| None` | `None` | Make body kinematic (user-driven motion) |
| `disable_gravity` | `bool \| None` | `None` | Disable gravity |
| `linear_damping` | `float \| None` | `None` | Linear damping |
| `angular_damping` | `float \| None` | `None` | Angular damping |
| `max_linear_velocity` | `float \| None` | `None` | Max linear velocity (m/s) |
| `max_angular_velocity` | `float \| None` | `None` | Max angular velocity (deg/s) |
| `max_depenetration_velocity` | `float \| None` | `None` | Max depenetration velocity (m/s) |
| `max_contact_impulse` | `float \| None` | `None` | Max contact impulse |
| `enable_gyroscopic_forces` | `bool \| None` | `None` | Enable gyroscopic forces |
| `retain_accelerations` | `bool \| None` | `None` | Carry over forces/accelerations over sub-steps |
| `solver_position_iteration_count` | `int \| None` | `None` | Per-body solver position iterations |
| `solver_velocity_iteration_count` | `int \| None` | `None` | Per-body solver velocity iterations |
| `sleep_threshold` | `float \| None` | `None` | Sleep threshold |
| `stabilization_threshold` | `float \| None` | `None` | Stabilization threshold |

### CollisionPropertiesCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `collision_enabled` | `bool \| None` | `None` | Enable/disable collisions |
| `contact_offset` | `float \| None` | `None` | Contact generation offset (m) |
| `rest_offset` | `float \| None` | `None` | Rest separation offset (m) |
| `torsional_patch_radius` | `float \| None` | `None` | Torsional friction radius (m) |
| `min_torsional_patch_radius` | `float \| None` | `None` | Min torsional friction radius (m) |

### MassPropertiesCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `mass` | `float \| None` | `None` | Mass (kg). Ignored if density is non-zero |
| `density` | `float \| None` | `None` | Density (kg/m^3) |

### JointDrivePropertiesCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `drive_type` | `Literal["force","acceleration"] \| None` | `None` | Drive type |
| `max_effort` | `float \| None` | `None` | Max effort (kg-m^2/s^2) |
| `max_velocity` | `float \| None` | `None` | Max velocity (m/s or rad/s) |
| `stiffness` | `float \| None` | `None` | Drive stiffness |
| `damping` | `float \| None` | `None` | Drive damping |

### FixedTendonPropertiesCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `tendon_enabled` | `bool \| None` | `None` | Enable/disable tendon |
| `stiffness` | `float \| None` | `None` | Spring stiffness on tendon length |
| `damping` | `float \| None` | `None` | Damping on tendon length and limits |
| `limit_stiffness` | `float \| None` | `None` | Stiffness on tendon length limits |
| `offset` | `float \| None` | `None` | Length offset for tendon actuation |
| `rest_length` | `float \| None` | `None` | Spring rest length |

### SpatialTendonPropertiesCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `tendon_enabled` | `bool \| None` | `None` | Enable/disable tendon |
| `stiffness` | `float \| None` | `None` | Spring stiffness |
| `damping` | `float \| None` | `None` | Damping |
| `limit_stiffness` | `float \| None` | `None` | Limit stiffness |
| `offset` | `float \| None` | `None` | Length offset |

### DeformableBodyPropertiesCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `deformable_enabled` | `bool \| None` | `None` | Enable deformable body |
| `kinematic_enabled` | `bool` | `False` | Make kinematic (user-driven) |
| `self_collision` | `bool \| None` | `None` | Enable self-collisions |
| `self_collision_filter_distance` | `float \| None` | `None` | Min penetration for self-collision contacts |
| `settling_threshold` | `float \| None` | `None` | Velocity threshold for sleep damping (m/s) |
| `sleep_damping` | `float \| None` | `None` | Additional damping below settling threshold |
| `sleep_threshold` | `float \| None` | `None` | Velocity threshold for sleep candidate (m/s) |
| `solver_position_iteration_count` | `int \| None` | `None` | Solver iterations [1, 255] |
| `vertex_velocity_damping` | `float \| None` | `None` | Artificial damping (approximates air drag) |
| `simulation_hexahedral_resolution` | `int` | `10` | Target resolution for simulation hex mesh |
| `collision_simplification` | `bool` | `True` | Simplify collision mesh |
| `collision_simplification_remeshing` | `bool` | `True` | Remesh before simplification |
| `collision_simplification_remeshing_resolution` | `int` | `0` | Remesh resolution (0 = heuristic) |
| `collision_simplification_target_triangle_count` | `int` | `0` | Target triangle count (0 = heuristic) |
| `collision_simplification_force_conforming` | `bool` | `True` | Force output to conform to input mesh |
| `contact_offset` | `float \| None` | `None` | Contact offset (m) |
| `rest_offset` | `float \| None` | `None` | Rest offset (m) |
| `max_depenetration_velocity` | `float \| None` | `None` | Max depenetration velocity (m/s) |
