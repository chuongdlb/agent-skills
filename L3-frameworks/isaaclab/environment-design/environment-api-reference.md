# Environment API Reference

Detailed attribute tables for all environment configuration classes and their base types.

## ManagerBasedEnvCfg

**File:** `source/isaaclab/isaaclab/envs/manager_based_env_cfg.py`

Base configuration for manager-based environments (non-RL and RL).

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `viewer` | `ViewerCfg` | `ViewerCfg()` | Viewport camera configuration |
| `sim` | `SimulationCfg` | `SimulationCfg()` | Physics simulation settings |
| `ui_window_class_type` | `type \| None` | `BaseEnvWindow` | UI window class (`None` to disable) |
| `seed` | `int \| None` | `None` | RNG seed (`None` = not set) |
| `decimation` | `int` | MISSING | Physics substeps per env step |
| `scene` | `InteractiveSceneCfg` | MISSING | Scene entity definitions |
| `recorders` | `object` | `DefaultEmptyRecorderManagerCfg()` | Recorder manager config |
| `observations` | `object` | MISSING | Observation manager config |
| `actions` | `object` | MISSING | Action manager config |
| `events` | `object` | `DefaultEventManagerCfg()` | Event manager config (default resets scene) |
| `num_rerenders_on_reset` | `int` | `0` | Extra render steps after reset for sensor data freshness |
| `wait_for_textures` | `bool` | `True` | Wait for asset textures to load |
| `xr` | `XrCfg \| None` | `None` | XR device configuration |
| `teleop_devices` | `DevicesCfg` | `DevicesCfg()` | Teleoperation device configuration |
| `export_io_descriptors` | `bool` | `False` | Export IO descriptors for the environment |
| `log_dir` | `str \| None` | `None` | Directory for logging artifacts |

### DefaultEventManagerCfg

Automatically included unless overridden:

```python
@configclass
class DefaultEventManagerCfg:
    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
```

## ManagerBasedRLEnvCfg

**File:** `source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py`

Extends `ManagerBasedEnvCfg` with RL-specific fields.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `ui_window_class_type` | `type \| None` | `ManagerBasedRLEnvWindow` | Overrides parent default |
| `is_finite_horizon` | `bool` | `False` | If `True`, no truncated signal sent to agent |
| `episode_length_s` | `float` | MISSING | Episode duration in seconds |
| `rewards` | `object` | MISSING | Reward manager config |
| `terminations` | `object` | MISSING | Termination manager config |
| `curriculum` | `object \| None` | `None` | Curriculum manager config |
| `commands` | `object \| None` | `None` | Command manager config |

Plus all attributes inherited from `ManagerBasedEnvCfg` above.

### Computed Properties (on ManagerBasedRLEnv)

| Property | Type | Formula |
|----------|------|---------|
| `num_envs` | `int` | `scene.num_envs` |
| `physics_dt` | `float` | `cfg.sim.dt` |
| `step_dt` | `float` | `cfg.sim.dt * cfg.decimation` |
| `max_episode_length_s` | `float` | `cfg.episode_length_s` |
| `max_episode_length` | `int` | `ceil(episode_length_s / step_dt)` |

## DirectRLEnvCfg

**File:** `source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py`

Standalone configuration for direct workflow environments (does not inherit from `ManagerBasedEnvCfg`).

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `viewer` | `ViewerCfg` | `ViewerCfg()` | Viewport camera configuration |
| `sim` | `SimulationCfg` | `SimulationCfg()` | Physics simulation settings |
| `ui_window_class_type` | `type \| None` | `BaseEnvWindow` | UI window class |
| `seed` | `int \| None` | `None` | RNG seed |
| `decimation` | `int` | MISSING | Physics substeps per env step |
| `is_finite_horizon` | `bool` | `False` | Finite vs infinite horizon |
| `episode_length_s` | `float` | MISSING | Episode duration in seconds |
| `scene` | `InteractiveSceneCfg` | MISSING | Scene entity definitions |
| `events` | `object \| None` | `None` | Event manager config (no default events) |
| `observation_space` | `SpaceType` | MISSING | Observation space spec |
| `state_space` | `SpaceType \| None` | `None` | Critic state space (asymmetric AC) |
| `observation_noise_model` | `NoiseModelCfg \| None` | `None` | Noise applied to observations |
| `action_space` | `SpaceType` | MISSING | Action space spec |
| `action_noise_model` | `NoiseModelCfg \| None` | `None` | Noise applied to actions |
| `num_rerenders_on_reset` | `int` | `0` | Extra render steps after reset |
| `wait_for_textures` | `bool` | `True` | Wait for asset textures |
| `xr` | `XrCfg \| None` | `None` | XR device configuration |
| `log_dir` | `str \| None` | `None` | Logging directory |

### SpaceType Encoding

Spaces can be specified as Python primitives or Gymnasium spaces:

| Gymnasium Space | Python Shorthand | Example |
|-----------------|-----------------|---------|
| `gym.spaces.Box` | `int` or `list[int]` | `7`, `[64, 64, 3]` |
| `gym.spaces.Discrete` | `set` (single element) | `{2}` |
| `gym.spaces.MultiDiscrete` | `list[set]` | `[{2}, {5}]` |
| `gym.spaces.Dict` | `dict` | `{"joints": 7, "rgb": [64, 64, 3]}` |
| `gym.spaces.Tuple` | `tuple` | `(7, [64, 64, 3], {2})` |

### Deprecated Attributes

| Deprecated | Replacement |
|------------|-------------|
| `num_observations` | `observation_space` |
| `num_actions` | `action_space` |
| `num_states` | `state_space` |
| `rerender_on_reset` | `num_rerenders_on_reset` |

## DirectMARLEnvCfg

**File:** `source/isaaclab/isaaclab/envs/direct_marl_env_cfg.py`

Configuration for multi-agent direct workflow environments.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `viewer` | `ViewerCfg` | `ViewerCfg()` | Viewport camera configuration |
| `sim` | `SimulationCfg` | `SimulationCfg()` | Physics simulation settings |
| `ui_window_class_type` | `type \| None` | `BaseEnvWindow` | UI window class |
| `seed` | `int \| None` | `None` | RNG seed |
| `decimation` | `int` | MISSING | Physics substeps per env step |
| `is_finite_horizon` | `bool` | `False` | Finite vs infinite horizon |
| `episode_length_s` | `float` | MISSING | Episode duration in seconds |
| `scene` | `InteractiveSceneCfg` | MISSING | Scene entity definitions |
| `events` | `object \| None` | `None` | Event manager config |
| `possible_agents` | `list[AgentID]` | MISSING | All possible agent IDs (immutable during training) |
| `observation_spaces` | `dict[AgentID, SpaceType]` | MISSING | Per-agent observation spaces |
| `action_spaces` | `dict[AgentID, SpaceType]` | MISSING | Per-agent action spaces |
| `state_space` | `SpaceType` | MISSING | Shared state space (see below) |
| `observation_noise_model` | `dict[AgentID, NoiseModelCfg \| None] \| None` | `None` | Per-agent observation noise |
| `action_noise_model` | `dict[AgentID, NoiseModelCfg \| None] \| None` | `None` | Per-agent action noise |
| `xr` | `XrCfg \| None` | `None` | XR device configuration |
| `log_dir` | `str \| None` | `None` | Logging directory |

### state_space Special Values

| Value | Behavior |
|-------|----------|
| `-1` | Auto-concatenate all agent observations |
| `0` | No state space (`state()` returns `None`) |
| `> 0` | Custom dimension; user must implement `_get_states()` |

### DirectMARLEnv Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_envs` | `int` | Number of parallel environments |
| `num_agents` | `int` | Current number of active agents (may change) |
| `max_num_agents` | `int` | Length of `possible_agents` (constant) |
| `agents` | `list[AgentID]` | Currently active agents |
| `possible_agents` | `list[AgentID]` | All possible agents |

## InteractiveSceneCfg

**File:** `source/isaaclab/isaaclab/scene/interactive_scene_cfg.py`

Base class for defining scene composition. Users subclass and add named attributes for each entity.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | `int` | MISSING | Number of parallel environments |
| `env_spacing` | `float` | MISSING | Distance between environment origins |
| `lazy_sensor_update` | `bool` | `True` | Update sensors only when `.data` is accessed |
| `replicate_physics` | `bool` | `True` | Share physics schemas across clones (faster setup) |
| `filter_collisions` | `bool` | `True` | Disable collisions between cloned envs |
| `clone_in_fabric` | `bool` | `False` | Use Fabric for cloning (requires `replicate_physics=True`) |

### Scene Entity Types

Entities are added as class attributes. The `InteractiveScene` parser categorizes them by config type:

| Config Type | Stored In | Example |
|-------------|-----------|---------|
| `TerrainImporterCfg` | `scene.terrain` | Ground planes, procedural terrain |
| `ArticulationCfg` | `scene.articulations["name"]` | Robots |
| `RigidObjectCfg` | `scene.rigid_objects["name"]` | Manipulated objects |
| `DeformableObjectCfg` | `scene.deformable_objects["name"]` | Soft bodies |
| `SensorBaseCfg` subclasses | `scene.sensors["name"]` | Cameras, contact sensors, ray casters |
| `AssetBaseCfg` | `scene.extras["name"]` | Lights, visual markers |

### Entity Ordering

The order of attributes in the config class determines the order of scene creation. Recommended order:

1. Terrain (ground plane, terrain importer)
2. Physics assets (articulations, rigid bodies)
3. Sensors (cameras, contact sensors, ray casters)
4. Non-physics extras (lights, visual-only assets)

## SimulationCfg

**File:** `source/isaaclab/isaaclab/sim/simulation_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `physics_prim_path` | `str` | `"/physicsScene"` | USD path for PhysicsScene prim |
| `device` | `str` | `"cuda:0"` | Compute device (`"cpu"`, `"cuda"`, `"cuda:N"`) |
| `dt` | `float` | `1/60` | Physics time-step in seconds |
| `render_interval` | `int` | `1` | Physics steps per render call |
| `gravity` | `tuple[float,float,float]` | `(0,0,-9.81)` | Gravity vector in m/s^2 |
| `enable_scene_query_support` | `bool` | `False` | Enable collision queries (raycasts, sweeps) |
| `use_fabric` | `bool` | `True` | Read physics buffers directly (bypass USD sync) |
| `physx` | `PhysxCfg` | `PhysxCfg()` | PhysX solver settings |
| `physics_material` | `RigidBodyMaterialCfg` | `RigidBodyMaterialCfg()` | Default physics material |
| `render` | `RenderCfg` | `RenderCfg()` | Omniverse RTX renderer settings |
| `create_stage_in_memory` | `bool` | `False` | Create stage in memory (faster startup) |
| `logging_level` | `Literal[...]` | `"WARNING"` | Log level |
| `save_logs_to_file` | `bool` | `True` | Write logs to file |
| `log_dir` | `str \| None` | `None` | Log file directory |

### PhysxCfg Key Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `solver_type` | `0 \| 1` | `1` | 0=PGS, 1=TGS (Temporal Gauss-Seidel) |
| `min_position_iteration_count` | `int` | `1` | Min solver position iterations |
| `max_position_iteration_count` | `int` | `255` | Max solver position iterations |
| `min_velocity_iteration_count` | `int` | `0` | Min solver velocity iterations |
| `max_velocity_iteration_count` | `int` | `255` | Max solver velocity iterations |
| `enable_ccd` | `bool` | `False` | Continuous collision detection |
| `enable_stabilization` | `bool` | `False` | Extra stabilization (for large dt) |
| `enable_enhanced_determinism` | `bool` | `False` | Determinism at cost of performance |
| `bounce_threshold_velocity` | `float` | `0.5` | Bounce threshold (m/s) |
| `gpu_max_rigid_contact_count` | `int` | `2**23` | GPU rigid contact buffer |
| `gpu_max_rigid_patch_count` | `int` | `5*2**15` | GPU rigid patch buffer |
| `gpu_found_lost_pairs_capacity` | `int` | `2**21` | GPU BP found/lost pairs |
| `gpu_collision_stack_size` | `int` | `2**26` | GPU collision stack |
| `gpu_heap_capacity` | `int` | `2**26` | GPU heap initial capacity |
| `gpu_max_num_partitions` | `int` | `8` | GPU pipeline partitions (power of 2, max 32) |
| `solve_articulation_contact_last` | `bool` | `False` | Prioritize contact resolution (better for gripping) |
| `enable_external_forces_every_iteration` | `bool` | `False` | External forces each TGS iteration |

### RenderCfg Key Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_translucency` | `bool \| None` | `None` | Translucent surfaces |
| `enable_reflections` | `bool \| None` | `None` | Reflections |
| `enable_global_illumination` | `bool \| None` | `None` | Diffuse GI |
| `antialiasing_mode` | `Literal[...] \| None` | `None` | `"Off"`, `"FXAA"`, `"DLSS"`, `"TAA"`, `"DLAA"` |
| `enable_dlssg` | `bool \| None` | `None` | DLSS Frame Generation (Ada Lovelace GPU) |
| `dlss_mode` | `0-3 \| None` | `None` | 0=Perf, 1=Balanced, 2=Quality, 3=Auto |
| `enable_shadows` | `bool \| None` | `None` | Shadow casting |
| `enable_ambient_occlusion` | `bool \| None` | `None` | Ambient occlusion |
| `rendering_mode` | `Literal[...] \| None` | `None` | `"performance"`, `"balanced"`, `"quality"` |
| `carb_settings` | `dict \| None` | `None` | Raw carb renderer key-value overrides |

`None` means the experience file default is used.

## ViewerCfg

**File:** `source/isaaclab/isaaclab/envs/common.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `eye` | `tuple[float,float,float]` | `(7.5, 7.5, 7.5)` | Camera position in meters |
| `lookat` | `tuple[float,float,float]` | `(0.0, 0.0, 0.0)` | Camera target position |
| `cam_prim_path` | `str` | `"/OmniverseKit_Persp"` | Camera prim for recording |
| `resolution` | `tuple[int,int]` | `(1280, 720)` | Camera resolution (width, height) |
| `origin_type` | `Literal[...]` | `"world"` | `"world"`, `"env"`, `"asset_root"`, `"asset_body"` |
