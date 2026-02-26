---
name: isaaclab-terrains-and-sensors
description: >
  Configures terrain generation/import and sensors (ContactSensor, RayCaster, Camera, FrameTransformer, IMU) within InteractiveScene.
layer: L3
domain: [robotics, locomotion]
source-project: IsaacLab
depends-on: [isaacsim-sensor-development, isaaclab-configclass-and-utilities]
tags: [terrains, sensors, height-field, curriculum]
---

# IsaacLab Terrains and Sensors

Terrain and sensor subsystems provide environment geometry and proprioceptive/exteroceptive feedback for RL training.

## Architecture

```
TerrainImporter (scene entity)
  ├── terrain_type="plane"      → flat ground
  ├── terrain_type="usd"        → custom USD mesh
  └── terrain_type="generator"  → TerrainGenerator
        ├── Height-field sub-terrains (HfRandomUniform, HfPyramidSloped, ...)
        └── Trimesh sub-terrains (MeshPyramidStairs, MeshRandomGrid, ...)

Sensors (scene entities)
  ├── ContactSensor     → contact forces, air time
  ├── RayCaster         → height scans, distance measurements
  ├── Camera            → RGB, depth, segmentation (path-traced)
  ├── TiledCamera       → RGB, depth (tiled rendering, faster)
  ├── FrameTransformer  → relative pose between frames
  └── Imu               → orientation, angular velocity, acceleration
```

## Terrain System

### TerrainImporter in Scene

Add terrain to an `InteractiveSceneCfg` as a named attribute:

```python
from isaaclab.terrains import TerrainImporterCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=my_terrain_generator_cfg,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        max_init_terrain_level=5,
        debug_vis=False,
    )
```

### TerrainImporterCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `prim_path` | str | MISSING | USD prim path for terrain |
| `terrain_type` | str | `"generator"` | `"generator"`, `"plane"`, `"usd"` |
| `terrain_generator` | TerrainGeneratorCfg | None | Generator config |
| `usd_path` | str | None | USD file (if terrain_type="usd") |
| `env_spacing` | float | None | Override env spacing |
| `max_init_terrain_level` | int | None | Max initial difficulty level |
| `physics_material` | RigidBodyMaterialCfg | default | Ground physics material |
| `visual_material` | VisualMaterialCfg | black preview | Visual material |

### TerrainGeneratorCfg

```python
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import *
from isaaclab.terrains.trimesh import *

my_terrain_cfg = TerrainGeneratorCfg(
    size=(8.0, 8.0),           # Each sub-terrain tile size
    num_rows=10,                # Rows (difficulty levels for curriculum)
    num_cols=20,                # Columns (terrain types)
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    curriculum=True,            # Enable difficulty progression
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.1),
        "stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
        ),
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
        ),
        "slopes": HfPyramidSlopedTerrainCfg(
            proportion=0.3,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
        ),
    },
)
```

### Sub-Terrain Types

**Height-Field** (grid-based, fast generation):

| Config | Description | Key Params |
|--------|-------------|------------|
| `HfRandomUniformTerrainCfg` | Random bumpy terrain | `noise_range`, `noise_step` |
| `HfPyramidSlopedTerrainCfg` | Pyramid slopes | `slope_range`, `platform_width` |
| `HfInvertedPyramidSlopedTerrainCfg` | Inverted slopes | `slope_range` |
| `HfPyramidStairsTerrainCfg` | Pyramid stairs | `step_height_range`, `step_width` |
| `HfInvertedPyramidStairsTerrainCfg` | Inverted stairs | `step_height_range` |
| `HfDiscreteObstaclesTerrainCfg` | Random obstacles | `obstacle_width/height_range`, `num_obstacles` |
| `HfWaveTerrainCfg` | Sinusoidal waves | `amplitude_range`, `num_waves` |
| `HfSteppingStonesTerrainCfg` | Stepping stones | `stone_height_max`, `stone_width/distance_range` |

**Trimesh** (mesh-based, more geometric detail):

| Config | Description | Key Params |
|--------|-------------|------------|
| `MeshPlaneTerrainCfg` | Flat plane | — |
| `MeshPyramidStairsTerrainCfg` | Stairs | `step_height_range`, `step_width`, `holes` |
| `MeshInvertedPyramidStairsTerrainCfg` | Inverted stairs | same |
| `MeshRandomGridTerrainCfg` | Random grid heights | `grid_width`, `grid_height_range` |
| `MeshRailsTerrainCfg` | Raised rails | `rail_thickness/height_range` |
| `MeshPitTerrainCfg` | Pit terrain | `pit_depth_range`, `double_pit` |
| `MeshBoxTerrainCfg` | Box terrain | `box_height_range`, `double_box` |
| `MeshGapTerrainCfg` | Gaps to cross | `gap_width_range` |
| `MeshFloatingRingTerrainCfg` | Floating rings | `ring_width/height_range` |
| `MeshStarTerrainCfg` | Star pattern bars | `num_bars`, `bar_width/height_range` |
| `MeshRepeatedPyramidsTerrainCfg` | Repeated pyramids | `object_params_start/end` |
| `MeshRepeatedBoxesTerrainCfg` | Repeated boxes | `object_params_start/end` |
| `MeshRepeatedCylindersTerrainCfg` | Repeated cylinders | `object_params_start/end` |

### Pre-built Configuration

```python
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

terrain_cfg = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG.replace(num_rows=10, num_cols=20),
)
```

### Terrain Curriculum

When `curriculum=True`, rows represent difficulty levels (0 = easiest, num_rows-1 = hardest). The `TerrainImporter.update_env_origins()` method promotes/demotes environments:

```python
# In a curriculum term or custom logic:
terrain = env.scene.terrain
terrain.update_env_origins(
    env_ids=env_ids,
    move_up=success_mask,     # bool tensor: promote on success
    move_down=failure_mask,   # bool tensor: demote on failure
)
```

## Sensors

### Adding Sensors to Scene

```python
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, ImuCfg, CameraCfg, TiledCameraCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Contact sensor on feet
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        track_air_time=True,
        force_threshold=1.0,
        history_length=3,
    )
    # Height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
    )
    # IMU sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
    )
```

### ContactSensorCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `prim_path` | str | MISSING | Body prim path (regex for multiple) |
| `track_air_time` | bool | False | Track air/contact time |
| `force_threshold` | float | 1.0 | Min force for contact detection |
| `track_pose` | bool | False | Track sensor pose |
| `track_friction_forces` | bool | False | Track friction forces |
| `filter_prim_paths_expr` | list | [] | Filter contacts with specific prims |
| `update_period` | float | 0.0 | Sensor update period |
| `history_length` | int | 0 | Past frames to store |

### RayCasterCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `prim_path` | str | MISSING | Sensor origin prim |
| `mesh_prim_paths` | list[str] | MISSING | Meshes to cast against |
| `pattern_cfg` | PatternBaseCfg | MISSING | Ray pattern (GridPatternCfg, etc.) |
| `offset` | OffsetCfg | identity | Position/rotation offset |
| `ray_alignment` | str | `"base"` | `"base"`, `"yaw"`, `"world"` |
| `max_distance` | float | 1e6 | Max ray distance |

### CameraCfg / TiledCameraCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `prim_path` | str | MISSING | Camera prim |
| `spawn` | PinholeCameraCfg | MISSING | Camera parameters |
| `width` | int | MISSING | Image width |
| `height` | int | MISSING | Image height |
| `data_types` | list[str] | `["rgb"]` | `"rgb"`, `"distance_to_image_plane"`, etc. |
| `offset` | OffsetCfg | identity | Offset with convention (opengl/ros/world) |
| `update_period` | float | 0.0 | Update period |

Use `TiledCameraCfg` for faster rendering (recommended for RL).

### FrameTransformerCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `prim_path` | str | MISSING | Source frame prim |
| `target_frames` | list[FrameCfg] | MISSING | Target frames to track |

### ImuCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `prim_path` | str | MISSING | IMU body prim |
| `offset` | OffsetCfg | identity | Position/rotation offset |
| `gravity_bias` | tuple | (0, 0, 9.81) | Gravity bias (m/s²) |

### Using Sensor Data in Observations

```python
from isaaclab.envs.mdp import observations as obs
from isaaclab.managers import SceneEntityCfg

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Height scan from ray caster
        height_scan = ObsTerm(
            func=obs.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
        )
        # IMU data
        imu_ang_vel = ObsTerm(
            func=obs.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        # Camera image
        camera_rgb = ObsTerm(
            func=obs.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"},
        )
```

## Reference Files

- [terrains-and-sensors-catalog.md](terrains-and-sensors-catalog.md) - Full attribute tables for all terrain and sensor config classes

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/isaaclab/isaaclab/terrains/terrain_generator.py` | TerrainGenerator class |
| `source/isaaclab/isaaclab/terrains/terrain_generator_cfg.py` | TerrainGeneratorCfg |
| `source/isaaclab/isaaclab/terrains/terrain_importer.py` | TerrainImporter with curriculum |
| `source/isaaclab/isaaclab/terrains/terrain_importer_cfg.py` | TerrainImporterCfg |
| `source/isaaclab/isaaclab/terrains/height_field/` | Height-field terrain types |
| `source/isaaclab/isaaclab/terrains/trimesh/` | Trimesh terrain types |
| `source/isaaclab/isaaclab/terrains/config/rough.py` | ROUGH_TERRAINS_CFG pre-built |
| `source/isaaclab/isaaclab/sensors/contact_sensor/` | Contact sensor |
| `source/isaaclab/isaaclab/sensors/ray_caster/` | RayCaster and patterns |
| `source/isaaclab/isaaclab/sensors/camera/` | Camera and TiledCamera |
| `source/isaaclab/isaaclab/sensors/frame_transformer/` | FrameTransformer |
| `source/isaaclab/isaaclab/sensors/imu/` | IMU sensor |
