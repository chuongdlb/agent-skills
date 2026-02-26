# Terrains and Sensors Catalog

Full attribute tables for all terrain and sensor configuration classes.

## TerrainGeneratorCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | TerrainGenerator | Generator class |
| `seed` | int \| None | None | Random seed |
| `curriculum` | bool | False | Enable difficulty progression |
| `size` | tuple[float, float] | MISSING | Sub-terrain tile size (x, y) meters |
| `border_width` | float | 0.0 | Border width in meters |
| `border_height` | float | 1.0 | Border height in meters |
| `num_rows` | int | 1 | Number of rows (difficulty levels) |
| `num_cols` | int | 1 | Number of columns (terrain types) |
| `color_scheme` | str | "none" | "height", "random", "none" |
| `horizontal_scale` | float | 0.1 | X/Y discretization (meters) |
| `vertical_scale` | float | 0.005 | Z discretization (meters) |
| `slope_threshold` | float \| None | 0.75 | Slope correction threshold |
| `sub_terrains` | dict[str, SubTerrainBaseCfg] | MISSING | Sub-terrain configs |
| `difficulty_range` | tuple[float, float] | (0.0, 1.0) | Difficulty value range |
| `use_cache` | bool | False | Cache generated terrains |
| `cache_dir` | str | "/tmp/isaaclab/terrains" | Cache directory |

## SubTerrainBaseCfg (Base)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `function` | Callable | MISSING | Generation function |
| `proportion` | float | 1.0 | Sampling probability |
| `size` | tuple[float, float] | (10.0, 10.0) | Tile size |
| `flat_patch_sampling` | dict \| None | None | Flat patch configs |

## Height-Field Sub-Terrains

### HfTerrainBaseCfg (Base)

| Attribute | Type | Default |
|-----------|------|---------|
| `border_width` | float | 0.0 |
| `horizontal_scale` | float | 0.1 |
| `vertical_scale` | float | 0.005 |
| `slope_threshold` | float \| None | None |

### HfRandomUniformTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_range` | tuple[float, float] | MISSING | Min/max height noise (m) |
| `noise_step` | float | MISSING | Min height change between points |
| `downsampled_scale` | float \| None | None | Sampling distance |

### HfPyramidSlopedTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `slope_range` | tuple[float, float] | MISSING | Slope range (radians) |
| `platform_width` | float | 1.0 | Center platform width |
| `inverted` | bool | False | Invert pyramid |

### HfPyramidStairsTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_height_range` | tuple[float, float] | MISSING | Step height range (m) |
| `step_width` | float | MISSING | Step width (m) |
| `platform_width` | float | 1.0 | Center platform width |
| `inverted` | bool | False | Invert stairs |

### HfDiscreteObstaclesTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `obstacle_height_mode` | str | "choice" | "choice" or "fixed" |
| `obstacle_width_range` | tuple[float, float] | MISSING | Obstacle width range |
| `obstacle_height_range` | tuple[float, float] | MISSING | Obstacle height range |
| `num_obstacles` | int | MISSING | Number of obstacles |
| `platform_width` | float | 1.0 | Center platform width |

### HfWaveTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `amplitude_range` | tuple[float, float] | MISSING | Wave amplitude range |
| `num_waves` | int | 1 | Number of waves |

### HfSteppingStonesTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `stone_height_max` | float | MISSING | Max stone height |
| `stone_width_range` | tuple[float, float] | MISSING | Stone width range |
| `stone_distance_range` | tuple[float, float] | MISSING | Distance between stones |
| `holes_depth` | float | -10.0 | Hole depth |
| `platform_width` | float | 1.0 | Center platform width |

## Trimesh Sub-Terrains

### MeshPyramidStairsTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `border_width` | float | 0.0 | Border width |
| `step_height_range` | tuple[float, float] | MISSING | Step height range |
| `step_width` | float | MISSING | Step width |
| `platform_width` | float | 1.0 | Center platform |
| `holes` | bool | False | Add holes |

### MeshRandomGridTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_width` | float | MISSING | Grid cell width |
| `grid_height_range` | tuple[float, float] | MISSING | Cell height range |
| `platform_width` | float | 1.0 | Center platform |
| `holes` | bool | False | Add holes |

### MeshRailsTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `rail_thickness_range` | tuple[float, float] | MISSING | Rail thickness range |
| `rail_height_range` | tuple[float, float] | MISSING | Rail height range |
| `platform_width` | float | 1.0 | Center platform |

### MeshPitTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `pit_depth_range` | tuple[float, float] | MISSING | Pit depth range |
| `platform_width` | float | 1.0 | Center platform |
| `double_pit` | bool | False | Two levels |

### MeshBoxTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `box_height_range` | tuple[float, float] | MISSING | Box height range |
| `platform_width` | float | 1.0 | Center platform |
| `double_box` | bool | False | Two levels |

### MeshGapTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `gap_width_range` | tuple[float, float] | MISSING | Gap width range |
| `platform_width` | float | 1.0 | Center platform |

### MeshFloatingRingTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `ring_width_range` | tuple[float, float] | MISSING | Ring width range |
| `ring_height_range` | tuple[float, float] | MISSING | Ring height range |
| `ring_thickness` | float | MISSING | Ring Z thickness |
| `platform_width` | float | 1.0 | Center platform |

### MeshStarTerrainCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_bars` | int | MISSING | Bars per side |
| `bar_width_range` | tuple[float, float] | MISSING | Bar width range |
| `bar_height_range` | tuple[float, float] | MISSING | Bar height range |
| `platform_width` | float | 1.0 | Center platform |

### MeshRepeatedObjectsTerrainCfg (Base)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `object_type` | str \| callable | MISSING | "cylinder", "box", "cone" |
| `object_params_start` | ObjectCfg | MISSING | Start curriculum params |
| `object_params_end` | ObjectCfg | MISSING | End curriculum params |
| `platform_width` | float | 1.0 | Center platform |

ObjectCfg: `num_objects: int`, `height: float`

## ROUGH_TERRAINS_CFG Pre-built

```python
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

# Contents: size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20
# Sub-terrains: pyramid_stairs(0.2), pyramid_stairs_inv(0.2),
#   boxes(0.2), random_rough(0.2), hf_pyramid_slope(0.1), hf_pyramid_slope_inv(0.1)
```

## Sensor Configs

### ContactSensorCfg (Full)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | ContactSensor | Sensor class |
| `prim_path` | str | MISSING | Body prim (regex) |
| `update_period` | float | 0.0 | Update period (s) |
| `history_length` | int | 0 | Past frames to store |
| `debug_vis` | bool | False | Visualization |
| `track_pose` | bool | False | Track sensor pose |
| `track_contact_points` | bool | False | Track contact locations |
| `track_friction_forces` | bool | False | Track friction |
| `max_contact_data_count_per_prim` | int | 4 | Max contacts per prim |
| `track_air_time` | bool | False | Track air/contact time |
| `force_threshold` | float | 1.0 | Contact detection threshold |
| `filter_prim_paths_expr` | list[str] | [] | Contact filter prims |

### RayCasterCfg (Full)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | RayCaster | Sensor class |
| `prim_path` | str | MISSING | Sensor origin prim |
| `update_period` | float | 0.0 | Update period (s) |
| `history_length` | int | 0 | Past frames |
| `mesh_prim_paths` | list[str] | MISSING | Meshes to cast against |
| `offset.pos` | tuple | (0, 0, 0) | Position offset |
| `offset.rot` | tuple | (1, 0, 0, 0) | Rotation offset (wxyz) |
| `ray_alignment` | str | "base" | "base", "yaw", "world" |
| `pattern_cfg` | PatternBaseCfg | MISSING | Ray pattern config |
| `max_distance` | float | 1e6 | Max ray distance |

### CameraCfg (Full)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | Camera | Sensor class |
| `prim_path` | str | MISSING | Camera prim |
| `spawn` | PinholeCameraCfg | MISSING | Camera params |
| `width` | int | MISSING | Image width (px) |
| `height` | int | MISSING | Image height (px) |
| `data_types` | list[str] | ["rgb"] | Data outputs |
| `offset.pos` | tuple | (0, 0, 0) | Position offset |
| `offset.rot` | tuple | (1, 0, 0, 0) | Rotation offset |
| `offset.convention` | str | "ros" | "opengl", "ros", "world" |
| `depth_clipping_behavior` | str | "none" | "max", "zero", "none" |
| `update_period` | float | 0.0 | Update period |
| `semantic_filter` | str | "*:*" | Semantic filter |

### TiledCameraCfg

Same as CameraCfg with `class_type = TiledCamera`. Uses tiled rendering (faster for RL).

### FrameTransformerCfg (Full)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | FrameTransformer | Sensor class |
| `prim_path` | str | MISSING | Source frame |
| `source_frame_offset.pos` | tuple | (0, 0, 0) | Source offset |
| `source_frame_offset.rot` | tuple | (1, 0, 0, 0) | Source rotation |
| `target_frames` | list[FrameCfg] | MISSING | Target frames |

FrameCfg: `prim_path: str`, `name: str | None`, `offset.pos`, `offset.rot`

### ImuCfg (Full)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type | Imu | Sensor class |
| `prim_path` | str | MISSING | IMU body prim |
| `offset.pos` | tuple | (0, 0, 0) | Position offset |
| `offset.rot` | tuple | (1, 0, 0, 0) | Rotation offset |
| `gravity_bias` | tuple | (0, 0, 9.81) | Gravity bias (m/s²) |
| `update_period` | float | 0.0 | Update period |

## Ray Pattern Configs

### GridPatternCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | float | MISSING | Grid spacing (m) |
| `size` | tuple[float, float] | MISSING | Grid size (length, width) |
| `direction` | tuple | (0, 0, -1) | Ray direction |
| `ordering` | str | "xy" | Point ordering |

### PinholeCameraPatternCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `focal_length` | float | 24.0 | Focal length (cm) |
| `horizontal_aperture` | float | 20.955 | Aperture (cm) |
| `width` | int | MISSING | Image width (px) |
| `height` | int | MISSING | Image height (px) |

### BpearlPatternCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizontal_fov` | float | 360.0 | Horizontal FOV (deg) |
| `horizontal_res` | float | 10.0 | Horizontal resolution (deg) |
| `vertical_ray_angles` | Sequence[float] | 32 angles | Vertical angles |
