---
name: isaacsim-asset-pipeline
description: >
  Imports, exports, and validates robot and scene assets in Isaac Sim — URDF/MJCF import, USD-to-URDF export, heightmap terrain generation, and asset conversion.
layer: L2
domain: [robotics, simulation]
source-project: IsaacSim
depends-on: [isaacsim-simulation-core]
tags: [urdf, mjcf, usd, assets, import-export]
---

# Isaac Sim Asset Pipeline

Asset pipeline extensions handle importing robot descriptions, exporting for external tools, generating procedural assets, and validating correctness.

## Extensions

| Extension | Purpose |
|-----------|---------|
| `isaacsim.asset.importer.urdf` | URDF → USD import |
| `isaacsim.asset.importer.mjcf` | MJCF (MuJoCo) → USD import |
| `isaacsim.asset.importer.heightmap` | Heightmap → USD terrain |
| `isaacsim.asset.exporter.urdf` | USD → URDF export |
| `isaacsim.asset.validation` | Asset validation rules |
| `isaacsim.asset.gen.conveyor` | Conveyor belt generation |
| `isaacsim.asset.gen.omap` | Occupancy map generation |

## URDF Import

The URDF importer uses a command pattern:

```python
import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf as urdf

# 1. Create import configuration
result, import_config = omni.kit.commands.execute(
    "URDFCreateImportConfig"
)

# 2. Configure import options
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = True
import_config.make_default_prim = True
import_config.create_physics_scene = True
import_config.default_drive_type = 1          # 1 = position, 2 = velocity
import_config.default_drive_strength = 1e4    # PD position gain
import_config.default_position_drive_damping = 1e2  # PD velocity gain
import_config.self_collision = False
import_config.distance_scale = 1.0

# 3. Parse URDF file
result, urdf_path = omni.kit.commands.execute(
    "URDFParseFile",
    urdf_path="/path/to/robot.urdf",
    import_config=import_config,
)

# 4. Import robot to stage
result, prim_path = omni.kit.commands.execute(
    "URDFImportRobot",
    urdf_path="/path/to/robot.urdf",
    import_config=import_config,
    dest_path="/World/Robot",
)
```

### Combined Parse and Import

```python
result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="/path/to/robot.urdf",
    import_config=import_config,
    dest_path="/World/Robot",
)
```

### Import Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `merge_fixed_joints` | bool | False | Merge links connected by fixed joints |
| `convex_decomp` | bool | False | Use convex decomposition for collision |
| `import_inertia_tensor` | bool | True | Import inertia from URDF |
| `fix_base` | bool | True | Fix the base link |
| `make_default_prim` | bool | True | Set as stage default prim |
| `create_physics_scene` | bool | True | Create PhysicsScene prim |
| `default_drive_type` | int | 1 | 1=position, 2=velocity |
| `default_drive_strength` | float | 1e4 | Position gain (Kp) |
| `default_position_drive_damping` | float | 1e2 | Velocity gain (Kd) |
| `self_collision` | bool | False | Enable self-collision |
| `distance_scale` | float | 1.0 | Scale factor for distances |

### Collision Approximation Modes

When `convex_decomp` is enabled, collision meshes are approximated:
- **Convex hull** - Single convex hull per link (fast, less accurate)
- **Convex decomposition** - Multiple convex hulls (slower, more accurate)
- **Mesh simplification** - Simplified triangle mesh
- **None** - Use original mesh (most accurate, slowest)

## MJCF Import (MuJoCo)

Similar command pattern for MuJoCo XML files:

```python
import omni.kit.commands

# 1. Create config
result, import_config = omni.kit.commands.execute(
    "MJCFCreateImportConfig"
)

# 2. Configure
import_config.fix_base = True
import_config.import_inertia_tensor = True
import_config.import_sites = False
import_config.visualize_collision_geoms = False
import_config.self_collision = False
import_config.create_physics_scene = True
import_config.make_default_prim = True

# 3. Import
result, prim_path = omni.kit.commands.execute(
    "MJCFCreateAsset",
    mjcf_path="/path/to/robot.xml",
    import_config=import_config,
    dest_path="/World/Robot",
)
```

### MJCF-Specific Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fix_base` | bool | True | Fix base link |
| `import_inertia_tensor` | bool | True | Use MJCF inertia |
| `import_sites` | bool | False | Import site markers |
| `visualize_collision_geoms` | bool | False | Show collision geometry |
| `self_collision` | bool | False | Enable self-collision |
| `create_physics_scene` | bool | True | Add PhysicsScene |
| `make_default_prim` | bool | True | Set as default prim |
| `create_instanceable_asset` | bool | False | Create instanceable version |

## URDF Export

Export USD articulations back to URDF for ROS consumption:

```python
from isaacsim.asset.exporter.urdf import UrdfExporter

exporter = UrdfExporter()
# Configure mesh export options:
# - mesh_dir: output directory for mesh files
# - mesh_path_prefix: "file://", "package://", "./" for mesh references
```

Uses `nvidia.srl.from_usd.to_urdf.UsdToUrdf` internally.

## Heightmap Import

Generate terrain from heightmap images:

```python
# Extension: isaacsim.asset.importer.heightmap
# Converts grayscale heightmap images to USD terrain geometry
```

## Asset Conversion (FBX/OBJ/glTF → USD)

For non-robot 3D assets, use the Kit asset converter:

```python
import omni.kit.asset_converter

converter = omni.kit.asset_converter.get_instance()

task = converter.create_converter_task(
    input_path="/path/to/model.fbx",
    output_path="/path/to/model.usd",
)
success = await task.wait_until_finished()
```

Supported formats: FBX, OBJ, glTF/GLB, STL.

## Conveyor Belt Generation

Create physics-enabled conveyor belts:

```python
import omni.kit.commands

# Creates an action graph with conveyor physics
omni.kit.commands.execute(
    "CreateConveyorBelt",
    prim_path="/World/Conveyor",
    # Applies RigidBodyAPI, CollisionAPI, PhysxSurfaceVelocityAPI
)
```

## Occupancy Map Generation

Generate 2D occupancy maps from 3D scenes:

```python
from isaacsim.asset.gen.omap import acquire_omap_interface

omap = acquire_omap_interface()
# Generate occupancy grid from scene geometry
```

Used by MobilityGen for navigation training data.

## Asset Validation

The `isaacsim.asset.validation` extension provides validation rules:

| Rule Category | Checks |
|---------------|--------|
| Robot Naming | Naming conventions, folder structure |
| Drive Rules | Drive API configuration |
| Material Rules | Material assignments |
| Joint Rules | Joint limits, drive parameters |
| Physics Rules | Physics parameters, collision setup |

## Reference Files

- [import-export-reference.md](import-export-reference.md) - Complete import/export API reference

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/extensions/isaacsim.asset.importer.urdf/` | URDF importer |
| `source/extensions/isaacsim.asset.importer.mjcf/` | MJCF importer |
| `source/extensions/isaacsim.asset.importer.heightmap/` | Heightmap import |
| `source/extensions/isaacsim.asset.exporter.urdf/` | URDF exporter |
| `source/extensions/isaacsim.asset.validation/` | Validation rules |
| `source/extensions/isaacsim.asset.gen.conveyor/` | Conveyor generation |
| `source/extensions/isaacsim.asset.gen.omap/` | Occupancy map |
| `source/standalone_examples/api/isaacsim.asset.importer.urdf/` | URDF examples |
| `source/standalone_examples/api/isaacsim.asset.importer.mjcf/` | MJCF examples |
