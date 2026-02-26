# Import/Export Reference

Complete API reference for asset import and export in Isaac Sim.

## URDFImportConfig Fields

All fields available on the import configuration object returned by `URDFCreateImportConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `merge_fixed_joints` | bool | False | Merge links connected by fixed joints into single body |
| `convex_decomp` | bool | False | Use convex decomposition for collision geometry |
| `import_inertia_tensor` | bool | True | Import inertia tensors from URDF (else auto-compute) |
| `fix_base` | bool | True | Fix the base link to world |
| `make_default_prim` | bool | True | Set imported robot as stage default prim |
| `create_physics_scene` | bool | True | Create a PhysicsScene prim if not present |
| `default_drive_type` | int | 1 | Joint drive type: 1=position, 2=velocity |
| `default_drive_strength` | float | 1e4 | Position drive stiffness (Kp) |
| `default_position_drive_damping` | float | 1e2 | Position drive damping (Kd) |
| `self_collision` | bool | False | Enable self-collision between links |
| `distance_scale` | float | 1.0 | Distance unit scale factor |
| `density` | float | 0.0 | Default density (0 = auto) |

### URDF Import Commands

```python
import omni.kit.commands

# Step 1: Create config
result, config = omni.kit.commands.execute("URDFCreateImportConfig")

# Step 2: Parse only (validates URDF)
result, parsed = omni.kit.commands.execute(
    "URDFParseFile",
    urdf_path="/path/to/robot.urdf",
    import_config=config,
)

# Step 3: Parse and import (single command)
result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="/path/to/robot.urdf",
    import_config=config,
    dest_path="/World/Robot",
)

# Alternative: Import only (after parsing)
result, prim_path = omni.kit.commands.execute(
    "URDFImportRobot",
    urdf_path="/path/to/robot.urdf",
    import_config=config,
    dest_path="/World/Robot",
)

# Parse from string
result, parsed = omni.kit.commands.execute(
    "URDFParseText",
    urdf_string=urdf_xml_string,
    import_config=config,
)
```

## MJCFImportConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fix_base` | bool | True | Fix base link |
| `import_inertia_tensor` | bool | True | Use MJCF inertia (else auto-compute) |
| `import_sites` | bool | False | Import MJCF site markers |
| `visualize_collision_geoms` | bool | False | Show collision geometry |
| `self_collision` | bool | False | Enable self-collision |
| `create_physics_scene` | bool | True | Create PhysicsScene prim |
| `make_default_prim` | bool | True | Set as default prim |
| `create_instanceable_asset` | bool | False | Create instanceable USD |
| `stage_units_per_meter` | float | 1.0 | Stage unit scale |

### MJCF Import Commands

```python
import omni.kit.commands

# Create config
result, config = omni.kit.commands.execute("MJCFCreateImportConfig")

# Import
result, prim_path = omni.kit.commands.execute(
    "MJCFCreateAsset",
    mjcf_path="/path/to/robot.xml",
    import_config=config,
    dest_path="/World/Robot",
)
```

## URDF Export

**Extension:** `isaacsim.asset.exporter.urdf`

Uses `nvidia.srl.from_usd.to_urdf.UsdToUrdf`:

```python
from nvidia.srl.from_usd.to_urdf import UsdToUrdf

converter = UsdToUrdf(
    usd_path="/path/to/robot.usd",
    output_dir="/path/to/output",
)
converter.convert()
```

### Export Options

| Option | Description |
|--------|-------------|
| `mesh_dir` | Output directory for mesh files |
| `mesh_path_prefix` | Prefix for mesh paths in URDF: `"file://"`, `"package://"`, `"./"` |

## Asset Converter (General 3D Formats)

For non-robot assets using `omni.kit.asset_converter`:

```python
import omni.kit.asset_converter

converter = omni.kit.asset_converter.get_instance()

# Create conversion task
task = converter.create_converter_task(
    input_path="/path/to/model.fbx",  # Source file
    output_path="/path/to/model.usd", # Output USD
)
success = await task.wait_until_finished()
```

### Supported Input Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| FBX | `.fbx` | Autodesk format, supports animation |
| OBJ | `.obj` | Wavefront, geometry only |
| glTF | `.gltf`, `.glb` | Khronos standard, PBR materials |
| STL | `.stl` | Triangle meshes, no materials |

## Common Import Issues and Solutions

### Issue: Robot falls through ground
**Cause:** Missing collision geometry or wrong collision approximation
**Fix:** Enable `convex_decomp` in import config, or manually add CollisionAPI

### Issue: Joints not moving
**Cause:** Drive type or gains misconfigured
**Fix:** Check `default_drive_type` (1=position, 2=velocity) and adjust `default_drive_strength` / `default_position_drive_damping`

### Issue: Robot explodes on simulation start
**Cause:** Self-intersecting collision meshes or extreme inertia values
**Fix:** Try `merge_fixed_joints=True`, disable `import_inertia_tensor`, or reduce physics timestep

### Issue: Inaccurate mass/inertia
**Cause:** URDF inertia values not imported or unit mismatch
**Fix:** Enable `import_inertia_tensor=True` and verify `distance_scale` matches URDF units

### Issue: Mesh files not found
**Cause:** URDF `mesh` tags use relative paths or `package://` protocol
**Fix:** Ensure mesh files are accessible; set `distance_scale` if units differ

### Issue: Links merged unexpectedly
**Cause:** `merge_fixed_joints` enabled
**Fix:** Set `merge_fixed_joints=False` to preserve all links

## Collision Approximation Modes

When generating collision geometry from visual meshes:

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| None (original mesh) | Slowest | Highest | Precise contact needed |
| Convex hull | Fast | Low | Simple shapes |
| Convex decomposition | Medium | Medium | Complex concave shapes |
| Mesh simplification | Fast | Medium | Detailed meshes needing simplification |

## Standalone Examples

### URDF Import Example

**File:** `source/standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py`

Demonstrates:
- Creating import configuration
- Setting physics scene properties (gravity, solver)
- Importing Carter robot from URDF
- Configuring wheel drive API for velocity control
- Running simulation loop

### MJCF Import Example

**File:** `source/standalone_examples/api/isaacsim.asset.importer.mjcf/mjcf_import.py`

Demonstrates:
- Creating MJCF import configuration
- Importing ANT robot from MuJoCo XML
- Clean stage setup
