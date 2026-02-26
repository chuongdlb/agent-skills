---
name: isaacsim-simulation-core
description: >
  Core Isaac Sim simulation API including World, SimulationContext, Scene, PhysicsContext singletons, prim wrappers, backend selection, and the experimental Core API.
layer: L2
domain: [robotics, simulation]
source-project: IsaacSim
depends-on: []
tags: [simulation, physics, usd, world, prim-wrappers]
---

# Isaac Sim Simulation Core

The core API provides the primary Python interface for controlling simulation state, managing scenes, and interacting with USD prims. It lives in `isaacsim.core.api` with prim wrappers in `isaacsim.core.prims`.

## Architecture

```
World (singleton, inherits SimulationContext)
  ├── Scene (object registry)
  │     ├── SingleXFormPrim / XFormPrim (multi-prim view)
  │     ├── SingleRigidPrim / RigidPrim
  │     ├── SingleArticulation / Articulation
  │     ├── Robot / RobotView
  │     ├── BaseSensor
  │     ├── GroundPlane
  │     └── ... (15+ object types)
  ├── PhysicsContext (physics settings)
  └── DataLogger
```

## World Singleton

`World` is the top-level API combining scene management and simulation control:

```python
from isaacsim.core.api import World

# Create (first call creates singleton, subsequent calls return it)
my_world = World(stage_units_in_meters=1.0)

# Retrieve existing instance
world = World.instance()

# Destroy singleton (call before creating new one)
World.clear_instance()
```

Constructor parameters:
- `physics_dt` (float) - Physics timestep, default `1/60`
- `rendering_dt` (float) - Rendering timestep, default `1/60`
- `stage_units_in_meters` (float) - Stage unit scale, default `1.0`
- `physics_prim_path` (str) - Physics scene prim, default `"/physicsScene"`
- `backend` (str) - `"numpy"`, `"torch"`, or `"warp"`
- `device` (str) - `None` (CPU) or `"cuda:0"` for GPU physics
- `set_defaults` (bool) - Apply default physics settings, default `True`

Default physics settings (when `set_defaults=True`):
- Gravity: -9.81 m/s^2
- CCD enabled, stabilization enabled
- GPU dynamics disabled
- Solver: TGS, Broadcast: MBP

## Simulation Lifecycle

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
import numpy as np

# 1. Create World
my_world = World(stage_units_in_meters=1.0)

# 2. Populate scene
from isaacsim.core.api.objects import DynamicCuboid
cube = my_world.scene.add(
    DynamicCuboid(prim_path="/World/cube", name="my_cube",
                  position=np.array([0, 0, 1.0]), size=0.5,
                  color=np.array([255, 0, 0]))
)
my_world.scene.add_default_ground_plane()

# 3. Reset (initializes physics, must be called before stepping)
my_world.reset()

# 4. Step loop
for i in range(500):
    my_world.step(render=True)
    pose = cube.get_world_pose()

# 5. Cleanup
simulation_app.close()
```

### Key Lifecycle Rules

- `reset()` must be called before any `step()` calls
- First `reset()` initializes physics views and articulation controllers
- `reset(soft=True)` preserves physics state, only resets registered object defaults
- `step(render=True)` advances both physics and rendering
- `step(render=False)` advances physics only (faster for headless)
- Multiple reset cycles are supported (outer loop with reset, inner loop with steps)

## Scene Management

`world.scene` is the Scene instance that registers and manages simulation objects:

```python
# Add objects
prim = my_world.scene.add(SingleRigidPrim(prim_path="/World/obj", name="obj"))

# Retrieve objects
obj = my_world.scene.get_object("obj")

# Check existence
exists = my_world.scene.object_exists("obj")

# Remove
my_world.scene.remove_object("obj")

# Ground planes
my_world.scene.add_default_ground_plane()
my_world.scene.add_ground_plane(size=100, z_position=-0.5)
```

Supported object types:
- `SingleXFormPrim` / `XFormPrim` - Transform wrappers
- `SingleRigidPrim` / `RigidPrim` - Rigid bodies
- `SingleGeometryPrim` / `GeometryPrim` - Geometry
- `SingleArticulation` / `Articulation` - Articulated systems
- `Robot` / `RobotView` - Robots
- `BaseSensor` - Sensors
- `GroundPlane` - Ground planes
- `SingleClothPrim` / `ClothPrim` - Cloth simulation
- `SingleDeformablePrim` / `DeformablePrim` - Deformable bodies
- `SingleParticleSystem` / `ParticleSystem` - Particles

## Prim Wrappers

All wrappers exist in two forms:
- `Single*` classes: wrap a single USD prim
- View classes (no prefix): wrap multiple prims via regex patterns for parallel access

### SingleXFormPrim (base for all single-prim wrappers)

```python
from isaacsim.core.prims import SingleXFormPrim

prim = SingleXFormPrim(prim_path="/World/my_prim", name="my_prim")
prim.set_world_pose(position=np.array([1, 0, 0]),
                    orientation=np.array([1, 0, 0, 0]))  # wxyz quaternion
pos, orient = prim.get_world_pose()
prim.set_local_scale(np.array([1, 1, 1]))
prim.set_visibility(True)
```

### SingleRigidPrim

```python
from isaacsim.core.prims import SingleRigidPrim

rigid = SingleRigidPrim(prim_path="/World/box", name="box",
                        mass=1.0, linear_velocity=np.array([0, 0, 1]))
rigid.set_linear_velocity(np.array([1, 0, 0]))
rigid.set_angular_velocity(np.array([0, 0, 1]))
vel = rigid.get_linear_velocity()
```

### SingleArticulation

```python
from isaacsim.core.prims import SingleArticulation

art = SingleArticulation(prim_path="/World/Robot", name="robot")
# After world.reset():
num_dofs = art.num_dof
positions = art.get_joint_positions()
art.set_joint_positions(np.zeros(num_dofs))
```

### Multi-Prim Views (XFormPrim, RigidPrim, Articulation)

```python
from isaacsim.core.prims import RigidPrim

# Match multiple prims via regex
view = RigidPrim(prim_paths_expr="/World/Env[0-9]*/box", name="boxes")
# After world.reset():
positions = view.get_world_poses()  # Shape: (N, 3) and (N, 4)
view.set_velocities(np.zeros((view.count, 6)))  # linear + angular
```

## Backend Selection

```python
# At World creation
world = World(backend="torch", device="cuda:0")

# Or per-module
from isaacsim.core.utils.torch.tensor_utils import clone_tensor
```

Backends: `"numpy"` (CPU, default), `"torch"` (CPU/GPU), `"warp"` (GPU).

## Experimental API

The experimental API (`isaacsim.core.experimental.*`) is Warp-based with automatic device/dtype conversion:

```python
from isaacsim.core.experimental.prims import XformPrim, RigidPrim, Articulation

# All wrappers are multi-prim (no Single* variants)
prim = XformPrim("/World/my_prim")
rigid = RigidPrim("/World/box")
art = Articulation("/World/Robot")
```

**Backend fallback order:** tensor → fabric → usdrt → usd (auto-selects fastest available).

Key differences from current API:
- No `Single*` prefix needed (all wrappers handle one or many prims)
- Uses `paths` parameter (not `prim_path` / `prim_paths_expr`)
- Warp arrays as native data type
- Automatic backend selection with `use_backend()` context manager

## Callbacks

```python
# Physics step callback (called every physics step)
my_world.add_physics_callback("my_cb", callback_fn=my_physics_step)

# Timeline callback (play/pause/stop events)
my_world.add_timeline_callback("my_cb", callback_fn=my_timeline_handler)

# Render callback
my_world.add_render_callback("my_cb", callback_fn=my_render_handler)

# Stage event callback
my_world.add_stage_callback("my_cb", callback_fn=my_stage_handler)
```

## Reference Files

- [class-hierarchy.md](class-hierarchy.md) - Full class hierarchy with methods
- [lifecycle-patterns.md](lifecycle-patterns.md) - Detailed lifecycle and callback patterns

## Key Repo Files

| File | Description |
|------|-------------|
| `source/extensions/isaacsim.core.api/python/impl/world/world.py` | World class |
| `source/extensions/isaacsim.core.api/python/impl/simulation_context/simulation_context.py` | SimulationContext |
| `source/extensions/isaacsim.core.api/python/impl/scenes/scene.py` | Scene class |
| `source/extensions/isaacsim.core.prims/python/impl/` | All prim wrappers |
| `source/extensions/isaacsim.core.experimental.prims/` | Experimental prim API |
| `source/standalone_examples/api/isaacsim.core.api/add_cubes.py` | Basic example |
| `source/standalone_examples/api/isaacsim.core.experimental/` | Experimental examples |
