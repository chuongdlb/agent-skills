# Class Hierarchy Reference

Detailed class hierarchy for the Isaac Sim core API. Source files are under `source/extensions/isaacsim.core.api/` and `source/extensions/isaacsim.core.prims/`.

## SimulationContext → World

### SimulationContext

**File:** `source/extensions/isaacsim.core.api/python/impl/simulation_context/simulation_context.py`

```python
class SimulationContext:
    _instance = None  # Singleton

    def __init__(self, physics_dt=None, rendering_dt=None,
                 stage_units_in_meters=None, physics_prim_path="/physicsScene",
                 sim_params=None, set_defaults=True, backend="numpy", device=None)
```

**Key Properties:**
- `app` - Kit Application interface
- `current_time_step_index` (int) - Physics step counter
- `current_time` (float) - Simulation time in seconds
- `stage` - Current USD stage
- `backend` (str) - "numpy", "torch", or "warp"
- `device` (str) - None for CPU, "cuda:N" for GPU
- `backend_utils` - Backend-specific utilities module
- `physics_sim_view` - Physics tensor view (available after reset)

**Key Methods:**
- `instance()` (classmethod) - Get singleton
- `clear_instance()` (classmethod) - Destroy singleton
- `get_physics_context()` - Returns PhysicsContext
- `set_simulation_dt(physics_dt, rendering_dt)`
- `get_physics_dt()` / `get_rendering_dt()`
- `is_playing()` / `is_stopped()`
- `play()` / `pause()` / `stop()`

### World (extends SimulationContext)

**File:** `source/extensions/isaacsim.core.api/python/impl/world/world.py`

```python
class World(SimulationContext):
    def __init__(self, physics_dt=None, rendering_dt=None,
                 stage_units_in_meters=None, physics_prim_path="/physicsScene",
                 sim_params=None, set_defaults=True, backend="numpy", device=None)
```

**Additional Properties:**
- `scene` (Scene) - Scene object registry

**Additional Methods:**
- `add_task(task: BaseTask)` - Register a task
- `get_task(name)` - Get task by name
- `get_current_tasks()` - All registered tasks
- `get_observations(task_name=None)` - Task observations
- `calculate_metrics(task_name=None)` - Task metrics
- `is_done(task_name=None)` - Task done state
- `get_data_logger()` - DataLogger instance
- `initialize_physics()` - Init physics views
- `reset(soft=False)` - Reset simulation
- `reset_async()` - Async reset
- `step(render=True, step_sim=True)` - Advance simulation
- `step_async(render=True)` - Async step
- `clear()` - Clear the stage
- `add_physics_callback(name, callback_fn)` - Per-step callback
- `add_timeline_callback(name, callback_fn)` - Timeline events
- `add_render_callback(name, callback_fn)` - Render events
- `add_stage_callback(name, callback_fn)` - Stage events

## Scene and SceneRegistry

**File:** `source/extensions/isaacsim.core.api/python/impl/scenes/scene.py`

```python
class Scene:
    def __init__(self)
```

**Properties:**
- `stage` - Current USD stage

**Methods:**
- `add(obj: SingleXFormPrim)` - Add object to registry
- `add_default_ground_plane(**kwargs)` - Create default ground
- `add_ground_plane(**kwargs)` - Create custom ground
- `get_object(name: str)` - Retrieve by name
- `object_exists(name: str)` - Check existence
- `remove_object(name: str, registry_only=False)`
- `clear(registry_only=False)` - Remove all objects
- `post_reset()` - Called after world.reset(), resets all objects

## Prim Wrapper Hierarchy

### Single-Prim Classes

```
SingleXFormPrim
  ├── SingleRigidPrim
  │     └── SingleArticulation
  │           └── Robot
  │                 ├── SingleManipulator  (isaacsim.robot.manipulators)
  │                 └── WheeledRobot       (isaacsim.robot.wheeled_robots)
  ├── SingleGeometryPrim
  ├── SingleClothPrim
  ├── SingleDeformablePrim
  ├── SingleParticleSystem
  └── BaseSensor                           (isaacsim.core.api)
        ├── Camera                          (isaacsim.sensors.camera)
        ├── ContactSensor                   (isaacsim.sensors.physics)
        ├── IMUSensor                       (isaacsim.sensors.physics)
        ├── RotatingLidarPhysX             (isaacsim.sensors.physx)
        └── LidarRtx                        (isaacsim.sensors.rtx)
```

### Multi-Prim View Classes

```
XFormPrim (Prim)
  ├── RigidPrim
  │     └── Articulation
  │           └── RobotView
  ├── GeometryPrim
  ├── ClothPrim
  ├── DeformablePrim
  └── ParticleSystem
```

### SingleXFormPrim

**File:** `source/extensions/isaacsim.core.prims/python/impl/single_xform_prim.py`

```python
class SingleXFormPrim:
    def __init__(self, prim_path, name, position=None, translation=None,
                 orientation=None, scale=None, visible=None)
```

**Methods:**
- `initialize(physics_sim_view=None)`
- `post_reset()`
- `get_world_pose()` → (position[3], orientation[4])
- `set_world_pose(position, orientation)`
- `get_local_pose()` → (position[3], orientation[4])
- `set_local_pose(translation, orientation)`
- `get_local_scale()` → scale[3]
- `set_local_scale(scale)`
- `set_visibility(visible: bool)`
- `get_visibility()` → bool
- `prim_path` (property) → str
- `name` (property) → str
- `prim` (property) → Usd.Prim

### SingleRigidPrim (extends SingleXFormPrim)

**File:** `source/extensions/isaacsim.core.prims/python/impl/single_rigid_prim.py`

**Additional constructor params:** `mass`, `density`, `linear_velocity`, `angular_velocity`

**Additional Methods:**
- `get_linear_velocity()` → vel[3]
- `set_linear_velocity(velocity)`
- `get_angular_velocity()` → vel[3]
- `set_angular_velocity(velocity)`
- `get_mass()` → float
- `set_mass(mass)`
- `get_density()` → float
- `set_density(density)`
- `get_default_state()` → state dict
- `set_default_state(position, orientation, linear_velocity, angular_velocity)`

### SingleArticulation (extends SingleRigidPrim)

**File:** `source/extensions/isaacsim.core.prims/python/impl/single_articulation.py`

**Properties:**
- `num_dof` → int
- `dof_names` → List[str]

**Additional Methods:**
- `get_joint_positions()` → positions[num_dof]
- `set_joint_positions(positions)`
- `get_joint_velocities()` → velocities[num_dof]
- `set_joint_velocities(velocities)`
- `get_joint_efforts()` → efforts[num_dof]
- `get_applied_action()` → ArticulationAction
- `apply_action(control_actions: ArticulationAction)`
- `get_articulation_controller()` → ArticulationController
- `set_joint_positions_default(positions)`
- `set_solver_position_iteration_count(count)`
- `set_solver_velocity_iteration_count(count)`

## Controller Classes

### ArticulationAction

```python
from isaacsim.core.utils.types import ArticulationAction

action = ArticulationAction(
    joint_positions=np.array([...]),     # Target positions
    joint_velocities=np.array([...]),    # Target velocities
    joint_efforts=np.array([...]),       # Target efforts/torques
    joint_indices=np.array([...]),       # Optional: specific joints
)
```

### BaseController

**File:** `source/extensions/isaacsim.core.api/python/impl/controllers/base_controller.py`

```python
class BaseController(ABC):
    def __init__(self, name: str)

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArticulationAction: ...
    def reset(self): ...
```

### ArticulationController

**File:** `source/extensions/isaacsim.core.api/python/impl/controllers/articulation_controller.py`

```python
class ArticulationController:
    def initialize(self, articulation_view)
    def apply_action(self, control_actions: ArticulationAction)
    def set_gains(self, kps, kds, save_to_usd=False)
    def get_gains() -> (kps, kds)
    def switch_control_mode(self, mode: str)  # "position", "velocity", "effort"
    def switch_dof_control_mode(self, dof_index: int, mode: str)
    def get_joint_limits() -> (lower, upper)
    def get_applied_action() -> ArticulationAction
    def set_max_efforts(efforts)
    def get_max_efforts() -> efforts
```

## PhysicsContext

**File:** `source/extensions/isaacsim.core.api/python/impl/physics_context/physics_context.py`

Configuration for the physics scene:

- `set_gravity(value)` - e.g., `set_gravity(Gf.Vec3f(0, 0, -9.81))`
- `set_physics_dt(dt)` - Physics timestep
- `enable_ccd(flag)` - Continuous collision detection
- `enable_stablization(flag)` - Solver stabilization
- `enable_gpu_dynamics(flag)` - GPU-accelerated dynamics
- `set_solver_type(solver)` - "TGS" or "PGS"
- `set_broadphase_type(broadphase)` - "MBP" or "SAP"

## BaseTask

**File:** `source/extensions/isaacsim.core.api/python/impl/tasks/base_task.py`

```python
class BaseTask:
    def __init__(self, name, offset=None)

    def set_up_scene(self, scene: Scene): ...  # Add objects to scene
    def get_task_objects() -> dict: ...         # Return task objects
    def pre_step(self, time_step_index, simulation_time): ...  # Called before step
    def post_reset(self): ...                   # Called after reset
    def get_observations() -> dict: ...         # Return observations
    def calculate_metrics() -> dict: ...        # Return metrics
    def is_done() -> bool: ...                  # Check if task complete
    def name (property) -> str
    def scene (property) -> Scene
```
