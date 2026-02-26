# Lifecycle Patterns

Detailed simulation lifecycle patterns, callback systems, and async vs sync usage.

## Simulation Lifecycle Sequence

### First Reset

```
World.__init__()
  └── Creates PhysicsContext, Scene, DataLogger

world.scene.add(obj)           # Populate scene (before reset)
world.add_task(task)           # Register tasks (before reset)

world.reset()                  # FIRST RESET
  ├── task.set_up_scene(scene) # Tasks populate scene
  ├── initialize_physics()     # Creates PhysicsSimulationView
  │   ├── Articulation views initialized
  │   ├── RigidBody views initialized
  │   └── Sensor views initialized
  ├── scene.post_reset()       # Resets all objects to defaults
  └── task.post_reset()        # Tasks reset internal state
```

### Step Loop

```
world.step(render=True)
  ├── task.pre_step(step_idx, sim_time)  # Pre-step callbacks
  ├── Physics step (1/60s default)
  │   └── physics_callback_fn()          # Registered physics callbacks
  ├── Render step (if render=True)
  │   └── render_callback_fn()           # Registered render callbacks
  └── Returns
```

### Subsequent Resets

```
world.reset(soft=False)  # Hard reset
  ├── Stops timeline
  ├── scene.post_reset()  # Resets ALL object states to defaults
  └── task.post_reset()

world.reset(soft=True)   # Soft reset
  ├── Does NOT stop timeline
  ├── scene.post_reset()  # Only resets registered default states
  └── task.post_reset()
```

## Standard Standalone Script Pattern

```python
import numpy as np
from isaacsim import SimulationApp

# 1. Create app (must be first Isaac import)
simulation_app = SimulationApp({"headless": False})

# 2. Import Isaac modules (AFTER SimulationApp creation)
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

# 3. Create world
my_world = World(stage_units_in_meters=1.0)

# 4. Add objects to scene
cube = my_world.scene.add(
    DynamicCuboid(prim_path="/World/cube", name="cube",
                  position=np.array([0, 0, 1.0]), size=0.5,
                  color=np.array([255, 0, 0]))
)
my_world.scene.add_default_ground_plane()

# 5. Reset + step loop (can repeat)
for episode in range(5):
    my_world.reset()
    for step in range(500):
        my_world.step(render=True)
        pos, orient = cube.get_world_pose()
        vel = cube.get_linear_velocity()

# 6. Cleanup
simulation_app.close()
```

## Extension Pattern (Async)

Inside extensions, use async patterns since the simulation is already running:

```python
import asyncio
import omni.ext
from isaacsim.core.api import World

class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._world = None
        self._running = False

    async def setup_scene(self):
        self._world = World.instance() or World()
        # Add objects...
        await self._world.reset_async()
        self._running = True

    async def run_loop(self):
        while self._running:
            await self._world.step_async(render=True)
            # Process observations...

    def on_shutdown(self):
        self._running = False
        if self._world:
            self._world.clear()
```

## Callback Registration

### Physics Step Callback

Called every physics step (before rendering):

```python
def my_physics_step(step_size: float):
    """step_size: physics dt in seconds"""
    positions = robot.get_joint_positions()
    # Compute and apply control

my_world.add_physics_callback("my_controller", callback_fn=my_physics_step)

# Remove when done
my_world.remove_physics_callback("my_controller")
```

### Timeline Callback

Called on play/pause/stop events:

```python
import omni.timeline

def my_timeline_handler(event):
    if event.type == int(omni.timeline.TimelineEventType.PLAY):
        print("Simulation started")
    elif event.type == int(omni.timeline.TimelineEventType.STOP):
        print("Simulation stopped")
    elif event.type == int(omni.timeline.TimelineEventType.PAUSE):
        print("Simulation paused")

my_world.add_timeline_callback("my_timeline", callback_fn=my_timeline_handler)
```

### Render Callback

Called every render frame:

```python
def my_render_step(event):
    # Read sensor data, update UI, etc.
    pass

my_world.add_render_callback("my_render", callback_fn=my_render_step)
```

### Stage Event Callback

Called on stage open/close events:

```python
def my_stage_handler(event):
    pass

my_world.add_stage_callback("my_stage", callback_fn=my_stage_handler)
```

## DataLogger

Record simulation data across frames:

```python
logger = my_world.get_data_logger()

def data_frame_fn(tasks, scene):
    return {
        "cube_position": cube.get_world_pose()[0],
        "cube_velocity": cube.get_linear_velocity(),
    }

logger.add_data_frame_logging_func(data_frame_fn)
logger.start()

# ... run simulation ...

logger.stop()
data = logger.get_data_log()
logger.save(log_path="/tmp/sim_data.json")
logger.reset()
```

## Task Integration Pattern

Tasks encapsulate reusable simulation scenarios:

```python
from isaacsim.core.api.tasks import BaseTask

class MyTask(BaseTask):
    def __init__(self, name="my_task", offset=None):
        super().__init__(name=name, offset=offset)

    def set_up_scene(self, scene):
        """Called during first world.reset(). Add objects here."""
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._cube = scene.add(DynamicCuboid(...))

    def get_observations(self):
        pos, orient = self._cube.get_world_pose()
        return {"position": pos, "orientation": orient}

    def pre_step(self, time_step_index, simulation_time):
        """Called before each physics step."""
        pass

    def post_reset(self):
        """Called after each world.reset()."""
        pass

    def is_done(self):
        return False

    def calculate_metrics(self):
        return {}

# Usage
world = World()
world.add_task(MyTask())
world.reset()

for i in range(100):
    world.step(render=True)
    obs = world.get_observations()
```

## Backend Switching

```python
# NumPy backend (CPU, default)
world = World(backend="numpy")

# PyTorch backend (CPU or GPU)
world = World(backend="torch", device="cuda:0")

# Warp backend (GPU)
world = World(backend="warp", device="cuda:0")
```

Backend affects return types of prim wrapper methods:
- `numpy`: returns `np.ndarray`
- `torch`: returns `torch.Tensor` on specified device
- `warp`: returns `wp.array`

## Multiple Environments (Cloner)

For parallel RL environments, use `isaacsim.core.cloner`:

```python
from isaacsim.core.cloner import GridCloner

cloner = GridCloner(spacing=2.0)
cloner.define_base_env("/World/Env")
prim_paths = cloner.generate_paths("/World/Env", num_paths=64)
cloner.clone(
    source_prim_path="/World/Env",
    prim_paths=prim_paths,
    copy_from_source=True,
)
```

Then use multi-prim view classes (`Articulation`, `RigidPrim`) with regex paths to access all environments in parallel.
