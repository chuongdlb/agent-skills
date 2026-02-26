---
name: isaacsim-synthetic-data
description: >
  Generates synthetic training data using Replicator — domain randomization, behavior scripting, SDG workflows, custom writers and annotators.
layer: L2
domain: [robotics, simulation, ml-training]
source-project: IsaacSim
depends-on: [isaacsim-simulation-core, isaacsim-sensor-development]
tags: [synthetic-data, domain-randomization, replicator, sdg]
---

# Isaac Sim Synthetic Data Generation

Isaac Sim uses NVIDIA's Omni.Replicator framework (`omni.replicator.core`) with Isaac-specific extensions for domain randomization, custom writers, grasping workflows, and mobility data generation.

## Extensions

| Extension | Purpose |
|-----------|---------|
| `isaacsim.replicator.examples` | SDG examples and tutorials |
| `isaacsim.replicator.domain_randomization` | Physics and visual randomization |
| `isaacsim.replicator.writers` | Custom writers (Pose, DOPE, YCB, PyTorch) |
| `isaacsim.replicator.grasping` | Grasping dataset generation |
| `isaacsim.replicator.mobility_gen` | Mobile robot scene generation |
| `isaacsim.replicator.behavior` | Behavior scripting for dynamic scenes |
| `isaacsim.replicator.synthetic_recorder` | Recording synthetic data |

## SDG Workflow

The standard pipeline follows: **Scene → Randomizers → Annotators → Writers → Trigger**

```python
import omni.replicator.core as rep
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

# 1. Scene setup
rep.create.cube(position=(0, 0, 0.5))
rep.create.light(light_type="distant")

# 2. Randomizers (run each frame)
with rep.trigger.on_frame():
    rep.randomizer.rotation()
    rep.randomizer.materials()

# 3. Camera with render product
camera = rep.create.camera(position=(2, 0, 1), look_at=(0, 0, 0))
render_product = rep.create.render_product(camera, (640, 480))

# 4. Annotators
rep.annotators.get("rgb").attach(render_product)
rep.annotators.get("semantic_segmentation").attach(render_product)
rep.annotators.get("bounding_box_2d_tight").attach(render_product)

# 5. Writer
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="/tmp/sdg_output",
    rgb=True,
    semantic_segmentation=True,
    bounding_box_2d_tight=True,
)
writer.attach(render_product)

# 6. Trigger
rep.orchestrator.run_until_complete(num_frames=100)

simulation_app.close()
```

## Domain Randomization

### Visual Randomization (via Replicator)

```python
import omni.replicator.core as rep

# Randomize object positions
with rep.trigger.on_frame():
    with rep.get.prims(semantics=[("class", "cube")]):
        rep.randomizer.position(
            position=rep.distribution.uniform((-1, -1, 0), (1, 1, 0))
        )
        rep.randomizer.rotation(
            rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360))
        )

# Randomize materials
with rep.trigger.on_frame():
    with rep.get.prims(semantics=[("class", "object")]):
        rep.randomizer.materials(
            materials=rep.distribution.choice(material_list)
        )

# Randomize lighting
with rep.trigger.on_frame():
    with rep.get.prims(path_pattern="/World/Light"):
        rep.randomizer.color(
            color=rep.distribution.uniform((0.5, 0.5, 0.5), (1, 1, 1))
        )
        rep.randomizer.intensity(
            intensity=rep.distribution.uniform(500, 5000)
        )
```

### Physics Randomization (Isaac-specific)

The `isaacsim.replicator.domain_randomization` extension provides OmniGraph nodes for physics parameter randomization:

- `OgnWritePhysicsSimulationContext` - Randomize gravity, global physics
- `OgnWritePhysicsRigidPrimView` - Randomize rigid body properties (mass, friction, restitution)
- `OgnWritePhysicsArticulationView` - Randomize joint properties (stiffness, damping, limits)

```python
from isaacsim.replicator.domain_randomization import physics_view

# Register views for randomization
physics_view.register_simulation_context(world)
physics_view.register_rigid_prim_view(rigid_prim_view)
```

## Annotators

Standard annotators from `omni.replicator.core`:

| Annotator | Output | Notes |
|-----------|--------|-------|
| `rgb` | RGBA image (H, W, 4) uint8 | Color image |
| `distance_to_image_plane` | Depth (H, W) float32 | Z-buffer depth |
| `distance_to_camera` | Depth (H, W) float32 | Euclidean depth |
| `normals` | Normals (H, W, 4) float32 | Surface normals |
| `semantic_segmentation` | Labels (H, W) uint32 | Per-pixel class |
| `instance_id_segmentation` | IDs (H, W) uint32 | Per-pixel instance |
| `instance_segmentation` | IDs (H, W) uint32 | Instance segmentation |
| `bounding_box_2d_tight` | BBox array | Tight 2D boxes |
| `bounding_box_2d_loose` | BBox array | Loose 2D boxes |
| `bounding_box_3d` | BBox array | 3D bounding boxes |
| `motion_vectors` | Motion (H, W, 4) float32 | Optical flow |
| `occlusion` | Occlusion (H, W) float32 | AO map |
| `pointcloud` | Points (N, 3) float32 | 3D point cloud |

## Writers

### Built-in Writers

**BasicWriter** (from omni.replicator.core):
```python
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir="/tmp/output", rgb=True,
                  semantic_segmentation=True, bounding_box_2d_tight=True)
writer.attach(render_product)
```

### Isaac Sim Custom Writers

**PoseWriter** - Object pose output for training pose estimators:
```python
from isaacsim.replicator.writers import PoseWriter

writer = PoseWriter(
    output_dir="/tmp/pose_data",
    class_name_to_index_map={"cube": 0, "sphere": 1},
    format="dope",  # or "centerpose"
)
```

**PyTorchWriter** - Direct conversion to PyTorch tensors:
```python
from isaacsim.replicator.writers import PyTorchWriter, PytorchListener

listener = PytorchListener()
writer = PyTorchWriter(
    output_dir="/tmp/pytorch_output",
    listener=listener,
    device="cuda",
)
# Access data in training loop:
# image_tensor = listener.data["rgb"]
```

**YCBVideoWriter** - YCB object dataset format:
```python
from isaacsim.replicator.writers import YCBVideoWriter
```

**DataVisualizationWriter** - Debug visualization:
```python
from isaacsim.replicator.writers import DataVisualizationWriter
```

### Custom Writer Pattern

```python
from omni.replicator.core import Writer, BackendDispatch

class MyWriter(Writer):
    def __init__(self, output_dir, **kwargs):
        self.backend = BackendDispatch(output_dir=output_dir)
        self.version = "1.0"
        self._frame_id = 0

    def write(self, data: dict) -> None:
        for annotator_name, annotator_data in data.items():
            # Process each annotator's output
            self.backend.write_image(
                f"frame_{self._frame_id}_{annotator_name}.png",
                annotator_data["data"]
            )
        self._frame_id += 1

# Register the writer
rep.WriterRegistry.register(MyWriter)
```

## Grasping SDG

Automated grasp dataset generation (`isaacsim.replicator.grasping`):

```python
from isaacsim.replicator.grasping import GraspingManager, GraspPhase

manager = GraspingManager()

# Configure grasp phases
phase = GraspPhase(
    name="close_gripper",
    simulation_steps=32,
    simulation_step_dt=1/60,
)
phase.add_joint("finger_joint1", target=0.0)
phase.add_joint("finger_joint2", target=0.0)

# Sampler configuration
manager.set_sampler_config({
    "sampler_type": "antipodal",
    "num_candidates": 100,
    "num_orientations": 1,
    "gripper_maximum_aperture": 0.08,
    "gripper_standoff_fingertips": 0.17,
})
```

## MobilityGen

Mobile robot scene generation for navigation training (`isaacsim.replicator.mobility_gen`):

```python
from isaacsim.replicator.mobility_gen import MobilityGenRobot

# Abstract base class for custom robots
class MyRobot(MobilityGenRobot):
    physics_dt = 1/60
    z_offset = 0.1
    occupancy_map_radius = 0.3
    occupancy_map_collision_radius = 0.35
    front_camera_type = Camera
    keyboard_linear_velocity_gain = 1.0
    keyboard_angular_velocity_gain = 1.0

    def build(self): ...
    def write_action(self, action): ...
```

## Behavior Scripting

The `isaacsim.replicator.behavior` extension enables scripted object behaviors for dynamic scene generation (falling objects, conveyor belts, etc.).

## Generation Modes

**Trigger-based:**
```python
rep.orchestrator.run_until_complete(num_frames=100)
```

**Timeline-based:**
```python
# SDG runs alongside simulation timeline
rep.orchestrator.run()
# ... simulation runs ...
rep.orchestrator.stop()
```

## Reference Files

- [sdg-workflows.md](sdg-workflows.md) - Step-by-step workflow documentation

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/extensions/isaacsim.replicator.examples/` | SDG examples |
| `source/extensions/isaacsim.replicator.domain_randomization/` | DR nodes |
| `source/extensions/isaacsim.replicator.writers/` | Custom writers |
| `source/extensions/isaacsim.replicator.grasping/` | Grasping workflow |
| `source/extensions/isaacsim.replicator.mobility_gen/` | MobilityGen |
| `source/extensions/isaacsim.replicator.behavior/` | Behavior scripts |
| `source/standalone_examples/replicator/` | Standalone examples |
