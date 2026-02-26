# SDG Workflows Reference

Step-by-step workflows for common synthetic data generation tasks in Isaac Sim.

## Basic SDG Pipeline

### Step 1: Create Scene

```python
import omni.replicator.core as rep

# Create objects with semantic labels
cube = rep.create.cube(
    position=(0, 0, 0.5),
    semantics=[("class", "cube")],
)
sphere = rep.create.sphere(
    position=(1, 0, 0.5),
    semantics=[("class", "sphere")],
)

# Add lighting
rep.create.light(light_type="distant", intensity=3000, rotation=(45, 0, 0))

# Add ground plane
rep.create.plane(scale=10)
```

### Step 2: Configure Randomizers

```python
with rep.trigger.on_frame():
    # Randomize positions
    with rep.get.prims(semantics=[("class", "cube")]):
        rep.randomizer.position(
            position=rep.distribution.uniform((-2, -2, 0.5), (2, 2, 0.5))
        )
        rep.randomizer.rotation(
            rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360))
        )

    # Randomize lighting
    with rep.get.prims(path_pattern="/World/Light*"):
        rep.randomizer.color(
            color=rep.distribution.uniform((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        )
```

### Step 3: Setup Camera and Render Product

```python
camera = rep.create.camera(
    position=(3, 0, 2),
    look_at=(0, 0, 0),
)
render_product = rep.create.render_product(camera, (1024, 768))
```

### Step 4: Attach Writer

```python
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="/tmp/sdg_basic",
    rgb=True,
    semantic_segmentation=True,
    bounding_box_2d_tight=True,
    bounding_box_3d=True,
    distance_to_image_plane=True,
)
writer.attach([render_product])
```

### Step 5: Generate

```python
# Trigger-based: generate N frames
rep.orchestrator.run_until_complete(num_frames=100)
```

## Domain Randomization Workflow

### Material Randomization

```python
# Create material pool
materials = [
    rep.create.material_omnipbr(diffuse_color=(1, 0, 0)),
    rep.create.material_omnipbr(diffuse_color=(0, 1, 0)),
    rep.create.material_omnipbr(diffuse_color=(0, 0, 1)),
]

with rep.trigger.on_frame():
    with rep.get.prims(semantics=[("class", "object")]):
        rep.randomizer.materials(
            materials=rep.distribution.choice(materials)
        )
```

### Texture Randomization

```python
with rep.trigger.on_frame():
    with rep.get.prims(semantics=[("class", "background")]):
        rep.randomizer.texture(
            textures=rep.distribution.choice(texture_paths)
        )
```

### Camera Randomization

```python
with rep.trigger.on_frame():
    with rep.get.prims(node_type="Camera"):
        rep.randomizer.position(
            position=rep.distribution.uniform((1, -2, 1), (4, 2, 3))
        )
        rep.randomizer.look_at(target=(0, 0, 0))
```

### Physics Parameter Randomization

Using Isaac Sim domain randomization OGN nodes:

```python
from isaacsim.replicator.domain_randomization import physics_view

# Register rigid body views
physics_view.register_rigid_prim_view(my_rigid_view)

# Randomize via OmniGraph nodes:
# - OgnWritePhysicsRigidPrimView: mass, friction, restitution
# - OgnWritePhysicsArticulationView: joint stiffness, damping
# - OgnWritePhysicsSimulationContext: gravity
```

## Custom Writer Implementation

```python
from omni.replicator.core import Writer, BackendDispatch, AnnotatorRegistry
import numpy as np
import json

class CustomTrainingWriter(Writer):
    """Write synthetic data in custom format for ML training."""

    def __init__(self, output_dir: str, image_format: str = "png"):
        self.backend = BackendDispatch(output_dir=output_dir)
        self.version = "1.0"
        self._frame_id = 0
        self._image_format = image_format
        # Define which annotators this writer needs
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            AnnotatorRegistry.get_annotator("semantic_segmentation"),
            AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
        ]

    def write(self, data: dict) -> None:
        """Called each frame with annotator data."""
        frame_dir = f"frame_{self._frame_id:06d}"

        # Save RGB image
        if "rgb" in data:
            self.backend.write_image(
                f"{frame_dir}/rgb.{self._image_format}",
                data["rgb"]["data"]
            )

        # Save bounding boxes as JSON
        if "bounding_box_2d_tight" in data:
            bbox_data = data["bounding_box_2d_tight"]["data"]
            labels = data["bounding_box_2d_tight"].get("info", {}).get("idToLabels", {})
            annotations = []
            for bbox in bbox_data:
                annotations.append({
                    "class": labels.get(str(bbox["semanticId"]), "unknown"),
                    "x_min": float(bbox["x_min"]),
                    "y_min": float(bbox["y_min"]),
                    "x_max": float(bbox["x_max"]),
                    "y_max": float(bbox["y_max"]),
                })
            self.backend.write_blob(
                f"{frame_dir}/annotations.json",
                json.dumps(annotations).encode()
            )

        self._frame_id += 1

# Register
rep.WriterRegistry.register(CustomTrainingWriter)

# Use
writer = rep.WriterRegistry.get("CustomTrainingWriter")
writer.initialize(output_dir="/tmp/custom_output")
writer.attach(render_products)
```

## Custom Annotator Implementation

```python
import omni.replicator.core as rep
import numpy as np

# Define custom annotator from OmniGraph node
def my_custom_annotator(data: dict) -> np.ndarray:
    """Post-process annotator data."""
    rgb = data["rgb"]
    # Custom processing
    grayscale = np.mean(rgb[:, :, :3], axis=2).astype(np.uint8)
    return grayscale

rep.AnnotatorRegistry.register_annotator_from_aov(
    name="my_grayscale",
    aov_name="LdrColor",
    function=my_custom_annotator,
)
```

## Grasping SDG Workflow

```python
from isaacsim.replicator.grasping import GraspingManager, GraspPhase

# 1. Configure manager
manager = GraspingManager()
manager.set_gripper_prim("/World/Gripper")
manager.set_object_prim("/World/Object")

# 2. Define grasp phases
pre_grasp = GraspPhase(name="pre_grasp", simulation_steps=16)
pre_grasp.add_joint("finger_joint1", target=0.04)
pre_grasp.add_joint("finger_joint2", target=0.04)

close_grasp = GraspPhase(name="close", simulation_steps=32)
close_grasp.add_joint("finger_joint1", target=0.0)
close_grasp.add_joint("finger_joint2", target=0.0)

# 3. Configure sampler
manager.set_sampler_config({
    "sampler_type": "antipodal",
    "num_candidates": 100,
    "num_orientations": 1,
    "gripper_maximum_aperture": 0.08,
    "gripper_standoff_fingertips": 0.17,
    "gripper_approach_direction": (0, 0, 1),
    "grasp_align_axis": (0, 1, 0),
    "orientation_sample_axis": (0, 1, 0),
    "random_seed": 42,
})

# 4. Run workflow
manager.run(output_dir="/tmp/grasping_data")
```

## Cosmos Writer Integration

```python
# For Cosmos format output (warehouse scenes)
writer = rep.WriterRegistry.get("CosmosWriter")
writer.initialize(
    output_dir="/tmp/cosmos_output",
    # Cosmos-specific configuration
)
writer.attach(render_products)
```

See: `source/standalone_examples/replicator/cosmos_writer_warehouse.py`

## Timeline-Based Generation

For dynamic scenes where physics simulation must advance:

```python
# Start SDG alongside simulation
rep.orchestrator.run()

# Simulation loop
for frame in range(1000):
    world.step(render=True)
    # Randomization happens automatically per frame

# Stop SDG
rep.orchestrator.stop()
```

## Multi-Camera Capture

```python
cameras = [
    rep.create.camera(position=pos, look_at=(0, 0, 0))
    for pos in [(3, 0, 2), (-3, 0, 2), (0, 3, 2), (0, -3, 2)]
]

render_products = [
    rep.create.render_product(cam, (640, 480))
    for cam in cameras
]

writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir="/tmp/multi_cam", rgb=True)
writer.attach(render_products)

rep.orchestrator.run_until_complete(num_frames=50)
```

## Standalone Example Categories

| Category | Path | Description |
|----------|------|-------------|
| AMR Navigation | `source/standalone_examples/replicator/amr_navigation.py` | Mobile robot SDG |
| Cosmos Writer | `source/standalone_examples/replicator/cosmos_writer_warehouse.py` | Cosmos format |
| Augmentation | `source/standalone_examples/replicator/augmentation/` | Data augmentation |
| Object SDG | `source/standalone_examples/replicator/object_based_sdg/` | Object-centric |
| Pose Generation | `source/standalone_examples/replicator/pose_generation/` | Pose estimation |
| Scene SDG | `source/standalone_examples/replicator/scene_based_sdg/` | Scene-level |
| Online Training | `source/standalone_examples/replicator/online_generation/` | Online with ShapeNet |
| MobilityGen | `source/standalone_examples/replicator/mobility_gen/` | Mobility data |
