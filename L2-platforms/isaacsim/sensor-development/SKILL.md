---
name: isaacsim-sensor-development
description: >
  Creates and configures sensors in Isaac Sim — cameras, physics-based sensors, PhysX lidar, RTX lidar/radar, annotator pipelines, and lens distortion.
layer: L2
domain: [robotics, simulation]
source-project: IsaacSim
depends-on: [isaacsim-simulation-core]
tags: [sensors, cameras, lidar, radar, annotators]
---

# Isaac Sim Sensor Development

Sensors are organized across four extensions by underlying technology:

| Extension | Sensors | Technology |
|-----------|---------|------------|
| `isaacsim.sensors.camera` | Camera (RGB, depth, stereo, fisheye) | Render-based |
| `isaacsim.sensors.physics` | Contact, Effort, IMU | Physics callbacks |
| `isaacsim.sensors.physx` | Rotating Lidar, Proximity | PhysX raycasting |
| `isaacsim.sensors.rtx` | RTX Lidar, RTX Radar | RTX ray-tracing |

All sensors inherit from `BaseSensor` (which extends `SingleXFormPrim`) in `isaacsim.core.api`.

## Sensor Lifecycle

```
1. ADD:        sensor = world.scene.add(Camera(...))
2. RESET:      world.reset()
3. INITIALIZE: sensor.initialize()  (automatic for most sensors)
4. CONFIGURE:  sensor.add_rgb_to_frame()  (attach annotators/data)
5. STEP:       world.step(render=True)
6. READ:       data = sensor.get_current_frame()
```

## Camera Sensor

The most feature-rich sensor. Supports RGB, depth, segmentation, bounding boxes, point clouds, and more.

```python
from isaacsim.sensors.camera import Camera
import numpy as np

camera = world.scene.add(
    Camera(
        prim_path="/World/camera",
        name="my_camera",
        position=np.array([2.0, 0.0, 1.5]),
        frequency=20,                      # Hz
        resolution=(640, 480),             # (width, height)
    )
)

world.reset()
camera.initialize()

# Attach annotators (choose what data to capture)
camera.add_rgb_to_frame()
camera.add_distance_to_image_plane_to_frame()  # depth
camera.add_semantic_segmentation_to_frame()
camera.add_bounding_box_2d_tight_to_frame()

# Step and read
world.step(render=True)
frame = camera.get_current_frame()
rgba = frame["rgba"]                     # np.ndarray (H, W, 4) uint8
depth = frame["distance_to_image_plane"] # np.ndarray (H, W) float32
```

### Supported Annotators

```python
camera.add_rgb_to_frame()
camera.add_normals_to_frame()
camera.add_motion_vectors_to_frame()
camera.add_occlusion_to_frame()
camera.add_distance_to_image_plane_to_frame()   # Depth
camera.add_distance_to_camera_to_frame()        # Euclidean depth
camera.add_bounding_box_2d_tight_to_frame()
camera.add_bounding_box_2d_loose_to_frame()
camera.add_bounding_box_3d_to_frame()
camera.add_semantic_segmentation_to_frame()
camera.add_instance_id_segmentation_to_frame()
camera.add_instance_segmentation_to_frame()
camera.add_pointcloud_to_frame()
```

Or use the generic method:
```python
camera.attach_annotator("custom_annotator_name", param1=value1)
camera.detach_annotator("normals")
```

### Annotator Device Selection

```python
# Set device at creation
camera = Camera(..., annotator_device="cuda")

# Or per-access
rgba = camera.get_rgba(device="cpu")
```

### Lens Distortion Models

```python
# OpenCV Pinhole model
camera.set_lens_distortion(
    model="pinhole",
    cx=320.0, cy=240.0,
    fx=500.0, fy=500.0,
    k1=-0.1, k2=0.01, p1=0.0, p2=0.0
)

# OpenCV Fisheye model
camera.set_lens_distortion(
    model="fisheye",
    cx=320.0, cy=240.0,
    fx=250.0, fy=250.0,
    k1=-0.05, k2=0.01, k3=0.0, k4=0.0
)
```

See standalone examples:
- `source/standalone_examples/api/isaacsim.sensors.camera/camera_opencv_pinhole.py`
- `source/standalone_examples/api/isaacsim.sensors.camera/camera_opencv_fisheye.py`

## Contact Sensor

Measures contact forces on a collision body:

```python
from isaacsim.sensors.physics import ContactSensor

contact = world.scene.add(
    ContactSensor(
        prim_path="/World/Robot/foot_link/contact_sensor",
        name="foot_contact",
        min_threshold=0,
        max_threshold=10000000,
        radius=0.1,
    )
)

world.reset()
contact.add_raw_contact_data_to_frame()

world.step(render=True)
frame = contact.get_current_frame()
# frame contains force, torque, contact point data
```

The parent prim must have `UsdPhysics.CollisionAPI` applied.

## IMU Sensor

Measures linear acceleration, angular velocity, and orientation:

```python
from isaacsim.sensors.physics import IMUSensor

imu = world.scene.add(
    IMUSensor(
        prim_path="/World/Robot/imu_link/imu",
        name="imu",
        frequency=100,
        linear_acceleration_filter_size=10,
        angular_velocity_filter_size=10,
        orientation_filter_size=10,
    )
)

world.reset()
world.step(render=True)
frame = imu.get_current_frame(read_gravity=True)
# frame["lin_acc"]: linear acceleration (3,)
# frame["ang_vel"]: angular velocity (3,)
# frame["orientation"]: quaternion (4,)
```

## Effort Sensor

Measures joint effort/torque:

```python
from isaacsim.sensors.physics import EffortSensor

effort = EffortSensor(
    prim_path="/World/Robot/joint1",
    sensor_period=0,        # 0 = every physics step
    use_latest_data=False,
    enabled=True,
)

world.reset()
world.step(render=True)
reading = effort.get_sensor_reading()
# reading.is_valid: bool
# reading.time: float
# reading.value: float (torque in Nm)
```

## PhysX Lidar (Rotating)

Raycast-based lidar using PhysX:

```python
from isaacsim.sensors.physx import RotatingLidarPhysX

lidar = world.scene.add(
    RotatingLidarPhysX(
        prim_path="/World/Robot/lidar",
        name="lidar",
        rotation_frequency=20.0,
    )
)

world.reset()
lidar.add_depth_data_to_frame()
lidar.add_point_cloud_data_to_frame()
lidar.enable_visualization()

world.step(render=True)
frame = lidar.get_current_frame()
```

**Key methods:**
- `set_fov(fov)` - Field of view
- `set_resolution(resolution)` - Angular resolution
- `set_valid_range(range)` - Min/max range
- `set_rotation_frequency(freq)` - Rotation speed

## RTX Lidar

Physically accurate lidar using RTX ray-tracing:

```python
from isaacsim.sensors.rtx import LidarRtx

lidar = world.scene.add(
    LidarRtx(
        prim_path="/World/Robot/rtx_lidar",
        name="rtx_lidar",
        config_file_name="HESAI_PandarXT_32",  # Predefined profile
    )
)

world.reset()
lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
lidar.enable_visualization()

world.step(render=True)
frame = lidar.get_current_frame()
```

### Available Lidar Profiles

Predefined sensor configs for: HESAI, NVIDIA, Ouster (OS0/OS1/OS2), SICK, SLAMTEC, Velodyne, ZVISION.

Config files: `source/extensions/isaacsim.sensors.rtx/data/lidar_configs/`

## RTX Radar

Material-aware radar simulation:

```python
# Created via omni.kit.commands
import omni.kit.commands
result, prim = omni.kit.commands.execute(
    "IsaacSensorCreateRtxRadar",
    path="/World/Robot/radar",
    parent=None,
)
```

## OmniGraph Integration

Sensors produce data that flows through OmniGraph computation graphs. ROS2 bridge nodes (`isaacsim.ros2.bridge`) consume sensor outputs via OmniGraph action graphs.

## Reference Files

- [sensor-types-reference.md](sensor-types-reference.md) - Complete sensor catalog with parameters and output formats

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/extensions/isaacsim.sensors.camera/` | Camera sensor |
| `source/extensions/isaacsim.sensors.physics/` | Contact, Effort, IMU |
| `source/extensions/isaacsim.sensors.physx/` | PhysX lidar |
| `source/extensions/isaacsim.sensors.rtx/` | RTX lidar, radar |
| `source/standalone_examples/api/isaacsim.sensors.camera/` | Camera examples |
| `source/standalone_examples/api/isaacsim.sensors.physics/` | Physics sensor examples |
| `source/standalone_examples/api/isaacsim.sensors.rtx/` | RTX sensor examples |
