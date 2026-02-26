# Sensor Types Reference

Complete catalog of Isaac Sim sensors with constructor parameters, output shapes, and configuration options.

## Camera Sensor

**Class:** `isaacsim.sensors.camera.Camera` (extends `BaseSensor`)
**File:** `source/extensions/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py`

### Constructor

```python
Camera(
    prim_path: str,              # USD prim path
    name: str = "camera",
    frequency: int = None,       # Capture frequency (Hz)
    dt: float = None,            # Capture period (1/frequency)
    resolution: tuple = None,    # (width, height) in pixels
    position: np.ndarray = None, # World position [x, y, z]
    orientation: np.ndarray = None, # Quaternion [w, x, y, z]
    translation: np.ndarray = None,
    render_product_path: str = None,
    annotator_device: str = None,  # None, "cpu", "cuda"
)
```

### Annotator Output Formats

| Annotator | Key in Frame | Shape | Dtype | Notes |
|-----------|-------------|-------|-------|-------|
| `rgb` | `"rgba"` | (H, W, 4) | uint8 | RGBA channels |
| `normals` | `"normals"` | (H, W, 4) | float32 | Surface normals + alpha |
| `motion_vectors` | `"motion_vectors"` | (H, W, 4) | float32 | Per-pixel motion |
| `occlusion` | `"occlusion"` | (H, W) | float32 | Ambient occlusion |
| `distance_to_image_plane` | `"distance_to_image_plane"` | (H, W) | float32 | Z-depth |
| `distance_to_camera` | `"distance_to_camera"` | (H, W) | float32 | Euclidean distance |
| `bounding_box_2d_tight` | `"bounding_box_2d_tight"` | (N,) | structured | Tight 2D bbox per object |
| `bounding_box_2d_loose` | `"bounding_box_2d_loose"` | (N,) | structured | Loose 2D bbox per object |
| `bounding_box_3d` | `"bounding_box_3d"` | (N,) | structured | 3D bbox per object |
| `semantic_segmentation` | `"semantic_segmentation"` | (H, W) | uint32 | Semantic class per pixel |
| `instance_id_segmentation` | `"instance_id_segmentation"` | (H, W) | uint32 | Instance ID per pixel |
| `instance_segmentation` | `"instance_segmentation"` | (H, W) | uint32 | Instance segmentation |
| `pointcloud` | `"pointcloud"` | (N, 3) | float32 | 3D point cloud |

### Key Methods

- `get_current_frame(clone=False)` → dict of annotator data
- `get_rgba(device=None)` → RGBA image
- `get_render_product_path()` → str
- `set_frequency(frequency)` / `set_dt(dt)`
- `pause()` / `resume()` - Pause/resume capture
- `attach_annotator(name, **kwargs)` / `detach_annotator(name)`

## Contact Sensor

**Class:** `isaacsim.sensors.physics.ContactSensor` (extends `BaseSensor`)
**File:** `source/extensions/isaacsim.sensors.physics/python/impl/contact_sensor.py`

### Constructor

```python
ContactSensor(
    prim_path: str,              # Must be under a prim with CollisionAPI
    name: str = "contact_sensor",
    frequency: int = None,
    dt: float = None,
    min_threshold: float = 0,    # Min force to register contact (N)
    max_threshold: float = 1e7,  # Max force threshold (N)
    radius: float = -1,          # Contact detection radius (-1 = use body)
    position: np.ndarray = None,
    translation: np.ndarray = None,
    orientation: np.ndarray = None,
)
```

### Key Methods

- `get_current_frame()` → dict with contact data
- `add_raw_contact_data_to_frame()` - Enable detailed contact reporting
- `set_dt(dt)` / `set_min_threshold(t)` / `set_max_threshold(t)` / `set_radius(r)`

### Output

Frame dict contains:
- Contact force vectors
- Contact torque vectors
- Contact point positions
- Contact body information

## IMU Sensor

**Class:** `isaacsim.sensors.physics.IMUSensor` (extends `BaseSensor`)
**File:** `source/extensions/isaacsim.sensors.physics/python/impl/imu_sensor.py`

### Constructor

```python
IMUSensor(
    prim_path: str,
    name: str = "imu_sensor",
    frequency: int = None,
    dt: float = None,
    position: np.ndarray = None,
    translation: np.ndarray = None,
    orientation: np.ndarray = None,
    linear_acceleration_filter_size: int = 1,
    angular_velocity_filter_size: int = 1,
    orientation_filter_size: int = 1,
)
```

### Output

```python
frame = imu.get_current_frame(read_gravity=True)
# frame["lin_acc"]     → np.ndarray (3,) - Linear acceleration (m/s^2)
# frame["ang_vel"]     → np.ndarray (3,) - Angular velocity (rad/s)
# frame["orientation"] → np.ndarray (4,) - Quaternion [w, x, y, z]
```

Setting `read_gravity=True` includes gravitational acceleration in the reading.

## Effort Sensor

**Class:** `isaacsim.sensors.physics.EffortSensor` (extends `SingleArticulation`)
**File:** `source/extensions/isaacsim.sensors.physics/python/impl/effort_sensor.py`

### Constructor

```python
EffortSensor(
    prim_path: str,              # Path to the joint prim
    sensor_period: float = 0,    # 0 = every physics step
    use_latest_data: bool = False,
    enabled: bool = True,
)
```

### Output

```python
reading = effort.get_sensor_reading()
# reading.is_valid → bool
# reading.time     → float (simulation time)
# reading.value    → float (joint effort in Nm)
```

## Rotating PhysX Lidar

**Class:** `isaacsim.sensors.physx.RotatingLidarPhysX` (extends `BaseSensor`)
**File:** `source/extensions/isaacsim.sensors.physx/python/impl/rotating_lidar_physX.py`

### Constructor

```python
RotatingLidarPhysX(
    prim_path: str,
    name: str = "lidar_physx",
    rotation_frequency: float = None,  # Hz (or use rotation_dt)
    rotation_dt: float = None,
    position: np.ndarray = None,
    translation: np.ndarray = None,
    orientation: np.ndarray = None,
    fov: tuple = None,          # (horizontal, vertical) in degrees
    resolution: tuple = None,   # (horizontal, vertical) in degrees
    valid_range: tuple = None,  # (min, max) in meters
)
```

### Key Methods

- `add_depth_data_to_frame()` - Enable depth readings
- `add_point_cloud_data_to_frame()` - Enable 3D point cloud
- `enable_visualization()` - Show lidar rays in viewport
- `set_rotation_frequency(freq)` / `set_fov(fov)` / `set_resolution(res)` / `set_valid_range(range)`

## RTX Lidar

**Class:** `isaacsim.sensors.rtx.LidarRtx` (extends `BaseSensor`)
**File:** `source/extensions/isaacsim.sensors.rtx/python/impl/lidar_rtx.py`

### Constructor

```python
LidarRtx(
    prim_path: str,
    name: str = "lidar_rtx",
    position: np.ndarray = None,
    translation: np.ndarray = None,
    orientation: np.ndarray = None,
    config_file_name: str = None,  # Predefined sensor config
)
```

### Available Sensor Configs

| Vendor | Models |
|--------|--------|
| HESAI | PandarXT_32, Pandar128, PandarQT64, AT128 |
| NVIDIA | GenericLidar |
| Ouster | OS0_128, OS1_32/64/128, OS2_32/64/128 |
| SICK | multiScan136 |
| SLAMTEC | RPLiDAR_S2E |
| Velodyne | VLP_16, VLS_128 |
| ZVISION | ML_30S |

### RTX Annotators

- `IsaacExtractRTXSensorPointCloudNoAccumulator` - Single-frame point cloud
- Custom annotators via `lidar.attach_annotator(name)`

### Output

Frame dict contains point cloud data with:
- XYZ positions
- Intensity values
- Range measurements

## RTX Radar

Created via commands rather than direct class instantiation:

```python
import omni.kit.commands

# Create radar sensor
result, prim = omni.kit.commands.execute(
    "IsaacSensorCreateRtxRadar",
    path="/World/Robot/radar",
    parent=None,
)
```

Output includes:
- Detection positions
- Velocity measurements
- Material-dependent radar cross-section

## Standalone Examples

| Sensor | Example | Path |
|--------|---------|------|
| Camera | Basic | `source/standalone_examples/api/isaacsim.sensors.camera/camera.py` |
| Camera | GPU annotators | `source/standalone_examples/api/isaacsim.sensors.camera/camera_annotator_device.py` |
| Camera | Pinhole lens | `source/standalone_examples/api/isaacsim.sensors.camera/camera_opencv_pinhole.py` |
| Camera | Fisheye lens | `source/standalone_examples/api/isaacsim.sensors.camera/camera_opencv_fisheye.py` |
| Camera | Stereo depth | `source/standalone_examples/api/isaacsim.sensors.camera/camera_stereoscopic_depth.py` |
| Camera | Pre-ISP | `source/standalone_examples/api/isaacsim.sensors.camera/camera_pre_isp_pipeline.py` |
| Contact | Ant robot | `source/standalone_examples/api/isaacsim.sensors.physics/contact_sensor.py` |
| IMU | Wheeled robot | `source/standalone_examples/api/isaacsim.sensors.physics/imu_sensor.py` |
| Effort | Joint torque | `source/standalone_examples/api/isaacsim.sensors.physics/effort_sensor.py` |
| PhysX Lidar | Rotating | `source/standalone_examples/api/isaacsim.sensors.physx/rotating_lidar_physX.py` |
| RTX Lidar | Point cloud | `source/standalone_examples/api/isaacsim.sensors.rtx/rotating_lidar_rtx.py` |
