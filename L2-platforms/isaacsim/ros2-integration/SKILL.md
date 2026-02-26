---
name: isaacsim-ros2-integration
description: >
  Integrates Isaac Sim with ROS 2 via OmniGraph-based bridge — topic pub/sub, camera/lidar/IMU streaming, clock sync, TF broadcasting.
layer: L2
domain: [robotics]
source-project: IsaacSim
depends-on: [isaacsim-simulation-core, isaacsim-sensor-development]
tags: [ros2, omnigraph, bridge, topics]
---

# Isaac Sim ROS 2 Integration

The ROS 2 bridge (`isaacsim.ros2.bridge`) connects Isaac Sim to ROS 2 through OmniGraph action graphs. Communication happens via OmniGraph nodes, not direct Python API calls.

## Architecture

```
Isaac Sim                          ROS 2
┌──────────────────────┐         ┌─────────────────┐
│ OmniGraph Action Graph │ ──────→│ ROS 2 Topics   │
│ ┌──────────────────┐ │         │ /camera/image   │
│ │OnPlaybackTick    │ │         │ /lidar/points   │
│ │  → ROS2Camera    │ │         │ /clock          │
│ │  → ROS2Clock     │ │         │ /tf             │
│ │  → ROS2TF        │ │         │ /joint_states   │
│ └──────────────────┘ │         └─────────────────┘
│                      │
│ ┌──────────────────┐ │ ←──────┐ /cmd_vel        │
│ │SubscribeTwist    │ │         │ /joint_commands  │
│ │  → DiffController│ │         └─────────────────┘
│ └──────────────────┘ │
└──────────────────────┘
```

## Supported ROS Distributions

- ROS 2 Humble (Ubuntu 22.04)
- ROS 2 Jazzy (Ubuntu 24.04)

Configuration: `exts."isaacsim.ros2.bridge".ros_distro = "system_default"`

## Quick Start: Camera Publishing

Create an OmniGraph action graph that publishes camera images to ROS 2:

```python
import omni.graph.core as og

# Create action graph
keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("CameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "CameraHelper.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "Clock.inputs:execIn"),
            ("ROS2Context.outputs:context", "CameraHelper.inputs:context"),
            ("ROS2Context.outputs:context", "Clock.inputs:context"),
        ],
        keys.SET_VALUES: [
            ("CameraHelper.inputs:cameraPrim", "/World/Camera"),
            ("CameraHelper.inputs:topicName", "camera/image"),
            ("CameraHelper.inputs:type", "rgb"),
            ("CameraHelper.inputs:frameId", "camera_frame"),
        ],
    },
)
```

## Publisher Nodes

30 OmniGraph nodes for publishing sensor and state data to ROS 2 topics:

### Camera Publishing

| Node | Topic Type | Description |
|------|-----------|-------------|
| `ROS2CameraHelper` | sensor_msgs/Image | RGB, depth, semantic images |
| `ROS2CameraInfoHelper` | sensor_msgs/CameraInfo | Camera intrinsics |
| `ROS2PublishImage` | sensor_msgs/Image | Raw image publisher |

Camera helper `type` options: `"rgb"`, `"depth"`, `"depth_pcl"`, `"semantic_segmentation"`, `"instance_segmentation"`, `"bbox_2d"`, `"bbox_3d"`.

### Lidar Publishing

| Node | Topic Type | Description |
|------|-----------|-------------|
| `ROS2RtxLidarHelper` | sensor_msgs/PointCloud2 | RTX lidar point clouds |
| `ROS2PublishPointCloud` | sensor_msgs/PointCloud2 | Generic point cloud |
| `ROS2PublishLaserScan` | sensor_msgs/LaserScan | 2D laser scan |

### Other Publishers

| Node | Topic Type | Description |
|------|-----------|-------------|
| `ROS2PublishClock` | rosgraph_msgs/Clock | Simulation clock |
| `ROS2PublishTransformTree` | tf2_msgs/TFMessage | TF tree |
| `ROS2PublishRawTransformTree` | tf2_msgs/TFMessage | Raw TF |
| `ROS2PublishJointState` | sensor_msgs/JointState | Joint states |
| `ROS2PublishOdometry` | nav_msgs/Odometry | Odometry |
| `ROS2PublishImu` | sensor_msgs/Imu | IMU data |
| `ROS2PublishSemanticLabels` | Custom | Semantic labels |
| `ROS2PublishBbox2D` | vision_msgs/Detection2DArray | 2D bounding boxes |
| `ROS2PublishBbox3D` | vision_msgs/Detection3DArray | 3D bounding boxes |
| `ROS2PublishAckermann` | ackermann_msgs/AckermannDriveStamped | Ackermann drive |

## Subscriber Nodes

| Node | Topic Type | Description |
|------|-----------|-------------|
| `ROS2SubscribeTwist` | geometry_msgs/Twist | Velocity commands |
| `ROS2SubscribeJointState` | sensor_msgs/JointState | Joint commands |
| `ROS2SubscribeAckermann` | ackermann_msgs/AckermannDriveStamped | Ackermann drive |
| `ROS2SubscribeClock` | rosgraph_msgs/Clock | External clock |
| `ROS2SubscribeTransformTree` | tf2_msgs/TFMessage | TF subscription |

## Service Nodes

| Node | Description |
|------|-------------|
| `ROS2ServiceClient` | ROS 2 service client |
| `ROS2ServiceServerRequest` | Handle service requests |
| `ROS2ServiceServerResponse` | Send service responses |
| `ROS2ServicePrim` | Prim-based service interface |

## Clock Synchronization

Always publish simulation clock to keep ROS 2 nodes synchronized:

```python
# In action graph:
("Clock", "isaacsim.ros2.bridge.ROS2PublishClock")
# Connect:
("OnPlaybackTick.outputs:tick", "Clock.inputs:execIn")
("ROS2Context.outputs:context", "Clock.inputs:context")
```

On the ROS 2 side, use `--ros-args -p use_sim_time:=true` to consume simulation clock.

## TF Broadcasting

Publish the USD transform tree as ROS 2 TF:

```python
("TFPublisher", "isaacsim.ros2.bridge.ROS2PublishTransformTree")
# Set target prims to broadcast their transforms
```

## Multi-Robot Setup

Use ROS 2 namespaces for multi-robot configurations:

```python
# Robot 1 action graph
og.Controller.set(
    og.Controller.attribute("inputs:nodeNamespace", robot1_context),
    "/robot1"
)

# Robot 2 action graph
og.Controller.set(
    og.Controller.attribute("inputs:nodeNamespace", robot2_context),
    "/robot2"
)
```

Topics become: `/robot1/camera/image`, `/robot2/camera/image`, etc.

## QoS Configuration

Use the `ROS2QoSProfile` node for custom Quality of Service:

```python
("QoS", "isaacsim.ros2.bridge.ROS2QoSProfile")
# Set reliability, durability, history depth, etc.
```

## NITROS Bridge

High-performance zero-copy GPU publishing mode:

```toml
# Enable in settings
exts."isaacsim.ros2.bridge".enable_nitros_bridge = true
```

NITROS enables direct GPU-to-GPU data transfer for image and point cloud data, bypassing CPU copies. Requires compatible NVIDIA hardware and drivers.

## Settings

```toml
exts."isaacsim.ros2.bridge".ros_distro = "system_default"
exts."isaacsim.ros2.bridge".publish_without_verification = false
exts."isaacsim.ros2.bridge".publish_multithreading_disabled = false
exts."isaacsim.ros2.bridge".enable_nitros_bridge = false
```

## Reference Files

- [ros2-patterns.md](ros2-patterns.md) - Common action graph configurations and node catalog

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/extensions/isaacsim.ros2.bridge/` | Main bridge extension |
| `source/extensions/isaacsim.ros2.bridge/nodes/` | C++ OGN node definitions |
| `source/extensions/isaacsim.ros2.bridge/python/nodes/` | Python OGN nodes |
| `source/extensions/isaacsim.ros2.sim_control/` | Simulation control via ROS 2 |
| `source/extensions/isaacsim.ros2.tf_viewer/` | TF visualization |
| `source/standalone_examples/api/isaacsim.ros2.bridge/` | Standalone examples |
