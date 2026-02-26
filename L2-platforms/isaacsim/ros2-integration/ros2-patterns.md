# ROS 2 Patterns Reference

Common OmniGraph action graph configurations for Isaac Sim ROS 2 integration.

## OmniGraph Node Catalog

### Execution and Context Nodes

| Node Type ID | Purpose |
|-------------|---------|
| `omni.graph.action.OnPlaybackTick` | Triggers on each simulation tick |
| `isaacsim.ros2.bridge.ROS2Context` | Creates ROS 2 context handle |
| `isaacsim.ros2.bridge.ROS2QoSProfile` | Quality of Service configuration |

### Publisher Node Types

| Node Type ID | ROS 2 Message Type |
|-------------|-------------------|
| `isaacsim.ros2.bridge.ROS2PublishClock` | rosgraph_msgs/Clock |
| `isaacsim.ros2.bridge.ROS2PublishTransformTree` | tf2_msgs/TFMessage |
| `isaacsim.ros2.bridge.ROS2PublishRawTransformTree` | tf2_msgs/TFMessage |
| `isaacsim.ros2.bridge.ROS2PublishImage` | sensor_msgs/Image |
| `isaacsim.ros2.bridge.ROS2PublishCameraInfo` | sensor_msgs/CameraInfo |
| `isaacsim.ros2.bridge.ROS2PublishPointCloud` | sensor_msgs/PointCloud2 |
| `isaacsim.ros2.bridge.ROS2PublishLaserScan` | sensor_msgs/LaserScan |
| `isaacsim.ros2.bridge.ROS2PublishJointState` | sensor_msgs/JointState |
| `isaacsim.ros2.bridge.ROS2PublishOdometry` | nav_msgs/Odometry |
| `isaacsim.ros2.bridge.ROS2PublishImu` | sensor_msgs/Imu |
| `isaacsim.ros2.bridge.ROS2PublishBbox2D` | vision_msgs/Detection2DArray |
| `isaacsim.ros2.bridge.ROS2PublishBbox3D` | vision_msgs/Detection3DArray |
| `isaacsim.ros2.bridge.ROS2PublishSemanticLabels` | Custom |
| `isaacsim.ros2.bridge.ROS2PublishAckermann` | ackermann_msgs/AckermannDriveStamped |
| `isaacsim.ros2.bridge.ROS2Publisher` | Generic (any message type) |

### Helper Nodes (Python-based, high-level)

| Node Type ID | Purpose |
|-------------|---------|
| `isaacsim.ros2.bridge.ROS2CameraHelper` | Camera publishing (handles render product) |
| `isaacsim.ros2.bridge.ROS2CameraInfoHelper` | Camera info with distortion |
| `isaacsim.ros2.bridge.ROS2RtxLidarHelper` | RTX lidar point cloud |

### Subscriber Node Types

| Node Type ID | ROS 2 Message Type |
|-------------|-------------------|
| `isaacsim.ros2.bridge.ROS2SubscribeTwist` | geometry_msgs/Twist |
| `isaacsim.ros2.bridge.ROS2SubscribeJointState` | sensor_msgs/JointState |
| `isaacsim.ros2.bridge.ROS2SubscribeAckermann` | ackermann_msgs/AckermannDriveStamped |
| `isaacsim.ros2.bridge.ROS2SubscribeClock` | rosgraph_msgs/Clock |
| `isaacsim.ros2.bridge.ROS2SubscribeTransformTree` | tf2_msgs/TFMessage |
| `isaacsim.ros2.bridge.ROS2Subscriber` | Generic (any message type) |

### Service Node Types

| Node Type ID | Purpose |
|-------------|---------|
| `isaacsim.ros2.bridge.ROS2ServiceClient` | Call ROS 2 services |
| `isaacsim.ros2.bridge.ROS2ServiceServerRequest` | Handle incoming requests |
| `isaacsim.ros2.bridge.ROS2ServiceServerResponse` | Send responses |
| `isaacsim.ros2.bridge.ROS2ServicePrim` | Prim-based service |

## Common Action Graph Configurations

### Camera Pipeline (RGB + Depth + Camera Info)

```python
import omni.graph.core as og

keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": "/CameraGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("RGBCamera", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("DepthCamera", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "RGBCamera.inputs:execIn"),
            ("OnTick.outputs:tick", "DepthCamera.inputs:execIn"),
            ("OnTick.outputs:tick", "Clock.inputs:execIn"),
            ("Context.outputs:context", "RGBCamera.inputs:context"),
            ("Context.outputs:context", "DepthCamera.inputs:context"),
            ("Context.outputs:context", "Clock.inputs:context"),
        ],
        keys.SET_VALUES: [
            ("RGBCamera.inputs:cameraPrim", "/World/Camera"),
            ("RGBCamera.inputs:topicName", "camera/image_raw"),
            ("RGBCamera.inputs:type", "rgb"),
            ("RGBCamera.inputs:frameId", "camera_link"),
            ("DepthCamera.inputs:cameraPrim", "/World/Camera"),
            ("DepthCamera.inputs:topicName", "camera/depth"),
            ("DepthCamera.inputs:type", "depth"),
            ("DepthCamera.inputs:frameId", "camera_link"),
        ],
    },
)
```

### Navigation Stack (Lidar + Odometry + TF + cmd_vel)

```python
keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": "/NavGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("LidarHelper", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ("Odometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ("TF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("TwistSub", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "LidarHelper.inputs:execIn"),
            ("OnTick.outputs:tick", "Odometry.inputs:execIn"),
            ("OnTick.outputs:tick", "TF.inputs:execIn"),
            ("OnTick.outputs:tick", "Clock.inputs:execIn"),
            ("OnTick.outputs:tick", "TwistSub.inputs:execIn"),
            ("Context.outputs:context", "LidarHelper.inputs:context"),
            ("Context.outputs:context", "Odometry.inputs:context"),
            ("Context.outputs:context", "TF.inputs:context"),
            ("Context.outputs:context", "Clock.inputs:context"),
            ("Context.outputs:context", "TwistSub.inputs:context"),
        ],
        keys.SET_VALUES: [
            ("TwistSub.inputs:topicName", "cmd_vel"),
        ],
    },
)
```

### Joint State Publishing

```python
keys = og.Controller.Keys
og.Controller.edit(
    {"graph_path": "/JointGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("JointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "JointState.inputs:execIn"),
            ("Context.outputs:context", "JointState.inputs:context"),
        ],
        keys.SET_VALUES: [
            ("JointState.inputs:targetPrim", "/World/Robot"),
            ("JointState.inputs:topicName", "joint_states"),
        ],
    },
)
```

## Namespace Configuration

For multi-robot setups, set the `nodeNamespace` on the ROS2Context:

```python
og.Controller.set(
    og.Controller.attribute("inputs:nodeNamespace", context_node),
    "/my_robot"
)
```

All topics published through this context will be prefixed: `/my_robot/camera/image`, etc.

## QoS Configuration

```python
# Create QoS node
("QoS", "isaacsim.ros2.bridge.ROS2QoSProfile")

# Configure
keys.SET_VALUES: [
    ("QoS.inputs:reliability", "reliable"),     # or "best_effort"
    ("QoS.inputs:durability", "volatile"),       # or "transient_local"
    ("QoS.inputs:history", "keep_last"),         # or "keep_all"
    ("QoS.inputs:depth", 10),                    # Queue depth
]

# Connect to publisher
keys.CONNECT: [
    ("QoS.outputs:qosProfile", "Publisher.inputs:qosProfile"),
]
```

## Standalone Examples

| Example | Path | Description |
|---------|------|-------------|
| camera_manual.py | `source/standalone_examples/api/isaacsim.ros2.bridge/camera_manual.py` | Manual camera setup |
| camera_periodic.py | Same directory | Periodic publishing |
| subscriber.py | Same directory | ROS 2 message receiving |
| clock.py | Same directory | Clock synchronization |
| rtx_lidar.py | Same directory | RTX lidar setup |
| carter_stereo.py | Same directory | Stereo camera |
| carter_multiple_robot_navigation.py | Same directory | Multi-robot nav |
| moveit.py | Same directory | MoveIt integration |
