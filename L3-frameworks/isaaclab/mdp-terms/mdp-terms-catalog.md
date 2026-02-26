# MDP Terms Catalog

Exhaustive reference of all built-in MDP terms shipped with IsaacLab.

---

## Observations (`isaaclab.envs.mdp.observations`)

### Root State

| Function | Key Params | Return Shape | Description |
|----------|-----------|--------------|-------------|
| `base_pos_z` | `asset_cfg` | (N, 1) | Root height in world frame |
| `base_lin_vel` | `asset_cfg` | (N, 3) | Root linear velocity in body frame |
| `base_ang_vel` | `asset_cfg` | (N, 3) | Root angular velocity in body frame |
| `projected_gravity` | `asset_cfg` | (N, 3) | Gravity direction in body frame |
| `root_pos_w` | `asset_cfg` | (N, 3) | Root position in env frame (minus env origin) |
| `root_quat_w` | `asset_cfg`, `make_quat_unique` | (N, 4) | Root orientation (w,x,y,z) in env frame |
| `root_lin_vel_w` | `asset_cfg` | (N, 3) | Root linear velocity in world frame |
| `root_ang_vel_w` | `asset_cfg` | (N, 3) | Root angular velocity in world frame |

All root state functions default to `asset_cfg=SceneEntityCfg("robot")`.

### Body State

| Function | Key Params | Return Shape | Description |
|----------|-----------|--------------|-------------|
| `body_pose_w` | `asset_cfg` (with `body_ids`) | (N, 7*num_bodies) | Body poses in env frame [x,y,z,qw,qx,qy,qz] flattened |
| `body_projected_gravity_b` | `asset_cfg` (with `body_ids`) | (N, 3*num_bodies) | Gravity projected onto body frames, flattened |

### Joint State

| Function | Key Params | Return Shape | Description |
|----------|-----------|--------------|-------------|
| `joint_pos` | `asset_cfg` | (N, num_joints) | Absolute joint positions (rad) |
| `joint_pos_rel` | `asset_cfg` | (N, num_joints) | Joint positions minus default positions |
| `joint_pos_limit_normalized` | `asset_cfg` | (N, num_joints) | Joint positions normalized to soft limits [-1, 1] |
| `joint_vel` | `asset_cfg` | (N, num_joints) | Joint velocities (rad/s) |
| `joint_vel_rel` | `asset_cfg` | (N, num_joints) | Joint velocities minus default velocities |
| `joint_effort` | `asset_cfg` | (N, num_joints) | Applied joint torques (N.m) |

All joint state functions default to `asset_cfg=SceneEntityCfg("robot")`. Use `joint_names` or `joint_ids` in the `SceneEntityCfg` to select specific joints.

### Sensor

| Function / Class | Key Params | Return Shape | Description |
|------------------|-----------|--------------|-------------|
| `height_scan` | `sensor_cfg`, `offset=0.5` | (N, num_rays) | Height scan from ray caster minus offset |
| `body_incoming_wrench` | `asset_cfg` (with `body_ids`) | (N, 6*num_bodies) | Incoming joint wrench (force+torque) in body frame |
| `imu_orientation` | `asset_cfg` (default `"imu"`) | (N, 4) | IMU orientation quaternion (w,x,y,z) in world frame |
| `imu_projected_gravity` | `asset_cfg` (default `"imu"`) | (N, 3) | Gravity projected onto IMU frame |
| `imu_ang_vel` | `asset_cfg` (default `"imu"`) | (N, 3) | IMU angular velocity in sensor frame (rad/s) |
| `imu_lin_acc` | `asset_cfg` (default `"imu"`) | (N, 3) | IMU linear acceleration in sensor frame (m/s^2) |
| `image` | `sensor_cfg`, `data_type="rgb"`, `normalize=True`, `convert_perspective_to_orthogonal=False` | (N, H, W, C) | Camera image (rgb, depth, distance_to_camera, normals) |
| `image_features` (class) | `sensor_cfg`, `data_type`, `model_name="resnet18"`, `model_device`, `model_zoo_cfg`, `inference_kwargs` | (N, feature_dim) | Extracted features from frozen encoder (ResNet, Theia) |

### Action

| Function | Key Params | Return Shape | Description |
|----------|-----------|--------------|-------------|
| `last_action` | `action_name=None` | (N, action_dim) | Previous action. If `action_name` given, returns that term's raw_actions |

### Command

| Function | Key Params | Return Shape | Description |
|----------|-----------|--------------|-------------|
| `generated_commands` | `command_name` | (N, cmd_dim) | Current command from the named command term |

### Time

| Function | Key Params | Return Shape | Description |
|----------|-----------|--------------|-------------|
| `current_time_s` | -- | (N, 1) | Current time in the episode (seconds) |
| `remaining_time_s` | -- | (N, 1) | Max remaining time in episode (seconds) |

---

## Rewards (`isaaclab.envs.mdp.rewards`)

### General

| Function / Class | Key Params | Description |
|------------------|-----------|-------------|
| `is_alive` | -- | 1.0 for non-terminated envs, 0.0 for terminated |
| `is_terminated` | -- | 1.0 for terminated (non-timeout) envs |
| `is_terminated_term` (class) | `term_keys=".*"` | Sum of specific termination term signals (excluding timeouts) |

### Root Penalties

| Function | Key Params | Description |
|----------|-----------|-------------|
| `lin_vel_z_l2` | `asset_cfg` | L2 penalty on z-axis linear velocity |
| `ang_vel_xy_l2` | `asset_cfg` | L2 penalty on xy angular velocity |
| `flat_orientation_l2` | `asset_cfg` | L2 penalty on xy projected gravity (non-flat orientation) |
| `base_height_l2` | `target_height`, `asset_cfg`, `sensor_cfg=None` | L2 penalty on root height from target. Optional ray-cast sensor adjusts for terrain |
| `body_lin_acc_l2` | `asset_cfg` (with `body_ids`) | L2 penalty on body linear accelerations |

### Joint Penalties

| Function | Key Params | Description |
|----------|-----------|-------------|
| `joint_torques_l2` | `asset_cfg` | L2 penalty on applied joint torques |
| `joint_vel_l1` | `asset_cfg` | L1 penalty on joint velocities |
| `joint_vel_l2` | `asset_cfg` | L2 penalty on joint velocities |
| `joint_acc_l2` | `asset_cfg` | L2 penalty on joint accelerations |
| `joint_deviation_l1` | `asset_cfg` | L1 penalty on deviation from default joint positions |
| `joint_pos_limits` | `asset_cfg` | Penalty for exceeding soft joint position limits |
| `joint_vel_limits` | `soft_ratio`, `asset_cfg` | Penalty for exceeding soft velocity limits (clipped to max 1 rad/s per joint) |
| `applied_torque_limits` | `asset_cfg` | Penalty for torque clipping (explicit actuators only) |

### Action Penalties

| Function | Key Params | Description |
|----------|-----------|-------------|
| `action_rate_l2` | -- | L2 penalty on action rate of change |
| `action_l2` | -- | L2 penalty on action magnitude |

### Contact Sensor

| Function | Key Params | Description |
|----------|-----------|-------------|
| `undesired_contacts` | `threshold`, `sensor_cfg` (with `body_ids`) | Count of contacts above threshold |
| `desired_contacts` | `sensor_cfg` (with `body_ids`), `threshold=1.0` | 1.0 when none of the desired contacts are present |
| `contact_forces` | `threshold`, `sensor_cfg` (with `body_ids`) | Sum of contact force violations above threshold |

### Velocity Tracking

| Function | Key Params | Description |
|----------|-----------|-------------|
| `track_lin_vel_xy_exp` | `std`, `command_name`, `asset_cfg` | Exponential reward for tracking xy linear velocity command |
| `track_ang_vel_z_exp` | `std`, `command_name`, `asset_cfg` | Exponential reward for tracking yaw angular velocity command |

---

## Terminations (`isaaclab.envs.mdp.terminations`)

### MDP Terminations

| Function | Key Params | Description |
|----------|-----------|-------------|
| `time_out` | -- | True when `episode_length_buf >= max_episode_length`. Use with `time_out=True` in config |
| `command_resample` | `command_name`, `num_resamples=1` | True when command has been resampled `num_resamples` times |

### Root Terminations

| Function | Key Params | Description |
|----------|-----------|-------------|
| `bad_orientation` | `limit_angle`, `asset_cfg` | True when angle between projected gravity and z-axis exceeds `limit_angle` (rad) |
| `root_height_below_minimum` | `minimum_height`, `asset_cfg` | True when root z-position is below `minimum_height` (flat terrain only) |

### Joint Terminations

| Function | Key Params | Description |
|----------|-----------|-------------|
| `joint_pos_out_of_limit` | `asset_cfg` | True when any joint position exceeds soft limits |
| `joint_pos_out_of_manual_limit` | `bounds: (float, float)`, `asset_cfg` | True when any joint position exceeds provided bounds |
| `joint_vel_out_of_limit` | `asset_cfg` | True when any joint velocity exceeds soft velocity limits |
| `joint_vel_out_of_manual_limit` | `max_velocity`, `asset_cfg` | True when any joint velocity exceeds `max_velocity` |
| `joint_effort_out_of_limit` | `asset_cfg` | True when applied torque differs from computed torque (clipping occurred) |

### Contact Sensor

| Function | Key Params | Description |
|----------|-----------|-------------|
| `illegal_contact` | `threshold`, `sensor_cfg` (with `body_ids`) | True when any contact force exceeds `threshold` |

---

## Events (`isaaclab.envs.mdp.events`)

### Randomization (typically mode="startup")

| Function / Class | Key Params | Description |
|------------------|-----------|-------------|
| `randomize_rigid_body_scale` | `scale_range`, `asset_cfg` | Randomize rigid body scale |
| `randomize_rigid_body_material` (class) | `asset_cfg`, `static_friction_range`, `dynamic_friction_range`, `restitution_range`, `num_buckets` | Randomize physics material properties using bucketed sampling |
| `randomize_rigid_body_mass` (class) | `asset_cfg`, `mass_distribution_params`, `operation` ("add"/"scale"/"abs") | Randomize body masses |
| `randomize_rigid_body_com` | `asset_cfg`, `com_range: dict[str, (float,float)]` | Randomize center of mass offset (x, y, z ranges) |
| `randomize_rigid_body_collider_offsets` | `asset_cfg`, `position_range`, `rotation_range` | Randomize collider position/rotation offsets |
| `randomize_physics_scene_gravity` | `gravity_distribution_params`, `operation` | Randomize gravity vector |
| `randomize_actuator_gains` (class) | `asset_cfg`, `stiffness_distribution_params`, `damping_distribution_params`, `operation` | Randomize actuator stiffness/damping |
| `randomize_joint_parameters` (class) | `asset_cfg`, plus optional `friction_distribution_params`, `armature_distribution_params`, `lower_limit_distribution_params`, `upper_limit_distribution_params`, `operation` | Randomize joint friction, armature, limits |
| `randomize_fixed_tendon_parameters` (class) | `asset_cfg`, plus optional `stiffness_distribution_params`, `damping_distribution_params`, `limit_stiffness_distribution_params`, `offset_distribution_params`, `rest_length_distribution_params` | Randomize fixed tendon parameters |
| `randomize_visual_texture_material` (class) | `asset_cfg`, plus texture params | Randomize visual textures on assets |
| `randomize_visual_color` (class) | `asset_cfg`, plus color params | Randomize visual colors on assets |

### Reset (mode="reset")

| Function | Key Params | Description |
|----------|-----------|-------------|
| `reset_root_state_uniform` | `pose_range: dict`, `velocity_range: dict`, `asset_cfg` | Reset root state with uniform sampling around defaults |
| `reset_root_state_with_random_orientation` | `pose_range: dict`, `velocity_range: dict`, `asset_cfg` | Reset root state with random full 3D orientation |
| `reset_root_state_from_terrain` | `pose_range`, `velocity_range`, `asset_cfg` | Reset root state on terrain (uses TerrainImporter) |
| `reset_joints_by_scale` | `position_range: (float, float)`, `velocity_range: (float, float)`, `asset_cfg` | Reset joint positions by scaling defaults |
| `reset_joints_by_offset` | `position_uniform_range: (float, float)`, `velocity_uniform_range: (float, float)`, `asset_cfg` | Reset joint positions by adding offset to defaults |
| `reset_nodal_state_uniform` | `position_uniform_range`, `velocity_uniform_range`, `asset_cfg` | Reset deformable object nodal state |
| `reset_scene_to_default` | `reset_joint_targets: bool = False` | Reset entire scene to default state |

### Push / External Forces (mode="reset" or "interval")

| Function | Key Params | Description |
|----------|-----------|-------------|
| `push_by_setting_velocity` | `velocity_range: dict`, `asset_cfg` | Apply random push by setting root velocity |
| `apply_external_force_torque` | `asset_cfg`, `force_range: (float,float)`, `torque_range: (float,float)` | Apply random external forces/torques to bodies |

---

## Actions (`isaaclab.envs.mdp.actions`)

### Joint-Space Actions

| Config Class | Action Class | Description |
|-------------|-------------|-------------|
| `JointPositionActionCfg` | `JointPositionAction` | Absolute joint position targets (offset by default joint pos) |
| `RelativeJointPositionActionCfg` | `RelativeJointPositionAction` | Relative (delta) joint position targets |
| `JointVelocityActionCfg` | `JointVelocityAction` | Joint velocity targets |
| `JointEffortActionCfg` | `JointEffortAction` | Direct joint effort/torque commands |
| `BinaryJointPositionActionCfg` | `BinaryJointPositionAction` | Binary open/close joint position commands |
| `BinaryJointVelocityActionCfg` | `BinaryJointVelocityAction` | Binary open/close joint velocity commands |
| `AbsBinaryJointPositionActionCfg` | `AbsBinaryJointPositionAction` | Threshold-based binary gripper from continuous input |
| `JointPositionToLimitsActionCfg` | `JointPositionToLimitsAction` | Joint positions rescaled to joint limits [-1,1] |
| `EMAJointPositionToLimitsActionCfg` | `EMAJointPositionToLimitsAction` | EMA-smoothed joint positions rescaled to limits |

Key config fields for `JointActionCfg` (base of position/velocity/effort):
- `joint_names: list[str]` -- Joint names or regex
- `scale: float | dict[str, float]` -- Scale factor (default 1.0)
- `offset: float | dict[str, float]` -- Offset factor (default 0.0)
- `preserve_order: bool` -- Maintain joint name order (default False)

### Task-Space Actions

| Config Class | Action Class | Description |
|-------------|-------------|-------------|
| `DifferentialInverseKinematicsActionCfg` | `DifferentialInverseKinematicsAction` | Differential IK with configurable controller |
| `OperationalSpaceControllerActionCfg` | `OperationalSpaceControllerAction` | Operational space control (force/impedance) |
| `PinkInverseKinematicsActionCfg` | `PinkInverseKinematicsAction` | PINK IK framework integration |
| `RMPFlowActionCfg` | `RMPFlowAction` | RMPFlow motion policy integration |

Key config fields for `DifferentialInverseKinematicsActionCfg`:
- `joint_names: list[str]` -- Controlled joint names
- `body_name: str` -- End-effector body name
- `body_offset: OffsetCfg | None` -- EE frame offset from body
- `controller: DifferentialIKControllerCfg` -- IK controller config

### Other Actions

| Config Class | Action Class | Description |
|-------------|-------------|-------------|
| `NonHolonomicActionCfg` | `NonHolonomicAction` | Non-holonomic base control with dummy joints |
| `SurfaceGripperBinaryActionCfg` | `SurfaceGripperBinaryAction` | Binary surface gripper control |

---

## Commands (`isaaclab.envs.mdp.commands`)

| Config Class | Command Class | Description |
|-------------|--------------|-------------|
| `UniformVelocityCommandCfg` | `UniformVelocityCommand` | Uniform velocity commands (lin_vel_x, lin_vel_y, ang_vel_z, heading) |
| `NormalVelocityCommandCfg` | `NormalVelocityCommand` | Gaussian-sampled velocity commands |
| `UniformPoseCommandCfg` | `UniformPoseCommand` | Uniform SE(3) pose commands |
| `UniformPose2dCommandCfg` | `UniformPose2dCommand` | Uniform 2D pose commands (x, y, heading) |
| `TerrainBasedPose2dCommandCfg` | `TerrainBasedPose2dCommand` | 2D pose commands sampled on terrain |
| `NullCommandCfg` | `NullCommand` | No-op command (for tasks without goals) |

Key config fields for `UniformVelocityCommandCfg`:
- `asset_name: str` -- Robot entity name
- `resampling_time_range: (float, float)` -- Time range before resampling (seconds)
- `heading_command: bool` -- Whether to use heading-based commands
- `heading_control_stiffness: float` -- P-gain for heading control
- `rel_standing_envs: float` -- Fraction of envs with zero velocity
- `rel_heading_envs: float` -- Fraction of envs using heading commands
- `ranges: Ranges` -- Inner configclass with `lin_vel_x`, `lin_vel_y`, `ang_vel_z`, `heading` ranges

---

## Curriculum (`isaaclab.envs.mdp.curriculums`)

| Class | Key Params | Description |
|-------|-----------|-------------|
| `modify_reward_weight` | `term_name`, `weight`, `num_steps` | Step-wise schedule to change a reward term's weight after `num_steps` |
| `modify_env_param` | `address` (dotted path), `modify_fn`, `modify_params` | Modify any env attribute at runtime via dotted path accessor |
| `modify_term_cfg` | `address` (simplified), `modify_fn`, `modify_params` | Like `modify_env_param` but uses short address (e.g. "rewards.my_term.weight") |

The `modify_fn` signature: `def modify_fn(env, env_ids, old_value, **modify_params) -> new_value | NO_CHANGE`

---

## Recorders (`isaaclab.envs.mdp.recorders`)

### Built-in Recorder Terms

| Config Class | Recorder Class | Hook | Description |
|-------------|---------------|------|-------------|
| `InitialStateRecorderCfg` | `InitialStateRecorder` | `record_post_reset` | Records initial scene state after reset |
| `PostStepStatesRecorderCfg` | `PostStepStatesRecorder` | `record_post_step` | Records scene state after each step |
| `PreStepActionsRecorderCfg` | `PreStepActionsRecorder` | `record_pre_step` | Records raw actions before each step |
| `PreStepFlatPolicyObservationsRecorderCfg` | `PreStepFlatPolicyObservationsRecorder` | `record_pre_step` | Records policy obs before each step |
| `PostStepProcessedActionsRecorderCfg` | `PostStepProcessedActionsRecorder` | `record_post_step` | Records processed actions after each step |

### Pre-built Manager Config

| Config Class | Description |
|-------------|-------------|
| `ActionStateRecorderManagerCfg` | Combines all 5 recorder terms above into a single manager config |

### RecorderTerm Lifecycle Hooks

| Method | When Called | Args |
|--------|-----------|------|
| `record_pre_reset(env_ids)` | Beginning of `env.reset()`, before reset | `env_ids` |
| `record_post_reset(env_ids)` | End of `env.reset()` | `env_ids` |
| `record_pre_step()` | Beginning of `env.step()`, after action processing | -- |
| `record_post_step()` | End of `env.step()` | -- |
| `record_post_physics_decimation_step()` | After each physics substep in decimation loop | -- |
| `close(file_path)` | On environment close | `file_path` |
