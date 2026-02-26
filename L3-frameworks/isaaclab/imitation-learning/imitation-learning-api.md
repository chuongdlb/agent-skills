# Imitation Learning API Reference

Full attribute tables for all imitation learning configuration classes.

## MimicEnvCfg

**File:** `source/isaaclab/isaaclab/envs/mimic_env_cfg.py`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `datagen_config` | DataGenConfig | DataGenConfig() | Data generation config |
| `subtask_configs` | dict[str, list[SubTaskConfig]] | {} | Per-EEF subtask configs |
| `task_constraint_configs` | list[SubTaskConstraintConfig] | [] | Multi-EEF constraints |
| `mimic_recorder_config` | RecorderManagerBaseCfg \| None | None | Recorder config |

## DataGenConfig (Full)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "demo" | Generation process name |
| `generation_guarantee` | bool | True | Retry until target success count |
| `generation_keep_failed` | bool | False | Keep failed generation trials |
| `max_num_failures` | int | 50 | Max consecutive failures |
| `seed` | int | 1 | Random seed |
| `source_dataset_path` | str | None | Path to source HDF5 |
| `generation_path` | str | None | Path for generated HDF5 |
| `generation_num_trials` | int | 10 | Number of trials to generate |
| `task_name` | str | None | Task identifier |
| `generation_select_src_per_subtask` | bool | False | Select source per subtask |
| `generation_select_src_per_arm` | bool | False | Select source per arm |
| `generation_transform_first_robot_pose` | bool | False | Transform initial robot pose |
| `generation_interpolate_from_last_target_pose` | bool | True | Interpolate from last target |
| `use_skillgen` | bool | False | Use skillgen annotation |
| `use_navigation_controller` | bool | False | Use navigation for locomotion |

## SubTaskConfig (Full)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `object_ref` | str | None | Object name in scene |
| `subtask_term_signal` | str | None | Termination signal key |
| `selection_strategy` | str | "random" | Demo selection strategy |
| `selection_strategy_kwargs` | dict | {} | Extra strategy params |
| `first_subtask_start_offset_range` | tuple[int, int] | (0, 0) | First subtask start offset |
| `subtask_start_offset_range` | tuple[int, int] | (0, 0) | Start offset (skillgen) |
| `subtask_term_offset_range` | tuple[int, int] | (0, 0) | Termination offset range |
| `action_noise` | float | 0.03 | Action noise amplitude |
| `num_interpolation_steps` | int | 5 | Interpolation step count |
| `num_fixed_steps` | int | 0 | Fixed steps per subtask |
| `apply_noise_during_interpolation` | bool | False | Noise during interpolation |
| `description` | str | "" | Subtask description |
| `next_subtask_description` | str | "" | Next subtask instructions |

## SubTaskConstraintType

```python
class SubTaskConstraintType(IntEnum):
    SEQUENTIAL = 0      # One arm before another
    COORDINATION = 1    # Coordinated multi-arm motion
```

## SubTaskConstraintCoordinationScheme

```python
class SubTaskConstraintCoordinationScheme(IntEnum):
    REPLAY = 0       # Replay source trajectory
    TRANSFORM = 1    # Full 6D transformation
    TRANSLATE = 2    # Translation only
```

## SubTaskConstraintConfig (Full)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `eef_subtask_constraint_tuple` | list[tuple] | — | Pairs of (eef_name, subtask_idx) |
| `constraint_type` | SubTaskConstraintType | None | SEQUENTIAL or COORDINATION |
| `sequential_min_time_diff` | int | -1 | Min time difference (sequential) |
| `coordination_scheme` | SubTaskConstraintCoordinationScheme | REPLAY | Coordination method |
| `coordination_scheme_pos_noise_scale` | float | 0.0 | Position noise scale |
| `coordination_scheme_rot_noise_scale` | float | 0.0 | Rotation noise scale |
| `coordination_synchronize_start` | bool | False | Synchronize subtask starts |

## RecorderManagerBaseCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_file_handler_class_type` | type | HDF5DatasetFileHandler | Handler class |
| `dataset_export_dir_path` | str | "/tmp/isaaclab/logs" | Export directory |
| `dataset_filename` | str | "dataset" | Filename |
| `dataset_export_mode` | DatasetExportMode | EXPORT_ALL | Export strategy |
| `export_in_record_pre_reset` | bool | True | Export on reset |
| `export_in_close` | bool | False | Export on close |

## RecorderTermCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_type` | type[RecorderTerm] | MISSING | Recorder term class |

## RecorderTerm Abstract Methods

| Method | Called When | Description |
|--------|------------|-------------|
| `record_pre_reset(env_ids)` | Before env reset | Capture pre-reset data |
| `record_post_reset(env_ids)` | After env reset | Capture post-reset data |
| `record_pre_step()` | Before env step | Capture pre-step data |
| `record_post_step()` | After env step | Capture post-step data |
| `record_post_physics_decimation_step()` | After physics step | High-frequency data |
| `close(file_path)` | On env close | Finalize recorder |

## Built-in RecorderTerm Classes

| Class | Records | Lifecycle Stage |
|-------|---------|-----------------|
| `InitialStateRecorderCfg` | Initial state | post_reset |
| `PreStepActionsRecorderCfg` | Raw actions | pre_step |
| `PreStepFlatPolicyObservationsRecorderCfg` | Policy observations | pre_step |
| `PostStepStatesRecorderCfg` | Post-step states | post_step |
| `PostStepProcessedActionsRecorderCfg` | Processed actions | post_step |

## RecorderManager Methods

| Method | Description |
|--------|-------------|
| `reset(env_ids)` | Reset episodes |
| `get_episode(env_id)` | Get EpisodeData |
| `add_to_episodes(key, value, env_ids)` | Add data |
| `set_success_to_episodes(env_ids, values)` | Mark success |
| `export_episodes(env_ids, demo_ids)` | Export to file |
| `close()` | Finalize and close |

## DataGenerator API

### Constructor

```python
DataGenerator(env, src_demo_datagen_info_pool, dataset_path, demo_keys)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `randomize_subtask_boundaries()` | Apply random offsets |
| `select_source_demo(eef_name, ...)` | Select source segment |
| `generate_eef_subtask_trajectory(env_id, eef_name, subtask_ind, ...)` | Generate transformed trajectory |
| `merge_eef_subtask_trajectory(env_id, eef_name, ...)` | Merge with interpolation |
| `async generate(env_id, success_term, ...)` | Full generation pipeline |

## DatagenInfo

Stores per-timestep data for source demo analysis:

| Attribute | Type | Description |
|-----------|------|-------------|
| `eef_pose` | Tensor[..., 4, 4] | End-effector poses |
| `object_poses` | dict[str, Tensor] | Object poses |
| `subtask_term_signals` | dict[str, Tensor] | Termination signals |
| `target_eef_pose` | Tensor[..., 4, 4] | Target EEF poses |
| `gripper_action` | Tensor[..., D] | Gripper actions |

## ManagerBasedRLMimicEnv Abstract Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_robot_eef_pose(eef_name, env_ids)` | Tensor[N, 4, 4] | Current EEF pose |
| `target_eef_pose_to_action(...)` | Tensor | Convert target to action |
| `action_to_target_eef_pose(action)` | dict[str, Tensor] | Action to target poses |
| `actions_to_gripper_actions(actions)` | dict[str, Tensor] | Extract gripper actions |
| `get_object_poses(env_ids)` | dict | Object pose dictionary |
| `get_subtask_term_signals(env_ids)` | dict[str, Tensor] | Termination signals |
| `get_subtask_start_signals(env_ids)` | dict[str, Tensor] | Start signals (optional) |

## CuroboPlannerCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `robot_config_file` | str | None | CuRobo robot config |
| `robot_name` | str | "" | Robot identifier |
| `ee_link_name` | str | None | End-effector link |
| `world_config_file` | str | "collision_table.yml" | World config |
| `num_trajopt_seeds` | int | 12 | Trajectory seeds |
| `num_graph_seeds` | int | 12 | Graph search seeds |
| `interpolation_dt` | float | 0.05 | Waypoint spacing |
| `trajopt_tsteps` | int | 32 | Optimization steps |
| `collision_activation_distance` | float | 0.0 | Activation distance |
| `approach_distance` | float | 0.05 | Approach distance |
| `retreat_distance` | float | 0.05 | Retreat distance |
| `enable_graph` | bool | True | Enable graph planning |
| `max_planning_attempts` | int | 15 | Max planning attempts |
| `enable_finetune_trajopt` | bool | True | Fine-tune trajectory |
