---
name: isaaclab-imitation-learning
description: >
  Implements IsaacLab imitation learning pipeline — demo recording, IsaacLab Mimic, dataset annotation/generation, motion planners, robomimic training.
layer: L3
domain: [robotics, manipulation]
source-project: IsaacLab
depends-on: [isaaclab-environment-design, isaaclab-robot-and-asset-config]
tags: [imitation-learning, mimic, demonstrations, robomimic]
---

# IsaacLab Imitation Learning

IsaacLab provides a complete 4-stage imitation learning pipeline: record demonstrations, annotate subtasks, generate synthetic datasets, and train policies.

## Architecture

```
Stage 1: Record ─────→ HDF5 dataset (human demos)
  └── RecorderManager + teleop devices

Stage 2: Annotate ───→ HDF5 dataset (with subtask boundaries)
  └── annotate_demos.py (auto or manual)

Stage 3: Generate ───→ HDF5 dataset (synthetic demos, 10-100x)
  └── DataGenerator + object-centric transforms

Stage 4: Train ──────→ Policy checkpoint
  └── robomimic (BC, etc.)
```

## Stage 1: Record Demonstrations

### RecorderManager

Records environment data at various lifecycle stages. Configured via `RecorderManagerBaseCfg`:

```python
from isaaclab.managers import RecorderManagerBaseCfg, DatasetExportMode

recorder_cfg = RecorderManagerBaseCfg(
    dataset_export_dir_path="/tmp/demos",
    dataset_filename="my_demos",
    dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
    export_in_record_pre_reset=True,
)
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_file_handler_class_type` | type | HDF5DatasetFileHandler | File handler |
| `dataset_export_dir_path` | str | "/tmp/isaaclab/logs" | Export directory |
| `dataset_filename` | str | "dataset" | File name (no extension) |
| `dataset_export_mode` | DatasetExportMode | EXPORT_ALL | Export strategy |
| `export_in_record_pre_reset` | bool | True | Export on episode reset |
| `export_in_close` | bool | False | Export on env close |

**DatasetExportMode options:**
- `EXPORT_NONE` — No export
- `EXPORT_ALL` — Export all episodes
- `EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES` — Separate success/failure
- `EXPORT_SUCCEEDED_ONLY` — Only successful episodes

### RecorderTerm

Recorder terms capture specific data at lifecycle events:

```python
from isaaclab.managers import RecorderTermCfg

@configclass
class MyRecorderCfg:
    actions = RecorderTermCfg(class_type=PreStepActionsRecorder)
    observations = RecorderTermCfg(class_type=PreStepFlatPolicyObservationsRecorder)
    states = RecorderTermCfg(class_type=PostStepStatesRecorder)
```

### Recording Script

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Lift-Franka-IK-Abs-Gripper-State-v0 \
    --teleop_device keyboard \
    --dataset_file demos.hdf5 \
    --num_demos 10
```

## Stage 2: Annotate Demonstrations

Annotates recorded demos with subtask boundaries for data generation.

```bash
# Automatic annotation (using subtask termination signals)
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --input_file demos.hdf5 \
    --output_file annotated_demos.hdf5 \
    --auto
```

## Stage 3: Generate Synthetic Dataset

### MimicEnvCfg

Configures the data generation process:

```python
from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg, DataGenConfig, SubTaskConfig,
    SubTaskConstraintConfig, SubTaskConstraintType,
)

mimic_cfg = MimicEnvCfg(
    datagen_config=DataGenConfig(
        generation_num_trials=1000,
        seed=42,
    ),
    subtask_configs={
        "ee": [
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="grasp",
                selection_strategy="nearest_neighbor_object",
                action_noise=0.03,
                num_interpolation_steps=5,
            ),
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="lift",
                selection_strategy="random",
                action_noise=0.03,
            ),
        ],
    },
)
```

### DataGenConfig

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "demo" | Generation name |
| `generation_guarantee` | bool | True | Retry until success |
| `generation_keep_failed` | bool | False | Keep failed trials |
| `max_num_failures` | int | 50 | Max failures before stopping |
| `seed` | int | 1 | Random seed |
| `source_dataset_path` | str | None | Source demo path |
| `generation_path` | str | None | Output path |
| `generation_num_trials` | int | 10 | Trials to generate |
| `generation_select_src_per_subtask` | bool | False | Per-subtask source selection |
| `generation_interpolate_from_last_target_pose` | bool | True | Interpolate from last target |

### SubTaskConfig

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `object_ref` | str | None | Object involved in subtask |
| `subtask_term_signal` | str | None | Termination signal name |
| `selection_strategy` | str | "random" | Source demo selection |
| `action_noise` | float | 0.03 | Action noise amplitude |
| `num_interpolation_steps` | int | 5 | Interpolation steps |
| `num_fixed_steps` | int | 0 | Fixed steps |
| `subtask_term_offset_range` | tuple | (0, 0) | Boundary offset range |

**Selection Strategies:**
- `"random"` — Random source demo
- `"nearest_neighbor_object"` — Closest object pose match
- `"nearest_neighbor_robot_distance"` — Closest robot pose match

### SubTaskConstraintConfig

For multi-arm coordination:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `eef_subtask_constraint_tuple` | list | — | Associated subtask pairs |
| `constraint_type` | SubTaskConstraintType | None | SEQUENTIAL or COORDINATION |
| `coordination_scheme` | enum | REPLAY | REPLAY, TRANSFORM, TRANSLATE |

### DataGenerator

The `DataGenerator` class transforms source demonstrations using object-centric coordinate frames:

1. Randomize subtask boundaries
2. Select source demo segment per subtask
3. Transform EEF trajectory using relative object poses
4. Merge subtask trajectories with interpolation
5. Execute in environment and record

### Generation Script

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --input_file annotated_demos.hdf5 \
    --output_file generated_dataset.hdf5 \
    --num_envs 64
```

### Consolidated Recording + Generation

Record and generate in real-time (one env for teleoperation, others generate):

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/consolidated_demo.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --num_demos 5 \
    --num_envs 16
```

## Stage 4: Train Policy

### Robomimic Training

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --dataset generated_dataset.hdf5 \
    --algo bc
```

### Evaluate Trained Policy

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --checkpoint path/to/model.pth
```

## Motion Planners

### CuRobo Integration

For collision-aware trajectory generation:

```python
from isaaclab_mimic.motion_planners.curobo import CuroboPlannerCfg

planner_cfg = CuroboPlannerCfg(
    robot_config_file="franka.yml",
    ee_link_name="panda_hand",
    world_config_file="collision_table.yml",
    num_trajopt_seeds=12,
    interpolation_dt=0.05,
    enable_graph=True,
)
```

## Mimic Environment Base Class

To make an environment mimic-compatible, subclass `ManagerBasedRLMimicEnv` and implement:

```python
class MyMimicEnv(ManagerBasedRLMimicEnv):
    def get_robot_eef_pose(self, eef_name, env_ids=None) -> torch.Tensor:
        """Return 4x4 EEF pose matrix."""

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, ...):
        """Convert target pose to environment action."""

    def get_object_poses(self, env_ids=None) -> dict:
        """Return object poses in scene."""

    def get_subtask_term_signals(self, env_ids=None) -> dict:
        """Return subtask termination signals."""
```

## HDF5 Dataset Format

```
dataset.hdf5
  └── data/
      ├── attrs: {total, env_args}
      ├── demo_0/
      │   ├── attrs: {num_samples, seed, success}
      │   ├── actions [N, action_dim]
      │   ├── obs/
      │   │   └── datagen_info/
      │   │       ├── eef_pose/{eef_name} [N, 4, 4]
      │   │       ├── object_pose/{obj_name} [N, 4, 4]
      │   │       └── subtask_term_signals/{signal} [N]
      │   └── states/
      └── demo_1/, demo_2/, ...
```

## Reference Files

- [imitation-learning-api.md](imitation-learning-api.md) - Full API tables for DataGenConfig, MimicEnvCfg, SubTaskConstraintType, DataGenerator, RecorderTermCfg
- [imitation-learning-workflows.md](imitation-learning-workflows.md) - Step-by-step recipes for the full IL pipeline

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py` | DataGenerator class |
| `source/isaaclab_mimic/isaaclab_mimic/datagen/generation.py` | Async generation loop |
| `source/isaaclab_mimic/isaaclab_mimic/datagen/selection_strategy.py` | Source selection strategies |
| `source/isaaclab_mimic/isaaclab_mimic/datagen/waypoint.py` | Waypoint/trajectory classes |
| `source/isaaclab/isaaclab/envs/mimic_env_cfg.py` | MimicEnvCfg, DataGenConfig, SubTaskConfig |
| `source/isaaclab/isaaclab/envs/manager_based_rl_mimic_env.py` | ManagerBasedRLMimicEnv base |
| `source/isaaclab/isaaclab/managers/recorder_manager.py` | RecorderManager |
| `source/isaaclab_mimic/isaaclab_mimic/motion_planners/` | Motion planners (CuRobo) |
| `scripts/tools/record_demos.py` | Demo recording script |
| `scripts/imitation_learning/isaaclab_mimic/annotate_demos.py` | Demo annotation |
| `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py` | Dataset generation |
| `scripts/imitation_learning/robomimic/train.py` | Robomimic training |
