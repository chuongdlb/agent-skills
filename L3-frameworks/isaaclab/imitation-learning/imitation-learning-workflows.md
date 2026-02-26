# Imitation Learning Workflows

Step-by-step recipes for the complete imitation learning pipeline.

## Recipe 1: Record Demos with Keyboard

```bash
# Step 1: Record teleoperated demonstrations
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Lift-Franka-IK-Abs-Gripper-State-v0 \
    --teleop_device keyboard \
    --dataset_file my_demos.hdf5 \
    --num_demos 10 \
    --step_hz 30

# Controls:
# W/S: X movement, A/D: Y movement, Q/E: Z movement
# Z/X: Roll, T/G: Pitch, C/V: Yaw
# K: Toggle gripper
# Enter: Save successful demo
# Backspace: Discard current demo
```

## Recipe 2: Record Demos with SpaceMouse

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Lift-Franka-IK-Abs-Gripper-State-v0 \
    --teleop_device spacemouse \
    --dataset_file my_demos.hdf5 \
    --num_demos 10
```

## Recipe 3: Annotate Subtasks

```bash
# Automatic annotation using subtask termination signals
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --input_file my_demos.hdf5 \
    --output_file annotated_demos.hdf5 \
    --auto

# With subtask start signals (for skillgen)
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --input_file my_demos.hdf5 \
    --output_file annotated_demos.hdf5 \
    --auto \
    --annotate_subtask_start_signals
```

## Recipe 4: Run Mimic Data Generation

```bash
# Generate 1000 demos from 10 source demos using 64 parallel envs
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --input_file annotated_demos.hdf5 \
    --output_file generated_dataset.hdf5 \
    --num_envs 64 \
    --headless
```

## Recipe 5: Train Robomimic Policy

```bash
# Train BC policy on generated dataset
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --dataset generated_dataset.hdf5 \
    --algo bc

# Evaluate trained policy
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --checkpoint path/to/model.pth \
    --num_envs 16
```

## Recipe 6: Consolidated Demo (Record + Generate Simultaneously)

```bash
# One env for teleoperation, rest for generation
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/consolidated_demo.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --num_demos 5 \
    --num_envs 16 \
    --teleop_env_index 0 \
    --generated_output_file generated.hdf5
```

## Recipe 7: Set Up New Mimic-Compatible Environment

### Step 1: Define Subtask Termination Signals

In your env config, define observation terms that serve as subtask boundary signals:

```python
# These return binary signals indicating subtask completion
def grasp_signal(env, ...) -> torch.Tensor:
    """Returns 1 when object is grasped, 0 otherwise."""
    contact_forces = env.scene["contact_sensor"].data.net_forces_w_history
    return (contact_forces.norm(dim=-1) > threshold).float()
```

### Step 2: Create MimicEnvCfg

```python
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, DataGenConfig, SubTaskConfig

@configclass
class MyMimicEnvCfg(MyBaseEnvCfg):
    mimic: MimicEnvCfg = MimicEnvCfg(
        datagen_config=DataGenConfig(
            generation_num_trials=1000,
            generation_guarantee=True,
        ),
        subtask_configs={
            "ee": [
                SubTaskConfig(
                    object_ref="target_object",
                    subtask_term_signal="grasp",
                    selection_strategy="nearest_neighbor_object",
                    action_noise=0.03,
                    num_interpolation_steps=5,
                ),
                SubTaskConfig(
                    object_ref="target_object",
                    subtask_term_signal="place",
                    selection_strategy="random",
                    action_noise=0.03,
                ),
            ],
        },
    )
```

### Step 3: Implement Mimic Env Class

```python
from isaaclab.envs import ManagerBasedRLMimicEnv

class MyMimicEnv(ManagerBasedRLMimicEnv):
    def get_robot_eef_pose(self, eef_name, env_ids=None):
        ee_frame = self.scene["ee_frame"]
        pos = ee_frame.data.target_pos_w[..., 0, :]
        quat = ee_frame.data.target_quat_w[..., 0, :]
        pose = torch.eye(4, device=self.device).repeat(len(pos), 1, 1)
        pose[:, :3, :3] = matrix_from_quat(quat)
        pose[:, :3, 3] = pos
        return pose

    def target_eef_pose_to_action(self, target_eef_pose_dict,
                                   gripper_action_dict, action_noise_dict, env_id):
        # Convert 4x4 pose to action format your env expects
        ...
        return action

    def get_object_poses(self, env_ids=None):
        obj = self.scene["object"]
        pos = obj.data.root_pos_w
        quat = obj.data.root_quat_w
        pose = make_4x4_pose(pos, quat)
        return {"target_object": pose}

    def get_subtask_term_signals(self, env_ids=None):
        return {
            "grasp": self._compute_grasp_signal(),
            "place": self._compute_place_signal(),
        }
```

### Step 4: Register

```python
gym.register(
    id="Isaac-MyTask-Mimic-v0",
    entry_point="my_pkg.env:MyMimicEnv",
    kwargs={
        "env_cfg_entry_point": "my_pkg.env_cfg:MyMimicEnvCfg",
    },
)
```

## Recipe 8: Robust Evaluation

```bash
# Evaluate policy robustness across multiple seeds
./isaaclab.sh -p scripts/imitation_learning/robomimic/robust_eval.py \
    --task Isaac-Lift-Franka-Mimic-v0 \
    --checkpoint path/to/model.pth \
    --num_envs 64 \
    --num_episodes 500
```

## Pipeline Summary

```
1. record_demos.py    →  demos.hdf5 (10 human demos)
                              ↓
2. annotate_demos.py  →  annotated.hdf5 (with subtask boundaries)
                              ↓
3. generate_dataset.py → generated.hdf5 (1000+ synthetic demos)
                              ↓
4. robomimic/train.py →  model.pth (trained policy)
                              ↓
5. robomimic/play.py  →  evaluation results
```

## Tips

- **Recording quality matters**: Spend time recording good demonstrations. Poor demos lead to poor generation.
- **Start with 5-10 demos**: Mimic can amplify a small set into thousands.
- **Use nearest_neighbor_object** for object-centric subtasks (grasping, placing).
- **Use random** for simple motions (approach, retract).
- **action_noise**: Start with 0.03, increase for more diversity.
- **num_interpolation_steps**: 5 is typical, increase for smoother transitions.
- **Parallel generation**: More `--num_envs` = faster generation. Use 32-128.
- **Check success rate**: If generation success rate is low (<50%), check subtask boundaries.
