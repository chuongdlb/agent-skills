---
name: isaaclab-rl-training
description: >
  Trains RL policies in IsaacLab using RSL-RL, RL-Games, Stable-Baselines3, or SKRL — PPO configs, checkpoints, multi-GPU, Ray tuning.
layer: L3
domain: [robotics, general-rl]
source-project: IsaacLab
depends-on: [isaaclab-environment-design, rl-theory-analyzer, K-Dense-AI/stable-baselines3, gymnasium-wrappers]
tags: [training, ppo, rsl-rl, multi-gpu]
---

# IsaacLab RL Training

IsaacLab supports four RL libraries through a unified wrapper architecture. RSL-RL is the primary library for locomotion and manipulation.

## Wrapper Architecture

```
Isaac Environment (ManagerBasedRLEnv / DirectRLEnv)
    ↓
(Optional) gym.wrappers (RecordVideo, etc.)
    ↓
RL Framework Wrapper [LAST — breaks gym interface]
    ├── RslRlVecEnvWrapper  → rsl_rl.runners.OnPolicyRunner
    ├── Sb3VecEnvWrapper    → stable_baselines3.PPO
    ├── RlGamesVecEnvWrapper → rl_games.torch_runner.Runner
    └── SkrlVecEnvWrapper   → skrl.utils.runner.Runner
```

The RL wrapper is always the **outermost** wrapper. It converts IsaacLab tensor observations to the format each library expects and handles device transfers.

## Supported Libraries

| Library | Wrapper | Algorithms | Best For |
|---------|---------|------------|----------|
| RSL-RL | `RslRlVecEnvWrapper` | PPO, Distillation | Locomotion, primary choice |
| RL-Games | `RlGamesVecEnvWrapper` | PPO (GPU-optimized) | High-throughput training |
| SB3 | `Sb3VecEnvWrapper` | PPO, SAC, etc. | Prototyping, benchmarks |
| SKRL | `SkrlVecEnvWrapper` | PPO, IPPO, MAPPO | Multi-agent, JAX support |

## RSL-RL Training Walkthrough

### 1. Define Agent Config

```python
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class MyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "my_task"
    run_name = ""
    logger = "tensorboard"
    obs_groups = {"policy": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### 2. Register Agent Config

In your task's `__init__.py`:

```python
gym.register(
    id="Isaac-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "my_pkg.env_cfg:MyEnvCfg",
        "rsl_rl_cfg_entry_point": "my_pkg.agents.rsl_rl_ppo_cfg:MyPPORunnerCfg",
    },
)
```

### 3. Train

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-MyTask-v0 \
    --num_envs 4096 \
    --headless
```

### 4. Evaluate

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-MyTask-v0 \
    --num_envs 32
```

## RSL-RL Config Classes

### RslRlOnPolicyRunnerCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed |
| `device` | str | "cuda:0" | Training device |
| `num_steps_per_env` | int | MISSING | Steps per env per iteration |
| `max_iterations` | int | MISSING | Total training iterations |
| `save_interval` | int | MISSING | Checkpoint save frequency |
| `experiment_name` | str | MISSING | Log folder name |
| `run_name` | str | "" | Run suffix |
| `logger` | str | "tensorboard" | "tensorboard", "wandb", "neptune" |
| `resume` | bool | False | Resume from checkpoint |
| `load_run` | str | ".*" | Run folder regex |
| `load_checkpoint` | str | "model_.*.pt" | Checkpoint regex |
| `obs_groups` | dict | MISSING | Observation group mapping |
| `clip_actions` | float \| None | None | Action clipping |
| `policy` | RslRlPpoActorCriticCfg | MISSING | Policy config |
| `algorithm` | RslRlPpoAlgorithmCfg | MISSING | Algorithm config |

### RslRlPpoActorCriticCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_name` | str | "ActorCritic" | Policy class |
| `init_noise_std` | float | MISSING | Initial action noise std |
| `noise_std_type` | str | "scalar" | "scalar" or "log" |
| `actor_hidden_dims` | list[int] | MISSING | Actor MLP layers |
| `critic_hidden_dims` | list[int] | MISSING | Critic MLP layers |
| `activation` | str | MISSING | "elu", "relu", "tanh" |
| `actor_obs_normalization` | bool | MISSING | Normalize actor obs |
| `critic_obs_normalization` | bool | MISSING | Normalize critic obs |

### RslRlPpoAlgorithmCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_learning_epochs` | int | MISSING | Epochs per iteration |
| `num_mini_batches` | int | MISSING | Mini-batches per epoch |
| `learning_rate` | float | MISSING | Learning rate |
| `schedule` | str | MISSING | "adaptive" or "fixed" |
| `gamma` | float | MISSING | Discount factor |
| `lam` | float | MISSING | GAE lambda |
| `entropy_coef` | float | MISSING | Entropy bonus |
| `desired_kl` | float | MISSING | Target KL divergence |
| `max_grad_norm` | float | MISSING | Gradient clipping |
| `value_loss_coef` | float | MISSING | Value loss weight |
| `use_clipped_value_loss` | bool | MISSING | Clip value loss |
| `clip_param` | float | MISSING | PPO clip range |

## CLI Arguments

### Common (all libraries)

| Flag | Description |
|------|-------------|
| `--task` | Task name (gym ID) |
| `--num_envs` | Override number of environments |
| `--seed` | Random seed |
| `--max_iterations` | Override max iterations |
| `--headless` | Run without GUI |
| `--device` | Compute device |
| `--video` | Record videos |
| `--video_length` | Video length (steps) |
| `--video_interval` | Steps between videos |

### RSL-RL Specific

| Flag | Description |
|------|-------------|
| `--experiment_name` | Log folder name |
| `--run_name` | Run suffix |
| `--resume` | Resume from checkpoint |
| `--load_run` | Run folder pattern |
| `--checkpoint` | Checkpoint file pattern |
| `--logger` | Logger type |
| `--log_project_name` | WandB/Neptune project |
| `--distributed` | Multi-GPU training |

## Checkpoint Management

```
logs/
  rsl_rl/{experiment_name}/{TIMESTAMP}/
    model_0.pt, model_50.pt, ...
  rl_games/{task}/nn/{task}.pth
  sb3/{task}/{TIMESTAMP}/model.zip
  skrl/{task}/{TIMESTAMP}/checkpoints/best_agent.pt
```

Resume training:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-MyTask-v0 --resume \
    --load_run "2024.*" --checkpoint "model_1000.pt"
```

## Multi-GPU Training

```bash
# 2-GPU training
python -m torch.distributed.run --nproc_per_node=2 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-MyTask-v0 --headless --distributed
```

Each GPU gets its own environment instance with unique seed.

## Ray Hyperparameter Tuning

```bash
python scripts/reinforcement_learning/ray/tuner.py \
    --cfg_file my_sweep_config.py
```

Ray Tune monitors TensorBoard logs and uses OptunaSearch for hyperparameter optimization.

## Policy Export

```python
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

export_policy_as_jit(actor_critic, path="policy.pt")
export_policy_as_onnx(actor_critic, path="policy.onnx")
```

## Reference Files

- [rl-training-configs.md](rl-training-configs.md) - Full config tables for RSL-RL, RL-Games, SB3, SKRL
- [rl-training-workflows.md](rl-training-workflows.md) - Step-by-step training recipes

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/isaaclab_rl/isaaclab_rl/rsl_rl/` | RSL-RL wrappers and configs |
| `source/isaaclab_rl/isaaclab_rl/rl_games/` | RL-Games wrapper |
| `source/isaaclab_rl/isaaclab_rl/sb3.py` | SB3 wrapper |
| `source/isaaclab_rl/isaaclab_rl/skrl.py` | SKRL wrapper |
| `scripts/reinforcement_learning/rsl_rl/train.py` | RSL-RL train script |
| `scripts/reinforcement_learning/rsl_rl/play.py` | RSL-RL evaluation |
| `scripts/reinforcement_learning/sb3/train.py` | SB3 train script |
| `scripts/reinforcement_learning/rl_games/train.py` | RL-Games train script |
| `scripts/reinforcement_learning/skrl/train.py` | SKRL train script |
| `scripts/reinforcement_learning/ray/tuner.py` | Ray hyperparameter tuning |
| `source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py` | Policy export (JIT/ONNX) |
