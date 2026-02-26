# RL Training Workflows

Step-by-step recipes for common RL training tasks.

## Recipe 1: Train Locomotion Policy (RSL-RL)

```bash
# 1. Train with 4096 environments
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 4096 \
    --headless

# 2. Evaluate the trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 32

# 3. Record video of policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 32 --video --video_length 300
```

## Recipe 2: Train Manipulation Policy (RSL-RL)

```bash
# Train Franka reach task
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-Franka-v0 \
    --num_envs 2048 \
    --headless

# Train with IK-based actions
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-Franka-IK-Rel-v0 \
    --num_envs 2048 \
    --headless
```

## Recipe 3: Resume Training

```bash
# Resume from latest checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 4096 \
    --headless --resume

# Resume from specific run and checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 4096 \
    --headless --resume \
    --load_run "2024-01-15.*" \
    --checkpoint "model_500.pt"
```

## Recipe 4: Evaluate Checkpoint

```bash
# Play with specific checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 32 \
    --load_run "2024-01-15.*" \
    --checkpoint "model_1000.pt"
```

## Recipe 5: Record Video

```bash
# Record video during training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-Franka-v0 \
    --num_envs 2048 \
    --headless \
    --video --video_length 200 --video_interval 5000

# Record video during evaluation
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Reach-Franka-v0 \
    --num_envs 1 \
    --video --video_length 500
```

## Recipe 6: Multi-GPU Training

```bash
# 2-GPU distributed training
python -m torch.distributed.run --nproc_per_node=2 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless --distributed

# 4-GPU distributed training
python -m torch.distributed.run --nproc_per_node=4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless --distributed
```

Each process gets its own GPU, environment set, and unique seed.

## Recipe 7: Ray Hyperparameter Sweep

```python
# sweep_config.py
from ray import tune

class MyJobCfg:
    task_name = "Isaac-Reach-Franka-v0"
    num_envs = 2048
    max_iterations = 500

    # Hyperparameters to sweep
    search_space = {
        "agent.algorithm.learning_rate": tune.loguniform(1e-4, 1e-2),
        "agent.algorithm.gamma": tune.uniform(0.95, 0.999),
        "agent.algorithm.entropy_coef": tune.loguniform(1e-4, 0.1),
        "agent.policy.actor_hidden_dims": tune.choice(
            [[128, 128], [256, 256], [256, 128, 64]],
        ),
    }
```

```bash
python scripts/reinforcement_learning/ray/tuner.py --cfg_file sweep_config.py
```

## Recipe 8: Train with SB3

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task Isaac-Cartpole-v0 \
    --num_envs 256 \
    --headless \
    --agent sb3_cfg_entry_point
```

## Recipe 9: Train with SKRL

```bash
# PyTorch backend
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Reach-Franka-v0 \
    --num_envs 2048 \
    --headless \
    --ml_framework torch

# JAX backend
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Reach-Franka-v0 \
    --num_envs 2048 \
    --headless \
    --ml_framework jax
```

## Recipe 10: Train with RL-Games

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Ant-v0 \
    --num_envs 4096 \
    --headless
```

## Recipe 11: WandB Logging

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 4096 \
    --headless \
    --logger wandb \
    --log_project_name my_project
```

## Recipe 12: Export Policy for Deployment

```python
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

# Load trained runner
runner = OnPolicyRunner(env, agent_cfg, ...)
runner.load("logs/rsl_rl/experiment/run/model_1500.pt")

# Export as TorchScript
export_policy_as_jit(
    runner.alg.actor_critic,
    path="exported_policy.pt",
)

# Export as ONNX
export_policy_as_onnx(
    runner.alg.actor_critic,
    path="exported_policy.onnx",
)
```

## Common Patterns

### Override Environment Config from CLI

```bash
# Override num_envs and physics dt via Hydra
python train.py task=Isaac-Reach-Franka-v0 \
    env.scene.num_envs=512 \
    env.sim.dt=0.01 \
    env.decimation=10
```

### Custom Training Script

```python
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Load configs
env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
agent_cfg = parse_agent_cfg(args.task, "rsl_rl_cfg_entry_point")

# Create env and wrap
env = gym.make(args.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

# Train
from rsl_rl.runners import OnPolicyRunner
runner = OnPolicyRunner(env, agent_cfg.to_dict(), ...)
runner.learn(num_learning_iterations=agent_cfg.max_iterations)
```
