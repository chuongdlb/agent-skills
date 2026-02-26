---
name: gpd-training-evaluation
description: >
  Stable-Baselines3 PPO training pipeline — environment setup, callbacks, reward thresholds, model saving, evaluation, inference loop.
layer: L3
domain: [drones, general-rl]
source-project: gym-pybullet-drones
depends-on: [gpd-rl-environments, pybullet-simulation-engine, K-Dense-AI/stable-baselines3]
tags: [training, ppo, sb3, evaluation]
---

# Training & Evaluation

End-to-end RL training pipeline using Stable-Baselines3 PPO with HoverAviary/MultiHoverAviary. Covers training, early stopping, model saving, evaluation, and visual inference.

## Pipeline Overview

```
1. Create environments ──→ make_vec_env(HoverAviary, ...)
2. Build model ──────────→ PPO('MlpPolicy', train_env)
3. Configure callbacks ──→ EvalCallback + StopTrainingOnRewardThreshold
4. Train ────────────────→ model.learn(total_timesteps=1e7)
5. Save ─────────────────→ model.save('final_model.zip')
6. Load best ────────────→ PPO.load('best_model.zip')
7. Evaluate ─────────────→ evaluate_policy(model, test_env)
8. Visual inference ─────→ model.predict(obs) in loop with GUI
```

## Training Setup

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('one_d_rpm')

# Single-agent
train_env = make_vec_env(HoverAviary,
                         env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                         n_envs=1, seed=0)
eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

# Multi-agent
train_env = make_vec_env(MultiHoverAviary,
                         env_kwargs=dict(num_drones=2, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                         n_envs=1, seed=0)
eval_env = MultiHoverAviary(num_drones=2, obs=DEFAULT_OBS, act=DEFAULT_ACT)
```

**Important:** Training environments use default `ctrl_freq=30` and `gui=False`.

## Model Configuration

```python
model = PPO('MlpPolicy', train_env, verbose=1)
```

Default PPO hyperparameters from SB3 are used (no custom tuning required for basic hover).

## Callbacks and Early Stopping

```python
target_reward = 474.  # for ONE_D_RPM single-agent (see reference)

callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=target_reward, verbose=1)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    verbose=1,
    best_model_save_path=filename + '/',
    log_path=filename + '/',
    eval_freq=1000,              # evaluate every 1000 steps
    deterministic=True,
    render=False
)
```

## Training

```python
model.learn(total_timesteps=int(1e7),
            callback=eval_callback,
            log_interval=100)

model.save(filename + '/final_model.zip')
```

## Output Files

```
results/save-MM.DD.YYYY_HH.MM.SS/
├── best_model.zip        # best model by eval reward
├── final_model.zip       # model at end of training
└── evaluations.npz       # timesteps + results arrays
```

**Reading evaluation results:**
```python
with np.load(filename + '/evaluations.npz') as data:
    timesteps = data['timesteps']
    results = data['results'][:, 0]
```

## Evaluation

```python
model = PPO.load(filename + '/best_model.zip')

# Quantitative
mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)

# Visual inference loop
test_env = HoverAviary(gui=True, obs=DEFAULT_OBS, act=DEFAULT_ACT)
obs, info = test_env.reset(seed=42, options={})
start = time.time()

for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    test_env.render()
    sync(i, start, test_env.CTRL_TIMESTEP)
    if terminated:
        obs = test_env.reset(seed=42, options={})

test_env.close()
```

## Logger Integration During Inference

```python
from gym_pybullet_drones.utils.Logger import Logger

logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ), num_drones=1)

# In inference loop (KIN observations only):
obs2 = obs.squeeze()
act2 = action.squeeze()
logger.log(drone=0,
           timestamp=i / test_env.CTRL_FREQ,
           state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
           control=np.zeros(12))

# After loop:
logger.plot()
```

Note: The state vector passed to Logger must be 20D. Since KIN observations are 12D (no quaternion or RPM), zeros are inserted for quaternion and the action is appended for the RPM slots.

## Running the Examples

```bash
# Train single-agent
python gym_pybullet_drones/examples/learn.py --multiagent false

# Train multi-agent
python gym_pybullet_drones/examples/learn.py --multiagent true

# Play back a trained model
python gym_pybullet_drones/examples/play.py --model_path results/best_model.zip
```

See [hyperparameter-reference.md](hyperparameter-reference.md) for reward thresholds and configurations.

## Key Source Files

| File | Purpose |
|------|---------|
| `gym_pybullet_drones/examples/learn.py` | Full training + evaluation script |
| `gym_pybullet_drones/examples/play.py` | Load and play back a trained model |
| `gym_pybullet_drones/envs/HoverAviary.py` | Single-agent training environment |
| `gym_pybullet_drones/envs/MultiHoverAviary.py` | Multi-agent training environment |
