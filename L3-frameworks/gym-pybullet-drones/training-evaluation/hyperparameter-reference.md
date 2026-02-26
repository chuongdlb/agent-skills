# Hyperparameter Reference

Reward thresholds, PPO defaults, and known working configurations.

## Target Reward Thresholds

From `examples/learn.py`:

| ActionType | Single-agent | Multi-agent (2 drones) |
|---|---|---|
| `ONE_D_RPM` | 474.0 | 949.5 |
| `RPM` | 467.0 | 920.0 |
| `PID` | 467.0 | 920.0 |
| `VEL` | 467.0 | 920.0 |
| `ONE_D_PID` | 467.0 | 920.0 |

Multi-agent thresholds are approximately 2x single-agent because the reward is summed across drones.

These thresholds are used with `StopTrainingOnRewardThreshold` to enable early stopping when the policy is sufficiently good.

## HoverAviary Reward Analysis

**Reward function:** `max(0, 2 - ||target_pos - cur_pos||^4)`

| Distance to target | Reward per step |
|---|---|
| 0.0 m | 2.0 (maximum) |
| 0.5 m | 1.9375 |
| 1.0 m | 1.0 |
| 1.19 m | ~0.0 |
| >1.19 m | 0.0 |

**Episode structure:**
- `EPISODE_LEN_SEC = 8` seconds
- `ctrl_freq = 30` Hz
- Steps per episode: `8 * 30 = 240`
- Maximum possible reward per episode: `240 * 2 = 480`
- ONE_D_RPM target (474) = ~98.75% of maximum

## PPO Default Configuration

SB3's PPO defaults used in `learn.py`:

| Parameter | Default value |
|---|---|
| Policy | `MlpPolicy` |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.0 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |

No custom tuning is applied in the example scripts.

## Training Configuration

| Parameter | Value |
|---|---|
| total_timesteps | 1e7 (local) / 1e2 (CI/pytest) |
| eval_freq | 1000 steps |
| deterministic eval | True |
| n_envs | 1 |
| seed | 0 |

## Environment Configuration for Training

| Parameter | HoverAviary | MultiHoverAviary |
|---|---|---|
| ctrl_freq | 30 | 30 |
| pyb_freq | 240 | 240 |
| num_drones | 1 | 2 (configurable) |
| gui | False | False |
| obstacles | True (from BaseRLAviary) | True |
| user_debug_gui | False (from BaseRLAviary) | False |

## Action Type Dimensions

| ActionType | Action dim | Obs dim (ctrl_freq=30) | Best suited for |
|---|---|---|---|
| `ONE_D_RPM` | 1 | 27 | Simplest, altitude-only |
| `ONE_D_PID` | 1 | 27 | PID-assisted altitude |
| `PID` | 3 | 57 | Position waypoints |
| `RPM` | 4 | 72 | Direct motor control |
| `VEL` | 4 | 72 | Velocity commands |

`ONE_D_RPM` is used as the default in `learn.py` because it converges fastest.

## Known Working Configurations

### Default (fastest convergence)
```python
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('one_d_rpm')
```
Converges to target reward in ~1-3 million steps.

### Full motor control
```python
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
```
Harder problem, may need 5-10 million steps.

## Truncation Conditions

### HoverAviary (single-agent)
- |x| > 1.5 m or |y| > 1.5 m or z > 2.0 m
- |roll| > 0.4 rad or |pitch| > 0.4 rad
- Time > EPISODE_LEN_SEC (8s)

### MultiHoverAviary (multi-agent)
- Any drone: |x| > 2.0 m or |y| > 2.0 m or z > 2.0 m
- Any drone: |roll| > 0.4 rad or |pitch| > 0.4 rad
- Time > EPISODE_LEN_SEC (8s)

## Evaluation Metrics

```python
mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
```

A well-trained ONE_D_RPM single-agent policy should achieve `mean_reward ≈ 474 ± 5`.
