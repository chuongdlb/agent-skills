---
name: gymnasium-wrappers
description: >
  Built-in Gymnasium wrappers for observation, action, reward, rendering, and common transformations — composable environment modifiers for preprocessing and monitoring.
layer: L2
domain: [general-rl]
source-project: Gymnasium
depends-on: [gymnasium-core-api, gymnasium-spaces]
tags: [wrappers, preprocessing, normalization, frame-stack, video]
---

# Gymnasium Wrappers

## Purpose

Wrappers modify environment behavior without changing the underlying implementation. They compose via the decorator pattern — each wraps the previous, forming a chain. Gymnasium provides ~35 built-in wrappers covering observation preprocessing, action transformation, reward shaping, rendering, and validation.

## When to Use

- Preprocessing observations (resize, normalize, frame-stack, grayscale)
- Transforming actions (clip, rescale, discretize, sticky actions)
- Shaping rewards (clip, normalize)
- Adding episode limits and statistics
- Recording video of agent behavior
- Converting between array backends (NumPy, JAX, PyTorch)
- Enforcing API contracts (order, type checking)

## Observation Wrappers

| Wrapper | What It Does | Modified Space |
|---------|-------------|----------------|
| `FrameStackObservation(env, stack_size)` | Stacks last N frames | Box with extra dim |
| `NormalizeObservation(env, epsilon)` | Running mean/std normalization | Same shape |
| `FlattenObservation(env)` | Flattens Dict/Tuple obs to 1D | Box |
| `FilterObservation(env, filter_keys)` | Keep only specified Dict keys | Dict subset |
| `ResizeObservation(env, shape)` | Resize image observations | Box with new shape |
| `GrayscaleObservation(env, keep_dim)` | RGB → grayscale | Box with 1 channel |
| `ReshapeObservation(env, shape)` | Reshape observation array | Box with new shape |
| `RescaleObservation(env, min_obs, max_obs)` | Linearly rescale bounds | Box with new bounds |
| `DtypeObservation(env, dtype)` | Cast observation dtype | Box with new dtype |
| `MaxAndSkipObservation(env, skip)` | Skip frames, max over last 2 | Same |
| `TimeAwareObservation(env, *, flatten)` | Append timestep to obs | Box (extended) |
| `DelayObservation(env, delay)` | Delay observations by N steps | Same |
| `AddRenderObservation(env, render_key)` | Add rendered frame to obs | Dict |
| `TransformObservation(env, func, obs_space)` | Custom function transform | User-specified |
| `AtariPreprocessing(env, ...)` | Standard Atari pipeline | Box(84,84) grayscale |
| `DiscretizeObservation(env, num_bins)` | Continuous → discrete bins | MultiDiscrete |

### Common Patterns

```python
# Atari-style preprocessing
env = gymnasium.make("ALE/Breakout-v5")
env = MaxAndSkipObservation(env, skip=4)
env = ResizeObservation(env, shape=(84, 84))
env = GrayscaleObservation(env)
env = FrameStackObservation(env, stack_size=4)

# Dict observation filtering
env = FilterObservation(env, filter_keys=["position", "velocity"])

# Running normalization
env = NormalizeObservation(env, epsilon=1e-8)
```

## Action Wrappers

| Wrapper | What It Does | Modified Space |
|---------|-------------|----------------|
| `ClipAction(env)` | Clips continuous actions to bounds | Same |
| `RescaleAction(env, min_action, max_action)` | Rescale action range | Box with new bounds |
| `StickyAction(env, repeat_action_probability)` | Repeat previous action with prob p | Same |
| `DiscretizeAction(env, num_bins)` | Continuous → discrete bins | MultiDiscrete |
| `TransformAction(env, func, action_space)` | Custom function transform | User-specified |

```python
# Rescale actions from [-1, 1] to environment's actual range
env = RescaleAction(env, min_action=-1.0, max_action=1.0)

# Clip to valid bounds (safety)
env = ClipAction(env)
```

## Reward Wrappers

| Wrapper | What It Does |
|---------|-------------|
| `ClipReward(env, min_reward, max_reward)` | Clip reward to range |
| `NormalizeReward(env, gamma, epsilon)` | Running return normalization |
| `TransformReward(env, func)` | Custom function transform |

```python
# Normalize rewards using discounted return statistics
env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
```

## Common Wrappers

| Wrapper | What It Does |
|---------|-------------|
| `TimeLimit(env, max_episode_steps)` | Truncate after N steps (sets `truncated=True`) |
| `Autoreset(env, *, autoreset_mode)` | Auto-reset on termination/truncation |
| `RecordEpisodeStatistics(env, buffer_length)` | Track episode return, length, time in `info` |
| `OrderEnforcing(env, disable_render_order_enforcing)` | Ensure `reset()` called before `step()` |
| `PassiveEnvChecker(env)` | Validate obs/action types and spaces |

```python
# TimeLimit auto-applied by make() when max_episode_steps is set
# RecordEpisodeStatistics adds info["episode"] = {"r": total_return, "l": length, "t": time}
```

## Rendering Wrappers

| Wrapper | What It Does |
|---------|-------------|
| `RecordVideo(env, video_folder, *, episode_trigger, step_trigger, ...)` | Save episode videos to disk |
| `HumanRendering(env)` | Convert `rgb_array` render to on-screen display |
| `RenderCollection(env, pop_frames, reset_clean)` | Collect rendered frames into list |
| `AddWhiteNoise(env, strength)` | Add noise to rendered frames |
| `ObstructView(env, left, right, top, bottom)` | Black out regions of rendered frames |

```python
# Record every 10th episode
env = RecordVideo(env, "videos/", episode_trigger=lambda ep: ep % 10 == 0)

# Convert rgb_array mode to human display
env = HumanRendering(env)
```

## Conversion Wrappers

| Wrapper | From | To |
|---------|------|----|
| `JaxToNumpy(env)` | JAX arrays | NumPy arrays |
| `JaxToTorch(env, device)` | JAX arrays | PyTorch tensors |
| `NumpyToTorch(env, device)` | NumPy arrays | PyTorch tensors |

## Wrapper Stacking Best Practices

1. **Inner to outer**: Observation transforms first, then action transforms, then monitoring
2. **Normalization last** (before RL algorithm) so it sees the final observation
3. **TimeLimit applied by `make()`** — don't double-apply
4. **RecordEpisodeStatistics outermost** to capture true episode metrics

Typical stack:

```python
env = gymnasium.make("HalfCheetah-v5")          # includes TimeLimit, OrderEnforcing
env = ClipAction(env)                             # clip before env sees action
env = NormalizeObservation(env)                   # normalize obs
env = NormalizeReward(env, gamma=0.99)            # normalize rewards
env = RecordVideo(env, "videos/")                 # record for debugging
```

## Key Source Files

| File | Contents |
|------|----------|
| `gymnasium/wrappers/__init__.py` | All wrapper exports |
| `gymnasium/wrappers/common.py` | TimeLimit, Autoreset, RecordEpisodeStatistics, etc. |
| `gymnasium/wrappers/rendering.py` | RecordVideo, HumanRendering, etc. |
| `gymnasium/wrappers/transform_observation.py` | TransformObservation |
| `gymnasium/wrappers/transform_action.py` | TransformAction |
| `gymnasium/wrappers/transform_reward.py` | TransformReward |
| `gymnasium/wrappers/numpy_to_torch.py` | NumpyToTorch |

## Reference Files

- [wrappers-catalog.md](wrappers-catalog.md) — Complete table of all wrappers with constructor params, modified spaces, and categories
