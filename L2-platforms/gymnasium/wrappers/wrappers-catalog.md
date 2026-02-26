# Wrappers Catalog — Full Constructor Signatures

## Observation Wrappers

### FrameStackObservation

```python
FrameStackObservation(env, stack_size: int, *, padding_type: str = "reset")
```

- Stacks last `stack_size` frames along a new first axis
- `padding_type`: `"reset"` (fill with reset obs) or `"zero"` (fill with zeros)
- Observation space: adds leading dimension of `stack_size`

### NormalizeObservation

```python
NormalizeObservation(env, epsilon: float = 1e-8)
```

- Maintains running mean and variance, normalizes `obs = (obs - mean) / sqrt(var + eps)`
- `update_running_mean`: attribute to toggle updates (set `False` at eval time)

### FlattenObservation

```python
FlattenObservation(env)
```

- Flattens Dict/Tuple observations into 1D Box using `flatten_space()`
- No parameters beyond env

### FilterObservation

```python
FilterObservation(env, filter_keys: Sequence[str])
```

- Keeps only specified keys from Dict observation space
- Errors if observation_space is not Dict

### ResizeObservation

```python
ResizeObservation(env, shape: tuple[int, int])
```

- Resizes image observations using `cv2.resize` (bilinear interpolation)
- `shape`: `(height, width)` — does not include channels

### GrayscaleObservation

```python
GrayscaleObservation(env, keep_dim: bool = False)
```

- Converts RGB → grayscale via weighted sum
- `keep_dim=True`: shape `(H, W, 1)` instead of `(H, W)`

### ReshapeObservation

```python
ReshapeObservation(env, shape: int | tuple[int, ...])
```

- Reshapes observation array to new shape (total elements must match)

### RescaleObservation

```python
RescaleObservation(env, min_obs: float | NDArray, max_obs: float | NDArray)
```

- Linearly maps observation from original [low, high] to [min_obs, max_obs]

### DtypeObservation

```python
DtypeObservation(env, dtype: type)
```

- Casts observation to specified numpy dtype

### MaxAndSkipObservation

```python
MaxAndSkipObservation(env, skip: int = 4)
```

- Repeats action for `skip` frames, returns element-wise max of last 2 frames
- Standard Atari frame-skip technique

### TimeAwareObservation

```python
TimeAwareObservation(env, *, flatten: bool = True, normalize_time: bool = True,
                     dict_time_key: str = "time")
```

- Appends normalized timestep to observation
- `flatten=True`: concatenate to 1D; `flatten=False`: use Dict
- `normalize_time`: divide by max_episode_steps if available

### DelayObservation

```python
DelayObservation(env, delay: int)
```

- Returns observation from `delay` steps ago
- Buffer initialized with reset observation

### AddRenderObservation

```python
AddRenderObservation(env, render_key: str = "render")
```

- Adds rendered frame to observation as Dict entry
- Requires `render_mode="rgb_array"` or similar

### TransformObservation

```python
TransformObservation(env, func: Callable[[ObsType], Any],
                     observation_space: Space | None = None)
```

- Applies arbitrary function to observations
- Must provide `observation_space` if shape/type changes

### AtariPreprocessing

```python
AtariPreprocessing(
    env,
    noop_max: int = 30,
    frame_skip: int = 4,
    screen_size: int = 84,
    terminal_on_life_loss: bool = False,
    grayscale_obs: bool = True,
    grayscale_newaxis: bool = False,
    scale_obs: bool = False,
)
```

- Full Atari preprocessing: noop starts, frame-skip with max, resize, grayscale
- `scale_obs`: divide by 255.0

### DiscretizeObservation

```python
DiscretizeObservation(env, num_bins: int | Sequence[int])
```

- Converts continuous Box observation to MultiDiscrete by binning
- `num_bins`: bins per dimension (scalar or per-dim array)

## Action Wrappers

### ClipAction

```python
ClipAction(env)
```

- Clips action to `action_space.low` / `action_space.high`
- No parameters — uses existing space bounds

### RescaleAction

```python
RescaleAction(env, min_action: float | NDArray, max_action: float | NDArray)
```

- Agent acts in [min_action, max_action], linearly mapped to original bounds
- Common: `RescaleAction(env, -1.0, 1.0)` to normalize all actions to [-1, 1]

### StickyAction

```python
StickyAction(env, repeat_action_probability: float)
```

- With probability p, repeats previous action instead of using new one
- Standard Atari stochasticity technique

### DiscretizeAction

```python
DiscretizeAction(env, num_bins: int | Sequence[int])
```

- Converts continuous Box action to MultiDiscrete by binning
- Inverse of DiscretizeObservation applied to actions

### TransformAction

```python
TransformAction(env, func: Callable[[ActType], Any],
                action_space: Space | None = None)
```

- Applies arbitrary function to actions before env.step()
- Must provide `action_space` if the agent's action space differs

## Reward Wrappers

### ClipReward

```python
ClipReward(env, min_reward: float = -1.0, max_reward: float = 1.0)
```

- Clips reward to [min_reward, max_reward]
- Common for Atari: `ClipReward(env, -1, 1)`

### NormalizeReward

```python
NormalizeReward(env, gamma: float = 0.99, epsilon: float = 1e-8)
```

- Normalizes reward using running variance of discounted returns
- `update_running_mean`: attribute to toggle (set `False` at eval)
- `gamma`: discount factor for return computation

### TransformReward

```python
TransformReward(env, func: Callable[[float], float])
```

- Applies arbitrary function to reward scalar

## Common Wrappers

### TimeLimit

```python
TimeLimit(env, max_episode_steps: int)
```

- Sets `truncated=True` after `max_episode_steps`
- Auto-applied by `gymnasium.make()` when `max_episode_steps` in EnvSpec

### Autoreset

```python
Autoreset(env, *, autoreset_mode: AutoresetMode | str = "next_step")
```

- Auto-resets when episode ends
- `"next_step"`: reset on next step call, return reset obs
- `"same_step"`: reset immediately, return reset obs in same step

### RecordEpisodeStatistics

```python
RecordEpisodeStatistics(env, buffer_length: int = 100)
```

- On episode end, adds `info["episode"] = {"r": return, "l": length, "t": wall_time}`
- Maintains rolling buffer of last `buffer_length` episodes

### OrderEnforcing

```python
OrderEnforcing(env, disable_render_order_enforcing: bool = False)
```

- Raises error if `step()` called before `reset()`
- Optionally enforces render order

### PassiveEnvChecker

```python
PassiveEnvChecker(env)
```

- Validates observation/action types and space containment on first step/reset
- No runtime overhead after initial checks

## Rendering Wrappers

### RecordVideo

```python
RecordVideo(
    env,
    video_folder: str,
    *,
    episode_trigger: Callable[[int], bool] | None = None,
    step_trigger: Callable[[int], bool] | None = None,
    video_length: int = 0,
    name_prefix: str = "rl-video",
    fps: int | None = None,
    disable_logger: bool = False,
)
```

- Records episodes to MP4 files in `video_folder`
- `episode_trigger`: function(episode_id) -> bool for which episodes to record
- `step_trigger`: function(step_id) -> bool for step-based recording
- `video_length`: max frames per video (0 = full episode)
- Requires `render_mode="rgb_array"` on the base env

### HumanRendering

```python
HumanRendering(env)
```

- Wraps an `rgb_array` env and displays frames in a pygame window
- Changes effective render_mode to `"human"`

### RenderCollection

```python
RenderCollection(env, pop_frames: bool = True, reset_clean: bool = True)
```

- Collects rendered frames; `render()` returns list of all collected frames
- `pop_frames`: clear buffer on render() call
- `reset_clean`: clear buffer on reset()

### AddWhiteNoise

```python
AddWhiteNoise(env, strength: float = 0.1)
```

- Adds Gaussian noise to rendered frames (for robustness testing)

### ObstructView

```python
ObstructView(env, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0)
```

- Blacks out border regions of rendered frames

## Conversion Wrappers

### JaxToNumpy

```python
JaxToNumpy(env)
```

- Converts JAX arrays in obs/reward/terminated/truncated to NumPy

### JaxToTorch

```python
JaxToTorch(env, device: torch.device | str | None = None)
```

- Converts JAX arrays to PyTorch tensors

### NumpyToTorch

```python
NumpyToTorch(env, device: torch.device | str | None = None)
```

- Converts NumPy arrays to PyTorch tensors on specified device
