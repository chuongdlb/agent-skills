---
name: gymnasium-core-api
description: >
  Core Gymnasium RL environment interface — Env base class, step/reset contract, Wrapper hierarchy, registration system, and standard RL loop patterns.
layer: L2
domain: [general-rl]
source-project: Gymnasium
depends-on: []
tags: [gymnasium, env, wrapper, registration, rl-loop]
---

# Gymnasium Core API

## Purpose

Provides the foundational `Env` interface that all Gymnasium-compatible environments implement. This is the standard API for RL environment interaction used by IsaacLab, gym-pybullet-drones, RL-Tools, Stable-Baselines3, and virtually all Python RL libraries.

## When to Use

- Creating or interfacing with any RL environment in Python
- Understanding step/reset return signatures
- Composing wrappers around environments
- Registering environments for `gymnasium.make()`
- Writing a standard RL training loop

## Env Base Class

```python
class Env(Generic[ObsType, ActType]):
    # Required attributes (set by subclass)
    metadata: dict[str, Any] = {"render_modes": []}
    render_mode: str | None = None
    observation_space: spaces.Space[ObsType]
    action_space: spaces.Space[ActType]

    # Managed by framework
    spec: EnvSpec | None = None
    _np_random: np.random.Generator | None = None
```

### Core Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `step` | `step(action: ActType)` | `(obs, reward, terminated, truncated, info)` |
| `reset` | `reset(*, seed=None, options=None)` | `(obs, info)` |
| `render` | `render()` | `RenderFrame \| list[RenderFrame] \| None` |
| `close` | `close()` | `None` |

### Step Return Tuple (v26+ API)

```python
obs, reward, terminated, truncated, info = env.step(action)
# obs: ObsType           — new observation
# reward: SupportsFloat  — scalar reward
# terminated: bool       — episode ended by environment (goal/failure)
# truncated: bool        — episode ended by time limit or external
# info: dict[str, Any]   — auxiliary diagnostic information
```

### Reset Return

```python
obs, info = env.reset(seed=42, options={"difficulty": "hard"})
# seed: sets np_random for reproducibility
# options: env-specific configuration
```

### Wrapper Attribute Access

```python
env.has_wrapper_attr("is_vector_env")   # check wrapper chain
env.get_wrapper_attr("render_mode")     # get from any wrapper
env.set_wrapper_attr("training", True)  # set on nearest wrapper
```

### Properties

```python
env.unwrapped      # innermost Env (bypasses all wrappers)
env.np_random      # np.random.Generator for reproducibility
env.np_random_seed # seed used to create np_random
```

## Wrapper Hierarchy

```
Wrapper(Env)                    — base: delegates all methods to env
├── ObservationWrapper(Wrapper) — override observation(obs) -> new_obs
├── RewardWrapper(Wrapper)      — override reward(reward) -> new_reward
└── ActionWrapper(Wrapper)      — override action(action) -> new_action
```

### Wrapper Pattern

```python
class MyWrapper(gymnasium.Wrapper):
    def __init__(self, env, my_param=1.0):
        super().__init__(env)
        self.my_param = my_param
        # optionally modify observation_space, action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward *= self.my_param  # modify reward
        return obs, reward, terminated, truncated, info
```

### ObservationWrapper Pattern

```python
class NormObs(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(-1, 1, shape=env.observation_space.shape)

    def observation(self, obs):
        return obs / 255.0  # only transform the observation
```

## Registration System

### register()

```python
gymnasium.register(
    id="namespace/EnvName-vN",         # required: unique ID
    entry_point="module.path:ClassName", # required: how to create
    reward_threshold=195.0,            # optional: "solved" threshold
    max_episode_steps=500,             # optional: auto-wraps TimeLimit
    order_enforce=True,                # optional: enforce reset-before-step
    disable_env_checker=False,         # optional: skip validation
    kwargs={"param": "default"},       # optional: default kwargs
    additional_wrappers=(              # optional: auto-applied wrappers
        WrapperSpec("name", "module:Cls", {"kwarg": val}),
    ),
    vector_entry_point="module:VecCls", # optional: custom vectorization
)
```

### make()

```python
env = gymnasium.make(
    "CartPole-v1",           # env ID or EnvSpec
    render_mode="human",     # passed to constructor
    max_episode_steps=1000,  # overrides registered value
    disable_env_checker=True,# skip PassiveEnvChecker
    **kwargs,                # forwarded to entry_point
)
```

**Wrapper application order** (inside to outside):
1. Entry point creates base env
2. `PassiveEnvChecker` (unless disabled)
3. `OrderEnforcing` (unless `order_enforce=False`)
4. `additional_wrappers` from EnvSpec
5. `TimeLimit` (if `max_episode_steps` set)
6. `RecordEpisodeStatistics` (if `render_mode` supports it)

### make_vec()

```python
vec_env = gymnasium.make_vec(
    "CartPole-v1",
    num_envs=4,
    vectorization_mode="sync",  # "sync" | "async" | "vector_entry_point"
    vector_kwargs={"shared_memory": True},
    wrappers=[lambda env: ClipAction(env)],
    **kwargs,
)
```

### ID Format

```
[namespace/]EnvName[-vN]
```

- `namespace`: optional, for third-party envs (e.g., `ALE/Breakout-v5`)
- `EnvName`: PascalCase name
- `-vN`: version number (integer)
- Examples: `CartPole-v1`, `phys2d/CartPole-v0`, `MyProject/CustomEnv-v2`

### Plugin System

Third-party packages register environments via entry points in `pyproject.toml`:

```toml
[project.entry-points."gymnasium.envs"]
__root__ = "my_package.envs"
```

The module's `register()` calls execute when Gymnasium loads.

## Standard RL Loop

```python
import gymnasium

env = gymnasium.make("CartPole-v1", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # or policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Utilities

### EzPickle

```python
class EzPickle:
    """Enables pickling via constructor args. Subclass and call super().__init__(*args, **kwargs)."""
```

### Seeding

```python
from gymnasium.utils.seeding import np_random
rng, seed = np_random(42)  # returns (Generator, int)
```

### RecordConstructorArgs

```python
class RecordConstructorArgs:
    """Records __init__ kwargs to _saved_kwargs for wrapper reproducibility."""
```

## Key Source Files

| File | Contents |
|------|----------|
| `gymnasium/core.py` | Env, Wrapper, ObservationWrapper, RewardWrapper, ActionWrapper |
| `gymnasium/envs/registration.py` | register(), make(), make_vec(), spec(), EnvSpec, WrapperSpec |
| `gymnasium/utils/seeding.py` | np_random(), RNG |
| `gymnasium/utils/ezpickle.py` | EzPickle |

## Reference Files

- [registration-reference.md](registration-reference.md) — Full EnvSpec/WrapperSpec field tables, register() full signature, make() wrapper order, plugin entry point format
