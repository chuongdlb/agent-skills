---
name: gymnasium-custom-environments
description: >
  Building custom Gymnasium environments — Env subclass pattern, render modes, FuncEnv stateless interface, registration, packaging, and env_checker validation.
layer: L2
domain: [general-rl]
source-project: Gymnasium
depends-on: [gymnasium-core-api, gymnasium-spaces]
tags: [custom-env, registration, funcenv, packaging, templates]
---

# Gymnasium Custom Environments

## Purpose

Guides the creation of custom RL environments that conform to the Gymnasium API. Covers the standard `Env` subclass pattern, functional (JAX-compatible) environments, registration for `gymnasium.make()`, packaging for distribution, and validation with `env_checker`.

## When to Use

- Creating a new RL environment from scratch
- Wrapping an existing simulator as a Gymnasium environment
- Building JAX-compilable environments with `FuncEnv`
- Packaging environments for distribution via pip
- Validating environment correctness

## Step-by-Step: Standard Env Subclass

### 1. Define the Class

```python
import gymnasium
from gymnasium import spaces
import numpy as np

class MyEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=5):
        super().__init__()
        self.size = size
        self.render_mode = render_mode

        # Define spaces
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Internal state (not part of API)
        self._agent_location = None
        self._target_location = None
```

### 2. Implement reset()

```python
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random

        self._agent_location = self.np_random.integers(0, self.size, size=2).astype(np.float32)
        self._target_location = self.np_random.integers(0, self.size, size=2).astype(np.float32)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
```

### 3. Implement step()

```python
    def step(self, action):
        direction = {0: [1, 0], 1: [-1, 0], 2: [0, 1], 3: [0, -1]}[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        ).astype(np.float32)

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1.0 if terminated else 0.0
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
```

### 4. Implement render() and close()

```python
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()  # return numpy array (H, W, 3) uint8
        elif self.render_mode == "human":
            self._render_human()  # display to screen (e.g., pygame)

    def close(self):
        # Clean up resources (pygame windows, connections, etc.)
        pass
```

### 5. Helper Methods

```python
    def _get_obs(self):
        return self._agent_location.copy()

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location)}
```

## Render Modes

| Mode | render() Returns | Use Case |
|------|-----------------|----------|
| `"human"` | `None` | Display to screen (pygame, OpenGL) |
| `"rgb_array"` | `NDArray[uint8]` shape `(H, W, 3)` | Video recording, headless |
| `"ansi"` | `str` | Text-based rendering |
| `"rgb_array_list"` | `list[NDArray]` | Accumulated frames |

- Set `metadata["render_modes"]` to advertise supported modes
- Set `metadata["render_fps"]` for video timing
- `render_mode` is set once at construction, not per-call

## FuncEnv — Stateless Functional Interface

For JAX-compilable environments that can be `jit`-compiled and `vmap`-vectorized.

```python
from gymnasium.experimental.functional import FuncEnv
import jax.numpy as jnp

class MyFuncEnv(FuncEnv):
    observation_space = spaces.Box(-1, 1, (4,))
    action_space = spaces.Box(-1, 1, (1,))

    def initial(self, rng, params=None):
        """Return initial state."""
        return jnp.zeros(4)

    def transition(self, state, action, rng, params=None):
        """Return next state (pure function, no side effects)."""
        return state + action * 0.1

    def observation(self, state, rng, params=None):
        """Extract observation from state."""
        return state

    def reward(self, state, action, next_state, rng, params=None):
        """Compute scalar reward."""
        return -jnp.sum(next_state ** 2)

    def terminal(self, state, rng, params=None):
        """Return whether episode is done."""
        return jnp.any(jnp.abs(state) > 10)
```

### Wrapping FuncEnv for Standard Use

```python
from gymnasium.experimental.functional_jax_env import FunctionalJaxEnv

env = FunctionalJaxEnv(MyFuncEnv(), render_mode="rgb_array")
# Now usable with standard gymnasium.Env interface
```

## Registration

### Basic Registration

```python
gymnasium.register(
    id="myproject/MyEnv-v0",
    entry_point="my_package.envs:MyEnv",
    max_episode_steps=200,
    reward_threshold=100.0,
    kwargs={"size": 10},
)
```

### Using the Registered Environment

```python
env = gymnasium.make("myproject/MyEnv-v0", size=20)  # kwargs override defaults
```

## Packaging for Distribution

### Project Structure

```
my-gym-envs/
├── pyproject.toml
└── my_package/
    ├── __init__.py
    └── envs/
        ├── __init__.py      # register() calls here
        └── my_env.py        # Env subclass
```

### pyproject.toml Entry Points

```toml
[project]
name = "my-gym-envs"
version = "0.1.0"
dependencies = ["gymnasium>=1.0"]

[project.entry-points."gymnasium.envs"]
my_namespace = "my_package.envs"
```

### envs/__init__.py

```python
from gymnasium import register

register(
    id="my_namespace/MyEnv-v0",
    entry_point="my_package.envs.my_env:MyEnv",
    max_episode_steps=200,
)
```

After `pip install my-gym-envs`, the environment is available:

```python
env = gymnasium.make("my_namespace/MyEnv-v0")
```

## EzPickle for Reproducibility

```python
from gymnasium.utils import EzPickle

class MyEnv(gymnasium.Env, EzPickle):
    def __init__(self, size=5, render_mode=None):
        EzPickle.__init__(self, size=size, render_mode=render_mode)
        # ... rest of __init__
```

- Enables `pickle.dumps(env)` / `pickle.loads()` using constructor args
- Required for `AsyncVectorEnv` (environments must be picklable)

## env_checker Validation

```python
from gymnasium.utils.env_checker import check_env

env = MyEnv()
check_env(env)  # raises warnings/errors for API violations
```

Checks include:
- `observation_space` and `action_space` are valid Space instances
- `reset()` returns `(obs, info)` with obs in observation_space
- `step()` returns correct 5-tuple with correct types
- `render()` returns correct type for render_mode
- `metadata` is properly structured
- Seeding produces reproducible results

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Not calling `super().reset(seed=seed)` | Always call super to seed `np_random` |
| Returning mutable observation references | Return `obs.copy()` to prevent aliasing |
| Missing `render_mode` in constructor | Accept and store `render_mode` parameter |
| Not setting `metadata["render_modes"]` | List all supported render modes |
| Using `random` module instead of `self.np_random` | Use `self.np_random` for reproducibility |
| Forgetting to call `close()` cleanup | Implement `close()` for resource cleanup |

## Key Source Files

| File | Contents |
|------|----------|
| `gymnasium/core.py` | Env, Wrapper base classes |
| `gymnasium/experimental/functional.py` | FuncEnv |
| `gymnasium/experimental/functional_jax_env.py` | FunctionalJaxEnv adapter |
| `gymnasium/utils/env_checker.py` | check_env() validation |
| `gymnasium/utils/ezpickle.py` | EzPickle |

## Reference Files

- [custom-env-template.md](custom-env-template.md) — Copy-paste templates for basic env, image-obs env, Dict-obs env, FuncEnv, registration snippet, pyproject.toml
