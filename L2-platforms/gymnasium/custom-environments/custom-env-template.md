# Custom Environment Templates

## Basic Environment Template

```python
import gymnasium
from gymnasium import spaces
import numpy as np


class BasicEnv(gymnasium.Env):
    """A minimal Gymnasium environment template.

    Observation: Box(low, high, shape)
    Action: Discrete(n)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, param1=1.0):
        super().__init__()
        self.param1 = param1
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self.np_random.standard_normal(4).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action):
        assert self.action_space.contains(action)

        # --- environment dynamics ---
        self._state += self.np_random.standard_normal(4).astype(np.float32) * 0.1

        observation = self._state.copy()
        reward = -float(np.sum(self._state ** 2))
        terminated = bool(np.any(np.abs(self._state) > 10))
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return None

    def close(self):
        pass
```

## Image Observation Environment Template

```python
import gymnasium
from gymnasium import spaces
import numpy as np


class ImageObsEnv(gymnasium.Env):
    """Environment with image observations (e.g., grid world with rendering)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(grid_size * 8, grid_size * 8, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)  # up, right, down, left

        self._agent_pos = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_pos = self.np_random.integers(0, self.grid_size, size=2)
        return self._render_obs(), {}

    def step(self, action):
        directions = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}
        self._agent_pos = np.clip(
            self._agent_pos + directions[action], 0, self.grid_size - 1
        )

        observation = self._render_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _render_obs(self):
        img = np.zeros((self.grid_size * 8, self.grid_size * 8, 3), dtype=np.uint8)
        r, c = self._agent_pos
        img[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = [255, 0, 0]
        return img

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_obs()
        return None

    def close(self):
        pass
```

## Dict Observation Environment Template

```python
import gymnasium
from gymnasium import spaces
import numpy as np


class DictObsEnv(gymnasium.Env):
    """Environment with Dict observation space (common for robotics)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "position": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
            "velocity": spaces.Box(-5, 5, shape=(3,), dtype=np.float32),
            "gripper_state": spaces.Discrete(2),
            "target": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

        self._pos = None
        self._vel = None
        self._target = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = self.np_random.uniform(-5, 5, size=3).astype(np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._target = self.np_random.uniform(-5, 5, size=3).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action):
        self._vel = np.clip(self._vel + action[:3] * 0.1, -5, 5).astype(np.float32)
        self._pos = np.clip(self._pos + self._vel * 0.1, -10, 10).astype(np.float32)

        dist = np.linalg.norm(self._pos - self._target)
        reward = -float(dist)
        terminated = bool(dist < 0.1)
        truncated = False
        info = {"distance": float(dist)}

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "position": self._pos.copy(),
            "velocity": self._vel.copy(),
            "gripper_state": 0,
            "target": self._target.copy(),
        }

    def render(self):
        return None

    def close(self):
        pass
```

## FuncEnv (JAX) Template

```python
import jax
import jax.numpy as jnp
from gymnasium.experimental.functional import FuncEnv
from gymnasium import spaces
import numpy as np


class MyFuncEnv(FuncEnv):
    """Stateless JAX-compatible functional environment.

    All methods are pure functions — no side effects, no self mutation.
    State is passed explicitly, enabling jit/vmap.
    """

    observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

    def initial(self, rng, params=None):
        """Return initial state."""
        return jax.random.normal(rng, shape=(4,))

    def transition(self, state, action, rng, params=None):
        """Compute next state (pure function)."""
        return state + action[0] * jnp.array([0.0, 0.1, 0.0, 0.1])

    def observation(self, state, rng, params=None):
        """Extract observation from state."""
        return state

    def reward(self, state, action, next_state, rng, params=None):
        """Compute scalar reward (pure function)."""
        return -jnp.sum(next_state ** 2)

    def terminal(self, state, rng, params=None):
        """Check if episode should terminate."""
        return jnp.any(jnp.abs(state) > 10.0)

    def state_info(self, state, params=None):
        """Optional: additional state info for debugging."""
        return {"energy": float(jnp.sum(state ** 2))}


# Wrap for standard Gymnasium usage:
# from gymnasium.experimental.functional_jax_env import FunctionalJaxEnv
# env = FunctionalJaxEnv(MyFuncEnv(), render_mode=None)
```

## Registration Snippet

```python
# In your_package/envs/__init__.py
from gymnasium import register

# Basic registration
register(
    id="myproject/BasicEnv-v0",
    entry_point="your_package.envs.basic_env:BasicEnv",
    max_episode_steps=200,
    reward_threshold=100.0,
    kwargs={"param1": 1.0},
)

# Image observation env
register(
    id="myproject/ImageEnv-v0",
    entry_point="your_package.envs.image_env:ImageObsEnv",
    max_episode_steps=500,
    kwargs={"grid_size": 8},
)

# Dict observation env with custom wrappers
from gymnasium.envs.registration import WrapperSpec

register(
    id="myproject/RobotEnv-v0",
    entry_point="your_package.envs.robot_env:DictObsEnv",
    max_episode_steps=1000,
    additional_wrappers=(
        WrapperSpec("FlattenObs", "gymnasium.wrappers:FlattenObservation", None),
    ),
)
```

## pyproject.toml Entry Point Template

```toml
[project]
name = "my-gym-envs"
version = "0.1.0"
description = "Custom Gymnasium environments"
requires-python = ">=3.9"
dependencies = [
    "gymnasium>=1.0",
    "numpy>=1.21",
]

[project.optional-dependencies]
rendering = ["pygame>=2.1"]

[project.entry-points."gymnasium.envs"]
myproject = "your_package.envs"

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"
```

After `pip install .`:

```python
import gymnasium
env = gymnasium.make("myproject/BasicEnv-v0")
```

## env_checker Usage

```python
from gymnasium.utils.env_checker import check_env

# Validate during development
env = BasicEnv(render_mode="rgb_array")
check_env(env)  # prints warnings, raises on critical errors

# Common issues check_env catches:
# - observation not in observation_space
# - step() returns wrong number of values
# - reset() doesn't return (obs, info) tuple
# - render() returns wrong type for render_mode
# - observation_space/action_space not set
# - metadata missing render_modes
```

## EzPickle Integration

```python
from gymnasium.utils import EzPickle

class MyEnv(gymnasium.Env, EzPickle):
    def __init__(self, param1=1.0, render_mode=None):
        EzPickle.__init__(self, param1=param1, render_mode=render_mode)
        super().__init__()
        # ... rest of init

# Now env is picklable (required for AsyncVectorEnv):
import pickle
env2 = pickle.loads(pickle.dumps(env))
```
