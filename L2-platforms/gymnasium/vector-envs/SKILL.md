---
name: gymnasium-vector-envs
description: >
  Gymnasium vector environment API — SyncVectorEnv, AsyncVectorEnv, batched step/reset semantics, autoreset modes, and vector wrappers for parallel RL.
layer: L2
domain: [general-rl]
source-project: Gymnasium
depends-on: [gymnasium-core-api, gymnasium-spaces]
tags: [vectorization, parallel, batched, async, sync]
---

# Gymnasium Vector Environments

## Purpose

Vector environments run multiple independent environment copies in parallel, producing batched observations and accepting batched actions. This is essential for efficient RL training — most modern algorithms (PPO, SAC) expect vectorized environments.

## When to Use

- Training RL agents that benefit from parallel environment rollouts
- Collecting experience from multiple environments simultaneously
- Comparing `SyncVectorEnv` (serial) vs `AsyncVectorEnv` (multiprocessing)
- Configuring autoreset behavior for continuous training
- Applying batch-level observation/reward normalization

## Creating Vector Environments

### Via make_vec()

```python
vec_env = gymnasium.make_vec("CartPole-v1", num_envs=8, vectorization_mode="sync")
vec_env = gymnasium.make_vec("HalfCheetah-v5", num_envs=16, vectorization_mode="async")
```

### Explicit Construction

```python
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

# Sync (serial for-loop, good for debugging)
vec_env = SyncVectorEnv([lambda: gymnasium.make("CartPole-v1") for _ in range(4)])

# Async (multiprocessing, good for CPU-heavy envs)
vec_env = AsyncVectorEnv(
    [lambda: gymnasium.make("CartPole-v1") for _ in range(8)],
    shared_memory=True,
)
```

## VectorEnv Base Class

```python
class VectorEnv(Generic[ObsType, ActType, ArrayType]):
    num_envs: int                          # number of parallel envs
    single_observation_space: Space        # space for ONE env
    single_action_space: Space             # space for ONE env
    observation_space: Space               # batched space (num_envs copies)
    action_space: Space                    # batched space (num_envs copies)
```

### Step Semantics

```python
obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
# obs: batched observations, shape (num_envs, *obs_shape)
# rewards: array shape (num_envs,)
# terminateds: bool array shape (num_envs,)
# truncateds: bool array shape (num_envs,)
# infos: dict of arrays
```

### Reset Semantics

```python
obs, infos = vec_env.reset(seed=42, options={})
# obs: batched observations for all envs
# seed: can be int (sequential seeding) or list[int]
```

## SyncVectorEnv

```python
SyncVectorEnv(
    env_fns: Sequence[Callable[[], Env]],
    copy: bool = True,                      # deep copy observations
    observation_mode: str | Space = "same",  # "same" or "different"
    autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP,
)
```

- Runs environments sequentially in a for-loop
- Best for: debugging, GPU-accelerated envs, fast envs where multiprocessing overhead is not worth it
- `copy=True`: prevents observation aliasing bugs
- `observation_mode="different"`: allows envs with different observation spaces

## AsyncVectorEnv

```python
AsyncVectorEnv(
    env_fns: Sequence[Callable[[], Env]],
    shared_memory: bool = True,     # use shared memory for obs transfer
    copy: bool = True,              # deep copy observations
    context: str | None = None,     # multiprocessing context ("fork", "spawn", "forkserver")
    daemon: bool = True,            # daemon worker processes
    worker: Callable | None = None, # custom worker function
    observation_mode: str | Space = "same",
    autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP,
)
```

- Runs environments in separate processes via `multiprocessing`
- Best for: CPU-heavy envs, physics simulations, many environments
- `shared_memory=True`: zero-copy observation transfer (faster but requires Box-like spaces)
- `context="spawn"`: safest on all platforms; `"fork"` fastest on Linux

## AutoresetMode

Controls what happens when an individual environment terminates or is truncated.

| Mode | Behavior |
|------|----------|
| `NEXT_STEP` | On the *next* `step()` call, the terminated env returns the reset observation. `info["final_observation"]` and `info["final_info"]` contain the terminal state. |
| `SAME_STEP` | Immediately resets in the same `step()` call. Returned obs is already the reset obs. |
| `DISABLED` | No auto-reset. User must manually reset terminated environments. |

### NEXT_STEP Pattern (default)

```python
obs, infos = vec_env.reset(seed=42)
for step in range(total_steps):
    actions = policy(obs)
    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

    # For terminated/truncated envs, obs is already the reset obs
    # The TRUE final observation is in:
    if "final_observation" in infos:
        for i, final_obs in enumerate(infos["final_observation"]):
            if final_obs is not None:
                # This was the actual last observation before reset
                pass
```

## Vector Wrappers

Vector wrappers operate on the batched level, modifying all environments simultaneously.

| Wrapper | What It Does |
|---------|-------------|
| `VectorizeTransformObservation(env, wrapper, ...)` | Apply single-env obs wrapper to vector env |
| `VectorizeTransformAction(env, wrapper, ...)` | Apply single-env action wrapper to vector env |
| `VectorizeTransformReward(env, wrapper, ...)` | Apply single-env reward wrapper to vector env |

### Vectorized Normalization

For observation and reward normalization across the batch:

```python
from gymnasium.wrappers.vector import (
    NormalizeObservation,   # running stats across all envs
    NormalizeReward,        # running return stats across all envs
)

vec_env = NormalizeObservation(vec_env)
vec_env = NormalizeReward(vec_env, gamma=0.99)
```

## Space Batching

```python
from gymnasium.vector.utils import batch_space, iterate

# batch_space creates the vectorized version of a space
batched = batch_space(single_space, n=4)
# Box(shape=(3,)) → Box(shape=(4, 3))
# Discrete(5) → MultiDiscrete([5, 5, 5, 5])

# iterate yields individual items from batched data
for single_obs in iterate(batched, batched_obs):
    pass
```

## Key Source Files

| File | Contents |
|------|----------|
| `gymnasium/vector/vector_env.py` | VectorEnv base class |
| `gymnasium/vector/sync_vector_env.py` | SyncVectorEnv |
| `gymnasium/vector/async_vector_env.py` | AsyncVectorEnv |
| `gymnasium/vector/utils/` | batch_space, iterate, shared memory utilities |
| `gymnasium/wrappers/vector/` | Vector-level wrappers |
