---
name: gymnasium-spaces
description: >
  Gymnasium space types for defining observation and action domains — Box, Discrete, Dict, Tuple, and composite spaces with sampling, containment, and flatten utilities.
layer: L2
domain: [general-rl]
source-project: Gymnasium
depends-on: [gymnasium-core-api]
tags: [spaces, box, discrete, dict, flatten]
---

# Gymnasium Spaces

## Purpose

Spaces define the valid structure and bounds for observations and actions. Every `Env` must set `observation_space` and `action_space` to a `Space` instance. Spaces support sampling, containment checking, serialization, and flattening.

## When to Use

- Defining observation/action domains for custom environments
- Sampling random actions for exploration
- Validating that observations/actions are within bounds
- Flattening complex spaces for neural network input
- Understanding what an environment expects/produces

## Space Base Class

```python
class Space(Generic[T_cov]):
    def __init__(self, shape=None, dtype=None, seed=None)

    def sample(self, mask=None, probability=None) -> T_cov
    def contains(self, x) -> bool
    def seed(self, seed=None) -> int | list[int] | dict[str, int]

    @property
    def shape(self) -> tuple[int, ...] | None
    @property
    def np_random(self) -> np.random.Generator
    @property
    def is_np_flattenable(self) -> bool

    def to_jsonable(self, sample_n) -> list
    def from_jsonable(self, sample_n) -> list
```

## Fundamental Spaces

### Box — Bounded continuous space

```python
Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
Box(low=np.array([0, -1]), high=np.array([1, 1]))
```

- `low`, `high`: scalar or array bounds (broadcast to shape)
- `dtype`: default `np.float32`
- `sample()`: uniform in [low, high] (or normal if unbounded)
- `is_bounded("both" | "below" | "above")`: check boundedness

### Discrete — Finite set {start, ..., start+n-1}

```python
Discrete(n=5)              # {0, 1, 2, 3, 4}
Discrete(n=3, start=-1)   # {-1, 0, 1}
```

- `n`: number of elements
- `start`: offset (default 0)
- `sample(mask=)`: binary mask to restrict sampling
- `dtype`: default `np.int64`

### MultiDiscrete — Product of Discrete spaces

```python
MultiDiscrete(nvec=[5, 3, 2])        # 3 independent discrete vars
MultiDiscrete(nvec=[3, 3], start=[1, 1])
```

- `nvec`: array of sizes per dimension
- `start`: per-dimension offsets (default all 0)
- `sample()` returns array of shape `nvec.shape`

### MultiBinary — Binary array

```python
MultiBinary(n=4)           # {0,1}^4
MultiBinary(n=[3, 2])      # {0,1}^(3x2)
```

- `n`: int or array shape
- `sample()` returns array of 0s and 1s

### Text — Variable-length string

```python
Text(min_length=1, max_length=10, charset=alphanumeric)
```

- `characters`: allowed character set (default alphanumeric)
- `min_length`, `max_length`: string length bounds

## Composite Spaces

### Dict — Named sub-spaces

```python
Dict({
    "position": Box(-10, 10, shape=(3,)),
    "velocity": Box(-1, 1, shape=(3,)),
    "gripper": Discrete(2),
})
```

- Keys are strings, values are any Space
- `sample()` returns `dict` with sampled values
- Supports `__getitem__`, iteration

### Tuple — Ordered sub-spaces

```python
Tuple((Discrete(5), Box(-1, 1, shape=(2,))))
```

- `sample()` returns `tuple` of sampled values
- Index-addressable

### Sequence — Variable-length sequences

```python
Sequence(Discrete(5))                       # variable-length
Sequence(Box(0, 1, shape=(3,)), seed=42)
```

- Feature space defines element type
- `sample()` returns `tuple` of variable length
- `stack`: whether to stack samples into arrays

### OneOf — Exclusive union

```python
OneOf([Discrete(3), Box(0, 1, shape=(2,))])
```

- Exactly one sub-space is active at a time
- `sample()` returns `(index, value)` tuple

### Graph — Graph-structured data

```python
Graph(
    node_space=Box(-1, 1, shape=(4,)),
    edge_space=Discrete(3),
)
```

- `sample(num_nodes=5, num_edges=8)` returns `GraphInstance`
- `GraphInstance(nodes, edges, edge_links)`

## Flatten Utilities

```python
from gymnasium.spaces.utils import flatten_space, flatten, unflatten, flatdim

flat_space = flatten_space(original_space)  # -> Box or MultiBinary
flat_obs = flatten(original_space, obs)      # -> 1D array
obs = unflatten(original_space, flat_obs)    # -> original structure
dim = flatdim(original_space)                # -> int (total flat size)
```

| Original Space | Flattened Space |
|---------------|-----------------|
| Box | Box (reshaped to 1D) |
| Discrete(n) | Box of one-hot (n,) |
| MultiDiscrete | Box of concatenated one-hots |
| MultiBinary | Box (reshaped to 1D) |
| Dict | Box (concatenated flattened values) |
| Tuple | Box (concatenated flattened values) |
| Text | Box (character indices) |

## Key Source Files

| File | Contents |
|------|----------|
| `gymnasium/spaces/space.py` | Space base class |
| `gymnasium/spaces/box.py` | Box |
| `gymnasium/spaces/discrete.py` | Discrete |
| `gymnasium/spaces/multi_discrete.py` | MultiDiscrete |
| `gymnasium/spaces/multi_binary.py` | MultiBinary |
| `gymnasium/spaces/text.py` | Text |
| `gymnasium/spaces/dict.py` | Dict |
| `gymnasium/spaces/tuple.py` | Tuple |
| `gymnasium/spaces/sequence.py` | Sequence |
| `gymnasium/spaces/oneof.py` | OneOf |
| `gymnasium/spaces/graph.py` | Graph, GraphInstance |
| `gymnasium/spaces/utils.py` | flatten_space, flatten, unflatten, flatdim |

## Reference Files

- [spaces-catalog.md](spaces-catalog.md) — Full constructor signatures, parameter tables, dtype defaults, sample() return types for all space types
