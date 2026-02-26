# Spaces Catalog — Full Signatures and Details

## Box

```python
Box(
    low: SupportsFloat | NDArray,    # lower bound (scalar or array)
    high: SupportsFloat | NDArray,   # upper bound (scalar or array)
    shape: Sequence[int] | None = None,  # inferred from low/high if None
    dtype: type = np.float32,        # element data type
    seed: int | Generator | None = None,
)
```

| Property | Type |
|----------|------|
| `low` | `NDArray` — lower bounds |
| `high` | `NDArray` — upper bounds |
| `shape` | `tuple[int, ...]` |
| `dtype` | `np.dtype` (default `float32`) |
| `is_np_flattenable` | `True` |

- `sample()` → `NDArray[dtype]` uniform in [low, high]
- For unbounded dims (±inf), samples from normal distribution
- `is_bounded("both" | "below" | "above")` → `bool`
- `contains(x)` checks shape, dtype, and bounds

## Discrete

```python
Discrete(
    n: int | np.integer,             # number of values
    seed: int | Generator | None = None,
    start: int | np.integer = 0,     # offset
    dtype: type = np.int64,
)
```

| Property | Type |
|----------|------|
| `n` | `int` — count of values |
| `start` | `int` — first value |
| `shape` | `()` |
| `dtype` | `np.int64` |
| `is_np_flattenable` | `True` |

- `sample()` → `np.int64` in {start, ..., start+n-1}
- `sample(mask=np.array([1,0,1,0,1]))` — binary mask restricts choices
- `sample(probability=np.array([0.5,0.1,0.1,0.1,0.2]))` — weighted sampling

## MultiDiscrete

```python
MultiDiscrete(
    nvec: NDArray[np.integer],       # sizes per dimension
    seed: int | Generator | None = None,
    start: NDArray[np.integer] | None = None,  # per-dim offsets
    dtype: type = np.int64,
)
```

| Property | Type |
|----------|------|
| `nvec` | `NDArray[np.integer]` |
| `start` | `NDArray[np.integer]` |
| `shape` | `nvec.shape` |
| `dtype` | `np.int64` |
| `is_np_flattenable` | `True` |

- `sample()` → `NDArray[int64]` with shape `nvec.shape`
- `sample(mask=)` — tuple of masks per dimension

## MultiBinary

```python
MultiBinary(
    n: int | Sequence[int],          # shape
    seed: int | Generator | None = None,
)
```

| Property | Type |
|----------|------|
| `n` | `int \| list` |
| `shape` | `(n,)` or `tuple(n)` |
| `dtype` | `np.int8` |
| `is_np_flattenable` | `True` |

- `sample()` → `NDArray[int8]` of 0s and 1s
- `sample(mask=)` — array of 0/1/2 (force 0, force 1, random)

## Text

```python
Text(
    min_length: int = 1,
    max_length: int = 1,
    charset: str = alphanumeric,     # string of allowed characters
    seed: int | Generator | None = None,
)
```

| Property | Type |
|----------|------|
| `min_length` | `int` |
| `max_length` | `int` |
| `characters` | `frozenset[str]` |
| `character_set` | `str` — sorted character string |
| `shape` | `()` |
| `dtype` | `str` |
| `is_np_flattenable` | `True` |

- `sample()` → `str` of random length in [min_length, max_length]
- `sample(mask=)` — tuple of `(length, char_mask)` or `(length, None)`

## Dict

```python
Dict(
    spaces: dict[str, Space] | None = None,
    seed: int | dict | None = None,
    **spaces_kwargs: Space,
)
```

| Property | Type |
|----------|------|
| `spaces` | `dict[str, Space]` |
| `shape` | `()` |
| `dtype` | `None` |
| `is_np_flattenable` | `True` (if all sub-spaces are) |

- `sample()` → `dict[str, Any]`
- `sample(mask={key: sub_mask})` — per-key masks
- `__getitem__(key)` → sub-space
- `__iter__()` → iterate over keys

## Tuple

```python
Tuple(
    spaces: Sequence[Space],
    seed: int | Sequence | None = None,
)
```

| Property | Type |
|----------|------|
| `spaces` | `tuple[Space, ...]` |
| `shape` | `()` |
| `dtype` | `None` |
| `is_np_flattenable` | `True` (if all sub-spaces are) |

- `sample()` → `tuple[Any, ...]`
- `sample(mask=(sub_mask1, sub_mask2))` — per-position masks
- `__getitem__(idx)` → sub-space

## Sequence

```python
Sequence(
    space: Space,
    seed: int | Generator | None = None,
    stack: bool = False,
)
```

| Property | Type |
|----------|------|
| `feature_space` | `Space` |
| `stack` | `bool` |
| `shape` | `()` |
| `dtype` | `None` |
| `is_np_flattenable` | `False` |

- `sample()` → `tuple[Any, ...]` of variable length (geometric distribution)
- `sample(mask=(length, sub_mask | None))` — fixed length sampling

## OneOf

```python
OneOf(
    spaces: Sequence[Space],
    seed: int | Sequence | None = None,
)
```

| Property | Type |
|----------|------|
| `spaces` | `tuple[Space, ...]` |
| `shape` | `()` |
| `dtype` | `None` |
| `is_np_flattenable` | `False` |

- `sample()` → `tuple[int, Any]` — `(space_index, sampled_value)`
- `sample(mask=(space_mask, sub_masks))` — restrict which spaces

## Graph

```python
Graph(
    node_space: Box | Discrete,
    edge_space: Box | Discrete | None = None,
    seed: int | Generator | None = None,
)
```

| Property | Type |
|----------|------|
| `node_space` | `Box \| Discrete` |
| `edge_space` | `Box \| Discrete \| None` |
| `shape` | `()` |
| `dtype` | `None` |
| `is_np_flattenable` | `False` |

- `sample(num_nodes=10, num_edges=None)` → `GraphInstance`
- `GraphInstance.nodes` — `NDArray` shape `(num_nodes, *node_shape)`
- `GraphInstance.edges` — `NDArray | None`
- `GraphInstance.edge_links` — `NDArray` shape `(num_edges, 2)`

## Flatten Utility Reference

```python
flatten_space(space: Space) -> Box | MultiBinary | Dict | Tuple
flatten(space: Space, x: T) -> NDArray
unflatten(space: Space, x: NDArray) -> T
flatdim(space: Space) -> int
```

### Flat Dimensions by Space Type

| Space | flatdim |
|-------|---------|
| `Box(shape=(3,4))` | `12` |
| `Discrete(5)` | `5` (one-hot) |
| `MultiDiscrete([3,4])` | `7` (sum of one-hots) |
| `MultiBinary(4)` | `4` |
| `Dict({"a": Box(3), "b": Discrete(2)})` | `5` |
| `Tuple((Box(3), Discrete(2)))` | `5` |
| `Text(max_length=5)` | varies |
