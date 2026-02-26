# Registration System Reference

## EnvSpec Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | (required) | Environment ID in `[namespace/]Name[-vN]` format |
| `entry_point` | `str \| Callable` | `None` | Module path `"mod:Class"` or callable |
| `reward_threshold` | `float \| None` | `None` | Reward threshold for "solved" |
| `nondeterministic` | `bool` | `False` | Whether env is stochastic even with same seed |
| `max_episode_steps` | `int \| None` | `None` | Auto-wraps with `TimeLimit` if set |
| `order_enforce` | `bool` | `True` | Auto-wraps with `OrderEnforcing` |
| `disable_env_checker` | `bool` | `False` | Skip `PassiveEnvChecker` wrapper |
| `kwargs` | `dict` | `{}` | Default kwargs passed to entry_point |
| `additional_wrappers` | `tuple[WrapperSpec, ...]` | `()` | Wrappers applied after env creation |
| `vector_entry_point` | `str \| Callable \| None` | `None` | Custom vectorized implementation |

### Derived Fields (set automatically)

| Field | Type | Description |
|-------|------|-------------|
| `namespace` | `str \| None` | Parsed from `id` |
| `name` | `str` | Parsed from `id` |
| `version` | `int \| None` | Parsed from `id` |

### EnvSpec Methods

```python
spec.make(**kwargs)                    # Create env from this spec
spec.to_json() -> str                 # Serialize to JSON
EnvSpec.from_json(json_str) -> EnvSpec # Deserialize
spec.pprint(                          # Pretty-print
    disable_print=False,
    include_entry_points=False,
    print_all=False,
) -> str | None
```

## WrapperSpec Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Wrapper display name |
| `entry_point` | `str` | Module path `"module:WrapperClass"` |
| `kwargs` | `dict[str, Any] \| None` | Constructor kwargs |

### Creating WrapperSpec from a Wrapper Class

```python
# Class method on Wrapper
MyWrapper.wrapper_spec(param1=val1) -> WrapperSpec
```

## register() Full Signature

```python
gymnasium.register(
    id: str,
    entry_point: EnvCreator | str | None = None,
    reward_threshold: float | None = None,
    nondeterministic: bool = False,
    max_episode_steps: int | None = None,
    order_enforce: bool = True,
    disable_env_checker: bool = False,
    additional_wrappers: tuple[WrapperSpec, ...] = (),
    vector_entry_point: VectorEnvCreator | str | None = None,
    kwargs: dict | None = None,
) -> None
```

## make() Wrapper Application Order

When `gymnasium.make(id, **kwargs)` is called, wrappers are applied inside-out:

```
OuterWrapper  ← 6. RecordEpisodeStatistics (if render_mode set)
  ↓
TimeLimit     ← 5. TimeLimit (if max_episode_steps set)
  ↓
Additional    ← 4. additional_wrappers from EnvSpec (in order)
  ↓
OrderEnforcing← 3. OrderEnforcing (if order_enforce=True)
  ↓
EnvChecker    ← 2. PassiveEnvChecker (unless disabled)
  ↓
BaseEnv       ← 1. entry_point(**kwargs) creates base env
```

## make_vec() Full Signature

```python
gymnasium.make_vec(
    id: str | EnvSpec,
    num_envs: int = 1,
    vectorization_mode: VectorizeMode | str | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
    **kwargs,
) -> VectorEnv
```

### VectorizeMode Enum

| Value | Description |
|-------|-------------|
| `"sync"` | `SyncVectorEnv` — serial execution |
| `"async"` | `AsyncVectorEnv` — multiprocessing |
| `"vector_entry_point"` | Use `vector_entry_point` from EnvSpec |

## Plugin Entry Point Format

In `pyproject.toml`:

```toml
[project.entry-points."gymnasium.envs"]
__root__ = "my_package.envs"
# OR for a namespace:
my_namespace = "my_package.envs"
```

The referenced module must call `gymnasium.register()` for each environment at import time.

### Namespace Context Manager

```python
with gymnasium.envs.registration.namespace("my_ns"):
    gymnasium.register(id="MyEnv-v0", entry_point="mod:Cls")
    # Registered as "my_ns/MyEnv-v0"
```

## Utility Functions

```python
gymnasium.spec("CartPole-v1") -> EnvSpec
gymnasium.pprint_registry(num_cols=3)
gymnasium.envs.registration.parse_env_id("ns/Name-v1") -> (ns, name, version)
gymnasium.envs.registration.get_env_id(ns, name, version) -> str
gymnasium.envs.registration.find_highest_version(ns, name) -> int | None
```

## Global Registry

```python
gymnasium.envs.registration.registry: dict[str, EnvSpec]
# All registered environments, keyed by ID
```
