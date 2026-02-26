# Configclass API Reference

Detailed reference for the `@configclass` decorator, its methods, inheritance rules, and AppLauncher.

## @configclass Decorator Internals

**File:** `source/isaaclab/isaaclab/utils/configclass.py`

The decorator wraps `dataclasses.dataclass` and adds:
1. Auto type-annotation from default values
2. Mutable default deep-copying (lists, dicts, nested configs)
3. Five utility methods: `to_dict`, `from_dict`, `replace`, `copy`, `validate`

### Method Signatures

```python
def to_dict(self) -> dict:
    """Convert config to nested dictionary. Nested @configclass instances become nested dicts."""

def from_dict(self, data: dict) -> None:
    """Update config in-place from a dictionary. Modifies existing instance."""

def replace(self, **kwargs) -> Self:
    """Return a NEW instance with specified fields replaced. Original is unchanged."""

def copy(self) -> Self:
    """Return a deep copy of this config instance."""

def validate(self) -> None:
    """Check for MISSING sentinel values. Raises TypeError listing all unset fields."""
```

### Inheritance Rules

1. Child `@configclass` inherits all parent fields
2. Child can override parent defaults
3. `class_type` convention: parent declares `class_type: type = MISSING`, child sets it
4. Multiple inheritance is supported (use with care)
5. `__post_init__` is called after `__init__` (standard dataclass behavior)

```python
@configclass
class BaseSensorCfg:
    class_type: type = MISSING
    prim_path: str = MISSING
    update_period: float = 0.0

@configclass
class CameraCfg(BaseSensorCfg):
    class_type: type = Camera          # Override with concrete type
    width: int = MISSING               # Add new field
    height: int = MISSING
    data_types: list[str] = ["rgb"]    # Mutable default is safe
```

### Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Forgetting `@configclass` | Fields won't have added methods | Always add decorator |
| Using `MISSING` without import | `MISSING` is not defined | `from dataclasses import MISSING` |
| Mutating shared defaults | N/A — auto deep-copy handles this | No action needed |
| Nested `replace()` | `cfg.replace(scene.num_envs=64)` fails | `cfg.replace(scene=cfg.scene.replace(num_envs=64))` |
| `__post_init__` with MISSING | Accessing MISSING field in `__post_init__` | Guard with `if self.field is not MISSING` |

### to_dict / from_dict Roundtrip

```python
@configclass
class MyCfg:
    val: int = 10
    nested: SubCfg = SubCfg()

cfg = MyCfg()
d = cfg.to_dict()
# d = {"val": 10, "nested": {"sub_val": ...}}

cfg2 = MyCfg()
cfg2.from_dict(d)  # Updates cfg2 in-place
```

### class_type Convention

Many IsaacLab subsystems use `class_type` to bind a config to its implementation:

```python
@configclass
class ActuatorBaseCfg:
    class_type: type = MISSING

@configclass
class ImplicitActuatorCfg(ActuatorBaseCfg):
    class_type: type = ImplicitActuator

# Framework instantiates: cfg.class_type(cfg, ...)
```

Systems using this pattern:
- `ActuatorBaseCfg` → all actuator types
- `SensorBaseCfg` → all sensor types
- `AssetBaseCfg` → Articulation, RigidObject, etc.
- `SpawnerCfg` → all spawner types
- `CommandTermCfg` → all command term types
- `ActionTermCfg` → all action term types
- `RecorderTermCfg` → all recorder term types
- `NoiseModelCfg` → noise model types
- `ModifierCfg` → modifier types
- `TerrainImporterCfg` → terrain importer
- `TerrainGeneratorCfg` → terrain generator

## AppLauncher Reference

**File:** `source/isaaclab/isaaclab/app/app_launcher.py`

### Initialization Pattern

```python
import argparse
from isaaclab.app import AppLauncher

# Option 1: With argparse
parser = argparse.ArgumentParser()
parser.add_argument("--my_arg", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Option 2: With dict
app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app

# Option 3: With kwargs
app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app
```

### CLI Flags (via add_app_launcher_args)

| Flag | Type | Default | Env Var | Description |
|------|------|---------|---------|-------------|
| `--headless` | bool | False | `HEADLESS` | Run without GUI |
| `--device` | str | `"cuda:0"` | — | Compute device (`"cpu"`, `"cuda"`, `"cuda:N"`) |
| `--livestream` | int | -1 | `LIVESTREAM` | WebRTC streaming (0=off, 1=public, 2=private) |
| `--enable_cameras` | bool | False | `ENABLE_CAMERAS` | Enable cameras in headless mode |
| `--experience` | str | `""` | `EXPERIENCE` | Kit experience file to load |
| `--rendering_mode` | str | `"balanced"` | — | `"performance"`, `"balanced"`, `"quality"` |
| `--xr` | bool | False | — | Enable XR mode |
| `--verbose` | bool | False | — | Verbose logging |
| `--info` | bool | False | — | Info-level logging |
| `--kit_args` | str | `""` | — | Additional Kit CLI args |

### Distributed Training Environment Variables

| Variable | Description |
|----------|-------------|
| `DEVICE_ID` | GPU device ID |
| `LOCAL_RANK` | Local process rank (multi-GPU) |
| `GLOBAL_RANK` | Global process rank (multi-node) |

### Properties

```python
app_launcher.app          # SimulationApp instance
app_launcher.local_rank   # Local rank for distributed training
app_launcher.global_rank  # Global rank for distributed training
```

### Critical Rule

**All Isaac Sim imports must come AFTER AppLauncher initialization:**

```python
# CORRECT
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
# Now safe to import
from isaaclab.envs import ManagerBasedRLEnv

# WRONG - will crash
from isaaclab.envs import ManagerBasedRLEnv  # Isaac Sim not initialized!
from isaaclab.app import AppLauncher
```
