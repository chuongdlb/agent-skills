# extension.toml Reference

Complete annotated schema for Isaac Sim extension manifests. Derived from real extensions: `isaacsim.core.api`, `isaacsim.sensors.camera`, `isaacsim.ros2.bridge`.

## [package] Section

```toml
[package]
version = "1.0.0"                           # Semantic version (required)
category = "Simulation"                     # Category: Simulation, SyntheticData, Utility
title = "My Extension Title"                # Human-readable title
description = "Brief description."          # Extension description
keywords = ["isaac", "robotics"]            # Searchable keywords
changelog = "docs/CHANGELOG.md"             # Path to changelog
readme = "docs/README.md"                   # Path to readme
preview_image = "data/preview.png"          # Preview image for extension browser
icon = "data/icon.png"                      # Extension icon
writeTarget.kit = true                      # Include in Kit build target
writeTarget.platform = true                 # OS-specific build (used for native plugins)
```

## [dependencies] Section

```toml
[dependencies]
# Required dependencies (extension won't load without these)
"isaacsim.core.api" = {}
"omni.graph" = {}
"omni.physx.tensors" = {}

# Optional dependencies (extension loads even if these are missing)
"omni.kit.material.library" = {optional = true}

# Version-constrained dependency
"omni.kit.pip_archive" = {}
```

Common Isaac Sim dependencies:
- `isaacsim.core.api` - Core simulation API (World, Scene, SimulationContext)
- `isaacsim.core.prims` - Prim wrapper classes
- `isaacsim.core.utils` - Utility functions
- `isaacsim.core.nodes` - Core OmniGraph nodes
- `isaacsim.core.deprecation_manager` - Handles deprecated node/setting migration
- `isaacsim.robot.schema` - Robot USD schema
- `isaacsim.storage.native` - Asset root path resolution
- `omni.graph` - OmniGraph framework
- `omni.replicator.core` - Replicator/SDG framework
- `omni.syntheticdata` - Synthetic data annotators
- `omni.physx` / `omni.physics.physx` - PhysX integration
- `omni.physx.tensors` - Tensor-based physics API
- `omni.pip.compute` - Pip packages (scipy, etc.)

## [[python.module]] Section

```toml
# Main Python module (required)
[[python.module]]
name = "isaacsim.my_domain.my_feature"

# Test module (set public = false to hide from public API)
[[python.module]]
name = "isaacsim.my_domain.my_feature.tests"
public = false

# Additional sample modules
[[python.module]]
name = "isaacsim.my_domain.my_feature.impl.samples.my_sample"
```

The module `name` must match the Python package directory structure under `python/`.

## [[native.plugin]] Section

For extensions with C++ plugins:

```toml
[[native.plugin]]
path = "bin/*.plugin"
recursive = false
```

## [settings] Section

Define default extension settings accessible via `carb.settings`:

```toml
[settings]
exts."isaacsim.my_domain.my_feature".my_setting = "default_value"
exts."isaacsim.my_domain.my_feature".my_flag = false
exts."isaacsim.my_domain.my_feature".my_number = 42
```

Read in Python:
```python
settings = carb.settings.get_settings()
value = settings.get("/exts/isaacsim.my_domain.my_feature/my_setting")
```

Real example from `isaacsim.ros2.bridge`:
```toml
[settings]
exts."isaacsim.ros2.bridge".ros_distro = "system_default"
exts."isaacsim.ros2.bridge".publish_without_verification = false
exts."isaacsim.ros2.bridge".enable_nitros_bridge = false
```

## [fswatcher.patterns] Section

File system watcher configuration for hot-reload during development:

```toml
[fswatcher.patterns]
include = ["*.ogn", "*.py", "*.toml"]
exclude = ["Ogn*Database.py"]

[fswatcher.paths]
exclude = ["*/bin", "*/__pycache__/*", "*/.git/*"]
```

## [[test]] Section

Test configuration for `repo test`:

```toml
[[test]]
timeout = 900                               # Test timeout in seconds

dependencies = [                            # Additional extensions needed for tests
    "isaacsim.core.cloner",
    "omni.kit.renderer.core",
    "isaacsim.test.utils",
]

# Standard test runtime arguments (copy this block for new extensions)
args = [
    "--enable", "omni.kit.loop-isaac",
    "--reset-user",
    "--vulkan",
    "--/app/asyncRendering=false",
    "--/app/asyncRenderingLowLatency=false",
    "--/app/file/ignoreUnsavedOnExit=true",
    "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
    "--/app/settings/persistent=false",
    "--/app/settings/persistent/physics/updateToUsd=false",
    "--/app/settings/persistent/physics/useFastCache=false",
    "--/app/settings/persistent/physics/numThreads=1",
    "--/app/settings/persistent/physics/updateTransformationsFromSDK=false",
    "--/app/settings/persistent/physics/updateVelocitiesToUsd=false",
    "--/app/settings/persistent/simulation/minFrameRate=15",
    "--/persistent/isaac/asset_root/default=/isaac-sim-assets/default",
    "--/persistent/isaac/asset_root/nvidia=/isaac-sim-assets/nvidia",
    "--/app/renderer/resolution/width=64",
    "--/app/renderer/resolution/height=64",
    "--/persistent/app/omniverse/gamepadCameraControl=false",
    "--no-window",
]

# Exclude known error patterns from test failure detection
stdoutFailPatterns.exclude = [
    "*[Error] [carb] [Plugin: ...] Dependency: [...] failed*",
    "*[Error] [rtx.postprocessing.plugin]*",
]
```

### Named Test Suites

Multiple `[[test]]` sections with different names:

```toml
[[test]]
name = "startup"                            # Startup smoke test
args = [
    "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
]

[[test]]
name = "doctest"                            # C++ doc tests
enabled = true
timeout = 900
pythonTests.include = []
pythonTests.exclude = ["*"]
cppTests.libraries = ["bin/${lib_prefix}myext.tests${lib_ext}"]
```

## Complete Minimal Example

```toml
[package]
version = "0.1.0"
category = "Simulation"
title = "My Isaac Sim Extension"
description = "Does something useful."
keywords = ["isaac"]
changelog = "docs/CHANGELOG.md"
readme = "docs/README.md"
preview_image = "data/preview.png"
icon = "data/icon.png"
writeTarget.kit = true

[dependencies]
"isaacsim.core.api" = {}

[[python.module]]
name = "isaacsim.my_domain.my_feature"

[[test]]
timeout = 600
args = [
    "--enable", "omni.kit.loop-isaac",
    "--reset-user",
    "--vulkan",
    "--/app/asyncRendering=false",
    "--no-window",
]
```
