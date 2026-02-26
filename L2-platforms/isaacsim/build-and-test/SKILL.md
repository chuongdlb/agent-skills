---
name: isaacsim-build-and-test
description: >
  Builds, configures, and tests the Isaac Sim repository — Premake5/Lua build system, Packman dependencies, pip packages, extension tests, CI/CD.
layer: L2
domain: [robotics, simulation]
source-project: IsaacSim
depends-on: [isaacsim-simulation-core]
tags: [build, testing, premake, ci-cd]
---

# Building and Testing Isaac Sim

Isaac Sim uses a multi-stage build pipeline based on NVIDIA's `repo_build` tooling with Premake5 for C++ builds, Packman for binary dependencies, and pip for Python packages.

## Build Quick Start

```bash
# Linux
./build.sh

# Windows
build.bat
```

The build script checks EULA acceptance, then delegates to `repo.sh build`.

### Common Build Flags

| Flag | Description |
|------|-------------|
| `-c` | Clean build |
| `-r` | Rebuild (clean + build) |
| `-d` | Debug configuration |
| `--release` | Release configuration |

## Build Pipeline

```
fetch → generate → build → stage → post-build

1. FETCH:      Packman downloads binary deps + pip installs Python packages
2. GENERATE:   Premake5 generates platform-specific build files
3. BUILD:      Compiler builds C++ plugins
4. STAGE:      Outputs staged to _build/ directory
5. POST-BUILD: VSCode settings, USD schema generation, extension caching
```

## Project Configuration (repo.toml)

The primary configuration file (`repo.toml`, ~1030 lines) controls all build behavior:

```toml
[repo]
name = "isaac-sim"

[repo_build]
# Premake-based build system
[repo_build.premake]
# Linux and Windows specific settings

[repo_build.fetch]
# Packman + pip dependency fetching
packman_target_files = [
    "deps/kit-sdk.packman.xml",
    "deps/omni-physics.packman.xml",
    "deps/ext-deps.packman.xml",
    "deps/isaac-sim.packman.xml",
]
pip_target_files = [
    "deps/pip.toml",
    "deps/pip_ml.toml",
    "deps/pip_lula.toml",
    "deps/pip_compute.toml",
]

[repo_build.stage]
# Staging configuration for _build/ output
```

## Premake5 Build System

Three Lua files configure the C++ build:

### premake5.lua (Base)
- Loads `repo_build` module
- Defines build paths for extensions
- Compiler settings: exceptions ON, RTTI ON
- Platform-specific compiler warnings

### premake5-isaacsim.lua (Isaac-Specific)
- PhysX library integration
- Debug/Release configurations (`_DEBUG`, `NDEBUG` defines)
- CUDA compilation support (NVCC)
- Boost library version management

### premake5-tests.lua (Tests)
- Test startup experience definitions
- Test groups: startup_tests, selector_tests, python_samples
- Native Python test definitions

## Dependency Management

### Packman (Binary Dependencies)

Binary dependencies are declared in XML manifests under `deps/`:

```
deps/
  kit-sdk.packman.xml          # Omniverse Kit SDK
  omni-physics.packman.xml     # PhysX integration
  kit-sdk-deps.packman.xml     # Kit SDK dependencies
  ext-deps.packman.xml         # Extension dependencies
  isaac-sim.packman.xml        # Isaac Sim specific (Lula, ROS2, Octomap, etc.)
```

Isaac-specific deps in `isaac-sim.packman.xml`:
- ROS 2 Humble and Jazzy libraries
- Lula motion generation (v0.10.1+)
- Octomap for 3D mapping
- TinyXML2, RapidJSON, Nlohmann JSON
- USD schemas (OmniIsaacSim)

### pip (Python Dependencies)

Python packages are declared in TOML files under `deps/`:

```
deps/
  pip.toml              # Core: numba, numpy, scipy, pandas
  pip_ml.toml           # ML: torch, etc.
  pip_lula.toml         # Lula Python bindings
  pip_compute.toml      # Compute: scipy, etc.
  pip_usd_to_urdf.toml  # nvidia.srl packages
  pip_cloud.toml        # Cloud storage
```

Packages are installed to `pip_prebundle/` for pre-installed availability.

## Extension Test Configuration

Tests are defined in the `[[test]]` section of each extension's `extension.toml`:

```toml
[[test]]
timeout = 900                    # Timeout in seconds

dependencies = [                 # Extra extensions needed for tests
    "isaacsim.test.utils",
    "omni.kit.renderer.core",
]

# Standard test args (copy this block for new extensions)
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

# Exclude known error patterns
stdoutFailPatterns.exclude = [
    "*[Error] [carb] [Plugin: ...]*",
]
```

## Running Tests

```bash
# Run all tests for an extension
repo test -e isaacsim.my_domain.my_feature

# Run specific test suite
repo test --suite pythontests

# Run with filter
repo test -f test_my_function
```

### Test Suites (from repo.toml)

| Suite | Purpose |
|-------|---------|
| `alltests` | Full test suite |
| `startuptests` | Smoke tests with shader compilation |
| `pythontests` | Main Python test suite |
| `benchmarks` | Performance benchmarks |
| `nativepythontests` | Native Python API tests |

### Python Test Buckets

Tests are organized into buckets:
- `deprecated` - Deprecated extension tests
- `asset` - Asset pipeline tests (`isaacsim.asset.*`)
- `core` - Core API tests
- `examples_tests` - Example tests
- `replicator` - SDG/Replicator tests
- `ros` - ROS2 integration tests
- `sensors` - Sensor tests
- `utils_gui` - Utility and GUI tests
- `robot` - Robot tests

## C/C++ Coding Style

Based on Omniverse Carbonite SDK guidelines:

| Element | Convention | Example |
|---------|-----------|---------|
| Namespaces | snake_case | `isaacsim::sensors` |
| Headers/Sources | PascalCase | `MyClass.h`, `MyClass.cpp` |
| Classes/Structs | PascalCase | `ContactSensor` |
| Constants | kCamelCase | `kDefaultTimeout` |
| Functions | camelCase | `computeForce()` |
| Private/Protected | _camelCase or m_camelCase | `_internalState`, `m_data` |
| Member Variables | camelCase or m_camelCase | `maxForce`, `m_maxForce` |
| Static Members | s_camelCase | `s_instance` |
| Global Static | g_camelCase | `g_logger` |
| Macros | MACRO_CASE | `ISAAC_ASSERT` |

Rules:
- No double underscores or `_Uppercase` prefixes (C++ reserved)
- Methods must start with verbs
- Use full English names, avoid abbreviations
- Doxygen-style documentation for public APIs

## Platform Support

| Platform | Architecture | Notes |
|----------|-------------|-------|
| Linux | x86_64 | Primary, GCC 11+ required |
| Linux | aarch64 | ARM support |
| Windows | x86_64 | MSVC required |

## CI/CD Pipeline

**File:** `.github/workflows/build-and-test.yml`

Triggers: push/PR to `main` and `public-main`

Steps:
1. Checkout with LFS and full history
2. Clean workspace (`git clean -fdx`)
3. Clear caches (Omniverse, Packman)
4. Build (`./build.sh`)
5. Accept EULA (`.eula_accepted`)
6. Run tests (startup, warmup, test suites)

Timeout: 240 minutes. Runs on self-hosted GPU runners.

## Environment Variables

```bash
# Required for ROS2 bridge
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Required for container environments
export OMNI_KIT_ALLOW_ROOT=1

# Set by build system
export LD_PRELOAD=<path>/libcarb.so
```

## Reference Files

- [build-system-reference.md](build-system-reference.md) - Annotated repo.toml and Premake5 reference
- [test-configuration.md](test-configuration.md) - Complete test setup reference

## Key Repo Paths

| Path | Description |
|------|-------------|
| `build.sh` / `build.bat` | Build entry points |
| `repo.toml` | Main build configuration |
| `repo_tools.toml` | Custom tool commands |
| `premake5.lua` | Base Premake5 config |
| `premake5-isaacsim.lua` | Isaac-specific build |
| `premake5-tests.lua` | Test definitions |
| `deps/*.packman.xml` | Binary dependency manifests |
| `deps/pip*.toml` | Python dependency configs |
| `tools/packman/` | Packman tool |
| `tools/repoman/` | Repoman tool |
| `.github/workflows/build-and-test.yml` | CI pipeline |
| `docs/overview/guidelines.rst` | Coding style guide |
