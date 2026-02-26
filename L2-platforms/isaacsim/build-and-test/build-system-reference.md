# Build System Reference

Annotated reference for Isaac Sim's build system components.

## repo.toml Key Sections

### [repo] - Repository Identity

```toml
[repo]
name = "isaac-sim"
```

### [repo_build] - Build Pipeline

```toml
[repo_build]
# Import base configuration from Kit Template
# Premake-based build system

[repo_build.premake]
# Premake5 project generation settings
# Platform-specific compiler configurations
```

### [repo_build.fetch] - Dependency Fetching

```toml
[repo_build.fetch]
# Packman binary dependencies
packman_target_files = [
    "deps/kit-sdk.packman.xml",       # Omniverse Kit SDK
    "deps/omni-physics.packman.xml",  # PhysX integration
    "deps/kit-sdk-deps.packman.xml",  # Kit SDK dependencies
    "deps/ext-deps.packman.xml",      # Extension dependencies
    "deps/isaac-sim.packman.xml",     # Isaac Sim specific
]

# Python pip dependencies
pip_target_files = [
    "deps/pip.toml",              # Core packages
    "deps/pip_ml.toml",           # ML packages (torch)
    "deps/pip_lula.toml",         # Lula motion planning
    "deps/pip_compute.toml",      # Scientific computing
    "deps/pip_usd_to_urdf.toml",  # USD-to-URDF conversion
    "deps/pip_cloud.toml",        # Cloud storage
]
```

### [repo_build.stage] - Output Staging

```toml
[repo_build.stage]
# Controls which files are copied to _build/
# allowed_from: ["_build/", "docs/source/"]
```

### [repo_build.commands] - Build Hooks

```toml
[repo_build.commands]
# Pre-build: cache extensions
pre_build = ["precache_exts"]

# Post-build: generate IDE settings, USD schemas
post_build = ["generate_vscode_settings", "edit_sysconfig", "usd"]
```

### [repo_test] - Test Configuration

```toml
[repo_test]
# Defines test suites and their configurations

[repo_test.suites.pythontests]
# Main Python test suite with test buckets:
# - deprecated, asset, core, examples_tests, replicator
# - ros, sensors, utils_gui, robot, other

[repo_test.suites.startuptests]
# Smoke tests with shader compilation

[repo_test.suites.benchmarks]
# Performance benchmarks
```

### [repo_publish] - Extension Publishing

```toml
[repo_publish]
enabled = true
platforms = ["windows-x86_64", "linux-x86_64", "linux-aarch64"]

# Extensions to publish
exts_include = ["isaacsim.*", "omni.isaac.*", "omni.kit.loop-isaac"]

# Signing and verification
signing = true
verify = true
```

### [repo_format] - Code Formatting

```toml
[repo_format.python]
maintain_legal_blurbs = false

# SPDX license preamble
preamble = """
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
```

## Premake5 Lua API

### premake5.lua Functions

```lua
-- Load repo_build module
local repo_build = require("omni/repo/build")

-- Define build extension directories
local ext_dirs = {
    "source/extensions",
}

-- Isaac Sim build settings
function isaacsim_build_settings()
    exceptionhandling("On")
    rtti("On")

    -- Platform-specific
    filter { "system:linux" }
        buildoptions { "-Wno-error=deprecated-declarations" }
    filter { "system:windows" }
        buildoptions { "/wd4996" }  -- Disable deprecation warnings
end
```

### premake5-isaacsim.lua Functions

```lua
-- PhysX integration
function isaacsim_physx_settings()
    -- Adds PhysX include paths and library paths
    -- Platform and config specific (Debug/Release)
end

-- CUDA compilation
function isaacsim_cuda_settings()
    -- NVCC compiler commands
    -- Compute capability flags
end

-- Boost library management
function isaacsim_boost_settings()
    -- Boost version and library paths
end
```

### premake5-tests.lua Functions

```lua
-- Test startup definitions
function define_startup_tests()
    -- startup_tests: main, streaming, extscache, xr.vr
    -- selector_tests: with auto-launch
    -- python_samples: native Python tests
end
```

## Packman XML Manifest Format

```xml
<?xml version="1.0" encoding="utf-8"?>
<project toolsVersion="5.6">
    <!-- Binary package dependency -->
    <dependency name="lula" linkPath="../_build/${platform}/${config}/lula">
        <package name="lula" version="0.10.1" />
    </dependency>

    <!-- Platform-specific dependency -->
    <dependency name="ros2-humble" linkPath="../_build/${platform}/${config}/ros2"
                platforms="linux-x86_64">
        <package name="ros2-humble" version="1.0.0" />
    </dependency>

    <!-- Multi-platform -->
    <dependency name="octomap" linkPath="../_build/${platform}/${config}/octomap">
        <package name="octomap" version="1.9.8"
                 platforms="linux-x86_64,linux-aarch64,windows-x86_64" />
    </dependency>
</project>
```

Key Isaac Sim packages in `deps/isaac-sim.packman.xml`:
- ROS 2 Humble (linux-x86_64)
- ROS 2 Jazzy (linux-x86_64, linux-aarch64)
- Lula motion generation
- Octomap 3D mapping
- TinyXML2, RapidJSON
- USD schemas (OmniIsaacSim)

## pip TOML Configuration Format

```toml
# deps/pip.toml
[[dependency]]
name = "numpy"
version = "1.24.0"

[[dependency]]
name = "scipy"
version = "1.10.0"

[[dependency]]
name = "numba"
version = "0.57.0"

[config]
target_dir = "pip_prebundle"
python_version = "3.11"
```

## Platform-Specific Build

### Linux x86_64

- Compiler: GCC 11+
- CXX ABI: enabled
- LD_PRELOAD: libcarb.so
- CUDA: NVCC from system or Packman

### Linux aarch64

- ARM-specific compiler flags
- Separate Packman packages

### Windows x86_64

- Compiler: MSVC
- Windows SDK required
- Different binary naming conventions

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `LD_PRELOAD` | Preload Carbonite library | `_build/linux-x86_64/release/libcarb.so` |
| `RMW_IMPLEMENTATION` | ROS 2 middleware | `rmw_fastrtps_cpp` |
| `LD_LIBRARY_PATH` | ROS 2 library paths | `_build/.../ros2/lib` |
| `OMNI_KIT_ALLOW_ROOT` | Allow root execution | `1` |
| `PYTHONPATH` | Python module paths | Set by build system |

## Build Output Structure

```
_build/
  linux-x86_64/
    release/
      apps/             # Kit application configs
      exts/             # Built extensions
      extscache/        # Extension cache
      kit/              # Kit SDK
      python/           # Python interpreter
      scripts/          # Runtime scripts
      standalone_examples/  # Copied examples
```

## Useful Build Commands

```bash
# Full clean rebuild
./build.sh -r

# Debug build
./build.sh -d

# List available repo commands
./repo.sh --help

# Run tests
./repo.sh test -e isaacsim.core.api

# Package for distribution
./repo.sh package

# Format code
./repo.sh format
```
