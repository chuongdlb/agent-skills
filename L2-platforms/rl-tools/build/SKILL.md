---
name: rltools-build
description: >
  CMake build system configuration for rl-tools — tiered targets (Core/Backend/Minimal/RLtools), backend auto-detection, embedded platform builds.
layer: L2
domain: [general-rl, embedded]
source-project: rl-tools-framework
depends-on: []
tags: [cmake, build, embedded, cpp17]
---

# RL-Tools Build System Configuration

## Overview

Configure CMake build systems for projects using the rl-tools header-only C++17 library. Covers target tiers, backend auto-detection, optimization flags, and embedded platform builds.

## When to Use This Skill

- User wants to create a new rl-tools project from scratch
- User wants to add rl-tools to an existing CMake project
- User needs to configure backends (MKL, OpenBLAS, CUDA)
- User wants to set up builds for embedded targets
- User has build errors or linking issues with rl-tools

## Reference Files

- Canonical project: `/home/ai/source/rl-tools-framework/example/CMakeLists.txt`
- Full build system: `/home/ai/source/rl-tools-framework/rl-tools/CMakeLists.txt`
- Backend detection: `/home/ai/source/rl-tools-framework/rl-tools/cmake/autodetect/`
- Optional deps: `/home/ai/source/rl-tools-framework/rl-tools/cmake/optional/`

## New Project Setup

### Minimal CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(my_project)

add_subdirectory(external/rl_tools)

add_executable(my_target src/main.cpp)
target_link_libraries(my_target PRIVATE RLtools::RLtools)

# Benchmark target (disables eval/checkpoint overhead)
add_executable(my_target_benchmark src/main.cpp)
target_compile_definitions(my_target_benchmark PRIVATE BENCHMARK)
target_link_libraries(my_target_benchmark PRIVATE RLtools::RLtools)

# Optimization flags for Release builds
if(NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(my_target PRIVATE -Ofast)
    if(NOT APPLE)
        target_compile_options(my_target PRIVATE -march=native)
    endif()
endif()
```

### Directory Structure

```
my_project/
  CMakeLists.txt
  external/
    rl_tools/              # git submodule
  include/
    my_env/
      my_env.h
      operations_generic.h
      operations_cpu.h
  src/
    main.cpp
```

### Adding rl-tools as Submodule

```bash
mkdir -p external
git submodule add https://github.com/rl-tools/rl-tools.git external/rl_tools
```

## Target Tiers

| Target | What It Provides | When to Use |
|--------|-----------------|-------------|
| `RLtools::Core` | Pure C++17, no external deps | Minimal builds, portability |
| `RLtools::Backend` | Core + auto-detected BLAS | Performance with BLAS |
| `RLtools::Minimal` | Backend + minimal optional deps | Standard development |
| `RLtools::RLtools` | Full library + optimization flags | Production training |

`RLtools::RLtools` automatically adds `-O3 -ffast-math -march=native` on GNU/Clang Release builds.

For most projects, use `RLtools::RLtools`. Use `RLtools::Core` only when targeting platforms without BLAS support or when you need zero external dependencies.

## Backend Configuration

Backends are auto-detected by the build system. To force a specific backend:

```cmake
# Force MKL
set(RL_TOOLS_BACKEND_ENABLE_MKL ON CACHE BOOL "" FORCE)

# Force OpenBLAS
set(RL_TOOLS_BACKEND_ENABLE_OPENBLAS ON CACHE BOOL "" FORCE)

# Disable BLAS entirely
set(RL_TOOLS_BACKEND_DISABLE_BLAS ON CACHE BOOL "" FORCE)
```

### CUDA Support
```cmake
enable_language(CUDA)
set(RL_TOOLS_BACKEND_ENABLE_CUDA ON CACHE BOOL "" FORCE)
```

## Optional Dependencies

### HDF5 (for checkpoints)
```cmake
set(RL_TOOLS_ENABLE_HDF5 ON CACHE BOOL "" FORCE)
# Requires HighFive: add_subdirectory(external/highfive)
```

### TensorBoard (for logging)
```cmake
set(RL_TOOLS_ENABLE_TENSORBOARD ON CACHE BOOL "" FORCE)
# Requires tensorboard_logger
```

### MuJoCo (for MuJoCo environments)
```cmake
set(RL_TOOLS_ENABLE_MUJOCO ON CACHE BOOL "" FORCE)
```

### JSON (for serialization)
```cmake
find_package(nlohmann_json REQUIRED)
```

## Embedded Platform Builds

### ESP32 (ESP-IDF)
```cmake
# In main/CMakeLists.txt
idf_component_register(
    SRCS "main.c" "rl_tools_adapter.cpp"
    INCLUDE_DIRS "."
)
target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)
```
Build: `idf.py build`

### Teensy (Arduino CLI)
Add rl_tools headers to Arduino library path. Use:
```bash
arduino-cli compile --fqbn teensy:avr:teensy41 --build-properties "build.flags.cpp=-std=c++17"
```

### ARM (Crazyflie)
Uses Kbuild/Make. Add C++17 flag:
```makefile
EXTRA_CFLAGS += -std=c++17
```

### PX4
```bash
make px4_fmu-v6c_default EXTERNAL_MODULES_LOCATION=<path>/external_modules
```

### WebAssembly (Emscripten)
```bash
emcmake cmake -B build_wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build_wasm
```

## Build Commands

```bash
# Standard build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With specific backend
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_MKL=ON
cmake --build build -j$(nproc)

# Run
./build/my_target
```

## Troubleshooting

### Common Issues

1. **C++17 not enabled**: Ensure `cmake_minimum_required(VERSION 3.10)` and target links `RLtools::Core` or higher (sets `cxx_std_17`)
2. **Missing BLAS**: Install MKL or OpenBLAS, or use `RLtools::Core`
3. **Header not found**: Check `add_subdirectory` path matches actual rl_tools location
4. **Linker errors with HDF5**: Ensure HighFive is added before rl_tools in CMake
5. **Slow training**: Use Release build (`-DCMAKE_BUILD_TYPE=Release`), link `RLtools::RLtools`
