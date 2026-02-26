---
name: rltools-environment
description: >
  Create custom RL environments for rl-tools C++17 — 3-file structure (env header, operations_generic, operations_cpu), JSON serialization, UI render.
layer: L2
domain: [general-rl]
source-project: rl-tools-framework
depends-on: [rltools-build]
tags: [environment, custom-env, cpp17]
---

# RL-Tools Environment Creation

## Overview

Create custom RL environments for the rl-tools header-only C++17 framework. Each environment consists of three files following a strict interface pattern. This skill scaffolds complete, compilable environments with all required operations.

## When to Use This Skill

- User wants to create a new RL environment or simulation
- User wants to add a custom task for policy training
- User wants to modify or extend an existing environment
- User asks about the environment interface or required functions

## Canonical Reference

The authoritative example is the `my_pendulum` environment:
- **Header**: `/home/ai/source/rl-tools-framework/example/include/my_pendulum/my_pendulum.h`
- **Operations**: `/home/ai/source/rl-tools-framework/example/include/my_pendulum/operations_generic.h`
- **CPU ops**: `/home/ai/source/rl-tools-framework/example/include/my_pendulum/operations_cpu.h`

Built-in environments for reference:
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/environments/pendulum/`
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/environments/acrobot/`
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/environments/car/`
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/environments/l2f/`

## File Structure

Every environment requires exactly 3 files:
```
include/<env_name>/
  <env_name>.h              — Struct definitions
  operations_generic.h      — Core free functions (device-agnostic)
  operations_cpu.h          — CPU-specific: JSON + UI render
```

## Step-by-Step Workflow

### Step 1: Define the Environment Header

Create `<env_name>.h` with these required types:
1. **Parameters struct** — Physical constants, limits, timestep (template on `T`)
2. **Specification struct** — Bundles `T`, `TI`, `PARAMETERS` type aliases
3. **State struct** — Must have `static constexpr TI DIM` member
4. **Observation struct** — Must have `static constexpr TI DIM` member
5. **Environment struct** — Inherits `rl_tools::rl::environments::Environment<T, TI>`, exposes:
   - `using Parameters`, `using State`, `using Observation`, `using ObservationPrivileged`
   - `static constexpr TI OBSERVATION_DIM`, `ACTION_DIM`, `EPISODE_STEP_LIMIT`

See `references/environment-header-template.md` for the complete template.

### Step 2: Implement Generic Operations

Create `operations_generic.h` with these **required** free functions in `namespace rl_tools`:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `malloc` | `(DEVICE&, ENV&)` | Allocate env resources |
| `free` | `(DEVICE&, ENV&)` | Free env resources |
| `init` | `(DEVICE&, ENV&)` | Initialize env |
| `initial_parameters` | `(DEVICE&, const ENV&, Parameters&)` | Default parameters |
| `sample_initial_parameters` | `(DEVICE&, const ENV&, Parameters&, RNG&)` | Randomized parameters |
| `initial_state` | `(DEVICE&, const ENV&, const Parameters&, State&)` | Fixed initial state |
| `sample_initial_state` | `(DEVICE&, const ENV&, const Parameters&, State&, RNG&)` | Random initial state |
| `step` | `(DEVICE&, const ENV&, const Parameters&, const State&, const Matrix<ACTION>&, State&, RNG&) → T` | Physics step, returns dt |
| `reward` | `(DEVICE&, const ENV&, const Parameters&, const State&, const Matrix<ACTION>&, const State&, RNG&) → T` | Reward function |
| `observe` | `(DEVICE&, const ENV&, const Parameters&, const State&, const Observation&, Matrix<OBS>&, RNG&)` | State → observation |
| `terminated` | `(DEVICE&, const ENV&, const Parameters&, State, RNG&) → bool` | Episode termination |

Key conventions:
- Actions are normalized to `[-1, 1]` by the algorithm; scale inside `step()`
- `step()` returns the timestep duration `dt`
- Use `static_assert` to validate matrix dimensions
- Access matrix elements with `get(matrix, row, col)` and `set(matrix, row, col, value)`
- Use `rl_tools::math::sin/cos/floor/etc.` for device-portable math
- Use `random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), min, max, rng)` for sampling

### Step 3: Add CPU Operations

Create `operations_cpu.h` with JSON serialization and UI render:

1. **`json(device, env, parameters)`** — Returns `"{}"` or parameter JSON
2. **`json(device, env, parameters, state)`** — Returns state as JSON string
3. **`get_ui(device, env)`** — Returns ES6 module string with `init()` and `render()` exports

The UI function uses a raw string literal: `R"RL_TOOLS_LITERAL(...)RL_TOOLS_LITERAL"`

Use https://studio.rl.tools for interactive render function development.

### Step 4: Wire Into Training

Add includes in `main.cpp`:
```cpp
#include "my_env/my_env.h"
#include "my_env/operations_generic.h"
#include "my_env/operations_cpu.h"
```

Define types:
```cpp
using PENDULUM_SPEC = MyEnvSpecification<T, TI>;
using ENVIRONMENT = MyEnv<PENDULUM_SPEC>;
```

### Step 5: Update CMakeLists.txt

No special CMake changes needed — header-only. Just ensure the include path covers your environment headers:
```cmake
target_include_directories(my_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

## Common Patterns

### Angle Wrapping
```cpp
template <typename DEVICE, typename T>
T angle_normalize(const DEVICE& dev, T x) {
    return f_mod_python(dev, (x + rl_tools::math::PI<T>), (2 * rl_tools::math::PI<T>)) - rl_tools::math::PI<T>;
}
```

### Clipping
```cpp
template <typename T>
T clip(T x, T min, T max) {
    return x < min ? min : (x > max ? max : x);
}
```

### Fourier Observation (angle → cos/sin)
```cpp
set(observation, 0, 0, rl_tools::math::cos(device.math, state.theta));
set(observation, 0, 1, rl_tools::math::sin(device.math, state.theta));
```

## Checklist Before Completion

- [ ] All 11 required free functions implemented
- [ ] `static_assert` on action and observation matrix dimensions
- [ ] State DIM and Observation DIM match the actual struct fields
- [ ] `step()` returns `dt`, not reward
- [ ] Actions scaled from normalized `[-1, 1]` range
- [ ] JSON serialization covers all state fields
- [ ] UI render function exports `init()` and `render()`
- [ ] Environment compiles with training config
