---
name: rltools-deploy-hardware
description: >
  Deploy trained rl-tools policies to hardware — ESP32, Teensy, Crazyflie, PX4, Betaflight platform adapters and firmware integration.
layer: L2
domain: [drones, embedded, sim-to-real]
source-project: rl-tools-framework
depends-on: [rltools-build, rltools-training]
tags: [deployment, embedded, esp32, crazyflie, px4]
---

# RL-Tools Hardware Deployment

## Overview

Deploy trained RL policies to embedded hardware platforms. All platforms use a common C adapter interface pattern wrapping the C++ inference executor. This skill covers policy export, firmware building, and timing synchronization for real-time control.

## When to Use This Skill

- User wants to deploy a trained policy to hardware
- User wants to export a policy to a C++ header
- User wants to build firmware for a specific platform
- User needs to configure the inference executor
- User wants to set up SITL before hardware deployment

## Supported Platforms

| Platform | Location | Device Type | Build System |
|----------|----------|-------------|-------------|
| Crazyflie | `crazyflie-controller/` | ARM Cortex-M4 | Make |
| PX4/Pixhawk | `px4/` | ARM Cortex-M7 | CMake/Make |
| ESP32 | `esp32/` | Xtensa/RISC-V | ESP-IDF |
| Teensy 4.1 | `teensy/` | ARM Cortex-M7 | Arduino CLI |
| Betaflight | `betaflight-firmware/` | STM32 | Make |
| iOS | `ios/` | Apple Silicon | Xcode |

## C Adapter Interface Pattern

Every platform implements the same interface boundary:

### Simple Interface
```c
// rl_tools_adapter.h
void rl_tools_init();
void rl_tools_reset();
void rl_tools_control(float* state, float* actions);
const char* rl_tools_get_checkpoint_name();
```

### L2F Interface (Quadrotors)
```c
// For platforms with timing synchronization
typedef struct {
    float position[3];
    float orientation[4];       // Quaternion: w, x, y, z
    float linear_velocity[3];
    float angular_velocity[3];
    float previous_action[4];
} RLtoolsInferenceApplicationsL2FObservation;

typedef struct {
    float action[4];
} RLtoolsInferenceApplicationsL2FAction;

void rl_tools_inference_applications_l2f_init();
void rl_tools_inference_applications_l2f_reset();
void rl_tools_inference_applications_l2f_control(
    uint64_t nanoseconds,
    RLtoolsInferenceApplicationsL2FObservation* obs,
    RLtoolsInferenceApplicationsL2FAction* action
);
```

## Deployment Pipeline

### Step 1: Export Policy

After training, export the policy to a C++ header file containing compile-time weight arrays:
```cpp
// Training produces: policy.h
// Contains constexpr arrays of network weights
#include "policy.h"
```

### Step 2: Create Adapter

```cpp
// rl_tools_adapter.cpp
#include <rl_tools/inference/applications/l2f/c_backend.h>
#include "policy.h"

// Configure device for target platform
using DEV_SPEC = rlt::devices::DefaultARMSpecification;  // or ESP32, etc.
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;

// Static allocation — no malloc at runtime
static DEVICE device;
static typename DEVICE::SPEC::RANDOM::ENGINE<> rng;
// ... executor state ...
```

### Step 3: Build Firmware

See platform-specific build commands below.

### Step 4: Flash and Test

## Platform Details

### Crazyflie

**Reference files:**
- `/home/ai/source/rl-tools-framework/crazyflie-controller/rl_tools_controller.c`
- `/home/ai/source/rl-tools-framework/crazyflie-controller/rl_tools_adapter.h`
- `/home/ai/source/rl-tools-framework/crazyflie-controller/rl_tools_adapter.cpp`

**Key details:**
- ARM Cortex-M4, `rlt::devices::arm::OPT<DEV_SPEC>`
- 500 Hz control loop, 5x oversampling to match 100 Hz training
- 32-step action history (input: 18 base dims + 32×4 action history = 146 dims)
- Static buffers, no dynamic allocation
- Rotation matrix preprocessing for orientation

**Build:**
```bash
cd crazyflie-controller
make  # Uses external/firmware + external/rl_tools
```

### PX4 Autopilot

**Reference files:**
- `/home/ai/source/rl-tools-framework/px4/external_modules/src/modules/rl_tools_policy/RLtoolsPolicy.hpp`
- `/home/ai/source/rl-tools-framework/px4/external_modules/src/modules/rl_tools_policy/RLtoolsPolicy.cpp`
- `/home/ai/source/rl-tools-framework/px4/external_modules/src/modules/rl_tools_policy/rl_tools_adapter.cpp`

**Key details:**
- PX4 ModuleBase + ScheduledWorkItem integration
- uORB topics: `vehicle_attitude`, `vehicle_angular_velocity`, `vehicle_local_position`
- Output: `actuator_motors` topic
- Nanosecond-precision timing with timestamp validation
- Multiple odometry sources supported

**Build:**
```bash
make px4_fmu-v6c_default EXTERNAL_MODULES_LOCATION=<path>/external_modules
```

### ESP32

**Reference files:**
- `/home/ai/source/rl-tools-framework/esp32/main/main.c`
- `/home/ai/source/rl-tools-framework/esp32/main/rl_tools_adapter.h`
- `/home/ai/source/rl-tools-framework/esp32/main/rl_tools_adapter.cpp`

**Key details:**
- ESP32-S3 with DSP acceleration: `rlt::devices::esp32::DSP<DEV_SPEC_S3>`
- Default ESP32: `rlt::devices::esp32::OPT<DEV_SPEC>`
- ESP-IDF component system

**Build:**
```bash
idf.py build
idf.py flash
```

### Teensy 4.1

**Reference files:**
- `/home/ai/source/rl-tools-framework/teensy/Arduino/rl_tools.ino`

**Key details:**
- Full SAC/PPO/TD3 training on microcontroller (not just inference)
- Memory regions: RAM1 (512KB code), RAM2 DMAMEM (512KB), PSRAM EXTMEM (16MB)
- Replay buffer stored in external PSRAM

**Build:**
```bash
arduino-cli compile --fqbn teensy:avr:teensy41 --build-properties "build.flags.cpp=-std=c++17"
```

### Betaflight

**Reference files:**
- `/home/ai/source/rl-tools-framework/betaflight-firmware/src/main/flight/rl_tools_adapter.h`
- `/home/ai/source/rl-tools-framework/betaflight-firmware/src/main/flight/rl_tools_adapter.cpp`

**Build:**
```bash
make TARGET=<board> RL_TOOLS_PATH=<path>/rl-tools
```

## Core Inference Files

- Executor: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/inference/executor/`
- L2F C backend: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/inference/applications/l2f/c_backend.h`
- L2F C interface: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/inference/applications/l2f/c_interface.h`

## Timing Synchronization

The inference executor handles rate matching between hardware control frequency and training frequency:

- **Intermediate frequency**: Hardware control rate (e.g., 500 Hz on Crazyflie)
- **Native frequency**: Training rate (e.g., 100 Hz)
- `force_sync` controls rate matching behavior
- Timing jitter/bias monitoring with configurable warning thresholds

## Key Design Principles

1. **Static allocation** — No dynamic memory at runtime
2. **C/C++ boundary** — Pure C interface for firmware integration
3. **Compile-time weights** — Policy embedded as constexpr arrays in header
4. **Template configuration** — Platform config via `RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG`
5. **Action history** — Maintained for temporal/recurrent policies
