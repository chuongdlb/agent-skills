# Platform Adapter Reference

## Adapter File Locations

| Platform | Adapter Header | Adapter Impl | Controller |
|----------|---------------|-------------|-----------|
| Crazyflie | `crazyflie-controller/rl_tools_adapter.h` | `crazyflie-controller/rl_tools_adapter.cpp` | `crazyflie-controller/rl_tools_controller.c` |
| PX4 | — | `px4/.../rl_tools_adapter.cpp` | `px4/.../RLtoolsPolicy.cpp` |
| ESP32 | `esp32/main/rl_tools_adapter.h` | `esp32/main/rl_tools_adapter.cpp` | `esp32/main/main.c` |
| Teensy | — | — | `teensy/Arduino/rl_tools.ino` |
| Betaflight | `betaflight-firmware/.../rl_tools_adapter.h` | `betaflight-firmware/.../rl_tools_adapter.cpp` | — |

## Device Type Selection

```cpp
// CPU (desktop training)
using DEVICE = rlt::devices::DEVICE_FACTORY<>;

// ARM (Crazyflie, Teensy)
using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;

// ESP32 (default)
using DEV_SPEC = rlt::devices::DefaultESP32Specification<rlt::devices::esp32::Hardware::DEFAULT>;
using DEVICE = rlt::devices::esp32::OPT<DEV_SPEC>;

// ESP32-S3 with DSP
using DEV_SPEC = rlt::devices::DefaultESP32Specification<rlt::devices::esp32::Hardware::S3>;
using DEVICE = rlt::devices::esp32::DSP<DEV_SPEC>;

// WebAssembly
// Device: rlt::devices::wasm32
```

## Minimal Adapter Template

```cpp
// rl_tools_adapter.h
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void rl_tools_init();
void rl_tools_reset();
void rl_tools_control(float* state, float* actions);
const char* rl_tools_get_checkpoint_name();

#ifdef __cplusplus
}
#endif
```

```cpp
// rl_tools_adapter.cpp
#include "rl_tools_adapter.h"

// Device-specific includes
#include <rl_tools/operations/arm.h>  // or esp32.h, cpu.h

// Policy blob (exported from training)
#include "policy.h"

namespace rlt = rl_tools;

using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
using T = float;
using TI = typename DEVICE::index_t;

// Static allocation
static DEVICE device;
static typename DEVICE::SPEC::RANDOM::ENGINE<> rng;
// ... model and buffer declarations ...

extern "C" {

void rl_tools_init() {
    rlt::malloc(device, model);
    // Load weights from policy.h
}

void rl_tools_reset() {
    // Reset any internal state
}

void rl_tools_control(float* state, float* actions) {
    // Copy state into input matrix
    // rlt::evaluate(device, model, input, output, buffer, rng);
    // Copy output to actions array
}

const char* rl_tools_get_checkpoint_name() {
    return "my_policy_v1";
}

}
```

## Memory Layout (Teensy)

```cpp
// RAM1 (512KB) - Stack + code
static auto loop_state_stack_vars;

// RAM2 DMAMEM (512KB) - Actor + buffers
DMAMEM static ActorType actor;
DMAMEM static BufferType buffer;

// PSRAM EXTMEM (16MB) - Large data structures
EXTMEM static ReplayBufferType replay_buffer;
EXTMEM static RunnerType runner;
```

## PX4 uORB Integration

```cpp
// Subscribe to state topics
_vehicle_attitude_sub = orb_subscribe(ORB_ID(vehicle_attitude));
_vehicle_angular_velocity_sub = orb_subscribe(ORB_ID(vehicle_angular_velocity));
_vehicle_local_position_sub = orb_subscribe(ORB_ID(vehicle_local_position));

// Publish motor commands
_actuator_motors_pub = orb_advertise(ORB_ID(actuator_motors), &_actuator_motors);
```
