---
name: rltools-testing
description: >
  Testing and benchmarking for rl-tools — GoogleTest, benchmark targets, environment correctness, NN inference speed, memory profiling for embedded.
layer: L2
domain: [general-rl]
source-project: rl-tools-framework
depends-on: [rltools-build]
tags: [testing, benchmarking, googletest]
---

# RL-Tools Testing & Benchmarking

## Overview

Test and benchmark rl-tools implementations. Covers unit testing with GoogleTest, training speed benchmarks, environment correctness validation, inference speed profiling, and embedded memory analysis.

## When to Use This Skill

- User wants to write tests for a custom environment
- User wants to benchmark training or inference speed
- User wants to validate environment step/reward/observe correctness
- User wants to compare convergence across configurations
- User wants to profile memory for embedded deployment

## Reference Files

### Test suite:
- `/home/ai/source/rl-tools-framework/rl-tools/tests/`

### Benchmark patterns:
- `/home/ai/source/rl-tools-framework/example/CMakeLists.txt` (benchmark target)
- `/home/ai/source/rl-tools-framework/crazyflie-firmware-benchmark/` (hardware benchmark)
- `/home/ai/source/rl-tools-framework/l2f-benchmark/` (L2F benchmark)
- `/home/ai/source/rl-tools-framework/rl-tools.github.io/` (web benchmark: `benchmark.html`)

## Unit Testing with GoogleTest

### CMake Setup

```cmake
# Enable testing
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_my_env tests/test_my_env.cpp)
target_link_libraries(test_my_env PRIVATE RLtools::RLtools GTest::gtest_main)
add_test(NAME test_my_env COMMAND test_my_env)
```

### Environment Correctness Tests

```cpp
#include <gtest/gtest.h>
#include <rl_tools/operations/cpu_mux.h>
#include "my_env/my_env.h"
#include "my_env/operations_generic.h"

namespace rlt = rl_tools;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;
using SPEC = MyEnvSpecification<T, TI>;
using ENV = MyEnv<SPEC>;

class EnvTest : public ::testing::Test {
protected:
    DEVICE device;
    ENV env;
    typename ENV::Parameters params;
    typename ENV::State state, next_state;
    typename DEVICE::SPEC::RANDOM::ENGINE<> rng;

    void SetUp() override {
        rlt::malloc(device, env);
        rlt::init(device, env);
        rlt::initial_parameters(device, env, params);
        rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 42);
    }

    void TearDown() override {
        rlt::free(device, env);
    }
};

TEST_F(EnvTest, InitialState) {
    rlt::initial_state(device, env, params, state);
    // Verify initial state is within expected bounds
    EXPECT_GE(state.x, -10.0f);
    EXPECT_LE(state.x, 10.0f);
}

TEST_F(EnvTest, SampleInitialState) {
    for (int i = 0; i < 100; i++) {
        rlt::sample_initial_state(device, env, params, state, rng);
        // Verify sampled state is within bounds
        EXPECT_GE(state.x, SPEC::PARAMETERS::INITIAL_STATE_MIN_X);
        EXPECT_LE(state.x, SPEC::PARAMETERS::INITIAL_STATE_MAX_X);
    }
}

TEST_F(EnvTest, StepProducesValidState) {
    rlt::initial_state(device, env, params, state);

    // Create action matrix
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    rlt::set(action, 0, 0, (T)0.5);

    T dt = rlt::step(device, env, params, state, action, next_state, rng);

    EXPECT_GT(dt, 0);
    EXPECT_FALSE(std::isnan(next_state.x));
    EXPECT_FALSE(std::isinf(next_state.x));

    rlt::free(device, action);
}

TEST_F(EnvTest, RewardIsBounded) {
    rlt::initial_state(device, env, params, state);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    rlt::set(action, 0, 0, (T)0.0);

    rlt::step(device, env, params, state, action, next_state, rng);
    T r = rlt::reward(device, env, params, state, action, next_state, rng);

    EXPECT_FALSE(std::isnan(r));
    EXPECT_FALSE(std::isinf(r));

    rlt::free(device, action);
}

TEST_F(EnvTest, ObservationDimension) {
    rlt::initial_state(device, env, params, state);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::OBSERVATION_DIM>> obs;
    rlt::malloc(device, obs);

    typename ENV::Observation obs_type;
    rlt::observe(device, env, params, state, obs_type, obs, rng);

    for (TI i = 0; i < ENV::OBSERVATION_DIM; i++) {
        EXPECT_FALSE(std::isnan(rlt::get(obs, 0, i)));
    }

    rlt::free(device, obs);
}

TEST_F(EnvTest, EpisodeTermination) {
    rlt::initial_state(device, env, params, state);
    bool terminated = rlt::terminated(device, env, params, state, rng);
    // Initial state should not be terminal
    EXPECT_FALSE(terminated);
}
```

### Running Tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cd build && ctest --output-on-failure
```

## Benchmark Targets

### Training Speed Benchmark

```cmake
add_executable(my_env_benchmark src/main.cpp)
target_compile_definitions(my_env_benchmark PRIVATE BENCHMARK)
target_link_libraries(my_env_benchmark PRIVATE RLtools::RLtools)

if(NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(my_env_benchmark PRIVATE -Ofast -march=native)
endif()
```

The `BENCHMARK` define disables:
- Evaluation episodes
- Checkpoint saving
- Trajectory recording
- ExTrack directory creation

This gives pure training throughput numbers.

### Measuring Training Time

```cpp
#include <chrono>
#include <iostream>

auto start = std::chrono::high_resolution_clock::now();
while (!rlt::step(device, ls)) {}
auto end = std::chrono::high_resolution_clock::now();

std::chrono::duration<double> elapsed = end - start;
std::cout << "Training time: " << elapsed.count() << " s" << std::endl;
std::cout << "Steps/sec: " << LOOP_CORE_PARAMETERS::STEP_LIMIT / elapsed.count() << std::endl;
```

### Inference Speed Benchmark

```cpp
#include <chrono>

// Warm up
for (int i = 0; i < 100; i++) {
    rlt::evaluate(device, actor, input, output, buffer, rng);
}

// Measure
auto start = std::chrono::high_resolution_clock::now();
constexpr int N = 10000;
for (int i = 0; i < N; i++) {
    rlt::evaluate(device, actor, input, output, buffer, rng);
}
auto end = std::chrono::high_resolution_clock::now();

double us_per_inference = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)N;
std::cout << "Inference: " << us_per_inference << " us/sample" << std::endl;
std::cout << "Throughput: " << 1e6 / us_per_inference << " samples/sec" << std::endl;
```

## Convergence Comparison

Run multi-seed experiments and compare:

```cpp
int main() {
    DEVICE device;
    for (TI seed = 0; seed < 10; seed++) {
        LOOP_STATE ls;
        ls.extrack_config.name = "convergence_test";
        rlt::malloc(device, ls);
        rlt::init(device, ls, seed);
        while (!rlt::step(device, ls)) {}
        rlt::free(device, ls);
    }
}
```

Compare `return.json` files across seeds using the zoo frontend or custom analysis.

## Memory Profiling (Embedded)

For embedded targets, track compile-time memory usage:

```cpp
// Print sizeof key structures
std::cout << "Loop state size: " << sizeof(LOOP_STATE) << " bytes" << std::endl;
std::cout << "Actor size: " << sizeof(decltype(ls.actor)) << " bytes" << std::endl;
std::cout << "Critic size: " << sizeof(decltype(ls.critic)) << " bytes" << std::endl;
```

For Teensy memory regions:
```cpp
// RAM1: Stack + static
// DMAMEM: Heap (actor, buffers)
// EXTMEM: PSRAM (replay buffer, runner)
DMAMEM static ActorType actor;
EXTMEM static ReplayBufferType replay_buffer;
```

## Hardware Benchmarks

- Crazyflie inference: `/home/ai/source/rl-tools-framework/crazyflie-firmware-benchmark/`
- L2F benchmark: `/home/ai/source/rl-tools-framework/l2f-benchmark/`
- Web benchmark: `/home/ai/source/rl-tools-framework/rl-tools.github.io/benchmark.html`

## Test Checklist

- [ ] Initial state is deterministic (`initial_state`)
- [ ] Sampled states are within parameter bounds (`sample_initial_state`)
- [ ] Step produces valid (non-NaN, non-Inf) next state
- [ ] Step returns positive dt
- [ ] Reward is bounded and valid
- [ ] Observation has correct dimension
- [ ] All observation elements are valid numbers
- [ ] Termination is `false` for initial state (unless env design requires it)
- [ ] Multiple episodes run without crash
- [ ] Training converges on simple test cases
