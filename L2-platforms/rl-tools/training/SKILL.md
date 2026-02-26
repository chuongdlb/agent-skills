---
name: rltools-training
description: >
  Configure and run RL training with rl-tools — SAC, TD3, PPO training, hyperparameters, network architecture, composable loop steps.
layer: L2
domain: [general-rl]
source-project: rl-tools-framework
depends-on: [rltools-build, rltools-neural-network]
tags: [training, sac, td3, ppo]
---

# RL-Tools Training Configuration

## Overview

Configure and run reinforcement learning training experiments using the rl-tools framework. Supports SAC, TD3, and PPO algorithms with composable loop steps for evaluation, checkpointing, experiment tracking, and trajectory saving.

## When to Use This Skill

- User wants to train a policy with SAC, TD3, or PPO
- User wants to tune hyperparameters or network architecture
- User wants to set up multi-seed experiments
- User asks about training configuration or loop steps
- User wants to compare algorithms on an environment

## Algorithm Selection Guide

| Algorithm | Type | Best For | Key Advantage |
|-----------|------|----------|---------------|
| **SAC** | Off-policy | Continuous control, sample efficiency | Automatic entropy tuning |
| **TD3** | Off-policy | Continuous control, stability | Twin critics reduce overestimation |
| **PPO** | On-policy | Robustness, parallel envs | Simple, reliable, parallelizable |

## Reference Files

### Algorithm configs:
- SAC: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/algorithms/sac/loop/core/config.h`
- TD3: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/algorithms/td3/loop/core/config.h`
- PPO: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/algorithms/ppo/loop/core/config.h`

### Zoo examples:
- SAC pendulum: `/home/ai/source/rl-tools-framework/rl-tools/src/rl/zoo/pendulum-v1/sac.h`
- TD3 pendulum: `/home/ai/source/rl-tools-framework/rl-tools/src/rl/zoo/pendulum-v1/td3.h`

### Canonical example:
- `/home/ai/source/rl-tools-framework/example/src/main.cpp`

### Loop steps:
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/extrack/`
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/evaluation/`
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/checkpoint/`
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/save_trajectories/`
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/timing/`
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/nn_analytics/`

## Training main.cpp Structure

Every training entry point follows this pattern:

```
1. Include operations headers (cpu_mux, nn, nn_models)
2. Include environment headers (env.h, operations_generic.h, operations_cpu.h)
3. Include algorithm loop config + operations
4. Include loop step configs + operations
5. Define type aliases (DEVICE, RNG, T, TI, ENVIRONMENT)
6. Define LOOP_CORE_PARAMETERS struct inheriting DefaultParameters
7. Stack loop step configs via template aliases
8. Define LOOP_STATE from final config
9. main(): malloc → init → while(!step) → free
```

See `references/training-templates.md` for complete templates for each algorithm.

## Key Parameters

### Common Parameters
```cpp
static constexpr TI STEP_LIMIT = 20000;          // Training steps
static constexpr TI ACTOR_HIDDEN_DIM = 64;        // Actor network width
static constexpr TI CRITIC_HIDDEN_DIM = 64;       // Critic network width
static constexpr TI ACTOR_NUM_LAYERS = 3;         // Actor depth
static constexpr TI CRITIC_NUM_LAYERS = 3;        // Critic depth
static constexpr TI EPISODE_STEP_LIMIT = 200;     // Max episode length
static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
```

### PPO-Specific
```cpp
static constexpr TI N_ENVIRONMENTS = 8;                    // Parallel envs
static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 128;  // Rollout length
static constexpr TI BATCH_SIZE = 128;                      // Minibatch size
static constexpr bool NORMALIZE_OBSERVATIONS = true;
struct PPO_PARAMETERS {
    static constexpr T GAMMA = 0.99;
    static constexpr T INITIAL_ACTION_STD = 2.0;
    static constexpr TI N_EPOCHS = 1;
    static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
};
```

### SAC/TD3-Specific
```cpp
static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;       // Buffer size
static constexpr TI N_ENVIRONMENTS = 1;                    // Usually 1
struct SAC_PARAMETERS {
    static constexpr TI ACTOR_BATCH_SIZE = 100;
    static constexpr TI CRITIC_BATCH_SIZE = 100;
};
static constexpr T ALPHA = 1.0;  // SAC entropy temperature
```

### Optimizer
```cpp
struct OPTIMIZER_PARAMETERS : rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY> {
    static constexpr T ALPHA = 0.001;  // Learning rate
};
```

## Loop Step Composition

Steps stack inside-out. Each wraps the previous config:

```cpp
// 1. Core algorithm loop
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<...>;

// 2. Experiment tracking directory structure
using LOOP_EXTRACK_CONFIG = rlt::rl::loop::steps::extrack::Config<LOOP_CORE_CONFIG>;

// 3. Periodic evaluation
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_EXTRACK_CONFIG, EVAL_PARAMS>;

// 4. Trajectory saving for replay
using LOOP_SAVE_TRAJ_CONFIG = rlt::rl::loop::steps::save_trajectories::Config<LOOP_EVAL_CONFIG, TRAJ_PARAMS>;

// 5. HDF5 checkpoints
using LOOP_CHECKPOINT_CONFIG = rlt::rl::loop::steps::checkpoint::Config<LOOP_SAVE_TRAJ_CONFIG, CKPT_PARAMS>;

// 6. Timing measurement
using LOOP_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_CHECKPOINT_CONFIG>;
```

### Required Includes Per Loop Step

| Step | Config include | Operations include |
|------|---------------|-------------------|
| extrack | (config included in core) | `rl/loop/steps/extrack/operations_cpu.h` |
| evaluation | (config included in core) | `rl/loop/steps/evaluation/operations_generic.h` |
| save_trajectories | (config included in core) | `rl/loop/steps/save_trajectories/operations_cpu.h` |
| checkpoint | `rl/loop/steps/checkpoint/config.h` | `rl/loop/steps/checkpoint/operations_cpu.h` |
| timing | (config included in core) | (included automatically) |

## Benchmark Mode

Use `#ifndef BENCHMARK` to conditionally include loop steps:
```cpp
#ifndef BENCHMARK
// Full training with eval, checkpoints, trajectories
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
#else
// Pure training speed measurement
using LOOP_CONFIG = LOOP_CORE_CONFIG;
#endif
```

Add benchmark target in CMake:
```cmake
add_executable(my_target_benchmark src/main.cpp)
target_compile_definitions(my_target_benchmark PRIVATE BENCHMARK)
```

## Multi-Seed Experiments

```cpp
int main() {
    DEVICE device;
    for (TI seed = 0; seed < 5; seed++) {
        LOOP_STATE ls;
        ls.extrack_config.name = "my_experiment";
        rlt::malloc(device, ls);
        rlt::init(device, ls, seed);
        while (!rlt::step(device, ls)) {}
        rlt::free(device, ls);
    }
}
```

## Activation Functions
```
RELU, TANH, FAST_TANH, SIGMOID, IDENTITY
```
Access: `rlt::nn::activation_functions::ActivationFunction::FAST_TANH`

## Checklist

- [ ] Correct algorithm include chain (config.h + operations)
- [ ] All loop step includes present (both config and operations)
- [ ] `STEP_LIMIT` appropriate for environment complexity
- [ ] Network dims appropriate for observation/action space
- [ ] `EPISODE_STEP_LIMIT` matches environment
- [ ] ExTrack name set before `init()`
- [ ] `malloc` → `init` → `step` loop → (`free`) order correct
- [ ] Benchmark target defined in CMake
