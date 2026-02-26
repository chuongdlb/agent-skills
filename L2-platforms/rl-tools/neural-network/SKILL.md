---
name: rltools-neural-network
description: >
  Custom neural network architectures in rl-tools — Dense, GRU, sample_and_squash layers, MLP/Sequential models, activation functions, optimizers, compile-time shapes.
layer: L2
domain: [general-rl]
source-project: rl-tools-framework
depends-on: [rltools-build]
tags: [neural-network, layers, mlp, cpp17]
---

# RL-Tools Neural Network Architecture

## Overview

Configure and customize neural network architectures in the rl-tools framework. All network dimensions are compile-time constants enabling zero-overhead abstraction. Supports MLP, Sequential, GRU, and specialized layers for RL.

## When to Use This Skill

- User wants to modify actor/critic network architecture
- User wants to add recurrent layers (GRU) for memory
- User wants to understand the layer/model system
- User wants to configure optimizers or learning rate schedules
- User asks about activation functions or network capacity

## Reference Files

### Layers:
- Dense: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn/layers/dense/`
- GRU: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn/layers/gru/`
- Sample & Squash: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn/layers/sample_and_squash/`
- Standardize: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn/layers/standardize/`
- Embedding: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn/layers/embedding/`
- TD3 Sampling: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn/layers/td3_sampling/`

### Models:
- MLP: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn_models/mlp/`
- Sequential: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn_models/sequential/`
- MLP Unconditional Stddev: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn_models/mlp_unconditional_stddev/`
- Multi-Agent Wrapper: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn_models/multi_agent_wrapper/`

### Optimizers:
- Adam: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn/optimizers/adam/`

### Activation Functions:
- `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/nn/activation_functions.h`

## Available Layers

| Layer | Purpose | Key Use Case |
|-------|---------|-------------|
| `dense` | Fully connected layer | Standard feedforward |
| `gru` | Gated Recurrent Unit | Memory/recurrent policies |
| `sample_and_squash` | Stochastic output with tanh squashing | SAC actor output |
| `standardize` | Running mean/std normalization | Observation normalization |
| `embedding` | Discrete → continuous mapping | Token embeddings |
| `td3_sampling` | Deterministic + noise | TD3 exploration |

## Available Models

### MLP (Multi-Layer Perceptron)
The standard model for most RL tasks. Configured via training parameters:

```cpp
static constexpr TI ACTOR_NUM_LAYERS = 3;    // Total layers (including output)
static constexpr TI ACTOR_HIDDEN_DIM = 64;   // Width of hidden layers
static constexpr auto ACTOR_ACTIVATION_FUNCTION =
    rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
```

Architecture for `NUM_LAYERS=3, HIDDEN_DIM=64`:
```
Input(OBS_DIM) → Dense(64, FAST_TANH) → Dense(64, FAST_TANH) → Dense(ACTION_DIM, IDENTITY)
```

### Sequential Model
For custom layer stacking beyond standard MLP:

```cpp
// Custom architecture: Standardize → Dense → GRU → Dense
using STANDARDIZE = rlt::nn::layers::standardize::Layer<...>;
using DENSE1 = rlt::nn::layers::dense::Layer<...>;
using GRU = rlt::nn::layers::gru::Layer<...>;
using DENSE2 = rlt::nn::layers::dense::Layer<...>;

using MODEL = rlt::nn_models::sequential::Build<
    STANDARDIZE,
    rlt::nn_models::sequential::Build<
        DENSE1,
        rlt::nn_models::sequential::Build<
            GRU,
            DENSE2
        >
    >
>;
```

### MLP Unconditional Stddev
Used for SAC actors — MLP for mean, separate learnable log_std parameter:

```cpp
// Automatically used by SAC's ConfigApproximatorsMLP
```

## Activation Functions

```cpp
rlt::nn::activation_functions::ActivationFunction::RELU
rlt::nn::activation_functions::ActivationFunction::TANH
rlt::nn::activation_functions::ActivationFunction::FAST_TANH   // Recommended
rlt::nn::activation_functions::ActivationFunction::SIGMOID
rlt::nn::activation_functions::ActivationFunction::IDENTITY    // Linear / output layer
```

`FAST_TANH` is recommended for training speed — uses a polynomial approximation.

## Optimizer Configuration

### Adam (Default)
```cpp
struct OPTIMIZER_PARAMETERS : rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY> {
    static constexpr T ALPHA = 0.001;      // Learning rate
    // Inherited defaults:
    // BETA_1 = 0.9
    // BETA_2 = 0.999
    // EPSILON = 1e-7 (TF style) or 1e-8 (PyTorch style)
};
```

Use `DEFAULT_PARAMETERS_TENSORFLOW` or `DEFAULT_PARAMETERS_PYTORCH` for matching epsilon conventions.

### Learning Rate Schedule (Manual)
```cpp
while (!rlt::step(device, ls)) {
    if (ls.step % 1000 == 0) {
        ls.actor_optimizer.parameters.alpha *= 0.99;
        ls.critic_optimizer.parameters.alpha *= 0.99;
    }
}
```

## Observation Normalization

For PPO, enable running mean/std normalization:
```cpp
static constexpr bool NORMALIZE_OBSERVATIONS = true;
```

This adds a `standardize` layer that tracks running statistics during training.

## Recurrent Policies (GRU)

For environments requiring memory (e.g., partially observable):
1. Use the `memory` environment as reference: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/environments/memory/`
2. Build Sequential model with GRU layer
3. Ensure proper hidden state management in the training loop

## Network Operations

```cpp
// Forward pass (training, tracks gradients)
rlt::forward(device, model, input, output);

// Evaluate (inference, no gradients)
rlt::evaluate(device, model, input, output, buffer, rng);

// Backward pass
rlt::backward(device, model, input, d_output, d_input, buffer);

// Update weights
rlt::update(device, optimizer, model);

// Zero gradients
rlt::zero_gradient(device, model);

// Memory management
rlt::malloc(device, model);
rlt::free(device, model);
```

## Compile-Time Shape System

All shapes are known at compile time:
- `Matrix<SPEC>` where SPEC encodes `ROWS`, `COLS`, element type
- Layer input/output dimensions are checked via `static_assert`
- Sequential model propagates shapes through the chain
- Mismatched dimensions produce compile errors (not runtime)

## Architecture Sizing Guidelines

| Environment Complexity | Hidden Dim | Num Layers | Notes |
|-----------------------|------------|------------|-------|
| Simple (Pendulum) | 32-64 | 2-3 | FAST_TANH |
| Medium (Acrobot, Car) | 64-128 | 3 | FAST_TANH or RELU |
| Complex (L2F, MuJoCo) | 128-256 | 3-4 | RELU or TANH |
| Memory required | 64-128 | 3 + GRU | Sequential model |
