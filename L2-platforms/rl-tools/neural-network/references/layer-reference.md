# Layer Reference

## Dense Layer
**Location**: `rl-tools/include/rl_tools/nn/layers/dense/`

Standard fully connected layer: `output = activation(W * input + b)`

Configuration via training parameters:
```cpp
static constexpr TI ACTOR_HIDDEN_DIM = 64;
static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
```

## GRU Layer
**Location**: `rl-tools/include/rl_tools/nn/layers/gru/`

Gated Recurrent Unit for sequence modeling. Use with Sequential model for recurrent policies.

Key: maintains hidden state across timesteps within an episode. Must be reset at episode boundaries.

## Sample and Squash Layer
**Location**: `rl-tools/include/rl_tools/nn/layers/sample_and_squash/`

Used by SAC for stochastic policies:
1. Takes mean and log_std as input
2. Samples from Gaussian: `z = mean + std * epsilon`
3. Applies tanh squashing: `action = tanh(z)`
4. Computes log probability with correction for squashing

Automatically configured by `ConfigApproximatorsMLP` in SAC.

## Standardize Layer
**Location**: `rl-tools/include/rl_tools/nn/layers/standardize/`

Running mean/standard deviation normalization:
- Tracks `mean` and `std` during training
- Normalizes: `output = (input - mean) / std`
- Enabled by `NORMALIZE_OBSERVATIONS = true` in PPO config

## Embedding Layer
**Location**: `rl-tools/include/rl_tools/nn/layers/embedding/`

Maps discrete indices to continuous vectors. Used for token-based inputs.

## TD3 Sampling Layer
**Location**: `rl-tools/include/rl_tools/nn/layers/td3_sampling/`

Adds clipped Gaussian noise for TD3 exploration:
- `action = actor(state) + clip(N(0, sigma), -c, c)`

## Activation Functions

| Function | Value | Notes |
|----------|-------|-------|
| `IDENTITY` | Linear (no activation) | Output layers |
| `RELU` | max(0, x) | Standard hidden layers |
| `TANH` | tanh(x) | Bounded output [-1, 1] |
| `FAST_TANH` | Polynomial approximation | Fastest, recommended for training |
| `SIGMOID` | 1/(1+exp(-x)) | Probability output |
