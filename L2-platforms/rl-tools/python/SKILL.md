---
name: rltools-python
description: >
  Python interface and integration for rl-tools — Gymnasium integration, pybind11 bindings, HDF5 checkpoints, tinyrl PyTorch extension.
layer: L2
domain: [general-rl]
source-project: rl-tools-framework
depends-on: [rltools-build, gymnasium-core-api]
tags: [python, bindings, gymnasium, pytorch]
---

# RL-Tools Python Integration

## Overview

Use rl-tools from Python through multiple binding approaches. Train policies, integrate with Gymnasium environments, load checkpoints, and deploy pre-trained models in Python applications.

## When to Use This Skill

- User wants to train rl-tools algorithms from Python
- User wants to integrate with Gymnasium or PyTorch
- User wants to load/save HDF5 checkpoints in Python
- User wants to use the L2F simulator from Python
- User wants to deploy a pre-trained policy in Python

## Available Python Packages

### 1. Official Bindings (`python-interface/`)

**Location**: `/home/ai/source/rl-tools-framework/python-interface/`

Primary Python interface using pybind11. Provides:
- SAC training from Python
- HDF5 checkpoint save/load
- Environment interaction

**Install:**
```bash
cd /home/ai/source/rl-tools-framework/python-interface
pip install .
# Development mode:
pip install -e .
```

**Documentation**: `/home/ai/source/rl-tools-framework/docs.rl.tools/docs/09-Python Interface.ipynb`

### 2. TinyRL (`tinyrl/`)

**Location**: `/home/ai/source/rl-tools-framework/tinyrl/`

PyTorch C++ extension wrapper. Integrates rl-tools training with the PyTorch ecosystem using `torch.utils.cpp_extension` for JIT compilation.

**Install:**
```bash
cd /home/ai/source/rl-tools-framework/tinyrl
pip install .
```

### 3. Alternative Bindings (`py-rl-tools/`)

**Location**: `/home/ai/source/rl-tools-framework/py-rl-tools/`

Alternative pybind11-based bindings with a different API surface.

### 4. L2F Simulator (`l2f/`)

**Location**: `/home/ai/source/rl-tools-framework/l2f/`

Python bindings for the Learning to Fly quadrotor simulator. Gymnasium-compatible interface for drone simulation.

**Install:**
```bash
cd /home/ai/source/rl-tools-framework/l2f
pip install .
```

### 5. Foundation Policy (`foundation-policy-python/`)

**Location**: `/home/ai/source/rl-tools-framework/foundation-policy-python/`

Pre-trained quadrotor control policy as a ready-to-use Python package.

## Usage Patterns

### Training with Official Bindings

```python
import rl_tools

# Configure and train
env = rl_tools.Environment(...)
config = rl_tools.SACConfig(
    step_limit=20000,
    actor_hidden_dim=64,
    critic_hidden_dim=64,
)
trainer = rl_tools.Trainer(env, config)
trainer.train()
```

### Gymnasium Integration

```python
import gymnasium as gym
from l2f import L2FEnv

# L2F provides Gymnasium-compatible wrapper
env = L2FEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

### HDF5 Checkpoint Interop

Checkpoints saved by C++ training can be loaded in Python:

```python
import h5py

with h5py.File("checkpoint.h5", "r") as f:
    # Navigate checkpoint structure
    print(list(f.keys()))

    # Access actor weights
    actor_weights = f["actor/layers/0/weights"][:]
    actor_biases = f["actor/layers/0/biases"][:]
```

### Foundation Policy Inference

```python
from foundation_policy import QuadrotorPolicy

policy = QuadrotorPolicy.load_pretrained()
action = policy.predict(observation)
```

## Build Requirements

All Python packages require:
- Python 3.8+
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- pybind11 (installed automatically via pip)
- CMake 3.10+

Optional:
- PyTorch (for tinyrl)
- HDF5 libraries (for checkpoint support)

## Key Documentation

- Python interface notebook: `/home/ai/source/rl-tools-framework/docs.rl.tools/docs/09-Python Interface.ipynb`
- ExTrack specification: `/home/ai/source/rl-tools-framework/docs.rl.tools/docs/10-Experiment Tracking.rst`
- Full docs site source: `/home/ai/source/rl-tools-framework/documentation/`

## Workflow: Python Training → C++ Deployment

1. Prototype in Python using official bindings or tinyrl
2. Save checkpoint as HDF5
3. Convert to C++ header for embedded deployment
4. Use `rltools-deploy-hardware` skill for firmware integration
