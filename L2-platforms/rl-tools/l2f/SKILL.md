---
name: rltools-l2f
description: >
  Learning to Fly (L2F) quadrotor development — quadrotor simulation, drone control policies, system identification, domain randomization, sim-to-real deployment.
layer: L2
domain: [drones, sim-to-real]
source-project: rl-tools-framework
depends-on: [rltools-build, rltools-training, rltools-environment]
tags: [quadrotor, l2f, sim-to-real, crazyflie]
---

# RL-Tools Learning to Fly (L2F)

## Overview

L2F is the quadrotor simulation, training, and deployment pipeline within rl-tools. It covers the full stack from physics simulation through policy training to hardware deployment on real drones, with domain randomization for sim-to-real transfer.

## When to Use This Skill

- User wants to train a quadrotor control policy
- User wants to configure quadrotor physics parameters
- User wants to run system identification from flight logs
- User wants to deploy a drone policy to hardware
- User wants to set up the L2F simulator in Python or browser
- User asks about domain randomization or sim-to-real

## Repository Map

| Repo | Role |
|------|------|
| `rl-tools/include/rl_tools/rl/environments/l2f/` | L2F environment definition |
| `rl-tools/include/rl_tools/inference/applications/l2f/` | L2F inference application |
| `l2f/` | L2F Python bindings |
| `learning-to-fly/` | Complete training-to-deployment pipeline |
| `raptor/` | Foundation policy (pre-trained) |
| `foundation-policy-python/` | Pre-trained policy Python package |
| `sysid.tools/` | System identification from flight logs |
| `l2f-studio/` | Browser-based drone simulator |
| `crazyflie-controller/` | Crazyflie hardware deployment |
| `betaflight-firmware/` | Betaflight hardware deployment |
| `px4/` | PX4 hardware deployment |
| `esp32/` | ESP32 deployment |

## L2F Environment

**Location**: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/environments/l2f/`

### State Space
```
position[3]          — x, y, z
orientation[4]       — Quaternion (w, x, y, z)
linear_velocity[3]   — vx, vy, vz
angular_velocity[3]  — wx, wy, wz
```
Total state dim: 13

### Observation Space
Configurable. Typical observation includes:
- Angular velocity (3)
- Orientation (rotation matrix: 9 or quaternion: 4)
- Linear velocity (3)
- Position (3)
- Previous action history (4 × N_history)

### Action Space
4-dimensional: Individual motor commands (normalized [-1, 1])

### Physics
- Rigid body dynamics with motor model
- Configurable mass, inertia, motor constants
- Aerodynamic drag model
- Motor response dynamics (first-order)

## Training Pipeline

### Step 1: Configure L2F Environment

Read the L2F environment files to understand available parameters:
```bash
ls /home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/environments/l2f/
```

### Step 2: Train Policy

Use SAC or TD3 (recommended for continuous control):
- Typical training: 100k-1M steps
- Network: 128-256 hidden dim, 3 layers
- Action history length: 16-32 steps for temporal policies

### Step 3: Domain Randomization

For sim-to-real transfer, randomize during training:
- Mass and inertia
- Motor constants and response time
- Aerodynamic coefficients
- Initial state distribution
- Sensor noise

Use `sample_initial_parameters()` to randomize per-episode.

### Step 4: System Identification

Use `sysid.tools/` to identify physical parameters from flight logs:
```bash
cd /home/ai/source/rl-tools-framework/sysid.tools/
```

### Step 5: Deploy to Hardware

Export policy and use the appropriate adapter. See `rltools-deploy-hardware` skill.

## Python Simulator

```python
from l2f import L2FEnv

env = L2FEnv()
obs, info = env.reset()

for _ in range(1000):
    action = policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Browser Simulator

**Location**: `/home/ai/source/rl-tools-framework/l2f-studio/`

Three.js + WebAssembly drone simulator:
- Real-time 3D visualization
- Gamepad support for manual control
- Policy inference via WASM-compiled C++

## Inference Application

The L2F inference application provides a complete wrapper for policy deployment:

**Location**: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/inference/applications/l2f/`

### C Backend (`c_backend.h`)
- Executor initialization and memory management
- Observation/action mapping between C and C++ types
- Timing synchronization (intermediate vs native frequency)
- Status monitoring (jitter, bias)

### C Interface (`c_interface.h`)
- `RLtoolsInferenceApplicationsL2FObservation` struct
- `RLtoolsInferenceApplicationsL2FAction` struct
- `rl_tools_inference_applications_l2f_init/reset/control` functions

## Foundation Policy

The `raptor/` repository contains pre-trained foundation policies for quadrotors:
- Trained with extensive domain randomization
- Generalizes across drone configurations
- Available as Python package (`foundation-policy-python/`)
- Can be fine-tuned for specific hardware

## Hardware Deployment Summary

| Platform | Controller | Build Command |
|----------|-----------|--------------|
| Crazyflie | `rl_tools_controller.c` | `make` |
| PX4 | `RLtoolsPolicy.cpp` | `make px4_fmu-v6c_default EXTERNAL_MODULES_LOCATION=...` |
| ESP32 | `main.c` | `idf.py build` |
| Betaflight | Flight controller integration | `make TARGET=<board>` |

## Key Design Decisions

- **100 Hz native control**: Training at 100 Hz, hardware may run faster with interpolation
- **Action history**: Critical for temporal policies — captures motor dynamics
- **Quaternion orientation**: w, x, y, z convention throughout
- **Normalized actions**: [-1, 1] mapped to motor commands in adapter
- **Domain randomization**: Essential for sim-to-real transfer
