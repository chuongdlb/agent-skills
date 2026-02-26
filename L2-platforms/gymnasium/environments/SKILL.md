---
name: gymnasium-environments
description: >
  Built-in Gymnasium environments — Classic Control, Box2D, Toy Text, MuJoCo, and functional/JAX variants with IDs, spaces, and dependency requirements.
layer: L2
domain: [general-rl]
source-project: Gymnasium
depends-on: [gymnasium-core-api, gymnasium-spaces]
tags: [environments, cartpole, mujoco, atari, classic-control]
---

# Gymnasium Built-in Environments

## Purpose

Gymnasium ships with ~25 built-in environments across 5 categories. These serve as standard benchmarks for RL algorithm development and testing, from simple control tasks (CartPole) to complex continuous locomotion (HalfCheetah, Humanoid).

## When to Use

- Testing and benchmarking RL algorithms
- Quick prototyping before moving to custom environments
- Comparing algorithm performance on standard tasks
- Teaching RL concepts with progressively harder environments

## Classic Control (no extra dependencies)

Simple environments with small state spaces, ideal for debugging and learning.

| Environment | ID | Action | Observation | Reward Threshold | Max Steps |
|------------|-----|--------|-------------|-----------------|-----------|
| CartPole | `CartPole-v1` | Discrete(2) | Box(4,) | 475.0 | 500 |
| MountainCar | `MountainCar-v0` | Discrete(3) | Box(2,) | -110.0 | 200 |
| Continuous MountainCar | `MountainCarContinuous-v0` | Box(1,) | Box(2,) | 90.0 | 999 |
| Pendulum | `Pendulum-v1` | Box(1,) | Box(3,) | None | 200 |
| Acrobot | `Acrobot-v1` | Discrete(3) | Box(6,) | -100.0 | 500 |

```python
env = gymnasium.make("CartPole-v1")
# obs: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
# action: 0 (push left) or 1 (push right)
```

## Box2D (requires `gymnasium[box2d]` / `box2d-py`)

Physics-based 2D environments with continuous control.

| Environment | ID | Action | Observation | Reward Threshold | Max Steps |
|------------|-----|--------|-------------|-----------------|-----------|
| LunarLander | `LunarLander-v3` | Discrete(4) | Box(8,) | 200.0 | 1000 |
| LunarLander Continuous | `LunarLanderContinuous-v3` | Box(2,) | Box(8,) | 200.0 | 1000 |
| BipedalWalker | `BipedalWalker-v3` | Box(4,) | Box(24,) | 300.0 | 1600 |
| BipedalWalker Hardcore | `BipedalWalkerHardcore-v3` | Box(4,) | Box(24,) | 300.0 | 2000 |
| CarRacing | `CarRacing-v3` | Box(3,) | Box(96,96,3) | 900.0 | 1000 |

```python
env = gymnasium.make("LunarLander-v3", render_mode="human")
# action: 0=noop, 1=left engine, 2=main engine, 3=right engine
# Key kwargs: gravity, enable_wind, wind_power, turbulence_power
```

## Toy Text (no extra dependencies)

Tabular/discrete environments for basic RL concepts.

| Environment | ID | Action | Observation | Reward Threshold | Max Steps |
|------------|-----|--------|-------------|-----------------|-----------|
| FrozenLake | `FrozenLake-v1` | Discrete(4) | Discrete(16) | 0.7 | 100 |
| FrozenLake 8x8 | `FrozenLake8x8-v1` | Discrete(4) | Discrete(64) | 0.85 | 200 |
| Taxi | `Taxi-v3` | Discrete(6) | Discrete(500) | 8.0 | 200 |
| Blackjack | `Blackjack-v1` | Discrete(2) | Tuple(Discrete, Discrete, Discrete) | None | None |
| CliffWalking | `CliffWalking-v1` | Discrete(4) | Discrete(48) | -13.0 | None |

```python
env = gymnasium.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
# action: 0=left, 1=down, 2=right, 3=up
```

## MuJoCo (requires `gymnasium[mujoco]` / `mujoco`)

Continuous control locomotion and manipulation tasks. Each environment has a v4 and v5 version; v5 is preferred.

| Environment | ID (v5) | Action dim | Obs dim | Reward Threshold | Max Steps |
|------------|---------|------------|---------|-----------------|-----------|
| HalfCheetah | `HalfCheetah-v5` | Box(6,) | Box(17,) | 4800.0 | 1000 |
| Hopper | `Hopper-v5` | Box(3,) | Box(11,) | 3800.0 | 1000 |
| Walker2d | `Walker2d-v5` | Box(6,) | Box(17,) | None | 1000 |
| Ant | `Ant-v5` | Box(8,) | Box(27,) | 6000.0 | 1000 |
| Humanoid | `Humanoid-v5` | Box(17,) | Box(376,) | None | 1000 |
| HumanoidStandup | `HumanoidStandup-v5` | Box(17,) | Box(376,) | None | 1000 |
| Swimmer | `Swimmer-v5` | Box(2,) | Box(8,) | 360.0 | 1000 |
| Reacher | `Reacher-v5` | Box(2,) | Box(10,) | -3.75 | 50 |
| Pusher | `Pusher-v5` | Box(7,) | Box(23,) | 0.0 | 100 |
| InvertedPendulum | `InvertedPendulum-v5` | Box(1,) | Box(4,) | 950.0 | 1000 |
| InvertedDoublePendulum | `InvertedDoublePendulum-v5` | Box(1,) | Box(9,) | 9100.0 | 1000 |

```python
env = gymnasium.make("HalfCheetah-v5", render_mode="human")
# Standard MuJoCo kwargs: xml_file, frame_skip, reset_noise_scale
# v5 adds: exclude_current_positions_from_observation
```

### v4 vs v5 Differences

- v5 uses the `mujoco` Python bindings (not `mujoco-py`)
- v5 has `exclude_current_positions_from_observation` parameter
- v5 default observation may differ in dimension from v4
- v4 environments are deprecated but still available

## Functional / JAX Environments

Stateless, JIT-compilable versions of select environments under the `phys2d/` and `tabular/` namespaces.

| Environment | ID | Notes |
|------------|-----|-------|
| CartPole (JAX) | `phys2d/CartPole-v1` | JAX-compatible, use with FunctionalJaxEnv |
| Pendulum (JAX) | `phys2d/Pendulum-v0` | JAX-compatible |
| Blackjack (tabular) | `tabular/Blackjack-v0` | Functional tabular version |
| CliffWalking (tabular) | `tabular/CliffWalking-v0` | Functional tabular version |

```python
# Use functional envs for hardware-accelerated RL
env = gymnasium.make("phys2d/CartPole-v1")
```

## Version Conventions

- Version number (`-vN`) increments when observation space, action space, reward function, or dynamics change
- Bug fixes that don't change the interface keep the same version
- Older versions remain available for reproducibility
- Always use the latest version for new projects

## Dependency Requirements

| Category | Install Command | Package |
|----------|----------------|---------|
| Classic Control | `pip install gymnasium` | (included) |
| Toy Text | `pip install gymnasium` | (included) |
| Box2D | `pip install gymnasium[box2d]` | `box2d-py`, `pygame` |
| MuJoCo | `pip install gymnasium[mujoco]` | `mujoco` |
| Atari | `pip install gymnasium[atari,accept-rom-license]` | `ale-py` |

## Key Source Files

| File | Contents |
|------|----------|
| `gymnasium/envs/__init__.py` | All environment registrations |
| `gymnasium/envs/classic_control/` | CartPole, MountainCar, Pendulum, Acrobot |
| `gymnasium/envs/box2d/` | LunarLander, BipedalWalker, CarRacing |
| `gymnasium/envs/toy_text/` | FrozenLake, Taxi, Blackjack, CliffWalking |
| `gymnasium/envs/mujoco/` | All MuJoCo environments |
| `gymnasium/envs/phys2d/` | JAX physics environments |
| `gymnasium/envs/tabular/` | Functional tabular environments |

## Reference Files

- [environment-catalog.md](environment-catalog.md) — Complete table of all environments with IDs, spaces, thresholds, steps, required deps, and key kwargs
