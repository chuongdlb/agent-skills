# Environment Catalog — Complete Reference

## Classic Control

### CartPole-v1

```python
gymnasium.make("CartPole-v1")
```

| Property | Value |
|----------|-------|
| Action Space | `Discrete(2)` — 0: push left, 1: push right |
| Observation Space | `Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf], (4,), float32)` |
| Observation | `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]` |
| Reward | +1 per step (alive bonus) |
| Termination | Pole angle > 12 deg, cart position > 2.4 |
| Truncation | 500 steps |
| Reward Threshold | 475.0 |

### MountainCar-v0

```python
gymnasium.make("MountainCar-v0")
```

| Property | Value |
|----------|-------|
| Action Space | `Discrete(3)` — 0: left, 1: no-op, 2: right |
| Observation Space | `Box([-1.2, -0.07], [0.6, 0.07], (2,), float32)` |
| Observation | `[car_position, car_velocity]` |
| Reward | -1 per step |
| Termination | Position >= 0.5 (reached flag) |
| Truncation | 200 steps |
| Reward Threshold | -110.0 |

### MountainCarContinuous-v0

```python
gymnasium.make("MountainCarContinuous-v0")
```

| Property | Value |
|----------|-------|
| Action Space | `Box(-1.0, 1.0, (1,), float32)` — force |
| Observation Space | `Box([-1.2, -0.07], [0.6, 0.07], (2,), float32)` |
| Reward | 100 at goal minus action cost (action^2 * 0.1) |
| Truncation | 999 steps |
| Reward Threshold | 90.0 |

### Pendulum-v1

```python
gymnasium.make("Pendulum-v1")
# kwargs: g=10.0
```

| Property | Value |
|----------|-------|
| Action Space | `Box(-2.0, 2.0, (1,), float32)` — torque |
| Observation Space | `Box(-1, 1, (3,), float64)` — [cos(θ), sin(θ), θ_dot] |
| Reward | `-(θ² + 0.1*θ_dot² + 0.001*torque²)` |
| Truncation | 200 steps |
| Reward Threshold | None |
| Key Kwargs | `g`: gravity (default 10.0) |

### Acrobot-v1

```python
gymnasium.make("Acrobot-v1")
```

| Property | Value |
|----------|-------|
| Action Space | `Discrete(3)` — torque: -1, 0, +1 |
| Observation Space | `Box(6,)` — [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot] |
| Reward | -1 per step |
| Termination | Tip reaches target height |
| Truncation | 500 steps |
| Reward Threshold | -100.0 |

## Box2D

### LunarLander-v3

```python
gymnasium.make("LunarLander-v3")
# kwargs: continuous=False, gravity=-10.0, enable_wind=False,
#         wind_power=15.0, turbulence_power=1.5
```

| Property | Value |
|----------|-------|
| Action Space | `Discrete(4)` or `Box(2,)` if continuous |
| Observation Space | `Box(8,)` — [x, y, vx, vy, angle, angular_vel, left_leg, right_leg] |
| Reward | +100..+140 landing, -100 crash, shaping for distance/velocity |
| Truncation | 1000 steps |
| Reward Threshold | 200.0 |
| Key Kwargs | `continuous`, `gravity`, `enable_wind`, `wind_power`, `turbulence_power` |

### BipedalWalker-v3

```python
gymnasium.make("BipedalWalker-v3")
# BipedalWalkerHardcore-v3 for harder version with obstacles
```

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, (4,), float32)` — joint torques |
| Observation Space | `Box(24,)` — hull angle/vel, joint angles/speeds, lidar |
| Reward | +300 for reaching far right, -100 for falling |
| Truncation | 1600 steps (2000 for Hardcore) |
| Reward Threshold | 300.0 |

### CarRacing-v3

```python
gymnasium.make("CarRacing-v3")
# kwargs: continuous=True, lap_complete_percent=0.95, domain_randomize=False
```

| Property | Value |
|----------|-------|
| Action Space | `Box([-1, 0, 0], [1, 1, 1], (3,))` — [steering, gas, brake] |
| Observation Space | `Box(0, 255, (96, 96, 3), uint8)` — top-down RGB |
| Reward | -0.1 per frame, +1000/N per track tile visited |
| Truncation | 1000 steps |
| Reward Threshold | 900.0 |
| Key Kwargs | `continuous`, `lap_complete_percent`, `domain_randomize` |

## Toy Text

### FrozenLake-v1

```python
gymnasium.make("FrozenLake-v1")
# kwargs: map_name="4x4", is_slippery=True, desc=None
# Also: FrozenLake8x8-v1
```

| Property | Value |
|----------|-------|
| Action Space | `Discrete(4)` — 0:left, 1:down, 2:right, 3:up |
| Observation Space | `Discrete(16)` (4x4) or `Discrete(64)` (8x8) |
| Reward | +1 at goal, 0 otherwise |
| Termination | Reach goal (G) or fall in hole (H) |
| Truncation | 100 steps (4x4) or 200 steps (8x8) |
| Key Kwargs | `map_name` ("4x4" or "8x8"), `is_slippery`, `desc` (custom map) |

### Taxi-v3

```python
gymnasium.make("Taxi-v3")
```

| Property | Value |
|----------|-------|
| Action Space | `Discrete(6)` — south, north, east, west, pickup, dropoff |
| Observation Space | `Discrete(500)` — encoded state |
| Reward | -1/step, +20 deliver, -10 illegal pickup/dropoff |
| Truncation | 200 steps |
| Reward Threshold | 8.0 |

### Blackjack-v1

```python
gymnasium.make("Blackjack-v1")
# kwargs: natural=False, sab=False
```

| Property | Value |
|----------|-------|
| Action Space | `Discrete(2)` — 0: stick, 1: hit |
| Observation Space | `Tuple(Discrete(32), Discrete(11), Discrete(2))` |
| Observation | `(player_sum, dealer_card, usable_ace)` |
| Reward | +1 win, -1 lose, 0 draw |
| Max Steps | None (episodic) |

### CliffWalking-v1

```python
gymnasium.make("CliffWalking-v1")
```

| Property | Value |
|----------|-------|
| Action Space | `Discrete(4)` — 0:up, 1:right, 2:down, 3:left |
| Observation Space | `Discrete(48)` — 4x12 grid |
| Reward | -1/step, -100 for cliff |
| Termination | Reach goal or fall off cliff |
| Reward Threshold | -13.0 |

## MuJoCo

### HalfCheetah-v5

```python
gymnasium.make("HalfCheetah-v5")
# kwargs: xml_file, frame_skip=5, forward_reward_weight=1.0,
#         ctrl_cost_weight=0.1, reset_noise_scale=0.1,
#         exclude_current_positions_from_observation=True
```

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, (6,), float32)` |
| Obs Space | `Box(17,)` (v5) or `Box(17,)` (v4) |
| Reward | forward_velocity * weight - ctrl_cost |
| Truncation | 1000 steps |
| Reward Threshold | 4800.0 |

### Hopper-v5

```python
gymnasium.make("Hopper-v5")
# kwargs: forward_reward_weight=1.0, ctrl_cost_weight=1e-3,
#         healthy_reward=1.0, terminate_when_unhealthy=True,
#         healthy_state_range=(-100, 100), healthy_z_range=(0.7, inf),
#         healthy_angle_range=(-0.2, 0.2), reset_noise_scale=5e-3
```

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, (3,), float32)` |
| Obs Space | `Box(11,)` |
| Reward | forward_velocity + healthy_reward - ctrl_cost |
| Termination | Unhealthy state (falling) |
| Truncation | 1000 steps |
| Reward Threshold | 3800.0 |

### Walker2d-v5

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, (6,), float32)` |
| Obs Space | `Box(17,)` |
| Reward | forward_velocity + healthy_reward - ctrl_cost |
| Termination | Unhealthy z-height or angle |
| Truncation | 1000 steps |

### Ant-v5

```python
gymnasium.make("Ant-v5")
# kwargs: use_contact_forces=False, ctrl_cost_weight=0.5
```

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, (8,), float32)` |
| Obs Space | `Box(27,)` (v5, without contacts) |
| Reward | forward_velocity + healthy_reward - ctrl_cost - contact_cost |
| Termination | Unhealthy z-height |
| Truncation | 1000 steps |
| Reward Threshold | 6000.0 |
| Key Kwargs | `use_contact_forces` (changes obs dim) |

### Humanoid-v5

| Property | Value |
|----------|-------|
| Action Space | `Box(-0.4, 0.4, (17,), float32)` |
| Obs Space | `Box(376,)` |
| Reward | forward_velocity + healthy_reward - ctrl_cost - contact_cost |
| Termination | Unhealthy z-height |
| Truncation | 1000 steps |

### HumanoidStandup-v5

| Property | Value |
|----------|-------|
| Action Space | `Box(-0.4, 0.4, (17,), float32)` |
| Obs Space | `Box(376,)` |
| Reward | z_height + healthy_reward - ctrl_cost - contact_cost |
| Truncation | 1000 steps |

### Swimmer-v5

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, (2,), float32)` |
| Obs Space | `Box(8,)` |
| Reward | forward_velocity - ctrl_cost |
| Truncation | 1000 steps |
| Reward Threshold | 360.0 |

### Reacher-v5

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, (2,), float32)` |
| Obs Space | `Box(10,)` |
| Reward | -distance_to_target - ctrl_cost |
| Truncation | 50 steps |
| Reward Threshold | -3.75 |

### Pusher-v5

| Property | Value |
|----------|-------|
| Action Space | `Box(-2, 2, (7,), float32)` |
| Obs Space | `Box(23,)` |
| Reward | -distance(tip,object) - distance(object,goal) |
| Truncation | 100 steps |
| Reward Threshold | 0.0 |

### InvertedPendulum-v5

| Property | Value |
|----------|-------|
| Action Space | `Box(-3, 3, (1,), float32)` |
| Obs Space | `Box(4,)` |
| Reward | +1 per step (alive) |
| Termination | Angle too large |
| Truncation | 1000 steps |
| Reward Threshold | 950.0 |

### InvertedDoublePendulum-v5

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, (1,), float32)` |
| Obs Space | `Box(9,)` |
| Reward | +10 per step - distance_penalty - velocity_penalty |
| Termination | Tip y-position below threshold |
| Truncation | 1000 steps |
| Reward Threshold | 9100.0 |
