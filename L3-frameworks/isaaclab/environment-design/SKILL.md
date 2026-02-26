---
name: isaaclab-environment-design
description: >
  Designs IsaacLab environments using Manager-Based or Direct paradigms, composes InteractiveScene, registers environments, configures decimation, implements DirectMARLEnv.
layer: L3
domain: [robotics, manipulation, locomotion, general-rl]
source-project: IsaacLab
depends-on: [isaacsim-simulation-core, isaaclab-configclass-and-utilities, rl-theory-analyzer, gymnasium-core-api]
tags: [environment, gymnasium, configclass, marl]
---

# IsaacLab Environment Design

IsaacLab provides two paradigms for building RL environments. Both share the same simulation backbone (`InteractiveScene`, `SimulationCfg`, physics stepping) but differ in how MDP logic is organized.

## Paradigm Comparison

| Aspect | Manager-Based | Direct |
|--------|--------------|--------|
| Base class | `ManagerBasedRLEnv` | `DirectRLEnv` / `DirectMARLEnv` |
| Config class | `ManagerBasedRLEnvCfg` | `DirectRLEnvCfg` / `DirectMARLEnvCfg` |
| Obs/Rewards/Terms | Declared as manager configs with `ObsTermCfg`, `RewTermCfg`, etc. | User implements `_get_observations()`, `_get_rewards()`, `_get_dones()` |
| Actions | `ActionTermCfg` in `ActionsCfg` | User implements `_pre_physics_step()`, `_apply_action()` |
| Scene setup | Fully declarative via `InteractiveSceneCfg` subclass | Override `_setup_scene()` for manual spawning and cloning |
| gym.register entry_point | `"isaaclab.envs:ManagerBasedRLEnv"` | `f"{__name__}.my_env:MyEnv"` (your class) |
| Multi-agent | Not supported | `DirectMARLEnv` with per-agent spaces |
| Best for | Rapid prototyping, reusable MDP terms | Maximum control, JIT-compiled rewards, MARL |

## Manager-Based Workflow

Configure the environment entirely through `@configclass` configs. The framework instantiates and manages all MDP components.

### ManagerBasedRLEnvCfg

Extends `ManagerBasedEnvCfg` with RL-specific fields:

```python
from dataclasses import MISSING
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    # Scene (required)
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=4.0)
    # MDP components (required)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # Optional
    events: EventCfg = EventCfg()
    commands: object | None = None
    curriculum: object | None = None

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.viewer.eye = (8.0, 0.0, 5.0)
```

Key fields inherited from `ManagerBasedEnvCfg`: `viewer`, `sim`, `seed`, `decimation`, `scene`, `observations`, `actions`, `events`, `recorders`, `num_rerenders_on_reset`, `wait_for_textures`, `xr`, `teleop_devices`, `export_io_descriptors`, `log_dir`.

Additional RL fields: `ui_window_class_type`, `is_finite_horizon`, `episode_length_s`, `rewards`, `terminations`, `curriculum`, `commands`.

### Manager Registration Order

Managers are loaded in this order (dependency-sensitive):
1. Command manager
2. Observation manager, Action manager, Event manager, Recorder manager (via `super().load_managers()`)
3. Termination manager
4. Reward manager
5. Curriculum manager

## Direct Workflow

Subclass `DirectRLEnv` and implement abstract methods directly. Gives maximum control over MDP logic.

### DirectRLEnvCfg

```python
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class MyDirectEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_space = 7          # int for Box, {N} for Discrete, [int,...] for multi-dim
    observation_space = 23
    state_space = 0           # 0 = no critic state space

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0)

    robot_cfg: ArticulationCfg = MY_ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
```

### DirectRLEnv Abstract Methods

| Method | Signature | Called | Purpose |
|--------|-----------|--------|---------|
| `_setup_scene` | `(self) -> None` | Once at init | Spawn assets, clone envs, register to scene |
| `_pre_physics_step` | `(self, actions: Tensor) -> None` | Once per step | Store/preprocess actions before decimation loop |
| `_apply_action` | `(self) -> None` | Each physics substep | Write action targets to sim |
| `_get_observations` | `(self) -> dict[str, Tensor]` | After step/reset | Return `{"policy": obs_tensor}` |
| `_get_rewards` | `(self) -> Tensor` | After step | Return `(num_envs,)` reward tensor |
| `_get_dones` | `(self) -> tuple[Tensor, Tensor]` | After step | Return `(terminated, time_outs)` bool tensors |
| `_reset_idx` | `(self, env_ids) -> None` | On reset | Reset specific envs (call `super()._reset_idx(env_ids)` first) |

## DirectMARLEnv (Multi-Agent)

For cooperative or competitive multi-agent tasks. Based on PettingZoo Parallel API concepts.

### DirectMARLEnvCfg

```python
@configclass
class MyMARLEnvCfg(DirectMARLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    possible_agents = ["agent_0", "agent_1"]
    action_spaces = {"agent_0": 3, "agent_1": 2}
    observation_spaces = {"agent_0": 10, "agent_1": 8}
    state_space = -1   # -1 = auto-concat all agent obs; 0 = none; >0 = custom dim
```

Abstract methods return dicts keyed by agent ID:
- `_get_observations() -> dict[AgentID, Tensor]`
- `_get_rewards() -> dict[AgentID, Tensor]`
- `_get_dones() -> tuple[dict[AgentID, Tensor], dict[AgentID, Tensor]]`
- `_pre_physics_step(actions: dict[AgentID, Tensor])`

Per-agent noise models: `observation_noise_model` and `action_noise_model` are `dict[AgentID, NoiseModelCfg | None]`.

## InteractiveScene Composition

The scene is defined by subclassing `InteractiveSceneCfg` and adding named attributes. Attribute order matters -- recommended: terrain, articulations/rigid bodies, sensors, lights.

```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    # terrain
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    # articulation
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # rigid object
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), ...),
    )
    # sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", ...)
    # light (non-physics)
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=500.0),
    )
```

### InteractiveSceneCfg Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | `int` | MISSING | Number of parallel environments |
| `env_spacing` | `float` | MISSING | Distance between env origins |
| `lazy_sensor_update` | `bool` | `True` | Only update sensors when `.data` accessed |
| `replicate_physics` | `bool` | `True` | Clone identical physics schemas (faster) |
| `filter_collisions` | `bool` | `True` | Disable collisions between cloned envs |
| `clone_in_fabric` | `bool` | `False` | Use Fabric for faster cloning (requires `replicate_physics=True`) |

### Prim Path Conventions

- `{ENV_REGEX_NS}` -- expands to `/World/envs/env_.*` for cloned per-env assets
- Non-env-scoped paths (e.g. `/World/ground`) are shared across all envs

## gym.register() Pattern

Every environment must be registered with Gymnasium to be discoverable by training scripts.

### Manager-Based Registration

```python
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env_cfg:MyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
```

### Direct Registration

```python
gym.register(
    id="Isaac-MyTask-Direct-v0",
    entry_point=f"{__name__}.my_env:MyEnv",   # points to your DirectRLEnv subclass
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env:MyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyPPORunnerCfg",
    },
)
```

Key points:
- `disable_env_checker=True` is always set (vectorized envs are not standard Gym)
- `env_cfg_entry_point` is required -- string path to config class or callable
- Agent cfg entry points follow pattern `{library}_cfg_entry_point` for PPO, `{library}_{algo}_cfg_entry_point` for other algorithms

## Decimation and Timing

```
physics_dt = sim.dt                          # e.g. 1/120 = 0.00833s
step_dt    = sim.dt * decimation             # e.g. 0.00833 * 2 = 0.01667s
render_dt  = sim.dt * sim.render_interval    # typically = step_dt

max_episode_length = ceil(episode_length_s / step_dt)
```

| Parameter | Location | Meaning |
|-----------|----------|---------|
| `sim.dt` | `SimulationCfg.dt` | Physics substep size (default 1/60) |
| `decimation` | `*EnvCfg.decimation` | Physics substeps per env step |
| `sim.render_interval` | `SimulationCfg.render_interval` | Physics substeps per render call |
| `episode_length_s` | `*RLEnvCfg.episode_length_s` | Episode duration in seconds |

Best practice: set `sim.render_interval = decimation` so rendering happens once per env step. If `render_interval < decimation`, multiple renders occur per step (warning issued).

### __post_init__ Timing Pattern

```python
def __post_init__(self) -> None:
    self.decimation = 4
    self.episode_length_s = 20.0
    self.sim.dt = 1 / 200
    self.sim.render_interval = self.decimation
```

## Hydra Overrides

Training scripts support Hydra CLI overrides for any config field:

```bash
# Override env config fields
python train.py --task Isaac-MyTask-v0 env.scene.num_envs=1024 env.decimation=4

# Override agent config fields
python train.py --task Isaac-MyTask-v0 agent.max_iterations=500

# Override nested fields
python train.py --task Isaac-MyTask-v0 env.rewards.track_velocity.weight=2.0
```

The `hydra_task_config` decorator converts configs to OmegaConf, applies overrides, then converts back:

```python
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    ...
```

## Project Templates

The `tools/template/` directory provides a generator for scaffolding new IsaacLab projects:

```bash
python -m isaaclab_tools.template  # interactive project generator
```

Available task templates:
- `manager-based_single-agent` -- manager-based cartpole with full MDP configs
- `direct_single-agent` -- direct cartpole with env class + cfg
- `direct_multi-agent` -- direct MARL cart-double-pendulum

Each template generates: env config, env class (direct only), MDP module (manager-based), `__init__.py` with `gym.register()`, and agent configs for supported RL libraries.

## Robot-Specific Config Subclass Pattern

Create task variants by subclassing and overriding in `__post_init__`:

```python
@configclass
class AnymalCFlatEnvCfg(AnymalCRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # disable sensors
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
```

Setting a scene entity or MDP term to `None` removes it from the environment.

## Reference Files

- [environment-api-reference.md](environment-api-reference.md) - Full attribute tables for all env config classes, SimulationCfg, and InteractiveSceneCfg
- [environment-templates.md](environment-templates.md) - Copy-paste skeletons for manager-based, direct, MARL envs, registration, and robot-specific subclasses

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/isaaclab/isaaclab/envs/manager_based_env_cfg.py` | ManagerBasedEnvCfg |
| `source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py` | ManagerBasedRLEnvCfg |
| `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` | ManagerBasedRLEnv class |
| `source/isaaclab/isaaclab/envs/manager_based_env.py` | ManagerBasedEnv base class |
| `source/isaaclab/isaaclab/envs/direct_rl_env.py` | DirectRLEnv class |
| `source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py` | DirectRLEnvCfg |
| `source/isaaclab/isaaclab/envs/direct_marl_env.py` | DirectMARLEnv class |
| `source/isaaclab/isaaclab/envs/direct_marl_env_cfg.py` | DirectMARLEnvCfg |
| `source/isaaclab/isaaclab/scene/interactive_scene_cfg.py` | InteractiveSceneCfg |
| `source/isaaclab/isaaclab/scene/interactive_scene.py` | InteractiveScene class |
| `source/isaaclab/isaaclab/sim/simulation_cfg.py` | SimulationCfg, PhysxCfg, RenderCfg |
| `source/isaaclab/isaaclab/envs/common.py` | ViewerCfg, SpaceType, AgentID, VecEnvObs |
| `source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py` | hydra_task_config decorator |
| `tools/template/` | Project template generator |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/` | Example manager-based env |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/` | Example direct env |
