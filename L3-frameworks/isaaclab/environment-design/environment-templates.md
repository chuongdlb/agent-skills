# Environment Templates

Copy-paste skeletons for creating IsaacLab environments. Replace placeholder names with your own.

## 1. Minimal Manager-Based Environment

### Scene Configuration

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Import a pre-defined robot config
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene with a robot and ground plane."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
```

### MDP Configurations

```python
import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization events."""

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )
```

### Full Environment Config

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass


@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment configuration."""

    # Scene
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=4.0)
    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
```

## 2. Minimal Direct Environment

### Config

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


@configclass
class MyDirectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # spaces
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # task parameters
    action_scale = 100.0
    max_cart_pos = 3.0
```

### Env Class

```python
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform


class MyDirectEnv(DirectRLEnv):
    cfg: MyDirectEnvCfg

    def __init__(self, cfg: MyDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # cache joint indices after sim is initialized
        self._cart_dof_idx, _ = self.robot.find_joints("slider_to_cart")
        self._pole_dof_idx, _ = self.robot.find_joints("cart_to_pole")

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        """Spawn assets, clone environments, register to scene."""
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone environments
        self.scene.clone_environments(copy_from_source=False)
        # filter collisions for CPU sim
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # register articulation with scene manager
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions before the decimation loop."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Write action targets to sim each physics substep."""
        self.robot.set_joint_effort_target(
            self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx
        )

    def _get_observations(self) -> dict:
        """Return observation dict with 'policy' key."""
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute scalar reward per environment."""
        alive = 1.0 - self.reset_terminated.float()
        terminated = -2.0 * self.reset_terminated.float()
        pole_pos = -1.0 * torch.square(self.joint_pos[:, self._pole_dof_idx[0]])
        cart_vel = -0.01 * torch.abs(self.joint_vel[:, self._cart_dof_idx[0]])
        return alive + terminated + pole_pos + cart_vel

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (terminated, time_outs) boolean tensors."""
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1
        )
        out_of_bounds = out_of_bounds | torch.any(
            torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1
        )
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specific environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            -0.25 * math.pi,
            0.25 * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
```

## 3. Minimal MARL Environment

### Config

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG  # isort:skip


@configclass
class MyMARLEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # multi-agent specification
    possible_agents = ["cart", "pendulum"]
    action_spaces = {"cart": 1, "pendulum": 1}
    observation_spaces = {"cart": 4, "pendulum": 3}
    state_space = -1  # auto-concat all agent observations

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # task parameters
    cart_action_scale = 100.0
    pendulum_action_scale = 50.0
    max_cart_pos = 3.0
```

### Env Class

```python
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform


class MyMARLEnv(DirectMARLEnv):
    cfg: MyMARLEnvCfg

    def __init__(self, cfg: MyMARLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints("slider_to_cart")
        self._pole_dof_idx, _ = self.robot.find_joints("cart_to_pole")
        self._pendulum_dof_idx, _ = self.robot.find_joints("pole_to_pendulum")

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Store per-agent actions."""
        self.actions = actions

    def _apply_action(self) -> None:
        """Apply each agent's action to the appropriate joint."""
        self.robot.set_joint_effort_target(
            self.actions["cart"] * self.cfg.cart_action_scale, joint_ids=self._cart_dof_idx
        )
        self.robot.set_joint_effort_target(
            self.actions["pendulum"] * self.cfg.pendulum_action_scale,
            joint_ids=self._pendulum_dof_idx,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return per-agent observation tensors."""
        observations = {
            "cart": torch.cat(
                (
                    self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(1),
                    self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(1),
                    self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(1),
                    self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(1),
                ),
                dim=-1,
            ),
            "pendulum": torch.cat(
                (
                    self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(1),
                    self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(1),
                    self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(1),
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Return per-agent reward tensors."""
        terminated = math.prod(self.terminated_dict.values())
        alive = 1.0 - terminated.float()
        return {
            "cart": alive + -1.0 * torch.square(self.joint_pos[:, self._pole_dof_idx[0]]),
            "pendulum": alive + -1.0 * torch.square(self.joint_pos[:, self._pendulum_dof_idx[0]]),
        }

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return per-agent (terminated, time_out) dicts."""
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1
        )
        terminated = {agent: out_of_bounds for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _get_states(self) -> torch.Tensor:
        """Only needed if state_space > 0. With state_space=-1, auto-concat is used."""
        raise NotImplementedError

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            -0.25 * math.pi, 0.25 * math.pi,
            joint_pos[:, self._pole_dof_idx].shape, joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
```

## 4. __init__.py Registration Pattern

### Manager-Based

```python
"""My custom environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

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

### Direct

```python
"""My custom direct environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-MyTask-Direct-v0",
    entry_point=f"{__name__}.my_env:MyDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env:MyDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

### Direct MARL

```python
"""My custom MARL environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-MyTask-MARL-Direct-v0",
    entry_point=f"{__name__}.my_marl_env:MyMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_marl_env:MyMARLEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)
```

### Agent Config Entry Point Naming Convention

| Library | Algorithm | Key Name |
|---------|-----------|----------|
| rsl_rl | PPO | `rsl_rl_cfg_entry_point` |
| skrl | PPO | `skrl_cfg_entry_point` |
| skrl | MAPPO | `skrl_mappo_cfg_entry_point` |
| skrl | IPPO | `skrl_ippo_cfg_entry_point` |
| sb3 | PPO | `sb3_cfg_entry_point` |
| rl_games | PPO | `rl_games_cfg_entry_point` |

Pattern: `{library}_cfg_entry_point` for the default algorithm (PPO), `{library}_{algorithm}_cfg_entry_point` for others.

## 5. Robot-Specific Config Subclass Pattern

Override a base environment config for a specific robot or terrain variant using `__post_init__`.

### Flat Terrain Variant

```python
from isaaclab.utils import configclass


@configclass
class MyRobotFlatEnvCfg(MyRobotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # change terrain to flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # disable height scanner sensor
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # adjust reward weights for flat terrain
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.feet_air_time.weight = 0.5

        # no terrain curriculum
        self.curriculum.terrain_levels = None
```

### Play/Evaluation Variant

```python
class MyRobotFlatEnvCfg_PLAY(MyRobotFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # fewer envs for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable observation noise
        self.observations.policy.enable_corruption = False

        # disable domain randomization
        self.events.base_external_force_torque = None
        self.events.push_robot = None
```

### Different Robot with .replace()

```python
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG


@configclass
class AnymalDRoughEnvCfg(AnymalCRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # swap robot
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # adjust actuator params if needed
        self.scene.robot.actuators["legs"].effort_limit = 120.0
```

### Multiple Registrations for Variants

```python
gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalCFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
    },
)
```
