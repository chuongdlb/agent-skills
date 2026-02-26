---
name: isaaclab-mdp-terms
description: >
  Implements MDP terms for IsaacLab manager-based environments across all 8 managers — Observation, Reward, Termination, Event, Action, Command, Curriculum, Recorder.
layer: L3
domain: [robotics, general-rl]
source-project: IsaacLab
depends-on: [isaaclab-environment-design, isaaclab-configclass-and-utilities]
tags: [mdp, observations, rewards, terminations, managers]
---

# IsaacLab MDP Terms

Manager-based environments in IsaacLab decompose the Markov Decision Process into 8 managers, each responsible for a distinct aspect of the environment logic. Terms are plugged into these managers through configuration classes.

## Architecture

```
ManagerBasedRLEnvCfg
  ├── observations: ObservationsCfg     -> ObservationManager
  ├── actions: ActionsCfg               -> ActionManager
  ├── commands: CommandsCfg             -> CommandManager
  ├── rewards: RewardsCfg               -> RewardManager
  ├── terminations: TerminationsCfg     -> TerminationManager
  ├── events: EventCfg                  -> EventManager
  ├── curriculum: CurriculumCfg         -> CurriculumManager
  └── recorder: RecorderCfg            -> RecorderManager
```

## The 8 Managers

| Manager | Role | Term Config | Term Type |
|---------|------|-------------|-----------|
| **Observation** | Compute obs tensors per group (policy, critic) | `ObservationTermCfg` | Function or class |
| **Reward** | Compute weighted scalar rewards | `RewardTermCfg` | Function or class |
| **Termination** | Compute boolean done signals | `TerminationTermCfg` | Function or class |
| **Event** | Randomize/reset simulation state | `EventTermCfg` | Function or class |
| **Action** | Process raw actions into asset commands | `ActionTermCfg` | Class only |
| **Command** | Generate goal commands (velocity, pose) | `CommandTermCfg` | Class only |
| **Curriculum** | Modify env params over training | `CurriculumTermCfg` | Function or class |
| **Recorder** | Record data at lifecycle stages | `RecorderTermCfg` | Class only |

## Manager Term Configuration Classes

### Common Base: ManagerTermBaseCfg

All function-based term configs inherit from this:

```python
@configclass
class ManagerTermBaseCfg:
    func: Callable | ManagerTermBase = MISSING
    params: dict[str, Any | SceneEntityCfg] = dict()
```

The `params` dict is passed as `**kwargs` to `func`. Any value that is a `SceneEntityCfg` is automatically resolved against the `InteractiveScene` before the term is called.

### ObservationTermCfg

```python
@configclass
class ObservationTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING     # Returns (num_envs, obs_dim)
    modifiers: list[ModifierCfg] | None = None       # Post-processing transforms
    noise: NoiseCfg | NoiseModelCfg | None = None    # Corruption model
    clip: tuple[float, float] | None = None          # Clamp after noise
    scale: tuple[float, ...] | float | None = None   # Multiply after clip
    history_length: int = 0                           # Past observations to buffer
    flatten_history_dim: bool = True                  # Flatten (N,H,D) -> (N,H*D)
```

Processing order: compute -> modifiers -> noise -> clip -> scale -> history.

### ObservationGroupCfg

Observation terms are organized into groups (e.g. `policy`, `critic`):

```python
@configclass
class ObservationGroupCfg:
    concatenate_terms: bool = True       # Concat all terms into single tensor
    concatenate_dim: int = -1            # Dimension to concat along
    enable_corruption: bool = False      # Enable noise on terms in this group
    history_length: int | None = None    # Override per-term history (None = per-term)
    flatten_history_dim: bool = True     # Override per-term flatten
```

### RewardTermCfg

```python
@configclass
class RewardTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING  # Returns (num_envs,)
    weight: float = MISSING                       # Multiplied with term output
```

### TerminationTermCfg

```python
@configclass
class TerminationTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING  # Returns (num_envs,) bool
    time_out: bool = False                        # True = episodic timeout, not failure
```

### EventTermCfg

```python
@configclass
class EventTermCfg(ManagerTermBaseCfg):
    func: Callable[..., None] = MISSING           # Returns void
    mode: str = MISSING                            # "startup", "reset", or "interval"
    interval_range_s: tuple[float, float] | None = None  # For mode="interval"
    is_global_time: bool = False                   # Same interval for all envs
    min_step_count_between_reset: int = 0          # Throttle reset-mode calls
```

### ActionTermCfg

```python
@configclass
class ActionTermCfg:
    class_type: type[ActionTerm] = MISSING  # Must subclass ActionTerm
    asset_name: str = MISSING               # Scene entity name
    debug_vis: bool = False
    clip: dict[str, tuple] | None = None    # Clip ranges by regex
```

### CommandTermCfg

```python
@configclass
class CommandTermCfg:
    class_type: type[CommandTerm] = MISSING        # Must subclass CommandTerm
    resampling_time_range: tuple[float, float] = MISSING  # Seconds
    debug_vis: bool = False
```

### CurriculumTermCfg

```python
@configclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    func: Callable[..., float | dict[str, float] | None] = MISSING
```

### RecorderTermCfg

```python
@configclass
class RecorderTermCfg:
    class_type: type[RecorderTerm] = MISSING  # Must subclass RecorderTerm
```

## SceneEntityCfg

Links a term to a scene entity (robot, sensor, object) and optionally selects specific joints or bodies:

```python
SceneEntityCfg(
    name: str,                                    # Scene entity name (e.g. "robot")
    joint_names: str | list[str] | None = None,   # Regex or list of joint names
    body_names: str | list[str] | None = None,     # Regex or list of body names
    joint_ids: list[int] | slice = slice(None),    # Auto-resolved from joint_names
    body_ids: list[int] | slice = slice(None),     # Auto-resolved from body_names
    preserve_order: bool = False,                  # Keep order from names list
)
```

The manager calls `SceneEntityCfg.resolve(scene)` at init, which converts name regexes to index lists. Term functions receive the resolved `SceneEntityCfg` with populated `joint_ids` / `body_ids`.

## Function-Based vs Class-Based Terms

### Function-based (observations, rewards, terminations, events, curriculum)

The simplest pattern. The function receives `env` as its first argument, plus any `params` as kwargs:

```python
def my_obs_term(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]
```

### Class-based (extends ManagerTermBase)

For terms that need persistent state. The class must implement `__call__` and optionally `reset`:

```python
class my_stateful_obs(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._running_mean = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is not None:
            self._running_mean[env_ids] = 0.0

    def __call__(self, env: ManagerBasedEnv, **kwargs) -> torch.Tensor:
        # compute and return observation
        ...
```

### Actions and Commands (always class-based)

These use `class_type` in their config instead of `func`, and subclass `ActionTerm` or `CommandTerm` respectively.

## Return Shapes

| Manager | Return Type | Shape |
|---------|------------|-------|
| Observation | `torch.Tensor` (float) | `(num_envs, obs_dim)` |
| Reward | `torch.Tensor` (float) | `(num_envs,)` |
| Termination | `torch.Tensor` (bool) | `(num_envs,)` |
| Event | `None` | void |
| Curriculum | `float \| dict \| None` | scalar for logging |

## Import Aliases

The codebase uses short aliases for term configs:

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
```

## Example 1: ObservationsCfg with PolicyCfg Group

```python
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Terms are evaluated in order and concatenated
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

## Example 2: RewardsCfg with Weighted Terms

```python
import math
import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    # Task rewards (positive weights)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Penalties (negative weights)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
```

## Example 3: Custom Observation Function

```python
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def ee_position_in_robot_frame(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """End-effector position relative to robot base. Returns (num_envs, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :3]
    root_pos_w = asset.data.root_pos_w
    return ee_pos_w - root_pos_w
```

## Example 4: Custom Reward Function

```python
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def reach_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Binary reward when end-effector is within threshold of target. Returns (num_envs,)."""
    robot: RigidObject = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    distance = torch.norm(robot.data.root_pos_w - target.data.root_pos_w, dim=-1)
    return (distance < threshold).float()
```

## Example 5: EventTermCfg for Randomization

```python
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

@configclass
class EventCfg:
    # mode="startup": runs once when the environment is created
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # mode="reset": runs whenever environments are reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )

    # mode="interval": runs periodically at random intervals
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
```

## Commonly Used Built-in Terms (Quick Reference)

### Observations (`isaaclab.envs.mdp`)

| Function | Description | Shape |
|----------|-------------|-------|
| `base_lin_vel` | Root linear velocity in body frame | (N, 3) |
| `base_ang_vel` | Root angular velocity in body frame | (N, 3) |
| `projected_gravity` | Gravity direction in body frame | (N, 3) |
| `joint_pos` | Joint positions | (N, num_joints) |
| `joint_pos_rel` | Joint positions minus defaults | (N, num_joints) |
| `joint_vel` | Joint velocities | (N, num_joints) |
| `last_action` | Previous action | (N, action_dim) |
| `generated_commands` | Current command vector | (N, cmd_dim) |
| `height_scan` | Terrain height scan | (N, num_rays) |

### Rewards (`isaaclab.envs.mdp`)

| Function | Description |
|----------|-------------|
| `track_lin_vel_xy_exp` | Exponential tracking of xy linear velocity commands |
| `track_ang_vel_z_exp` | Exponential tracking of yaw angular velocity commands |
| `lin_vel_z_l2` | Penalize vertical linear velocity |
| `ang_vel_xy_l2` | Penalize roll/pitch angular velocity |
| `joint_torques_l2` | Penalize joint torques |
| `action_rate_l2` | Penalize action rate of change |
| `undesired_contacts` | Penalize contacts above threshold |

### Terminations (`isaaclab.envs.mdp`)

| Function | Description |
|----------|-------------|
| `time_out` | Episode length exceeded (set `time_out=True`) |
| `bad_orientation` | Robot tilted beyond `limit_angle` |
| `illegal_contact` | Contact force exceeds `threshold` |

### Events (`isaaclab.envs.mdp`)

| Function | Typical Mode | Description |
|----------|-------------|-------------|
| `randomize_rigid_body_material` | startup | Randomize friction/restitution |
| `randomize_rigid_body_mass` | startup | Add/scale body masses |
| `reset_root_state_uniform` | reset | Reset root pose and velocity |
| `reset_joints_by_scale` | reset | Reset joint positions by scaling defaults |
| `push_by_setting_velocity` | interval | Apply random velocity pushes |

## Reference Files

- [mdp-terms-catalog.md](mdp-terms-catalog.md) - Exhaustive catalog of all built-in observation, reward, termination, event, action, command, and curriculum terms
- [mdp-terms-patterns.md](mdp-terms-patterns.md) - Code templates for writing every custom term type (observation, reward, termination, event, action, command, curriculum, recorder)

## Key Repo Paths

| Path | Description |
|------|-------------|
| `source/isaaclab/isaaclab/envs/mdp/observations.py` | Built-in observation term functions |
| `source/isaaclab/isaaclab/envs/mdp/rewards.py` | Built-in reward term functions |
| `source/isaaclab/isaaclab/envs/mdp/terminations.py` | Built-in termination term functions |
| `source/isaaclab/isaaclab/envs/mdp/events.py` | Built-in event term functions |
| `source/isaaclab/isaaclab/envs/mdp/curriculums.py` | Built-in curriculum term classes |
| `source/isaaclab/isaaclab/envs/mdp/actions/` | Built-in action term classes and configs |
| `source/isaaclab/isaaclab/envs/mdp/commands/` | Built-in command term classes and configs |
| `source/isaaclab/isaaclab/envs/mdp/recorders/` | Built-in recorder term classes and configs |
| `source/isaaclab/isaaclab/managers/manager_term_cfg.py` | All term config class definitions |
| `source/isaaclab/isaaclab/managers/scene_entity_cfg.py` | SceneEntityCfg definition and resolution |
| `source/isaaclab/isaaclab/managers/manager_base.py` | ManagerTermBase base class |
| `source/isaaclab/isaaclab/managers/observation_manager.py` | ObservationManager implementation |
| `source/isaaclab/isaaclab/managers/action_manager.py` | ActionTerm base class and ActionManager |
| `source/isaaclab/isaaclab/managers/command_manager.py` | CommandTerm base class and CommandManager |
| `source/isaaclab/isaaclab/managers/recorder_manager.py` | RecorderTerm base class and RecorderManager |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` | Reference locomotion env config |
