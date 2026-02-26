# MDP Terms Patterns

Code templates for writing each custom term type. Copy and adapt these patterns to create new MDP terms.

---

## 1. Custom Observation Function

The simplest observation term pattern. Receives `env` and optional params, returns `(num_envs, obs_dim)` float tensor.

```python
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def joint_pos_sine_encoding(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sine-encoded joint positions. Returns (num_envs, num_joints * 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.cat([torch.sin(joint_pos), torch.cos(joint_pos)], dim=-1)
```

Usage in config:

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm

joint_encoding = ObsTerm(func=joint_pos_sine_encoding)
# Or with specific joints:
arm_encoding = ObsTerm(
    func=joint_pos_sine_encoding,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*"])},
)
```

### With @generic_io_descriptor (for ONNX/IO metadata export)

```python
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape, record_dtype

@generic_io_descriptor(
    units="rad",
    observation_type="JointState",
    on_inspect=[record_shape, record_dtype],
)
def joint_pos_sine_encoding(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sine-encoded joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.cat([torch.sin(joint_pos), torch.cos(joint_pos)], dim=-1)
```

---

## 2. Custom Observation Class (ManagerTermBase subclass)

Use when you need persistent state (running averages, caches, pre-loaded models).

```python
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg


class joint_velocity_running_avg(ManagerTermBase):
    """Exponential moving average of joint velocities.

    Provides a smoothed velocity signal that is more stable than raw velocities.

    Params:
        alpha: EMA smoothing factor. Lower = smoother. Default 0.1.
        asset_cfg: The robot asset configuration.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Read params from config
        self._alpha = cfg.params.get("alpha", 0.1)
        asset_name = cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name
        asset: Articulation = env.scene[asset_name]
        num_joints = asset.num_joints
        # Persistent state
        self._ema = torch.zeros(env.num_envs, num_joints, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._ema[:] = 0.0
        else:
            self._ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedEnv,
        alpha: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
        self._ema = self._alpha * vel + (1.0 - self._alpha) * self._ema
        return self._ema.clone()
```

Usage in config:

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm

smooth_vel = ObsTerm(func=joint_velocity_running_avg, params={"alpha": 0.2})
```

---

## 3. Custom Reward Function

Receives `env: ManagerBasedRLEnv` and params, returns `(num_envs,)` float tensor.

```python
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def ee_reach_target_exp(
    env: ManagerBasedRLEnv,
    std: float,
    ee_body_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for reaching a target pose.

    Computes exp(-||ee_pos - target_pos||^2 / std^2). Returns (num_envs,).

    Args:
        std: Standard deviation for the exponential kernel.
        ee_body_name: Name of the end-effector body in the asset.
        command_name: Name of the command term providing target positions.
        asset_cfg: The robot scene entity.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get end-effector position (select the body by name)
    ee_cfg = SceneEntityCfg(asset_cfg.name, body_names=[ee_body_name])
    ee_cfg.resolve(env.scene)
    ee_pos_w = asset.data.body_pos_w[:, ee_cfg.body_ids[0], :3]
    # Get target position from command manager
    target = env.command_manager.get_command(command_name)[:, :3]
    # Compute exponential reward
    distance_sq = torch.sum(torch.square(ee_pos_w - target), dim=-1)
    return torch.exp(-distance_sq / std**2)
```

Usage in config:

```python
from isaaclab.managers import RewardTermCfg as RewTerm

reach_reward = RewTerm(
    func=ee_reach_target_exp,
    weight=5.0,
    params={"std": 0.1, "ee_body_name": "panda_hand", "command_name": "ee_pose"},
)
```

---

## 4. Custom Reward Class

Use when the reward requires persistent state or one-time setup.

```python
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg


class cumulative_distance_penalty(ManagerTermBase):
    """Penalize cumulative distance traveled since last reset.

    Params:
        max_distance: Distance at which penalty saturates. Default 10.0.
        asset_cfg: The robot scene entity.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._total_distance = torch.zeros(env.num_envs, device=env.device)
        self._prev_pos = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._total_distance[:] = 0.0
        else:
            self._total_distance[env_ids] = 0.0
            asset = self._env.scene["robot"]
            self._prev_pos[env_ids] = asset.data.root_pos_w[env_ids].clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        max_distance: float = 10.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        pos = asset.data.root_pos_w[:, :3]
        self._total_distance += torch.norm(pos - self._prev_pos, dim=-1)
        self._prev_pos = pos.clone()
        return torch.clamp(self._total_distance / max_distance, max=1.0)
```

---

## 5. Custom Termination Function

Returns `(num_envs,)` boolean tensor. True = environment should terminate.

```python
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def base_contact_with_ground(
    env: ManagerBasedRLEnv,
    min_height: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot base drops below a minimum height.

    Args:
        min_height: Minimum acceptable base height (meters).
        asset_cfg: The robot scene entity.

    Returns:
        Boolean tensor of shape (num_envs,). True where base is too low.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < min_height
```

Usage in config:

```python
from isaaclab.managers import TerminationTermCfg as DoneTerm

# Regular termination (counts as failure)
fell_over = DoneTerm(func=base_contact_with_ground, params={"min_height": 0.15})

# Timeout termination (does not count as failure for RL)
time_limit = DoneTerm(func=mdp.time_out, time_out=True)
```

---

## 6. Custom Event Function (mode="startup")

Called once when the environment is created. Used for one-time randomization.
Receives `env` and `env_ids` (all env indices at startup).

```python
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


def randomize_joint_stiffness_startup(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    stiffness_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize joint stiffness once at environment creation.

    Args:
        env: The environment instance.
        env_ids: The environment indices (all envs at startup).
        stiffness_range: (min, max) range for uniform sampling.
        asset_cfg: The robot scene entity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_joints = asset.num_joints
    random_stiffness = math_utils.sample_uniform(
        stiffness_range[0], stiffness_range[1],
        size=(len(env_ids), num_joints),
        device=env.device,
    )
    asset.write_joint_stiffness_to_sim(random_stiffness, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
```

Usage in config:

```python
from isaaclab.managers import EventTermCfg as EventTerm

stiffness_rand = EventTerm(
    func=randomize_joint_stiffness_startup,
    mode="startup",
    params={"stiffness_range": (80.0, 120.0)},
)
```

---

## 7. Custom Event Function (mode="reset")

Called whenever environments are reset. `env_ids` contains indices of resetting envs.

```python
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


def reset_object_on_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset an object to a random position on a table surface.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        position_range: Dict with "x", "y", "z" keys mapping to (min, max) ranges.
        asset_cfg: The object scene entity.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    object_asset: RigidObject = env.scene[asset_cfg.name]
    default_state = object_asset.data.default_root_state[env_ids].clone()

    # Randomize position
    for axis_idx, axis in enumerate(["x", "y", "z"]):
        if axis in position_range:
            low, high = position_range[axis]
            default_state[:, axis_idx] += torch.empty(len(env_ids), device=env.device).uniform_(low, high)

    # Add environment origins
    default_state[:, :3] += env.scene.env_origins[env_ids]

    # Write back
    object_asset.write_root_pose_to_sim(default_state[:, :7], env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)
```

Usage in config:

```python
from isaaclab.managers import EventTermCfg as EventTerm

reset_obj = EventTerm(
    func=reset_object_on_table,
    mode="reset",
    params={
        "position_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.05)},
        "asset_cfg": SceneEntityCfg("object"),
    },
)
```

---

## 8. Custom Event Function (mode="interval")

Called periodically at random intervals during the episode. Used for ongoing disturbances.

```python
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def apply_random_force_impulse(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    force_magnitude_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply a random horizontal force impulse to the robot base.

    Args:
        env: The environment instance.
        env_ids: Environments where the interval has triggered.
        force_magnitude_range: (min, max) magnitude of the applied force (N).
        asset_cfg: The robot scene entity.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    num = len(env_ids)

    # Random direction in xy-plane
    angle = torch.rand(num, device=env.device) * 2.0 * 3.14159
    magnitude = torch.empty(num, device=env.device).uniform_(*force_magnitude_range)
    forces = torch.zeros(num, 1, 3, device=env.device)
    forces[:, 0, 0] = magnitude * torch.cos(angle)
    forces[:, 0, 1] = magnitude * torch.sin(angle)

    torques = torch.zeros_like(forces)
    asset.set_external_force_and_torque(forces, torques, body_ids=asset_cfg.body_ids, env_ids=env_ids)
```

Usage in config:

```python
from isaaclab.managers import EventTermCfg as EventTerm

random_push = EventTerm(
    func=apply_random_force_impulse,
    mode="interval",
    interval_range_s=(5.0, 10.0),     # Apply every 5-10 seconds
    params={
        "force_magnitude_range": (50.0, 200.0),
        "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
    },
)
```

---

## 9. Custom Action Term (ActionTerm subclass)

Actions always require a class. Must implement `action_dim`, `raw_actions`, `processed_actions`, `process_actions()`, and `apply_actions()`.

```python
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


@configclass
class ScaledJointPositionActionCfg(ActionTermCfg):
    """Configuration for scaled joint position action with velocity limit."""

    class_type: type[ActionTerm] = None  # Set below after class definition
    joint_names: list[str] = []
    scale: float = 1.0
    max_velocity: float = 1.0  # rad/s limit on joint velocity


class ScaledJointPositionAction(ActionTerm):
    """Joint position action with velocity-limited smoothing.

    Limits the rate of change of joint position targets to prevent
    sudden jerky motions.
    """

    cfg: ScaledJointPositionActionCfg

    def __init__(self, cfg: ScaledJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset: Articulation = self._asset
        # Resolve joint names
        self._joint_ids, _ = asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # Buffers
        self._raw_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._current_targets = asset.data.default_joint_pos[:, self._joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Called once per environment step."""
        self._raw_actions = actions.clone()
        # Scale raw actions to target positions
        desired = self._asset.data.default_joint_pos[:, self._joint_ids] + actions * self.cfg.scale
        # Velocity-limit the targets
        dt = self._env.step_dt
        delta = desired - self._current_targets
        max_delta = self.cfg.max_velocity * dt
        delta = torch.clamp(delta, -max_delta, max_delta)
        self._current_targets += delta
        self._processed_actions = self._current_targets.clone()

    def apply_actions(self):
        """Called once per simulation step (possibly multiple times per env step)."""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is not None:
            self._current_targets[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]


# Wire up the config class_type
ScaledJointPositionActionCfg.class_type = ScaledJointPositionAction
```

Usage in config:

```python
@configclass
class ActionsCfg:
    arm = ScaledJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-7]"],
        scale=0.5,
        max_velocity=2.0,
    )
```

---

## 10. Custom Command Term (CommandTerm subclass)

Commands generate goal signals for the agent. Must implement `command` property, `_update_metrics()`, `_resample_command()`, and `_update_command()`.

```python
import torch
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.command_manager import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.utils import configclass


@configclass
class RandomTargetHeightCommandCfg(CommandTermCfg):
    """Config for random target height command generator."""

    class_type: type = None  # Set below
    resampling_time_range: tuple[float, float] = (5.0, 10.0)
    height_range: tuple[float, float] = (0.3, 0.8)


class RandomTargetHeightCommand(CommandTerm):
    """Generates random target heights for the robot.

    Command tensor shape: (num_envs, 1) containing target height in meters.
    """

    cfg: RandomTargetHeightCommandCfg

    def __init__(self, cfg: RandomTargetHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Command buffer
        self._target_height = torch.zeros(env.num_envs, 1, device=env.device)
        # Metrics
        self.metrics["height_error"] = torch.zeros(env.num_envs, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        """The target height command. Shape: (num_envs, 1)."""
        return self._target_height

    def _update_metrics(self):
        """Update tracking metrics (called every step)."""
        robot = self._env.scene["robot"]
        current_height = robot.data.root_pos_w[:, 2:3]
        self.metrics["height_error"] = torch.abs(current_height - self._target_height).squeeze(-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample command for given env indices."""
        n = len(env_ids)
        self._target_height[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
            *self.cfg.height_range
        )

    def _update_command(self):
        """Update command each step (no-op if command is static)."""
        pass


# Wire up config
RandomTargetHeightCommandCfg.class_type = RandomTargetHeightCommand
```

Usage in config:

```python
@configclass
class CommandsCfg:
    target_height = RandomTargetHeightCommandCfg(
        resampling_time_range=(5.0, 10.0),
        height_range=(0.3, 0.8),
    )
```

---

## 11. Custom Curriculum Term

Curriculum terms modify environment parameters over training. Receives `env`, `env_ids`, and params. Returns a value for logging or `None`.

### Function-based

```python
import torch
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedRLEnv


def increase_velocity_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    initial_range: float,
    final_range: float,
    num_steps_to_final: int,
) -> float | None:
    """Linearly increase the velocity command range over training.

    Returns the current velocity range for logging.
    """
    progress = min(env.common_step_counter / num_steps_to_final, 1.0)
    current_range = initial_range + (final_range - initial_range) * progress

    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.ranges.lin_vel_x = (-current_range, current_range)
    command_term.cfg.ranges.lin_vel_y = (-current_range, current_range)

    return current_range
```

Usage in config:

```python
from isaaclab.managers import CurriculumTermCfg as CurrTerm

vel_curriculum = CurrTerm(
    func=increase_velocity_range,
    params={
        "command_name": "base_velocity",
        "initial_range": 0.5,
        "final_range": 2.0,
        "num_steps_to_final": 100000,
    },
)
```

### Class-based (using modify_reward_weight)

```python
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab.envs.mdp as mdp

# Increase penalty weight after N training steps
increase_penalty = CurrTerm(
    func=mdp.modify_reward_weight,
    params={
        "term_name": "joint_torques",
        "weight": -5.0e-4,
        "num_steps": 50000,
    },
)
```

---

## 12. Custom Recorder Term

Recorder terms capture data at specific lifecycle stages. Subclass `RecorderTerm` and override the relevant `record_*` methods. Each method returns `(key, value)` or `(None, None)`.

```python
import torch
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class JointStateRecorder(RecorderTerm):
    """Records joint positions and velocities at each step."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def record_pre_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record joint state before the step."""
        robot = self._env.scene["robot"]
        return "joint_state", {
            "positions": robot.data.joint_pos.clone(),
            "velocities": robot.data.joint_vel.clone(),
        }

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record initial joint state after reset."""
        robot = self._env.scene["robot"]
        if env_ids is None:
            env_ids = slice(None)
        return "initial_joint_state", {
            "positions": robot.data.joint_pos[env_ids].clone(),
            "velocities": robot.data.joint_vel[env_ids].clone(),
        }


@configclass
class JointStateRecorderCfg(RecorderTermCfg):
    """Configuration for the joint state recorder."""
    class_type: type[RecorderTerm] = JointStateRecorder
```

Usage in config:

```python
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.utils import configclass

@configclass
class MyRecorderCfg(RecorderManagerBaseCfg):
    joint_recorder = JointStateRecorderCfg()
```

To use the pre-built recorder that captures actions, states, and observations:

```python
from isaaclab.envs.mdp.recorders import ActionStateRecorderManagerCfg

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    recorder: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
```

---

## Summary: Term Signature Quick Reference

| Term Type | First Args | Returns | Config `func` / `class_type` |
|-----------|-----------|---------|-------------------------------|
| Observation (fn) | `env: ManagerBasedEnv, **params` | `Tensor (N, D)` | `func=` |
| Observation (cls) | `__init__(cfg, env)`, `__call__(env, **params)` | `Tensor (N, D)` | `func=` |
| Reward (fn) | `env: ManagerBasedRLEnv, **params` | `Tensor (N,)` | `func=` |
| Reward (cls) | `__init__(cfg, env)`, `__call__(env, **params)` | `Tensor (N,)` | `func=` |
| Termination (fn) | `env: ManagerBasedRLEnv, **params` | `Tensor (N,)` bool | `func=` |
| Event (fn) | `env: ManagerBasedEnv, env_ids: Tensor, **params` | `None` | `func=` |
| Curriculum (fn) | `env: ManagerBasedRLEnv, env_ids: Sequence, **params` | `float \| dict \| None` | `func=` |
| Action (cls) | `__init__(cfg, env)`, `process_actions(actions)`, `apply_actions()` | -- | `class_type=` |
| Command (cls) | `__init__(cfg, env)`, `command` property | `Tensor (N, D)` | `class_type=` |
| Recorder (cls) | `__init__(cfg, env)`, `record_*(...)` | `(str, Tensor\|dict)` | `class_type=` |
