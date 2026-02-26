# Templates Reference

Copy-paste scaffolds for common extension tasks.

## Template: New RL Environment

```python
"""gym_pybullet_drones/envs/MyTaskAviary.py"""
import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class MyTaskAviary(BaseRLAviary):
    """Single agent RL problem: [describe task]."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        self.TARGET_POS = np.array([0, 0, 1])
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS - state[0:3])
        return max(0, 2 - dist ** 4)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        return np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.0001

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0
                or abs(state[7]) > 0.4 or abs(state[8]) > 0.4):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {}
```

## Template: New Multi-Agent RL Environment

```python
"""gym_pybullet_drones/envs/MyMultiAviary.py"""
import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class MyMultiAviary(BaseRLAviary):
    """Multi-agent RL problem: [describe task]."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
        self.TARGET_POS = self.INIT_XYZS + np.array([[0, 0, 1] for _ in range(num_drones)])

    def _computeReward(self):
        ret = 0
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            ret += max(0, 2 - np.linalg.norm(self.TARGET_POS[i, :] - state[0:3]) ** 4)
        return ret

    def _computeTerminated(self):
        total_dist = sum(np.linalg.norm(self.TARGET_POS[i, :] - self._getDroneStateVector(i)[0:3])
                         for i in range(self.NUM_DRONES))
        return total_dist < 0.0001

    def _computeTruncated(self):
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            if (abs(state[0]) > 2.0 or abs(state[1]) > 2.0 or state[2] > 2.0
                    or abs(state[7]) > 0.4 or abs(state[8]) > 0.4):
                return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {}
```

## Template: New Control Environment

```python
"""gym_pybullet_drones/envs/MyCtrlAviary.py"""
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class MyCtrlAviary(BaseAviary):
    """Multi-drone environment for control applications."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder)

    def _actionSpace(self):
        lo = np.array([[0.] * 4 for _ in range(self.NUM_DRONES)])
        hi = np.array([[self.MAX_RPM] * 4 for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _observationSpace(self):
        lo = np.array([[-np.inf] * 16 + [0.] * 4 for _ in range(self.NUM_DRONES)])
        hi = np.array([[np.inf] * 16 + [self.MAX_RPM] * 4 for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _computeObs(self):
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    def _preprocessAction(self, action):
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    def _computeReward(self):
        return -1

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {}
```

## Template: New Controller

```python
"""gym_pybullet_drones/control/MyControl.py"""
import numpy as np

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


class MyControl(BaseControl):
    """Custom controller for [describe]."""

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        # Initialize gains, mixer matrices, etc.
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.reset()

    def reset(self):
        super().reset()
        # Reset internal state (errors, integrals, etc.)

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel,
                       cur_ang_vel, target_pos, target_rpy=np.zeros(3),
                       target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        self.control_counter += 1
        # Compute desired RPMs
        rpm = np.array([self.HOVER_RPM] * 4)  # placeholder
        pos_e = target_pos - cur_pos
        yaw_e = 0.0
        return rpm, pos_e, yaw_e
```

Note: `BaseControl.__init__` does not set `HOVER_RPM`. If needed, compute it as `sqrt(GRAVITY / (4 * KF))`.

## Template: URDF File

```xml
<?xml version="1.0" ?>
<robot name="my_drone">
  <properties arm="0.05" kf="3.16e-10" km="7.94e-12" thrust2weight="2.5"
    max_speed_kmh="50" gnd_eff_coeff="11.36859" prop_radius="3e-2"
    drag_coeff_xy="9.1785e-7" drag_coeff_z="10.311e-7"
    dw_coeff_1="2267.18" dw_coeff_2=".16" dw_coeff_3="-.11" />

  <link name="base_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="2e-5" ixy="0" ixz="0" iyy="2e-5" iyz="0" izz="3e-5"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="./cf2.dae" scale="1 1 1"/></geometry>
      <material name="grey"><color rgba=".5 .5 .5 1"/></material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><cylinder radius="0.08" length="0.03"/></geometry>
    </collision>
  </link>

  <!-- Motor 0: front-right (X-config example) -->
  <link name="prop0_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0.035 -0.035 0"/>
      <mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop0_joint" type="fixed">
    <parent link="base_link"/><child link="prop0_link"/>
  </joint>

  <!-- Motor 1: rear-right -->
  <link name="prop1_link">
    <inertial>
      <origin rpy="0 0 0" xyz="-0.035 -0.035 0"/>
      <mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop1_joint" type="fixed">
    <parent link="base_link"/><child link="prop1_link"/>
  </joint>

  <!-- Motor 2: rear-left -->
  <link name="prop2_link">
    <inertial>
      <origin rpy="0 0 0" xyz="-0.035 0.035 0"/>
      <mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop2_joint" type="fixed">
    <parent link="base_link"/><child link="prop2_link"/>
  </joint>

  <!-- Motor 3: front-left -->
  <link name="prop3_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0.035 0.035 0"/>
      <mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="prop3_joint" type="fixed">
    <parent link="base_link"/><child link="prop3_link"/>
  </joint>

  <!-- Center of mass -->
  <link name="center_of_mass_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="center_of_mass_joint" type="fixed">
    <parent link="base_link"/><child link="center_of_mass_link"/>
  </joint>
</robot>
```

## Template: Gymnasium Registration Entry

```python
# Add to gym_pybullet_drones/__init__.py
register(
    id='my-task-v0',
    entry_point='gym_pybullet_drones.envs:MyTaskAviary',
)
```

## Template: Example Script

```python
"""gym_pybullet_drones/examples/my_example.py"""
import time
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
logger = Logger(logging_freq_hz=int(env.CTRL_FREQ), num_drones=1)

obs, info = env.reset()
START = time.time()
for i in range(10 * env.CTRL_FREQ):
    rpm, _, _ = ctrl.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP, state=obs[0],
        target_pos=np.array([0, 0, 1]))
    action = rpm.reshape(1, 4)
    obs, reward, terminated, truncated, info = env.step(action)
    logger.log(drone=0, timestamp=i / env.CTRL_FREQ, state=obs[0])
    env.render()
    sync(i, START, env.CTRL_TIMESTEP)

env.close()
logger.plot()
```

## Template: Test Function

```python
# Add to tests/test_examples.py
def test_my_example():
    from gym_pybullet_drones.examples.my_example import run
    run(gui=False, plot=False, output_folder='tmp')
```
