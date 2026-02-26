# Pre-Built Robot Catalog

Complete catalog of every pre-built robot configuration available in `isaaclab_assets`. All configs are `ArticulationCfg` instances.

## Manipulators

| Config Variable | Import Path | Actuator Type | Description |
|----------------|-------------|---------------|-------------|
| `FRANKA_PANDA_CFG` | `isaaclab_assets.robots.franka` | ImplicitActuator | Franka Emika Panda with Panda hand. Stiffness 80, damping 4 for arm. |
| `FRANKA_PANDA_HIGH_PD_CFG` | `isaaclab_assets.robots.franka` | ImplicitActuator | Franka Panda with stiffer PD (stiffness 400, damping 80). Gravity disabled. For IK control. |
| `FRANKA_ROBOTIQ_GRIPPER_CFG` | `isaaclab_assets.robots.franka` | ImplicitActuator | Franka with Robotiq 2F-85 gripper variant. Gravity disabled. Higher effort limits. |
| `KINOVA_JACO2_N7S300_CFG` | `isaaclab_assets.robots.kinova` | ImplicitActuator | Kinova JACO2 7-DOF arm with 3-finger gripper. Per-joint effort/stiffness/damping dicts. |
| `KINOVA_JACO2_N6S300_CFG` | `isaaclab_assets.robots.kinova` | ImplicitActuator | Kinova JACO2 6-DOF arm with 3-finger gripper. |
| `KINOVA_GEN3_N7_CFG` | `isaaclab_assets.robots.kinova` | ImplicitActuator | Kinova Gen3 7-DOF arm, no gripper. |
| `UR10_CFG` | `isaaclab_assets.robots.universal_robots` | ImplicitActuator | UR10 arm. Stiffness 800, damping 40. |
| `UR10e_CFG` | `isaaclab_assets.robots.universal_robots` | ImplicitActuator | UR10e arm. Per-group stiffness/damping (shoulder, elbow, wrist). Gravity disabled. |
| `UR10_LONG_SUCTION_CFG` | `isaaclab_assets.robots.universal_robots` | ImplicitActuator | UR10 with long suction gripper variant. Gravity disabled. |
| `UR10_SHORT_SUCTION_CFG` | `isaaclab_assets.robots.universal_robots` | ImplicitActuator | UR10 with short suction gripper variant. Gravity disabled. |
| `UR10e_ROBOTIQ_GRIPPER_CFG` | `isaaclab_assets.robots.universal_robots` | ImplicitActuator | UR10e with Robotiq 2F-140 gripper. Includes drive, finger, and passive actuator groups. |
| `UR10e_ROBOTIQ_2F_85_CFG` | `isaaclab_assets.robots.universal_robots` | ImplicitActuator | UR10e with Robotiq 2F-85 gripper. Includes drive, finger, and passive actuator groups. |
| `SAWYER_CFG` | `isaaclab_assets.robots.sawyer` | ImplicitActuator | Rethink Robotics Sawyer 7-DOF arm. Per-group effort limits. |
| `OPENARM_BI_CFG` | `isaaclab_assets.robots.openarm` | ImplicitActuator | OpenArm bimanual (two arms). Per-joint velocity/effort limits from motor spec sheets. |
| `OPENARM_BI_HIGH_PD_CFG` | `isaaclab_assets.robots.openarm` | ImplicitActuator | OpenArm bimanual with stiffer PD (stiffness 400, damping 80). Gravity disabled. |
| `OPENARM_UNI_CFG` | `isaaclab_assets.robots.openarm` | ImplicitActuator | OpenArm unimanual (one arm). |
| `OPENARM_UNI_HIGH_PD_CFG` | `isaaclab_assets.robots.openarm` | ImplicitActuator | OpenArm unimanual with stiffer PD. Gravity disabled. |

## Quadrupeds

| Config Variable | Import Path | Actuator Type | Description |
|----------------|-------------|---------------|-------------|
| `UNITREE_A1_CFG` | `isaaclab_assets.robots.unitree` | DCMotor | Unitree A1. saturation_effort=33.5, effort_limit=33.5. Init height 0.42m. |
| `UNITREE_GO1_CFG` | `isaaclab_assets.robots.unitree` | ActuatorNetMLP | Unitree Go1. Uses learned MLP actuator model (unitree_go1.pt). Init height 0.4m. |
| `UNITREE_GO2_CFG` | `isaaclab_assets.robots.unitree` | DCMotor | Unitree Go2. saturation_effort=23.5. Init height 0.4m. |
| `ANYMAL_B_CFG` | `isaaclab_assets.robots.anymal` | ActuatorNetLSTM | ANYmal-B with ANYdrive 3.0 LSTM actuator net. Init height 0.6m. |
| `ANYMAL_C_CFG` | `isaaclab_assets.robots.anymal` | ActuatorNetLSTM | ANYmal-C with ANYdrive 3.0 LSTM actuator net. Init height 0.6m. |
| `ANYMAL_D_CFG` | `isaaclab_assets.robots.anymal` | ActuatorNetLSTM | ANYmal-D with ANYdrive 3.0 LSTM actuator net (same net as ANYmal-C). Init height 0.6m. |
| `SPOT_CFG` | `isaaclab_assets.robots.spot` | DelayedPD + RemotizedPD | Boston Dynamics Spot. Hips: DelayedPD (0-4 step delay). Knees: RemotizedPD with lookup table. Init height 0.5m. |

### ANYmal Actuator Presets

These standalone actuator configs can be reused:

| Config Variable | Import Path | Type | Description |
|----------------|-------------|------|-------------|
| `ANYDRIVE_3_SIMPLE_ACTUATOR_CFG` | `isaaclab_assets.robots.anymal` | DCMotor | ANYdrive 3.x simple model. saturation=120, effort=80, velocity=7.5 |
| `ANYDRIVE_3_LSTM_ACTUATOR_CFG` | `isaaclab_assets.robots.anymal` | ActuatorNetLSTM | ANYdrive 3.0 LSTM model. Uses anydrive_3_lstm_jit.pt |

### Go1 Actuator Preset

| Config Variable | Import Path | Type | Description |
|----------------|-------------|------|-------------|
| `GO1_ACTUATOR_CFG` | `isaaclab_assets.robots.unitree` | ActuatorNetMLP | Go1 MLP model. Uses unitree_go1.pt. pos_scale=-1.0 |

## Bipeds and Humanoids

| Config Variable | Import Path | Actuator Type | Description |
|----------------|-------------|---------------|-------------|
| `H1_CFG` | `isaaclab_assets.robots.unitree` | ImplicitActuator | Unitree H1 humanoid. 3 actuator groups: legs, feet, arms. Init height 1.05m. |
| `H1_MINIMAL_CFG` | `isaaclab_assets.robots.unitree` | ImplicitActuator | H1 with fewer collision meshes for faster simulation. |
| `G1_CFG` | `isaaclab_assets.robots.unitree` | ImplicitActuator | Unitree G1 humanoid. 3 groups: legs, feet, arms (includes fingers). Init height 0.74m. |
| `G1_MINIMAL_CFG` | `isaaclab_assets.robots.unitree` | ImplicitActuator | G1 with fewer collision meshes for faster simulation. |
| `G1_29DOF_CFG` | `isaaclab_assets.robots.unitree` | DCMotor + Implicit | G1 29-DOF for locomanipulation. Legs/feet: DCMotor. Waist/arms/hands: Implicit. Configurable fix_root_link. |
| `G1_INSPIRE_FTP_CFG` | `isaaclab_assets.robots.unitree` | DCMotor + Implicit | G1 29-DOF with Inspire 5-finger hand. Fixed base, gravity disabled. For grasping tasks. |
| `CASSIE_CFG` | `isaaclab_assets.robots.cassie` | ImplicitActuator | Agility Cassie biped. 2 groups: legs, toes. Init height 0.9m. |
| `DIGIT_V4_CFG` | `isaaclab_assets.robots.agility` | ImplicitActuator | Agility Digit V4 humanoid. Stiffness/damping from USD. Init height 1.05m. |
| `GR1T2_CFG` | `isaaclab_assets.robots.fourier` | ImplicitActuator | Fourier GR1T2 humanoid with 6-DOF hands. Gravity disabled. Init height 0.95m. |
| `GR1T2_HIGH_PD_CFG` | `isaaclab_assets.robots.fourier` | ImplicitActuator | GR1T2 with high PD gains (stiffness 4400) for pick-place manipulation. Created via .replace(). |
| `AGIBOT_A2D_CFG` | `isaaclab_assets.robots.agibot` | ImplicitActuator | AGIBot A2D wheeled humanoid. Body, head, left/right arm, left/right gripper groups. |
| `GALBOT_ONE_CHARLIE_CFG` | `isaaclab_assets.robots.galbot` | ImplicitActuator | Galbot One Charlie humanoid. PD from USD. Head, leg, left/right arm, left gripper groups. |
| `HUMANOID_CFG` | `isaaclab_assets.robots.humanoid` | ImplicitActuator | MuJoCo Humanoid (21-DOF). Per-joint stiffness/damping. Init height 1.34m. |
| `HUMANOID_28_CFG` | `isaaclab_assets.robots.humanoid_28` | ImplicitActuator | MuJoCo Humanoid 28-DOF. Stiffness/damping from USD. Init height 0.8m. |

## Dexterous Hands

| Config Variable | Import Path | Actuator Type | Description |
|----------------|-------------|---------------|-------------|
| `ALLEGRO_HAND_CFG` | `isaaclab_assets.robots.allegro` | ImplicitActuator | Wonik Allegro Hand. 16 joints. Gravity disabled. effort_limit_sim=0.5, stiffness=3.0. |
| `SHADOW_HAND_CFG` | `isaaclab_assets.robots.shadow_hand` | ImplicitActuator | Shadow Dexterous Hand. 24 joints with tendons. Per-joint effort limits. fixed_tendons_props and joint_drive_props used. |
| `KUKA_ALLEGRO_CFG` | `isaaclab_assets.robots.kuka_allegro` | ImplicitActuator | Kuka LBR iiwa 7 arm + Allegro Hand. Gravity disabled. Solver iterations 32/1. Per-joint stiffness/damping/friction. |

## Mobile Platforms

| Config Variable | Import Path | Actuator Type | Description |
|----------------|-------------|---------------|-------------|
| `RIDGEBACK_FRANKA_PANDA_CFG` | `isaaclab_assets.robots.ridgeback_franka` | ImplicitActuator | Clearpath Ridgeback base + Franka arm. Base: velocity control (stiffness=0, damping=1e5). |

## Aerial Robots

| Config Variable | Import Path | Actuator Type | Description |
|----------------|-------------|---------------|-------------|
| `CRAZYFLIE_CFG` | `isaaclab_assets.robots.quadcopter` | ImplicitActuator | Bitcraze Crazyflie quadcopter. Dummy actuator (stiffness=0, damping=0). Init motor velocities 200 rad/s. |

## Classic Control / Benchmark

| Config Variable | Import Path | Actuator Type | Description |
|----------------|-------------|---------------|-------------|
| `ANT_CFG` | `isaaclab_assets.robots.ant` | ImplicitActuator | MuJoCo Ant. 8 joints. Stiffness=0, damping=0 (torque-controlled). Init height 0.5m. |
| `CARTPOLE_CFG` | `isaaclab_assets.robots.cartpole` | ImplicitActuator | Classic cartpole. Cart: damping=10. Pole: free (damping=0). Init height 2.0m. |
| `CART_DOUBLE_PENDULUM_CFG` | `isaaclab_assets.robots.cart_double_pendulum` | ImplicitActuator | Cart with double pendulum. Cart: damping=10. Both pendulums free. Init height 2.0m. |

## Other / Specialized

| Config Variable | Import Path | Actuator Type | Description |
|----------------|-------------|---------------|-------------|
| `PICK_AND_PLACE_CFG` | `isaaclab_assets.robots.pick_and_place` | ImplicitActuator | Simple 3-axis gantry robot with suction cup. X/Y/Z prismatic joints. Velocity-controlled. |

## Usage Examples

### Import and Use a Pre-Built Config

```python
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

scene_cfg.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

### Create a Variant with .copy()

```python
from isaaclab_assets.robots.unitree import H1_CFG

my_h1 = H1_CFG.copy()
my_h1.spawn.articulation_props.fix_root_link = True
my_h1.actuators["legs"].stiffness = {".*_hip_yaw": 200.0, ".*_hip_roll": 200.0, ".*_hip_pitch": 300.0, ".*_knee": 300.0, "torso": 300.0}
```

### Create a Variant with .replace()

```python
from isaaclab_assets.robots.fourier import GR1T2_CFG
from isaaclab.actuators import ImplicitActuatorCfg

my_gr1t2 = GR1T2_CFG.replace(
    actuators={
        "trunk": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*"],
            stiffness=4400,
            damping=40.0,
            armature=0.01,
        ),
        # ... other groups
    },
)
```

### Swap Actuator Model

```python
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG, ANYDRIVE_3_SIMPLE_ACTUATOR_CFG

# Switch from LSTM actuator net to simple DC motor
anymal_simple = ANYMAL_C_CFG.copy()
anymal_simple.actuators = {"legs": ANYDRIVE_3_SIMPLE_ACTUATOR_CFG}
```
