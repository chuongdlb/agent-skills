# Action & Observation Reference

Complete dispatch tables for `_preprocessAction()` and `_computeObs()` in `BaseRLAviary`.

## Action Dispatch Table

`_preprocessAction(action)` converts normalized `[-1, 1]` actions to `(NUM_DRONES, 4)` RPM arrays.

### ActionType.RPM (dim=4)
```python
rpm[k,:] = HOVER_RPM * (1 + 0.05 * action[k,:])
```
Each motor is independently controlled. The 0.05 scaling means full range [-1,1] maps to [0.95, 1.05] × HOVER_RPM.

### ActionType.PID (dim=3)
```python
state = _getDroneStateVector(k)
next_pos = _calculateNextStep(current_position=state[0:3], destination=target, step_size=1)
rpm[k,:], _, _ = ctrl[k].computeControl(
    control_timestep=CTRL_TIMESTEP,
    cur_pos=state[0:3], cur_quat=state[3:7],
    cur_vel=state[10:13], cur_ang_vel=state[13:16],
    target_pos=next_pos
)
```
The 3D action is interpreted as a target position. `_calculateNextStep` interpolates toward the target at `step_size=1`.

### ActionType.VEL (dim=4)
```python
state = _getDroneStateVector(k)
v_unit = target[0:3] / ||target[0:3]||     # direction from first 3 elements
speed = SPEED_LIMIT * |target[3]|            # magnitude from 4th element
rpm[k,:] = ctrl[k].computeControl(
    cur_pos=state[0:3], cur_quat=state[3:7],
    cur_vel=state[10:13], cur_ang_vel=state[13:16],
    target_pos=state[0:3],                   # hold current position
    target_rpy=[0, 0, state[9]],             # keep current yaw
    target_vel=speed * v_unit                # desired velocity vector
)
```
Where `SPEED_LIMIT = 0.03 * MAX_SPEED_KMH * (1000/3600)`.

### ActionType.ONE_D_RPM (dim=1)
```python
rpm[k,:] = np.repeat(HOVER_RPM * (1 + 0.05 * target), 4)
```
Single scalar repeated to all 4 motors. Simplest action space for altitude control.

### ActionType.ONE_D_PID (dim=1)
```python
state = _getDroneStateVector(k)
rpm[k,:] = ctrl[k].computeControl(
    cur_pos=state[0:3], cur_quat=state[3:7],
    cur_vel=state[10:13], cur_ang_vel=state[13:16],
    target_pos=state[0:3] + 0.1 * [0, 0, target[0]]   # vertical offset
)
```
Single scalar controls vertical position offset. PID handles stabilization.

## Action Space Shapes

| ActionType | Action dim per drone | Total space shape |
|---|---|---|
| RPM | 4 | (NUM_DRONES, 4) |
| PID | 3 | (NUM_DRONES, 3) |
| VEL | 4 | (NUM_DRONES, 4) |
| ONE_D_RPM | 1 | (NUM_DRONES, 1) |
| ONE_D_PID | 1 | (NUM_DRONES, 1) |

All actions bounded in `[-1, +1]`.

## Action Buffer

```python
self.ACTION_BUFFER_SIZE = int(ctrl_freq // 2)    # last 0.5 sec of actions
self.action_buffer = deque(maxlen=ACTION_BUFFER_SIZE)
```

Every call to `_preprocessAction` appends the raw action to the buffer. The buffer is appended to KIN observations.

## Observation Dispatch Table

### ObservationType.KIN

**Base observation (12D per drone):**
```python
obs = _getDroneStateVector(i)                    # 20D state
obs_12 = [obs[0:3], obs[7:10], obs[10:13], obs[13:16]]
#          x,y,z    r,p,y      vx,vy,vz    wx,wy,wz
```

Note: quaternion (indices 3-6) and RPMs (indices 16-19) are **excluded** from RL observations.

**Action buffer appended:**
```python
for i in range(ACTION_BUFFER_SIZE):
    ret = np.hstack([ret, action_buffer[i]])
```

**Total observation dimensions per drone:**

| ActionType | Base | Buffer entries | Buffer dim each | Total |
|---|---|---|---|---|
| RPM | 12 | ctrl_freq//2 | 4 | 12 + 4*(ctrl_freq//2) |
| PID | 12 | ctrl_freq//2 | 3 | 12 + 3*(ctrl_freq//2) |
| VEL | 12 | ctrl_freq//2 | 4 | 12 + 4*(ctrl_freq//2) |
| ONE_D_RPM | 12 | ctrl_freq//2 | 1 | 12 + 1*(ctrl_freq//2) |
| ONE_D_PID | 12 | ctrl_freq//2 | 1 | 12 + 1*(ctrl_freq//2) |

For `ctrl_freq=30`: buffer size = 15.
- RPM/VEL: 12 + 4*15 = 72D per drone
- PID: 12 + 3*15 = 57D per drone
- ONE_D_*: 12 + 1*15 = 27D per drone

### ObservationType.RGB

```python
shape = (NUM_DRONES, IMG_RES[1], IMG_RES[0], 4)  # (N, 48, 64, 4)
dtype = np.float32, range [0, 255]
```

Images captured every `IMG_CAPTURE_FREQ` steps (24 FPS). Between captures, the last frame is reused.

Landmarks loaded for RGB observations:
```
block.urdf    at [ 1,  0, 0.1]
cube_small    at [ 0,  1, 0.1]
duck_vhacd    at [-1,  0, 0.1]
teddy_vhacd   at [ 0, -1, 0.1]
```

## Observation Space Bounds (KIN)

```python
obs_lower = [[-inf, -inf, 0, -inf*9...], ...]     # z >= 0
obs_upper = [[+inf]*12, ...]
# Action buffer bounds: [-1, +1]
```

## Integrated PID Controller

For `ActionType.PID`, `VEL`, and `ONE_D_PID`, BaseRLAviary creates:
```python
self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(num_drones)]
```
Only supports CF2X and CF2P drone models. Sets `KMP_DUPLICATE_LIB_OK=True` for macOS compatibility.
