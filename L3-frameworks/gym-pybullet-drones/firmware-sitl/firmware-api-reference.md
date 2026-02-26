# Firmware API Reference

Data structures, sensor pipeline, and protocol specifications for CFAviary and BetaAviary.

## pycffirmware Data Structures (CFAviary)

### control_t
Output from the controller. Contains thrust and torque commands.
```
control_t.thrust    # float: collective thrust
control_t.roll      # float: roll torque
control_t.pitch     # float: pitch torque
control_t.yaw       # float: yaw torque
```

### setpoint_t
Desired state set by commands or the high-level planner.
```
setpoint_t.position.x/y/z          # target position (m)
setpoint_t.velocity.x/y/z          # target velocity (m/s)
setpoint_t.acceleration.x/y/z      # target acceleration (m/s^2)
setpoint_t.attitudeQuaternion.x/y/z/w  # target attitude
setpoint_t.attitudeRate.roll/pitch/yaw  # target angular rates (deg/s)
setpoint_t.attitude.roll/pitch/yaw      # target Euler angles (deg)
setpoint_t.mode.x/y/z              # modeAbs or modeDisable
setpoint_t.mode.quat               # modeAbs or modeDisable
setpoint_t.mode.roll/pitch/yaw     # modeAbs or modeDisable
setpoint_t.timestamp               # ms
```

### sensorData_t
Simulated sensor readings fed to the controller.
```
sensorData_t.acc.x/y/z             # accelerometer (Gs, body frame)
sensorData_t.gyro.x/y/z            # gyroscope (deg/s, body frame)
sensorData_t.mag.x/y/z             # magnetometer (gauss) - unused
sensorData_t.baro.pressure         # barometer (hPa) - unused
sensorData_t.baro.temperature      # temperature (C) - unused
sensorData_t.interruptTimestamp     # microseconds
```

### state_t
Current state estimate (ground truth from PyBullet in SITL).
```
state_t.attitude.roll/pitch/yaw    # Euler angles (deg, CF2 convention: pitch inverted)
state_t.attitudeQuaternion.x/y/z/w # quaternion
state_t.position.x/y/z             # position (m)
state_t.velocity.x/y/z             # velocity (m/s)
state_t.acc.x/y/z                  # acceleration (Gs, without gravity on z)
```

## CFAviary Sensor Pipeline

### Low-Pass Filters

```python
# Initialization
acclpf = [firm.lpf2pData() for _ in range(3)]
gyrolpf = [firm.lpf2pData() for _ in range(3)]
for i in range(3):
    firm.lpf2pInit(acclpf[i], firmware_freq, GYRO_LPF_CUTOFF_FREQ)    # 80 Hz
    firm.lpf2pInit(gyrolpf[i], firmware_freq, ACCEL_LPF_CUTOFF_FREQ)  # 30 Hz

# Per-step application
sensorData.gyro.x = firm.lpf2pApply(gyrolpf[0], gyro_x)
sensorData.acc.x = firm.lpf2pApply(acclpf[0], acc_x)
```

### Rate Estimation

Angular rates and acceleration are estimated from finite differences:
```python
cur_rotation_rates = (cur_rpy - prev_rpy) / firmware_dt        # rad/s
cur_acc = (cur_vel - prev_vel) / firmware_dt / 9.8 + [0,0,1]  # Gs (global)
body_acc = body_rot.apply(cur_acc)                              # Gs (body)
```

### State Update (CF convention)

```python
# Attitude: CF2 convention has inverted pitch
attitude_t.roll = roll_deg
attitude_t.pitch = -pitch_deg    # legacy inversion
attitude_t.yaw = yaw_deg
```

### Setpoint Mode Flags

For `sendFullStateCmd`:
```python
setpoint.mode.x = firm.modeAbs      # absolute position
setpoint.mode.y = firm.modeAbs
setpoint.mode.z = firm.modeAbs
setpoint.mode.quat = firm.modeAbs   # absolute quaternion
setpoint.mode.roll = firm.modeDisable
setpoint.mode.pitch = firm.modeDisable
setpoint.mode.yaw = firm.modeDisable
```

## CFAviary High-Level Commander API

```python
# Initialize
firm.crtpCommanderHighLevelInit()

# Tell state for planner
firm.crtpCommanderHighLevelTellState(state)

# Update time
firm.crtpCommanderHighLevelUpdateTime(time_s)

# Get computed setpoint
firm.crtpCommanderHighLevelGetSetpoint(setpoint, state)

# Commands
firm.crtpCommanderHighLevelTakeoff(height, duration)
firm.crtpCommanderHighLevelTakeoffYaw(height, duration, yaw)
firm.crtpCommanderHighLevelTakeoffWithVelocity(height, vel, relative)
firm.crtpCommanderHighLevelLand(height, duration)
firm.crtpCommanderHighLevelLandYaw(height, duration, yaw)
firm.crtpCommanderHighLevelLandWithVelocity(height, vel, relative)
firm.crtpCommanderHighLevelGoTo(x, y, z, yaw, duration_s, relative)
firm.crtpCommanderHighLevelStop()
```

## CFAviary Controller Stepping

```python
# Controller tick logic
if time - last_att_pid > 0.002 and time - last_pos_pid > 0.01:
    tick = 0    # Run BOTH position and attitude controller
elif time - last_att_pid > 0.002:
    tick = 2    # Run attitude controller only
else:
    tick = 1    # Run neither

# Step controller
firm.controllerMellinger(control, setpoint, sensorData, state, tick)
# or
firm.controllerPid(control, setpoint, sensorData, state, tick)
```

## CFAviary Power Distribution

Converts `control_t` to motor PWMs:

```python
# X-configuration (QUAD_FORMATION_X = True)
r = control.roll / 2
p = control.pitch / 2
motor0 = motorsGetPWM(limitThrust(control.thrust - r + p + control.yaw))
motor1 = motorsGetPWM(limitThrust(control.thrust - r - p - control.yaw))
motor2 = motorsGetPWM(limitThrust(control.thrust + r - p + control.yaw))
motor3 = motorsGetPWM(limitThrust(control.thrust + r + p - control.yaw))

# Brushed motor PWM conversion
def motorsGetPWM(thrust):
    thrust = thrust / 65536 * 60
    volts = -0.0006239 * thrust^2 + 0.088 * thrust
    percentage = min(1, volts / SUPPLY_VOLTAGE)   # SUPPLY_VOLTAGE = 3V
    return percentage * MAX_PWM
```

## BetaAviary UDP Protocol

### FDM Packet (State → Betaflight)

```python
struct.pack('@dddddddddddddddddd',
    timestamp,                          # double: seconds
    w_body[0], -w_body[1], -w_body[2],  # double[3]: angular velocity (ENU→NED)
    0, 0, 0,                            # double[3]: linear acceleration
    1, 0, 0, 0,                         # double[4]: orientation quat (w,x,y,z)
    0, 0, 0,                            # double[3]: velocity
    0, 0, 0,                            # double[3]: position
    1.0                                 # double: pressure (hPa)
)
# Total: 18 doubles = 144 bytes
# Sent to port 9003 + 10*drone_id
```

### RC Packet (Commands → Betaflight)

```python
struct.pack('@dHHHHHHHHHHHHHHHH',
    timestamp,                          # double: seconds
    roll, pitch, throttle, yaw,         # uint16[4]: channels 1000-2000
    aux1, aux2, ..., aux12              # uint16[12]: aux channels
)
# Total: 1 double + 16 uint16 = 40 bytes
# Sent to port 9004 + 10*drone_id
```

**RC channel conventions:**
- Throttle: 1000 (min) to 2000 (max), positive up
- Roll: 1500 center, positive right
- Pitch: 1500 center, positive forward
- Yaw: 1500 center, positive CCW
- Aux1: 1000 (disarmed), 1500 (armed)

### PWM Packet (Betaflight → Motors)

```python
data = sock_pwm.recvfrom(16)           # 4 floats = 16 bytes
pwms = struct.unpack('@ffff', data)
# Received on port 9002 + 10*drone_id
```

### CTBR to Betaflight Channel Conversion

```python
def ctbr2beta(thrust, roll, pitch, yaw):
    MIN_CHANNEL, MAX_CHANNEL = 1000, 2000
    MAX_RATE = 360     # deg/s
    MAX_THRUST = 40.9  # N

    mid = 1500
    d = 500
    thrust_ch = thrust / MAX_THRUST * d * 2 + MIN_CHANNEL
    rates_ch = np.array([roll, pitch, -yaw]) / pi * 180 / MAX_RATE * d + mid
    return clip(thrust_ch, 1000, 2000), *clip(rates_ch, 1000, 2000)
```

## BetaAviary Motor Remapping

Betaflight SITL uses different motor ordering than PyBullet:

```python
remapped = [action[2], action[1], action[3], action[0]]  # BF→PyBullet
rpm = sqrt(MAX_THRUST / 4 / KF * remapped)
```

## Delay Configuration (CFAviary)

```python
ACTION_DELAY = 0   # firmware loops between command and motor response
SENSOR_DELAY = 0   # firmware loops between motion and sensor reading
STATE_DELAY = 0    # not implemented, keep at 0
```

Non-zero delays use history buffers for more realistic simulation but are experimental.
