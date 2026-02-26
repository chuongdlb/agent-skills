# Logger Format Reference

State vector index mapping, file formats, and plot layout for the Logger class.

## State Vector Index Mapping

### Input Format (BaseAviary 20D → Logger)

| Input index | Field | Logger index | Logger field |
|---|---|---|---|
| 0 | pos_x | 0 | pos_x |
| 1 | pos_y | 1 | pos_y |
| 2 | pos_z | 2 | pos_z |
| 3 | quat_x | — | dropped |
| 4 | quat_y | — | dropped |
| 5 | quat_z | — | dropped |
| 6 | quat_w | — | dropped |
| 7 | roll | 6 | roll |
| 8 | pitch | 7 | pitch |
| 9 | yaw | 8 | yaw |
| 10 | vel_x | 3 | vel_x |
| 11 | vel_y | 4 | vel_y |
| 12 | vel_z | 5 | vel_z |
| 13 | ang_vel_x | 9 | ang_vel_x |
| 14 | ang_vel_y | 10 | ang_vel_y |
| 15 | ang_vel_z | 11 | ang_vel_z |
| 16 | rpm0 | 12 | rpm0 |
| 17 | rpm1 | 13 | rpm1 |
| 18 | rpm2 | 14 | rpm2 |
| 19 | rpm3 | 15 | rpm3 |

### Reordering Code

```python
self.states[drone, :, step] = np.hstack([
    state[0:3],     # pos (3)
    state[10:13],   # vel (3)
    state[7:10],    # rpy (3)
    state[13:20]    # ang_vel (3) + rpm (4) = 7
])
# Total: 3 + 3 + 3 + 7 = 16 stored values
```

## Logger Internal Arrays

```python
self.timestamps    # shape: (num_drones, T)
self.states        # shape: (num_drones, 16, T)
self.controls      # shape: (num_drones, 12, T)
self.counters      # shape: (num_drones,) — current step index per drone
```

### Control Vector (12D)

```
controls[drone, 0:3, :]   — target pos_x, pos_y, pos_z
controls[drone, 3:6, :]   — target vel_x, vel_y, vel_z
controls[drone, 6:9, :]   — target roll, pitch, yaw
controls[drone, 9:12, :]  — target ang_vel_x, ang_vel_y, ang_vel_z
```

## NPZ File Structure

Output of `logger.save()`:

```python
# File: results/save-flight-MM.DD.YYYY_HH.MM.SS.npy
with np.load('file.npy') as data:
    timestamps = data['timestamps']   # (num_drones, T)
    states = data['states']           # (num_drones, 16, T)
    controls = data['controls']       # (num_drones, 12, T)
```

## CSV File Structure

Output of `logger.save_as_csv(comment)`:

```
results/save-flight-{comment}-MM.DD.YYYY_HH.MM.SS/
├── x0.csv           # position x, drone 0
├── y0.csv           # position y, drone 0
├── z0.csv           # position z, drone 0
├── vx0.csv          # velocity x
├── vy0.csv          # velocity y
├── vz0.csv          # velocity z
├── r0.csv           # roll
├── p0.csv           # pitch
├── ya0.csv          # yaw
├── wx0.csv          # angular velocity x
├── wy0.csv          # angular velocity y
├── wz0.csv          # angular velocity z
├── rr0.csv          # roll rate (finite diff)
├── pr0.csv          # pitch rate (finite diff)
├── yar0.csv         # yaw rate (finite diff)
├── rpm0-0.csv       # motor 0 RPM
├── rpm1-0.csv       # motor 1 RPM
├── rpm2-0.csv       # motor 2 RPM
├── rpm3-0.csv       # motor 3 RPM
├── pwm0-0.csv       # motor 0 PWM (derived)
├── pwm1-0.csv       # motor 1 PWM (derived)
├── pwm2-0.csv       # motor 2 PWM (derived)
└── pwm3-0.csv       # motor 3 PWM (derived)
```

Each CSV has 2 columns: `time, value`.

PWM is derived from RPM: `pwm = (rpm - 4070.3) / 0.2685`

For multi-drone, files are suffixed with drone index: `x0.csv`, `x1.csv`, etc.

Angular rates (rr, pr, yar) are computed via finite differences:
```python
rdot = hstack([0, (states[j, 6, 1:] - states[j, 6, 0:-1]) * LOGGING_FREQ_HZ])
```

## Plot Layout (10x2 Grid)

`logger.plot()` generates a `fig, axs = plt.subplots(10, 2)`:

### Column 0 (Left)

| Row | States index | Y-label | Description |
|---|---|---|---|
| 0 | 0 | x (m) | Position X |
| 1 | 1 | y (m) | Position Y |
| 2 | 2 | z (m) | Position Z |
| 3 | 6 | r (rad) | Roll |
| 4 | 7 | p (rad) | Pitch |
| 5 | 8 | y (rad) | Yaw |
| 6 | 9 | wx | Angular velocity X |
| 7 | 10 | wy | Angular velocity Y |
| 8 | 11 | wz | Angular velocity Z |
| 9 | — | time | Time vs time (identity) |

### Column 1 (Right)

| Row | States index | Y-label | Description |
|---|---|---|---|
| 0 | 3 | vx (m/s) | Velocity X |
| 1 | 4 | vy (m/s) | Velocity Y |
| 2 | 5 | vz (m/s) | Velocity Z |
| 3 | computed | rdot (rad/s) | Roll rate (finite diff) |
| 4 | computed | pdot (rad/s) | Pitch rate (finite diff) |
| 5 | computed | ydot (rad/s) | Yaw rate (finite diff) |
| 6 | 12 | RPM0 / PWM0 | Motor 0 |
| 7 | 13 | RPM1 / PWM1 | Motor 1 |
| 8 | 14 | RPM2 / PWM2 | Motor 2 |
| 9 | 15 | RPM3 / PWM3 | Motor 3 |

## Color/Linestyle Cycling

```python
plt.rc('axes', prop_cycle=(
    cycler('color', ['r', 'g', 'b', 'y']) +
    cycler('linestyle', ['-', '--', ':', '-.'])
))
```

| Drone | Color | Linestyle |
|---|---|---|
| 0 | Red | Solid |
| 1 | Green | Dashed |
| 2 | Blue | Dotted |
| 3 | Yellow | Dash-dot |

Cycles for >4 drones.

## Plot Formatting

```python
fig.subplots_adjust(
    left=0.06, bottom=0.05, right=0.99, top=0.98,
    wspace=0.15, hspace=0.0
)
```

All subplots have grid enabled and legends in upper-right corner. Labels are `drone_0`, `drone_1`, etc.

## Dynamic Array Growth

When `duration_sec=0` (default), arrays grow dynamically:

```python
if current_counter >= timestamps.shape[1]:
    timestamps = np.concatenate((timestamps, np.zeros((NUM_DRONES, 1))), axis=1)
    states = np.concatenate((states, np.zeros((NUM_DRONES, 16, 1))), axis=2)
    controls = np.concatenate((controls, np.zeros((NUM_DRONES, 12, 1))), axis=2)
```

When `duration_sec > 0`, arrays are preallocated for better performance:
```python
timestamps = np.zeros((num_drones, duration_sec * logging_freq_hz))
states = np.zeros((num_drones, 16, duration_sec * logging_freq_hz))
```
