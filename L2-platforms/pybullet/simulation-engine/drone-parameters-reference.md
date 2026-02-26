# Drone Parameters Reference

Complete URDF properties, motor positions, and inertia for all supported drone models.

## URDF XML Schema

Every drone URDF follows this structure:

```xml
<?xml version="1.0" ?>
<robot name="...">
  <properties arm="..." kf="..." km="..." thrust2weight="..." max_speed_kmh="..."
    gnd_eff_coeff="..." prop_radius="..." drag_coeff_xy="..." drag_coeff_z="..."
    dw_coeff_1="..." dw_coeff_2="..." dw_coeff_3="..." />
  <link name="base_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="..."/>
      <inertia ixx="..." ixy="0.0" ixz="0.0" iyy="..." iyz="0.0" izz="..."/>
    </inertial>
    <visual>...</visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><cylinder radius="..." length="..."/></geometry>
    </collision>
  </link>
  <!-- prop0_link through prop3_link (motor positions) -->
  <!-- center_of_mass_link -->
</robot>
```

The `<properties>` element contains aerodynamic coefficients. Motor positions are encoded as the `<origin>` of each `prop_link`.

## Drone Specifications

### CF2X — Crazyflie 2.0 X-Configuration

| Property | Value |
|----------|-------|
| Mass | 0.027 kg |
| Arm length | 0.0397 m |
| KF (thrust coeff) | 3.16e-10 |
| KM (torque coeff) | 7.94e-12 |
| Thrust-to-weight | 2.25 |
| Max speed | 30 km/h |
| Ixx | 1.4e-5 |
| Iyy | 1.4e-5 |
| Izz | 2.17e-5 |
| Collision radius | 0.06 m |
| Collision height | 0.025 m |

**Motor positions (X-config):**
```
prop0: ( 0.028, -0.028, 0)   # front-right
prop1: (-0.028, -0.028, 0)   # rear-right
prop2: (-0.028,  0.028, 0)   # rear-left
prop3: ( 0.028,  0.028, 0)   # front-left
```

### CF2P — Crazyflie 2.0 +-Configuration

| Property | Value |
|----------|-------|
| Mass | 0.027 kg |
| Arm length | 0.0397 m |
| KF (thrust coeff) | 3.16e-10 |
| KM (torque coeff) | 7.94e-12 |
| Thrust-to-weight | 2.25 |
| Max speed | 30 km/h |
| Ixx | 2.3951e-5 |
| Iyy | 2.3951e-5 |
| Izz | 3.2347e-5 |
| Collision radius | 0.06 m |
| Collision height | 0.025 m |

**Motor positions (+-config):**
```
prop0: ( 0.0397,  0,      0)   # front (+X)
prop1: ( 0,       0.0397, 0)   # left  (+Y)
prop2: (-0.0397,  0,      0)   # rear  (-X)
prop3: ( 0,      -0.0397, 0)   # right (-Y)
```

### RACE — Racing Drone X-Configuration

| Property | Value |
|----------|-------|
| Mass | 0.830 kg |
| Arm length | 0.109 m |
| KF (thrust coeff) | 8.47e-9 |
| KM (torque coeff) | 2.13e-11 |
| Thrust-to-weight | 4.17 |
| Max speed | 200 km/h |
| Ixx | 3.113e-3 |
| Iyy | 3.113e-3 |
| Izz | 3.113e-3 |
| Collision radius | 0.06 m |
| Collision height | 0.025 m |

**Motor positions (X-config):**
```
prop0: ( 0.085,  0.0675, 0)   # front-right
prop1: (-0.085,  0.0675, 0)   # rear-right
prop2: (-0.085, -0.0675, 0)   # rear-left
prop3: ( 0.085, -0.0675, 0)   # front-left
```

## Computed Constants

These are derived from URDF parameters in `BaseAviary.__init__()`:

```python
GRAVITY = G * M                                     # weight (N)
HOVER_RPM = sqrt(GRAVITY / (4 * KF))                # RPM for hover
MAX_RPM = sqrt((THRUST2WEIGHT_RATIO * GRAVITY) / (4 * KF))
MAX_THRUST = 4 * KF * MAX_RPM^2
J = diag([IXX, IYY, IZZ])                           # inertia matrix
J_INV = inv(J)                                       # inverse inertia
```

**CF2X computed values:**
```
GRAVITY = 9.8 * 0.027 = 0.2646 N
HOVER_RPM ≈ 14468 RPM
MAX_RPM ≈ 21702 RPM
```

## URDF Parsing

`_parseURDFParameters()` extracts values from the XML:

| XML path | Parameter |
|----------|-----------|
| `URDF_TREE[0].attrib['arm']` | Arm length L |
| `URDF_TREE[0].attrib['kf']` | Thrust coefficient KF |
| `URDF_TREE[0].attrib['km']` | Torque coefficient KM |
| `URDF_TREE[1][0][1].attrib['value']` | Mass M |
| `URDF_TREE[1][0][2].attrib['ixx']` | Moment of inertia Ixx |
| `URDF_TREE[1][2][1][0].attrib['length']` | Collision height |
| `URDF_TREE[1][2][1][0].attrib['radius']` | Collision radius |

## Motor Numbering Convention

PyBullet link indices for propellers:
- Link 0: `prop0_link` (motor 0)
- Link 1: `prop1_link` (motor 1)
- Link 2: `prop2_link` (motor 2)
- Link 3: `prop3_link` (motor 3)
- Link 4: `center_of_mass_link`

Yaw torque alternating signs: `-motor0 + motor1 - motor2 + motor3`

## URDF File Locations

```
gym_pybullet_drones/assets/cf2x.urdf
gym_pybullet_drones/assets/cf2p.urdf
gym_pybullet_drones/assets/racer.urdf
gym_pybullet_drones/assets/cf2.dae        # shared mesh
```
