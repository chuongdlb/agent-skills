---
name: rltools-experiment-tracking
description: >
  ExTrack experiment tracking and visualization — experiment directories, evaluation intervals, trajectory replay, return.json metrics, dashboards.
layer: L2
domain: [general-rl]
source-project: rl-tools-framework
depends-on: [rltools-build, rltools-web-visualization]
tags: [experiment-tracking, extrack, visualization]
---

# RL-Tools Experiment Tracking (ExTrack)

## Overview

ExTrack is rl-tools' built-in experiment tracking system. It creates a structured directory hierarchy for organizing training runs, checkpoints, trajectories, and metrics. Combined with the ExTrack UI library, it enables interactive visualization and comparison of experiments.

## When to Use This Skill

- User wants to configure experiment tracking for training
- User wants to save and replay trajectories
- User wants to compare runs across seeds or algorithms
- User wants to build visualization dashboards
- User asks about the ExTrack directory format

## Reference Files

### Core:
- ExTrack loop step: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/extrack/`
- Evaluation step: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/evaluation/`
- Save trajectories: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/save_trajectories/`
- Checkpoint step: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/rl/loop/steps/checkpoint/`

### Visualization:
- Zoo frontend: `/home/ai/source/rl-tools-framework/zoo.rl.tools/`
- ExTrack UI lib: `/home/ai/source/rl-tools-framework/extrack-ui-lib/`

### Data:
- CI runs: `/home/ai/source/rl-tools-framework/zoo-runs-ci/`
- Curated runs: `/home/ai/source/rl-tools-framework/zoo-runs-curated/`

### Spec:
- `/home/ai/source/rl-tools-framework/docs.rl.tools/docs/10-Experiment Tracking.rst`

## ExTrack Directory Structure

```
experiments/
  <timestamp>/                              # e.g., 2024-01-15_14-30-00
    <hash>_<name>/                          # e.g., a1b2c3_my_experiment
      <env>_<algo>/                         # e.g., pendulum_sac
        <seed>/                             # e.g., 0, 1, 2
          steps/
            <step_number>/                  # e.g., 1000, 2000
              checkpoint.h5                 # HDF5 model weights
              trajectories.json.gz          # Compressed episode data
          return.json                       # Evaluation return metrics
          ui.esm.js                         # Environment render function
          description.txt                   # Human-readable description
```

## Configuration in Code

### Enable ExTrack

```cpp
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>

using LOOP_EXTRACK_CONFIG = rlt::rl::loop::steps::extrack::Config<LOOP_CORE_CONFIG>;
```

Set the experiment name before `init()`:
```cpp
LOOP_STATE ls;
ls.extrack_config.name = "my_experiment";
rlt::malloc(device, ls);
rlt::init(device, ls, seed);
```

### Configure Evaluation

```cpp
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>

template <typename NEXT>
struct LOOP_EVAL_PARAMETERS : rlt::rl::loop::steps::evaluation::Parameters<TYPE_POLICY, TI, NEXT> {
    static constexpr TI EVALUATION_INTERVAL = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / 10;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_EXTRACK_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_EXTRACK_CONFIG>>;
```

### Configure Trajectory Saving

```cpp
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>

struct LOOP_SAVE_TRAJ_PARAMS : rlt::rl::loop::steps::save_trajectories::Parameters<TYPE_POLICY, TI, LOOP_EVAL_CONFIG> {
    static constexpr TI INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / 3;
    static constexpr TI INTERVAL = INTERVAL_TEMP == 0 ? 1 : INTERVAL_TEMP;
    static constexpr TI NUM_EPISODES = 10;
};
using LOOP_SAVE_TRAJ_CONFIG = rlt::rl::loop::steps::save_trajectories::Config<LOOP_EVAL_CONFIG, LOOP_SAVE_TRAJ_PARAMS>;
```

### Configure Checkpointing

```cpp
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>

// Requires HDF5 support enabled in CMake
```

## Data Formats

### return.json
```json
{
  "steps": [1000, 2000, 3000, 4000, 5000],
  "returns": [-150.5, -80.3, -45.2, -20.1, -8.7]
}
```

### trajectories.json.gz
Gzip-compressed JSON. Each entry contains timestep data with state, action, and reward for episode replay.

### ui.esm.js
ES6 module exported by the environment's `get_ui()` function. Contains `init()` and `render()` for browser-based visualization.

## Visualization Tools

### Zoo Frontend (`zoo.rl.tools/`)
Browse and visualize training runs:
- Chart.js learning curves
- Three.js 3D environment replay
- ExTrack UI integration
- Loads from `zoo-runs-ci/` and `zoo-runs-curated/`

### ExTrack UI Library (`extrack-ui-lib/`)
Reusable JavaScript components:
- **Chart.js wrapper** — Learning curve plots with seed aggregation
- **Three.js wrapper** — 3D environment visualization
- **ACE editor** — In-browser code editing
- **pako** — zlib decompression for `.json.gz` files

### Building Custom Dashboards
```html
<script type="module">
  import { loadExperiment, plotReturns } from './extrack-ui-lib/index.js';

  const experiment = await loadExperiment('experiments/2024-01-15/...');
  plotReturns(document.getElementById('chart'), experiment);
</script>
```

## Run Comparison

ExTrack's directory structure naturally supports comparison:

```
experiments/timestamp/hash_name/
  pendulum_sac/
    0/return.json    # Seed 0
    1/return.json    # Seed 1
    2/return.json    # Seed 2
  pendulum_td3/
    0/return.json    # Different algorithm, same env
```

The zoo frontend aggregates across seeds automatically, showing mean and confidence intervals.

## Requirements for Trajectory Replay

For trajectory saving to work, the environment must implement in `operations_cpu.h`:
1. `json(device, env, parameters)` — Serialize parameters
2. `json(device, env, parameters, state)` — Serialize state per timestep
3. `get_ui(device, env)` — Return ES6 render module

Without these, ExTrack still works for return metrics and checkpoints, but trajectory replay won't function.
