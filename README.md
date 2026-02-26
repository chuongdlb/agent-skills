# Agent Skills Repository

A consolidated collection of **46 agent skills** organized in a 4-layer architecture + meta category, following patterns used by top RL/AI labs (Nvidia Isaac, DeepMind Acme, OpenAI Gym, Meta Habitat, Berkeley RLlib).

**Key design principle**: Organize by **abstraction layer** (not domain), with domain as cross-cutting metadata tags.

## Architecture

```
                    ┌─────────────────────────────┐
                    │         meta (3 skills)      │  Orchestration & methodology
                    │  rl-innovator, book-reader,  │  (cross-layer)
                    │  book-reader-agent-prompt     │
                    └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  L3-frameworks (14 skills)                                              │
│  Domain frameworks & applications built on platforms                    │
│  ┌──────────────────────────┐  ┌──────────────────────────────────┐    │
│  │ isaaclab/ (8 skills)     │  │ gym-pybullet-drones/ (6 skills) │    │
│  │ environment-design       │  │ rl-environments                  │    │
│  │ mdp-terms, rl-training   │  │ control-systems                  │    │
│  │ robot-and-asset-config   │  │ training-evaluation              │    │
│  │ terrains-and-sensors     │  │ logging-testing                  │    │
│  │ controllers-and-teleop   │  │ firmware-sitl                    │    │
│  │ configclass-and-utils    │  │ extending-the-codebase           │    │
│  │ imitation-learning       │  │                                  │    │
│  └────────────┬─────────────┘  └──────────────┬───────────────────┘    │
│               │                                │                        │
├───────────────┼────────────────────────────────┼────────────────────────┤
│  L2-platforms (25 skills)    │                  │                        │
│  Simulation engines & platforms                 │                        │
│  ┌────────────┴────────┐  ┌─┴────────────┐  ┌─┴──────────────────┐    │
│  │ isaacsim/ (8)       │  │ pybullet/ (1)│  │ rl-tools/ (10)     │    │
│  │ simulation-core     │  │ simulation-  │  │ build              │    │
│  │ asset-pipeline      │  │ engine       │  │ neural-network     │    │
│  │ robotics            │  │              │  │ training           │    │
│  │ sensor-development  │  │              │  │ environment        │    │
│  │ synthetic-data      │  │              │  │ l2f                │    │
│  │ ros2-integration    │  │              │  │ deploy-hardware    │    │
│  │ build-and-test      │  │              │  │ python, testing    │    │
│  │ extensions          │  │              │  │ experiment-tracking│    │
│  │                     │  │              │  │ web-visualization  │    │
│  └─────────────────────┘  └──────────────┘  └────────────────────┘    │
│                                                                         │
│  ┌──────────────────────────────────────────────┐                      │
│  │ gymnasium/ (6)                                │                      │
│  │ core-api, spaces, wrappers, vector-envs,      │                      │
│  │ environments, custom-environments             │                      │
│  └──────────────────────────────────────────────┘                      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  L1-core (1 skill)                                                      │
│  Core tools, programming, reusable infrastructure                       │
│  ┌──────────────────────┐                                               │
│  │ rl-implementer       │  Python RL implementations (tabular → DQN)   │
│  └──────────────────────┘                                               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  L0-theory (3 skills)                                                   │
│  Foundational RL/math knowledge (project-agnostic)                      │
│  ┌──────────────────────┐ ┌─────────────────────┐ ┌──────────────────┐ │
│  │ rl-theory-analyzer   │ │ rl-convergence-     │ │ rl-algorithm-    │ │
│  │                      │ │ prover              │ │ designer         │ │
│  └──────────────────────┘ └─────────────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Skill Count by Layer

| Layer | Count | Description |
|-------|-------|-------------|
| meta | 3 | Orchestration & methodology (cross-layer) |
| L0-theory | 3 | Foundational RL/math knowledge |
| L1-core | 1 | Core tools & reusable infrastructure |
| L2-platforms | 25 | Simulation engines & platforms (IsaacSim, Gymnasium, PyBullet, RL-Tools) |
| L3-frameworks | 14 | Domain frameworks (IsaacLab, gym-pybullet-drones) |
| **Total** | **46** | |

## Skill Count by Source Project

| Source Project | Count | Layers |
|----------------|-------|--------|
| Book-Mathematical-Foundation-of-RL | 7 | meta, L0, L1 |
| Gymnasium | 6 | L2 |
| IsaacSim | 8 | L2 |
| IsaacLab | 8 | L3 |
| gym-pybullet-drones | 7 | L2 (1), L3 (6) |
| rl-tools-framework | 10 | L2 |

## Skill Index

### meta
| Skill | Description |
|-------|-------------|
| [rl-innovator](meta/rl-innovator/SKILL.md) | Orchestrates RL skills for novel algorithm exploration |
| [book-reader](meta/book-reader/SKILL.md) | Textbook knowledge extraction methodology |
| [book-reader-agent-prompt](meta/book-reader-agent-prompt/SKILL.md) | Prompt template for book-reading agents |

### L0-theory
| Skill | Description |
|-------|-------------|
| [rl-theory-analyzer](L0-theory/rl-theory-analyzer/SKILL.md) | Mathematical analysis of RL algorithms |
| [rl-convergence-prover](L0-theory/rl-convergence-prover/SKILL.md) | Convergence proofs using SA theorems |
| [rl-algorithm-designer](L0-theory/rl-algorithm-designer/SKILL.md) | Algorithm design from composable patterns |

### L1-core
| Skill | Description |
|-------|-------------|
| [rl-implementer](L1-core/rl-implementer/SKILL.md) | Python RL implementations (tabular through DQN) |

### L2-platforms/gymnasium
| Skill | Description |
|-------|-------------|
| [core-api](L2-platforms/gymnasium/core-api/SKILL.md) | Env base class, step/reset contract, registration |
| [spaces](L2-platforms/gymnasium/spaces/SKILL.md) | Box, Discrete, Dict, Tuple, composite spaces |
| [wrappers](L2-platforms/gymnasium/wrappers/SKILL.md) | Observation, action, reward, rendering wrappers |
| [vector-envs](L2-platforms/gymnasium/vector-envs/SKILL.md) | SyncVectorEnv, AsyncVectorEnv, batched semantics |
| [environments](L2-platforms/gymnasium/environments/SKILL.md) | Classic Control, Box2D, Toy Text, MuJoCo |
| [custom-environments](L2-platforms/gymnasium/custom-environments/SKILL.md) | Env subclass, FuncEnv, registration, packaging |

### L2-platforms/isaacsim
| Skill | Description |
|-------|-------------|
| [simulation-core](L2-platforms/isaacsim/simulation-core/SKILL.md) | World, Scene, prim wrappers, physics |
| [asset-pipeline](L2-platforms/isaacsim/asset-pipeline/SKILL.md) | URDF/MJCF import, USD conversion |
| [robotics](L2-platforms/isaacsim/robotics/SKILL.md) | Robot configs, motion planning, controllers |
| [sensor-development](L2-platforms/isaacsim/sensor-development/SKILL.md) | Cameras, lidar, radar, annotators |
| [synthetic-data](L2-platforms/isaacsim/synthetic-data/SKILL.md) | Replicator SDG, domain randomization |
| [ros2-integration](L2-platforms/isaacsim/ros2-integration/SKILL.md) | ROS 2 bridge, topics, TF |
| [build-and-test](L2-platforms/isaacsim/build-and-test/SKILL.md) | Premake build, Packman deps, CI/CD |
| [extensions](L2-platforms/isaacsim/extensions/SKILL.md) | Omniverse Kit extensions, OmniGraph |

### L2-platforms/pybullet
| Skill | Description |
|-------|-------------|
| [simulation-engine](L2-platforms/pybullet/simulation-engine/SKILL.md) | BaseAviary, physics models, step loop |

### L2-platforms/rl-tools
| Skill | Description |
|-------|-------------|
| [build](L2-platforms/rl-tools/build/SKILL.md) | CMake build system, tiered targets |
| [neural-network](L2-platforms/rl-tools/neural-network/SKILL.md) | NN architectures, layers, optimizers |
| [python](L2-platforms/rl-tools/python/SKILL.md) | Python bindings, Gymnasium integration |
| [testing](L2-platforms/rl-tools/testing/SKILL.md) | GoogleTest, benchmarking, profiling |
| [experiment-tracking](L2-platforms/rl-tools/experiment-tracking/SKILL.md) | ExTrack directories, metrics, dashboards |
| [web-visualization](L2-platforms/rl-tools/web-visualization/SKILL.md) | WASM, Canvas 2D, Three.js, Chart.js |
| [training](L2-platforms/rl-tools/training/SKILL.md) | SAC/TD3/PPO training pipelines |
| [environment](L2-platforms/rl-tools/environment/SKILL.md) | Custom C++17 RL environments |
| [l2f](L2-platforms/rl-tools/l2f/SKILL.md) | Learning to Fly quadrotor pipeline |
| [deploy-hardware](L2-platforms/rl-tools/deploy-hardware/SKILL.md) | ESP32, Teensy, Crazyflie, PX4 deployment |

### L3-frameworks/isaaclab
| Skill | Description |
|-------|-------------|
| [configclass-and-utilities](L3-frameworks/isaaclab/configclass-and-utilities/SKILL.md) | @configclass, math utilities, buffers |
| [environment-design](L3-frameworks/isaaclab/environment-design/SKILL.md) | Manager-Based & Direct env paradigms |
| [mdp-terms](L3-frameworks/isaaclab/mdp-terms/SKILL.md) | 8 MDP managers, obs/reward/termination terms |
| [robot-and-asset-config](L3-frameworks/isaaclab/robot-and-asset-config/SKILL.md) | ArticulationCfg, actuators, robot catalog |
| [terrains-and-sensors](L3-frameworks/isaaclab/terrains-and-sensors/SKILL.md) | Terrain generation, sensor configs |
| [controllers-and-teleop](L3-frameworks/isaaclab/controllers-and-teleop/SKILL.md) | IK, operational space, teleoperation |
| [rl-training](L3-frameworks/isaaclab/rl-training/SKILL.md) | RSL-RL/RL-Games/SB3/SKRL training |
| [imitation-learning](L3-frameworks/isaaclab/imitation-learning/SKILL.md) | Demo recording, Mimic, robomimic |

### L3-frameworks/gym-pybullet-drones
| Skill | Description |
|-------|-------------|
| [rl-environments](L3-frameworks/gym-pybullet-drones/rl-environments/SKILL.md) | BaseRLAviary, action/obs types, rewards |
| [control-systems](L3-frameworks/gym-pybullet-drones/control-systems/SKILL.md) | PID, MRAC, Betaflight controllers |
| [training-evaluation](L3-frameworks/gym-pybullet-drones/training-evaluation/SKILL.md) | SB3 PPO training, callbacks, eval |
| [logging-testing](L3-frameworks/gym-pybullet-drones/logging-testing/SKILL.md) | Logger, pytest, debug tools |
| [firmware-sitl](L3-frameworks/gym-pybullet-drones/firmware-sitl/SKILL.md) | Crazyflie & Betaflight SITL |
| [extending-the-codebase](L3-frameworks/gym-pybullet-drones/extending-the-codebase/SKILL.md) | Extension templates & registration |

## File Structure

Each skill directory contains:
- `SKILL.md` — Main skill file with YAML frontmatter and content
- Optional reference files (API docs, catalogs, templates)
- Optional `references/` subdirectory for additional docs

See [SKILL-SPEC.md](SKILL-SPEC.md) for the frontmatter specification.

## Machine-Readable Index

[registry.json](registry.json) contains all 46 skills with paths, layers, domains, dependencies, and tags.
