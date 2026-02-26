# Skill Specification

This document defines the YAML frontmatter schema for SKILL.md files in the agent-skills repository.

## Frontmatter Schema

Every `SKILL.md` file must begin with a YAML frontmatter block:

```yaml
---
name: skill-name-here
description: >
  A concise description of what the skill does and when to use it.
layer: L2
domain: [robotics, simulation]
source-project: ProjectName
depends-on: [other-skill-name, another-skill]
tags: [keyword1, keyword2, keyword3]
---
```

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique identifier for the skill. Lowercase, hyphenated. Must match an entry in `registry.json`. |
| `description` | string | What the skill does and when to invoke it. 1-2 sentences. |
| `layer` | enum | One of: `meta`, `L0`, `L1`, `L2`, `L3` |
| `domain` | string[] | Cross-cutting domain tags. See Domain Vocabulary below. |
| `source-project` | string | Original project this skill was extracted from. |
| `depends-on` | string[] | Names of skills this skill references or builds upon. Empty array if none. |
| `tags` | string[] | Free-form keywords for search and filtering. |

## Layer Definitions

| Layer | Name | Description | Examples |
|-------|------|-------------|----------|
| `meta` | Orchestration | Cross-layer methodology and orchestration skills | rl-innovator, book-reader |
| `L0` | Theory | Foundational knowledge, project-agnostic | rl-theory-analyzer |
| `L1` | Core | Reusable tools, programming, infrastructure | rl-implementer |
| `L2` | Platforms | Simulation engines and platform tools | isaacsim-*, pybullet-*, rltools-* |
| `L3` | Frameworks | Domain frameworks built on L2 platforms | isaaclab-*, gpd-* |

## Domain Vocabulary

Recommended domain tags (use consistently):

| Domain | Description |
|--------|-------------|
| `general-rl` | Domain-agnostic reinforcement learning |
| `robotics` | General robotics |
| `manipulation` | Robot manipulation tasks |
| `locomotion` | Robot locomotion tasks |
| `drones` | Quadrotor / UAV systems |
| `simulation` | Simulation infrastructure |
| `embedded` | Embedded systems deployment |
| `sim-to-real` | Sim-to-real transfer |
| `ml-training` | Machine learning training infrastructure |
| `general` | General-purpose (methodology, tools) |

## Depends-on Rules

- Every entry in `depends-on` must reference a valid `name` from another skill's frontmatter
- Dependencies typically flow downward: L3 → L2 → L1 → L0
- Cross-layer dependencies (e.g., L3 → L0) are allowed for theory references
- `meta` skills may depend on any layer

### External Skill References

For prerequisites from external skill repositories, use the `org/skill-name` format:

```yaml
depends-on: [local-skill-name, K-Dense-AI/stable-baselines3]
```

| Prefix | Repository | Location |
|--------|-----------|----------|
| `K-Dense-AI/` | [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) | `~/.claude/skills/` (installed) |

External references are validated separately from local skill names. Only add external references where there is a **specific, direct prerequisite** — the external skill teaches a library or concept that this skill directly uses.

## Directory Convention

```
skill-name/
├── SKILL.md                    # Required: main skill file with frontmatter
├── optional-reference.md       # Optional: API docs, catalogs, etc.
└── references/                 # Optional: subdirectory for multiple refs
    └── reference-file.md
```

## Adding a New Skill

1. Create a directory under the appropriate layer
2. Write `SKILL.md` with valid frontmatter
3. Add reference files if needed
4. Update `registry.json` (or regenerate it)
5. Verify all `depends-on` references are valid
