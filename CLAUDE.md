# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A collection of **46 Claude Code agent skills** for reinforcement learning and robotics, organized by abstraction layer. Skills are installed to `~/.claude/skills/` where Claude Code loads them automatically.

## Commands

```bash
python install-skills.py              # Install all skills to ~/.claude/skills/
python install-skills.py --dry-run    # Preview without changes
python install-skills.py --list       # Show install status
python install-skills.py --uninstall  # Remove all installed skills
```

There is no build system, linter, or test suite.

## Architecture

Skills are organized by **abstraction layer** (not domain):

| Layer | Purpose | Examples |
|-------|---------|---------|
| `meta` | Cross-layer orchestration | rl-innovator, book-reader |
| `L0-theory` | Project-agnostic foundational knowledge | rl-theory-analyzer, rl-convergence-prover |
| `L1-core` | Reusable tools and infrastructure | rl-implementer |
| `L2-platforms` | Simulation engines (grouped by platform) | isaacsim/, gymnasium/, rl-tools/, pybullet/ |
| `L3-frameworks` | Domain frameworks built on L2 | isaaclab/, gym-pybullet-drones/ |

Dependencies flow downward: L3 → L2 → L1 → L0. Meta skills may depend on any layer.

## Skill File Structure

Each skill is a directory containing:
- **`SKILL.md`** (required) — YAML frontmatter + markdown body
- Optional reference files (API docs, catalogs, templates)

### Frontmatter Schema (SKILL-SPEC.md)

All 6 fields are required:
```yaml
---
name: skill-name          # Unique, lowercase-hyphenated, must match registry.json
description: >            # 1-2 sentences: what it does and when to invoke it
  ...
layer: L2                 # meta | L0 | L1 | L2 | L3
domain: [robotics, simulation]  # From defined vocabulary in SKILL-SPEC.md
source-project: ProjectName
depends-on: [other-skill] # Valid skill names; use K-Dense-AI/name for external
tags: [keyword1, keyword2]
---
```

## Registry

`registry.json` is the machine-readable index of all skills. It must stay in sync with the actual skill directories. Each entry mirrors the frontmatter fields plus `path` and `reference-files`.

## Install Script Behavior

`install-skills.py` reads `registry.json`, then for each skill:
1. Parses `SKILL.md`, strips the agent-skills frontmatter (layer, domain, source-project, depends-on, tags)
2. Generates simplified user-level frontmatter (name, description, license, metadata)
3. Writes the new `SKILL.md` to `~/.claude/skills/<name>/`
4. Symlinks all reference files (not copies)

## Adding a New Skill

1. Create directory under the appropriate layer (e.g., `L2-platforms/gymnasium/new-skill/`)
2. Write `SKILL.md` with valid frontmatter per SKILL-SPEC.md
3. Add reference files if needed
4. Add entry to `registry.json` with matching `name`, `path`, and `reference-files`
5. Verify all `depends-on` references point to existing skill names
6. Update counts in `README.md` (total_skills in registry.json, layer counts, source project table)
7. Run `python install-skills.py` to install
