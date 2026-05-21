# Agent Skills Repository

A collection of **14 agent skills** organized by layer — theory, meta-orchestration, and knowledge-base management.

**Key design principle**: Organize by **abstraction layer** (not domain), with domain as cross-cutting metadata tags.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  meta (12 skills)                                                       │
│  Orchestration & methodology (cross-layer)                              │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ rl-innovator, book-reader, book-reader-agent-prompt,            │    │
│  │ book-to-knowledge-base                                          │    │
│  │ paper-extractor, tex-source-paper-extractor, kb-integrator,     │    │
│  │ paper-discoverer, kb-query, kb-maintenance, kb-lint,            │    │
│  │ publication-scout                                               │    │
│  └──────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────┤
│  L0-theory (2 skills)                                                   │
│  Foundational RL/math knowledge (project-agnostic)                      │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ rl-methodology         Analysis, proofs, design patterns,       │    │
│  │                        templates + KB (Zhao 2024)               │    │
│  │ rl-training-protocol   Project-agnostic deep-RL hygiene: env-   │    │
│  │                        symmetry, init/schedule asserts, drift,  │    │
│  │                        seed gating, mechanism attribution,      │    │
│  │                        PPO update & data-pipeline checks        │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Claude Installation

Install all skills as user-level Claude Code skills:

```bash
python install-skills.py
```

This creates skills in `~/.claude/skills/` with adapted frontmatter. Reference files are symlinked to avoid duplication.

Other commands:

```bash
python install-skills.py --dry-run    # Preview without changes
python install-skills.py --list       # Show install status
python install-skills.py --uninstall  # Remove all installed skills
```

## Codex Build / Installation

Build Codex-compatible skill folders into `dist/codex-skills/`:

```bash
python build_codex_skills.py
```

Install them directly into `~/.codex/skills/`:

```bash
python build_codex_skills.py --install
```

Other Codex commands:

```bash
python build_codex_skills.py --dry-run
python build_codex_skills.py --list-installed
python build_codex_skills.py --uninstall
python build_codex_skills.py --skill gymnasium-core-api --skill book-reader
python build_codex_skills.py --copy-files   # copy references instead of symlinking
```

The Codex build keeps the source skill content intact and adapts each skill into:

- Minimal Codex `SKILL.md` frontmatter (`name`, `description`)
- Generated `agents/openai.yaml`
- Symlinked or copied bundled reference files

## Skill Count by Layer

| Layer | Count | Description |
|-------|-------|-------------|
| meta | 12 | Orchestration & methodology (cross-layer) |
| L0-theory | 2 | Foundational RL/math knowledge |
| **Total** | **14** | |

## Skill Count by Source Project

| Source Project | Count | Layers |
|----------------|-------|--------|
| Book-Mathematical-Foundation-of-RL | 5 | meta, L0 |
| rl-escape-dense-forest | 1 | L0 |
| papers-kb | 8 | meta |

## Skill Index

### meta
| Skill | Description |
|-------|-------------|
| [rl-innovator](meta/rl-innovator/SKILL.md) | Orchestrates RL skills for novel algorithm exploration |
| [book-reader](meta/book-reader/SKILL.md) | Textbook knowledge extraction methodology |
| [book-reader-agent-prompt](meta/book-reader-agent-prompt/SKILL.md) | Prompt template for book-reading agents |
| [book-to-knowledge-base](meta/book-to-knowledge-base/SKILL.md) | Lossless PDF-to-markdown knowledge base conversion |
| [paper-extractor](meta/paper-extractor/SKILL.md) | Extract structured contributions from a research paper PDF |
| [tex-source-paper-extractor](meta/tex-source-paper-extractor/SKILL.md) | Extract or enrich a paper card from an arXiv paper's LaTeX (.tex) source |
| [kb-integrator](meta/kb-integrator/SKILL.md) | Classify novelty and update topic files, index, timeline, registry |
| [paper-discoverer](meta/paper-discoverer/SKILL.md) | Search databases and GitHub for new papers, produce ranked list |
| [kb-query](meta/kb-query/SKILL.md) | Answer research questions against the knowledge base |
| [kb-maintenance](meta/kb-maintenance/SKILL.md) | Full KB update cycle: discover, filter, extract, integrate, report |
| [kb-lint](meta/kb-lint/SKILL.md) | Health-check the KB: mechanical and semantic checks |
| [publication-scout](meta/publication-scout/SKILL.md) | Discover publishable research gaps with 5-Gates review |

### L0-theory
| Skill | Description |
|-------|-------------|
| [rl-methodology](L0-theory/rl-methodology/SKILL.md) | Analysis, proofs, design patterns, templates + extensible knowledge base |
| [rl-training-protocol](L0-theory/rl-training-protocol/SKILL.md) | Project-agnostic deep-RL hygiene across six failure-mode classes |

## File Structure

Each skill directory contains:
- `SKILL.md` — Main skill file with YAML frontmatter and content
- Optional reference files (API docs, catalogs, templates)
- Optional `references/` subdirectory for additional docs

See [SKILL-SPEC.md](SKILL-SPEC.md) for the frontmatter specification.

## Machine-Readable Index

[registry.json](registry.json) contains all 14 skills with paths, layers, domains, dependencies, and tags.
