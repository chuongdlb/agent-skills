#!/usr/bin/env python3
"""Build or install Codex-compatible skills from this repository.

The source of truth remains the skill directories in this repo. This script
adapts them into Codex's skill layout by:

1. Rewriting SKILL.md frontmatter to Codex's minimal format
2. Generating agents/openai.yaml metadata
3. Symlinking or copying all bundled reference files

Usage:
    python build_codex_skills.py
    python build_codex_skills.py --install
    python build_codex_skills.py --output-dir /tmp/codex-skills
    python build_codex_skills.py --skill gymnasium-core-api --skill book-reader
    python build_codex_skills.py --list-installed
    python build_codex_skills.py --uninstall
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SCRIPT_DIR / "registry.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dist" / "codex-skills"
CODEX_SKILLS_DIR = Path.home() / ".codex" / "skills"


def load_registry() -> list[dict]:
    """Load skill entries from registry.json."""
    with open(REGISTRY_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    return data["skills"]


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter, body) for a SKILL.md file."""
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md is missing opening frontmatter delimiter")

    closing = text.find("\n---\n", 4)
    if closing == -1:
        raise ValueError("SKILL.md is missing closing frontmatter delimiter")

    frontmatter = text[4:closing]
    body = text[closing + 5 :]
    return frontmatter, body


def load_skill_body(skill_path: Path) -> str:
    """Read a SKILL.md file and return only the markdown body."""
    text = skill_path.read_text(encoding="utf-8")
    _, body = split_frontmatter(text)
    return body.lstrip("\n")


def yaml_quote(value: str) -> str:
    """Render a YAML-safe quoted scalar using JSON string escaping."""
    return json.dumps(value, ensure_ascii=False)


def extract_display_name(entry: dict, body: str) -> str:
    """Prefer the first H1 in the skill body; fall back to a humanized name."""
    match = re.search(r"^#\s+(.+?)\s*$", body, flags=re.MULTILINE)
    if match:
        heading = match.group(1).strip()
        for separator in (" — ", " - ", ": "):
            if separator in heading:
                return heading.split(separator, 1)[0].strip()
        return heading

    tokens = entry["name"].split("-")
    replacements = {
        "api": "API",
        "gymnasium": "Gymnasium",
        "isaacsim": "IsaacSim",
        "isaaclab": "IsaacLab",
        "pybullet": "PyBullet",
        "rl": "RL",
        "ros2": "ROS 2",
        "sitl": "SITL",
        "mdp": "MDP",
        "l2f": "L2F",
    }
    humanized = [replacements.get(token, token.capitalize()) for token in tokens]
    return " ".join(humanized)


def render_codex_skill_md(entry: dict, body: str) -> str:
    """Generate Codex-compatible SKILL.md contents."""
    return "\n".join(
        [
            "---",
            f"name: {yaml_quote(entry['name'])}",
            f"description: {yaml_quote(entry['description'])}",
            "---",
            "",
            body.rstrip(),
            "",
        ]
    )


def render_openai_yaml(entry: dict, display_name: str) -> str:
    """Generate minimal Codex UI metadata for a skill."""
    default_prompt = (
        f"Use the {display_name} skill when relevant. {entry['description']}"
    )
    return "\n".join(
        [
            "interface:",
            f"  display_name: {yaml_quote(display_name)}",
            f"  short_description: {yaml_quote(entry['description'])}",
            f"  default_prompt: {yaml_quote(default_prompt)}",
            "",
        ]
    )


def reset_dir(path: Path, *, dry_run: bool = False) -> None:
    """Remove an existing directory tree and recreate it empty."""
    if dry_run:
        return
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link(source: Path, target: Path, *, copy_files: bool) -> None:
    """Materialize a file in the output tree."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy_files:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def build_skill(
    entry: dict,
    destination_root: Path,
    *,
    copy_files: bool = False,
    dry_run: bool = False,
) -> dict:
    """Build a single Codex-formatted skill into destination_root."""
    source_dir = SCRIPT_DIR / entry["path"]
    source_skill_md = source_dir / "SKILL.md"
    if not source_skill_md.exists():
        return {"name": entry["name"], "status": "missing_source", "files": 0}

    body = load_skill_body(source_skill_md)
    display_name = extract_display_name(entry, body)
    target_dir = destination_root / entry["name"]
    files = 0

    if not dry_run:
        reset_dir(target_dir)
        (target_dir / "agents").mkdir(parents=True, exist_ok=True)
        (target_dir / "SKILL.md").write_text(
            render_codex_skill_md(entry, body),
            encoding="utf-8",
        )
        (target_dir / "agents" / "openai.yaml").write_text(
            render_openai_yaml(entry, display_name),
            encoding="utf-8",
        )

    for source_file in source_dir.rglob("*"):
        if not source_file.is_file():
            continue
        if source_file.name == "SKILL.md":
            continue

        relative_path = source_file.relative_to(source_dir)
        if relative_path == Path("agents/openai.yaml"):
            continue

        target_path = target_dir / relative_path
        files += 1

        if not dry_run:
            copy_or_link(source_file, target_path, copy_files=copy_files)

    return {"name": entry["name"], "status": "built", "files": files}


def list_installed(entries: list[dict]) -> None:
    """Print Codex install status for each registered skill."""
    installed = 0
    print(f"{'Skill':<45} {'Status':<15} {'Path'}")
    print("-" * 96)
    for entry in entries:
        path = CODEX_SKILLS_DIR / entry["name"] / "SKILL.md"
        status = "installed" if path.exists() else "not installed"
        if path.exists():
            installed += 1
        print(f"{entry['name']:<45} {status:<15} {path.parent}")
    print("-" * 96)
    print(
        f"Total: {len(entries)} skills "
        f"({installed} installed, {len(entries) - installed} not installed)"
    )


def uninstall(entries: list[dict], *, dry_run: bool = False) -> int:
    """Remove installed Codex skills managed by this repository."""
    removed = 0
    for entry in entries:
        target_dir = CODEX_SKILLS_DIR / entry["name"]
        if not target_dir.exists():
            continue
        removed += 1
        print(f"{'would_remove' if dry_run else 'removed'}: {entry['name']}")
        if not dry_run:
            shutil.rmtree(target_dir)
    return removed


def filter_entries(entries: list[dict], names: list[str] | None) -> list[dict]:
    """Filter registry entries by skill name."""
    if not names:
        return entries

    wanted = set(names)
    filtered = [entry for entry in entries if entry["name"] in wanted]
    found = {entry["name"] for entry in filtered}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown skill names: {', '.join(missing)}")
    return filtered


def build(entries: list[dict], destination_root: Path, *, copy_files: bool, dry_run: bool) -> int:
    """Build the selected skills into a destination directory."""
    if not dry_run:
        destination_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for entry in entries:
        result = build_skill(
            entry,
            destination_root,
            copy_files=copy_files,
            dry_run=dry_run,
        )
        total_files += result["files"]
        print(f"{result['status']}: {result['name']} (+{result['files']} files)")

    print(
        f"\n{'Would build' if dry_run else 'Built'} {len(entries)} skills "
        f"in {destination_root} with {total_files} linked/copied files"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or install Codex-compatible skills from registry.json"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--install",
        action="store_true",
        help="Install the generated skills into ~/.codex/skills",
    )
    group.add_argument(
        "--list-installed",
        action="store_true",
        help="Show install status in ~/.codex/skills",
    )
    group.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove installed Codex skills from ~/.codex/skills",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Build output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy bundled reference files instead of symlinking them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the filesystem",
    )
    parser.add_argument(
        "--skill",
        action="append",
        dest="skills",
        help="Build only the named skill. Pass multiple times for more than one skill.",
    )
    args = parser.parse_args(argv)

    if not REGISTRY_PATH.exists():
        print(f"Error: registry.json not found at {REGISTRY_PATH}", file=sys.stderr)
        return 1

    entries = filter_entries(load_registry(), args.skills)

    if args.list_installed:
        list_installed(entries)
        return 0

    if args.uninstall:
        removed = uninstall(entries, dry_run=args.dry_run)
        action = "Would remove" if args.dry_run else "Removed"
        print(f"\n{action} {removed} skills from {CODEX_SKILLS_DIR}")
        return 0

    destination_root = CODEX_SKILLS_DIR if args.install else args.output_dir
    if args.install:
        print(
            f"{'[DRY RUN] ' if args.dry_run else ''}"
            f"Installing Codex skills into {destination_root}"
        )
    else:
        print(
            f"{'[DRY RUN] ' if args.dry_run else ''}"
            f"Building Codex skills into {destination_root}"
        )

    return build(
        entries,
        destination_root,
        copy_files=args.copy_files,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
