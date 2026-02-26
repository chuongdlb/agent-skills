#!/usr/bin/env python3
"""Install agent-skills as user-level Claude Code skills.

Creates skills in ~/.claude/skills/ with adapted frontmatter and symlinked
reference files.

Usage:
    python install-skills.py              # Install all skills
    python install-skills.py --uninstall  # Remove installed skills
    python install-skills.py --list       # Show install status
    python install-skills.py --dry-run    # Preview without changes
"""

import argparse
import json
import os
import sys
from pathlib import Path


SKILLS_DIR = Path.home() / ".claude" / "skills"
SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SCRIPT_DIR / "registry.json"

# Fields to drop from agent-skills frontmatter (not used by Claude Code)
AGENT_ONLY_FIELDS = {"layer", "domain", "source-project", "depends-on", "tags"}


def load_registry() -> list[dict]:
    """Load skill entries from registry.json."""
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    return data["skills"]


def parse_skill_md(skill_path: Path) -> tuple[dict, str]:
    """Parse a SKILL.md file into (frontmatter_dict, body_text).

    Returns the raw frontmatter fields as a dict and the body content
    (everything after the closing --- delimiter).
    """
    text = skill_path.read_text()

    # Expect file to start with ---
    if not text.startswith("---"):
        raise ValueError(f"SKILL.md missing opening '---': {skill_path}")

    # Find closing ---
    end = text.index("\n---\n", 3)
    body = text[end + 5:]  # everything after closing ---\n

    return body


def generate_frontmatter(entry: dict) -> str:
    """Generate user-level Claude Code frontmatter from a registry entry."""
    # Escape description for YAML - use block scalar if it contains special chars
    desc = entry["description"]

    lines = [
        "---",
        f"name: {entry['name']}",
        f"description: >",
        f"  {desc}",
        "license: MIT License",
        "metadata:",
        "    skill-author: agent-skills",
        "---",
    ]
    return "\n".join(lines)


def install_skill(entry: dict, dry_run: bool = False) -> dict:
    """Install a single skill. Returns a status dict."""
    name = entry["name"]
    source_dir = SCRIPT_DIR / entry["path"]
    target_dir = SKILLS_DIR / name
    source_skill_md = source_dir / "SKILL.md"

    result = {"name": name, "files_linked": 0, "status": "installed"}

    if not source_skill_md.exists():
        result["status"] = "missing_source"
        return result

    if dry_run:
        result["status"] = "would_install"
        # Count reference files
        ref_count = 0
        for item in source_dir.rglob("*"):
            if item.is_file() and item.name != "SKILL.md":
                ref_count += 1
        result["files_linked"] = ref_count
        return result

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Parse source SKILL.md and generate new one with user-level frontmatter
    body = parse_skill_md(source_skill_md)
    frontmatter = generate_frontmatter(entry)
    new_content = frontmatter + "\n" + body

    target_skill_md = target_dir / "SKILL.md"
    target_skill_md.write_text(new_content)

    # Symlink all other files (reference files, subdirectories)
    for item in source_dir.rglob("*"):
        if item.is_file() and item.name != "SKILL.md":
            # Compute relative path within the skill directory
            rel = item.relative_to(source_dir)
            target_path = target_dir / rel

            # Create parent dirs if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing symlink/file before creating
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()

            target_path.symlink_to(item.resolve())
            result["files_linked"] += 1

    return result


def uninstall_skill(entry: dict, dry_run: bool = False) -> dict:
    """Uninstall a single skill. Returns a status dict."""
    name = entry["name"]
    target_dir = SKILLS_DIR / name
    result = {"name": name, "status": "not_installed"}

    if not target_dir.exists():
        return result

    if dry_run:
        result["status"] = "would_remove"
        return result

    # Remove all files and subdirectories
    for item in sorted(target_dir.rglob("*"), reverse=True):
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    target_dir.rmdir()

    result["status"] = "removed"
    return result


def list_skills(entries: list[dict]) -> None:
    """Print install status for all skills."""
    installed = 0
    not_installed = 0

    print(f"{'Skill':<45} {'Status':<15} {'Path'}")
    print("-" * 90)

    for entry in entries:
        name = entry["name"]
        target_dir = SKILLS_DIR / name
        target_skill = target_dir / "SKILL.md"

        if target_skill.exists():
            status = "installed"
            installed += 1
        else:
            status = "not installed"
            not_installed += 1

        print(f"{name:<45} {status:<15} {target_dir}")

    print("-" * 90)
    print(f"Total: {len(entries)} skills ({installed} installed, {not_installed} not installed)")


def main():
    parser = argparse.ArgumentParser(
        description="Install agent-skills as user-level Claude Code skills"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--uninstall", action="store_true",
        help="Remove all installed agent-skills"
    )
    group.add_argument(
        "--list", action="store_true",
        help="Show install status of all skills"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without modifying anything"
    )
    args = parser.parse_args()

    if not REGISTRY_PATH.exists():
        print(f"Error: registry.json not found at {REGISTRY_PATH}", file=sys.stderr)
        sys.exit(1)

    entries = load_registry()
    print(f"Found {len(entries)} skills in registry.json")

    if args.list:
        list_skills(entries)
        return

    if args.uninstall:
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Uninstalling skills...")
        removed = 0
        for entry in entries:
            result = uninstall_skill(entry, dry_run=args.dry_run)
            if result["status"] in ("removed", "would_remove"):
                print(f"  {result['status']}: {result['name']}")
                removed += 1
        print(f"\n{'Would remove' if args.dry_run else 'Removed'} {removed} skills")
        return

    # Install
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Installing skills to {SKILLS_DIR}/...")
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    installed = 0
    total_refs = 0
    errors = []

    for entry in entries:
        try:
            result = install_skill(entry, dry_run=args.dry_run)
            status = result["status"]
            refs = result["files_linked"]
            total_refs += refs

            ref_info = f" (+{refs} ref files)" if refs > 0 else ""
            print(f"  {status}: {result['name']}{ref_info}")

            if status in ("installed", "would_install"):
                installed += 1
        except Exception as e:
            errors.append((entry["name"], str(e)))
            print(f"  ERROR: {entry['name']}: {e}")

    print(f"\n{'Would install' if args.dry_run else 'Installed'} {installed} skills, {total_refs} reference files linked")
    if errors:
        print(f"{len(errors)} errors occurred", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
