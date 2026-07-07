"""Registry ↔ filesystem ↔ frontmatter consistency checks.

Catches the defect class where a skill is removed or renamed but the
registry, another skill's depends-on, a test, or a doc still references it.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "registry.json").read_text(encoding="utf-8"))
SKILLS = REGISTRY["skills"]
NAMES = {s["name"] for s in SKILLS}

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def frontmatter(skill):
    text = (REPO / skill["path"] / "SKILL.md").read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    assert m, f"{skill['name']}: missing or malformed frontmatter"
    return m.group(1)


def test_total_skills_matches_entry_count():
    assert REGISTRY["total_skills"] == len(SKILLS)


def test_skill_names_unique():
    assert len(NAMES) == len(SKILLS)


def test_every_skill_dir_has_skill_md():
    for s in SKILLS:
        path = REPO / s["path"] / "SKILL.md"
        assert path.is_file(), f"{s['name']}: no SKILL.md at {s['path']}"


def test_frontmatter_name_matches_registry():
    for s in SKILLS:
        fm = frontmatter(s)
        m = re.search(r"^name:\s*(\S+)\s*$", fm, re.MULTILINE)
        assert m, f"{s['name']}: frontmatter has no name field"
        assert m.group(1) == s["name"], (
            f"{s['path']}: frontmatter name {m.group(1)!r} != registry {s['name']!r}"
        )


def test_depends_on_references_resolve():
    for s in SKILLS:
        for dep in s.get("depends-on", []):
            if "/" in dep:  # external org/skill reference per SKILL-SPEC.md
                continue
            assert dep in NAMES, (
                f"{s['name']}: depends-on {dep!r} does not match any registered skill"
            )


def test_frontmatter_depends_on_matches_registry():
    for s in SKILLS:
        fm = frontmatter(s)
        m = re.search(r"^depends-on:\s*\[(.*?)\]\s*$", fm, re.MULTILINE)
        fm_deps = (
            [d.strip() for d in m.group(1).split(",") if d.strip()] if m else []
        )
        assert fm_deps == s.get("depends-on", []), (
            f"{s['name']}: frontmatter depends-on {fm_deps} "
            f"!= registry {s.get('depends-on', [])}"
        )


def test_reference_files_exist():
    for s in SKILLS:
        for ref in s.get("reference-files", []):
            assert (REPO / ref).is_file(), (
                f"{s['name']}: reference file missing: {ref}"
            )
