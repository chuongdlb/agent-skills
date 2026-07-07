import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_codex_skills.py"
SPEC = importlib.util.spec_from_file_location("build_codex_skills", MODULE_PATH)
build_codex_skills = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_codex_skills)


def test_build_codex_skills_generates_codex_layout(tmp_path):
    output_dir = tmp_path / "codex-skills"

    exit_code = build_codex_skills.main(
        [
            "--output-dir",
            str(output_dir),
            "--skill",
            "rl-methodology",
            "--skill",
            "book-reader",
        ]
    )

    assert exit_code == 0

    rl_skill = output_dir / "rl-methodology"
    rl_skill_md = rl_skill / "SKILL.md"
    rl_openai_yaml = rl_skill / "agents" / "openai.yaml"
    rl_reference = rl_skill / "knowledge-base" / "INDEX.md"

    assert rl_skill_md.exists()
    assert rl_openai_yaml.exists()
    assert rl_reference.exists()
    assert rl_reference.is_symlink()

    skill_text = rl_skill_md.read_text(encoding="utf-8")
    assert 'name: "rl-methodology"' in skill_text
    assert 'description: "Comprehensive RL methodology' in skill_text
    assert "layer:" not in skill_text
    assert "depends-on:" not in skill_text
    assert "# RL Methodology" in skill_text

    openai_text = rl_openai_yaml.read_text(encoding="utf-8")
    assert 'display_name: "RL Methodology"' in openai_text
    assert 'short_description: "Comprehensive RL methodology' in openai_text
    assert 'default_prompt: "Use the RL Methodology skill when relevant.' in openai_text

    book_skill = output_dir / "book-reader"
    book_openai_yaml = book_skill / "agents" / "openai.yaml"
    assert book_openai_yaml.exists()
    assert 'display_name: "Book Reader Agent"' in book_openai_yaml.read_text(
        encoding="utf-8"
    )
