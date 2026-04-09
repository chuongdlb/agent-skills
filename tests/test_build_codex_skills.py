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
            "gymnasium-core-api",
            "--skill",
            "book-reader",
        ]
    )

    assert exit_code == 0

    gym_skill = output_dir / "gymnasium-core-api"
    gym_skill_md = gym_skill / "SKILL.md"
    gym_openai_yaml = gym_skill / "agents" / "openai.yaml"
    gym_reference = gym_skill / "registration-reference.md"

    assert gym_skill_md.exists()
    assert gym_openai_yaml.exists()
    assert gym_reference.exists()
    assert gym_reference.is_symlink()

    skill_text = gym_skill_md.read_text(encoding="utf-8")
    assert 'name: "gymnasium-core-api"' in skill_text
    assert 'description: "Core Gymnasium RL environment interface' in skill_text
    assert "layer:" not in skill_text
    assert "depends-on:" not in skill_text
    assert "# Gymnasium Core API" in skill_text

    openai_text = gym_openai_yaml.read_text(encoding="utf-8")
    assert 'display_name: "Gymnasium Core API"' in openai_text
    assert 'short_description: "Core Gymnasium RL environment interface' in openai_text
    assert 'default_prompt: "Use the Gymnasium Core API skill when relevant.' in openai_text

    book_skill = output_dir / "book-reader"
    book_openai_yaml = book_skill / "agents" / "openai.yaml"
    assert book_openai_yaml.exists()
    assert 'display_name: "Book Reader Agent"' in book_openai_yaml.read_text(
        encoding="utf-8"
    )
