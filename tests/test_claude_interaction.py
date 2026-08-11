"""Tests for claude_interaction.ClaudeInteraction (template discovery/loading only).

None of these tests call the Claude API.
"""

import pytest

from claude_interaction import ClaudeInteraction


@pytest.fixture
def prompt_dir(tmp_path, sample_prompt):
    (tmp_path / "genomics_prompt.json").write_text(sample_prompt.to_json())
    (tmp_path / "aardvark_prompt.json").write_text(sample_prompt.to_json())
    return tmp_path


def test_list_available_templates_sorted_alphabetically(prompt_dir):
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    templates = interaction.list_available_templates()

    assert [t["filename"].split("/")[-1] for t in templates] == [
        "aardvark_prompt.json",
        "genomics_prompt.json",
    ]


def test_load_prompt_template_noninteractive_uses_first_template(prompt_dir):
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    loaded = interaction.load_prompt_template(interactive=False)

    assert loaded is not None
    assert loaded.research_area == "Test Area"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    with pytest.raises(ValueError):
        ClaudeInteraction(api_key=None, prompt_dir="prompt")
