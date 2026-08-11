"""Tests for utils.validation."""

from utils.validation import validate_prompt


def test_validate_prompt_passes_for_well_formed_prompt(sample_prompt):
    result = validate_prompt(sample_prompt)

    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_prompt_flags_missing_examples(sample_prompt):
    sample_prompt.examples = []

    result = validate_prompt(sample_prompt)

    assert result["valid"] is False
    assert any("example" in err.lower() for err in result["errors"])


def test_validate_prompt_warns_on_short_description(sample_prompt):
    sample_prompt.description = "Too short."

    result = validate_prompt(sample_prompt)

    assert any("description" in w.lower() for w in result["warnings"])
