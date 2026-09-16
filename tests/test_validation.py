"""Tests for bioinformatics_prompts.utils.validation."""

import logging

from bioinformatics_prompts.utils.validation import export_all_prompts, validate_prompt


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


def test_export_all_prompts_logs_instead_of_printing(sample_prompt, tmp_path, caplog, capsys):
    """export_all_prompts reports progress through the logger, not stdout.

    Scoped to this function deliberately: `claude_interaction.py` still prints,
    and those calls are #14's to remove.
    """
    with caplog.at_level(logging.INFO, logger="bioinformatics_prompts.utils.validation"):
        export_all_prompts({"genomics": sample_prompt}, str(tmp_path))

    assert "Exported 1 prompts" in caplog.text
    assert capsys.readouterr().out == ""
