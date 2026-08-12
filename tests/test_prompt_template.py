"""Tests for bioinformatics_prompts.prompt.templates.prompt_template."""

from bioinformatics_prompts.prompt.templates.prompt_template import BioinformaticsPrompt, FewShotExample


def test_generate_prompt_includes_sections_and_query(sample_prompt):
    rendered = sample_prompt.generate_prompt("What tool should I use?")

    assert "# Test Area Research Context" in rendered
    assert "## Key Concepts" in rendered
    assert "Concept A" in rendered
    assert "## Common Tools" in rendered
    assert "## Current Query\nWhat tool should I use?" in rendered


def test_json_round_trip_preserves_generated_prompt(sample_prompt):
    query = "How do I call variants?"
    original = sample_prompt.generate_prompt(query)

    reloaded = BioinformaticsPrompt.from_json(sample_prompt.to_json())

    assert reloaded.generate_prompt(query) == original


def test_few_shot_example_format():
    example = FewShotExample(query="Q?", context="C", response="R")

    assert example.format() == "User Query: Q?\nContext: C\nResponse: R\n"
