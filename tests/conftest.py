"""Shared test fixtures."""

import pytest

from bioinformatics_prompts.prompt.templates.prompt_template import BioinformaticsPrompt, FewShotExample


@pytest.fixture
def sample_prompt():
    """A minimal but valid BioinformaticsPrompt for use in tests."""
    return BioinformaticsPrompt(
        research_area="Test Area",
        description=(
            "A sufficiently long description used only for testing purposes, "
            "well past the minimum length checked by validate_prompt."
        ),
        key_concepts=["Concept A", "Concept B", "Concept C"],
        common_tools=["Tool A", "Tool B", "Tool C"],
        common_file_formats=[
            {"name": "FASTA", "description": "Sequence format"},
            {"name": "VCF", "description": "Variant format"},
        ],
        examples=[
            FewShotExample(
                query="How do I run a basic alignment?",
                context="User has paired-end reads and a reference genome.",
                response="Run the following:\n```bash\nbwa mem ref.fa reads.fq > out.sam\n```",
            )
        ],
        references=["Some Reference et al. (2020)."],
    )
