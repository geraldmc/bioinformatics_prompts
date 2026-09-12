"""Tests for the DSPy-based template router.

None of these tests call the real Claude API or a real dspy.LM; DSPy's
DummyLM stands in for the LM backend everywhere a DSPy call happens.
"""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from bioinformatics_prompts.claude_interaction import ClaudeInteraction
from bioinformatics_prompts.dspy_modules.router import TemplateRouter, match_area
from bioinformatics_prompts.prompt.templates.prompt_template import BioinformaticsPrompt, FewShotExample


# ---------------------------------------------------------------------------
# match_area — plain-dict unit tests, no DSPy involved at all
# ---------------------------------------------------------------------------

AREAS = [
    {"research_area": "Genomics", "description": "DNA sequencing and assembly"},
    {"research_area": "Single-Cell Genomics", "description": "Single-cell RNA-seq analysis"},
]


def test_match_area_exact_match():
    assert match_area("Genomics", AREAS) == AREAS[0]


def test_match_area_case_insensitive():
    assert match_area("genomics", AREAS) == AREAS[0]


def test_match_area_substring_fallback():
    # Predicted string contains extra text around a listed area name.
    assert match_area("The best match is Single-Cell Genomics.", AREAS) == AREAS[1]


def test_match_area_no_match_returns_none():
    assert match_area("Astrophysics", AREAS) is None


# ---------------------------------------------------------------------------
# TemplateRouter — DSPy module tested in isolation via dspy.configure(DummyLM)
# ---------------------------------------------------------------------------

def test_template_router_forward_returns_predicted_research_area():
    dspy.configure(lm=DummyLM([{"research_area": "Genomics"}]))

    router = TemplateRouter()
    prediction = router(question="How do I assemble a genome?", areas=AREAS)

    assert prediction.research_area == "Genomics"


# ---------------------------------------------------------------------------
# ClaudeInteraction integration — DummyLM injected via monkeypatched
# configure_claude_lm, since ClaudeInteraction.route_template() would
# otherwise call the real configure_claude_lm() and build a real,
# network-calling dspy.LM.
# ---------------------------------------------------------------------------

def _stub_configure_claude_lm(dummy_lm):
    """Build a drop-in replacement for configure_claude_lm that configures
    the given DummyLM instead of a real dspy.LM."""

    def _configure(model, api_key=None):
        dspy.configure(lm=dummy_lm)
        return dummy_lm

    return _configure


@pytest.fixture
def genomics_prompt():
    return BioinformaticsPrompt(
        research_area="Genomics",
        description=(
            "DNA sequencing, genome assembly, and variant calling workflows, "
            "well past the minimum length checked by validate_prompt."
        ),
        key_concepts=["Assembly", "Variant calling"],
        common_tools=["BWA", "GATK"],
        common_file_formats=[{"name": "FASTA", "description": "Sequence format"}],
        examples=[
            FewShotExample(
                query="How do I align reads?",
                context="Paired-end reads and a reference genome.",
                response="Run bwa mem.",
            )
        ],
        references=[],
    )


@pytest.fixture
def single_cell_prompt():
    return BioinformaticsPrompt(
        research_area="Single-Cell Genomics",
        description=(
            "Single-cell RNA-seq and multi-omics analysis, including clustering "
            "and trajectory inference, well past the minimum length checked."
        ),
        key_concepts=["Clustering", "Trajectory inference"],
        common_tools=["Scanpy", "Seurat"],
        common_file_formats=[{"name": "H5AD", "description": "AnnData format"}],
        examples=[
            FewShotExample(
                query="How do I cluster cells?",
                context="A processed single-cell count matrix.",
                response="Use Leiden clustering.",
            )
        ],
        references=[],
    )


@pytest.fixture
def prompt_dir(tmp_path, genomics_prompt, single_cell_prompt):
    (tmp_path / "genomics_prompt.json").write_text(genomics_prompt.to_json())
    (tmp_path / "single_cell_prompt.json").write_text(single_cell_prompt.to_json())
    return tmp_path


def test_list_available_templates_includes_description(prompt_dir):
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    templates = interaction.list_available_templates()

    descriptions = {t["research_area"]: t["description"] for t in templates}
    assert descriptions["Genomics"].startswith("DNA sequencing")
    assert descriptions["Single-Cell Genomics"].startswith("Single-cell RNA-seq")


def test_route_template_returns_matched_template(prompt_dir, monkeypatch):
    dummy_lm = DummyLM([{"research_area": "Genomics"}])
    monkeypatch.setattr(
        "bioinformatics_prompts.claude_interaction.configure_claude_lm",
        _stub_configure_claude_lm(dummy_lm),
    )

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    matched = interaction.route_template("How do I assemble a bacterial genome?")

    assert matched is not None
    assert matched["research_area"] == "Genomics"


def test_route_template_no_match_returns_none(prompt_dir, monkeypatch):
    dummy_lm = DummyLM([{"research_area": "Astrophysics"}])
    monkeypatch.setattr(
        "bioinformatics_prompts.claude_interaction.configure_claude_lm",
        _stub_configure_claude_lm(dummy_lm),
    )

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    matched = interaction.route_template("How do I assemble a bacterial genome?")

    assert matched is None


def test_load_prompt_template_by_query_loads_matched_template(prompt_dir, monkeypatch):
    dummy_lm = DummyLM([{"research_area": "Single-Cell Genomics"}])
    monkeypatch.setattr(
        "bioinformatics_prompts.claude_interaction.configure_claude_lm",
        _stub_configure_claude_lm(dummy_lm),
    )

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    loaded = interaction.load_prompt_template_by_query("How do I cluster my cells?")

    assert loaded is not None
    assert loaded.research_area == "Single-Cell Genomics"
    assert interaction.prompt_template is loaded


def test_load_prompt_template_by_query_no_match_returns_none(prompt_dir, monkeypatch):
    dummy_lm = DummyLM([{"research_area": "Astrophysics"}])
    monkeypatch.setattr(
        "bioinformatics_prompts.claude_interaction.configure_claude_lm",
        _stub_configure_claude_lm(dummy_lm),
    )

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    loaded = interaction.load_prompt_template_by_query("How do I cluster my cells?")

    assert loaded is None
