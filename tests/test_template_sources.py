"""The authored templates and the committed JSON must not drift apart.

Each of the 14 templates exists twice: ``prompt/templates/<module>.py`` is what
a maintainer edits, and ``prompt/<variable>.json`` is what the runtime actually
reads -- ``list_available_templates()`` globs the JSON and nothing imports the
modules. Until #25 the only thing keeping the two in step was a per-module
``if __name__ == "__main__":`` block that a human had to remember to run.

These tests are that enforcement. They also pin the two facts
``scripts/regenerate_template_json.py`` relies on: every module declares
exactly one ``*_prompt`` object, and the JSON file is named after that
variable rather than after the module (``ai.py`` produces
``artificial_intelligence_prompt.json``).
"""

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import bioinformatics_prompts
from bioinformatics_prompts.prompt import templates
from bioinformatics_prompts.prompt_template import BioinformaticsPrompt

PROMPT_DIR = Path(bioinformatics_prompts.__file__).resolve().parent / "prompt"
REPO_ROOT = Path(__file__).resolve().parent.parent
REGENERATOR = REPO_ROOT / "scripts" / "regenerate_template_json.py"

# The fix, quoted verbatim in every drift failure so the message is actionable
# on its own -- a red build here means someone edited a module and did not
# regenerate, and they should not have to go find the command.
REGENERATE_HINT = "regenerate with:  python scripts/regenerate_template_json.py"

TEMPLATE_MODULES = sorted(info.name for info in pkgutil.iter_modules(templates.__path__))


def declared_prompts(module_name: str) -> dict:
    """The ``*_prompt`` objects a template module defines, by variable name."""
    module = importlib.import_module(f"{templates.__name__}.{module_name}")
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and name.endswith("_prompt")
        and isinstance(value, BioinformaticsPrompt)
    }


def sole_prompt(module_name: str) -> tuple:
    """The single (variable name, prompt) pair a template module declares."""
    declared = declared_prompts(module_name)
    assert len(declared) == 1, f"{module_name}.py declares {sorted(declared)}"
    return next(iter(declared.items()))


def test_there_are_fourteen_template_modules():
    assert len(TEMPLATE_MODULES) == 14, TEMPLATE_MODULES


@pytest.mark.parametrize("module_name", TEMPLATE_MODULES)
def test_every_template_module_declares_exactly_one_prompt(module_name):
    """The rule the generated filename is derived from.

    A module with two ``*_prompt`` objects would silently have one of them
    dropped by the regenerator; a module with none would be skipped entirely.
    """
    declared = declared_prompts(module_name)
    assert len(declared) == 1, (
        f"{module_name}.py declares {len(declared)} BioinformaticsPrompt objects "
        f"({sorted(declared)}); the regenerator derives the JSON filename from "
        "the variable name and needs exactly one"
    )


@pytest.mark.parametrize("module_name", TEMPLATE_MODULES)
def test_committed_json_matches_the_authored_source(module_name):
    """The invariant this whole file exists for."""
    variable, prompt = sole_prompt(module_name)
    committed = PROMPT_DIR / f"{variable}.json"

    assert committed.is_file(), (
        f"{module_name}.py declares {variable} but {committed.name} does not "
        f"exist; {REGENERATE_HINT}"
    )
    assert committed.read_text(encoding="utf-8") == prompt.to_json(), (
        f"{committed.name} has drifted from {module_name}.py; {REGENERATE_HINT}"
    )


def test_no_orphan_json_files():
    """Every committed template has a module that can regenerate it.

    An orphan would keep working -- the runtime reads JSON -- while being
    unreachable from the authoring source, which is the drift this file is
    meant to make impossible.
    """
    authored = {
        variable for module_name in TEMPLATE_MODULES for variable in declared_prompts(module_name)
    }
    # rglob, not glob: list_available_templates() reads "**/*.json", so a stray
    # file in a subdirectory would be loaded at runtime and must be caught here.
    committed = {path.stem for path in PROMPT_DIR.rglob("*.json")}

    assert committed == authored, (
        f"JSON with no authoring module: {sorted(committed - authored)}; "
        f"modules with no JSON: {sorted(authored - committed)}"
    )


# ---------------------------------------------------------------------------
# The regenerator
# ---------------------------------------------------------------------------


def test_the_regenerator_reproduces_every_committed_file(tmp_path):
    """The script that produces the JSON, run for real against a scratch dir.

    The tests above compare the committed files to ``to_json()`` in memory, so
    they would stay green even if the script itself rotted. This runs it as a
    subprocess -- the way a maintainer does -- and writes nothing into the tree.
    """
    result = subprocess.run(
        [sys.executable, str(REGENERATOR), "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    committed = sorted(PROMPT_DIR.glob("*.json"))
    assert {path.name for path in tmp_path.glob("*.json")} == {
        path.name for path in committed
    }, "the regenerator wrote a different set of files than is committed"

    for path in committed:
        assert (tmp_path / path.name).read_bytes() == path.read_bytes(), (
            f"the regenerator does not reproduce {path.name}"
        )


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------

# One realistic query per research area, carried over from the per-module
# `if __name__ == "__main__":` blocks that #25 deleted. Each of those blocks
# ended by asserting that the JSON it had just written regenerated a prompt
# identical to the one built from Python -- a real invariant that ran only when
# a human executed the module by hand. It now runs on every test run instead,
# and the queries survive as data rather than being deleted with the blocks.
SAMPLE_QUERIES = {
    "artificial_intelligence_prompt": (
        "How can I use deep learning to analyze single-cell RNA-seq data?"
    ),
    "bioinformatics_tools_prompt": (
        "What considerations should I make when selecting alignment tools for my "
        "RNA-seq experiment?"
    ),
    "blockchain_bioinformatics_prompt": (
        "How can I implement blockchain to ensure privacy and security of patient "
        "genomic data?"
    ),
    "data_standardization_prompt": (
        "What standards should I follow when preparing my multi-omics dataset for "
        "publication?"
    ),
    "epigenomics_prompt": (
        "How do I analyze histone modification ChIP-seq data to identify cell-type "
        "specific enhancers?"
    ),
    "genomics_prompt": (
        "I have Illumina paired-end reads from a bacterial sample. How can I "
        "assemble and annotate the genome?"
    ),
    "gwas_prompt": "How do I interpret GWAS results when I have hundreds of significant hits?",
    "metagenomics_prompt": (
        "How can I compare the taxonomic composition between different environmental "
        "samples?"
    ),
    "ngs_sequencing_prompt": (
        "What are the best practices for variant calling in whole genome sequencing data?"
    ),
    "precision_medicine_prompt": (
        "What are the best practices for interpreting and reporting incidental "
        "findings from whole genome sequencing?"
    ),
    "sequence_analysis_prompt": (
        "How do I compare protein sequences from different bacterial species?"
    ),
    "single_cell_genomics_prompt": (
        "How do I interpret cell clusters in my single-cell RNA-seq data from a "
        "mixed tissue sample?"
    ),
    "synthetic_biology_prompt": (
        "How do I design a genetic circuit for biosensing environmental toxins?"
    ),
    "workflow_automation_prompt": (
        "How do I design a scalable and reproducible bioinformatics workflow for "
        "bacterial genome assembly and annotation?"
    ),
}


def test_every_template_has_a_sample_query():
    """The query table is data about the 14 templates and must not go stale."""
    authored = {
        variable for module_name in TEMPLATE_MODULES for variable in declared_prompts(module_name)
    }
    assert set(SAMPLE_QUERIES) == authored, (
        f"templates with no sample query: {sorted(authored - set(SAMPLE_QUERIES))}; "
        f"queries for no template: {sorted(set(SAMPLE_QUERIES) - authored)}"
    )


AUTHORED_BY_VARIABLE = {
    variable: prompt
    for module_name in TEMPLATE_MODULES
    for variable, prompt in declared_prompts(module_name).items()
}


@pytest.mark.parametrize("variable", sorted(SAMPLE_QUERIES))
def test_committed_json_round_trips_to_an_identical_prompt(variable):
    """A template loaded from JSON generates the same prompt as the source object.

    Stronger than the byte comparison above, and in a different direction: this
    is what a consumer actually receives, so it would catch a to_json() /
    from_json() pair that dropped a field on both sides consistently.
    """
    query = SAMPLE_QUERIES[variable]
    committed = (PROMPT_DIR / f"{variable}.json").read_text(encoding="utf-8")

    loaded = BioinformaticsPrompt.from_json(committed)

    assert loaded.generate_prompt(query) == AUTHORED_BY_VARIABLE[variable].generate_prompt(
        query
    ), f"{variable}.json generates a different prompt than its authoring module"
    assert loaded.to_json() == committed, f"{variable}.json does not survive a load/dump cycle"
