"""The documentation site must keep up with the package (#24).

Two things here can silently fall out of step with the code, and neither is
caught by building the site -- `mkdocs build --strict` fails on a broken *link*,
not on a name nobody thought to document:

- the API reference names its twelve targets explicitly, so a name added to
  `__all__` would simply go undocumented, and
- the template catalogue is generated at build time from the JSON, so nothing
  in the repo records how many pages it should produce.

**These tests read text and JSON only.** They must never import mkdocs or any
docs plugin: the core-only CI job installs `--no-default-groups --group test`
and runs this whole suite without the `docs` dependency group.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

import bioinformatics_prompts

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
REFERENCE = DOCS / "reference"
PROMPT_DIR = Path(bioinformatics_prompts.__file__).resolve().parent / "prompt"

# `::: bioinformatics_prompts.utils.validation.validate_prompt` -> the last
# segment is the name a reader looks up.
DIRECTIVE = re.compile(r"^:::\s+(?P<target>[\w.]+)\s*$", re.MULTILINE)


def documented_names() -> set:
    """Every object the reference pages point mkdocstrings at, by leaf name."""
    found = set()
    for page in sorted(REFERENCE.glob("*.md")):
        for match in DIRECTIVE.finditer(page.read_text(encoding="utf-8")):
            found.add(match.group("target").rsplit(".", 1)[-1])
    return found


# ---------------------------------------------------------------------------
# The API reference
# ---------------------------------------------------------------------------


def test_every_exported_name_is_documented():
    """The reference covers exactly `__all__` -- no more, no less.

    The pages name their targets explicitly rather than rendering the package
    wholesale, which keeps them readable and independent of how Griffe resolves
    re-exported aliases. The cost is a list that could go stale; this is what
    stops it.
    """
    exported = set(bioinformatics_prompts.__all__)
    documented = documented_names()

    assert documented == exported, (
        f"exported but undocumented: {sorted(exported - documented)}; "
        f"documented but not exported: {sorted(documented - exported)}. "
        f"Add a `::: bioinformatics_prompts.<module>.<name>` line under {REFERENCE}"
    )


def test_reference_directives_point_at_importable_objects():
    """A `:::` target that no longer resolves renders as an empty section.

    mkdocs only warns about this in some configurations, and a blank heading is
    easy to miss in a rendered page.
    """
    for page in sorted(REFERENCE.glob("*.md")):
        for match in DIRECTIVE.finditer(page.read_text(encoding="utf-8")):
            target = match.group("target")
            module_path, _, attribute = target.rpartition(".")
            module = importlib.import_module(module_path)
            assert hasattr(module, attribute), (
                f"{page.name} documents {target}, which does not exist"
            )


# ---------------------------------------------------------------------------
# The template catalogue
# ---------------------------------------------------------------------------


def load_catalogue_module():
    """Import scripts/template_catalogue.py by path.

    `scripts/` is maintainer tooling, deliberately outside the package and off
    sys.path (#25), so the test reaches it the same way the docs build does.
    """
    path = REPO_ROOT / "scripts" / "template_catalogue.py"
    spec = importlib.util.spec_from_file_location("template_catalogue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE_FILES = sorted(PROMPT_DIR.glob("*.json"))


def test_there_are_fourteen_templates_to_catalogue():
    assert len(TEMPLATE_FILES) == 14, [p.name for p in TEMPLATE_FILES]


@pytest.mark.parametrize("path", TEMPLATE_FILES, ids=lambda p: p.stem)
def test_every_template_renders_a_catalogue_page(path):
    """One page per template, carrying what a reader came to look up."""
    catalogue = load_catalogue_module()
    data = json.loads(path.read_text(encoding="utf-8"))

    page = catalogue.render_page(data)

    assert page.startswith(f"# {data['research_area']}"), (
        "the page must open with the research area as its title"
    )
    assert data["description"].split(".")[0] in page
    for concept in data["key_concepts"]:
        assert concept in page, f"key concept missing from the page: {concept}"
    for fmt in data["common_file_formats"]:
        assert fmt["name"] in page, f"file format missing from the page: {fmt['name']}"
    assert data["examples"][0]["query"] in page, "the first example must appear"


def test_catalogue_covers_every_template_and_nothing_else():
    """The generator discovers templates rather than carrying a list."""
    catalogue = load_catalogue_module()

    discovered = {path.stem for path, _ in catalogue.iter_templates(PROMPT_DIR)}

    assert discovered == {path.stem for path in TEMPLATE_FILES}


# ---------------------------------------------------------------------------
# The README
# ---------------------------------------------------------------------------

README = REPO_ROOT / "README.md"

# It was 481 lines and 10 top-level sections when #24 was written, having grown
# 36 lines in #25 alone because a contributor topic had nowhere else to go. The
# budget is what stops it silently becoming a documentation site again; raising
# it should take an argument, not a commit.
README_LINE_BUDGET = 90


def test_readme_stays_a_landing_page():
    lines = README.read_text(encoding="utf-8").splitlines()

    assert len(lines) <= README_LINE_BUDGET, (
        f"README is {len(lines)} lines, over the {README_LINE_BUDGET}-line budget. "
        "New prose belongs in docs/, not here"
    )


def test_readme_points_at_the_documented_import_path():
    """The contradiction #24 settled must not come back.

    The README told people to `from bioinformatics_prompts.utils.validation
    import validate_prompt` while #21 reserved the right to move anything
    outside `__all__`. The name is exported now, so the deep path should be
    gone from the docs a reader follows first.
    """
    text = README.read_text(encoding="utf-8")

    assert "bioinformatics_prompts.utils.validation" not in text, (
        "README reaches past the public surface; import validate_prompt from "
        "the package root instead"
    )


def test_readme_links_to_the_documentation_site():
    text = README.read_text(encoding="utf-8")

    assert "geraldmc.github.io/bioinformatics_prompts" in text, (
        "the landing page must link to the site the rest of the docs moved to"
    )


def test_catalogue_summary_lists_every_page():
    """literate-nav builds the Templates section from this file.

    A page missing from SUMMARY.md still builds, but is unreachable from the
    navigation -- so it would be invisible rather than broken.
    """
    catalogue = load_catalogue_module()

    summary = catalogue.render_summary(catalogue.iter_templates(PROMPT_DIR))

    for path in TEMPLATE_FILES:
        assert f"{path.stem}.md" in summary, f"{path.stem} missing from SUMMARY.md"
    assert summary.count("\n-") + summary.count("- ") >= len(TEMPLATE_FILES)
