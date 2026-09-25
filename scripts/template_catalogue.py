"""Render the bundled prompt templates as documentation pages (#24).

The 14 templates are the substance of this package and could previously only be
browsed by opening JSON files. This turns each one into a page.

Reads the **JSON**, not the authoring modules in `prompt/templates/`, because
the JSON is what `list_available_templates()` globs — so the catalogue
describes what a consumer actually gets. That also gives the template format a
second consumer, which notices a breaking change.

Pure stdlib on purpose: `tests/test_docs.py` imports this module, and the
core-only CI job runs the test suite without the `docs` dependency group. The
mkdocs side lives in `gen_template_pages.py` beside it, which is a thin wrapper.

Like `regenerate_template_json.py` beside it, this is maintainer tooling and is
not shipped in the wheel. Unlike it, nothing here is ever written to disk: the
pages are generated into the docs build every time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Tuple

Template = Tuple[Path, dict]


def iter_templates(prompt_dir: Path) -> Iterator[Template]:
    """Yield (path, parsed template) for every JSON file, in filename order.

    Args:
        prompt_dir: The directory holding the template JSON, normally the
            package's `prompt/`.

    Yields:
        Each template file paired with its parsed contents.
    """
    for path in sorted(prompt_dir.glob("*.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def _title(data: dict) -> str:
    return data["research_area"]


def render_page(data: dict) -> str:
    """Render one template as a Markdown page.

    The section order follows the template's own structure rather than an
    invented one, so a reader who has seen `generate_prompt()` output
    recognises it.

    Args:
        data: A parsed template, as `BioinformaticsPrompt.to_json()` writes it.

    Returns:
        The complete Markdown page, ending in a single newline.
    """
    out = [f"# {_title(data)}", "", data["description"], ""]

    out += ["## Key concepts", ""]
    out += [f"- {concept}" for concept in data["key_concepts"]]
    out += [""]

    out += ["## Common tools", ""]
    out += [f"- {tool}" for tool in data["common_tools"]]
    out += [""]

    out += ["## Common file formats", "", "| Format | Description |", "|---|---|"]
    out += [
        f"| `{fmt['name']}` | {fmt['description']} |"
        for fmt in data["common_file_formats"]
    ]
    out += [""]

    out += [
        "## Examples",
        "",
        "The few-shot examples this template sends to Claude alongside your question.",
        "",
    ]
    for index, example in enumerate(data["examples"], 1):
        # Collapsed: a single example response runs to well over a thousand
        # characters, and several of them would bury the sections above.
        out += [
            f'??? example "Example {index} — {example["query"]}"',
            "",
            f"    **Context.** {example['context']}",
            "",
        ]
        out += [f"    {line}" if line else "" for line in example["response"].splitlines()]
        out += [""]

    if data.get("references"):
        out += ["## References", ""]
        out += [f"- {reference}" for reference in data["references"]]
        out += [""]

    return "\n".join(out).rstrip("\n") + "\n"


def render_index(templates: Iterable[Template]) -> str:
    """Render the catalogue's landing page.

    Gives the section a real page to link to -- `catalogue/` alone is a
    directory, which MkDocs cannot resolve as a link target -- and lets a
    reader scan all 14 areas before opening one.

    Args:
        templates: The (path, data) pairs `iter_templates` yields.

    Returns:
        The complete Markdown page.
    """
    out = [
        "# Templates",
        "",
        "The 14 research areas bundled with the package. Each carries key concepts,",
        "common tools, file formats and worked examples, and is what",
        "`load_template()` returns.",
        "",
        "| Research area | Covers |",
        "|---|---|",
    ]
    for path, data in templates:
        # First sentence only: several descriptions run to three or four.
        summary = data["description"].split(". ")[0].rstrip(".")
        out.append(f"| [{_title(data)}]({path.stem}.md) | {summary} |")
    out += [
        "",
        "These pages are generated from the template JSON on every docs build, so",
        "they cannot drift from what the package actually ships. To write your own,",
        "see [Creating a custom template](../guide/custom-templates.md).",
    ]
    return "\n".join(out) + "\n"


def render_summary(templates: Iterable[Template]) -> str:
    """Render the literate-nav file for the Templates section.

    mkdocs.yml delegates that section with a trailing slash (`Templates:
    catalogue/`), so this file -- not a hand-maintained `nav:` list -- decides
    which pages appear and in what order.

    Args:
        templates: The (path, data) pairs `iter_templates` yields.

    Returns:
        A Markdown list: the landing page, then one link per template.
    """
    # Front matter, not a list item: literate-nav marks its own nav file
    # NOT_IN_NAV rather than excluded, so the page is built and would otherwise
    # turn up in site search as a stray list of links.
    lines = ["---", "search:", "  exclude: true", "---", ""]
    lines += ["- [Overview](index.md)"]
    lines += [f"- [{_title(data)}]({path.stem}.md)" for path, data in templates]
    return "\n".join(lines) + "\n"
