#!/usr/bin/env python
"""Regenerate the committed template JSON from the authored Python modules.

The 14 templates are authored as Python (``prompt/templates/*.py``) because
their examples are long-form markdown with embedded code fences: JSON has no
multi-line string literal, so the same content on disk is one 1,600-character
line that no reviewer can read in a diff. The runtime, however, reads only the
JSON. This script is the bridge, and
``tests/test_template_sources.py::test_committed_json_matches_the_authored_source``
is what stops the two from drifting apart when someone forgets to run it.

Usage::

    python scripts/regenerate_template_json.py       # rewrite the committed files
    python scripts/regenerate_template_json.py --output-dir DIR

It replaces the 14 per-module ``if __name__ == "__main__":`` blocks that used
to do this one template at a time.

This is a maintainer tool and deliberately lives outside the package: it is not
shipped in the wheel, because a consumer who wants their own templates points
``ClaudeInteraction(prompt_dir=...)`` at a directory rather than regenerating
ours.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Iterator, Tuple

import bioinformatics_prompts
from bioinformatics_prompts.prompt import templates
from bioinformatics_prompts.prompt_template import BioinformaticsPrompt

DEFAULT_OUTPUT_DIR = Path(bioinformatics_prompts.__file__).resolve().parent / "prompt"


def iter_authored_templates() -> Iterator[Tuple[str, str, BioinformaticsPrompt]]:
    """Yield (module name, variable name, prompt) for every template module.

    The output filename comes from the *variable* name, not the module name --
    ``ai.py`` declares ``artificial_intelligence_prompt`` and produces
    ``artificial_intelligence_prompt.json``. Every module declares exactly one
    such object, which is asserted both here and in the test suite.
    """
    for info in sorted(pkgutil.iter_modules(templates.__path__), key=lambda i: i.name):
        module = importlib.import_module(f"{templates.__name__}.{info.name}")
        declared = [
            (name, value)
            for name, value in vars(module).items()
            if not name.startswith("_")
            and name.endswith("_prompt")
            and isinstance(value, BioinformaticsPrompt)
        ]
        if len(declared) != 1:
            raise SystemExit(
                f"{info.name}.py declares {len(declared)} BioinformaticsPrompt objects "
                f"({sorted(name for name, _ in declared)}); expected exactly one, since "
                "the JSON filename is derived from the variable name"
            )
        variable, prompt = declared[0]
        yield info.name, variable, prompt


def regenerate(output_dir: Path) -> list[Path]:
    """Write every template's JSON into ``output_dir``; return what was written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for module_name, variable, prompt in iter_authored_templates():
        path = output_dir / f"{variable}.json"
        # No trailing newline: this is what BioinformaticsPrompt.to_json()
        # returns, and the committed files match it byte for byte.
        path.write_text(prompt.to_json(), encoding="utf-8")
        written.append(path)
        print(f"{module_name}.py -> {path.name}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "where to write the JSON files "
            "(default: the package's prompt/ directory, i.e. rewrite in place)"
        ),
    )
    args = parser.parse_args(argv)

    written = regenerate(args.output_dir)
    print(f"regenerated {len(written)} templates in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
