# Contributing

## Development setup

Dependencies and the virtual environment are managed with
[uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone https://github.com/geraldmc/bioinformatics_prompts.git
cd bioinformatics_prompts

# Install dependencies into a local .venv (creates one automatically)
uv sync
```

Run any command in that environment with `uv run <command>` (e.g.
`uv run bioinformatics-prompts`), or activate it directly with
`source .venv/bin/activate`.

The dependency groups are split so CI can install less than a developer does:

| Group | Contents | Installed by |
|---|---|---|
| `test` | pytest, pytest-cov, mypy | `uv sync --no-default-groups --group test` |
| `dev` | `test` plus dspy | a bare `uv sync` |
| `docs` | mkdocs, Material, mkdocstrings, gen-files, literate-nav | `uv sync --group docs` |

`docs` is not part of `dev` on purpose: the everyday contributor loop should not
pay for the documentation toolchain, and the `core-only` CI job proves the test
suite needs none of it.

## Testing

```bash
# Run the test suite
uv run pytest

# Run with a coverage report
uv run pytest --cov
```

## Building the docs

```bash
uv sync --group docs
uv run mkdocs serve          # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict # what CI runs; any warning is an error
```

`--strict` is not optional in CI. Link and anchor validation are raised above
their defaults in `mkdocs.yml`, so a dead cross-reference fails the build rather
than shipping as a broken link.

Two parts of the site are generated and must not be committed:

- **The API reference** renders from the docstrings in `src/`. mkdocstrings
  reads them statically through Griffe, so the package is never imported — the
  docs build cannot be broken by an import-time regression, and needs neither
  `dspy` nor an installed wheel.
- **The template catalogue** is written at build time by
  `docs/gen_template_pages.py` from the template JSON. `docs/templates/` never
  exists on disk.

`tests/test_docs.py` guards both: it asserts the reference documents exactly
`__all__`, and that the catalogue renders a page per template. It reads text
and JSON only and never imports mkdocs, so it runs in the `core-only` job too.

## Changing a bundled template

Each of the 14 templates exists twice:
`src/bioinformatics_prompts/prompt/templates/<area>.py` is the authored source,
and `src/bioinformatics_prompts/prompt/<name>_prompt.json` is what the runtime
reads. **Edit the Python, never the JSON**, then regenerate:

```bash
# Edit the template
$EDITOR src/bioinformatics_prompts/prompt/templates/genomics.py

# Rewrite every JSON file from its module
uv run python scripts/regenerate_template_json.py

# Commit both the module and the regenerated JSON
git add src/bioinformatics_prompts/prompt/
```

The JSON is generated, but it is committed rather than built on install, because
it is the package data the wheel ships. `tests/test_template_sources.py`
compares every committed file to its module on every test run, so forgetting the
regeneration step fails the suite — on all five Python versions in CI — with a
message naming the template and the command to fix it.

Templates are authored in Python rather than as data because their examples are
long-form markdown with embedded code fences. JSON has no multi-line string
literal, so one example is a single 1,600-character line, and a one-word change
to it would arrive in a pull request as a whole rewritten line.

To ship *different* templates rather than change these, you do not need any of
this — see [Creating a custom template](guide/custom-templates.md).

## Continuous integration

`.github/workflows/tests.yml` runs on every push and pull request:

- **`test`** — the suite against the locked dependency set on Python 3.10, 3.11,
  3.12, 3.13 and 3.14. Dependencies install with `uv sync --locked`, so a
  `uv.lock` that has drifted from `pyproject.toml` fails the build rather than
  being silently re-resolved.
- **`core-only`** — installs without the `routing` extra and asserts that
  importing the package pulls in neither `dspy` nor the OpenAI SDK, that routing
  fails with an actionable error, and that the rest of the suite still passes.
  This is the configuration consumers get and the one a developer checkout never
  reproduces, since `dspy` stays in the `dev` group.
- **`build`** — `uv build`, then a check that the wheel still ships all 14
  prompt template JSON files and that the installed package can load them.
- **`docs`** — `mkdocs build --strict`.
- **`deploy-docs`** — publishes this site to GitHub Pages. Runs on `main` only.

No secrets are configured or required: the suite fakes every network-facing
call, so CI never contacts the Claude API.
