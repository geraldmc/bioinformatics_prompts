# Using the library

## The public API

Everything below is importable directly from `bioinformatics_prompts`:

| Name | What it is |
|---|---|
| `ClaudeInteraction` | the client: loads templates, generates prompts, talks to Claude |
| `BioinformaticsPrompt` | a research-area template |
| `FewShotExample` | one worked example inside a template |
| `TemplateInfo` | one entry from `list_available_templates()` (a `TypedDict`) |
| `validate_prompt` | check a template for missing examples and thin descriptions |
| `ValidationResult` | what `validate_prompt` returns (a `TypedDict`) |
| `BioinformaticsPromptsError` | base class for every error this package raises |
| `MissingAPIKeyError`, `TemplateNotFoundError`, `TemplateLoadError`, `NoTemplateLoadedError`, `RoutingUnavailableError` | the five specific errors — see [Errors](errors.md) |

Anything not in that list is an implementation detail and may move without
notice — including `cli`, `cli_chat`, `matching` and `dspy_modules`. The [API
reference](../reference/index.md) documents exactly these twelve names, and a
test asserts it stays that way.

`bioinformatics_prompts.__version__` reports the installed version, read from
package metadata so `pyproject.toml` stays the single source of truth. It is
resolved on first access rather than at import time — `importlib.metadata`
costs more to import than the rest of this package combined, and an attribute
most callers never read should not be charged to every import.

The package ships a **`py.typed`** marker, so mypy and pyright use its
annotations instead of treating it as untyped. `tests/test_typing_contract.py`
type-checks a consumer written from these pages on every CI run, so what is
documented here is what a type checker will accept.

## Programmatic usage

```python
from bioinformatics_prompts import ClaudeInteraction

# Initialize with API key
api_key = "your_anthropic_api_key"  # or set as environment variable
interaction = ClaudeInteraction(api_key=api_key)

# Load a template by research area, or by filename stem. Matching is exact:
# a name that doesn't exist raises TemplateNotFoundError listing the valid ones.
interaction.load_template("Genomics")

# Or list what's available first
# Each entry is a TemplateInfo: filename, research_area, description.
# Entries describe a template, not its position — number them yourself if
# you are presenting a menu.
templates = interaction.list_available_templates()
for number, t in enumerate(templates, 1):
    print(f"{number}. {t['research_area']}")

# Ask a question using the loaded template
response = interaction.ask_claude("How do I identify SNPs in my bacterial genome?")
print(response)

# To inspect the prompt that would be sent, ask for it directly
print(interaction.generate_prompt("How do I identify SNPs?"))
```

## Using your own templates

`ClaudeInteraction(prompt_dir=...)` — and the CLI's `--prompt-dir` — point the
loader at any directory of template JSON, replacing the bundled 14. You do not
need to modify the package to ship your own; see
[Creating a custom template](custom-templates.md).

## Choosing a model

`ClaudeInteraction(model=...)` accepts an explicit model id. If omitted, no
model is chosen at construction time — the first time a request is actually
sent, the client queries Anthropic's Models API and picks the most recently
released Sonnet-tier model as a reasonable middle-of-the-lineup default, then
caches that choice for the lifetime of the instance. If the query fails (no
network, invalid key, etc.), it falls back to a hardcoded constant
(`FALLBACK_MODEL` in `claude_interaction.py`). Pass `model=` explicitly to skip
this resolution entirely.
