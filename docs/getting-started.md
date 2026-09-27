# Getting started

## Install

This package is installable, but not published to PyPI. Add it to a uv-managed
project directly from a local path or a git URL:

```bash
uv add /path/to/bioinformatics_prompts
# or
uv add git+https://github.com/geraldmc/bioinformatics_prompts.git
```

Then import it like any other package:

```python
from bioinformatics_prompts import ClaudeInteraction, BioinformaticsPrompt
```

!!! tip "Automatic routing is optional"

    Routing a query to a template automatically needs the `routing` extra,
    which is deliberately not installed by default — see
    [Automatic routing](guide/routing.md).

## Configure a key

Set your Anthropic API key in the environment, or in a `.env` file:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key for accessing Claude |
| `CLAUDE_API_KEY` | Accepted as an alternative |

You can also pass it directly as `ClaudeInteraction(api_key=...)`. Listing
templates needs no key at all.

## Your first question

```python
from bioinformatics_prompts import ClaudeInteraction

interaction = ClaudeInteraction()          # reads the key from the environment
interaction.load_template("Genomics")

print(interaction.ask_claude("How do I identify SNPs in my bacterial genome?"))
```

`load_template` matches exactly, on the research area or the filename stem. A
name that does not exist raises `TemplateNotFoundError` listing the valid ones.

To see what is available first:

```python
for number, t in enumerate(interaction.list_available_templates(), 1):
    print(f"{number}. {t['research_area']}")
```

Or browse them here: **[Templates](catalogue/index.md)**.

## Without writing code

Installing the package also installs a `bioinformatics-prompts` command:

```bash
uv run bioinformatics-prompts list-templates
uv run bioinformatics-prompts chat
```

See the [CLI guide](guide/cli.md).
