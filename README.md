# Bioinformatics Prompts

[![Tests](https://github.com/geraldmc/bioinformatics_prompts/actions/workflows/tests.yml/badge.svg)](https://github.com/geraldmc/bioinformatics_prompts/actions/workflows/tests.yml)

A Python package for generating and using bioinformatics-specific prompts with Anthropic's Claude AI.

## Overview

This package provides a framework for creating, validating, and utilizing domain-specific prompts for bioinformatics research. It helps researchers generate more focused and effective interactions with Large Language Models like Claude by providing context-rich templates with key concepts, tools, file formats, and examples relevant to specific bioinformatics subfields.

The package follows the OPTIMAL model (Optimization of Prompts Through Iterative Mentoring and Assessment with an LLM chatbot) described in the paper "Empowering beginners in bioinformatics with ChatGPT" by [Shue et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10299548/)

## Features

- Structured templates for different bioinformatics research areas
- Few-shot examples to guide LLM responses
- Validation utilities to ensure prompt quality
- Seamless integration with Anthropic's Claude API
- Interactive conversation mode
- Template selection interface
- JSON serialization for easy template sharing

## Project Structure

```
bioinformatics-prompts/
├── .github/
│   └── workflows/
│       └── tests.yml         # CI: test matrix + wheel build
│
├── src/
│   └── bioinformatics_prompts/
│       ├── __init__.py        # Public API (see "Public API" below)
│       ├── claude_interaction.py  # Claude API integration
│       ├── cli.py                 # Click CLI entry point (chat/list-templates/route)
│       ├── cli_chat.py            # Interactive REPL and template picker (CLI only)
│       ├── exceptions.py          # Exception hierarchy (stdlib-only)
│       ├── matching.py            # Research-area name matching (no DSPy)
│       ├── prompt_template.py     # Data model: BioinformaticsPrompt, FewShotExample
│       ├── py.typed               # PEP 561 marker: annotations are usable downstream
│       ├── dspy_modules/           # DSPy-based automatic template routing ([routing] extra)
│       │   ├── __init__.py
│       │   ├── lm.py
│       │   └── router.py
│       ├── prompt/                # Prompt implementations
│       │   ├── __init__.py
│       │   ├── templates/         # Template definitions
│       │   │   ├── __init__.py
│       │   │   ├── ai.py
│       │   │   ├── bioinformatics_tools.py
│       │   │   ├── blockchain_bioinformatics.py
│       │   │   ├── data_standardization.py
│       │   │   ├── epigenomics.py
│       │   │   ├── genomics.py
│       │   │   ├── gwas.py
│       │   │   ├── metagenomics.py
│       │   │   ├── ngs_sequencing.py
│       │   │   ├── precision_medicine.py
│       │   │   ├── sequence_analysis.py
│       │   │   ├── single_cell.py
│       │   │   ├── synthetic_biology.py
│       │   │   └── workflow_automation.py
│       │   │
│       │   ├── artificial_intelligence_prompt.json
│       │   ├── bioinformatics_tools_prompt.json
│       │   ├── blockchain_bioinformatics_prompt.json
│       │   ├── data_standardization_prompt.json
│       │   ├── epigenomics_prompt.json
│       │   ├── genomics_prompt.json
│       │   ├── gwas_prompt.json
│       │   ├── metagenomics_prompt.json
│       │   ├── ngs_sequencing_prompt.json
│       │   ├── precision_medicine_prompt.json
│       │   ├── sequence_analysis_prompt.json
│       │   ├── single_cell_genomics_prompt.json
│       │   ├── synthetic_biology_prompt.json
│       │   └── workflow_automation_prompt.json
│       │
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           └── validation.py
│
├── tests/                    # Test suite (pytest)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_claude_interaction.py
│   ├── test_prompt_template.py
│   ├── test_router.py
│   ├── test_templates_importable.py
│   └── test_validation.py
│
├── LICENSE                   # MIT License
├── pyproject.toml            # Project metadata, dependencies, tool config
├── uv.lock                   # Locked dependency versions
└── README.md                 # This file
```

## Installation

### As a contributor

Dependencies and the virtual environment are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone https://github.com/geraldmc/bioinformatics-prompts.git
cd bioinformatics-prompts

# Install dependencies into a local .venv (creates one automatically)
uv sync
```

Run any command in that environment with `uv run <command>` (e.g. `uv run bioinformatics-prompts`), or activate it directly with `source .venv/bin/activate`.

### As a dependency of another local project

This package is installable, but not published to PyPI. Add it to another
uv-managed project directly from a local path or a git URL:

```bash
uv add /path/to/bioinformatics_prompts
# or
uv add git+https://github.com/geraldmc/bioinformatics_prompts.git
```

Then import it like any other package:

```python
from bioinformatics_prompts import ClaudeInteraction, BioinformaticsPrompt
```

#### The `routing` extra

Automatic template routing is **optional**, because it depends on DSPy, which
pulls in `litellm` and through it the OpenAI SDK. The base install deliberately
skips all of that — roughly 20 packages instead of 70, and an import that costs
milliseconds rather than half a second.

```bash
# Base install: everything except automatic routing
uv add git+https://github.com/geraldmc/bioinformatics_prompts.git

# With routing
uv add "bioinformatics-prompts[routing] @ git+https://github.com/geraldmc/bioinformatics_prompts.git"
```

Without the extra, every feature except `route_template()` and
`load_template_by_query()` works normally. Those two raise
`RoutingUnavailableError` (a subclass of `ImportError`) with an install hint:

```python
from bioinformatics_prompts import RoutingUnavailableError

try:
    matched = interaction.route_template("How do I call variants?")
except RoutingUnavailableError:
    matched = None  # fall back to explicit template selection
```

### CLI usage

Installing the package also installs a `bioinformatics-prompts` command with
three subcommands:

```bash
# Start the interactive conversation mode (also the default with no subcommand)
uv run bioinformatics-prompts chat
uv run bioinformatics-prompts

# List available prompt templates (no API key required)
uv run bioinformatics-prompts list-templates

# Route a query to the best-matching template without starting a chat
uv run bioinformatics-prompts route "How do I call variants from a VCF file?"
```

Global options, available before any subcommand, map onto
`ClaudeInteraction`'s constructor arguments:

```bash
uv run bioinformatics-prompts --api-key YOUR_KEY --model claude-sonnet-5 --prompt-dir /path/to/templates chat
```

- `--api-key` — Claude API key. Defaults to the `CLAUDE_API_KEY` or
  `ANTHROPIC_API_KEY` environment variable (or a `.env` file).
- `--model` — Default Claude model to use.
- `--prompt-dir` — Directory containing prompt template JSON files.

`list-templates` only reads local template files, so it works without an API
key configured; `chat` and `route` require one.

## Usage

### Public API

Everything below is importable directly from `bioinformatics_prompts`:

| Name | What it is |
|---|---|
| `ClaudeInteraction` | the client: loads templates, generates prompts, talks to Claude |
| `BioinformaticsPrompt` | a research-area template |
| `FewShotExample` | one worked example inside a template |
| `TemplateInfo` | one entry from `list_available_templates()` (a `TypedDict`) |
| `BioinformaticsPromptsError` | base class for every error this package raises |
| `MissingAPIKeyError`, `TemplateNotFoundError`, `TemplateLoadError`, `NoTemplateLoadedError`, `RoutingUnavailableError` | the five specific errors — see [Errors](#errors) |

Anything not in that list is an implementation detail and may move without
notice — including `cli`, `cli_chat`, `matching` and `dspy_modules`. The
exception is `bioinformatics_prompts.utils.validation`, which is documented
under [Validating a Prompt Template](#validating-a-prompt-template) and is
imported from its own module rather than the top level.

`bioinformatics_prompts.__version__` reports the installed version, read from
package metadata so `pyproject.toml` stays the single source of truth. It is
resolved on first access rather than at import time — `importlib.metadata`
costs more to import than the rest of this package combined, and an attribute
most callers never read should not be charged to every import.

The package ships a **`py.typed`** marker, so mypy and pyright use its
annotations instead of treating it as untyped. `tests/test_typing_contract.py`
type-checks a consumer written from this README on every CI run, so what is
documented here is what a type checker will accept.

### Basic Usage with Interactive Mode

The interactive conversation is a **CLI feature**, not a library one — the
library never reads stdin or writes to stdout:

```bash
uv run bioinformatics-prompts chat
```

This starts a terminal-based conversation where you may:
- Select a bioinformatics topic template
- Ask questions within that domain
- Get contextually-aware responses from Claude
- Switch templates or reset the conversation as needed

### Programmatic Usage

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

#### Errors

The library **raises**; it never reports failure through its return value and
never prompts on stdin. Everything it raises descends from
`BioinformaticsPromptsError`:

| Exception | Raised when |
|---|---|
| `MissingAPIKeyError` | no API key passed or in the environment (also a `ValueError`) |
| `TemplateNotFoundError` | no template matches the requested name, or the directory is empty |
| `TemplateLoadError` | a template file was found but could not be read or parsed |
| `NoTemplateLoadedError` | an operation needing a template ran before one was loaded |
| `RoutingUnavailableError` | routing requested without the `routing` extra (also an `ImportError`) |

All six are exported from the top level, so
`from bioinformatics_prompts import TemplateNotFoundError` works; they also
remain importable from `bioinformatics_prompts.exceptions`.

Errors from the Claude API propagate unchanged as `anthropic.AnthropicError`
subclasses, so you can catch the SDK's own typed hierarchy — `RateLimitError`,
`AuthenticationError` and the rest — rather than a flattened wrapper.

#### Choosing a model

`ClaudeInteraction(model=...)` accepts an explicit model id. If omitted, no
model is chosen at construction time — the first time a request is actually
sent, the client queries Anthropic's Models API and picks the most recently
released Sonnet-tier model as a reasonable middle-of-the-lineup default, then
caches that choice for the lifetime of the instance. If the query fails (no
network, invalid key, etc.), it falls back to a hardcoded constant
(`FALLBACK_MODEL` in `claude_interaction.py`). Pass `model=` explicitly to
skip this resolution entirely.

#### Automatic template routing

Requires the optional `routing` extra (see
[The `routing` extra](#the-routing-extra) above); without it these calls raise
`RoutingUnavailableError`.

Instead of naming a template explicitly with `load_template()`, you can route a
user's query to the best-matching template using a small DSPy-based router:

```python
interaction = ClaudeInteraction(api_key=api_key)

# Picks a template automatically based on the query text, or returns None
# if no good match is found (fall back to load_template(name) in that case).
loaded = interaction.load_template_by_query(
    "How do I call variants from bacterial WGS reads?"
)
```

This uses `dspy.LM("anthropic/<model>", ...)` under the hood (see
`dspy_modules/lm.py` and `dspy_modules/router.py`), resolving the model the
same way as `send_to_claude` (`self.default_model` if set, else
`FALLBACK_MODEL`). The `bioinformatics-prompts route` subcommand exercises
this directly (see [CLI usage](#cli-usage) above); it is not yet wired into the
`chat` subcommand's interactive loop, which still uses the numbered menu.

### Creating a Custom Template

```python
from bioinformatics_prompts import BioinformaticsPrompt, FewShotExample

# Create a custom prompt template
custom_prompt = BioinformaticsPrompt(
    research_area="The bioinformatics research area of interest",
    description="Description of your area...",
    key_concepts=["Concept 1", "Concept 2"],
    common_tools=["Tool 1", "Tool 2"],
    common_file_formats=[
        {"name": "Format1", "description": "Description of format"}
    ],
    examples=[
        FewShotExample(
            query="Example question?",
            context="Context for the example",
            response="Detailed response with examples..."
        )
    ],
    references=["Reference 1", "Reference 2"]
)

# Save the template for reuse
with open("custom_prompt.json", "w") as f:
    f.write(custom_prompt.to_json())
```

### Validating a Prompt Template

```python
from bioinformatics_prompts.utils.validation import validate_prompt

# Validate your template
validation_result = validate_prompt(custom_prompt)
print(validation_result)
```

### Logging

The package logs through the standard library's `logging` module, under the
`bioinformatics_prompts` logger. Following the guidance in the Python logging
HOWTO, it installs **only a `NullHandler`** and configures nothing else — no
handlers, no formatters, no levels. Importing it will never alter logging
configuration your application has already set up, and it produces no log
output until you configure a handler.

To see the package's log records, configure logging as you normally would:

```python
import logging

logging.basicConfig(level=logging.INFO)          # or dictConfig, or your own handlers
logging.getLogger("bioinformatics_prompts").setLevel(logging.DEBUG)  # optional
```

## Testing

```bash
# Run the test suite
uv run pytest

# Run with a coverage report
uv run pytest --cov
```

### Continuous integration

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

No secrets are configured or required: the suite fakes every network-facing
call, so CI never contacts the Claude API.

## Available Prompt Templates

The package includes pre-built templates for various bioinformatics research areas:

- **Artificial Intelligence in Bioinformatics**: ML/DL for biological data
- **Bioinformatics Tool Selection**: Evaluation and selection of appropriate tools
- **Blockchain in Bioinformatics**: Blockchain applications for biological data
- **Data Standardization**: FAIR principles and bioinformatics data management
- **Epigenomics**: DNA methylation, histone modifications, chromatin structure
- **Genomics**: DNA sequencing, assembly, variant calling
- **GWAS**: Genome-wide association studies
- **Metagenomics**: Microbial community analysis
- **NGS Sequencing Analysis**: Next-generation sequencing data processing
- **Precision Medicine**: Clinical genomics and personalized healthcare
- **Sequence Analysis**: Alignments, motif finding, phylogenetics
- **Single-Cell Genomics**: Single-cell RNA-seq and multi-omics analysis
- **Synthetic Biology**: Genetic circuit and metabolic pathway engineering
- **Workflow Automation**: Pipeline design and optimization

## Environment Variables

- `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`: Your Anthropic API key for accessing Claude

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Partly inspired by the paper "Empowering beginners in bioinformatics with ChatGPT" by Evelyn Shue et al. (2023)
- Based on the OPTIMAL model (Optimization of Prompts Through Iterative Mentoring and Assessment with an LLM chatbot)
- Uses Anthropic's Claude API for advanced AI interactions