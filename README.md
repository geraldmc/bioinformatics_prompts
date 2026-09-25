# Bioinformatics Prompts

[![Tests](https://github.com/geraldmc/bioinformatics_prompts/actions/workflows/tests.yml/badge.svg)](https://github.com/geraldmc/bioinformatics_prompts/actions/workflows/tests.yml)

A Python package for generating and using bioinformatics-specific prompts with
Anthropic's Claude AI.

**📖 [Documentation](https://geraldmc.github.io/bioinformatics_prompts/)** —
guides, all 14 templates, and the full API reference.

## Overview

This package provides a framework for creating, validating, and utilizing
domain-specific prompts for bioinformatics research. It helps researchers
generate more focused and effective interactions with Large Language Models like
Claude by providing context-rich templates with key concepts, tools, file
formats, and examples relevant to specific bioinformatics subfields.

The package follows the OPTIMAL model (Optimization of Prompts Through Iterative
Mentoring and Assessment with an LLM chatbot) described in the paper "Empowering
beginners in bioinformatics with ChatGPT" by
[Shue et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10299548/)

Fourteen research areas ship with it — genomics, single-cell, GWAS,
epigenomics, metagenomics, precision medicine and more. Browse them in the
[template catalogue](https://geraldmc.github.io/bioinformatics_prompts/catalogue/).

## Install

Installable, but not published to PyPI. Add it from a local path or a git URL:

```bash
uv add git+https://github.com/geraldmc/bioinformatics_prompts.git
```

Automatic template routing is optional and deliberately excluded from the base
install; add it with the `routing` extra. See
[Getting started](https://geraldmc.github.io/bioinformatics_prompts/getting-started/).

## Example

```python
from bioinformatics_prompts import ClaudeInteraction

interaction = ClaudeInteraction()          # reads ANTHROPIC_API_KEY
interaction.load_template("Genomics")

print(interaction.ask_claude("How do I identify SNPs in my bacterial genome?"))
```

Or from the terminal, no code required:

```bash
uv run bioinformatics-prompts list-templates
uv run bioinformatics-prompts chat
```

## Documentation

| | |
|---|---|
| [Getting started](https://geraldmc.github.io/bioinformatics_prompts/getting-started/) | Install, configure a key, first question |
| [Guide](https://geraldmc.github.io/bioinformatics_prompts/guide/library/) | The library, CLI, errors, routing, custom templates |
| [Templates](https://geraldmc.github.io/bioinformatics_prompts/catalogue/) | All 14, browsable |
| [API reference](https://geraldmc.github.io/bioinformatics_prompts/reference/) | Every exported name, generated from the source |
| [Contributing](https://geraldmc.github.io/bioinformatics_prompts/contributing/) | Development setup, tests, CI |

## Development

```bash
git clone https://github.com/geraldmc/bioinformatics_prompts.git
cd bioinformatics_prompts
uv sync
uv run pytest
```

See [Contributing](https://geraldmc.github.io/bioinformatics_prompts/contributing/)
for the dependency groups, the docs build, and how to change a bundled template.

## License

MIT. See [LICENSE](LICENSE).
