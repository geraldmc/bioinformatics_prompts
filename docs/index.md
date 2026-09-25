# Bioinformatics Prompts

A Python package for generating and using bioinformatics-specific prompts with
Anthropic's Claude AI.

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

## Features

- Structured templates for different bioinformatics research areas
- Few-shot examples to guide LLM responses
- Validation utilities to ensure prompt quality
- Seamless integration with Anthropic's Claude API
- Interactive conversation mode
- Template selection interface
- JSON serialization for easy template sharing

## Where to go next

- **[Getting started](getting-started.md)** — install it, and send your first question.
- **[Using the library](guide/library.md)** — the client, the public API, model selection.
- **[CLI](guide/cli.md)** — three subcommands, no code required.
- **[Templates](catalogue/index.md)** — all 14 bundled templates, browsable.
- **[API reference](reference/index.md)** — every exported name, generated from the source.
- **[Contributing](contributing.md)** — development setup, tests and CI.

## License

MIT. See [LICENSE](https://github.com/geraldmc/bioinformatics_prompts/blob/main/LICENSE).

## Acknowledgements

- Partly inspired by the paper "Empowering beginners in bioinformatics with ChatGPT" by Evelyn Shue et al. (2023)
- Based on the OPTIMAL model (Optimization of Prompts Through Iterative Mentoring and Assessment with an LLM chatbot)
- Uses Anthropic's Claude API for advanced AI interactions
