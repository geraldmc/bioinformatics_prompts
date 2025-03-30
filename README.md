# Bioinformatics Prompts

A package for generating and using bioinformatics-specific prompts with Anthropic's Claude AI.

## Overview

This package provides a structured framework for creating, validating, and utilizing domain-specific prompts for bioinformatics research. It helps researchers generate more focused and effective interactions with Large Language Models like Claude.

The package includes:
- Template structures for different bioinformatics disciplines
- Few-shot examples to guide LLM responses
- Validation utilities to ensure prompt quality
- Integration with Anthropic's API for seamless interactions

## Project Structure

```
bioinformatics-prompts/
├── prompt/                   # Prompt implementations
│   ├── __init__.py
│   └── artificial_intelligence_prompt.json
│   └── bioinformatics_tools_prompt.json
│   └── blockchain_bioinformatics_prompt.json
│   └── data_standardization_prompt.json
│   └── epigenomics_prompt.json
│   └── genomics_prompt.json
│   └── GWAS_prompt.json
│   └── metagenomics_prompt.json
│   └── ngs_sequencing_prompt.json
│   └── precision_medicine_prompt.json
│   └── sequence_analysis_prompt.json
│   └── single_cell_genomics_prompt.json
│   └── sythetic_biology_prompt.json
│   └── workflow_automation_prompt.json
├── templates/                # Base template classes
│   ├── __init__.py
│   └── prompt_template.py
├── tests/                    # Test modules
│   └── __init__.py
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── formatting.py
│   ├── logging.py
│   └── validation.py
├── claude_interaction.py     # Claude API integration
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bioinformatics-prompts.git
cd bioinformatics-prompts

# Install dependencies using uv
uv pip install -r requirements.txt

# Or just...
pip install -r requirements.txt
```

## Usage

### Creating a Prompt Template

```python
from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a genomics prompt template
genomics_prompt = BioinformaticsPrompt(
    discipline="Genomics",
    description="Description of genomics...",
    key_concepts=["Concept 1", "Concept 2"],
    common_tools=["Tool 1", "Tool 2"],
    common_file_formats=[
        {"name": "FASTQ", "description": "Raw sequencing reads with quality scores"}
    ],
    examples=[
        FewShotExample(
            query="How do I identify SNPs?",
            context="User has Illumina paired-end reads...",
            response="Detailed response with code examples..."
        )
    ],
    references=["Reference 1", "Reference 2"]
)

# Save the template
with open("my_prompt.json", "w") as f:
    f.write(genomics_prompt.to_json())
```

### Validating a Prompt

```python
from utils.validation import validate_prompt

validation_result = validate_prompt(genomics_prompt)
print(validation_result)
```

### Using Prompts with Claude API

```python
import os
from claude_interaction import ClaudeInteraction

# Set up Claude interaction
api_key = os.environ.get("ANTHROPIC_API_KEY")
claude = ClaudeInteraction(api_key)

# Load a template
genomics_template = claude.load_prompt_template("prompt/genomics_prompt.json")

# Ask a bioinformatics question
question = "How do I assemble a bacterial genome from Illumina paired-end reads?"
response = claude.ask_question(question, genomics_template)
print(response)

# Ask a follow-up question
follow_up = "What quality control steps should I include?"
response = claude.ask_question(follow_up)
print(response)
```

## Pre-built Templates

The package includes pre-built templates for various bioinformatics disciplines:

- **Genomics**: DNA sequencing, assembly, variant calling, etc.
- **Proteomics**: Protein identification, quantification, and analysis
- **Epigenomics**: DNA methylation, histone modifications, etc.
- **Metabolomics**: Metabolite identification and quantification
- **Metagenomics**: Analysis of microbial communities
- **Sequence Analysis**: Alignments, motif finding, etc.

## Creating Custom Templates

To create a custom template:

1. Define your discipline-specific details (tools, concepts, file formats)
2. Create few-shot examples with representative queries and responses
3. Use the `BioinformaticsPrompt` class to structure your template
4. Validate your template using the validation utilities
5. Export your template as JSON for reuse

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Partly inspired by the paper "Empowering beginners in bioinformatics with ChatGPT" by Evelyn Shue et al.
- Based on the OPTIMAL model (Optimization of Prompts Through Iterative Mentoring and Assessment with an LLM chatbot)
- Currently uses Anthropic's Claude API for advanced AI interactions. Other models planned for the future. 