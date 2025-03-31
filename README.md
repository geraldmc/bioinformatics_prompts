# Bioinformatics Prompts

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
├── prompt/                   # Prompt implementations
│   ├── __init__.py
│   ├── templates/            # Template definitions
│   │   ├── __init__.py
│   │   ├── ai.py
│   │   ├── bioinformatics_tools.py
│   │   ├── blockchain_bioinformatics.py
│   │   ├── data_standardization.py
│   │   ├── epigenomics.py
│   │   ├── genomics.py
│   │   ├── gwas.py
│   │   ├── metagenomics.py
│   │   ├── ngs_sequencing.py
│   │   ├── precision_medicine.py
│   │   ├── prompt_template.py  # Base template class
│   │   ├── sequence_analysis.py
│   │   ├── single-cell.py
│   │   ├── synthetic_biology.py
│   │   └── workflow_automation.py
│   │
│   ├── artificial_intelligence_prompt.json
│   ├── bioinformatics_tools_prompt.json
│   ├── blockchain_bioinformatics_prompt.json
│   ├── data_standardization_prompt.json
│   ├── epigenomics_prompt.json
│   ├── genomics_prompt.json
│   ├── gwas_prompt.json
│   ├── metagenomics_prompt.json
│   ├── ngs_sequencing_prompt.json
│   ├── precision_medicine_prompt.json
│   ├── sequence_analysis_prompt.json
│   ├── single_cell_genomics_prompt.json
│   ├── synthetic_biology_prompt.json
│   └── workflow_automation_prompt.json
│
├── tests/                    # Test modules (placeholder directory)
│   └── __init__.py
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── formatting.py
│   ├── logging.py
│   └── validation.py
│
├── claude_interaction.py     # Claude API integration
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/geraldmc/bioinformatics-prompts.git
cd bioinformatics-prompts

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage with Interactive Mode

The simplest way to use this package is through the interactive conversation mode:

```python
from claude_interaction import ClaudeInteraction

# Initialize (will use ANTHROPIC_API_KEY from environment variables)
interaction = ClaudeInteraction()

# Start interactive conversation with template selection
interaction.start_conversation()
```

This will start a terminal-based conversation where you may:
- Select a bioinformatics topic template
- Ask questions within that domain
- Get contextually-aware responses from Claude
- Switch templates or reset the conversation as needed

### Programmatic Usage

```python
from claude_interaction import ClaudeInteraction

# Initialize with API key
api_key = "your_anthropic_api_key"  # or set as environment variable
interaction = ClaudeInteraction(api_key=api_key)

# Load a specific template
interaction.load_prompt_template(interactive=False)  # Will load first available template

# Or list and select from available templates
templates = interaction.list_available_templates()
for t in templates:
    print(f"{t['id']}. {t['research_area']}")

# Ask a question using the loaded template
response = interaction.ask_claude(
    "How do I identify SNPs in my bacterial genome?",
    show_prompt=True  # Show the generated prompt for debugging
)
print(response)
```

### Creating a Custom Template

```python
from prompt.templates.prompt_template import BioinformaticsPrompt, FewShotExample

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
with open("prompt/custom_prompt.json", "w") as f:
    f.write(custom_prompt.to_json())
```

### Validating a Prompt Template

```python
from utils.validation import validate_prompt

# Validate your template
validation_result = validate_prompt(custom_prompt)
print(validation_result)
```

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