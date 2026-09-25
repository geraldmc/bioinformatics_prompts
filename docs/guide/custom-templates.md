# Creating a custom template

A template is a `BioinformaticsPrompt`: a research area's context, plus worked
examples that show Claude the shape of a good answer. The 14 bundled ones are
built exactly this way — see [Templates](../catalogue/index.md) for what they contain.

## Building one

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

## Validating it

`validate_prompt` checks for the things that quietly make a template weak — a
thin description, too few concepts or tools, no examples, or example responses
with no code block in them:

```python
from bioinformatics_prompts import validate_prompt

result = validate_prompt(custom_prompt)

if not result["valid"]:
    for error in result["errors"]:
        print(f"error: {error}")

for warning in result["warnings"]:
    print(f"warning: {warning}")
```

`result` is a [`ValidationResult`](../reference/templates.md#bioinformatics_prompts.utils.validation.ValidationResult) —
`valid`, `warnings` and `errors`. `valid` goes False only for errors; warnings
describe a template that will work but could be better.

It is worth running against the bundled templates too:

```python
from bioinformatics_prompts import ClaudeInteraction, validate_prompt

interaction = ClaudeInteraction(require_api_key=False)
template = interaction.load_template("Genomics")

print(validate_prompt(template))
# {'valid': True, 'warnings': ["FewShotExample 1 response doesn't contain code blocks", ...], 'errors': []}
```

## Using it

Put your JSON files in a directory of their own and point the loader at it —
this replaces the bundled 14 rather than adding to them:

```python
from bioinformatics_prompts import ClaudeInteraction

interaction = ClaudeInteraction(prompt_dir="/path/to/my/templates")
interaction.load_template("The bioinformatics research area of interest")
```

The CLI takes the same directory with `--prompt-dir`. Nothing about this
requires modifying or rebuilding the package.

Before sending anything, you can inspect exactly what Claude will receive:

```python
print(interaction.generate_prompt("How do I analyze RNA-seq data from a non-model organism?"))
```

!!! note "Changing a bundled template instead"

    The 14 shipped templates are authored as Python and the JSON is generated
    from it. That is a contributor workflow, not a consumer one — see
    [Contributing](../contributing.md#changing-a-bundled-template).
