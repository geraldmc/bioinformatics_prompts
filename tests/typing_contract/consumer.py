"""A consumer of this package's documented API, for mypy to check.

Never executed — it is type-checked by tests/test_typing_contract.py and
nothing else. Every call here appears in the README, so a type error in this
file is a promise `py.typed` makes and the package does not keep.

Keep it to *documented* usage. This file is a contract, not a test of how
much of the package happens to be annotated.
"""

from bioinformatics_prompts import (
    BioinformaticsPrompt,
    ClaudeInteraction,
    FewShotExample,
    TemplateInfo,
    TemplateNotFoundError,
)
from bioinformatics_prompts.utils.validation import validate_prompt


def main() -> None:
    interaction = ClaudeInteraction(api_key="key")

    # README, "Programmatic Usage": entries are TemplateInfo, numbered by the caller.
    entry: TemplateInfo = interaction.list_available_templates()[0]
    print(entry["research_area"].upper(), entry["filename"].lower(), entry["description"])

    prompt: BioinformaticsPrompt = interaction.load_template_file(entry)

    try:
        prompt = interaction.load_template("Genomics")
    except TemplateNotFoundError as e:       # README, "Errors"
        print(str(e))

    # README, "Choosing a model": omitting the model means "resolve one for me",
    # and passing None explicitly must mean the same thing.
    interaction.send_to_claude("question", model=None)
    interaction.ask_claude("question", model=None)

    # README, "Creating a Custom Template".
    BioinformaticsPrompt(
        research_area="X",
        description="d",
        key_concepts=[],
        common_tools=[],
        common_file_formats=[],
        examples=[FewShotExample(query="q", context="c", response="r")],
    )

    # README, "Validating a Prompt Template".
    result = validate_prompt(prompt)
    if result["valid"]:
        for message in result["warnings"] + result["errors"]:
            print(message.upper())
