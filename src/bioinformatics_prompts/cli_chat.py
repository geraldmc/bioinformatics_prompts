"""Interactive terminal chat, and the numbered template picker it uses.

This is the terminal half of what used to live in `claude_interaction.py`. A
library must not own a REPL — an importing web app or notebook would inherit
stdout noise and a blocking `input()` — but a CLI legitimately does, so the
behaviour moved here rather than disappearing.

Everything here is built on the library's public surface:
`list_available_templates()`, `load_template()`, `ask_claude()`.
"""

from typing import Optional

import anthropic
import click

from bioinformatics_prompts.claude_interaction import ClaudeInteraction
from bioinformatics_prompts.exceptions import BioinformaticsPromptsError

EXIT_COMMANDS = ("quit", "exit", "bye")


def select_template(interaction: ClaudeInteraction) -> Optional[str]:
    """Present the numbered template menu and return the chosen research area.

    Returns None if the user quits. Raises TemplateNotFoundError if the prompt
    directory holds no templates — a configuration problem, not a choice.
    """
    templates = interaction.list_available_templates()

    click.echo("\nAvailable research areas (prompt templates):")
    for template in templates:
        click.echo(f"{template['id']}. {template['research_area']}")

    while True:
        choice = click.prompt(
            "\nSelect a template by number (or 'q' to quit)",
            default="",
            show_default=False,
        )

        if choice.strip().lower() == "q":
            return None

        try:
            choice_idx = int(choice)
        except ValueError:
            click.echo("Please enter a valid number")
            continue

        selected = next((t for t in templates if t["id"] == choice_idx), None)
        if selected is None:
            click.echo(
                f"Invalid selection. Please choose a number between 1 and {len(templates)}"
            )
            continue

        return selected["research_area"]


def _load_selected_template(interaction: ClaudeInteraction) -> bool:
    """Run the picker and load the choice. Returns False if the user quit."""
    research_area = select_template(interaction)
    if research_area is None:
        return False

    interaction.load_template(research_area)
    click.echo(f"Loaded template: {research_area}")
    return True


def run_chat(interaction: ClaudeInteraction, use_template: bool = True) -> None:
    """Start an interactive conversation with Claude in the terminal."""
    interaction.reset_conversation()

    if use_template and interaction.prompt_template is None:
        if not _load_selected_template(interaction):
            click.echo("No template selected. Exiting conversation.")
            return

    click.echo("\n=== Starting conversation with Claude ===")
    click.echo("Type 'quit', 'exit', or 'bye' to end the conversation")
    click.echo("Type 'reset' to clear the conversation history")
    click.echo("Type 'template' to load a different template")
    click.echo("================================================\n")

    while True:
        user_query = click.prompt("\nYou", default="", show_default=False)
        command = user_query.strip().lower()

        if command in EXIT_COMMANDS:
            click.echo("Ending conversation. Goodbye!")
            return

        if command == "reset":
            interaction.reset_conversation()
            click.echo("Conversation history has been reset.")
            continue

        if command == "template":
            _load_selected_template(interaction)
            continue

        # Errors are reported and the loop continues: a failed turn should not
        # end the session. This catch-and-report is correct *here* — it is what
        # made the same code wrong inside the library.
        try:
            response = interaction.ask_claude(
                user_query, use_template=use_template, use_history=True
            )
        except (BioinformaticsPromptsError, anthropic.AnthropicError) as e:
            click.echo(f"Error: {e}", err=True)
            continue

        click.echo(f"\nClaude: {response}")
