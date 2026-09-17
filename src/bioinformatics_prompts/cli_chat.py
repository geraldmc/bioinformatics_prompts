"""Interactive terminal chat, and the numbered template picker it uses.

This is the terminal half of what used to live in `claude_interaction.py`. A
library must not own a REPL — an importing web app or notebook would inherit
stdout noise and a blocking `input()` — but a CLI legitimately does, so the
behaviour moved here rather than disappearing.

Everything here is built on the library's public surface:
`list_available_templates()`, `load_template()`, `ask_claude()`.
"""

from typing import Dict, Optional

import anthropic
import click

from bioinformatics_prompts.claude_interaction import ClaudeInteraction
from bioinformatics_prompts.exceptions import BioinformaticsPromptsError

EXIT_COMMANDS = ("quit", "exit", "bye")


def select_template(interaction: ClaudeInteraction) -> Optional[Dict[str, str]]:
    """Present the numbered template menu and return the chosen template dict.

    Returns the entry from list_available_templates() that the user picked, or
    None if they quit.

    It returns the whole dict rather than just the research area on purpose. A
    name has to be re-resolved by scanning the list, and that scan stops at the
    first match — so when two templates share a research_area (or one file's
    area equals another's stem, or several unreadable files both fall back to
    "Unknown"), the user's actual choice would be silently swapped for whichever
    came first.
    """
    templates = interaction.list_available_templates()

    click.echo("\nAvailable research areas (prompt templates):")
    for number, template in enumerate(templates, 1):
        click.echo(f"{number}. {template['research_area']}")

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

        if not 1 <= choice_idx <= len(templates):
            # The menu numbers the list it just printed, so the range is always
            # 1..len and always contiguous. That was not true while the number
            # came from the library: it was assigned before unreadable files
            # were discarded, so the menu could offer 1 and 3 and then reject 2.
            click.echo(
                f"Invalid selection. Please choose a number between 1 and {len(templates)}"
            )
            continue

        return templates[choice_idx - 1]


def _load_selected_template(interaction: ClaudeInteraction) -> bool:
    """Run the picker and load the choice. Returns False if the user quit."""
    selected = select_template(interaction)
    if selected is None:
        return False

    # Load the file the user actually picked, not a re-resolution of its name.
    interaction.load_template_file(selected)
    click.echo(f"Loaded template: {selected['research_area']}")
    return True


def run_chat(interaction: ClaudeInteraction, use_template: bool = True) -> None:
    """Start an interactive conversation with Claude in the terminal."""
    interaction.reset_conversation()

    if use_template and interaction.prompt_template is None:
        # Retry rather than eject: a malformed template file is a reason to pick
        # again, not to refuse to start.
        while interaction.prompt_template is None:
            try:
                if not _load_selected_template(interaction):
                    click.echo("No template selected. Exiting conversation.")
                    return
            except BioinformaticsPromptsError as e:
                click.echo(f"Error: {e}", err=True)

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
            # A malformed template file must not end the session and discard
            # the conversation so far — report it and let the user pick again.
            try:
                _load_selected_template(interaction)
            except BioinformaticsPromptsError as e:
                click.echo(f"Error: {e}", err=True)
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
