import click
from dotenv import load_dotenv

from bioinformatics_prompts.claude_interaction import ClaudeInteraction


def _build_interaction(ctx, *, require_api_key=True):
    try:
        return ClaudeInteraction(
            api_key=ctx.obj["api_key"],
            prompt_dir=ctx.obj["prompt_dir"],
            model=ctx.obj["model"],
            require_api_key=require_api_key,
        )
    except ValueError as e:
        raise click.ClickException(
            f"{e} Set ANTHROPIC_API_KEY or CLAUDE_API_KEY, or pass --api-key."
        ) from e


@click.group(invoke_without_command=True)
@click.option("--api-key", default=None, help="Claude API key. Defaults to CLAUDE_API_KEY/ANTHROPIC_API_KEY.")
@click.option("--model", default=None, help="Default Claude model to use.")
@click.option("--prompt-dir", default=None, help="Directory containing prompt template JSON files.")
@click.pass_context
def cli(ctx, api_key, model, prompt_dir):
    """Generate and use bioinformatics-specific prompts with Claude."""
    load_dotenv()
    ctx.obj = {"api_key": api_key, "model": model, "prompt_dir": prompt_dir}
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@cli.command()
@click.pass_context
def chat(ctx):
    """Start an interactive conversation with Claude."""
    _build_interaction(ctx).start_conversation()


@cli.command(name="list-templates")
@click.pass_context
def list_templates(ctx):
    """List available prompt templates."""
    interaction = _build_interaction(ctx, require_api_key=False)
    for template in interaction.list_available_templates():
        click.echo(f"{template['id']}. {template['research_area']}")


@cli.command()
@click.argument("query")
@click.pass_context
def route(ctx, query):
    """Route a query to the best-matching prompt template."""
    interaction = _build_interaction(ctx)
    matched = interaction.route_template(query)
    if matched:
        click.echo(f"Matched template: {matched['research_area']}")
