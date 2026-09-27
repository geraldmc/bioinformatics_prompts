# CLI

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

## Global options

Available before any subcommand, these map onto `ClaudeInteraction`'s
constructor arguments:

```bash
uv run bioinformatics-prompts --api-key YOUR_KEY --model claude-sonnet-5 --prompt-dir /path/to/templates chat
```

| Option | Meaning |
|---|---|
| `--api-key` | Claude API key. Defaults to the `CLAUDE_API_KEY` or `ANTHROPIC_API_KEY` environment variable (or a `.env` file). |
| `--model` | Default Claude model to use. See [Choosing a model](library.md#choosing-a-model). |
| `--prompt-dir` | Directory containing prompt template JSON files. |

`list-templates` only reads local template files, so it works without an API key
configured; `chat` and `route` require one.

## Interactive mode

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

!!! note

    `route` exercises [automatic routing](routing.md) directly. It is not yet
    wired into `chat`'s interactive loop, which still uses the numbered menu.
