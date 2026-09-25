# Automatic routing

Instead of naming a template explicitly with `load_template()`, you can route a
user's query to the best-matching template using a small DSPy-based router.

## The `routing` extra

Automatic template routing is **optional**, because it depends on DSPy, which
pulls in `litellm` and through it the OpenAI SDK. The base install deliberately
skips all of that — roughly 20 packages instead of 70, and an import that costs
milliseconds rather than half a second.

```bash
# Base install: everything except automatic routing
uv add git+https://github.com/geraldmc/bioinformatics_prompts.git

# With routing
uv add "bioinformatics-prompts[routing] @ git+https://github.com/geraldmc/bioinformatics_prompts.git"
```

Without the extra, every feature except `route_template()` and
`load_template_by_query()` works normally. Those two raise
`RoutingUnavailableError` (a subclass of `ImportError`) with an install hint:

```python
from bioinformatics_prompts import RoutingUnavailableError

try:
    matched = interaction.route_template("How do I call variants?")
except RoutingUnavailableError:
    matched = None  # fall back to explicit template selection
```

CI runs a dedicated `core-only` job that installs without the extra and asserts
that importing the package pulls in neither `dspy` nor the OpenAI SDK, so this
separation cannot quietly erode.

## Routing a query

```python
interaction = ClaudeInteraction(api_key=api_key)

# Picks a template automatically based on the query text, or returns None
# if no good match is found (fall back to load_template(name) in that case).
loaded = interaction.load_template_by_query(
    "How do I call variants from bacterial WGS reads?"
)
```

This uses `dspy.LM("anthropic/<model>", ...)` under the hood (see
`dspy_modules/lm.py` and `dspy_modules/router.py`), resolving the model the same
way as `send_to_claude` (`self.default_model` if set, else `FALLBACK_MODEL`).

The `bioinformatics-prompts route` subcommand exercises this directly (see the
[CLI guide](cli.md)); it is not yet wired into the `chat` subcommand's
interactive loop, which still uses the numbered menu.
