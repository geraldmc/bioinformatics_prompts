# Errors and logging

## Errors

The library **raises**; it never reports failure through its return value and
never prompts on stdin. Everything it raises descends from
`BioinformaticsPromptsError`:

| Exception | Raised when |
|---|---|
| `MissingAPIKeyError` | no API key passed or in the environment (also a `ValueError`) |
| `TemplateNotFoundError` | no template matches the requested name, or the directory is empty |
| `TemplateLoadError` | a template file was found but could not be read or parsed |
| `NoTemplateLoadedError` | an operation needing a template ran before one was loaded |
| `RoutingUnavailableError` | routing requested without the `routing` extra (also an `ImportError`) |

All six are exported from the top level, so
`from bioinformatics_prompts import TemplateNotFoundError` works; they also
remain importable from `bioinformatics_prompts.exceptions`.

Because they share a base class, one clause catches everything the package
raises:

```python
from bioinformatics_prompts import BioinformaticsPromptsError, ClaudeInteraction

try:
    interaction = ClaudeInteraction()
    interaction.load_template("Genomics")
except BioinformaticsPromptsError as exc:
    print(f"could not start: {exc}")
```

Errors from the Claude API propagate unchanged as `anthropic.AnthropicError`
subclasses, so you can catch the SDK's own typed hierarchy — `RateLimitError`,
`AuthenticationError` and the rest — rather than a flattened wrapper.

Full signatures: [Exceptions](../reference/exceptions.md).

## Logging

The package logs through the standard library's `logging` module, under the
`bioinformatics_prompts` logger. Following the guidance in the Python logging
HOWTO, it installs **only a `NullHandler`** and configures nothing else — no
handlers, no formatters, no levels. Importing it will never alter logging
configuration your application has already set up, and it produces no log
output until you configure a handler.

To see the package's log records, configure logging as you normally would:

```python
import logging

logging.basicConfig(level=logging.INFO)          # or dictConfig, or your own handlers
logging.getLogger("bioinformatics_prompts").setLevel(logging.DEBUG)  # optional
```
