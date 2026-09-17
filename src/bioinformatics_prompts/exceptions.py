"""Exceptions raised by this package.

Deliberately imports nothing beyond the standard library, so it stays importable
in a core-only install — without the `routing` extra, and without the API client.
That constraint is why these do not live under `dspy_modules/`.
"""


class BioinformaticsPromptsError(Exception):
    """Base class for every exception this package raises."""


class MissingAPIKeyError(BioinformaticsPromptsError, ValueError):
    """No Claude API key was passed or found in the environment.

    Also subclasses ValueError because that is what this condition raised
    before the exception hierarchy existed, and callers (including cli.py)
    catch it that way.
    """


class TemplateNotFoundError(BioinformaticsPromptsError):
    """No template matched the requested name, or the prompt directory is empty."""


class TemplateLoadError(BioinformaticsPromptsError):
    """A template file was located but could not be read or parsed."""


class NoTemplateLoadedError(BioinformaticsPromptsError):
    """An operation requiring a loaded template ran before one was loaded.

    Unlike MissingAPIKeyError this does *not* subclass ValueError: the
    ValueError it replaces was caught by nothing, so there is no contract to
    preserve.
    """


class RoutingUnavailableError(BioinformaticsPromptsError, ImportError):
    """Template routing was requested but the `routing` extra is not installed.

    Also subclasses ImportError so that callers with an existing
    `except ImportError` handler keep working.
    """

    def __init__(self, message: str | None = None):
        super().__init__(
            message
            or (
                "Template routing requires the 'routing' extra, which is not "
                "installed. Install it with:\n"
                "    uv add 'bioinformatics-prompts[routing]'\n"
                "or:\n"
                "    pip install 'bioinformatics-prompts[routing]'"
            )
        )
