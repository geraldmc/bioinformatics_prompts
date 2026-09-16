"""Exceptions raised by this package.

Deliberately imports nothing beyond the standard library, so it stays importable
in a core-only install — without the `routing` extra, and without the API client.
That constraint is why these do not live under `dspy_modules/`.
"""


class BioinformaticsPromptsError(Exception):
    """Base class for every exception this package raises."""


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
