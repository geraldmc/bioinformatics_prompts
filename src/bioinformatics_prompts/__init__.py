import logging

# Installed before the submodule imports below, deliberately: anything a
# submodule logs while the package is still importing would otherwise find this
# logger handler-less and fall through to logging.lastResort, printing to
# stderr. Libraries configure no handlers of their own beyond a NullHandler;
# installing real ones is the prerogative of the importing application. See the
# Python logging HOWTO, "Configuring Logging for a Library".
logging.getLogger(__name__).addHandler(logging.NullHandler())

# The stdlib module is not part of this package's namespace. Deleted rather
# than aliased so `bioinformatics_prompts.logging` is an AttributeError instead
# of a second name for the stdlib module — this package had a real
# utils/logging.py until #12 removed it, so the name is a live trap.
del logging

from bioinformatics_prompts.claude_interaction import (  # noqa: E402
    ClaudeInteraction,
    TemplateInfo,
)
from bioinformatics_prompts.exceptions import (  # noqa: E402
    BioinformaticsPromptsError,
    MissingAPIKeyError,
    NoTemplateLoadedError,
    RoutingUnavailableError,
    TemplateLoadError,
    TemplateNotFoundError,
)
from bioinformatics_prompts.prompt_template import (  # noqa: E402
    BioinformaticsPrompt,
    FewShotExample,
)

# The exceptions are exported alongside the types on purpose. exceptions.py
# imports nothing beyond the stdlib (#14) and claude_interaction already pulls
# five of the six in, so naming them here reaches no further than importing the
# package already does — including RoutingUnavailableError, which describes the
# optional `routing` extra but never touches dspy (#11).
__all__ = [
    "BioinformaticsPrompt",
    "BioinformaticsPromptsError",
    "ClaudeInteraction",
    "FewShotExample",
    "MissingAPIKeyError",
    "NoTemplateLoadedError",
    "RoutingUnavailableError",
    "TemplateInfo",
    "TemplateLoadError",
    "TemplateNotFoundError",
]
