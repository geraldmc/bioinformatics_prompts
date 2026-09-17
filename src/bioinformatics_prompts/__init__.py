import logging

# Installed before the submodule imports below, deliberately: anything a
# submodule logs while the package is still importing would otherwise find this
# logger handler-less and fall through to logging.lastResort, printing to
# stderr. Libraries configure no handlers of their own beyond a NullHandler;
# installing real ones is the prerogative of the importing application. See the
# Python logging HOWTO, "Configuring Logging for a Library".
logging.getLogger(__name__).addHandler(logging.NullHandler())

from bioinformatics_prompts.claude_interaction import ClaudeInteraction  # noqa: E402
from bioinformatics_prompts.prompt_template import (  # noqa: E402
    BioinformaticsPrompt,
)

__all__ = ["ClaudeInteraction", "BioinformaticsPrompt"]
