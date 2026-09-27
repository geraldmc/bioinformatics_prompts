"""Bioinformatics-specific prompt templates for Anthropic's Claude.

The package bundles 14 research-area templates -- genomics, single-cell, GWAS
and so on -- each carrying key concepts, common tools, file formats and
few-shot examples. `ClaudeInteraction` loads one and wraps a user's question in
it before sending it to Claude.

Everything this package promises is named in `__all__` below and documented in
the API reference. Anything else may move without notice.

    >>> from bioinformatics_prompts import ClaudeInteraction
    >>> client = ClaudeInteraction(require_api_key=False)
    >>> len(client.list_available_templates())
    14
"""

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
from bioinformatics_prompts.utils.validation import (  # noqa: E402
    ValidationResult,
    validate_prompt,
)

# The exceptions are exported alongside the types on purpose. exceptions.py
# imports nothing beyond the stdlib (#14) and claude_interaction already pulls
# five of the six in, so naming them here reaches no further than importing the
# package already does — including RoutingUnavailableError, which describes the
# optional `routing` extra but never touches dspy (#11).
#
# validate_prompt and ValidationResult joined in #24. utils/validation.py
# imports only the stdlib and prompt_template, which claude_interaction already
# pulls in, so this costs the import graph two first-party modules and no new
# dependency at all -- measured, not assumed.
# Its three siblings (batch_validate_prompts, run_test_query,
# export_all_prompts) are deliberately not exported: nothing documents them.
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
    "ValidationResult",
    "validate_prompt",
]


def __getattr__(name: str) -> str:
    """Resolve `__version__` lazily, per PEP 562.

    Reading it from installed metadata keeps pyproject.toml the single source
    of truth, but `importlib.metadata` is expensive: importing it costs ~63
    modules and roughly doubles this package's import time (measured 102 -> 165
    modules, 9.1ms -> 16.7ms on a core-only install). #11 cut the import graph
    from 1906 modules to ~104 and #21 held it there, so that is not a cost to
    pay on every import for an attribute almost nothing reads. Deferring it
    here means only a caller that actually asks pays.
    """
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("bioinformatics-prompts")
        except PackageNotFoundError:      # running from an uninstalled source tree
            return "0.0.0.dev0"

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
