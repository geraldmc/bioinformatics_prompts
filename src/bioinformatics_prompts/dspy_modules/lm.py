"""Helpers for configuring DSPy's LM backend to use Claude via litellm."""

import os
from typing import Optional

import dspy


def configure_claude_lm(model: str, api_key: Optional[str] = None) -> dspy.LM:
    """
    Configure DSPy's global LM to use Claude via litellm, and return it.

    Args:
        model: Claude model id (e.g. "claude-sonnet-4-6"). Required — callers
            that need a default should resolve one themselves (e.g.
            `self.default_model or FALLBACK_MODEL` in ClaudeInteraction)
            before calling this. Not imported here to avoid a circular
            import with claude_interaction.py.
        api_key: Anthropic API key. If None, resolved from the
            CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variables
            (same precedence as ClaudeInteraction.__init__).

    Returns: The configured dspy.LM instance.
    """
    key = api_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    lm = dspy.LM(f"anthropic/{model}", api_key=key)
    dspy.configure(lm=lm)
    return lm
