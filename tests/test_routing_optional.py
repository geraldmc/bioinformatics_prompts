"""Tests for DSPy being an optional extra rather than a hard dependency.

The package must be importable, and usable for everything except routing,
without `dspy` installed (#11). Two different techniques appear here:

- A **subprocess** for the import guard, because the pytest session itself
  imports dspy via tests/test_router.py, so an in-process `sys.modules` check
  would depend on collection order.
- A **blocked-import fixture** for the failure-mode tests, which simulates a
  missing extra in an environment where dspy *is* installed.
"""

import subprocess
import sys

import pytest

from bioinformatics_prompts.exceptions import RoutingUnavailableError


def _stdout_of(script: str) -> str:
    """Run `script` in a fresh interpreter and return its stripped stdout."""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def prompt_dir(tmp_path, sample_prompt):
    (tmp_path / "genomics_prompt.json").write_text(sample_prompt.to_json())
    return tmp_path


# ---------------------------------------------------------------------------
# The headline guard: importing the package must not import dspy
# ---------------------------------------------------------------------------


def test_importing_the_package_does_not_import_dspy():
    """The whole point of the extra: dspy stays out of the import graph."""
    output = _stdout_of(
        "import sys\n"
        "import bioinformatics_prompts\n"
        "print('dspy' in sys.modules)\n"
    )

    assert output == "False"


def test_importing_the_package_does_not_import_openai():
    """dspy -> litellm -> openai: a Claude library must not drag in the OpenAI SDK."""
    output = _stdout_of(
        "import sys\n"
        "import bioinformatics_prompts\n"
        "print('openai' in sys.modules)\n"
    )

    assert output == "False"


def test_deep_import_of_prompt_template_stays_light():
    """A consumer wanting only the dataclasses pays only for the dataclasses.

    The parent package's __init__ runs first on any submodule import, so this
    is the case that was previously indistinguishable from a full import.
    """
    output = _stdout_of(
        "import sys\n"
        "from bioinformatics_prompts.prompt.templates.prompt_template import BioinformaticsPrompt\n"
        "print('dspy' in sys.modules)\n"
    )

    assert output == "False"


# ---------------------------------------------------------------------------
# match_area must be reachable without the extra
# ---------------------------------------------------------------------------


def test_match_area_is_importable_without_dspy():
    """Pure dict logic, so it must not live behind a dspy import."""
    output = _stdout_of(
        "import sys\n"
        "from bioinformatics_prompts.matching import match_area\n"
        "areas = [{'research_area': 'Genomics'}]\n"
        "assert match_area('Genomics', areas) == areas[0]\n"
        "print('dspy' in sys.modules)\n"
    )

    assert output == "False"


# ---------------------------------------------------------------------------
# Failure mode when the extra is missing
# ---------------------------------------------------------------------------


@pytest.fixture
def dspy_unavailable(monkeypatch):
    """Simulate a core-only install in an environment where dspy is installed.

    Evicts dspy and the package's dspy-importing modules from sys.modules, then
    installs a meta_path finder that raises ModuleNotFoundError for dspy, so the
    deferred import inside route_template() fails exactly as it would without
    the extra.
    """

    class _BlockDspy:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "dspy" or fullname.startswith("dspy."):
                raise ModuleNotFoundError(f"No module named {fullname!r}")
            return None

    for name in list(sys.modules):
        if name.startswith("dspy") or name.startswith(
            "bioinformatics_prompts.dspy_modules"
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.setattr(sys, "meta_path", [_BlockDspy(), *sys.meta_path])


def test_route_template_raises_routing_unavailable(prompt_dir, dspy_unavailable):
    """A missing extra is a configuration error, not a routing miss.

    None already means "no template matched", so returning it here would
    disguise a broken install as an ordinary negative result.
    """
    from bioinformatics_prompts import ClaudeInteraction

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    with pytest.raises(RoutingUnavailableError) as excinfo:
        interaction.route_template("How do I assemble a bacterial genome?")

    message = str(excinfo.value)
    assert "routing" in message, "the error must name the extra"
    assert "pip install" in message or "uv add" in message, (
        "the error must carry an actionable install hint"
    )


def test_routing_unavailable_is_an_import_error():
    """Subclassing ImportError keeps `except ImportError` handlers working."""
    assert issubclass(RoutingUnavailableError, ImportError)


def test_load_template_by_query_propagates_the_error(prompt_dir, dspy_unavailable):
    """The convenience wrapper must not swallow it into its None return."""
    from bioinformatics_prompts import ClaudeInteraction

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    with pytest.raises(RoutingUnavailableError):
        interaction.load_template_by_query("How do I assemble a genome?")
