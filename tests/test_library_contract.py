"""Tests for the library contract: raise, don't print-and-return (#14).

Two defects are guarded here:

1. Failures were *returned* as strings (`f"Error: {e}"`) or as `None`, so a
   consumer could only detect them by sniffing the return value — and a real
   Claude response about error handling can begin "Error: " too.
2. The library owned a terminal UI, including a REPL and a blocking `input()`
   that `ask_claude` would trigger on a fresh instance.
"""

from pathlib import Path

import pytest

from bioinformatics_prompts import ClaudeInteraction
from bioinformatics_prompts.exceptions import (
    BioinformaticsPromptsError,
    MissingAPIKeyError,
    NoTemplateLoadedError,
    TemplateLoadError,
    TemplateNotFoundError,
)

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture
def prompt_dir(tmp_path, sample_prompt):
    (tmp_path / "genomics_prompt.json").write_text(sample_prompt.to_json())
    return tmp_path


class _BoomClient:
    """An anthropic client whose request raises, as a real API failure would."""

    class _Messages:
        def create(self, **kwargs):
            raise RuntimeError("upstream exploded")

    def __init__(self):
        self.messages = self._Messages()
        self.models = self

    def list(self):
        return []


# ---------------------------------------------------------------------------
# Failures are raised, not returned
# ---------------------------------------------------------------------------


def test_send_to_claude_propagates_api_errors(monkeypatch):
    """The defect: this used to return "Error: upstream exploded" as if it were
    Claude's answer. Asserting with pytest.raises rather than a prefix check,
    since prefix-sniffing is the anti-pattern being removed."""
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: _BoomClient())

    interaction = ClaudeInteraction(api_key="test-key", model="claude-sonnet-5")

    with pytest.raises(RuntimeError, match="upstream exploded"):
        interaction.send_to_claude("hello")


def test_send_to_claude_never_returns_an_error_string(monkeypatch):
    """A returned value is always Claude's text, never a smuggled error."""
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: _BoomClient())

    interaction = ClaudeInteraction(api_key="test-key", model="claude-sonnet-5")

    try:
        result = interaction.send_to_claude("hello")
    except RuntimeError:
        return  # raising is the correct behaviour
    pytest.fail(f"returned {result!r} instead of raising")


def test_ask_claude_raises_when_no_template_loaded(prompt_dir):
    """Must not fall back to the interactive picker and block on stdin.

    If this regresses, the test hangs rather than failing — which is exactly
    what a consuming web app would experience.
    """
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    with pytest.raises(NoTemplateLoadedError):
        interaction.ask_claude("How do I assemble a genome?")


def test_generate_prompt_raises_when_no_template_loaded(prompt_dir):
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    with pytest.raises(NoTemplateLoadedError):
        interaction.generate_prompt("How do I assemble a genome?")


def test_route_template_raises_on_empty_prompt_dir(tmp_path):
    """An empty directory is a configuration error, not a routing miss."""
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(tmp_path))

    with pytest.raises(TemplateNotFoundError):
        interaction.route_template("anything")


# ---------------------------------------------------------------------------
# load_template: explicit, exact, and it raises
# ---------------------------------------------------------------------------


def test_load_template_by_research_area(prompt_dir):
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    loaded = interaction.load_template("Test Area")

    assert loaded.research_area == "Test Area"
    assert interaction.prompt_template is loaded


def test_load_template_by_filename_stem(prompt_dir):
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    assert interaction.load_template("genomics_prompt").research_area == "Test Area"


def test_load_template_is_exact_not_substring(prompt_dir):
    """A partial name must fail loudly rather than guess.

    Substring matching is right for routing, where the input is fuzzy LLM
    output. It is wrong here: it would make adding a template change what an
    existing call returns.
    """
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    with pytest.raises(TemplateNotFoundError):
        interaction.load_template("Test")


def test_load_template_error_lists_the_valid_names(prompt_dir):
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    with pytest.raises(TemplateNotFoundError, match="Test Area"):
        interaction.load_template("nonexistent")


def test_load_template_raises_on_corrupt_file(tmp_path):
    """A findable-but-unparseable template is a load failure, not 'not found'."""
    (tmp_path / "broken_prompt.json").write_text("{not valid json")

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(tmp_path))

    with pytest.raises((TemplateLoadError, TemplateNotFoundError)):
        interaction.load_template("broken_prompt")


def test_old_load_prompt_template_is_gone(prompt_dir):
    """The blocking, interactive-by-default entry point no longer exists."""
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    assert not hasattr(interaction, "load_prompt_template")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_every_package_error_shares_one_base():
    for exc in (
        MissingAPIKeyError,
        TemplateNotFoundError,
        TemplateLoadError,
        NoTemplateLoadedError,
    ):
        assert issubclass(exc, BioinformaticsPromptsError)


def test_missing_api_key_error_is_also_a_value_error():
    """Preserves the documented constructor contract and cli.py's handler."""
    assert issubclass(MissingAPIKeyError, ValueError)


def test_no_template_loaded_error_is_not_a_value_error():
    """Nothing catches the ValueError it replaces, so it gets a clean type."""
    assert not issubclass(NoTemplateLoadedError, ValueError)


# ---------------------------------------------------------------------------
# The library owns no terminal UI
# ---------------------------------------------------------------------------


def test_library_modules_have_no_interactive_io():
    """No input() or print() on any library code path.

    Three deliberate exclusions:
      - `cli*.py` — a CLI is exactly where terminal I/O belongs.
      - `prompt/templates/*.py` — these carry `print(` inside example-code
        string literals, not as executable calls.
      - anything under `if __name__ == "__main__":` — a module's demo entry
        point is a script, and never runs on import.
    """
    offenders = []

    for path in sorted(SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(SRC_ROOT)
        if "prompt/templates" in relative.as_posix() or relative.name.startswith("cli"):
            continue

        source = path.read_text()
        # Scan only what executes on import.
        importable = source.split('if __name__ == "__main__":')[0]
        for needle in ("input(", "print("):
            if needle in importable:
                offenders.append(f"{relative}: {needle}")

    assert offenders == []


def test_start_conversation_is_no_longer_on_the_library_class():
    assert not hasattr(ClaudeInteraction, "start_conversation")


# ---------------------------------------------------------------------------
# Regressions found reviewing #20
# ---------------------------------------------------------------------------


def test_picker_loads_the_exact_file_selected(tmp_path, monkeypatch, sample_prompt):
    """Two templates can share a research_area, so selecting by name is lossy.

    The picker used to return a research_area that load_template() then
    re-resolved by scanning from the top, silently loading the first match
    instead of the entry the user chose.

    This drives the real picker with a simulated keystroke. Asserting against
    load_template_file() directly would pass even with the bug present, since
    the defect lives in how the picker hands its choice on.
    """
    from bioinformatics_prompts import cli_chat

    for stem, marker in (("aardvark_prompt", "FIRST"), ("zebra_prompt", "SECOND")):
        sample_prompt.key_concepts = [marker, "B", "C"]
        (tmp_path / f"{stem}.json").write_text(sample_prompt.to_json())

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(tmp_path))
    templates = interaction.list_available_templates()
    assert len({t["research_area"] for t in templates}) == 1, "fixture needs a shared name"

    # The user types "2" — the second entry, zebra.
    monkeypatch.setattr(cli_chat.click, "prompt", lambda *a, **kw: "2")

    assert cli_chat._load_selected_template(interaction) is True

    assert interaction.prompt_template.key_concepts[0] == "SECOND", (
        "picker loaded a different file than the one selected"
    )


def test_route_miss_is_reported_not_silent(tmp_path, monkeypatch, sample_prompt):
    """A routing miss must be visible: exit 0 with no output reads as success."""
    from click.testing import CliRunner

    import bioinformatics_prompts.cli as cli_module
    from bioinformatics_prompts.cli import cli

    monkeypatch.setattr(cli_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(ClaudeInteraction, "route_template", lambda self, query: None)

    result = CliRunner().invoke(cli, ["--api-key", "test-key", "route", "unroutable"])

    assert result.exit_code == 0
    assert result.output.strip(), "a miss produced no output at all"
    assert "No matching template" in result.output


def test_bad_template_pick_does_not_end_the_chat_session(tmp_path, monkeypatch, sample_prompt):
    """A malformed template must not destroy an in-progress conversation.

    Uses a file that parses as JSON but fails the model schema, so it survives
    list_available_templates()' JSONDecodeError filter and actually reaches
    load_template_file() — the path that raises TemplateLoadError.
    """
    from bioinformatics_prompts import cli_chat

    (tmp_path / "good_prompt.json").write_text(sample_prompt.to_json())
    (tmp_path / "broken_prompt.json").write_text('{"research_area": "Broken Area"}')

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(tmp_path))
    templates = interaction.list_available_templates()
    broken = next(t for t in templates if t["research_area"] == "Broken Area")

    # The bad file is listed, so the picker can reach it.
    assert broken is not None
    with pytest.raises(TemplateLoadError):
        interaction.load_template_file(broken)

    # Drive the REPL: load the good template, then pick the broken one, then quit.
    good_id = next(t["id"] for t in templates if t["research_area"] == "Test Area")
    replies = iter([str(good_id), "template", str(broken["id"]), "quit"])
    monkeypatch.setattr(cli_chat.click, "prompt", lambda *a, **kw: next(replies))

    cli_chat.run_chat(interaction)  # must return normally, not raise

    # The good template is still loaded and history survived the bad pick.
    assert interaction.prompt_template.research_area == "Test Area"
