"""Tests for bioinformatics_prompts.cli.

None of these tests call the real Claude API or Claude Code CLI; network-facing
calls and interactive prompts are faked/monkeypatched.
"""

import pytest
from click.testing import CliRunner

import bioinformatics_prompts.cli as cli_module
from bioinformatics_prompts.cli import cli
from bioinformatics_prompts.claude_interaction import ClaudeInteraction


@pytest.fixture
def prompt_dir(tmp_path, sample_prompt):
    (tmp_path / "genomics_prompt.json").write_text(sample_prompt.to_json())
    (tmp_path / "aardvark_prompt.json").write_text(sample_prompt.to_json())
    return tmp_path


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def no_real_dotenv(monkeypatch):
    """Prevent a real local .env from leaking a key into these tests."""
    monkeypatch.setattr(cli_module, "load_dotenv", lambda: None)


def test_bare_invocation_falls_through_to_chat(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(ClaudeInteraction, "start_conversation", lambda self: calls.append("chat"))

    result = runner.invoke(cli, ["--api-key", "test-key"])

    assert result.exit_code == 0
    assert calls == ["chat"]


def test_explicit_chat_subcommand(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(ClaudeInteraction, "start_conversation", lambda self: calls.append("chat"))

    result = runner.invoke(cli, ["--api-key", "test-key", "chat"])

    assert result.exit_code == 0
    assert calls == ["chat"]


def test_chat_with_no_api_key_configured(runner, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    result = runner.invoke(cli, ["chat"])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "API key" in result.output


def test_route_with_no_api_key_configured(runner, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    result = runner.invoke(cli, ["route", "some query"])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "API key" in result.output


def test_list_templates_with_no_api_key_configured(runner, monkeypatch, prompt_dir):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    result = runner.invoke(cli, ["--prompt-dir", str(prompt_dir), "list-templates"])

    assert result.exit_code == 0


def test_list_templates_output_format(runner, prompt_dir):
    result = runner.invoke(cli, ["--api-key", "test-key", "--prompt-dir", str(prompt_dir), "list-templates"])

    assert result.exit_code == 0
    assert result.output == "1. Test Area\n2. Test Area\n"
    assert "description" not in result.output.lower()


def test_route_hit(runner, monkeypatch):
    monkeypatch.setattr(
        ClaudeInteraction, "route_template", lambda self, query: {"research_area": "Genomics"}
    )

    result = runner.invoke(cli, ["--api-key", "test-key", "route", "align some reads"])

    assert result.exit_code == 0
    assert "Genomics" in result.output


def test_route_miss(runner, monkeypatch):
    monkeypatch.setattr(ClaudeInteraction, "route_template", lambda self, query: None)

    result = runner.invoke(cli, ["--api-key", "test-key", "route", "align some reads"])

    assert result.exit_code == 0


def test_route_does_not_also_chat(runner, monkeypatch):
    monkeypatch.setattr(
        ClaudeInteraction, "route_template", lambda self, query: {"research_area": "Genomics"}
    )

    def _fail_if_called(self, *args, **kwargs):
        raise AssertionError("ask_claude should not be called by route")

    monkeypatch.setattr(ClaudeInteraction, "ask_claude", _fail_if_called)

    result = runner.invoke(cli, ["--api-key", "test-key", "route", "align some reads"])

    assert result.exit_code == 0


def test_global_options_passthrough(runner, prompt_dir):
    captured = {}
    real_init = ClaudeInteraction.__init__

    def _capture_init(self, api_key=None, prompt_dir=None, model=None, require_api_key=True):
        captured["api_key"] = api_key
        captured["prompt_dir"] = prompt_dir
        captured["model"] = model
        return real_init(
            self, api_key=api_key, prompt_dir=prompt_dir, model=model, require_api_key=require_api_key
        )

    import unittest.mock

    with unittest.mock.patch.object(ClaudeInteraction, "__init__", _capture_init):
        result = runner.invoke(
            cli,
            [
                "--api-key",
                "test-key",
                "--model",
                "claude-sonnet-5",
                "--prompt-dir",
                str(prompt_dir),
                "list-templates",
            ],
        )

    assert result.exit_code == 0
    assert captured == {
        "api_key": "test-key",
        "prompt_dir": str(prompt_dir),
        "model": "claude-sonnet-5",
    }


def test_load_dotenv_called_during_group_dispatch(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(cli_module, "load_dotenv", lambda: calls.append("load_dotenv"))
    monkeypatch.setattr(ClaudeInteraction, "start_conversation", lambda self: None)

    runner.invoke(cli, ["--api-key", "test-key", "chat"])

    assert calls == ["load_dotenv"]
