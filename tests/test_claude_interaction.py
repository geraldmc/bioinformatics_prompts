"""Tests for claude_interaction.ClaudeInteraction.

None of these tests call the real Claude API; network-facing calls are faked.
"""

import pytest

import bioinformatics_prompts.claude_interaction as claude_interaction_module
from bioinformatics_prompts.claude_interaction import FALLBACK_MODEL, ClaudeInteraction, main


class _FakeModel:
    def __init__(self, model_id):
        self.id = model_id


class _FakeModelsResource:
    def __init__(self, model_ids=None, error=None):
        self._model_ids = model_ids or []
        self._error = error
        self.calls = 0

    def list(self):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return [_FakeModel(model_id) for model_id in self._model_ids]


class _FakeMessage:
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]


class _FakeMessagesResource:
    def create(self, **kwargs):
        return _FakeMessage("fake response")


class _FakeClient:
    def __init__(self, model_ids=None, error=None):
        self.models = _FakeModelsResource(model_ids=model_ids, error=error)
        self.messages = _FakeMessagesResource()


@pytest.fixture
def prompt_dir(tmp_path, sample_prompt):
    (tmp_path / "genomics_prompt.json").write_text(sample_prompt.to_json())
    (tmp_path / "aardvark_prompt.json").write_text(sample_prompt.to_json())
    return tmp_path


def test_list_available_templates_sorted_alphabetically(prompt_dir):
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    templates = interaction.list_available_templates()

    assert [t["filename"].split("/")[-1] for t in templates] == [
        "aardvark_prompt.json",
        "genomics_prompt.json",
    ]


def test_load_prompt_template_noninteractive_uses_first_template(prompt_dir):
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir=str(prompt_dir))

    loaded = interaction.load_prompt_template(interactive=False)

    assert loaded is not None
    assert loaded.research_area == "Test Area"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    with pytest.raises(ValueError):
        ClaudeInteraction(api_key=None, prompt_dir="prompt")


def test_default_model_not_resolved_at_construction():
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir="prompt")

    assert interaction.default_model is None


def test_resolve_default_model_picks_most_recent_sonnet():
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir="prompt")
    client = _FakeClient(
        model_ids=["claude-opus-5", "claude-sonnet-5", "claude-sonnet-4-5", "claude-haiku-4-5"]
    )

    resolved = interaction._resolve_default_model(client)

    assert resolved == "claude-sonnet-5"


def test_resolve_default_model_falls_back_on_no_sonnet_match():
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir="prompt")
    client = _FakeClient(model_ids=["claude-opus-5", "claude-haiku-4-5"])

    resolved = interaction._resolve_default_model(client)

    assert resolved == FALLBACK_MODEL


def test_resolve_default_model_falls_back_on_exception():
    interaction = ClaudeInteraction(api_key="test-key", prompt_dir="prompt")
    client = _FakeClient(error=RuntimeError("network down"))

    resolved = interaction._resolve_default_model(client)

    assert resolved == FALLBACK_MODEL


def test_default_prompt_dir_is_cwd_independent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    interaction = ClaudeInteraction(api_key="test-key")
    templates = interaction.list_available_templates()

    assert len(templates) > 0


def test_main_loads_dotenv_and_starts_conversation(monkeypatch):
    calls = []
    monkeypatch.setattr(claude_interaction_module, "load_dotenv", lambda: calls.append("load_dotenv"))
    monkeypatch.setattr(
        ClaudeInteraction, "start_conversation", lambda self: calls.append("start_conversation")
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    main()

    assert calls == ["load_dotenv", "start_conversation"]


def test_send_to_claude_caches_resolved_model(monkeypatch):
    fake_client = _FakeClient(model_ids=["claude-opus-5", "claude-sonnet-5"])
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: fake_client)

    interaction = ClaudeInteraction(api_key="test-key", prompt_dir="prompt")

    interaction.send_to_claude("hello")
    interaction.send_to_claude("hello again")

    assert interaction.default_model == "claude-sonnet-5"
    assert fake_client.models.calls == 1
