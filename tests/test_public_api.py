"""Tests for the package's declared public surface (#21).

Separate from test_library_contract.py, which covers how the library *behaves*
(raise, don't print-and-return). This file covers what the package *declares*:
what `__all__` names, what the annotations promise, and where things live.

The defect these guard against is a signature that is not true. Annotations are
inert until a PEP 561 marker ships, so a wrong one can sit unnoticed and then
start being believed the moment `py.typed` lands (#13).
"""

import importlib
import pkgutil
import typing

import pytest

from bioinformatics_prompts import ClaudeInteraction
from bioinformatics_prompts.claude_interaction import TemplateInfo


def test_template_entry_matches_its_declared_type(tmp_path, sample_prompt):
    """The entries really are TemplateInfo: same keys, same value types.

    The defect: list_available_templates() was annotated List[Dict[str, str]]
    while "id" held an int, so a checker approved `t["id"].upper()` and the
    call raised AttributeError at runtime.
    """
    (tmp_path / "genomics_prompt.json").write_text(sample_prompt.to_json())

    entry = ClaudeInteraction(
        api_key="test-key", prompt_dir=str(tmp_path)
    ).list_available_templates()[0]

    declared = typing.get_type_hints(TemplateInfo)

    assert set(entry) == set(declared), "runtime keys differ from the declared type"
    for key, expected_type in declared.items():
        assert isinstance(entry[key], expected_type), (
            f"{key!r} is {type(entry[key]).__name__}, declared as {expected_type.__name__}"
        )


def test_template_entry_is_a_plain_dict(tmp_path, sample_prompt):
    """TemplateInfo is a typing construct, not a runtime wrapper.

    Consumers subscript these entries and so do cli.py and cli_chat.py, so the
    typing change must not have made them something else.
    """
    (tmp_path / "genomics_prompt.json").write_text(sample_prompt.to_json())

    entry = ClaudeInteraction(
        api_key="test-key", prompt_dir=str(tmp_path)
    ).list_available_templates()[0]

    assert type(entry) is dict


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------


def test_data_model_is_not_in_the_template_data_package():
    """The model moved to bioinformatics_prompts.prompt_template.

    It used to sit in prompt/templates/ beside the 14 template *content*
    modules, so reaching FewShotExample — the element type of
    BioinformaticsPrompt's own `examples` argument — took five path segments
    through a directory of data.
    """
    importlib.import_module("bioinformatics_prompts.prompt_template")

    # Assembled rather than written out, so a future search-and-replace over
    # the old dotted path cannot quietly turn this into the new one.
    old_path = ".".join(
        ["bioinformatics_prompts", "prompt", "templates", "prompt_template"]
    )
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old_path)


def test_template_package_holds_only_template_data():
    """Every module under prompt/templates/ defines a template, nothing else.

    This is what lets that directory be excluded from coverage wholesale (#13)
    rather than by a hand-maintained list that goes stale whenever a template
    is added. The previous exception was prompt_template.py, which was real
    code at 98% coverage; a glob would have silently discarded that signal.
    """
    from bioinformatics_prompts.prompt import templates

    for info in pkgutil.iter_modules(templates.__path__):
        module = importlib.import_module(f"{templates.__name__}.{info.name}")
        defined = [
            name
            for name, value in vars(module).items()
            if not name.startswith("_") and name.endswith("_prompt")
        ]
        assert defined, f"{info.name} defines no *_prompt object; is it really data?"
