"""Tests for the package's declared public surface (#21).

Separate from test_library_contract.py, which covers how the library *behaves*
(raise, don't print-and-return). This file covers what the package *declares*:
what `__all__` names, what the annotations promise, and where things live.

The defect these guard against is a signature that is not true. Annotations are
inert until a PEP 561 marker ships, so a wrong one can sit unnoticed and then
start being believed the moment `py.typed` lands (#13).
"""

import typing

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
