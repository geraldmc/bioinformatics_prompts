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


# ---------------------------------------------------------------------------
# What the package exports
# ---------------------------------------------------------------------------

EXPECTED_EXPORTS = [
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


def test_all_is_exactly_the_declared_surface():
    """Pinned deliberately: adding a name here is an API decision, not a detail.

    FewShotExample is the load-bearing one. It is the element type of
    BioinformaticsPrompt's `examples` argument, so before this the exported
    class could not be constructed from the exported names alone.
    """
    import bioinformatics_prompts

    assert bioinformatics_prompts.__all__ == EXPECTED_EXPORTS


def test_every_exported_name_resolves():
    import bioinformatics_prompts

    missing = [
        name for name in bioinformatics_prompts.__all__
        if not hasattr(bioinformatics_prompts, name)
    ]
    assert missing == []


def test_star_import_binds_every_exported_name():
    namespace = {}
    exec("from bioinformatics_prompts import *", namespace)

    bound = {name for name in namespace if not name.startswith("__")}
    assert bound == set(EXPECTED_EXPORTS)


def test_terminal_ui_and_routing_internals_are_not_exported():
    """cli_chat owns the REPL #14 moved out of the library; match_area is
    routing internals that #11 split out only so it stayed importable without
    the extra. Neither is part of the surface a consumer programs against.
    """
    import bioinformatics_prompts

    for name in ("run_chat", "select_template", "match_area", "cli", "cli_chat"):
        assert name not in bioinformatics_prompts.__all__


def test_stdlib_logging_is_not_a_package_attribute():
    """__init__ imports logging for its NullHandler, which used to leave
    `bioinformatics_prompts.logging` resolving to the stdlib module — next to
    the real API in dir(), and a live trap because this package had a real
    utils/logging.py until #12 removed it.
    """
    import bioinformatics_prompts

    assert not hasattr(bioinformatics_prompts, "logging")


def test_exceptions_are_importable_without_the_routing_extra():
    """RoutingUnavailableError is exported from the top level even though it is
    about an optional extra: exceptions.py is stdlib-only by design (#14), so
    exporting it reaches no further than the rest of the package already does.
    """
    import bioinformatics_prompts

    assert issubclass(bioinformatics_prompts.RoutingUnavailableError, ImportError)
    assert issubclass(
        bioinformatics_prompts.RoutingUnavailableError,
        bioinformatics_prompts.BioinformaticsPromptsError,
    )
