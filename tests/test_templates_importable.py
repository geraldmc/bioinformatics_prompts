"""Regression test: every module under prompt.templates must be importable.

Guards against filenames like the historical `single-cell.py`, whose hyphen
made it invisible to `pkgutil` and unimportable via any dotted path.
"""

import importlib
import pkgutil

from bioinformatics_prompts.prompt import templates


def test_all_template_modules_are_importable():
    module_names = [info.name for info in pkgutil.iter_modules(templates.__path__)]

    assert "single_cell" in module_names

    for name in module_names:
        importlib.import_module(f"{templates.__name__}.{name}")
