"""Tests for what the built distribution declares about itself (#13).

Distinct from test_public_api.py, which covers the importable surface. These
cover packaging: the version, the metadata a consuming project reads, and the
coverage configuration that keeps the report honest.
"""

import subprocess
import sys
from importlib.metadata import version


def _in_fresh_interpreter(code: str) -> str:
    """Run code in a clean interpreter and return its stdout.

    Needed because pytest itself imports importlib.metadata to discover
    plugins, so sys.modules in-process can never show whether *this package*
    pulled it in.
    """
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_version_matches_installed_metadata():
    import bioinformatics_prompts

    assert bioinformatics_prompts.__version__ == version("bioinformatics-prompts")


def test_version_is_not_part_of_the_exported_surface():
    """#21 pinned __all__ to exactly ten names. A dunder must not disturb it."""
    import bioinformatics_prompts

    assert "__version__" not in bioinformatics_prompts.__all__

    namespace = {}
    exec("from bioinformatics_prompts import *", namespace)
    assert "__version__" not in namespace


def test_unknown_attribute_still_raises():
    """Resolving __version__ lazily must not swallow genuine typos."""
    import bioinformatics_prompts

    try:
        bioinformatics_prompts.no_such_attribute
    except AttributeError as e:
        assert "no_such_attribute" in str(e)
    else:
        raise AssertionError("an unknown attribute did not raise AttributeError")


def test_importing_the_package_does_not_pull_in_importlib_metadata():
    """The regression guard for how __version__ is resolved.

    Resolving it eagerly at module scope costs ~63 modules and roughly doubles
    import time — measured 102 -> 165 modules, 9.1ms -> 16.7ms on a core-only
    install. #11 cut this package's import graph from 1906 modules to ~104 and
    #21 held it there, so that cost is not worth paying on every import for an
    attribute almost nothing reads.
    """
    out = _in_fresh_interpreter(
        "import sys, bioinformatics_prompts;"
        "print('importlib.metadata' in sys.modules)"
    )
    assert out == "False", "importing the package eagerly loaded importlib.metadata"


def test_reading_version_loads_importlib_metadata_on_demand():
    """The other half: lazy must still actually work."""
    out = _in_fresh_interpreter(
        "import sys, bioinformatics_prompts;"
        "v = bioinformatics_prompts.__version__;"
        "print('importlib.metadata' in sys.modules, v)"
    )
    assert out.startswith("True "), out


def test_distribution_declares_its_license_and_project_urls():
    """A consumer installing from a git URL must get a distribution that
    states its own license, author and home."""
    from importlib.metadata import metadata

    meta = metadata("bioinformatics-prompts")

    assert meta["License-Expression"] == "MIT"
    assert meta["Author"] == "geraldmc"

    urls = {
        entry.split(",", 1)[0].strip(): entry.split(",", 1)[1].strip()
        for entry in meta.get_all("Project-URL") or []
    }
    assert set(urls) == {"Homepage", "Repository", "Issues"}
    assert all("github.com/geraldmc/bioinformatics_prompts" in u for u in urls.values())

    classifiers = meta.get_all("Classifier") or []
    assert "Topic :: Scientific/Engineering :: Bio-Informatics" in classifiers
    assert any(c.startswith("Development Status") for c in classifiers)
    assert not any("License ::" in c for c in classifiers), (
        "PEP 639 supersedes the license classifier; carrying both invites conflicts"
    )


def test_coverage_omits_template_data_by_pattern_not_by_name():
    """The hand-maintained list silently went stale whenever a template was
    added — the new module just started counting toward coverage with nobody
    noticing. A pattern cannot drift because it names no modules.

    Safe only while prompt/templates/ holds nothing but template data; that
    invariant is guarded by
    test_public_api.py::test_template_package_holds_only_template_data.
    """
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    omit = tomllib.loads(pyproject.read_text())["tool"]["coverage"]["run"]["omit"]

    named_modules = [
        entry
        for entry in omit
        if "prompt/templates/" in entry and not entry.endswith("*.py")
    ]
    assert named_modules == [], (
        f"template modules are enumerated by name and will go stale: {named_modules}"
    )
    assert any("prompt/templates" in entry for entry in omit), (
        "template data must still be excluded from the coverage report"
    )
