"""Regression tests: the package must not touch the host application's logging.

The defect these guard against (#12) was `utils/logging.py` calling
`logger.remove()` at import time, destroying whatever logging configuration the
importing application had already installed.

Several checks run in a subprocess. That is deliberate: pytest installs its own
logging handlers, and module import state is session-global, so an in-process
assertion about `sys.modules` or handler lists would depend on test collection
order rather than on the package's behaviour.
"""

import logging
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    """Run `script` in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )


def _stdout_of(script: str) -> str:
    """Run `script` in a fresh interpreter and return its stripped stdout."""
    return _run_in_subprocess(script).stdout.strip()


def test_package_logger_has_only_a_null_handler():
    """The one handler a library is permitted to install, and nothing else."""
    output = _stdout_of(
        "import logging\n"
        "import bioinformatics_prompts\n"
        "handlers = logging.getLogger('bioinformatics_prompts').handlers\n"
        "print([type(h).__name__ for h in handlers])\n"
    )

    assert output == "['NullHandler']"


def test_importing_utils_leaves_host_root_logging_untouched():
    """A host's stdlib logging configuration survives importing the package.

    Note: this is a *forward* guard, not a regression guard. The original
    defect was loguru-side only, so this assertion also held before the fix.
    It exists to catch a future `basicConfig()` or `addHandler()` call sneaking
    into library import paths.
    """
    output = _stdout_of(
        "import logging, sys\n"
        "root = logging.getLogger()\n"
        "root.addHandler(logging.StreamHandler(sys.stderr))\n"
        "before = list(root.handlers)\n"
        "import bioinformatics_prompts.utils.validation\n"
        "print(before == list(root.handlers))\n"
    )

    assert output == "True"


#: Ways library code could seize control of logging configuration. The first two
#: are the original loguru defect; the rest are the stdlib forms a regression
#: would realistically take now that loguru is gone. Verified that none of these
#: appear inside the template modules' example-code string literals, so scanning
#: every module under src/ is safe.
FORBIDDEN_IN_LIBRARY_SOURCE = (
    "logger.remove()",
    "loguru",
    "logging.basicConfig(",
    ".handlers.clear()",
    "removeHandler(",
)


def test_library_source_never_seizes_logging_configuration():
    """No library module may install, remove, or clear logging handlers.

    Handler configuration is the importing application's prerogative; the sole
    exception is the NullHandler in the top-level `__init__.py`.
    """
    source_files = sorted(SRC_ROOT.rglob("*.py"))

    # Guard against the scan silently passing over an empty tree, e.g. if the
    # suite is ever run against an installed copy without the repo layout.
    assert SRC_ROOT.is_dir(), f"source tree not found at {SRC_ROOT}"
    assert len(source_files) > 20, f"only {len(source_files)} modules scanned"

    offenders = []

    for path in source_files:
        source = path.read_text()
        for needle in FORBIDDEN_IN_LIBRARY_SOURCE:
            if needle in source:
                offenders.append(f"{path.relative_to(SRC_ROOT)}: {needle}")

    assert offenders == []


def test_package_logger_is_silent_without_host_configuration():
    """With no handlers configured by the host, the package emits nothing.

    Without the NullHandler, stdlib logging falls back to `logging.lastResort`,
    which writes the record to stderr — exactly the unsolicited output a library
    must not produce. Asserting on stderr, not stdout, is what makes this a real
    check.
    """
    result = _run_in_subprocess(
        "import logging\n"
        "import bioinformatics_prompts\n"
        "logging.getLogger('bioinformatics_prompts.probe').warning('should not appear')\n"
    )

    # Asserting on the record's own text rather than on stderr being wholly
    # empty: the import pulls in anthropic/dspy/litellm, any of which may emit
    # an unrelated warning on a dependency bump.
    assert "should not appear" not in result.stderr


def test_null_handler_does_not_block_host_configuration():
    """A NullHandler must not suppress records once the host configures logging."""
    output = _stdout_of(
        "import logging\n"
        "import bioinformatics_prompts\n"
        "records = []\n"
        "handler = logging.Handler()\n"
        "handler.emit = records.append\n"
        "logging.getLogger('bioinformatics_prompts').addHandler(handler)\n"
        "logging.getLogger('bioinformatics_prompts.probe').warning('hello')\n"
        "print(len(records))\n"
    )

    assert output == "1"
