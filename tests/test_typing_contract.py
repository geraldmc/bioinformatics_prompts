"""Guards the promise `py.typed` makes (#13).

Shipping a PEP 561 marker tells every downstream type checker to believe this
package's annotations. That is only worth doing if they are true, so this runs
mypy over a consumer written from the README and requires a clean result.

Deliberately narrow: it checks the *consumer's* view, not the package's own
internals, which still carry annotation gaps. Making those clean is a
type-checker-adoption question, tracked separately (#19).
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

CONSUMER = Path(__file__).parent / "typing_contract" / "consumer.py"


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None,
    # find_spec, not shutil.which: the venv's bin/ is not necessarily on PATH
    # when pytest is invoked as `.venv/bin/python -m pytest`, so a PATH probe
    # silently skips this test in exactly the setup CI uses.
    reason="mypy is not installed",
)
def test_documented_usage_type_checks_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--ignore-missing-imports", str(CONSUMER)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "py.typed promises these annotations are usable; mypy disagrees:\n"
        + result.stdout
        + result.stderr
    )
