"""Every place that states a version must state the same one.

There are five, because each serves something that cannot read the others: meson needs
its own at configure time, pip reads pyproject before any code runs, and `__version__`
has to answer without the package being installed. None of them is removable without
giving something up, so instead of one source of truth there is one test.

It is here because the drift already happened: meson.build sat at 1.0.1 against a 1.0.6
package through two releases. Nothing was mis-built, since pyproject is what
meson-python packages from, but `meson dist` produced a tarball named 1.0.1 and anyone
reading meson.build got the wrong answer. A mismatch is silent everywhere it matters
until it is embarrassing, which is the kind worth a test rather than a convention.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: every file that names a version, and how to get it out
SOURCES = {
    "pyproject.toml": lambda p: tomllib.loads(p.read_text())["project"]["version"],
    "agent/pyproject.toml": lambda p: tomllib.loads(p.read_text())["project"]["version"],
    "meson.build": lambda p: re.search(r"^\s*version:\s*'([^']+)'",
                                       p.read_text(), re.M).group(1),
    "rexgraph/__init__.py": lambda p: re.search(r'^__version__\s*=\s*"([^"]+)"',
                                                p.read_text(), re.M).group(1),
    "agent/agent/__init__.py": lambda p: re.search(r'^__version__\s*=\s*"([^"]+)"',
                                                   p.read_text(), re.M).group(1),
}


def _declared() -> dict[str, str]:
    found = {}
    for name, extract in SOURCES.items():
        path = ROOT / name
        if not path.exists():                       # installed tree, not a checkout
            continue
        found[name] = extract(path)
    return found


@pytest.mark.skipif(not (ROOT / "pyproject.toml").exists(),
                    reason="not a source checkout")
def test_every_declared_version_agrees():
    declared = _declared()
    assert declared, "no version declarations found at all"
    distinct = set(declared.values())
    assert len(distinct) == 1, (
        "version declarations disagree:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in sorted(declared.items())))


@pytest.mark.skipif(not (ROOT / "pyproject.toml").exists(),
                    reason="not a source checkout")
def test_the_runtime_version_is_the_declared_one():
    """What the package REPORTS has to be what the packaging says.

    Separate from the file comparison above: that one reads text, this one imports. A
    stale build directory can serve a different `__version__` than the source declares,
    and that is the case where a user's bug report cites a version nobody shipped.
    """
    import rexgraph

    assert rexgraph.__version__ == _declared()["pyproject.toml"]


@pytest.mark.skipif(not (ROOT / "pyproject.toml").exists(),
                    reason="not a source checkout")
def test_the_version_is_a_release_number():
    """Three dot-separated numbers, optionally a pre-release suffix.

    Guards the bump itself rather than taste: `1.0.6-dev`, a stray quote, or an empty
    string all pass an equality check between five identical mistakes.
    """
    version = _declared()["pyproject.toml"]
    assert re.fullmatch(r"\d+\.\d+\.\d+([abrc]\d+|\.dev\d+|\.post\d+)?", version), version
