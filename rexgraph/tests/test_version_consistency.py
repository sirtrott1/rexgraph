"""Every place that states a version must state the same one.

There are eleven, because each serves something that cannot read the others: meson needs
its own at configure time, pip reads pyproject before any code runs, and `__version__`
has to answer without the package being installed. None of them is removable without
giving something up, so instead of one source of truth there is one test.

It matters more than it did. rcdb, rcql and system are their own distributions now, and
the wire formats broke this release, so every inter-distribution floor was raised to
>=1.1.3. That floor rejects a pre-1.1.3 sibling; it does not pin the five to each other,
and a later release can still resolve an older one unless its own floor moves. What this
test guarantees is the half that is checkable here: this source release states 1.1.3 in
every one of the eleven places that state a version.

It is here because the drift already happened: meson.build sat at 1.0.1 against a 1.0.6
package through two releases. Nothing was mis-built, since pyproject is what
meson-python packages from, but `meson dist` produced a tarball named 1.0.1 and anyone
reading meson.build got the wrong answer. A mismatch is silent everywhere it matters
until it is embarrassing, which is the kind worth a test rather than a convention.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[2]

def _toml_version(p):
    return tomllib.loads(p.read_text())["project"]["version"]


def _module_version(p):
    return re.search(r'^__version__\s*=\s*"([^"]+)"', p.read_text(), re.M).group(1)


#: every file that names a version, and how to get it out.
#:
#: There are eleven now rather than five, because rcdb, rcql and system became
#: distributions of their own. A package declaring one version while its distribution
#: declares another is what makes a floor meaningless, and each package's own test proves
#: only that it agrees with ITSELF, so the cross-distribution check lives here.
SOURCES = {
    "pyproject.toml": _toml_version,
    "agent/pyproject.toml": _toml_version,
    "rcdb/pyproject.toml": _toml_version,
    "rcql/pyproject.toml": _toml_version,
    "system/pyproject.toml": _toml_version,
    "meson.build": lambda p: re.search(r"^\s*version:\s*'([^']+)'",
                                       p.read_text(), re.M).group(1),
    "rexgraph/__init__.py": _module_version,
    "agent/agent/__init__.py": _module_version,
    "rcdb/rcdb/__init__.py": _module_version,
    "rcql/rcql/__init__.py": _module_version,
    "system/system/__init__.py": _module_version,
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
