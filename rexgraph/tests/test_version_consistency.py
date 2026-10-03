"""Check all five distribution versions and their sibling dependency floors.

The eleven version declarations cover pyproject metadata, Meson and package
__version__ values. Runtime imports must report the declared release.
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


# Version declarations and their readers.
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


@pytest.mark.skipif(not (ROOT / "pyproject.toml").exists(), reason="not a source checkout")
def test_sibling_dependency_floors_cover_the_current_shared_contracts():
    version = _declared()["pyproject.toml"]
    for name in ("agent", "rcdb", "rcql", "system"):
        project = tomllib.loads((ROOT / name / "pyproject.toml").read_text())["project"]
        dependencies = project.get("dependencies", []) + [
            item for extra in project.get("optional-dependencies", {}).values() for item in extra]
        for dependency in dependencies:
            if re.match(r"rexgraph(?:-(?:rcdb|rcql|system|agent))?>=", dependency):
                assert dependency.split(">=", 1)[1] == version, (name, dependency, version)


@pytest.mark.skipif(not (ROOT / "pyproject.toml").exists(),
                    reason="not a source checkout")
def test_the_version_is_a_release_number():
    """Three dot separated numbers, optionally a pre release suffix.

    Guards the bump itself rather than taste: `1.0.6-dev`, a stray quote, or an empty
    string all pass an equality check between identical invalid declarations.
    """
    version = _declared()["pyproject.toml"]
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+|\.post\d+)?", version), version


def _version_command(tmp_path, monkeypatch):
    import runpy

    command = runpy.run_path(str(ROOT / "scripts/set_version.py"))
    for name in command["TARGETS"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setitem(command["main"].__globals__, "ROOT", tmp_path)
    return command


def test_version_command_updates_every_distribution_and_floor(tmp_path, monkeypatch):
    command = _version_command(tmp_path, monkeypatch)
    original = {name: (tmp_path / name).read_bytes() for name in SOURCES}
    assert command["main"](["1.4.0rc1", "--dry-run"]) == 0
    assert original == {name: (tmp_path / name).read_bytes() for name in SOURCES}
    assert command["main"](["1.4.0rc1"]) == 0
    assert {extract(tmp_path / name) for name, extract in SOURCES.items()} == {"1.4.0rc1"}
    for package in ("agent", "rcdb", "rcql", "system"):
        project = tomllib.loads((tmp_path / package / "pyproject.toml").read_text())["project"]
        dependencies = project.get("dependencies", []) + [
            item for extra in project.get("optional-dependencies", {}).values() for item in extra]
        for dependency in dependencies:
            if dependency.startswith("rexgraph"):
                if dependency.startswith(project["name"] + "["):
                    continue
                assert dependency.endswith(">=1.4.0rc1"), dependency


def test_version_command_validates_all_targets_before_writing(tmp_path, monkeypatch):
    command = _version_command(tmp_path, monkeypatch)
    (tmp_path / "system/system/__init__.py").write_text("# missing version\n")
    original = {name: (tmp_path / name).read_bytes() for name in SOURCES}
    assert command["main"](["1.4.0"]) == 3
    assert original == {name: (tmp_path / name).read_bytes() for name in SOURCES}
