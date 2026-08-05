"""The versions the app reports have to be the versions in the source.

The agent version was read from installed package metadata while the core was read
from its module. In an editable checkout that is two sources: the agent reported the
version it happened to be installed at (1.0.0) next to a core that had moved to
1.0.6, in the status bar, where the mismatch reads as a real fact about the build.
Both now come from their module, and these tests pin pyproject to match.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEMVER = re.compile(r"^\d+\.\d+\.\d+")


def _pyproject_version(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    pytest.fail(f"no static version in {path}")


def test_agent_module_and_pyproject_agree():
    import agent
    assert agent.__version__ == _pyproject_version(ROOT / "agent" / "pyproject.toml")


def test_core_module_and_pyproject_agree():
    import rexgraph
    assert rexgraph.__version__ == _pyproject_version(ROOT / "pyproject.toml")


def test_both_versions_are_semver():
    import agent
    import rexgraph
    assert SEMVER.match(agent.__version__), agent.__version__
    assert SEMVER.match(rexgraph.__version__), rexgraph.__version__


def test_launch_reports_the_source_versions_not_install_metadata():
    """What the status bar shows comes from the modules, so an editable checkout
    reports what it actually is."""
    import agent
    import rexgraph
    from agent.server.launch import _agent_version, _core_version
    assert _agent_version() == agent.__version__
    assert _core_version() == rexgraph.__version__
