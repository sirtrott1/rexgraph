"""The version the package reports has to be the version in its source.

These ship together and their formats depend on each other, so a package reporting one
version while its distribution declares another makes the floors meaningless: the whole
point of raising them was that a mismatched pair cannot be installed.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        m = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    pytest.fail("no static version in pyproject.toml")


def test_module_and_pyproject_agree():
    import rcql
    assert rcql.__version__ == _pyproject_version()


def test_the_version_is_semver():
    import rcql
    assert re.match(r"^\d+\.\d+\.\d+", rcql.__version__), rcql.__version__
