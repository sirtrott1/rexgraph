"""Packaging metadata: a shipped subsystem must have an install path.

A module that hard-imports a third-party package, with no extra that declares it, cannot be
enabled by ANY `pip install rexgraph-agent[...]` invocation - the feature ships dead. That is a
distribution bug the runtime test suite cannot see, because it only ever observes the import
failing in an environment somebody set up by hand.
"""
import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def meta():
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _extras(meta) -> dict[str, list[str]]:
    return meta["project"].get("optional-dependencies", {})


def _resolve(extras: dict[str, list[str]], name: str, _seen=None) -> set[str]:
    """Flatten one extra, following `rexgraph-agent[a,b]` self-references."""
    _seen = _seen if _seen is not None else set()
    if name in _seen:
        return set()
    _seen.add(name)
    out: set[str] = set()
    for req in extras.get(name, []):
        m = re.fullmatch(r"rexgraph-agent\[([^\]]+)\]", req.strip())
        if m:
            for sub in m.group(1).split(","):
                out |= _resolve(extras, sub.strip(), _seen)
        else:
            out.add(re.split(r"[<>=!\[ ]", req.strip(), 1)[0].lower())
    return out


def test_object_storage_is_installable(meta):
    """agent/agent/objectstore.py hard-imports fsspec; some extra has to declare it."""
    declared = set()
    for name in _extras(meta):
        declared |= _resolve(_extras(meta), name)
    declared |= {re.split(r"[<>=!\[ ]", d, 1)[0].lower() for d in meta["project"]["dependencies"]}
    assert "fsspec" in declared, "objectstore.py needs fsspec and no extra installs it"


def test_dev_extra_is_self_sufficient_for_the_test_suite(meta):
    """[dev] documents itself as self-sufficient for pytest. agent/tests/test_store_interop.py
    exercises the object store, so fsspec has to be reachable from [dev]."""
    assert "fsspec" in _resolve(_extras(meta), "dev")
