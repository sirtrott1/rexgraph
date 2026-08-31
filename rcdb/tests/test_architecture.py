"""The boundary that makes this a package rather than a folder.

rcdb must not import the application. That is not tidiness: a store that needs the agent
to describe a complex cannot be installed, tested or reasoned about on its own, and the
rule erodes silently the moment someone needs one convenient thing from the other side.
So it is a test, not a convention.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "rcdb"
REPO = pathlib.Path(__file__).resolve().parents[2]


def _modules():
    return sorted(PACKAGE.glob("*.py"))


def test_the_package_has_modules_to_check():
    assert len(_modules()) >= 6, [p.name for p in _modules()]


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_rcdb_does_not_import_the_agent(path):
    """Checked by parsing rather than by grepping, so a name inside a string or a comment
    is not a false positive and an import hidden inside a function is not a false pass."""
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names
                          if a.name == "agent" or a.name.startswith("agent.")]
        elif (isinstance(node, ast.ImportFrom) and node.module
                and (node.module == "agent" or node.module.startswith("agent."))):
            offenders.append(node.module)
    assert not offenders, f"{path.name} imports {offenders}"


@pytest.mark.parametrize("name", ["rcdb", "rcdb_index", "rexstore", "objectstore",
                                  "rcdb_protected_index"])
def test_the_agent_modules_are_compatibility_surfaces(name):
    """Each is a re-export, so the thirty-odd modules importing agent.rcdb still work."""
    path = REPO / "agent" / "agent" / f"{name}.py"
    if not path.is_dir() and not path.exists():
        pytest.skip(f"{name} is not present in this checkout")
    text = path.read_text()
    assert "from rcdb import" in text, f"{name} does not re-export the package"
    assert len(text.splitlines()) < 30, f"{name} carries an implementation, not a surface"


def test_a_store_works_with_no_hooks_configured():
    """The point of the injection: with nothing supplied the store still stores.

    Importing the agent installs its policy, so this puts the hooks back to nothing first
    and then restores them, rather than assuming the order tests happened to run in.
    """
    from rexgraph.graph import RexGraph

    from rcdb import configure_hooks
    from rcdb.core import (
        _ACTIVITY_HOOK,
        _PRIVACY_HOOK,
        _SCOPE_HOOK,
        _SIMILARITY_HOOK,
        MemoryStore,
    )
    saved = (_ACTIVITY_HOOK, _SCOPE_HOOK, _PRIVACY_HOOK, _SIMILARITY_HOOK)
    configure_hooks()
    try:
        store = MemoryStore()
        rex = RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2])
        store.put("r1", rex, meta={"vertex_labels": ["a", "b"]})
        back = store.get("r1")
        assert back is not None and (back.nV, back.nE) == (rex.nV, rex.nE)
        assert store.get_record("r1").signature["nE"] == rex.nE
    finally:
        configure_hooks(activity=saved[0], scope=saved[1], privacy=saved[2],
                        similarity=saved[3])


def test_the_agent_installs_its_policy():
    """And with the agent present, all four arrive."""
    pytest.importorskip("agent")
    import agent  # noqa: F401  - importing is what installs them

    from rcdb import core
    assert core._ACTIVITY_HOOK is not None
    assert core._SCOPE_HOOK is not None
    assert core._PRIVACY_HOOK is not None
    assert core._SIMILARITY_HOOK is not None


def test_the_public_surface_is_reachable_from_the_package():
    """Every name in __all__ resolves, and the surface is not quietly narrowed.

    The agent re-exports this package dynamically, so a name dropped from __init__ still
    reaches anyone importing agent.rcdb and nothing fails there. That masking is exactly
    why this checks the package DIRECTLY: the first version of this __init__ exported a
    fraction of the surface and the agent suite stayed green.
    """
    import rcdb
    missing = [n for n in rcdb.__all__ if not hasattr(rcdb, n)]
    assert not missing, f"__all__ names that do not resolve: {missing}"

    # the surface a caller is entitled to expect, spanning every submodule
    expected = {
        "RCStore", "MemoryStore", "FileStore", "SQLStore", "RexStore", "ObjectStore",
        "ComplexRecord", "open_store", "default_store", "register_backend",
        "available_backends", "configure_hooks", "structural_signature",
        "serialize_complex", "deserialize_complex", "find_similar", "lineage",
        "IndexPolicy", "SearchRelation", "build_search_relation", "term_token",
    }
    assert expected <= set(rcdb.__all__), sorted(expected - set(rcdb.__all__))


def test_safetensors_is_a_base_dependency_not_an_extra():
    """A dependency every write needs must not be optional.

    Every backend's put goes through serialize_complex, which writes safetensors bytes,
    and index.py reads and writes the tensor index with it. Declared as an extra, a base
    install imports cleanly and then fails on its first put, which is the worst place to
    learn a dependency is missing. This was a real regression: the extraction moved
    safetensors from base into a search extra, and a system-site test environment that
    happened to have safetensors installed hid it from every suite.
    """
    import re
    import tomllib
    from pathlib import Path

    meta = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    project = meta["project"]
    base = {re.split(r"[<>=!\[ ]", d, maxsplit=1)[0].lower() for d in project["dependencies"]}

    # what a bare install must be able to do: import a store and complete a write
    for required in ("rexgraph", "numpy", "safetensors"):
        assert required in base, (
            f"{required} is needed by every put but is not a base dependency; "
            f"base declares {sorted(base)}"
        )

    # and an extra must not re-declare it, which is how the demotion reads as harmless
    for name, deps in project.get("optional-dependencies", {}).items():
        if name == "dev":
            continue
        names = {re.split(r"[<>=!\[ ]", d, maxsplit=1)[0].lower() for d in deps}
        assert "safetensors" not in names, f"extra {name!r} re-declares a base dependency"
