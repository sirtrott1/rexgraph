"""Packaging metadata: a shipped subsystem must have an install path.

A module that hard-imports a third-party package, with no extra that declares it, cannot be
enabled by ANY `pip install rexgraph-agent[...]` invocation - the feature ships dead. That is a
distribution bug the runtime test suite cannot see, because it only ever observes the import
failing in an environment somebody set up by hand.
"""
import re
from pathlib import Path

import pytest
import tomllib

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
            out.add(re.split(r"[<>=!\[ ]", req.strip(), maxsplit=1)[0].lower())
    return out


def test_object_storage_is_installable(meta):
    """agent/agent/objectstore.py hard-imports fsspec; some extra has to declare it."""
    declared = set()
    for name in _extras(meta):
        declared |= _resolve(_extras(meta), name)
    declared |= {re.split(r"[<>=!\[ ]", d, maxsplit=1)[0].lower()
                 for d in meta["project"]["dependencies"]}
    assert "fsspec" in declared, "objectstore.py needs fsspec and no extra installs it"


def test_dev_extra_is_self_sufficient_for_the_test_suite(meta):
    """[dev] documents itself as self-sufficient for pytest. agent/tests/test_store_interop.py
    exercises the object store, so fsspec has to be reachable from [dev]."""
    assert "fsspec" in _resolve(_extras(meta), "dev")


def test_container_sealing_is_installable(meta):
    """agent/agent/kms.py needs an AEAD implementation; some extra has to declare it."""
    declared = set()
    for name in _extras(meta):
        declared |= _resolve(_extras(meta), name)
    declared |= {re.split(r"[<>=!\[ ]", d, maxsplit=1)[0].lower()
                 for d in meta["project"]["dependencies"]}
    assert "cryptography" in declared, "kms.py needs cryptography and no extra installs it"


def test_dev_extra_can_run_the_sealing_tests(meta):
    """[dev] documents itself as self-sufficient for pytest, and agent/tests/test_kms.py
    seals a real bundle."""
    assert "cryptography" in _resolve(_extras(meta), "dev")


def _agent_modules():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "agent"
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _module_level_imports(path):
    """Top-level package names imported at MODULE scope, so an install must provide them.

    An import inside a function is a soft dependency: the module loads without it and only
    a caller reaching that path pays. One at module scope is not optional at all.
    """
    import ast
    tree = ast.parse(path.read_text())
    out = set()
    for node in tree.body:                           # module scope only, deliberately
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_a_sibling_imported_at_module_scope_is_a_declared_dependency(meta):
    """The gap that let the store move out of the agent without the agent saying so.

    agent.metrics and agent.scoring re-export from rcdb.analytics, and agent.rcdb and its
    four siblings are surfaces over the package, all at module scope. An install without
    rexgraph-rcdb therefore fails at import, which no other test here would have caught.
    """
    siblings = {"rcdb": "rexgraph-rcdb", "rcql": "rexgraph-rcql",
                "system": "rexgraph-system", "rexgraph": "rexgraph"}
    declared = {re.split(r"[<>=!\[ ]", d, maxsplit=1)[0].lower()
                for d in meta["project"]["dependencies"]}
    needed = {}
    for path, chain in _reachable_at_import().items():
        for name in _module_level_imports(path) & siblings.keys():
            needed.setdefault(siblings[name], []).append(" -> ".join(chain))
    missing = {dist: how for dist, how in needed.items() if dist not in declared}
    assert not missing, (
        f"imported at module scope but not a base dependency: {missing}")


def _reachable_at_import():
    """Modules actually loaded by `import agent`, and the chain that reaches each.

    Reachability is the question, not presence. agent.rcql_runtime imports rcql at module
    scope but is itself reached only through agent.__getattr__, so an install without
    rexgraph-rcql still imports the agent and only a caller asking for the runtime pays.
    A module scope import inside a module nothing loads at import is a soft dependency.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "agent"
    seen, out, todo = set(), {}, [("__init__", ["agent"])]
    while todo:
        name, chain = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        path = root / f"{name.replace('.', '/')}.py"
        if not path.exists():
            path = root / name.replace(".", "/") / "__init__.py"
        if not path.exists():
            continue
        out[path] = chain
        import ast
        for node in ast.parse(path.read_text()).body:
            # Within the package both forms appear: `from .metrics import x` and
            # `from agent.metrics import x`. Following only one of them made this walk
            # miss most of the graph and the test pass while the dependency was absent.
            child = None
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 1:
                    child = node.module
                elif node.level == 0 and node.module.startswith("agent."):
                    child = node.module[len("agent."):]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("agent."):
                        todo.append((alias.name[len("agent."):],
                                     [*chain, alias.name]))
            if child:
                todo.append((child, [*chain, f"agent.{child}"]))
    return out
