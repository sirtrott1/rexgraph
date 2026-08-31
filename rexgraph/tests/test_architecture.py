"""Core does not import the application built on top of it.

rexgraph/coordinator.py reached into agent.local_runtime for GPU detection, on the
reasoning that a host is a fact about a machine rather than about the math. The reasoning
was sound and the direction was wrong: it made the core depend on the thing built on it,
so rexgraph could not be installed or reasoned about alone. The probe moved to
rexgraph.hardware; this keeps it moved.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN = ("agent", "rcdb", "rcql", "system")


def _modules():
    return sorted(p for p in PACKAGE.rglob("*.py")
                  if "__pycache__" not in p.parts and p.parent.name != "tests")


def test_there_are_modules_to_check():
    assert len(_modules()) > 20, len(_modules())


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_core_does_not_import_what_is_built_on_it(path):
    """Parsed rather than grepped, so a name in a string is not a false positive and an
    import inside a function is not a false pass: the coordinator's was inside one."""
    offenders = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names
                          if a.name.split(".")[0] in FORBIDDEN]
        elif (isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                and node.module.split(".")[0] in FORBIDDEN):
            offenders.append(node.module)
    assert not offenders, f"{path.relative_to(PACKAGE)} imports {offenders}"


def test_the_gpu_probe_is_reachable_from_core():
    """It has to actually be here, not merely absent from the agent."""
    from rexgraph.hardware import detect_gpus, drm_devices, gpu_probes
    assert callable(detect_gpus) and callable(drm_devices)
    assert set(gpu_probes()) >= {"amdgpu", "intel", "nvidia", "apple"}


def test_bus_topology_asks_core_for_its_input():
    """The whole point of the move: this used to work only with the agent installed."""
    from rexgraph.coordinator import detect_bus_topology
    result = detect_bus_topology()
    assert result is None or isinstance(result, dict)


def _readme_imports():
    """Yield (readme, module, names) for every `from X import ...` in a python fence.

    Only rexgraph modules are checked. A README may legitimately show agent or rcdb usage,
    and those packages are not importable from this suite.
    """
    import re

    repo = PACKAGE.parent
    for readme in (repo / "README.md", PACKAGE / "io" / "README.md"):
        if not readme.exists():
            continue
        for fence in re.findall(r"```python\n(.*?)```", readme.read_text(), re.S):
            for mod, raw in re.findall(r"^from (rexgraph[.\w]*) import ([^\n(]+)$", fence, re.M):
                names = [n.strip().split(" as ")[0].strip() for n in raw.split(",") if n.strip()]
                yield readme.relative_to(repo), mod, names


def test_documented_imports_exist():
    """Every name a README tells a reader to import actually resolves.

    Documentation drifts from code silently, because nothing executes it. Two rounds of
    this repo's docs shipped names that were never in the source: four wrong signatures in
    one README and eight nonexistent functions in another, both written from memory of
    what the API ought to be called. A reader following either would get ImportError on
    line one. This executes the import lines so that class of error cannot survive a test
    run.

    A module that will not import for want of an optional dependency is skipped. A module
    that imports but lacks the documented NAME is a failure, which is the actual defect.
    """
    import importlib

    checked = 0
    missing = []
    for readme, mod, names in _readme_imports():
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue  # optional backend, not a documentation defect
        for n in names:
            checked += 1
            if hasattr(m, n):
                continue
            # `from package import submodule` resolves through the import system even when
            # the submodule is not re-exported, so attribute lookup alone under-reports
            try:
                importlib.import_module(f"{mod}.{n}")
            except ImportError:
                missing.append(f"{readme}: from {mod} import {n}")
    assert checked, "no documented rexgraph imports were found to check"
    assert not missing, "documented names that do not exist:\n  " + "\n  ".join(missing)
