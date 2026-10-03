"""Phase -1D dependency direction and compatibility contracts."""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import numpy as np

from rexgraph import RexGraph


def _import_targets(module) -> set[str]:
    source = inspect.getsource(module)
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_legacy_state_modules_are_true_aliases():
    pairs = {
        "rexgraph.io.rex_state": "rexgraph.state",
        "rexgraph.io.temporal_state": "rexgraph.temporal_state",
        "rexgraph.io.partition_state": "rexgraph.partition_state",
        "rexgraph.io.model_state": "rexgraph.model_codec",
        "rexgraph.io.field_state": "rexgraph.field_codec",
        "rexgraph.io.section_state": "rexgraph.section_codec",
        "rexgraph.io.span_state": "rexgraph.span_codec",
    }
    for legacy, neutral in pairs.items():
        assert importlib.import_module(legacy) is importlib.import_module(neutral)


def test_state_identity_has_one_implementation():
    from rexgraph.object_identity import object_digest, state_object_digest
    from rexgraph.io.catalog import object_digest as catalog_object_digest
    from rexgraph.io.catalog import state_object_digest as catalog_state_object_digest
    from rexgraph.state import to_state

    rex = RexGraph(
        sources=np.array([0, 1], dtype=np.int32),
        targets=np.array([1, 2], dtype=np.int32),
    )
    state = to_state(rex)
    assert catalog_object_digest is object_digest
    assert catalog_state_object_digest is state_object_digest
    assert object_digest(rex) == state_object_digest(state)


def test_semantic_and_model_layers_do_not_import_concrete_io():
    modules = [
        "rexgraph.state",
        "rexgraph.temporal_state",
        "rexgraph.partition_state",
        "rexgraph.object_identity",
        "rexgraph.model_state",
        "rexgraph.model_runtime",
        "rexgraph.model_contract",
        "rexgraph.artifacts",
        "rexgraph.flow.cochain",
        "rexgraph.flow.model",
        "rexgraph.nn.lifecycle",
        "rexgraph.nn.optim",
    ]
    for name in modules:
        module = importlib.import_module(name)
        bad = sorted(t for t in _import_targets(module) if t == "rexgraph.io" or t.startswith("rexgraph.io."))
        assert bad == [], f"{name} imports concrete IO: {bad}"


def test_all_flow_and_nn_sources_obey_artifact_boundary():
    import rexgraph.flow
    import rexgraph.nn

    for package in (rexgraph.flow, rexgraph.nn):
        root = Path(package.__file__).resolve().parent
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                assert not any(n == "rexgraph.io" or n.startswith("rexgraph.io.") for n in names), path


def test_model_coordinate_contract_is_the_runtime_export():
    from rexgraph.model_contract import model_coordinates as contract
    from rexgraph.model_runtime import model_coordinates as runtime

    assert runtime is contract


def test_artifact_port_accepts_application_backend(monkeypatch):
    import rexgraph.artifacts as artifacts

    calls = []

    class Backend:
        def save_vectors(self, matrix, labels, path, **kwargs):
            calls.append(("vectors", path, kwargs))
            return "vectors"

        def save_rex_artifact(self, rex, path, *, tensors, metadata, format):
            calls.append(("save", path, format, dict(tensors), dict(metadata)))
            return "saved"

        def load_rex_artifact(self, path, *, format):
            calls.append(("load", path, format))
            return artifacts.ArtifactPayload("rex", {"x": 1}, {"m": 2})

    monkeypatch.setattr(artifacts, "_backend", None)
    artifacts.register_artifact_backend(Backend())
    assert artifacts.save_vectors([[1]], ["x"], "v") == "vectors"
    assert artifacts.save_rex_artifact("r", "p", tensors={"a": 1}, metadata={"b": 2}) == "saved"
    payload = artifacts.load_rex_artifact("p")
    assert payload.object == "rex" and payload.tensors == {"x": 1} and payload.metadata == {"m": 2}
    assert [c[0] for c in calls] == ["vectors", "save", "load"]


def test_flow_checkpoint_uses_model_contract_not_lifecycle_directly():
    import rexgraph.flow.cochain as cochain

    targets = _import_targets(cochain)
    assert "rexgraph.nn.lifecycle" not in targets
    assert "rexgraph.model_contract" in targets


def test_checkpoint_contract_bootstraps_registered_lifecycle_provider():
    import rexgraph.model_contract as contract
    import rexgraph.nn.lifecycle as lifecycle

    assert contract._checkpoint_capture is lifecycle.capture_checkpoint
