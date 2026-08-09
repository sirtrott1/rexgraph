"""Tests for file I/O integration: rexgraph.io format round-trips through
the app (safetensors upload/export), training-data export (the fixed
safetensors bugs), downloadable exports, and custom-model registration."""

import os
import tempfile

import pytest

# rexgraph.io layer: safetensors dispatch

def _make_safetensors_complex(suffix=".safetensors"):
    from rexgraph.graph import RexGraph
    from rexgraph.io.safetensors_bridge import rex_to_safetensors
    r = RexGraph.from_graph([0, 1, 2, 3, 0], [1, 2, 3, 4, 4])
    p = tempfile.mktemp(suffix=suffix)
    rex_to_safetensors(r, p)
    return r, p


class TestIoLayer:
    def test_io_load_dispatches_safetensors(self):
        """Regression: io.load fell through to HDF5 for .safetensors."""
        from rexgraph.io import load as io_load
        r, p = _make_safetensors_complex()
        r2 = io_load(p)
        assert r2.nV == r.nV and r2.nE == r.nE

    def test_auto_rex_accepts_safetensors(self):
        """Regression: auto_rex raised 'Unsupported file format: .safetensors'."""
        from agent.auto import auto_rex
        r, p = _make_safetensors_complex()
        r2 = auto_rex(p)
        assert r2.nV == r.nV


# training exporter (the fixed safetensors bugs)

class TestTrainingExporter:
    def _exporter(self):
        from agent.training import TrainingExporter
        return TrainingExporter.from_texts(
            ["cells signal receptors genes express proteins pathways",
             "neurons fire synapses transmit signals dendrites axons"],
            doc_ids=["a", "b"])

    def test_feature_names_is_property_list(self):
        te = self._exporter()
        names = te.feature_names           # property, not a method
        assert isinstance(names, list) and len(names) > 0

    def test_export_features_safetensors(self):
        """Regression: KeyError on reserved 'n_features' metadata key."""
        te = self._exporter()
        p = tempfile.mktemp(suffix=".safetensors")
        te.export_features(p)
        assert os.path.getsize(p) > 0
        from safetensors.numpy import load_file
        d = load_file(p)
        assert "features" in d

    def test_export_training_pairs(self):
        te = self._exporter()
        p = tempfile.mktemp(suffix=".safetensors")
        te.export_training_pairs(p, target="summary")
        assert os.path.getsize(p) > 0


# server: upload/export/download I/O + custom models

@pytest.fixture(scope="module")
def client():
    from agent.server.app import app
    from agent.server.auth import get_auth_manager
    from fastapi.testclient import TestClient
    get_auth_manager().disable_auth(persist=False)  # default posture; a prior run may have enabled it
    with TestClient(app) as c:
        yield c


class TestServerIO:
    def test_upload_safetensors_complex(self, client):
        _, p = _make_safetensors_complex()
        with open(p, "rb") as f:
            r = client.post("/api/upload",
                            files={"file": ("g.safetensors", f.read(), "application/octet-stream")},
                            data={"options": "{}"})
        assert r.status_code == 200
        assert r.json()["nV"] == 5

    def test_export_session_safetensors_download(self, client):
        _, p = _make_safetensors_complex()
        with open(p, "rb") as f:
            sid = client.post("/api/upload",
                              files={"file": ("g.safetensors", f.read(), "application/octet-stream")},
                              data={"options": "{}"}).json()["session_id"]
        r = client.get(f"/api/v1/export/session/{sid}?format=safetensors")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        assert len(r.content) > 0
        # and the exported bytes load back as a complex
        p2 = tempfile.mktemp(suffix=".safetensors")
        with open(p2, "wb") as f:
            f.write(r.content)
        from rexgraph.io import load as io_load
        assert io_load(p2).nV == 5

    def test_export_session_hdf5_download(self, client):
        _, p = _make_safetensors_complex()
        with open(p, "rb") as f:
            sid = client.post("/api/upload",
                              files={"file": ("g.safetensors", f.read(), "application/octet-stream")},
                              data={"options": "{}"}).json()["session_id"]
        r = client.get(f"/api/v1/export/session/{sid}?format=hdf5")
        assert r.status_code == 200 and len(r.content) > 0

    def test_training_download_after_corpus_build(self, client):
        client.post("/api/v1/corpus/add",
                    data={"text": "cells signal receptors genes express proteins pathways", "doc_id": "d1"})
        client.post("/api/v1/corpus/add",
                    data={"text": "neurons fire synapses transmit signals dendrites axons", "doc_id": "d2"})
        b = client.post("/api/v1/corpus/build", data={"depth": "quick"})
        assert b.status_code == 200
        r = client.get("/api/v1/model/training/download?fmt=safetensors")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        assert len(r.content) > 0
        # pairs format too
        r2 = client.get("/api/v1/model/training/download?fmt=pairs")
        assert r2.status_code == 200 and len(r2.content) > 0

    def test_custom_model_registration(self, client):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "config.json"), "w") as f:
            f.write("{}")
        r = client.post("/api/v1/models/set-path",
                        json={"model_id": "my-model", "path": d, "model_type": "transformers"})
        assert r.status_code == 200
        assert r.json()["status"] == "registered"
        # shows up in the registry
        lst = client.get("/api/v1/models/list").json()["models"]
        assert any((m.get("model_id") or m.get("id")) == "my-model" for m in lst)
        # can be assigned to the chat pipeline stage
        sp = client.post("/api/v1/models/set-pipeline",
                         json={"purpose": "chat", "model_id": "my-model"})
        assert sp.status_code == 200
