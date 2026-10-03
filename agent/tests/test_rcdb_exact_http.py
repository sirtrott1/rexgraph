"""The actual RCDB routes retain declared metadata across JSON transport."""
from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.server.routes import rcdb as routes
from rcdb import ComplexRecord, MemoryStore
from rexgraph import Absent, ExactArray, RexGraph
from rexgraph.value_codec import pack_value


def test_get_list_query_and_put_use_one_lossless_record_codec(monkeypatch):
    store = MemoryStore()
    meta = {"vertex_labels": ["a", "b"], "weight": Fraction(1, 7),
            "samples": np.arange(3, dtype=np.int16),
            "exact": ExactArray.from_values([Fraction(1, 7), Absent, 2**1000])}
    rex = RexGraph.from_graph([0], [1])
    store.put("record", rex, meta=meta, analytics=False)
    monkeypatch.setattr(routes, "_store", lambda: store)
    rex._agent_meta = meta
    monkeypatch.setattr(routes, "_rex_from_body", lambda body: (rex, "fixture"))
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        responses = [client.get("/v1/db/get/record"), client.get("/v1/db/list"),
                     client.post("/v1/db/query", json={"min_nE": 1}),
                     client.post("/v1/db/put", json={"id": "new"})]
    for response in responses:
        assert response.status_code == 200, response.text
        body = response.json()
        record = ComplexRecord.from_dict(body["records"][0] if "records" in body else body)
        assert pack_value(record.meta) == pack_value(meta)
        assert record.signature["nE"] == 1


def test_generic_metadata_routes_preserve_types_and_graph_routes_refuse_them(monkeypatch):
    from contextlib import closing
    with closing(MemoryStore()) as store:
        store.put_record("value", {"q": Fraction(1, 7)}, meta={"q": Fraction(2, 7)})
        store.put("graph", RexGraph.from_graph([0], [1]), analytics=False)
        monkeypatch.setattr(routes, "_store", lambda: store)
        app = FastAPI()
        app.include_router(routes.router)
        with TestClient(app) as client:
            response = client.post("/v1/db/query", json={"record_type": "NativeValue"})
            assert response.status_code == 200, response.text
            record = ComplexRecord.from_dict(response.json()["records"][0])
            assert record.id == "value" and not record.is_complex and record.meta["q"] == Fraction(2, 7)
            assert client.get("/v1/db/export/value").status_code == 400
            assert client.post("/v1/db/similar", json={"id": "value"}).status_code == 400
