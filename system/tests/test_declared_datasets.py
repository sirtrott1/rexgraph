"""System registers declared data without a separate reader/construction path."""
from io import StringIO

from fastapi.testclient import TestClient

from rcql import SourcePolicy
from rexgraph import RelationSpec
from rexgraph.io.declaration import DatasetDeclaration
from rexgraph.io.records import RecordSchema, RecordField
from system import register_dataset, remove_source
from system.server.app import app


def test_registered_dataset_queries_and_native_download():
    declaration = DatasetDeclaration("csv", RecordSchema((RecordField("a"), RecordField("b"))),
                                     RelationSpec("pair", ("a", "b")))
    source = register_dataset("declared", declaration, StringIO("a,b\nx,y\n"), policy=SourcePolicy.allow("read"))
    try:
        with TestClient(app) as client:
            response = client.post("/api/query", json={"query": 'FROM DATASET("declared") RETURN BETTI(0)'})
            assert response.status_code == 200, response.text
            assert response.json()["values"] == [1]
            response = client.post("/api/query", json={"query": 'FROM DATASET("declared") RETURN BETTI(0)', "result_format": "native"})
            assert response.status_code == 200
            from rcql import Result
            restored = Result.from_bytes(response.content)
            assert restored.values == (1,) and source.digest in restored.provenance[0]["source_state"]["name"]
            exact = client.post("/api/query", json={"query": 'FROM DATASET("declared") RETURN BETTI(0)', "exactness": "exact"})
            assert exact.status_code == 200 and exact.json()["native_plan"]["evaluation_policy"]["requested"] == "exact"
            refusal = client.post("/api/query", json={"query": 'FROM DATASET("declared") RETURN $value',
                "params": {"value": .5}, "exactness": "exact"})
            assert refusal.status_code == 400 and "approximate" in refusal.text
    finally:
        remove_source("declared")
