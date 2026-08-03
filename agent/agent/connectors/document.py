"""agent.connectors.document: the document/NoSQL connector (MongoDB shape).

Wraps the engine's Mongo schema inference: collections are vertices, inferred
``*_id`` references are edges, junction collections become co-participation
faces. Read-only sampling - it infers structure from a sample of documents,
never returning document values.

    read(source) -> (rex, meta)

``source`` may be an in-memory ``{collection: [docs]}`` mapping (inferred with
no live service) or a ``mongodb://host/dbname`` URI (sampled live via pymongo).
"""
from __future__ import annotations

from typing import Any

from . import BaseConnector, Capabilities, ConnectorError


class DocumentConnector(BaseConnector):
    def __init__(self, sample: int = 100):
        self.sample = sample

    def capabilities(self) -> Capabilities:
        return Capabilities(modality=False, faces=True, schemes=("mongodb",))

    def read(self, source: Any) -> tuple[Any, dict[str, Any]]:
        from ..schema_complex import infer_mongo_schema, reflect_mongo, schema_to_rex
        if isinstance(source, dict):
            model = infer_mongo_schema(source)
            tag = "mongodb://in-memory"
        else:
            uri = str(source)
            db = uri.rstrip("/").rsplit("/", 1)[-1]
            model = reflect_mongo(uri, db, sample=self.sample)
            from ..secrets import mask_uri
            tag = mask_uri(uri)
        rex, sm = schema_to_rex(model)
        if rex is None:
            raise ConnectorError("no inferred references - nothing to form a complex")
        return self.result(rex, vertex_labels=list(sm["vertex_labels"]),
                           edges=[tuple(e) for e in sm["edges"]], source=tag)
