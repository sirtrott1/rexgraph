"""agent.connectors.semantic: the semantic connector (RDFS/OWL/SPARQL shape).

Wraps the engine's ontology mapping: subClassOf = gradient edge, definition =
face, symmetric/equivalent = bigon, object property = typed edge. A subsumption
cycle with no defining face surfaces as an inconsistency under the same Hodge
readout the schema path uses.

    read(source) -> (rex, meta)

``source`` is an iterable of ``(subject, predicate, object)`` triples - the RDF
the host already has; no live triple store required.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
from . import BaseConnector, Capabilities, ConnectorError


class SemanticConnector(BaseConnector):
    def capabilities(self) -> Capabilities:
        return Capabilities(faces=True, schemes=("ontology", "rdf", "owl"))

    def read(self, source: Any) -> Tuple[Any, Dict[str, Any]]:
        from ..ontology_complex import parse_rdf, ontology_to_rex
        model = parse_rdf(list(source or []))
        rex, meta = ontology_to_rex(model)
        if rex is None:
            raise ConnectorError("no subsumption/definition structure in the triples")
        # Rebuild edges in the engine's own order (model.edges minus self-loops)
        # so the contract's edge list aligns 1:1 with the rex's edges.
        edges = [(a, b) for (a, b, _pred, _kind) in model.edges if a != b]
        return self.result(rex, vertex_labels=list(meta["vertex_labels"]),
                           edges=edges, source=meta.get("source", "ontology"))
