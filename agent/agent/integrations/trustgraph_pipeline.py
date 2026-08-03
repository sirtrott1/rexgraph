"""
TrustGraph full integration pipeline.

End-to-end workflow: ingest from TrustGraph, build relational complex,
analyze, interpret, write enrichment back.

Usage:

    pipe = TrustGraphPipeline(url="http://localhost:8088/")

    # Analyze an entire flow
    result = pipe.analyze_flow("default")

    # Analyze specific documents
    result = pipe.analyze_documents(["doc1", "doc2"])

    # Score a RAG query
    confidence = pipe.query_confidence(
        flow="default",
        entities=["Metformin", "Diabetes", "AMPK"],
    )

    # Decompose a query signal
    decomp = pipe.decompose_query_signal(
        flow="default",
        signal=relevance_scores,
    )

    # Write structural enrichments back
    n = pipe.write_enrichment(result)

Standalone mode (no running TrustGraph):

    pipe = TrustGraphPipeline.standalone()
    result = pipe.analyze_triples(triples, contexts=contexts)
"""

from __future__ import annotations

import numpy as np

from agent.engine import DecisionEngine, EngineResult
from agent.integrations.trustgraph_adapter import (
    TrustGraphAdapter,
)


class TrustGraphPipeline:
    """End-to-end pipeline from TrustGraph to structural analysis.

    Connects the DecisionEngine to TrustGraph's API for document
    ingestion, triple export, complex construction, analysis, and
    enrichment write-back.
    """

    def __init__(
        self,
        url: str = None,
        token: str = None,
        workspace: str = "default",
        timeout: int = 60,
    ):
        self.adapter = TrustGraphAdapter(
            url=url, token=token,
            workspace=workspace, timeout=timeout,
        )
        self.engine = DecisionEngine()
        self._last_result: EngineResult | None = None

    @classmethod
    def standalone(cls) -> TrustGraphPipeline:
        """Create a pipeline for standalone use (no TG instance)."""
        return cls(url=None)

    # Flow-level analysis

    def analyze_flow(
        self,
        flow: str = "default",
        depth: str = None,
        signal: np.ndarray = None,
    ) -> EngineResult:
        """Export all triples from a TG flow and run full analysis.

        Uses BulkClient.export_triples to stream triples, builds
        a context matrix from named graphs (document provenance),
        constructs the relational complex with algebraic face
        selection, and runs the full analysis pipeline.

        Parameters
        ----------
        flow : str
            TrustGraph flow identifier.
        depth : str, optional
            Override analysis depth ('quick', 'standard', 'full').
        signal : ndarray, optional
            Edge signal to decompose (e.g., query relevance).

        Returns
        -------
        EngineResult
        """
        triples = list(self.adapter.bulk.export_triples(flow=flow))

        # Build context matrix from named graphs
        contexts = self._contexts_from_named_graphs(triples)

        result = self.engine.run(
            triples,
            contexts=contexts if contexts else None,
            signal=signal,
            depth=depth,
        )
        self._last_result = result
        return result

    # Document-level analysis

    def analyze_documents(
        self,
        document_ids: list[str],
        flow: str = "default",
        depth: str = None,
    ) -> EngineResult:
        """Analyze specific documents from the TG library.

        Exports triples, filters to those from the specified
        documents, uses document membership as the context matrix.

        Parameters
        ----------
        document_ids : list of str
        flow : str
        depth : str, optional

        Returns
        -------
        EngineResult
        """
        all_triples = list(
            self.adapter.bulk.export_triples(flow=flow)
        )

        # Filter to triples from the specified documents
        doc_set = set(document_ids)
        filtered = []
        for t in all_triples:
            g = getattr(t, "g", None) or "default"
            if g in doc_set or g == "default":
                filtered.append(t)

        if not filtered:
            filtered = all_triples

        # Build context from document IDs
        contexts = {
            doc_id: self._entities_in_document(all_triples, doc_id)
            for doc_id in document_ids
        }

        result = self.engine.run(
            filtered, contexts=contexts, depth=depth,
        )
        self._last_result = result
        return result

    # Standalone triple analysis

    def analyze_triples(
        self,
        triples: list,
        contexts: dict[str, list[str]] = None,
        signal: np.ndarray = None,
        depth: str = None,
    ) -> EngineResult:
        """Analyze triples without a running TG instance.

        Parameters
        ----------
        triples : list of SimpleTriple, Triple, or (s, p, o) tuples
        contexts : dict, optional
        signal : ndarray, optional
        depth : str, optional

        Returns
        -------
        EngineResult
        """
        result = self.engine.run(
            triples, contexts=contexts, signal=signal,
            depth=depth,
        )
        self._last_result = result
        return result

    # Query confidence

    def query_confidence(
        self,
        entities: list[str],
        flow: str = "default",
        rex=None,
        meta: dict = None,
    ) -> dict:
        """Score structural confidence for a set of entities.

        If a previous analysis result is available (from
        analyze_flow or analyze_triples), uses the cached complex.
        Otherwise exports triples and builds a new complex.

        Parameters
        ----------
        entities : list of str
            Entity names to score.
        flow : str
            TG flow (used if no cached result).
        rex : RexGraph, optional
            Pre-built complex (overrides cache and flow).
        meta : dict, optional
            Metadata with vertex_labels.

        Returns
        -------
        dict with confidence scores per view.
        """
        if rex is None:
            if self._last_result is not None:
                rex = self._last_result.rex
                meta = self._last_result.meta
            else:
                result = self.analyze_flow(flow)
                rex = result.rex
                meta = result.meta

        labels = meta.get("vertex_labels", [])
        indices = []
        for name in entities:
            if name in labels:
                indices.append(labels.index(name))

        if not indices:
            return {
                "confidence": "NONE",
                "reason": f"No matching entities found for {entities}",
            }

        return self.adapter.subgraph_confidence(rex, indices)

    # Signal decomposition

    def decompose_query_signal(
        self,
        signal: np.ndarray,
        signal_name: str = "query",
        flow: str = "default",
        rex=None,
    ) -> dict:
        """Decompose an edge signal on the knowledge graph.

        Parameters
        ----------
        signal : f64[nE]
        signal_name : str
        flow : str
        rex : RexGraph, optional

        Returns
        -------
        dict with Hodge decomposition, channel character, face/void
        dipole.
        """
        if rex is None:
            if self._last_result is not None:
                rex = self._last_result.rex
            else:
                result = self.analyze_flow(flow)
                rex = result.rex

        return self.adapter.decompose_signal(
            rex, signal, signal_name
        )

    # Enrichment write-back

    def write_enrichment(
        self,
        result: EngineResult = None,
        flow: str = "default",
    ) -> int:
        """Write structural enrichment triples back to TrustGraph.

        Parameters
        ----------
        result : EngineResult, optional
            If None, uses the last cached result.
        flow : str
            TG flow to write to.

        Returns
        -------
        int
            Number of triples written.
        """
        if result is None:
            result = self._last_result
        if result is None:
            raise RuntimeError(
                "No analysis result available. "
                "Run analyze_flow or analyze_triples first."
            )

        return self.adapter.write_enrichment_triples(
            result.rex, result.analysis, flow=flow,
        )

    # Helpers

    def _contexts_from_named_graphs(self, triples) -> dict | None:
        """Extract document contexts from triple named graphs."""
        from agent.integrations.trustgraph_adapter import (
            _extract_context_id,
            _triple_to_strings,
        )

        groups = {}
        for t in triples:
            ctx = _extract_context_id(t)
            if ctx == "default":
                continue
            s, p, o = _triple_to_strings(t)
            if ctx not in groups:
                groups[ctx] = set()
            groups[ctx].add(s)
            groups[ctx].add(o)

        if len(groups) <= 1:
            return None

        return {k: list(v) for k, v in groups.items()}

    def _entities_in_document(self, triples, doc_id) -> list[str]:
        """Find all entities mentioned in a specific document."""
        from agent.integrations.trustgraph_adapter import (
            _extract_context_id,
            _triple_to_strings,
        )

        entities = set()
        for t in triples:
            ctx = _extract_context_id(t)
            if ctx == doc_id:
                s, _, o = _triple_to_strings(t)
                entities.add(s)
                entities.add(o)
        return list(entities)
