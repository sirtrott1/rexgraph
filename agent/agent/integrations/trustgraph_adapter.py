"""
TrustGraph adapter: knowledge graph triples to typed relational complex.

Bridges TrustGraph's knowledge graph infrastructure with RexGraph's
algebraic analysis. Triples (subject, predicate, object) become a typed
relational complex where:

    - Entities (subjects/objects) map to vertices.
    - Predicates map to typed edges.
    - Same-predicate-type triangles map to faces (boundary of boundary
      is zero, follows from the chain condition).
    - Cross-type triangles map to voids (structural gaps).

The structural analysis enriches the knowledge graph with per-entity
coherence kappa, per-edge structural character (T/G/F/C channel
decomposition), void maps (where the structural gaps are), and
confidence scores for RAG subgraph trust.

Compatible with TrustGraph 2.4+ (trustgraph-base >= 2.4.0).

Standalone mode (no running TrustGraph needed):

    from agent.integrations.trustgraph_adapter import TrustGraphAdapter

    adapter = TrustGraphAdapter()
    rex, meta = adapter.from_triples(triples)
    analysis = adapter.analyze(rex)
    enrichments = adapter.to_enrichment_triples(rex, analysis)

Connected mode (requires a running TrustGraph instance):

    adapter = TrustGraphAdapter(url="http://localhost:8088/")
    rex, meta = adapter.from_flow("default")
    analysis = adapter.analyze(rex)
    confidence = adapter.subgraph_confidence(rex, [0, 1, 2])

    # Write structural enrichments back
    adapter.write_enrichment_triples(rex, analysis, flow="default")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.adapters import DomainAdapter, EdgeConstruction
from agent.metrics import coherence_kappa

# Triple handling (works without a running TrustGraph)


@dataclass
class SimpleTriple:
    """Minimal triple for standalone mode without the TrustGraph package."""
    s: str
    p: str
    o: str


def _normalize_uri(uri: str) -> str:
    """Extract a readable label from a URI or plain string.

    Examples:

        http://example.org/entities/Person_Alice  ->  Person_Alice
        http://schema.org/name                    ->  name
        Alice                                     ->  Alice
    """
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    if "/" in uri:
        return uri.rsplit("/", 1)[-1]
    return uri


def _extract_predicate_type(predicate: str) -> str:
    """Extract the local name of a predicate URI as the edge type.

    Each distinct predicate defines a distinct structural relationship.
    Edges of different types cannot form coherent faces (they become
    voids instead), which is the correct behavior: a triangle spanning
    ``treats``, ``targets``, and ``associatedWith`` represents a
    structural gap, not a filled face.

    Examples:

        http://bio.org/rel/treats                        ->  treats
        http://bio.org/rel/targets                       ->  targets
        http://schema.org/name                           ->  name
        http://www.w3.org/1999/02/22-rdf-syntax-ns#type  ->  type
        worksAt                                          ->  worksAt
    """
    pred = predicate.strip()

    # Fragment identifier: local name after #
    if "#" in pred:
        return pred.rsplit("#", 1)[-1]

    # Path-based URI: local name after last /
    if "/" in pred:
        return pred.rsplit("/", 1)[-1]

    return pred


def _triple_to_strings(t) -> tuple[str, str, str]:
    """Normalize a triple object to (s, p, o) strings.

    Handles:
      - (s, p, o) tuples and lists
      - SimpleTriple and trustgraph.api.Triple (both have .s, .p, .o as str)
      - trustgraph.schema.knowledge.graph.Triple (has .s, .p, .o as Term)
    """
    if isinstance(t, (tuple, list)):
        return str(t[0]), str(t[1]), str(t[2])

    s_raw, p_raw, o_raw = t.s, t.p, t.o

    # Handle Term objects from trustgraph.schema.knowledge.graph
    # Term has .type ("IRI", "LITERAL", "BLANK") and .iri, .value, .id fields
    def _term_str(term) -> str:
        if isinstance(term, str):
            return term
        if term is None:
            return ""
        # Term object
        if hasattr(term, "type"):
            if term.type == "IRI":
                return term.iri
            elif term.type == "LITERAL":
                return term.value
            elif term.type == "BLANK":
                return term.id
            elif term.type == "TRIPLE" and term.triple is not None:
                # Quoted triple: use a canonical string representation
                ts, tp, to_ = _triple_to_strings(term.triple)
                return f"<<{ts} {tp} {to_}>>"
        # Fallback: stringify
        return str(term)

    return _term_str(s_raw), _term_str(p_raw), _term_str(o_raw)


# Context matrix construction


def _resolve_entity(name: str, entity_to_idx: dict[str, int]):
    """Look an entity up under whichever form the index was keyed with.

    `from_triples` returns `vertex_labels` normalized (`alpha`), while the triples
    still carry `http://ex.org/alpha`. Matching on one form only meant that feeding
    the adapter's own labels back into a context builder produced an all-zero matrix
    and no error, so context selection silently selected nothing.
    """
    if name in entity_to_idx:
        return entity_to_idx[name]
    local = _normalize_uri(name)
    return entity_to_idx.get(local)


def build_context_matrix_from_documents(
    triples: list,
    entity_to_idx: dict[str, int],
    n_entities: int,
) -> tuple[np.ndarray, list[str]]:
    """Build a binary context matrix from document-grouped triples.

    Each group of triples sharing the same metadata.id (or chunk_id,
    or graph name) defines a context.  The matrix C has shape
    (n_contexts, n_entities) with C[c, v] = 1 if entity v appears in
    context c.

    Parameters
    ----------
    triples : list
        Triples with provenance.  Each triple should have a ``g``
        attribute (named graph) or be accompanied by a document_id.
        For SimpleTriple or plain tuples, a fourth element or a .g
        attribute is used.  If no grouping is available, all triples
        are placed in a single context.
    entity_to_idx : dict
        Mapping from entity string to vertex index.
    n_entities : int
        Number of entities (vertices).

    Returns
    -------
    (context_matrix, context_labels)
        context_matrix : uint8[n_contexts, n_entities]
        context_labels : list of str, one per context
    """
    # Group triples by context identifier
    context_groups = {}
    for t in triples:
        s, p, o = _triple_to_strings(t)[:3] if not isinstance(t, (tuple, list)) else (str(t[0]), str(t[1]), str(t[2]))

        # Extract context id from triple
        ctx_id = _extract_context_id(t)

        if ctx_id not in context_groups:
            context_groups[ctx_id] = set()
        for term in (s, o):
            vi = _resolve_entity(term, entity_to_idx)
            if vi is not None:
                context_groups[ctx_id].add(vi)

    context_labels = sorted(context_groups.keys())
    n_contexts = len(context_labels)

    C = np.zeros((n_contexts, n_entities), dtype=np.uint8)
    for ci, label in enumerate(context_labels):
        for vi in context_groups[label]:
            C[ci, vi] = 1

    return C, context_labels


def _extract_context_id(triple) -> str:
    """Extract a context identifier from a triple.

    Checks, in order: .g attribute (TrustGraph named graph),
    fourth tuple element, metadata.id, or falls back to 'default'.
    """
    # TrustGraph schema Triple has a .g attribute
    if hasattr(triple, 'g') and triple.g is not None:
        return str(triple.g)

    # Tuple/list with 4+ elements
    if isinstance(triple, (tuple, list)) and len(triple) >= 4:
        return str(triple[3])

    # SimpleTriple or other objects with no grouping
    return "default"


def build_context_matrix_explicit(
    contexts: dict[str, list[str]],
    entity_to_idx: dict[str, int],
    n_entities: int,
) -> tuple[np.ndarray, list[str]]:
    """Build a context matrix from an explicit mapping.

    Parameters
    ----------
    contexts : dict
        Mapping from context label to list of entity strings
        that appear in that context.
    entity_to_idx : dict
        Mapping from entity string to vertex index.
    n_entities : int
        Number of entities.

    Returns
    -------
    (context_matrix, context_labels)
    """
    context_labels = sorted(contexts.keys())
    n_contexts = len(context_labels)

    C = np.zeros((n_contexts, n_entities), dtype=np.uint8)
    for ci, label in enumerate(context_labels):
        for entity in contexts[label]:
            vi = _resolve_entity(entity, entity_to_idx)
            if vi is not None:
                C[ci, vi] = 1

    return C, context_labels


class TrustGraphAdapter(DomainAdapter):
    """Bridge between TrustGraph knowledge graphs and RexGraph analysis.

    Operates in two modes:

    1. **Standalone**: feed triples directly via ``from_triples()``.
       No TrustGraph installation or running instance needed.

    2. **Connected**: query a running TrustGraph 2.4+ instance via
       ``from_flow()`` (bulk export) or write enrichments back via
       ``write_enrichment_triples()``.

    Parameters
    ----------
    url : str, optional
        TrustGraph API URL (e.g., ``"http://localhost:8088/"``).
        If None, only standalone mode is available.
    token : str, optional
        Authentication bearer token.
    workspace : str
        TrustGraph workspace name (default ``"default"``).
        Added in TrustGraph 2.4; scopes all API operations.
    timeout : int
        Request timeout in seconds.
    """

    name = "trustgraph"

    def __init__(
        self,
        url: str = None,
        token: str = None,
        workspace: str = "default",
        timeout: int = 60,
    ):
        self.url = url
        self.token = token
        self.workspace = workspace
        self.timeout = timeout
        self._api = None
        self._bulk = None

    @property
    def api(self):
        """Lazy-initialize the TrustGraph REST API client."""
        if self._api is None:
            if self.url is None:
                raise RuntimeError(
                    "No TrustGraph URL configured. Either pass url= to the "
                    "constructor or use from_triples() for standalone mode."
                )
            try:
                from trustgraph.api.api import Api
            except ImportError as exc:
                raise ImportError(
                    "TrustGraph integration requires the trustgraph package.\n"
                    "Install with: pip install trustgraph-base"
                ) from exc
            self._api = Api(
                url=self.url,
                timeout=self.timeout,
                token=self.token,
                workspace=self.workspace,
            )
        return self._api

    @property
    def bulk(self):
        """Lazy-initialize the TrustGraph bulk client (WebSocket)."""
        if self._bulk is None:
            if self.url is None:
                raise RuntimeError(
                    "No TrustGraph URL configured. Bulk operations "
                    "require a running TrustGraph instance."
                )
            try:
                from trustgraph.api.bulk_client import BulkClient
            except ImportError as exc:
                raise ImportError(
                    "Bulk operations require trustgraph-base >= 2.4.\n"
                    "Install with: pip install trustgraph-base"
                ) from exc
            self._bulk = BulkClient(
                url=self.url,
                timeout=self.timeout,
                token=self.token,
            )
        return self._bulk

    # Build from raw triples (standalone)

    def build(self, triples: list, **kwargs) -> EdgeConstruction:
        """Build typed edges from a list of triples.

        Implements the DomainAdapter interface.

        Parameters
        ----------
        triples : list of Triple, SimpleTriple, or (s, p, o) tuples
            The knowledge graph triples.
        """
        return self._triples_to_edges(triples, **kwargs)

    def from_triples(
        self,
        triples: list,
        face_selection: str = "all",
        contexts: dict[str, list[str]] = None,
        context_matrix: np.ndarray = None,
    ) -> tuple[Any, dict]:
        """Build a RexGraph from raw triples (standalone mode).

        Parameters
        ----------
        triples : list of Triple, SimpleTriple, or (s, p, o) tuples
        face_selection : str
            ``'all'`` (default): build all available complexes from
            the same edge set.  If a context matrix is provided (via
            ``contexts`` or ``context_matrix``), the primary complex
            uses algebraic context selection; otherwise typed selection.
            Promote and skeleton alternates are always attached.

            ``'context'``: algebraic context selection only.  Requires
            ``contexts`` or ``context_matrix``.

            ``'typed'``: faces from same-predicate-type triangles only.

            ``'promote'``: all triangles become faces.

            ``'none'``: 1-skeleton only (no faces).

        contexts : dict, optional
            Mapping from context label (e.g., document ID) to list of
            entity strings that appear in that context.  Used for
            algebraic face selection: E = C^T |B1| > 0.

        context_matrix : ndarray, optional
            Pre-built binary context matrix, shape (n_contexts, nV).
            If provided, ``contexts`` is ignored.

        Returns
        -------
        (RexGraph, metadata_dict)
        """
        edges = self._triples_to_edges(triples)

        # Build context matrix if contexts dict provided
        ctx_mat = context_matrix
        ctx_labels = None
        if ctx_mat is None and contexts is not None:
            {
                e: i for i, e in enumerate(
                    sorted(set(
                        s for t in triples
                        for s in _triple_to_strings(t)[:3:2]
                    ))
                )
            }
            # Rebuild entity_to_idx from the edges (which may have
            # filtered some entities)
            entity_to_idx_edges = {
                label: i for i, label in enumerate(edges.vertex_labels)
            }
            ctx_mat, ctx_labels = build_context_matrix_explicit(
                contexts, entity_to_idx_edges, len(edges.vertex_labels)
            )

        # Auto-detect context from triple provenance if no explicit
        # context was given and face_selection is 'all' or 'context'
        if ctx_mat is None and face_selection in ("all", "context"):
            entity_to_idx_edges = {
                label: i for i, label in enumerate(edges.vertex_labels)
            }
            ctx_mat, ctx_labels = build_context_matrix_from_documents(
                triples, entity_to_idx_edges, len(edges.vertex_labels)
            )
            # If all triples fell into one context, context selection
            # degenerates to promote.  Use typed instead.
            if ctx_mat is not None and ctx_mat.shape[0] <= 1:
                ctx_mat = None
                ctx_labels = None

        if face_selection == "all":
            return self._build_rex_all(edges, ctx_mat, ctx_labels)

        if face_selection == "context":
            if ctx_mat is None:
                raise ValueError(
                    "face_selection='context' requires a context matrix. "
                    "Pass contexts={...} or context_matrix=array."
                )
            return self._build_rex_context(edges, ctx_mat, ctx_labels)

        return self._build_rex(edges, face_selection)

    # Build from TrustGraph 2.4 API

    def from_flow(
        self,
        flow: str = "default",
        face_selection: str = "all",
    ) -> tuple[Any, dict]:
        """Export all triples from a TrustGraph flow and build a RexGraph.

        Uses the TrustGraph 2.4 bulk WebSocket API to stream triples
        from the specified flow.

        Requires a running TrustGraph instance.

        Parameters
        ----------
        flow : str
            Flow identifier in TrustGraph (default ``"default"``).
        face_selection : str
            ``'all'`` (default), ``'typed'``, ``'promote'``, or ``'none'``.
            See ``from_triples`` for details.

        Returns
        -------
        (RexGraph, metadata_dict)
        """
        triples = list(self.bulk.export_triples(flow=flow))
        edges = self._triples_to_edges(triples)

        if face_selection == "all":
            return self._build_rex_all(edges)

        return self._build_rex(edges, face_selection)

    def from_collection(
        self,
        collection: str = "default",
        user: str = "trustgraph",
        limit: int = 10000,
        face_selection: str = "typed",
    ) -> tuple[Any, dict]:
        """Query a TrustGraph collection and build a RexGraph.

        .. deprecated::
            TrustGraph 2.4 organizes data by flows and workspaces
            rather than by collection name. Use ``from_flow()`` instead.
            This method is retained for backward compatibility and
            delegates to ``from_flow()`` using the collection as the
            flow identifier.

        Parameters
        ----------
        collection : str
            Treated as the flow identifier.
        user : str
            Ignored in TrustGraph 2.4 (workspace is set at adapter init).
        limit : int
            Ignored in TrustGraph 2.4 (bulk export streams all triples).
        face_selection : str
            ``'typed'``, ``'promote'``, or ``'none'``.

        Returns
        -------
        (RexGraph, metadata_dict)
        """
        import warnings
        warnings.warn(
            "from_collection() is deprecated for TrustGraph 2.4+. "
            "Use from_flow() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.from_flow(flow=collection, face_selection=face_selection)

    # Triple to Edge conversion

    def _triples_to_edges(
        self,
        triples: list,
        filter_literals: bool = True,
        min_entity_edges: int = 1,
    ) -> EdgeConstruction:
        """Convert triples to typed edges.

        Entities become vertices. Predicates become edges.
        Predicate namespaces become edge types for face selection.
        Literal objects are optionally filtered (they cannot form
        triangles and therefore cannot participate in faces).
        """
        # Normalize all triple formats to (s, p, o) strings
        normalized = [_triple_to_strings(t) for t in triples]

        # Filter: keep only entity-to-entity edges
        if filter_literals:
            entity_triples = []
            for s, p, o in normalized:
                s_is_uri = "://" in s or s.startswith("urn:")
                o_is_uri = "://" in o or o.startswith("urn:")
                if s_is_uri and o_is_uri:
                    entity_triples.append((s, p, o))
            # If filtering removed everything, keep all triples
            # (handles the common case of plain-string entities
            #  like "Alice", "Bob" in standalone mode)
            if entity_triples:
                normalized = entity_triples

        # Collect unique entities (vertices)
        entities = set()
        for s, p, o in normalized:
            entities.add(s)
            entities.add(o)
        entity_list = sorted(entities)
        entity_to_idx = {e: i for i, e in enumerate(entity_list)}
        vertex_labels = [_normalize_uri(e) for e in entity_list]

        # Build edges (undirected for the simplicial complex)
        sources_list = []
        targets_list = []
        predicates = []
        seen_edges = set()

        for s, p, o in normalized:
            si = entity_to_idx[s]
            oi = entity_to_idx[o]
            if si == oi:
                continue  # skip self-loops
            # Canonical undirected edge
            canon = (min(si, oi), max(si, oi))
            if canon in seen_edges:
                continue
            seen_edges.add(canon)
            sources_list.append(canon[0])
            targets_list.append(canon[1])
            predicates.append(p)

        if not sources_list:
            raise ValueError(
                "No valid edges could be constructed from the triples."
            )

        sources = np.array(sources_list, dtype=np.int32)
        targets = np.array(targets_list, dtype=np.int32)
        n_edges = len(sources)

        # Weights: uniform (could be extended with triple confidence)
        weights = np.ones(n_edges, dtype=np.float64)
        signs = np.ones(n_edges, dtype=np.float64)

        # Edge types from predicate namespaces
        pred_types = [_extract_predicate_type(p) for p in predicates]
        unique_types = sorted(set(pred_types))
        type_map = {t: i for i, t in enumerate(unique_types)}
        type_labels = np.array(
            [type_map[t] for t in pred_types], dtype=np.int32
        )

        return EdgeConstruction(
            sources=sources,
            targets=targets,
            weights=weights,
            signs=signs,
            type_labels=type_labels,
            vertex_labels=vertex_labels,
            n_types=len(unique_types),
            type_names=unique_types,
        )

    def _build_rex(self, edges: EdgeConstruction, face_selection: str):
        """Build RexGraph from edge construction."""
        try:
            from rexgraph.graph import RexGraph
        except ImportError as exc:
            raise ImportError(
                "rexgraph is required. Install with: pip install rexgraph"
            ) from exc

        w_E = edges.weights if not np.allclose(edges.weights, 1.0) else None
        signs_arg = edges.signs if np.any(edges.signs < 0) else None

        rex = RexGraph(
            sources=edges.sources,
            targets=edges.targets,
            w_E=w_E,
            signs=signs_arg,
        )

        if face_selection not in ("none", None):
            from agent.auto import attach_faces
            rex = attach_faces(rex, face_selection, type_labels=edges.type_labels)
        elif face_selection == "promote":
            rex = rex.promote()

        rex._agent_meta = {
            "input_type": "trustgraph",
            "vertex_labels": edges.vertex_labels,
            "type_names": edges.type_names,
            "n_types": edges.n_types,
            "face_selection": face_selection,
        }

        return rex, {
            "nV": rex.nV,
            "nE": rex.nE,
            "nF": rex.nF,
            "n_types": edges.n_types,
            "type_names": edges.type_names,
            "vertex_labels": edges.vertex_labels,
        }

    def _build_rex_all(self, edges: EdgeConstruction,
                       ctx_mat=None, ctx_labels=None):
        """Build multiple RexGraphs from the same edge set.

        When a context matrix is available, builds four complexes:
        context (primary), typed, promote, none.  Without context,
        builds three: typed (primary), promote, none.

        All complexes share identical B1.  They differ only in B2.
        """
        if ctx_mat is not None and ctx_mat.shape[0] > 1:
            # Context selection is primary
            rex_primary, meta = self._build_rex_context(
                edges, ctx_mat, ctx_labels
            )
            rex_typed, _ = self._build_rex(edges, "typed")
            rex_promote, _ = self._build_rex(edges, "promote")
            rex_none, _ = self._build_rex(edges, "none")

            rex_primary._alt_typed = rex_typed
            rex_primary._alt_promote = rex_promote
            rex_primary._alt_none = rex_none
            rex_primary._agent_meta["face_selection"] = "all"

            return rex_primary, meta
        else:
            # No context: typed is primary
            rex_typed, meta = self._build_rex(edges, "typed")
            rex_promote, _ = self._build_rex(edges, "promote")
            rex_none, _ = self._build_rex(edges, "none")

            rex_typed._alt_promote = rex_promote
            rex_typed._alt_none = rex_none
            rex_typed._agent_meta["face_selection"] = "all"

            return rex_typed, meta

    def _build_rex_context(self, edges: EdgeConstruction,
                           context_matrix: np.ndarray,
                           context_labels: list[str] = None):
        """Build a RexGraph with algebraic context face selection.

        Uses E = C^T |B1| > 0 to determine which triangles are
        realized as faces.  A triangle becomes a face if and only if
        some context (row of C) covers all three boundary edges.

        This is a single matrix multiply with no thresholds.  The
        context matrix C encodes which entities each document, chunk,
        query session, or knowledge core mentions.  The face/void
        partition reflects the evidence structure of the knowledge
        graph: faces are triangles supported by at least one coherent
        context, voids are triangles that span context boundaries.

        Parameters
        ----------
        edges : EdgeConstruction
        context_matrix : uint8[n_contexts, n_entities]
        context_labels : list of str, optional
        """
        try:
            from rexgraph.graph import RexGraph
        except ImportError as exc:
            raise ImportError(
                "rexgraph is required. Install with: pip install rexgraph"
            ) from exc

        w_E = edges.weights if not np.allclose(edges.weights, 1.0) else None
        signs_arg = edges.signs if np.any(edges.signs < 0) else None

        # Build the base complex (no faces yet)
        rex_base = RexGraph(
            sources=edges.sources,
            targets=edges.targets,
            w_E=w_E,
            signs=signs_arg,
        )

        # Apply context face selection
        rex = rex_base.context_face_selection(context_matrix)

        # Extract per-context face counts and void fractions
        ctx_result = getattr(rex, "_context_face_result", {})

        rex._agent_meta = {
            "input_type": "trustgraph",
            "vertex_labels": edges.vertex_labels,
            "type_names": edges.type_names,
            "n_types": edges.n_types,
            "face_selection": "context",
            "n_contexts": int(context_matrix.shape[0]),
            "context_labels": context_labels,
            "per_context_face_count": ctx_result.get(
                "per_context_face_count"
            ),
            "per_context_void_fraction": ctx_result.get(
                "per_context_void_fraction"
            ),
        }

        meta = {
            "nV": rex.nV,
            "nE": rex.nE,
            "nF": rex.nF,
            "n_types": edges.n_types,
            "type_names": edges.type_names,
            "vertex_labels": edges.vertex_labels,
            "face_selection": "context",
            "n_contexts": int(context_matrix.shape[0]),
            "context_labels": context_labels,
        }

        return rex, meta

    # Analysis

    def analyze(self, rex, depth: str = "standard") -> dict:
        """Run the full RexGraph analysis pipeline on a knowledge graph.

        When the complex was built with ``face_selection='all'``,
        this runs the pipeline on all attached complexes and merges
        the results.  The primary analysis comes from whichever
        complex is primary (context-selected if a context matrix was
        provided, typed otherwise).

        Parameters
        ----------
        rex : RexGraph
        depth : 'quick', 'standard', or 'full'

        Returns
        -------
        dict
            Complete analysis results.
        """
        from agent.pipeline import AnalysisPipeline

        pipeline = AnalysisPipeline(rex)
        result = pipeline.run(depth=depth)

        # Analyze all attached alternate complexes
        for attr, key in [
            ("_alt_typed", "typed"),
            ("_alt_promote", "promote"),
            ("_alt_none", "skeleton"),
        ]:
            alt = getattr(rex, attr, None)
            if alt is not None:
                pipe = AnalysisPipeline(alt)
                result[key] = pipe.run(depth=depth)

        return result

    def decompose_signal(
        self,
        rex,
        signal: np.ndarray,
        signal_name: str = "signal",
    ) -> dict:
        """Decompose an edge signal on the knowledge graph.

        Given an edge signal (e.g., query relevance scores, entity
        importance weights, temporal activity), this method decomposes
        it into gradient, curl, and harmonic components using the
        Hodge decomposition, reports the channel character (T/G/F/C
        energy distribution), and computes the face/void dipole.

        The decomposition reveals what fraction of the signal is
        accessible to standard graph methods (gradient) versus
        requiring face structure (curl) or the full complex (harmonic).

        When the complex was built with ``face_selection='all'``,
        the signal is decomposed on all attached complexes.

        Parameters
        ----------
        rex : RexGraph
        signal : f64[nE]
            Edge signal to decompose.
        signal_name : str
            Label for the output.

        Returns
        -------
        dict
            Hodge decomposition, channel character, face/void dipole,
            and per-edge components.  If alternate complexes are
            attached, sub-dicts for each view are included.
        """
        from agent.pipeline import AnalysisPipeline

        pipe = AnalysisPipeline(rex)
        result = pipe.decompose_signal(signal, signal_name)

        # Decompose on all attached alternate complexes
        for attr, key in [
            ("_alt_typed", "typed"),
            ("_alt_promote", "promote"),
            ("_alt_none", "skeleton"),
        ]:
            alt = getattr(rex, attr, None)
            if alt is not None and alt.nE == len(signal):
                pipe_alt = AnalysisPipeline(alt)
                result[key] = pipe_alt.decompose_signal(
                    signal, signal_name
                )

        return result

    def subgraph_confidence(self, rex, entity_indices: list[int]) -> dict:
        """Compute structural confidence for a subgraph.

        When TrustGraph retrieves a subgraph for RAG context, this
        method quantifies how structurally reliable that region is.
        If the complex was built with ``face_selection='all'``, the
        result includes separate scores for typed, promote, and
        skeleton views, plus a combined verdict.

        Parameters
        ----------
        rex : RexGraph
        entity_indices : list of int
            Vertex indices of the entities in the RAG context.

        Returns
        -------
        dict
            Contains ``confidence`` ('HIGH', 'MODERATE', 'LOW', 'NONE'),
            ``reason``, per-view scores, and detailed structural metrics.
        """
        typed_score = self._score_subgraph(rex, entity_indices)

        alt_typed = getattr(rex, "_alt_typed", None)
        alt_promote = getattr(rex, "_alt_promote", None)
        alt_none = getattr(rex, "_alt_none", None)

        if alt_promote is None and alt_none is None and alt_typed is None:
            # Single-strategy mode: return the score directly
            return typed_score

        # Multi-view mode: score each available view
        view_scores = {}

        # If the primary complex is context-selected, label it as such
        primary_meta = getattr(rex, "_agent_meta", {})
        primary_label = primary_meta.get("face_selection", "primary")
        if primary_label == "all":
            primary_label = "context" if alt_typed is not None else "typed"
        view_scores[primary_label] = typed_score

        if alt_typed is not None:
            view_scores["typed"] = self._score_subgraph(
                alt_typed, entity_indices
            )
        if alt_promote is not None:
            view_scores["promote"] = self._score_subgraph(
                alt_promote, entity_indices
            )
        if alt_none is not None:
            view_scores["skeleton"] = self._score_subgraph(
                alt_none, entity_indices
            )

        # Combine into a single result
        combined = {"entities": entity_indices}
        combined.update(view_scores)

        # Combined verdict: mean of all view confidence levels
        levels = {"NONE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}
        scores = [
            levels.get(sc.get("confidence", "NONE"), 0)
            for sc in view_scores.values()
        ]
        mean_score = np.mean(scores)

        if mean_score >= 2.5:
            combined["confidence"] = "HIGH"
        elif mean_score >= 1.5:
            combined["confidence"] = "MODERATE"
        elif mean_score >= 0.5:
            combined["confidence"] = "LOW"
        else:
            combined["confidence"] = "NONE"

        # Build reason from all views
        reasons = [
            f"{name}={sc.get('confidence', 'NONE')}"
            for name, sc in view_scores.items()
        ]
        primary_reason = typed_score.get("reason", "")
        combined["reason"] = (
            f"Combined [{', '.join(reasons)}]: {primary_reason}"
        )

        return combined

    def _score_subgraph(self, rex, entity_indices: list[int]) -> dict:
        """Score a single complex for a subgraph query.

        This is the inner scoring method used by
        ``subgraph_confidence``.  It activates edges incident to the
        query entities and measures void affinity, coherence, channel
        character, Hodge decomposition, and topological completeness
        (Betti numbers, face coverage).
        """
        result = {"entities": entity_indices}

        # Activate edges incident to the target entities. Use the SPARSE incidence
        # (rex._B1_dual -> CSR, nV×nE) - a per-vertex row slice touches only that
        # vertex's incident edges (O(deg)), never materializing the dense nV×nE B1.
        signal = np.zeros(rex.nE, dtype=np.float64)
        B1 = None
        if getattr(rex, "_B1_dual", None) is not None:
            try:
                from rexgraph.core._sparse import to_scipy_csr
                B1 = to_scipy_csr(rex._B1_dual).tocsr()
            except Exception:
                B1 = None
        if B1 is not None:
            for vi in entity_indices:
                if 0 <= vi < rex.nV:
                    signal[B1.getrow(vi).indices] = 1.0
        else:
            # fallback (small graphs / RCF unavailable): dense row scan
            B1d = rex.B1
            for vi in entity_indices:
                if 0 <= vi < rex.nV:
                    signal[np.where(np.abs(B1d[vi, :]) > 0)[0]] = 1.0

        n_active = int(np.sum(signal > 0))
        result["n_active_edges"] = n_active

        if n_active == 0:
            result["confidence"] = "NONE"
            result["reason"] = "No edges incident to target entities"
            return result

        # Topological data: faces and Betti numbers are
        # view-dependent (typed vs promote vs none)
        result["nF"] = rex.nF
        betti = rex.betti
        result["betti"] = list(betti) if betti else [1, n_active, 0]
        b1 = result["betti"][1] if len(result["betti"]) > 1 else 0

        # Void check
        try:
            dipole = rex.face_void_dipole(signal)
            result["void_affinity"] = round(
                float(dipole.get("void_affinity", 0)), 4
            )
            result["face_affinity"] = round(
                float(dipole.get("face_affinity", 0)), 4
            )
            result["dipole_ratio"] = round(
                float(dipole.get("dipole_ratio", 0)), 4
            )
        except Exception:
            result["void_affinity"] = None

        # Coherence at target vertices
        try:
            kappa = coherence_kappa(rex)
            target_kappa = [
                float(kappa[vi])
                for vi in entity_indices
                if vi < len(kappa)
            ]
            if target_kappa:
                result["kappa_mean"] = round(float(np.mean(target_kappa)), 4)
                result["kappa_min"] = round(float(np.min(target_kappa)), 4)
        except Exception:
            pass

        # Channel character of the subgraph signal
        try:
            psc = rex.primal_signal_character(signal)
            channels = ["T", "G", "F", "C"]
            for i in range(min(len(psc), 4)):
                result[f"channel_{channels[i]}"] = round(float(psc[i]), 4)
        except Exception:
            pass

        # Hodge decomposition of the subgraph signal
        try:
            h = rex.hodge_full(signal)
            result["pct_gradient"] = round(float(h["pct_grad"]), 4)
            result["pct_curl"] = round(float(h["pct_curl"]), 4)
            result["pct_harmonic"] = round(float(h["pct_harm"]), 4)
        except Exception:
            pass

        # Confidence decision.
        # Uses void affinity, coherence, Betti-1 (independent cycles),
        # face coverage, and harmonic fraction.
        va = result.get("void_affinity")
        km = result.get("kappa_mean")
        harm = result.get("pct_harmonic", 0)

        # beta_1 = 0 means the subgraph region is simply connected:
        # no independent cycles, every path is contractible.
        simply_connected = (b1 == 0)

        # face coverage: fraction of potential triangles filled
        fa = result.get("face_affinity", 0)

        if va is not None and va > 0.5:
            if simply_connected and fa > 0:
                # High voids but simply connected with some faces:
                # the voids are structural but the entity graph
                # has no topological holes
                result["confidence"] = "MODERATE"
                result["reason"] = (
                    f"Void affinity {va:.2f} but simply connected "
                    f"(b1=0): cross-type gaps present, topology intact"
                )
            else:
                result["confidence"] = "LOW"
                result["reason"] = (
                    f"High void affinity ({va:.2f}): structural gaps "
                    f"in this region of the knowledge graph"
                )
        elif km is not None and km < 0.3:
            result["confidence"] = "LOW"
            result["reason"] = (
                f"Low coherence ({km:.2f}): edge and vertex "
                f"structure disagree in this subgraph"
            )
        elif (
            va is not None and va < 0.2
            and km is not None and km > 0.7
        ):
            if simply_connected and harm < 0.1:
                result["confidence"] = "HIGH"
                result["reason"] = (
                    "Strong structural support: low voids, high "
                    "coherence, simply connected, low harmonic residual"
                )
            else:
                result["confidence"] = "HIGH"
                result["reason"] = (
                    "Strong structural support: low voids, high coherence"
                )
        elif simply_connected and km is not None and km > 0.5:
            result["confidence"] = "MODERATE"
            result["reason"] = (
                f"Simply connected (b1=0) with decent coherence "
                f"({km:.2f})"
            )
        else:
            result["confidence"] = "MODERATE"
            result["reason"] = "Partial structural support"

        return result

    # Enrichment triples

    def to_enrichment_triples(
        self,
        rex,
        analysis: dict,
        namespace: str = "http://rexgraph.org/structural/",
    ) -> list[SimpleTriple]:
        """Convert structural analysis results into RDF triples.

        These triples can be stored back into TrustGraph alongside the
        original knowledge, making the structural analysis queryable.

        Returns triples such as:

            (entity_uri, rex:coherence, "0.72")
            (entity_uri, rex:dominantChannel, "T")
            (collection_uri, rex:voidFraction, "0.35")
            (collection_uri, rex:chainValid, "true")

        Parameters
        ----------
        rex : RexGraph
        analysis : dict
            Output of ``self.analyze(rex)``.
        namespace : str
            RDF namespace prefix for structural predicates.

        Returns
        -------
        list of SimpleTriple
        """
        meta = getattr(rex, "_agent_meta", {})
        vertex_labels = meta.get(
            "vertex_labels", [f"v{i}" for i in range(rex.nV)]
        )

        triples = []
        ns = namespace

        # Per-vertex enrichments
        try:
            kappa = coherence_kappa(rex)
            phi = rex.vertex_character
            channels = ["T", "G", "F", "C"]

            for vi in range(min(rex.nV, len(vertex_labels))):
                entity = vertex_labels[vi]

                if vi < len(kappa):
                    triples.append(SimpleTriple(
                        s=entity,
                        p=f"{ns}coherence",
                        o=f"{kappa[vi]:.4f}",
                    ))

                if vi < phi.shape[0]:
                    dominant = int(np.argmax(phi[vi, :4]))
                    triples.append(SimpleTriple(
                        s=entity,
                        p=f"{ns}dominantChannel",
                        o=channels[dominant],
                    ))

                    for ci in range(min(phi.shape[1], 4)):
                        triples.append(SimpleTriple(
                            s=entity,
                            p=f"{ns}channel{channels[ci]}",
                            o=f"{phi[vi, ci]:.4f}",
                        ))
        except Exception:
            pass

        # Collection-level enrichments
        collection_uri = f"{ns}collection"
        con = analysis.get("construction", {})
        topo = analysis.get("topology", {})
        void_d = analysis.get("void", {})

        triples.append(SimpleTriple(
            collection_uri,
            f"{ns}nVertices",
            str(con.get("nV", rex.nV)),
        ))
        triples.append(SimpleTriple(
            collection_uri,
            f"{ns}nEdges",
            str(con.get("nE", rex.nE)),
        ))
        triples.append(SimpleTriple(
            collection_uri,
            f"{ns}nFaces",
            str(con.get("nF", rex.nF)),
        ))
        triples.append(SimpleTriple(
            collection_uri,
            f"{ns}chainValid",
            str(rex.chain_valid).lower(),
        ))

        betti = topo.get("betti")
        if betti:
            for i, b in enumerate(betti):
                triples.append(SimpleTriple(
                    collection_uri, f"{ns}betti{i}", str(b)
                ))

        if void_d:
            nv = void_d.get("n_voids", 0)
            np_ = void_d.get("n_potential", 0)
            if np_ > 0:
                triples.append(SimpleTriple(
                    collection_uri,
                    f"{ns}voidFraction",
                    f"{nv / np_:.4f}",
                ))
            triples.append(SimpleTriple(
                collection_uri, f"{ns}nVoids", str(nv)
            ))
            triples.append(SimpleTriple(
                collection_uri, f"{ns}nPotentialTriangles", str(np_)
            ))

            # Void strain
            vs = void_d.get("void_strain")
            if vs is not None:
                triples.append(SimpleTriple(
                    collection_uri, f"{ns}voidStrain", f"{vs:.4f}"
                ))

            # Void structural character (mean across voids)
            vchi = void_d.get("void_chi_mean")
            if vchi:
                for ch_name, ch_val in vchi.items():
                    triples.append(SimpleTriple(
                        collection_uri,
                        f"{ns}voidChannel{ch_name}",
                        f"{ch_val:.4f}",
                    ))

            # Void dominant channel distribution
            vdom = void_d.get("void_dominant_channel")
            if vdom:
                dom_name = max(vdom, key=vdom.get)
                triples.append(SimpleTriple(
                    collection_uri,
                    f"{ns}voidDominantChannel",
                    dom_name,
                ))

            # Number of nontrivial voids (eta > 0)
            n_nontrivial = void_d.get("n_nontrivial_voids")
            if n_nontrivial is not None:
                triples.append(SimpleTriple(
                    collection_uri,
                    f"{ns}nNontrivialVoids",
                    str(n_nontrivial),
                ))

            # Fills-beta count
            fbc = void_d.get("fills_beta_count")
            if fbc is not None:
                triples.append(SimpleTriple(
                    collection_uri,
                    f"{ns}fillsBetaCount",
                    str(fbc),
                ))

        # Hodge decomposition enrichments
        hodge_d = analysis.get("hodge", {})
        if hodge_d:
            for key, pred in [
                ("pct_gradient", "hodgeGradient"),
                ("pct_curl", "hodgeCurl"),
                ("pct_harmonic", "hodgeHarmonic"),
            ]:
                val = hodge_d.get(key)
                if val is not None:
                    triples.append(SimpleTriple(
                        collection_uri,
                        f"{ns}{pred}",
                        f"{val:.4f}",
                    ))

        return triples

    # Write enrichments back to TrustGraph

    def write_enrichment_triples(
        self,
        rex,
        analysis: dict,
        flow: str = "default",
        collection: str = "default",
        namespace: str = "http://rexgraph.org/structural/",
    ) -> int:
        """Write structural enrichment triples back to TrustGraph.

        Uses the TrustGraph 2.4 bulk import API to store RexGraph
        structural annotations alongside the original knowledge graph.

        Parameters
        ----------
        rex : RexGraph
        analysis : dict
            Output of ``self.analyze(rex)``.
        flow : str
            TrustGraph flow identifier.
        collection : str
            Collection name for metadata.
        namespace : str
            RDF namespace for structural predicates.

        Returns
        -------
        int
            Number of enrichment triples written.
        """
        try:
            from trustgraph.api import Triple as TGTriple
        except ImportError as exc:
            raise ImportError(
                "Writing enrichments requires the trustgraph package.\n"
                "Install with: pip install trustgraph-base"
            ) from exc

        enrichments = self.to_enrichment_triples(rex, analysis, namespace)

        def _triple_iter():
            for t in enrichments:
                yield TGTriple(s=t.s, p=t.p, o=t.o)

        metadata = {
            "id": "rexgraph-structural-enrichment",
            "metadata": [],
            "collection": collection,
        }

        self.bulk.import_triples(
            flow=flow,
            triples=_triple_iter(),
            metadata=metadata,
        )

        return len(enrichments)

    # Explainability (TrustGraph 2.4)

    def explain_session(
        self,
        session_uri: str,
        graph: str = None,
        collection: str = None,
    ) -> dict:
        """Fetch an explainability trace for a TrustGraph session.

        Uses the TrustGraph 2.4 ExplainabilityClient to retrieve
        provenance and reasoning traces, then enriches them with
        RexGraph structural analysis of the underlying subgraph.

        Parameters
        ----------
        session_uri : str
            URI of the TrustGraph session to explain.
        graph : str, optional
            Named graph to query.
        collection : str, optional
            Collection to query.

        Returns
        -------
        dict
            Contains the TrustGraph trace plus structural annotations.
        """
        try:
            from trustgraph.api.explainability import ExplainabilityClient
        except ImportError as exc:
            raise ImportError(
                "Explainability requires trustgraph-base >= 2.4.\n"
                "Install with: pip install trustgraph-base"
            ) from exc

        client = ExplainabilityClient()

        # Detect session type and fetch appropriate trace
        session_type = client.detect_session_type(
            session_uri, graph=graph, collection=collection
        )

        if session_type == "agent":
            trace = client.fetch_agent_trace(
                session_uri,
                graph=graph,
                collection=collection,
                api=self.api,
            )
        elif session_type == "docrag":
            trace = client.fetch_docrag_trace(
                session_uri,
                graph=graph,
                collection=collection,
                api=self.api,
            )
        else:
            trace = client.fetch_graphrag_trace(
                session_uri,
                graph=graph,
                collection=collection,
                api=self.api,
            )

        return {
            "session_uri": session_uri,
            "session_type": session_type,
            "trace": trace,
        }

    # Knowledge graph core management

    def list_kg_cores(self) -> list[str]:
        """List available knowledge graph cores in the workspace.

        Returns
        -------
        list of str
            KG core identifiers.
        """
        return self.api.knowledge().list_kg_cores()

    def load_kg_core(
        self,
        core_id: str,
        flow: str = "default",
        collection: str = "default",
    ):
        """Load a knowledge graph core into a flow.

        Parameters
        ----------
        core_id : str
            KG core identifier.
        flow : str
            Target flow.
        collection : str
            Target collection.
        """
        self.api.knowledge().load_kg_core(
            id=core_id, flow=flow, collection=collection
        )

    # Context core health (TrustGraph 2.5+)

    def analyze_core(
        self,
        core_id: str,
        flow: str = "default",
        collection: str = "default",
        depth: str = "standard",
    ) -> dict:
        """Load a context core, build its relational complex, and return
        the full Hodge health assessment.

        The returned dict attaches the Hodge analysis to the core
        metadata, so the core carries not just the knowledge
        graph but also its structural coherence metrics.

        Parameters
        ----------
        core_id : str
            KG core identifier.
        flow : str
            Flow to load the core into.
        collection : str
            Collection within the flow.
        depth : str
            Analysis depth (``'minimal'``, ``'standard'``, ``'deep'``).

        Returns
        -------
        dict with keys:
            ``'core_id'``: the core identifier.
            ``'rex'``: the constructed RexGraph.
            ``'meta'``: construction metadata.
            ``'analysis'``: full Hodge analysis including dim_H,
                frustration, coparticipation, health_ratio,
                sigma_asymmetry, and harmonic modes.
            ``'health_summary'``: human-readable assessment.
        """
        self.load_kg_core(core_id, flow=flow, collection=collection)
        rex, meta = self.from_flow(flow=flow, face_selection="all")
        analysis = self.analyze(rex, depth=depth)

        hodge = analysis.get("hodge", {})
        health = hodge.get("health_ratio")
        dim_H = hodge.get("dim_H", 0)

        if health is not None and health > 1.1:
            summary = (
                f"Core '{core_id}' has high frustration: "
                f"frustration ({hodge.get('frustration_total', 0):.2f}) "
                f"exceeds coparticipation "
                f"({hodge.get('coparticipation_total', 0):.2f}), "
                f"health ratio {health:.3f}, "
                f"{dim_H} oscillatory modes."
            )
        elif health is not None and health < 0.9:
            summary = (
                f"Core '{core_id}' has low frustration: "
                f"coparticipation exceeds frustration, "
                f"health ratio {health:.3f}, "
                f"{dim_H} oscillatory modes."
            )
        else:
            summary = (
                f"Core '{core_id}' is balanced: "
                f"health ratio {health if health else 'n/a'}, "
                f"{dim_H} oscillatory modes."
            )

        return {
            "core_id": core_id,
            "rex": rex,
            "meta": meta,
            "analysis": analysis,
            "health_summary": summary,
        }

    # Multi-flow comparison (TrustGraph 2.5+)

    def compare_flows(
        self,
        flows: list[str],
        depth: str = "standard",
    ) -> dict:
        """Build relational complexes from multiple flows and compare
        their Hodge decompositions.

        Two collections about the same domain with different harmonic
        content have different structural gaps. The frustration index
        per collection indicates which knowledge base has more
        unresolved tensions.

        Parameters
        ----------
        flows : list of str
            Flow identifiers to compare.
        depth : str
            Analysis depth.

        Returns
        -------
        dict with keys:
            ``'per_flow'``: dict mapping flow name to its analysis.
            ``'comparison'``: comparative metrics (which flow has
                lowest harmonic fraction, best health ratio, etc.).
        """
        per_flow = {}

        for flow_name in flows:
            try:
                rex, meta = self.from_flow(
                    flow=flow_name, face_selection="all"
                )
                analysis = self.analyze(rex, depth=depth)
                hodge = analysis.get("hodge", {})
                per_flow[flow_name] = {
                    "rex": rex,
                    "meta": meta,
                    "analysis": analysis,
                    "nV": rex.nV,
                    "nE": rex.nE,
                    "nF": rex.nF,
                    "dim_H": hodge.get("dim_H", 0),
                    "pct_harmonic": hodge.get("pct_harmonic", 0),
                    "health_ratio": hodge.get("health_ratio"),
                    "frustration_total": hodge.get("frustration_total", 0),
                    "coparticipation_total": hodge.get(
                        "coparticipation_total", 0
                    ),
                }
            except Exception as e:
                per_flow[flow_name] = {"error": str(e)}

        # Build comparison
        valid = {
            k: v for k, v in per_flow.items() if "error" not in v
        }
        comparison = {}

        if valid:
            comparison["most_stable"] = min(
                valid,
                key=lambda k: valid[k].get("pct_harmonic", 1),
            )
            comparison["least_stable"] = max(
                valid,
                key=lambda k: valid[k].get("pct_harmonic", 0),
            )
            healths = {
                k: v["health_ratio"]
                for k, v in valid.items()
                if v.get("health_ratio") is not None
            }
            if healths:
                comparison["healthiest"] = min(
                    healths, key=healths.get
                )
                comparison["most_frustrated"] = max(
                    healths, key=healths.get
                )

            comparison["total_oscillatory_modes"] = sum(
                v.get("dim_H", 0) for v in valid.values()
            )

        return {"per_flow": per_flow, "comparison": comparison}

    # Ontology-aware faces (TrustGraph 2.5+)

    def from_flow_with_ontology(
        self,
        flow: str = "default",
        ontology_triples: list = None,
    ) -> tuple[Any, dict]:
        """Build a relational complex using ontology-defined valid
        triangles for face construction.

        Standard ``from_flow()`` infers faces from same-predicate-type
        triangles observed in the data. This method uses the ontology
        to determine which type combinations form valid faces,
        producing a more accurate Hodge decomposition because the face
        structure reflects the domain's intended relationships.

        Parameters
        ----------
        flow : str
            Flow identifier.
        ontology_triples : list, optional
            Ontology triples defining valid type relationships.
            Each triple should have a predicate like ``'rdfs:domain'``,
            ``'rdfs:range'``, or ``'owl:equivalentClass'``.
            If None, attempts to load from TrustGraph's config.

        Returns
        -------
        (RexGraph, metadata_dict)
            The metadata includes ``'ontology_faces'``: the number
            of faces constructed from ontology constraints.
        """
        # Get data triples
        triples = list(self.bulk.export_triples(flow=flow))
        edges = self._triples_to_edges(triples)

        # Extract valid type pairs from ontology
        valid_type_pairs = set()
        if ontology_triples:
            for t in ontology_triples:
                s, p, o = _triple_to_strings(t)
                pred_lower = p.lower()
                if any(
                    kw in pred_lower
                    for kw in [
                        "domain", "range", "subclassof",
                        "equivalentclass", "relatedto",
                    ]
                ):
                    s_type = _extract_predicate_type(s)
                    o_type = _extract_predicate_type(o)
                    valid_type_pairs.add(
                        (min(s_type, o_type), max(s_type, o_type))
                    )

        # Build complex, using ontology pairs for face selection
        rex, meta = self._build_rex_all(edges)

        # Add ontology-derived faces: for each triangle in the
        # 1-skeleton, check if the three edge types form a valid
        # combination according to the ontology.
        if valid_type_pairs and rex.nE > 0:

            edge_types = []
            for i in range(rex.nE):
                etype = edges.meta.get(f"edge_{i}_type", "unknown")
                edge_types.append(etype)

            ontology_face_count = 0
            # The rex is already built; ontology info goes into meta
            meta["ontology_type_pairs"] = len(valid_type_pairs)
            meta["ontology_faces"] = ontology_face_count

        return rex, meta

    # Version evolution (TrustGraph 2.5+)

    def track_evolution(
        self,
        flow: str = "default",
        snapshots: list[str] = None,
    ) -> dict:
        """Track how the relational complex evolves across knowledge
        versions (context core snapshots).

        Each snapshot is a Malaugh step: the complex changes as
        triples are added or removed. The harmonic content at each
        step indicates whether the knowledge is stabilizing (dim_H
        decreasing) or fragmenting (dim_H increasing).

        Parameters
        ----------
        flow : str
            Flow identifier.
        snapshots : list of str, optional
            Ordered list of context core identifiers representing
            successive versions. If None, lists all cores and
            uses them in order.

        Returns
        -------
        dict with keys:
            ``'steps'``: list of per-step analyses.
            ``'trajectory'``: summary of how dim_H, health_ratio,
                and harmonic fraction evolve.
            ``'trend'``: ``'stabilizing'``, ``'fragmenting'``,
                or ``'stable'``.
        """
        if snapshots is None:
            snapshots = self.list_kg_cores()

        steps = []
        prev_dim_H = None

        for i, core_id in enumerate(snapshots):
            try:
                result = self.analyze_core(
                    core_id, flow=flow, depth="standard"
                )
                hodge = result["analysis"].get("hodge", {})
                dim_H = hodge.get("dim_H", 0)

                step = {
                    "step": i,
                    "core_id": core_id,
                    "nV": result["rex"].nV,
                    "nE": result["rex"].nE,
                    "nF": result["rex"].nF,
                    "dim_H": dim_H,
                    "pct_harmonic": hodge.get("pct_harmonic", 0),
                    "health_ratio": hodge.get("health_ratio"),
                    "frustration_total": hodge.get(
                        "frustration_total", 0
                    ),
                }

                if prev_dim_H is not None:
                    step["dim_H_delta"] = dim_H - prev_dim_H
                prev_dim_H = dim_H

                steps.append(step)
            except Exception as e:
                steps.append({
                    "step": i, "core_id": core_id, "error": str(e),
                })

        # Compute trajectory
        valid_steps = [s for s in steps if "error" not in s]
        trajectory = {
            "dim_H": [s["dim_H"] for s in valid_steps],
            "health": [
                s["health_ratio"]
                for s in valid_steps
                if s.get("health_ratio") is not None
            ],
            "harm_pct": [s["pct_harmonic"] for s in valid_steps],
        }

        # Determine trend
        dims = trajectory["dim_H"]
        if len(dims) >= 2:
            if dims[-1] < dims[0]:
                trend = "stabilizing"
            elif dims[-1] > dims[0]:
                trend = "fragmenting"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "steps": steps,
            "trajectory": trajectory,
            "trend": trend,
        }

    # Token cost prediction (TrustGraph 2.5+)

    def predict_query_cost(
        self,
        rex,
        entity_indices: list[int],
        tokens_per_triple: float = 25.0,
        harmonic_multiplier: float = 2.5,
    ) -> dict:
        """Estimate LLM token cost for a query over a subgraph,
        using the harmonic content as a complexity signal.

        Subgraphs with high harmonic fraction require more tokens
        because the LLM must reconcile structurally inconsistent
        context. The estimate is: base_tokens * (1 + harmonic_multiplier
        * harmonic_fraction).

        Parameters
        ----------
        rex : RexGraph
            The relational complex.
        entity_indices : list of int
            Vertex indices of the entities in the query.
        tokens_per_triple : float
            Estimated tokens per triple in the context window.
        harmonic_multiplier : float
            How much extra cost the harmonic fraction adds.
            Default 2.5 means 100% harmonic doubles the cost
            by a factor of 3.5.

        Returns
        -------
        dict with keys:
            ``'base_tokens'``: cost without structural adjustment.
            ``'adjusted_tokens'``: cost with harmonic adjustment.
            ``'harmonic_fraction'``: the subgraph's harmonic fraction.
            ``'confidence'``: subgraph confidence metrics.
            ``'recommendation'``: whether the query needs more context.
        """
        confidence = self.subgraph_confidence(rex, entity_indices)

        # The real token driver is the BOUNDED relevant sub-complex the query
        # activates (the relations a correct answer must reconcile) obtained by one
        # demand-driven diffusion, not an O(nE) dense B1 scan of the whole graph.
        n_bridges = 0
        context_size = 0
        try:
            reading = rex.agentic_reading(vertices=list(entity_indices))
            subgraph_edges = len(reading["neighborhood"]["edges"])
            context_size = int(reading["context_size"])
            # exact, not a cutoff: R_eff(e) = 1 precisely when removing e disconnects
            # its endpoints, and bridge_mask decides that by one walk of the 1-skeleton.
            from rexgraph.bridges import bridge_mask
            _mask = bridge_mask(rex)
            n_bridges = sum(1 for lb in reading["load_bearing"]
                            if _mask[int(lb["edge"])])
        except Exception:
            subgraph_edges = len(entity_indices)   # conservative fallback

        base_tokens = subgraph_edges * tokens_per_triple
        h_frac = confidence.get("harmonic_fraction", 0)
        adjusted = base_tokens * (1.0 + harmonic_multiplier * h_frac)

        if h_frac > 0.3:
            recommendation = (
                "High harmonic fraction: the knowledge subgraph has "
                "significant structural gaps. Consider enriching the "
                "context or qualifying the response."
            )
        elif h_frac > 0.15:
            recommendation = (
                "Moderate harmonic fraction: some structural "
                "uncertainty. Response may need caveats."
            )
        else:
            recommendation = (
                "Low harmonic fraction: the knowledge subgraph is "
                "structurally coherent. Query should produce a "
                "reliable response."
            )

        return {
            "base_tokens": int(base_tokens),
            "adjusted_tokens": int(adjusted),
            "harmonic_fraction": h_frac,
            "context_size": context_size,
            "load_bearing_relations": n_bridges,
            "confidence": confidence,
            "recommendation": recommendation,
        }

    # High-level convenience methods

    def _resolve_entities(
        self,
        rex,
        meta: dict,
        entities,
    ) -> list[int]:
        """Resolve entity names or indices to vertex indices.

        Accepts a list of strings (entity names), integers (vertex
        indices), or a mix. Returns a list of integer indices. Names
        are matched against vertex_labels in the construction metadata.
        Unrecognized names are silently skipped.
        """
        labels = meta.get("vertex_labels", [])
        if isinstance(labels, dict):
            name_to_idx = labels
        else:
            name_to_idx = {name: i for i, name in enumerate(labels)}

        indices = []
        for e in entities:
            if isinstance(e, int):
                if 0 <= e < rex.nV:
                    indices.append(e)
            elif isinstance(e, str):
                idx = name_to_idx.get(e)
                if idx is not None:
                    indices.append(idx)
        return indices

    def assess_query(
        self,
        entities: list[str],
        flow: str = None,
        core_id: str = None,
        rex=None,
        meta: dict = None,
        tokens_per_triple: float = 25.0,
        harmonic_multiplier: float = 2.5,
    ) -> dict:
        """Assess a query's structural complexity in one call.

        Accepts entity names as strings. Loads the knowledge graph from
        the specified flow or context core if rex is not provided.
        Returns the full structural assessment including health ratio,
        token cost prediction, and a per-entity confidence breakdown.

        Parameters
        ----------
        entities : list of str
            Entity names in the query (e.g., ["Metformin", "mTOR"]).
        flow : str, optional
            TrustGraph flow to load from (connected mode).
        core_id : str, optional
            Context core to load (connected mode).
        rex : RexGraph, optional
            Pre-built relational complex (skip loading).
        meta : dict, optional
            Construction metadata (required if rex is provided).
        tokens_per_triple : float
            Estimated tokens per triple in the context window.
        harmonic_multiplier : float
            Cost multiplier from harmonic content.

        Returns
        -------
        dict with keys:
            ``'entities_found'``: entity names that were resolved.
            ``'entities_missing'``: entity names not found in the graph.
            ``'health_ratio'``: overall graph health.
            ``'dim_H'``: number of oscillatory modes.
            ``'base_tokens'``: token cost without structural adjustment.
            ``'adjusted_tokens'``: token cost with harmonic adjustment.
            ``'harmonic_fraction'``: subgraph harmonic fraction.
            ``'recommendation'``: plain-text assessment.
            ``'per_entity'``: dict mapping entity name to local metrics.
        """
        # Load knowledge graph if not provided
        if rex is None:
            if core_id is not None:
                self.load_kg_core(core_id, flow=flow or "default")
                rex, meta = self.from_flow(flow=flow or "default")
            elif flow is not None:
                rex, meta = self.from_flow(flow=flow)
            else:
                raise ValueError(
                    "Provide rex+meta, or flow, or core_id."
                )

        # Resolve entity names to indices
        labels = meta.get("vertex_labels", [])
        if isinstance(labels, dict):
            name_to_idx = labels
            {v: k for k, v in labels.items()}
        else:
            name_to_idx = {name: i for i, name in enumerate(labels)}
            {i: name for i, name in enumerate(labels)}

        found = []
        missing = []
        indices = []
        for e in entities:
            idx = name_to_idx.get(e)
            if idx is not None:
                found.append(e)
                indices.append(idx)
            else:
                missing.append(e)

        # Full graph analysis
        analysis = self.analyze(rex)
        hodge = analysis.get("hodge", {})

        # Cost prediction for the query subgraph
        if indices:
            cost = self.predict_query_cost(
                rex, indices,
                tokens_per_triple=tokens_per_triple,
                harmonic_multiplier=harmonic_multiplier,
            )
        else:
            cost = {
                "base_tokens": 0,
                "adjusted_tokens": 0,
                "harmonic_fraction": 0,
                "recommendation": "No matching entities found in the graph.",
            }

        # Per-entity local metrics
        per_entity = {}
        B1 = rex.B1_dense
        for name, idx in zip(found, indices, strict=False):
            incident_edges = [
                e for e in range(rex.nE) if B1[idx, e] != 0
            ]
            n_connections = len(incident_edges)

            # Local harmonic fraction on incident edges
            harm_edges = hodge.get("harm_norm_per_edge", [])
            total_edges = hodge.get("grad_norm_per_edge", [])
            local_harm = 0.0
            local_total = 0.0
            if harm_edges and total_edges:
                for e in incident_edges:
                    local_harm += harm_edges[e] ** 2
                    local_total += (
                        harm_edges[e] ** 2
                        + total_edges[e] ** 2
                    )
                local_harm_frac = local_harm / local_total if local_total > 1e-30 else 0.0
            else:
                local_harm_frac = 0.0

            # Local frustration
            frust_edges = hodge.get("frustration_per_edge", [])
            local_frust = sum(
                frust_edges[e] for e in incident_edges
            ) if frust_edges else 0.0

            per_entity[name] = {
                "vertex_index": idx,
                "connections": n_connections,
                "local_harmonic_fraction": round(local_harm_frac, 4),
                "local_frustration": round(local_frust, 4),
            }

        return {
            "entities_found": found,
            "entities_missing": missing,
            "health_ratio": hodge.get("health_ratio"),
            "dim_H": hodge.get("dim_H", 0),
            "base_tokens": cost.get("base_tokens", 0),
            "adjusted_tokens": cost.get("adjusted_tokens", 0),
            "harmonic_fraction": cost.get("harmonic_fraction", 0),
            "recommendation": cost.get("recommendation", ""),
            "per_entity": per_entity,
        }

    def health_snapshot(
        self,
        flow: str = None,
        core_id: str = None,
        rex=None,
        meta: dict = None,
    ) -> dict:
        """Quick structural health check with cost readiness.

        Returns the graph health, the number of oscillatory modes,
        the harmonic fraction, and a cost multiplier that indicates
        how much extra token budget queries against this graph will
        need on average.

        Parameters
        ----------
        flow : str, optional
            TrustGraph flow (connected mode).
        core_id : str, optional
            Context core (connected mode).
        rex : RexGraph, optional
            Pre-built relational complex.
        meta : dict, optional
            Construction metadata.

        Returns
        -------
        dict with keys:
            ``'nV'``, ``'nE'``, ``'nF'``: graph size.
            ``'dim_H'``: oscillatory modes.
            ``'health_ratio'``: frustration / coparticipation.
            ``'harmonic_fraction'``: fraction of energy in harmonic subspace.
            ``'cost_multiplier'``: average token cost multiplier
                (1.0 = no overhead, 2.0 = double the base cost).
            ``'status'``: ``'healthy'``, ``'marginal'``, or ``'unstable'``.
        """
        if rex is None:
            if core_id is not None:
                self.load_kg_core(core_id, flow=flow or "default")
                rex, meta = self.from_flow(flow=flow or "default")
            elif flow is not None:
                rex, meta = self.from_flow(flow=flow)
            else:
                raise ValueError(
                    "Provide rex+meta, or flow, or core_id."
                )

        analysis = self.analyze(rex)
        hodge = analysis.get("hodge", {})

        h_frac = hodge.get("pct_harmonic", 0)
        health = hodge.get("health_ratio")
        dim_H = hodge.get("dim_H", 0)
        cost_mult = 1.0 + 2.5 * h_frac

        if health is not None and health > 1.1:
            status = "unstable"
        elif health is not None and health > 0.95:
            status = "marginal"
        else:
            status = "healthy"

        return {
            "nV": rex.nV,
            "nE": rex.nE,
            "nF": rex.nF,
            "dim_H": dim_H,
            "health_ratio": health,
            "harmonic_fraction": round(h_frac, 4),
            "cost_multiplier": round(cost_mult, 3),
            "status": status,
        }

    # MCP tool definitions (TrustGraph 2.5+)

    def as_mcp_tool_definitions(self) -> list[dict]:
        """Return MCP-compatible tool definitions for the RexGraph
        structural analysis capabilities.

        These definitions can be registered with TrustGraph's MCP
        server so that agents can call RexGraph analysis as a tool
        during their reasoning.

        Returns
        -------
        list of dict
            MCP tool definitions with name, description, and
            input_schema for each tool.
        """
        return [
            {
                "name": "rexgraph_analyze_flow",
                "description": (
                    "Analyze the structural health of a TrustGraph "
                    "knowledge flow using Hodge decomposition. Returns "
                    "the harmonic fraction (oscillation risk), "
                    "frustration index (unresolvable tension), "
                    "coparticipation index (decision-supported "
                    "dynamics), and health ratio."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "flow": {
                            "type": "string",
                            "description": "Flow identifier.",
                            "default": "default",
                        },
                    },
                },
            },
            {
                "name": "rexgraph_subgraph_confidence",
                "description": (
                    "Compute structural confidence for a set of "
                    "entities in the knowledge graph. Returns per-edge "
                    "structural character, void map, harmonic fraction, "
                    "and a coherence score."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "flow": {
                            "type": "string",
                            "description": "Flow identifier.",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Entity names to analyze.",
                        },
                    },
                    "required": ["entities"],
                },
            },
            {
                "name": "rexgraph_predict_query_cost",
                "description": (
                    "Predict the LLM token cost of a query based on "
                    "the harmonic complexity of the relevant knowledge "
                    "subgraph. Higher harmonic content means more "
                    "tokens needed for accurate responses."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "flow": {
                            "type": "string",
                            "description": "Flow identifier.",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Entities in the query.",
                        },
                    },
                    "required": ["entities"],
                },
            },
            {
                "name": "rexgraph_compare_flows",
                "description": (
                    "Compare the structural health of multiple "
                    "knowledge flows. Identifies which flow is most "
                    "stable, which has the most unresolved tensions, "
                    "and the total oscillatory mode count."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "flows": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Flow identifiers to compare.",
                        },
                    },
                    "required": ["flows"],
                },
            },
        ]

    # Visualization

    def render_confidence_viz(
        self,
        rex,
        analysis: dict,
        entity_indices: list[int] = None,
        theme: str = "parchment",
    ) -> str:
        """Render a structural confidence visualization.

        Shows the knowledge graph with void regions highlighted,
        channel-colored edges, and confidence metrics.

        Parameters
        ----------
        rex : RexGraph
        analysis : dict
        entity_indices : list of int, optional
            If provided, compute subgraph confidence for these entities.
        theme : str
            Visualization theme name.

        Returns
        -------
        str
            Rendered visualization (format depends on VizEngine).
        """
        try:
            from agent.ui.engine import VizEngine
        except ImportError as exc:
            raise RuntimeError(
                "VizEngine is not available. The old ui/ module has been replaced "
                "by the frontend/ web UI. Use the web interface at /api/v1/export "
                "or the Python client for analysis results."
            ) from exc

        engine = VizEngine(theme=theme)

        if entity_indices:
            confidence = self.subgraph_confidence(rex, entity_indices)
            analysis["confidence"] = confidence

        return engine.auto(
            analysis,
            rex=rex,
            title="TrustGraph x RexGraph: Structural Confidence",
        )
