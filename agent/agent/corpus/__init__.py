"""
Cross-document corpus analysis: orchestration over rexgraph Cython kernels.

Takes multiple documents of any supported format, builds per-document
relational complexes via ``auto_rex`` (CSV, JSON, DataFrames, Parquet,
Arrow, HDF5, Zarr, .rex bundles, raw text, images, PDFs), resolves
shared entities, constructs cross-document complexes via the existing
``_joins`` and ``_cross_complex`` kernels, runs BIOES temporal tagging
via ``_temporal``, and provides propagator-based query matching via
``_query.spectral_propagate()``.

Every operation delegates to compiled Cython kernels.

Usage:

    from agent.corpus import CorpusBuilder

    corpus = CorpusBuilder()
    corpus.add_document("paper_2024.pdf", date="2024-01-15")
    corpus.add_document("paper_2025.pdf", date="2025-06-01")
    corpus.build()

    # Cross-document void analysis (missing information)
    voids = corpus.cross_document_voids()

    # BIOES temporal tagging across the document sequence
    tags = corpus.temporal_tags()

    # Propagator-based query matching
    sections = corpus.query("TSMC semiconductor supply chain")

    # TrustGraph triples with provenance
    triples = corpus.to_triples()
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import contextlib

logger = logging.getLogger(__name__)


#: The face rule a document complex is built under: the canonical one.
#:
#: A face is a filled cycle of any gon, solved from B1 c = 0. Named here as well
#: because a document complex, the query complex it is scored against and the chunks
#: taken from it must all be built the same way; two complexes under different rules
#: are not comparable.
from agent.auto import FACE_RULE as DOC_FACE_RULE


# Data classes
@dataclass

class DocumentRecord:
    """A single document in the corpus."""

    doc_id: str
    source: str = ""
    date: str | None = None
    text: str = ""

    # Per-document RexGraph (set after build)
    rex: Any = None
    edge_construction: Any = None
    analysis: dict[str, Any] = field(default_factory=dict)
    vertex_labels: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result of a propagator-based corpus query."""

    query_text: str
    ranked_sections: list[dict[str, Any]] = field(default_factory=list)
    query_character: np.ndarray | None = None
    query_rex: Any = None


# Entity extraction (lightweight, no external NLP)
def _extract_entities(text: str, min_len: int = 3) -> list[str]:
    """Extract candidate entities from text.

    Uses capitalization and noun-phrase heuristics.
    Returns deduplicated, lowercased entity strings.
    """
    # Capitalized multi-word phrases (e.g. "United States", "Machine Learning")
    cap_phrases = re.findall(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text,
    )
    # Single capitalized words (not at sentence start)
    single_caps = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+\b', text)
    # Also grab mid-sentence capitals
    mid_caps = re.findall(r'(?<=[a-z]\s)([A-Z][a-zA-Z]{2,})\b', text)

    entities = set()
    for phrase in cap_phrases:
        normalized = phrase.lower().strip()
        if len(normalized) >= min_len:
            entities.add(normalized)

    for word in single_caps + mid_caps:
        normalized = word.lower().strip()
        if len(normalized) >= min_len:
            entities.add(normalized)

    return sorted(entities)


# Corpus builder
# Ranking
#
# One mechanism, in agent.scoring: the interfacing vector (Poisson lift -> typed
# channel operators -> bilinear score). What used to be here was a label Jaccard
# plus a cosine between MEAN structural characters plus a hand-rolled spectral term,
# blended under fixed 0.3/0.35/0.35 weights: three approximations of the thing the
# library already computes exactly. Lexical overlap is now a candidate prefilter
# only; it decides what to look at, not what is relevant.


def score_document(doc, query_ec, query_chi=None, mode="hybrid") -> float:
    """Score a document against a query. `query_chi`/`mode` are accepted and ignored:
    they selected between the old blends, and there is one mechanism now."""
    from agent.scoring import interfacing_score
    return interfacing_score(getattr(doc, "rex", None),
                             getattr(doc, "vertex_labels", []) or [],
                             getattr(query_ec, "vertex_labels", []) or [])["score"]


def score_document_full(doc, query_ec) -> dict:
    """`score_document` plus the typed character and the bundle's diagnostics."""
    from agent.scoring import interfacing_score
    return interfacing_score(getattr(doc, "rex", None),
                             getattr(doc, "vertex_labels", []) or [],
                             getattr(query_ec, "vertex_labels", []) or [])


def count_shared_entities(labels_a, labels_b) -> int:
    """Count entities shared between two label sets."""
    return len({str(x).lower() for x in (labels_a or [])}
               & {str(x).lower() for x in (labels_b or [])})


class CorpusBuilder:
    """Cross-document corpus analysis using existing Cython kernels.

    Accepts any input type that auto_rex handles: CSV edge lists,
    feature matrices, DataFrames, JSON, Parquet, Arrow IPC, HDF5,
    Zarr, .rex bundles, raw text, images, and PDFs.

    All mathematical operations delegate to compiled rexgraph code:
    - ``_cross_complex``: entity alignment, kappa comparison, void comparison
    - ``_joins``: inner/outer/left join of relational complexes
    - ``_temporal``: BIOES tagging, edge/face lifecycle, phase detection
    - ``_query``: spectral propagation, signal imputation
    - ``_persistence``: persistence diagrams, bottleneck distance
    - ``_fiber``: structural character cosine similarity on Δ³
    """

    def __init__(
        self,
        strategy: str = "text",
        ocr_client=None,
        **adapter_kwargs,
    ):
        self.strategy = strategy
        self.ocr_client = ocr_client
        self.adapter_kwargs = adapter_kwargs

        self.documents: list[DocumentRecord] = []
        self._built = False

        # Cross-document state (populated by build())
        self._merged_rex = None
        self._shared_labels: list[str] = []
        self._doc_edge_types: np.ndarray | None = None
        self._temporal_snapshots: list = []

    # Document ingestion
    def add_document(
        self,
        source: str,
        doc_id: str | None = None,
        date: str | None = None,
        text: str | None = None,
        edge_construction: Any | None = None,
    ) -> str:
        """Add a document to the corpus.

        Parameters
        ----------
        source : str
            File path (CSV, JSON, PDF, image, Parquet, HDF5, .rex, etc.)
            or raw text.  Type is auto-detected by ``auto_rex``.
        doc_id : str, optional
            Unique document identifier.  Auto-generated if None.
        date : str, optional
            Document date for chronological ordering (ISO format).
        text : str, optional
            Pre-extracted text.  If None, OCR is run on ``source``.

        Returns
        -------
        str : the doc_id assigned
        """
        if doc_id is None:
            doc_id = f"doc_{len(self.documents):04d}"

        self.documents.append(DocumentRecord(
            doc_id=doc_id,
            source=source,
            date=date,
            text=text or "",
            edge_construction=edge_construction,
        ))
        self._built = False
        return doc_id

    def add_text(self, text: str, doc_id: str | None = None,
                 date: str | None = None) -> str:
        """Add raw text as a document (no OCR needed)."""
        return self.add_document(
            source="<text>", doc_id=doc_id, date=date, text=text,
        )

    def add_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: list[str] | None = None,
        date: str | None = None,
    ) -> list[str]:
        """Walk a directory and add each supported file as a document.

        Parameters
        ----------
        directory : str
            Root directory to scan.
        recursive : bool
            Walk subdirectories (default True).
        extensions : list of str, optional
            File extensions to include (e.g. ``['.pdf', '.csv']``).
            If None, includes all supported types.
        date : str, optional
            Date applied to all documents in the directory.

        Returns
        -------
        list of str : doc_ids of added documents
        """
        import os
        from pathlib import Path

        supported = {
            ".csv", ".tsv", ".json", ".parquet", ".arrow",
            ".h5", ".hdf5", ".zarr", ".rex",
            ".pdf",
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
            ".txt", ".md",
        }
        # every registered reader too, so a format added to agent.adapters.formats
        # is ingestable without editing this set. It was a literal, so the
        # scientific containers were readable by auto_rex and silently skipped
        # here: a plan that promised them and a build that dropped them.
        try:
            from agent.adapters.formats import available_extensions
            supported = supported | set(available_extensions())
        except Exception:
            pass
        if extensions:
            allowed = {e if e.startswith(".") else f".{e}" for e in extensions}
        else:
            allowed = supported

        doc_ids = []
        walker = os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]

        for root, dirs, files in walker:
            dirs.sort()
            for name in sorted(files):
                ext = Path(name).suffix.lower()
                if ext not in allowed:
                    continue
                filepath = os.path.join(root, name)
                rel = os.path.relpath(filepath, directory)
                doc_id = rel.replace(os.sep, "/").rsplit(".", 1)[0]

                # Text files: read content directly
                if ext in (".txt", ".md"):
                    try:
                        with open(filepath, errors="replace") as f:
                            text = f.read()
                        did = self.add_text(text, doc_id=doc_id, date=date)
                    except Exception as e:
                        logger.warning("Skipping %s: %s", filepath, e)
                        continue
                else:
                    did = self.add_document(
                        source=filepath, doc_id=doc_id, date=date,
                    )
                doc_ids.append(did)

        return doc_ids

    # Build
    def build(self, depth: str = "standard", stage_callback=None) -> None:
        """Build per-document complexes and cross-document structure.

        Accepts any input that auto_rex handles: CSV, JSON, DataFrames,
        Parquet, Arrow, HDF5, Zarr, .rex bundles, raw text, images, and
        PDFs.  Each document becomes a RexGraph; cross-document analysis
        uses the existing Cython kernels.

        Parameters
        ----------
        depth : str
            Analysis depth ('quick', 'standard', 'full').
        stage_callback : callable, optional
            ``callback(doc_id, stage_name, stage_data)`` invoked as each
            analysis stage completes, so a server can stream per-stage
            progress instead of a single opaque "analysis" step.

        Steps:
            1. Per-document RexGraph via auto_rex (type auto-detected)
            2. Analysis pipeline per document
            3. Entity resolution across documents (align_by_labels)
            4. Temporal snapshot construction (for BIOES)
        """
        if len(self.documents) == 0:
            raise ValueError("No documents in corpus")

        from agent import cache as _cache
        from agent.auto import auto_rex, build_rex_from_edges
        from agent.pipeline import AnalysisPipeline

        for doc in self.documents:
            # Content-addressed cache: skip rebuild + analysis when we've
            # seen identical input at this depth before.
            cache_key = None
            if getattr(doc, "edge_construction", None) is None:
                try:
                    content = doc.text or doc.source or doc.doc_id
                    # The face rule is part of what the complex IS, so it belongs in
                    # the key. It lives in the effective kwargs rather than in
                    # adapter_kwargs, so keying on adapter_kwargs alone served a
                    # complex built under a different rule as a hit.
                    eff = dict(self.adapter_kwargs)
                    eff.setdefault("face_selection", DOC_FACE_RULE)
                    extra = repr(sorted(eff.items()))
                    cache_key = _cache.content_key(content, depth=depth, extra=extra)
                    c_rex, c_analysis, c_meta = _cache.get_rex_and_analysis(cache_key)
                    if c_rex is not None and c_analysis is not None:
                        doc.rex = c_rex
                        doc.analysis = c_analysis
                        doc.meta = c_meta or getattr(c_rex, "_agent_meta", {})
                        doc.vertex_labels = list(doc.meta.get("vertex_labels", []))
                        if not doc.text:
                            st = doc.meta.get("source_text", "")
                            if st:
                                doc.text = st
                        if stage_callback is not None:
                            stage_callback(doc.doc_id, "cache_hit", {"cached": True})
                        continue
                except Exception:
                    cache_key = None

            try:
                if getattr(doc, "edge_construction", None) is not None:
                    # An adapter already built the edges outside auto_rex
                    # (e.g. OCR-layout so document structure is preserved,
                    # or a single-cell / L-R construction). Use them
                    # directly so we don't re-route through the flat
                    # TextAdapter.
                    # Faces are asked for, not assumed, and a document complex
                    # wants them: with none, curl is identically 0 and every loop
                    # a document contains reads as harmonic instead. "typed" is
                    # what this path produced before faces became a request, kept
                    # so the reading did not change silently. It is a type filter
                    # over triangles, which is not what a face is; the rule this
                    # should use is an open decision.
                    rex = build_rex_from_edges(
                        doc.edge_construction,
                        face_selection=DOC_FACE_RULE,
                        input_type=getattr(
                            doc.edge_construction, "input_type",
                            "edge_construction",
                        ),
                    )
                else:
                    kw = dict(self.adapter_kwargs)
                    kw.setdefault("face_selection", DOC_FACE_RULE)   # keyed above
                    if doc.text:
                        rex = auto_rex(doc.text, **kw)
                    elif doc.source and doc.source != "<text>":
                        rex = auto_rex(doc.source, **kw)
                    else:
                        logger.warning("No source for %s, skipping", doc.doc_id)
                        continue
            except Exception as e:
                logger.warning("Failed to build rex for %s: %s", doc.doc_id, e)
                continue

            if rex is None or rex.nE == 0:
                logger.warning("Empty complex for %s, skipping", doc.doc_id)
                continue

            doc.rex = rex
            meta = getattr(rex, "_agent_meta", {})
            doc.vertex_labels = list(meta.get("vertex_labels", []))
            doc.meta = meta
            # Store source text on doc if not already set (needed for chunking)
            if not doc.text and doc.source and doc.source != "<text>":
                try:
                    from pathlib import Path as _Path
                    source_text = meta.get("source_text", "")
                    if not source_text:
                        p = _Path(doc.source)
                        if p.exists() and p.suffix.lower() in ('.txt', '.csv', '.tsv', '.json'):
                            source_text = p.read_text(errors='replace')[:50000]
                    if source_text and len(source_text.strip()) > 10:
                        doc.text = source_text
                except Exception:
                    pass

            pipe = AnalysisPipeline(rex)
            if stage_callback is not None:
                _did = doc.doc_id

                def _cb(stage_name, stage_data, _doc_id=_did):
                    with contextlib.suppress(Exception):
                        stage_callback(_doc_id, stage_name, stage_data)

                pipe.on_stage(_cb)
            doc.analysis = pipe.run(depth=depth)

            # Populate the cache for next time (best-effort).
            if cache_key is not None:
                with contextlib.suppress(Exception):
                    _cache.store_rex_and_analysis(
                        cache_key, rex, doc.analysis, doc.meta,
                    )

        # Sort by date if available
        dated = [d for d in self.documents if d.date]
        if dated:
            dated.sort(key=lambda d: d.date)
            undated = [d for d in self.documents if not d.date]
            self.documents = dated + undated

        self._build_temporal_snapshots()
        self._built = True

    def _build_temporal_snapshots(self):
        """Construct snapshot sequences for _temporal kernels."""
        self._temporal_snapshots = []
        for doc in self.documents:
            if doc.rex is None:
                continue
            rex = doc.rex
            self._temporal_snapshots.append((
                rex.sources.copy(),
                rex.targets.copy(),
            ))

    # Cross-document analysis (calls Cython kernels)
    def cross_document_kappa(
        self,
        doc_a: int = 0,
        doc_b: int = 1,
    ) -> dict[str, Any]:
        """Compare coherence across two documents.

        Calls ``_cross_complex.align_by_labels()`` and
        ``_cross_complex.cross_complex_kappa()``.
        """
        self._ensure_built()
        from rexgraph.core._cross_complex import (
            align_by_labels,
            cross_complex_kappa,
        )

        da, db = self.documents[doc_a], self.documents[doc_b]
        if da.rex is None or db.rex is None:
            return {"error": "One or both documents have no RexGraph"}

        shared_labels, idx_a, idx_b = align_by_labels(da.vertex_labels, db.vertex_labels)

        kappa_a = da.analysis.get("relational", {}).get(
            "kappa_per_vertex", [],
        )
        kappa_b = db.analysis.get("relational", {}).get(
            "kappa_per_vertex", [],
        )

        if not kappa_a or not kappa_b:
            return {"n_shared": len(idx_a), "kappa_comparison": None}

        result = cross_complex_kappa(
            np.array(kappa_a, dtype=np.float64),
            np.array(kappa_b, dtype=np.float64),
            idx_a, idx_b,
        )
        return result

    def cross_document_voids(
        self,
        doc_a: int = 0,
        doc_b: int = 1,
    ) -> dict[str, Any]:
        """Compare void structure across two documents.

        Calls ``_cross_complex.cross_complex_void_fraction()``.
        """
        self._ensure_built()
        from rexgraph.core._cross_complex import (
            cross_complex_void_fraction,
        )

        da, db = self.documents[doc_a], self.documents[doc_b]
        void_a = da.analysis.get("void", {})
        void_b = db.analysis.get("void", {})

        return cross_complex_void_fraction(
            void_a.get("n_voids", 0),
            void_a.get("n_potential", 0),
            void_b.get("n_voids", 0),
            void_b.get("n_potential", 0),
        )

    def cross_document_bridge(
        self,
        doc_a: int = 0,
        doc_b: int = 1,
    ) -> dict[str, Any]:
        """Full cross-document structural bridge.

        Calls ``_cross_complex.cross_complex_bridge()``.
        """
        self._ensure_built()
        from rexgraph.core._cross_complex import (
            align_by_labels,
            cross_complex_bridge,
        )

        da, db = self.documents[doc_a], self.documents[doc_b]
        if da.rex is None or db.rex is None:
            return {"error": "One or both documents have no RexGraph"}

        shared_labels, idx_a, idx_b = align_by_labels(da.vertex_labels, db.vertex_labels)

        kappa_a = np.array(
            da.analysis.get("relational", {}).get("kappa_per_vertex", []),
            dtype=np.float64,
        )
        kappa_b = np.array(
            db.analysis.get("relational", {}).get("kappa_per_vertex", []),
            dtype=np.float64,
        )

        if kappa_a.size == 0 or kappa_b.size == 0:
            return {"n_shared": len(idx_a)}

        void_a = da.analysis.get("void", {})
        void_b = db.analysis.get("void", {})

        return cross_complex_bridge(
            kappa_a, kappa_b, idx_a, idx_b,
            void_a.get("n_voids", 0),
            void_a.get("n_potential", 0),
            void_b.get("n_voids", 0),
            void_b.get("n_potential", 0),
        )

    # Temporal BIOES tagging (calls _temporal kernels)
    def temporal_tags(
        self,
        phase_tol: float = 0.0,
        min_phase_len: int = 2,
    ) -> dict[str, Any]:
        """Run BIOES temporal tagging across the document sequence.

        Each document is a "timestep."  Calls
        ``_temporal.compute_bioes_full()`` with Betti numbers
        from each document's analysis.
        """
        self._ensure_built()
        from rexgraph.core._temporal import (
            compute_bioes_full,
            edge_lifecycle,
        )

        if len(self._temporal_snapshots) < 2:
            return {"error": "Need at least 2 documents for temporal analysis"}

        # Extract Betti numbers per document
        docs_with_rex = [d for d in self.documents if d.rex is not None]
        T = len(docs_with_rex)

        beta0 = np.zeros(T, dtype=np.int64)
        beta1 = np.zeros(T, dtype=np.int64)
        for i, doc in enumerate(docs_with_rex):
            betti = doc.analysis.get("topology", {}).get("betti", [0, 0])
            beta0[i] = betti[0] if len(betti) > 0 else 0
            beta1[i] = betti[1] if len(betti) > 1 else 0

        tags, ec, bc, dc, p_start, p_end, p_b0, p_b1 = compute_bioes_full(
            self._temporal_snapshots,
            beta0, beta1,
            directed=False,
            phase_tol=phase_tol,
            min_phase_len=min_phase_len,
        )

        # Edge lifecycle across documents
        lifecycle = edge_lifecycle(self._temporal_snapshots, directed=False)

        tag_names = {0: "B", 1: "I", 2: "O", 3: "E", 4: "S"}

        return {
            "tags": [tag_names.get(int(t), "?") for t in tags],
            "tags_raw": tags,
            "edge_counts": ec,
            "edges_born": bc,
            "edges_died": dc,
            "n_phases": len(p_start),
            "phase_start": p_start,
            "phase_end": p_end,
            "lifecycle": lifecycle,
            "doc_ids": [d.doc_id for d in docs_with_rex],
        }

    def metrics(self) -> dict:
        """Per-DOCUMENT and per-CORPUS information metrics: each built document's
        structural perplexity (effective modes), coherence, and varentropy reliability
        gap, plus their corpus-level distribution and diversity (the effective number
        of coherence-distinct documents). Same Rényi calculus as the token/response
        metrics; see agent.metrics."""
        from agent.metrics import corpus_metrics, structural_metrics
        docs = [d for d in self.documents if d.rex is not None]
        per_document = []
        for d in docs:
            try:
                sm = structural_metrics(d.rex)
                sm["doc_id"] = d.doc_id
                sm["coherence"] = round(float(np.asarray(d.rex.coherence).mean()), 4)
                per_document.append(sm)
            except Exception:
                continue
        return {
            "n_documents": len(docs),
            "per_document": per_document,
            "corpus": corpus_metrics([d.rex for d in docs]),
        }

    # Query matching (calls _query and _fiber kernels)
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> QueryResult:
        """Find the most structurally relevant document sections.

        Modes:
            chi:      structural character cosine + Jaccard overlap
            spectral: propagator-based matching via _query.spectral_propagate
            hybrid:   combine both (weighted average)
        """
        self._ensure_built()
        from agent.adapters.text import TextAdapter
        from rexgraph.graph import RexGraph

        # Build query RexGraph
        ta = TextAdapter()
        qec = ta.build(query_text, min_count=1, max_vocab=200)

        if not qec.vertex_labels:
            return QueryResult(query_text=query_text)

        # If query has no edges (single word), skip RexGraph but still score by vocabulary
        q_chi = None
        q_rex = None
        if qec.nE > 0:
            q_rex = RexGraph(
                sources=qec.sources,
                targets=qec.targets,
            )
            if qec.n_types > 1:
                from agent.auto import attach_faces
                q_rex = attach_faces(q_rex, DOC_FACE_RULE, type_labels=qec.type_labels)
            with contextlib.suppress(Exception):
                q_chi = q_rex.structural_character

        # Score each document by structural similarity
        results = []
        for doc in self.documents:
            if doc.rex is None:
                continue

            score = self._score_document(doc, qec, q_chi, mode)
            results.append({
                "doc_id": doc.doc_id,
                "source": doc.source,
                "score": score,
                "n_shared_entities": self._count_shared_entities(
                    qec.vertex_labels, doc.vertex_labels,
                ),
                "kappa_mean": doc.analysis.get(
                    "relational", {},
                ).get("kappa_mean", 0),
            })

        # doc_id breaks ties: without a ranking term that varies with vocabulary,
        # every non-matching document scores exactly 0, and enumeration order
        # would otherwise decide the tail differently per caller.
        results.sort(key=lambda r: (-r["score"], str(r["doc_id"])))
        return QueryResult(
            query_text=query_text,
            ranked_sections=results[:top_k],
            query_character=q_chi.mean(axis=0) if q_chi is not None else None,
            query_rex=q_rex,
        )

    # Ranking lives in agent.scoring; these stay as the historical entry points.
    def _score_document(self, doc, query_ec, query_chi=None, mode="hybrid") -> float:
        return score_document(doc, query_ec, query_chi, mode)

    _count_shared_entities = staticmethod(count_shared_entities)

    def to_triples(self) -> list:
        """Generate TrustGraph triples with cross-document provenance.

        Calls ``TrustGraphAdapter.to_enrichment_triples()`` per
        document, then adds cross-document provenance triples.
        """
        self._ensure_built()
        from agent.integrations.trustgraph_adapter import TrustGraphAdapter

        tga = TrustGraphAdapter()
        all_triples = []

        for doc in self.documents:
            if doc.rex is None:
                continue

            triples = tga.to_enrichment_triples(doc.rex, doc.analysis)
            # Add provenance triple for each entity
            from agent.integrations.trustgraph_adapter import SimpleTriple
            for t in triples:
                # Prefix subject with doc_id for provenance
                all_triples.append(SimpleTriple(
                    s=f"{doc.doc_id}:{t.s}",
                    p=t.p,
                    o=t.o,
                ))

        # Cross-document consistency triples. Only document PAIRS that share ≥1
        # entity can emit a triple (the n_shared > 0 gate), so build an inverted
        # index (entity label -> docs containing it) once and bridge only the
        # co-occurring pairs - instead of the old all-pairs O(D²) scan that ran a
        # full alignment for every pair, including the (usually many) that share
        # nothing. Output is identical; a corpus with a hub entity in every doc
        # still bridges those pairs (they genuinely share), only faster to reach.
        from agent.integrations.trustgraph_adapter import SimpleTriple
        posting: dict[str, list[int]] = {}
        for i, doc in enumerate(self.documents):
            if doc.rex is None:
                continue
            for lab in doc.vertex_labels:
                posting.setdefault(lab, []).append(i)
        candidate_pairs = set()
        for docs_with_label in posting.values():
            if len(docs_with_label) > 1:
                for a_pos in range(len(docs_with_label)):
                    for b_pos in range(a_pos + 1, len(docs_with_label)):
                        i, j = docs_with_label[a_pos], docs_with_label[b_pos]
                        candidate_pairs.add((i, j) if i < j else (j, i))
        for i, j in sorted(candidate_pairs):
            da, db = self.documents[i], self.documents[j]
            bridge = self.cross_document_bridge(i, j)
            n_shared = bridge.get("n_shared", 0)
            if n_shared > 0:
                kappa_data = bridge.get("kappa", {})
                corr = kappa_data.get("correlation", 0)
                all_triples.append(SimpleTriple(
                    s=da.doc_id,
                    p="http://rexgraph.org/corpus/shared_entities_with",
                    o=f"{db.doc_id}:{n_shared}",
                ))
                all_triples.append(SimpleTriple(
                    s=da.doc_id,
                    p="http://rexgraph.org/corpus/kappa_correlation_with",
                    o=f"{db.doc_id}:{round(corr, 4) if corr else 0}",
                ))

        return all_triples

    def trustgraph_analysis(self, depth: str = "standard") -> dict:
        """Run TrustGraph ontology enrichment over the whole corpus.

        Generates enrichment triples for every document (KEGG / GO /
        CellPhoneDB-style ontology mappings via the TrustGraph adapter),
        then runs the standalone TrustGraph engine over them and returns
        a JSON-safe summary.  This is the pipeline hook the manual
        workflow used to produce its enrichment triples.

        Returns a dict with ``available`` False and a ``reason`` when the
        TrustGraph integration cannot run, so callers can treat it as an
        optional stage.
        """
        self._ensure_built()
        summary: dict = {"available": False}
        try:
            from agent.integrations.trustgraph_pipeline import TrustGraphPipeline
        except Exception as e:
            summary["reason"] = f"TrustGraph integration unavailable: {e}"
            return summary

        try:
            triples = self.to_triples()
        except Exception as e:
            summary["reason"] = f"triple generation failed: {e}"
            return summary

        summary["n_triples"] = len(triples)
        if not triples:
            summary["reason"] = "no enrichment triples produced"
            return summary

        try:
            tgp = TrustGraphPipeline.standalone()
            result = tgp.analyze_triples(triples, depth=depth)
        except Exception as e:
            summary["reason"] = f"engine run failed: {e}"
            summary["available"] = True  # triples still generated
            return summary

        summary["available"] = True
        # Pull compact, serialisable fields off the EngineResult.
        try:
            meta = getattr(result, "meta", {}) or {}
            summary["n_entities"] = meta.get("nV") or meta.get("n_entities")
            summary["n_relations"] = meta.get("nE") or meta.get("n_relations")
        except Exception:
            pass
        analysis = getattr(result, "analysis", None)
        if isinstance(analysis, dict):
            # Keep only small scalar-ish fields to stay SSE-friendly.
            keep = {}
            for k, v in analysis.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    keep[k] = v
                elif isinstance(v, dict) and k in ("topology", "hodge"):
                    keep[k] = {
                        kk: vv for kk, vv in v.items()
                        if isinstance(vv, (int, float, str, bool, list))
                    }
            summary["analysis"] = keep
        interp = getattr(result, "interpretation", None)
        if isinstance(interp, dict):
            summary["interpretation"] = {
                k: v for k, v in interp.items()
                if isinstance(v, (int, float, str, bool, list))
            }
        return summary

    # Persistence comparison
    def persistence_distance(
        self,
        doc_a: int = 0,
        doc_b: int = 1,
    ) -> dict[str, float]:
        """Compute bottleneck and Wasserstein distances between
        persistence diagrams of two documents.

        Calls ``_persistence.persistence_diagram()``,
        ``_persistence.bottleneck_distance()``, and
        ``_persistence.wasserstein_distance()``.
        """
        self._ensure_built()
        # Persistence diagrams require filtrations, so use the edge weight filtration
        from rexgraph.core._persistence import (
            bottleneck_distance,
            persistence_diagram,
            persistence_entropy,
            persistence_landscape,
            wasserstein_distance,
        )

        da, db = self.documents[doc_a], self.documents[doc_b]
        if da.rex is None or db.rex is None:
            return {"error": "One or both documents have no RexGraph"}

        try:
            ra, rb = da.rex, db.rex

            filt_v_a = np.zeros(ra.nV, dtype=np.float64)
            filt_e_a = ra.w_E.copy() if ra.w_E is not None else np.ones(ra.nE, dtype=np.float64)
            filt_f_a = np.zeros(ra.nF, dtype=np.float64) if ra.nF > 0 else np.array([], dtype=np.float64)

            filt_v_b = np.zeros(rb.nV, dtype=np.float64)
            filt_e_b = rb.w_E.copy() if rb.w_E is not None else np.ones(rb.nE, dtype=np.float64)
            filt_f_b = np.zeros(rb.nF, dtype=np.float64) if rb.nF > 0 else np.array([], dtype=np.float64)

            dgm_a = persistence_diagram(filt_v_a, filt_e_a, filt_f_a,
                                        ra._boundary_ptr, ra._boundary_idx,
                                        ra._B2_col_ptr, ra._B2_row_idx)
            dgm_b = persistence_diagram(filt_v_b, filt_e_b, filt_f_b,
                                        rb._boundary_ptr, rb._boundary_idx,
                                        rb._B2_col_ptr, rb._B2_row_idx)

            pairs_a = dgm_a['pairs'] if isinstance(dgm_a, dict) else dgm_a
            pairs_b = dgm_b['pairs'] if isinstance(dgm_b, dict) else dgm_b

            bn = bottleneck_distance(pairs_a, pairs_b)
            ws = wasserstein_distance(pairs_a, pairs_b)

            result = {
                "bottleneck": float(bn),
                "wasserstein": float(ws),
                "dgm_a_size": len(pairs_a),
                "dgm_b_size": len(pairs_b),
            }

            try:
                grid = np.linspace(0, float(np.max(filt_e_a) + 1), 50)
                land_a = persistence_landscape(pairs_a, grid, k_max=3)
                land_b = persistence_landscape(pairs_b, grid, k_max=3)
                if isinstance(land_a, np.ndarray) and isinstance(land_b, np.ndarray):
                    result["landscape_distance"] = float(np.linalg.norm(land_a - land_b))
            except Exception:
                pass

            try:
                ent_a = persistence_entropy(pairs_a)
                ent_b = persistence_entropy(pairs_b)
                result["entropy_a"] = float(ent_a)
                result["entropy_b"] = float(ent_b)
                result["entropy_delta"] = float(abs(ent_a - ent_b))
            except Exception:
                pass

            return result
        except Exception as e:
            return {"error": str(e)}

    # Cross-dataset comparison
    def cross_dataset_comparison(self, metric: str = "bottleneck") -> dict:
        """Compare structural invariants across *all* documents at once.

        Produces the multi-dataset comparison the Poincaré critical-surface
        analysis needed: a pairwise persistence-distance matrix plus a
        per-document invariant table (betti, Hodge fractions, kappa) and
        shared-entity / kappa-correlation bridges.  Every distance comes
        from the compiled ``_persistence`` kernels via
        :meth:`persistence_distance`.

        Parameters
        ----------
        metric : str
            Which persistence-distance field to place in the matrix
            ('bottleneck', 'wasserstein', or 'landscape_distance').
        """
        self._ensure_built()
        docs = [d for d in self.documents if d.rex is not None]
        n = len(docs)
        ids = [d.doc_id for d in docs]

        # Per-document invariant table.
        invariants = []
        for d in docs:
            rel = d.analysis.get("relational", {}) if d.analysis else {}
            hodge = d.analysis.get("hodge", {}) if d.analysis else {}
            topo = d.analysis.get("topology", {}) if d.analysis else {}
            invariants.append({
                "doc_id": d.doc_id,
                "nV": d.rex.nV, "nE": d.rex.nE, "nF": d.rex.nF,
                "betti": topo.get("betti"),
                "kappa_mean": rel.get("kappa_mean"),
                "pct_gradient": hodge.get("pct_gradient"),
                "pct_curl": hodge.get("pct_curl"),
                "pct_harmonic": hodge.get("pct_harmonic"),
            })

        # Pairwise persistence-distance matrix (symmetric, zero diagonal).
        matrix = [[0.0] * n for _ in range(n)]
        errors = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    pd = self.persistence_distance(
                        self.documents.index(docs[i]),
                        self.documents.index(docs[j]),
                    )
                    val = pd.get(metric)
                    if val is None:
                        val = pd.get("bottleneck", 0.0)
                    val = float(val) if val is not None else 0.0
                except Exception as e:
                    val = 0.0
                    errors.append(f"{ids[i]}~{ids[j]}: {e}")
                matrix[i][j] = matrix[j][i] = val

        # Shared-entity / kappa bridges.
        bridges = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    b = self.cross_document_bridge(
                        self.documents.index(docs[i]),
                        self.documents.index(docs[j]),
                    )
                    bridges.append({
                        "a": ids[i], "b": ids[j],
                        "n_shared": b.get("n_shared", 0),
                        "kappa_correlation": (b.get("kappa", {}) or {}).get(
                            "correlation"
                        ),
                    })
                except Exception:
                    pass

        return {
            "n_documents": n,
            "doc_ids": ids,
            "metric": metric,
            "distance_matrix": matrix,
            "invariants": invariants,
            "bridges": bridges,
            "errors": errors,
        }

    # Utilities
    # RCDB persistence
    #
    # One record per document, so the store IS the corpus rather than somewhere a
    # corpus gets copied to. Everything retrieval needs (labels, source text) already
    # rides in the rex's _agent_meta and round-trips through the canonical serializer;
    # what goes in `meta` is only what a reader needs WITHOUT opening the blob.

    def persist(self, store=None, *, prefix: str = "", tags: list[str] | None = None,
                valid_from=None) -> list[str]:
        """Write each built document into an RCStore. Returns the ids written.

        Re-persisting an unchanged corpus is a no-op on version numbers: each document
        goes through `version_if_changed`, so repeated ingests do not spam the lineage.
        """
        from agent import rcdb
        self._ensure_built()
        store = store or rcdb.default_store()
        written = []
        for doc in self.documents:
            if doc.rex is None:
                continue
            rid = f"{prefix}{doc.doc_id}"
            # vertex_labels has to be in meta: structural_signature falls back to the
            # rex's _agent_meta only when meta is absent, so an explicit meta without
            # labels silently drops labels_sample/n_labels from the signature and the
            # cheap prefilter goes blind.
            meta = {
                "doc_id": rid,
                "corpus_doc_id": doc.doc_id,
                "source": doc.source or "<text>",
                "date": doc.date,
                "vertex_labels": list(doc.vertex_labels),
                "input_type": (doc.meta or {}).get("input_type", "text"),
            }
            rcdb.version_if_changed(store, rid, doc.rex, meta=meta,
                                    tags=list(tags or []), valid_from=valid_from)
            written.append(rid)
        return written

    @classmethod
    def from_store(cls, store=None, *, ids: Sequence[str] | None = None,
                   prefix: str = "", as_of=None, valid_at=None,
                   limit: int = 1000, **kwargs) -> CorpusBuilder:
        """Rehydrate a corpus from an RCStore.

        `as_of`/`valid_at` read the store bitemporally, so a corpus can be
        reconstructed as it stood at a transaction or validity time.
        """
        from agent import rcdb
        store = store or rcdb.default_store()
        corpus = cls(**kwargs)
        if ids is None:
            ids = [r.id for r in store.list(limit=limit)
                   if not prefix or r.id.startswith(prefix)]
        for rid in ids:
            rex = store.get(rid, as_of=as_of, valid_at=valid_at)
            if rex is None:
                continue
            rec = store.get_record(rid, as_of=as_of, valid_at=valid_at)
            rmeta = getattr(rex, "_agent_meta", {}) or {}
            recmeta = (rec.meta if rec is not None else {}) or {}
            doc = DocumentRecord(
                doc_id=rid[len(prefix):] if prefix and rid.startswith(prefix) else rid,
                source=recmeta.get("source", "<store>"),
                date=recmeta.get("date"),
                text=rmeta.get("source_text", "") or "",
            )
            doc.rex = rex
            doc.meta = rmeta
            doc.vertex_labels = list(rmeta.get("vertex_labels")
                                     or recmeta.get("vertex_labels") or [])
            corpus.documents.append(doc)
        corpus._rehydrate()
        return corpus

    def _rehydrate(self):
        """Mark a store-loaded corpus built without re-running the per-document
        pipeline: the rexes are already the analyzed ones that were persisted."""
        for doc in self.documents:
            if doc.rex is not None and not doc.analysis:
                doc.analysis = {}
        self._build_temporal_snapshots()
        self._built = True

    def _ensure_built(self):
        if not self._built:
            raise RuntimeError("Call build() before analysis")

    @property
    def n_documents(self) -> int:
        return len(self.documents)

    @property
    def document_ids(self) -> list[str]:
        return [d.doc_id for d in self.documents]

    def summary(self) -> str:
        """Human-readable corpus summary."""
        lines = [f"Corpus: {self.n_documents} documents"]
        for doc in self.documents:
            status = f"{doc.rex.nV}V {doc.rex.nE}E {doc.rex.nF}F" if doc.rex else "no rex"
            lines.append(f"  {doc.doc_id}: {status}")
            if doc.date:
                lines.append(f"    date: {doc.date}")
        return "\n".join(lines)
