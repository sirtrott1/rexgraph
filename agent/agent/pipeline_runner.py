"""
agent.pipeline_runner: end-to-end document analysis pipeline.

    files -> OCR -> corpus -> Hodge chunk -> query -> LLM -> hallucination check -> rechunk

Every step calls compiled Cython kernels. This module is glue.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


#: the gate could not be evaluated, so the caller proceeds unannotated
GATE_OK = "ok"
#: measurable but weak coverage: answer, and say the context is thin
GATE_WARN = "warn"
#: the query and the source share no vocabulary, so nothing was retrieved
GATE_REFUSE = "refuse"


def _context_quality_gate(source_rex, source_labels, query) -> dict:
    """Structural-coverage gate: does the source cover the query well enough to
    trust the retrieved context?

    Returns {"verdict", "reasons", "score", "n_shared"} where verdict is one of
    GATE_OK, GATE_WARN, GATE_REFUSE.

    REFUSE is reserved for a MEASURED zero: the query supplies terms and none of
    them name a vertex, so retrieval returned nothing and any answer would come
    from the model alone rather than the source. Being unable to evaluate is not
    the same finding and stays permissive.

    WARN is read off `quality_gate`, which is q(x) = x / (x + median(|x|)) per
    channel. That puts its own fixed point at 0.5, where a channel scores exactly
    the median magnitude, so the comparison is against the statistic the kernel
    already normalises by rather than against a tuned constant.

    The channel-scored path uses dense L0-eigenbasis Cython kernels that assume a
    FULL nV x nV basis. On the universal sparse path the bundle carries only a
    truncated L0 basis (k<<nV) for nV>2000, so feeding it to those kernels reads
    out of bounds (C-level segfault). Guard on the full basis and skip the gate
    when it is unavailable.
    """
    def verdict(v, *reasons, score=None, n_shared=None):
        return {"verdict": v, "reasons": list(reasons),
                "score": score, "n_shared": n_shared}

    if not source_rex or source_rex.nE == 0:
        return verdict(GATE_OK, "no source complex to measure against")
    try:
        # A token that names a vertex counts however short it is: identifiers are
        # routinely two characters, and dropping them by length would report zero
        # coverage for a query that names its subject exactly. The length rule only
        # decides which NON-matching tokens are substantive enough to judge on.
        index_of = {str(lbl).lower(): i for i, lbl in enumerate(source_labels)}
        tokens = [w.lower().strip(".,;:!?()[]\"'") for w in query.split()]
        shared = [index_of[t] for t in tokens if t in index_of]
        usable = [t for t in tokens if len(t) > 2 or t in index_of]
        if not usable:
            return verdict(GATE_OK, "query carries no terms substantive enough to match")
        if not shared:
            return verdict(
                GATE_REFUSE,
                f"none of the {len(usable)} query terms name a vertex in the source",
                n_shared=0)

        sb = source_rex.spectral_bundle
        evecs_L0 = sb.get('evecs_L0')
        if evecs_L0 is None or evecs_L0.shape[1] != source_rex.nV:
            # truncated basis: dense kernels unsafe, so the channel score is
            # unavailable. The vocabulary overlap above still stands.
            return verdict(GATE_OK, "truncated L0 basis, channel score unavailable",
                           n_shared=len(shared))

        from rexgraph.core._interfacing import (
            build_edge_signal,
            build_response_operators,
            build_vertex_source,
            channel_scores,
            quality_gate,
        )
        deg = source_rex.degree.astype(np.float64)
        vw = 1.0 / np.log(deg + np.e)
        ti = np.array(shared, dtype=np.int32)
        tw = np.ones(len(shared), dtype=np.float64)
        rho = build_vertex_source(ti, tw, vw, source_rex.nV)
        B1 = np.ascontiguousarray(source_rex.B1, dtype=np.float64)
        psi = build_edge_signal(
            rho, B1, sb['evals_L0'],
            np.ascontiguousarray(evecs_L0, dtype=np.float64),
            source_rex.nV, source_rex.nE,
        )
        resp_ops = build_response_operators(
            B1, sb['evals_L0'], evecs_L0,
            source_rex.g_channel_operator, source_rex.L_frustration,
            source_rex.nV, source_rex.nE,
        )
        ch_scores = channel_scores(
            psi, resp_ops['S_T'], resp_ops['S_G'], resp_ops['S_F'],
            psi, source_rex.nE,
        )
        gate = quality_gate(ch_scores.reshape(1, -1))
        if not isinstance(gate, np.ndarray):
            return verdict(GATE_OK, "gate returned no score", n_shared=len(shared))
        score = float(gate.mean())
        if score <= 0.5:
            return verdict(GATE_WARN,
                           f"channel score {score:.3f} at or below the gate's median "
                           "fixed point",
                           score=score, n_shared=len(shared))
        return verdict(GATE_OK, score=score, n_shared=len(shared))
    except Exception as exc:
        return verdict(GATE_OK, f"gate could not be evaluated: {exc}")


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    documents: list = field(default_factory=list)
    corpus_summary: dict = field(default_factory=dict)
    temporal: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    query_result: dict = field(default_factory=dict)
    model_response: str = ""
    hallucination_report: dict = field(default_factory=dict)
    ontology: dict = field(default_factory=dict)
    elapsed: float = 0.0


class PipelineRunner:
    """Runs the full OCR -> analysis -> LLM pipeline."""

    # "read" = load documents into the corpus: parse tables/JSON/text directly,
    # or OCR image/PDF files. Named domain-agnostically (it is NOT always OCR).
    PHASES = ["read", "corpus", "analysis", "chunking", "query", "model", "hallucination"]

    OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    def __init__(self, ocr_client=None, model_url=None, max_vocab=200):
        self._ocr_client = ocr_client
        self._model_url = model_url
        self.max_vocab = max_vocab
        self._callbacks = []
        self._last_corpus = None
        self._stage_callback = None

    def on_phase(self, callback):
        """Register a callback for phase progress: callback(phase_name, data_dict)."""
        self._callbacks.append(callback)

    def _emit(self, phase, data):
        for cb in self._callbacks:
            with contextlib.suppress(Exception):
                cb(phase, data)

    def run(
        self,
        files: list[str] | None = None,
        texts: list[str] | None = None,
        doc_ids: list[str] | None = None,
        query: str | None = None,
        max_rechunk: int = 2,
        depth: str = "standard",
        ontology: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline.

        Pass ``files`` for auto-detected processing, or ``texts``/``doc_ids``
        for pre-extracted text (when OCR is cached in the caller).

        Files are routed by type via auto_rex:
            PDF/images  -> OCR -> text -> CorpusBuilder
            CSV/TSV     -> csv_loader -> RexGraph
            JSON        -> json_loader -> RexGraph
            .txt/other  -> read as text -> TextAdapter

        When ``ontology`` is True, an optional TrustGraph enrichment stage
        runs after chunking (audit 1.1).
        """
        t0 = time.time()
        result = PipelineResult()

        # Per-analysis-stage callback -> surfaced as "analysis" phase events
        # so the frontend can show real progress during the long
        # eigendecomposition/Hodge stages (audit 4.2).
        def stage_cb(doc_id, stage_name, stage_data):
            payload = {"status": "running", "doc_id": doc_id, "stage": stage_name}
            if isinstance(stage_data, dict) and "error" in stage_data:
                payload["stage_error"] = stage_data["error"]
            self._emit("analysis", payload)

        self._stage_callback = stage_cb

        # Step 1: build one corpus from whatever inputs are present.
        # A mixed batch (OCR texts + direct CSV/JSON/text files) must
        # include BOTH - previously the texts branch won and direct files
        # were silently dropped (audit P2).
        have_texts = texts is not None and doc_ids is not None
        have_files = bool(files)
        if not have_texts and not have_files:
            result.elapsed = time.time() - t0
            return result

        if have_texts:
            self._emit("read", {"status": "skipped", "n_texts": len(texts)})
        if have_files:
            self._emit("read", {"status": "running", "n_files": len(files)})
        corpus = self._build_corpus(
            texts if have_texts else None,
            doc_ids if have_texts else None,
            files if have_files else None,
            depth,
        )
        if have_files:
            self._emit("read", {"status": "done"})

        if corpus is None:
            result.elapsed = time.time() - t0
            return result

        # Step 2: extract doc info
        self._emit("corpus", {"status": "running", "n_docs": len(corpus.documents)})
        result.documents = self._extract_doc_info(corpus)
        result.corpus_summary = corpus.summary() if corpus._built else {}
        self._emit("corpus", {"status": "done", "n_documents": len(result.documents)})

        # Temporal
        try:
            result.temporal = corpus.temporal_tags()
        except Exception as e:
            logger.warning("Temporal tags failed: %s", e)

        # Step 3: chunk each document
        self._emit("chunking", {"status": "running"})
        all_chunks = self._chunk_documents(corpus)
        result.chunks = [
            {
                "doc_id": doc_id,
                "chunks": [
                    {
                        "idx": c.idx,
                        "text_preview": c.text[:150],
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "n_edges": c.n_edges,
                        "kappa": c.kappa,
                        "hodge_gradient": c.hodge_gradient,
                        "hodge_curl": c.hodge_curl,
                        "hodge_harmonic": c.hodge_harmonic,
                        "dominant_channel": c.dominant_channel,
                    }
                    for c in chunks
                ],
            }
            for doc_id, chunks in all_chunks
        ]
        total_chunks = sum(len(dc["chunks"]) for dc in result.chunks)
        self._emit("chunking", {"status": "done", "total_chunks": total_chunks})

        # Optional TrustGraph ontology enrichment (audit 1.1)
        if ontology:
            self._emit("ontology", {"status": "running"})
            try:
                result.ontology = corpus.trustgraph_analysis(depth=depth)
                self._emit("ontology", {
                    "status": "done",
                    "available": result.ontology.get("available", False),
                    "n_triples": result.ontology.get("n_triples", 0),
                })
            except Exception as e:
                logger.warning("Ontology stage failed: %s", e)
                result.ontology = {"available": False, "reason": str(e)}
                self._emit("ontology", {"status": "error", "error": str(e)})

        # Step 4: query + LLM + hallucination check
        if query:
            self._emit("query", {"status": "running", "query": query})
            result.query_result, result.model_response, result.hallucination_report = \
                self._query_and_check(corpus, all_chunks, texts, query, max_rechunk)
            self._emit("query", {"status": "done"})

        result.elapsed = time.time() - t0
        return result

    def _ocr_files(self, files):
        """OCR each file, return (texts, doc_ids)."""
        from agent.integrations.unlimited_ocr import (
            create_ocr_client,
            is_image_file,
            is_pdf_file,
        )
        client = self._ocr_client or create_ocr_client()

        texts = []
        doc_ids = []
        for path in files:
            try:
                if is_pdf_file(path):
                    result = client.ocr_pdf(path)
                    text = result.full_text
                elif is_image_file(path):
                    result = client.ocr_image(path)
                    text = result.text
                else:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        text = f.read()

                if text and len(text.strip()) > 10:
                    texts.append(text)
                    doc_ids.append(Path(path).stem)
                else:
                    logger.warning("No text extracted from %s", path)
            except Exception as e:
                logger.warning("Failed to process %s: %s", path, e)

        return texts, doc_ids

    def _add_file_to_corpus(self, corpus, path):
        """Add one direct file to the corpus, routed by type.

        OCR-extension files are OCR'd and routed through the OCRAdapter's
        structure-aware layout path (audit 2.1). Everything else is added
        as a source path so auto_rex picks the right adapter (CSV/JSON/
        feature/text) - never flattened to word co-occurrence (audit P3).
        """
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in self.OCR_EXTENSIONS:
            text = self._ocr_single_file(path)
            if text and len(text.strip()) > 10:
                text = self._sanitize_text(text)
                ec = None
                try:
                    from agent.adapters.ocr import OCRAdapter
                    ec = OCRAdapter().build_from_text(text, strategy="layout")
                    if ec is None or ec.nE == 0:
                        ec = None
                except Exception as e:
                    logger.warning(
                        "OCRAdapter layout failed for %s (%s); "
                        "using flat text.", p.name, e)
                    ec = None
                corpus.add_document(
                    source=path, doc_id=p.stem, text=text,
                    edge_construction=ec,
                )
        else:
            corpus.add_document(source=path, doc_id=p.stem)

    def _build_corpus(self, texts, doc_ids, files, depth):
        """Build a single corpus from OCR texts and/or direct files."""
        from agent.corpus import CorpusBuilder
        corpus = CorpusBuilder(max_vocab=self.max_vocab)
        if texts and doc_ids:
            for text, doc_id in zip(texts, doc_ids, strict=False):
                corpus.add_text(text, doc_id=doc_id)
        for path in (files or []):
            self._add_file_to_corpus(corpus, path)
        if not corpus.documents:
            return None
        corpus.build(depth=depth, stage_callback=self._stage_callback)
        self._last_corpus = corpus
        return corpus

    def _build_corpus_from_files(self, files, depth):
        """Build corpus from file paths (thin wrapper over _build_corpus)."""
        return self._build_corpus(None, None, files, depth)

    def _build_corpus_from_texts(self, texts, doc_ids, depth):
        """Build corpus from pre-extracted texts (thin wrapper)."""
        return self._build_corpus(texts, doc_ids, None, depth)

    def _ocr_single_file(self, path):
        """OCR a single PDF or image file."""
        from agent.integrations.unlimited_ocr import create_ocr_client, is_image_file, is_pdf_file
        client = self._ocr_client or create_ocr_client()
        try:
            if is_pdf_file(path):
                return client.ocr_pdf(path).full_text
            elif is_image_file(path):
                return client.ocr_image(path).text
        except Exception as e:
            logger.warning("OCR failed for %s: %s", path, e)
        return ""

    @staticmethod
    def _sanitize_text(text):
        """Clean OCR output to prevent degenerate co-occurrence graphs."""
        if not text:
            return text
        text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = text.split('\n')
        lines = [ln for ln in lines if len(ln.strip()) > 15 or not ln.strip()]
        text = '\n'.join(lines)
        if len(text) > 80000:
            text = text[:80000]
        return text

    def _extract_doc_info(self, corpus):
        """Extract per-document analysis info."""
        docs = []
        for doc in corpus.documents:
            d = {"doc_id": doc.doc_id}
            if doc.rex:
                d["nV"] = doc.rex.nV
                d["nE"] = doc.rex.nE
                d["nF"] = doc.rex.nF
                topo = doc.analysis.get("topology", {})
                d["betti"] = topo.get("betti", [])
                rel = doc.analysis.get("relational", {})
                d["kappa_mean"] = rel.get("kappa_mean")
                d["chi_mean"] = rel.get("chi_mean")
                hodge = doc.analysis.get("hodge", {})
                d["hodge"] = {
                    "gradient": hodge.get("pct_gradient"),
                    "curl": hodge.get("pct_curl"),
                    "harmonic": hodge.get("pct_harmonic"),
                }
            docs.append(d)
        return docs

    def _chunk_documents(self, corpus):
        """Hodge-chunk each document."""
        from agent.adapters.text import TextAdapter
        from agent.chunking import hodge_chunk

        results = []
        ta = TextAdapter()
        for doc in corpus.documents:
            if doc.rex is None:
                continue
            if not doc.text or len(doc.text.strip()) < 50:
                continue
            try:
                ec = ta.build(doc.text, min_count=1, max_vocab=self.max_vocab)
                chunks = hodge_chunk(
                    doc.rex, ec.edge_spans, ec.sentence_spans, ec.source_text,
                )
                results.append((doc.doc_id, chunks))
            except Exception as e:
                logger.warning("Chunking failed for %s: %s", doc.doc_id, e)

        return results

    def _query_and_check(self, corpus, all_chunks, texts, query, max_rechunk):
        """Run query, call LLM, check for hallucinations."""
        from agent.hallucination import detect_hallucinations_exchange

        # Spectral query across corpus
        qr = corpus.query(query, top_k=3, mode="hybrid")
        query_dict = {
            "query": qr.query_text,
            "ranked": qr.ranked_sections,
        }

        # Build context from top-ranked chunks (kappa-gated + quality-gated)
        context_parts = []
        source_rex = None
        source_labels = []
        source_text = ""

        for ranked in (qr.ranked_sections or [])[:2]:
            doc_id = ranked.get("doc_id", "")
            for did, chunks in all_chunks:
                if did == doc_id:
                    for chunk in chunks:
                        if chunk.kappa > 0.5 or chunk.kappa > 0.2:
                            context_parts.append(chunk.text)
                    break

            for doc in corpus.documents:
                if doc.doc_id == doc_id and doc.rex is not None:
                    source_rex = doc.rex
                    source_labels = doc.vertex_labels
                    source_text = doc.text or ""
                    break

        context = "\n\n".join(context_parts)

        # Structural coverage of the query by the source
        gate = _context_quality_gate(source_rex, source_labels, query)
        gate_report = {
            "context_gate": gate["verdict"],
            "context_gate_reasons": gate["reasons"],
            "context_score": gate["score"],
            "context_sufficient": gate["verdict"] != GATE_REFUSE,
        }

        # Refusing here is the whole point of a measured zero: the source shares no
        # vocabulary with the query, so the model would be answering from itself and
        # the retrieved context would be decoration on top of that.
        if gate["verdict"] == GATE_REFUSE:
            why = "; ".join(gate["reasons"])
            logger.warning("Context gate refused: %s", why)
            return query_dict, (
                "The source does not cover this query: "
                f"{why}. Answering would not be grounded in it."
            ), gate_report

        # Call LLM
        model_response = ""
        try:
            model_response = self._call_model(query, context)
        except Exception as e:
            logger.warning("Model call failed: %s", e)
            return query_dict, f"Error: {e}", gate_report

        if gate["verdict"] == GATE_WARN:
            logger.warning("Context gate weak: %s", "; ".join(gate["reasons"]))
            model_response = (
                "[thin context: " + "; ".join(gate["reasons"]) + "]\n\n"
                + model_response
            )

        # Hallucination check via exchange complex
        hall_report = dict(gate_report)
        if source_text and model_response:
            report = detect_hallucinations_exchange(
                source_text, model_response,
            )
            hall_report.update({
                "overall_score": report.overall_score,
                "kappa_correlation": report.kappa_correlation,
                "n_shared": report.n_shared_entities,
                "n_flags": report.n_flags,
                "summary": report.summary(),
            })

        return query_dict, model_response, hall_report

    def _call_model(self, query, context):
        """Send query + context to the model."""
        import httpx

        url = self._model_url
        if not url:
            from agent.cli.serve import find_running_server
            url = find_running_server()
        if not url:
            url = os.environ.get("CHAT_MODEL_URL", "") or os.environ.get("UNLIMITED_OCR_URL", "")
        if not url:
            raise RuntimeError("No GPU server running")

        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"Document context:\n{context[:12000]}",
            })
        messages.append({"role": "user", "content": query})

        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{url}/v1/chat/completions",
                json={"messages": messages, "max_tokens": 1024, "temperature": 0.7},
            )
            resp.raise_for_status()
            data = resp.json()
            text = ""
            for choice in data.get("choices", []):
                text += choice.get("message", {}).get("content", "")
            return text
