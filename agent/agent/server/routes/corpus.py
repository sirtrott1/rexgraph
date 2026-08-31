"""agent.server.routes.corpus: workspace-scoped corpus analysis."""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile

from rexgraph.io._compat import json_sanitize

from agent.server.auth import TokenEntry, WorkspaceState, require_auth, require_workspace

router = APIRouter(prefix="/v1/corpus")


@router.post("/add-text")
async def add_text_json(
    body: dict = Body(...),
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """Add text as a relational complex and keep it as a .rex document.

    Body: {text, doc_id?, date?, persist?}. The text is built into a complex the
    same way an uploaded .txt is (words as vertices, sentence co-occurrence as
    relations) and written as a .rex bundle. The bundle carries the source text in
    its own metadata, so the document is one file and the text is not a sidecar.

    `persist: false` adds it to the corpus without writing the bundle.
    """
    text = body.get("text")
    if not text or not str(text).strip():
        raise HTTPException(400, "Provide 'text'")
    corpus = ws.get_corpus()
    doc_id = body.get("doc_id")
    if doc_id:
        # Checked before anything is built, so a refused id costs nothing. The save
        # checks again on the id the corpus actually assigned.
        from agent.server.persistence import doc_path
        try:
            doc_path(ws.name, str(doc_id))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    ws.record_activity(token.user_id, "add_document", doc_id or "text")
    did = corpus.add_text(str(text), doc_id=doc_id, date=body.get("date"))

    out = {"doc_id": did, "n_documents": corpus.n_documents}
    if body.get("persist") is False:
        return out
    # Build so the document HAS a complex: add_text records the text and defers
    # construction, so persisting before this would write nothing.
    try:
        corpus.build(depth=body.get("depth", "quick"))
    except Exception as e:
        raise HTTPException(500, f"Could not build a complex from the text: {e}") from e
    doc = next((d for d in corpus.documents if d.doc_id == did), None)
    if doc is None or doc.rex is None:
        raise HTTPException(500, "The text produced no complex")
    from agent.server.persistence import save_document_rex
    out["path"] = save_document_rex(ws.name, did, doc.rex)
    out["nV"], out["nE"] = int(doc.rex.nV), int(doc.rex.nE)
    out["vertex_labels"] = (doc.vertex_labels or [])[:12]
    return out


@router.post("/add")
async def add_document(
    file: UploadFile = File(None), path: str = Form(None),
    text: str = Form(None), doc_id: str = Form(None), date: str = Form(None),
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    corpus = ws.get_corpus()
    ws.record_activity(token.user_id, "add_document", doc_id or "file")
    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1] or ".bin"
        from agent.server.persistence import staging_dir
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False,
                                         dir=str(staging_dir(ws.name))) as tmp:
            content = await file.read(); tmp.write(content); source = tmp.name
        did = corpus.add_document(source=source, doc_id=doc_id or file.filename, date=date)
    elif text:
        did = corpus.add_text(text, doc_id=doc_id, date=date)
    elif path:
        # Which directories a caller may read from, and the test that decides it.
        # The home directory is NOT here any more. It was, and it holds ~/.ssh, ~/.aws
        # and the deployment's own credential store, so the widest entry in the
        # allow-list contained the secrets the allow-list exists to protect. An
        # operator who wants a directory says so in REXGRAPH_ALLOWED_DIRS.
        from ..handles import path_allowed
        resolved = os.path.realpath(path)
        if not path_allowed(resolved):
            raise HTTPException(403, "Path outside allowed directories")
        if os.path.isdir(resolved):
            ids = corpus.add_directory(resolved, date=date)
            return {"doc_ids": ids, "count": len(ids)}
        did = corpus.add_document(source=resolved, doc_id=doc_id, date=date)
    else:
        raise HTTPException(400, "Provide file, text, or path")
    return {"doc_id": did, "n_documents": corpus.n_documents}


@router.post("/build")
async def build_corpus(
    depth: str = Form("standard"),
    ontology: bool = Form(False),
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    corpus = ws.get_corpus()
    ws.record_activity(token.user_id, "build", ws.name)
    if corpus.n_documents == 0:
        raise HTTPException(400, "No documents in corpus")
    try:
        corpus.build(depth=depth)
    except Exception as e:
        raise HTTPException(500, f"Build failed: {e}") from e
    docs = []
    for doc in corpus.documents:
        d = {"doc_id": doc.doc_id, "source": doc.source, "date": doc.date}
        if doc.rex:
            d["nV"] = doc.rex.nV; d["nE"] = doc.rex.nE; d["nF"] = doc.rex.nF
            topo = doc.analysis.get("topology", {})
            d["betti"] = topo.get("betti", [])
            rel = doc.analysis.get("relational", {})
            d["kappa_mean"] = rel.get("kappa_mean"); d["chi_mean"] = rel.get("chi_mean")
            hodge = doc.analysis.get("hodge", {})
            d["hodge"] = {"gradient": hodge.get("pct_gradient"), "curl": hodge.get("pct_curl"), "harmonic": hodge.get("pct_harmonic")}
        docs.append(d)
    out = {"n_documents": corpus.n_documents, "documents": docs}
    if ontology:
        try:
            out["ontology"] = corpus.trustgraph_analysis(depth=depth)
        except Exception as e:
            out["ontology"] = {"available": False, "reason": str(e)}
    return out


@router.post("/query")
async def query_corpus(
    query: str = Form(...), top_k: int = Form(5), mode: str = Form("hybrid"),
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    corpus = ws.get_corpus()
    ws.record_activity(token.user_id, "query", query[:50])
    if not corpus._built:
        raise HTTPException(400, "Build the corpus first")
    result = corpus.query(query, top_k=top_k, mode=mode)
    return {"query": result.query_text, "mode": mode, "ranked": result.ranked_sections,
            "query_character": result.query_character.tolist() if result.query_character is not None else None}


@router.get("/temporal")
async def temporal_tags(
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    corpus = ws.get_corpus()
    if not corpus._built:
        raise HTTPException(400, "Build the corpus first")
    # temporal_tags returns numpy arrays and a tuple of arrays, which FastAPI's
    # encoder cannot serialize, so the request died after the handler succeeded.
    # Only reachable with two or more documents; one document short-circuits.
    return json_sanitize(corpus.temporal_tags(), nan="null")


@router.get("/bridge/{doc_a}/{doc_b}")
async def cross_document_bridge(
    doc_a: int, doc_b: int,
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    corpus = ws.get_corpus()
    if not corpus._built:
        raise HTTPException(400, "Build the corpus first")
    if doc_a >= corpus.n_documents or doc_b >= corpus.n_documents:
        raise HTTPException(400, "Invalid document index")
    return corpus.cross_document_bridge(doc_a, doc_b)


@router.get("/voids/{doc_a}/{doc_b}")
async def cross_document_voids(
    doc_a: int, doc_b: int,
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    corpus = ws.get_corpus()
    if not corpus._built:
        raise HTTPException(400, "Build the corpus first")
    return corpus.cross_document_voids(doc_a, doc_b)


@router.get("/persistence/{doc_a}/{doc_b}")
async def persistence_distance(
    doc_a: int, doc_b: int,
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    corpus = ws.get_corpus()
    if not corpus._built:
        raise HTTPException(400, "Build the corpus first")
    return corpus.persistence_distance(doc_a, doc_b)


@router.get("/summary")
async def corpus_summary(
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    corpus = ws.get_corpus()
    return {"n_documents": corpus.n_documents, "built": corpus._built,
            "doc_ids": corpus.document_ids, "summary": corpus.summary() if corpus._built else "",
            "workspace": ws.name, "user": token.user_id}


@router.get("/metrics")
async def get_corpus_metrics(
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    """Per-document and per-corpus information metrics: each built document's
    structural perplexity (effective modes), coherence, and varentropy reliability
    gap, plus the corpus-level distribution and diversity (the effective number of
    coherence-distinct documents). Same Rényi calculus as the token/response metrics."""
    corpus = ws.get_corpus()
    if not corpus._built:
        raise HTTPException(400, "Build the corpus first")
    return corpus.metrics()


@router.post("/compare")
async def compare_datasets(
    metric: str = Form("bottleneck"),
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    """Cross-dataset structural comparison across all corpus documents.

    Computes a pairwise persistence-distance matrix plus per-document
    invariants and shared-entity bridges.
    """
    corpus = ws.get_corpus()
    if not corpus._built:
        raise HTTPException(400, "Build the corpus first")
    if corpus.n_documents < 2:
        raise HTTPException(400, "Need at least 2 documents to compare")
    try:
        return corpus.cross_dataset_comparison(metric=metric)
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {e}") from e


@router.post("/trustgraph")
async def trustgraph_enrichment(
    depth: str = Form("standard"),
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    """Run TrustGraph ontology enrichment across the corpus."""
    corpus = ws.get_corpus()
    if not corpus._built:
        raise HTTPException(400, "Build the corpus first")
    try:
        return corpus.trustgraph_analysis(depth=depth)
    except Exception as e:
        raise HTTPException(500, f"TrustGraph analysis failed: {e}") from e


@router.post("/reset")
async def reset_corpus(
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    ws.corpus = None; ws.trackers.clear()
    return {"status": "reset", "workspace": ws.name}


@router.post("/fusion")
async def ocr_fusion(
    file: UploadFile = File(...), backends: str = Form("paddleocr,offline"),
    token: TokenEntry = Depends(require_auth), ws: WorkspaceState = Depends(require_workspace),
):
    from agent.integrations.ocr_fusion import OCRFusion
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=tempfile.gettempdir()) as tmp:
        content = await file.read(); tmp.write(content); path = tmp.name
    try:
        backend_list = [b.strip() for b in backends.split(",") if b.strip()]
        fusion = OCRFusion(backends=backend_list)
        report = fusion.compare(path)
        return {"n_backends": report.n_backends, "best_coherence": report.best_coherence,
                "summary": report.summary()}
    finally:
        os.unlink(path)
