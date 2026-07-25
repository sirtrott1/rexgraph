"""
model_introspect - run the RCF relational math on the model's own internals, pulled
live from the running llama.cpp server (Tier-1 bridge: embeddings + logits over the
OpenAI-compatible API - no PyTorch, no C++ patch).

The model's embedding geometry becomes a relational complex analyzed by the same
compiled Cython kernels + moment engine, reading the model at inference on the
Vulkan/llama.cpp stack. (Raw per-layer attention is Tier-2: it needs
a ggml patch to expose the tensors, then a zero-copy float* -> the Cython extern/pointer
ABI - kept as a separate optional module, not vendored.)
"""
from __future__ import annotations

import numpy as np


def _resolve_url():
    from agent import chat_model
    t = chat_model._resolve()
    url = getattr(t, "url", "") or ""
    if not url:
        raise RuntimeError(
            "No local model server configured. Start one via the Models tab (Local) or "
            "`python -m agent.local_runtime start MODEL.gguf`.")
    return url.rstrip("/"), getattr(t, "model", "")


def embed(texts, url=None, model=None, timeout: float = 60.0) -> np.ndarray:
    """Embedding vectors for ``texts`` from the server's ``/v1/embeddings`` (launch the
    server with embeddings enabled, e.g. ``llama-server --embeddings``). Returns
    (len(texts), dim) float64."""
    import httpx
    if url is None:
        try:                                     # prefer the dedicated beehive embedding worker
            from agent import local_runtime
            url = local_runtime.embed_url()
        except Exception:
            url = None
    if url is None:
        url, m = _resolve_url()
        model = m if model is None else model
    r = httpx.post(url + "/v1/embeddings",
                   json={"input": list(texts), "model": model or ""}, timeout=timeout)
    r.raise_for_status()
    data = r.json().get("data", [])
    return np.array([d["embedding"] for d in data], dtype=np.float64)


def _complex_from_vectors(V: np.ndarray, labels, top_p: float = 0.9) -> dict:
    """Vectors -> relational complex -> RCF metrics. The shared analysis body: build a cosine
    graph over the rows, sparsify with the nucleus (top_p) rule, and read structural
    perplexity, coherence, Betti, and the load-bearing (bridge) pairs via effective
    resistance. Used by both live embeddings and a reloaded corpus so the math lives once."""
    from rexgraph.graph import RexGraph
    from agent import metrics as _M
    from agent.integrations.huggingface_analyzer import extract_attention_rex

    labels = [str(x)[:48] for x in labels]
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] < 3:
        return {"n_items": int(V.shape[0]), "note": "need >= 3 items to form a complex"}
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    S = np.clip(Vn @ Vn.T, 0.0, None)                 # cosine similarity, positive part
    ed = extract_attention_rex(S, top_p=top_p)        # reuse the nucleus sparsifier
    if ed["sources"].size < 3:
        return {"n_items": V.shape[0], "note": "too few edges above the nucleus"}
    rex = RexGraph.from_graph(ed["sources"], ed["targets"], w_E=ed["weights"])
    rex._agent_meta = {"vertex_labels": labels}
    eff = rex._effective_resistance_batch(np.arange(rex.nE))
    order = np.argsort(-eff)[:8]
    bridges = []
    for e in order:
        s, t = int(ed["sources"][e]), int(ed["targets"][e])
        bridges.append({
            "from": labels[s] if s < len(labels) else str(s),
            "to": labels[t] if t < len(labels) else str(t),
            "effective_resistance": round(float(eff[e]), 4)})
    return {
        "n_items": int(V.shape[0]),
        "structural": _M.structural_metrics(rex),
        "coherence_mean": round(float(np.asarray(rex.coherence).mean()), 4),
        "betti": [int(b) for b in rex.betti],
        "bridges": bridges,   # load-bearing concept links in the embedding space
    }


def embedding_complex(texts, url=None, top_p: float = 0.9, persist: str = None) -> dict:
    """The model's EMBEDDING GEOMETRY as a relational complex: embed the items and run the
    RCF moment engine on the cosine graph - "which concepts are central vs bridge vs
    frustrated in the model's own representation space." Tier-1: no PyTorch, no C++ patch -
    the compiled Cython core reading the C++ engine's output over the API. If ``persist`` is
    a path, the embedding matrix is saved (via ``model_io``/``rexgraph.io``) so it can be
    re-analyzed later with ``embedding_complex_from_corpus`` without re-embedding."""
    labels = [str(t)[:48] for t in texts]
    url_r, model = (None, None)
    try:
        url_r, model = _resolve_url()
    except Exception:
        pass
    V = embed(texts, url=url or url_r)
    if persist and V.shape[0] >= 1:
        from agent import model_io
        model_io.save_embedding_corpus(V, np.array(labels), persist, model=model,
                                       source=url or url_r, top_p=top_p)
    return _complex_from_vectors(V, labels, top_p)


def embedding_complex_from_corpus(path: str, top_p: float = 0.9) -> dict:
    """Re-run the embedding-geometry analysis on a corpus saved by ``embedding_complex(...,
    persist=path)`` - no server call, no re-embedding. Same math as the live path."""
    from agent import model_io
    V, labels, _names, _meta = model_io.load_embedding_corpus(path)
    labs = list(labels) if labels is not None else list(range(V.shape[0]))
    return _complex_from_vectors(V, labs, top_p)
