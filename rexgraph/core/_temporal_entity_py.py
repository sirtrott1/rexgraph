"""
rexgraph.core._temporal_entity_py: Pure-Python fallback for entity-level
BIOES tagging. Same API as _temporal_entity.pyx, uses numpy vectorization.

Import order: try Cython first, fall back to this.

    try:
        from rexgraph.core._temporal_entity import entity_bioes_matrix
    except ImportError:
        from rexgraph.core._temporal_entity_py import entity_bioes_matrix
"""

import numpy as np

TAG_B = 0
TAG_I = 1
TAG_O = 2
TAG_E = 3
TAG_S = 4


def entity_bioes_matrix(birth, death, T):
    """Build N × T BIOES tag matrix. Vectorized numpy."""
    N = len(birth)
    tags = np.full((N, T), TAG_O, dtype=np.int32)

    d_eff = np.where(death < 0, T, death)
    spans = d_eff - birth

    for i in range(N):
        b, d, s = int(birth[i]), int(d_eff[i]), int(spans[i])
        if s <= 0:
            continue
        elif s == 1:
            tags[i, b] = TAG_S
        else:
            tags[i, b] = TAG_B
            tags[i, d - 1] = TAG_E
            tags[i, b + 1:d - 1] = TAG_I

    return tags


def entity_bioes_gapped(snapshots, edge_ids, directed=False):
    """Gap-aware per-entity BIOES. Pure Python."""

    T = len(snapshots)
    N = len(edge_ids)

    # Build presence matrix
    eid_to_idx = {int(eid): i for i, eid in enumerate(edge_ids)}
    presence = np.zeros((N, T), dtype=np.uint8)

    for t in range(T):
        src, tgt = snapshots[t]
        for j in range(len(src)):
            s, tg = int(src[j]), int(tgt[j])
            if not directed and s > tg:
                s, tg = tg, s
            key = s * 2147483648 + tg
            idx = eid_to_idx.get(key)
            if idx is not None:
                presence[idx, t] = 1

    # Tag from presence
    tags = np.full((N, T), TAG_O, dtype=np.int32)
    n_spans = np.zeros(N, dtype=np.int32)

    for i in range(N):
        in_span = False
        span_start = -1
        spans = 0
        for t in range(T):
            if presence[i, t]:
                if not in_span:
                    span_start = t
                    in_span = True
            else:
                if in_span:
                    _tag_span(tags, i, span_start, t - 1)
                    spans += 1
                    in_span = False
        if in_span:
            _tag_span(tags, i, span_start, T - 1)
            spans += 1
        n_spans[i] = spans

    return tags, n_spans


def _tag_span(tags, row, start, end):
    if start == end:
        tags[row, start] = TAG_S
    else:
        tags[row, start] = TAG_B
        tags[row, end] = TAG_E
        tags[row, start + 1:end] = TAG_I


def vertex_lifecycle(snapshots, directed=False):
    """Per-vertex birth/death. Pure Python."""
    T = len(snapshots)
    first_seen = {}
    last_seen = {}

    for t in range(T):
        src, tgt = snapshots[t]
        for j in range(len(src)):
            for v in [int(src[j]), int(tgt[j])]:
                if v not in first_seen:
                    first_seen[v] = t
                last_seen[v] = t

    vids = np.array(sorted(first_seen.keys()), dtype=np.int32)
    birth = np.array([first_seen[v] for v in vids], dtype=np.int32)
    death = np.array(
        [-1 if last_seen[v] == T - 1 else last_seen[v] + 1
         for v in vids], dtype=np.int32)

    return vids, birth, death


def cross_document_stats(birth, death, doc_boundaries, T):
    """Cross-document statistics. Pure Python."""
    N = len(birth)
    d_eff = np.where(death < 0, T, death)
    spans = d_eff - birth

    chunk_doc = np.zeros(T, dtype=np.int32)
    cur = 0
    for b_idx in doc_boundaries:
        chunk_doc[b_idx:] = cur + 1
        cur += 1

    n_cross = n_within = n_single = 0
    cross_lifespans = []
    hist = np.zeros(T + 1, dtype=np.int32)

    for i in range(N):
        s = int(spans[i])
        if s <= 0:
            continue
        hist[min(s, T)] += 1
        if s == 1:
            n_single += 1
            n_within += 1
        else:
            b_doc = chunk_doc[int(birth[i])]
            d_doc = chunk_doc[min(int(d_eff[i]) - 1, T - 1)]
            if b_doc != d_doc:
                n_cross += 1
                cross_lifespans.append(s)
            else:
                n_within += 1

    return {
        "n_total": N,
        "n_cross_doc": n_cross,
        "n_within_doc": n_within,
        "n_singleton": n_single,
        "lifespan_histogram": hist,
        "cross_doc_lifespans": np.array(cross_lifespans, dtype=np.int32),
        "mean_lifespan": float(spans[spans > 0].mean()) if (spans > 0).any() else 0.0,
    }


def persistence_spectrum(birth, death, T):
    """Persistence spectrum. Pure Python."""
    d_eff = np.where(death < 0, T, death).astype(np.float64)
    b_f = birth.astype(np.float64)
    lifespans = d_eff - b_f
    pairs = np.column_stack([b_f, d_eff])
    order = np.argsort(-lifespans)
    return lifespans[order], pairs[order]
