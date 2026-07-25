# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._temporal_entity - Entity-level BIOES tagging for
cross-document NLP on relational complexes.

Extends _temporal with per-entity and per-relationship lifecycle
tracking.  Tags individual entities/edges with B-I-O-E-S across
the chunk/document sequence.

Hot loops are pure C with typed memoryviews.  No Python object
access in inner paths.

Functions
---------
entity_bioes_matrix   - N × T tag matrix from birth/death, single nogil pass
entity_bioes_gapped   - gap-aware tagging (re-appearance = new span)
vertex_lifecycle      - per-vertex birth/death tracking
cross_document_stats  - summary statistics with document boundaries
persistence_spectrum  - lifespan distribution for persistence analysis
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cimport cython

from rexgraph.core._common cimport i32, i64, f64

np.import_array()

# Tags - same values as _temporal.pyx
cdef enum:
    TAG_B = 0
    TAG_I = 1
    TAG_O = 2
    TAG_E = 3
    TAG_S = 4


# Edge encoding (mirrors _temporal.pyx)

cdef inline i64 _encode_undirected(i32 s, i32 t) noexcept nogil:
    if s <= t:
        return (<i64>s) * 2147483648LL + <i64>t
    return (<i64>t) * 2147483648LL + <i64>s

cdef inline i64 _encode_directed(i32 s, i32 t) noexcept nogil:
    return (<i64>s) * 2147483648LL + <i64>t


# Pure C inner loops

cdef void _tag_contiguous(i32 *row, i32 b, i32 d) noexcept nogil:
    """Tag a contiguous birth-death span in a pre-zeroed (O-filled) row."""
    cdef i32 span = d - b
    cdef i32 t
    if span <= 0:
        return
    if span == 1:
        row[b] = TAG_S
    else:
        row[b] = TAG_B
        row[d - 1] = TAG_E
        for t in range(b + 1, d - 1):
            row[t] = TAG_I

cdef void _tag_gapped(const np.uint8_t *prow, i32 *trow,
                       i32 *n_spans, i32 T) noexcept nogil:
    """Tag one entity row from its presence vector.  Pure C."""
    cdef i32 t, span_start, in_span, spans
    in_span = 0
    span_start = 0
    spans = 0

    for t in range(T):
        if prow[t] == 1:
            if not in_span:
                span_start = t
                in_span = 1
        else:
            if in_span:
                _tag_contiguous(trow, span_start, t)
                spans += 1
                in_span = 0

    if in_span:
        _tag_contiguous(trow, span_start, T)
        spans += 1

    n_spans[0] = spans


# entity_bioes_matrix

def entity_bioes_matrix(np.ndarray[i32, ndim=1] birth,
                        np.ndarray[i32, ndim=1] death,
                        i32 T):
    """Build the full N × T BIOES tag matrix in one nogil pass.

    Parameters
    ----------
    birth : i32[N]   - per-entity first-seen snapshot index
    death : i32[N]   - per-entity snapshot AFTER last presence (-1 = alive)
    T     : int      - total number of snapshots

    Returns
    -------
    tags : i32[N, T]  - tag matrix (0=B 1=I 2=O 3=E 4=S)
    """
    cdef Py_ssize_t N = birth.shape[0]
    cdef np.ndarray[i32, ndim=2] tags = np.full((N, T), TAG_O, dtype=np.int32)
    cdef i32 *tptr = <i32 *>tags.data
    cdef i32 *bptr = <i32 *>birth.data
    cdef i32 *dptr = <i32 *>death.data
    cdef Py_ssize_t i
    cdef i32 b, d

    with nogil:
        for i in range(N):
            b = bptr[i]
            d = dptr[i]
            if d < 0:
                d = T
            _tag_contiguous(tptr + i * T, b, d)

    return tags


# entity_bioes_gapped

def entity_bioes_gapped(list snapshots,
                        np.ndarray[i64, ndim=1] edge_ids,
                        bint directed=False):
    """Gap-aware per-entity BIOES tagging.

    Tracks actual per-snapshot presence and creates separate B-I-E
    spans for each contiguous appearance.  An entity absent in the
    middle of its lifespan gets O tags during the gap.

    Parameters
    ----------
    snapshots  : list of (i32 src, i32 tgt) per timestep
    edge_ids   : i64[N] sorted unique IDs from edge_lifecycle
    directed   : bool

    Returns
    -------
    tags    : i32[N, T]
    n_spans : i32[N]  - contiguous-appearance count per entity
    """
    cdef Py_ssize_t T = len(snapshots), N = edge_ids.shape[0]
    cdef Py_ssize_t t, j, nE

    # Presence matrix  (row-major, N × T)
    cdef np.ndarray[np.uint8_t, ndim=2] presence = np.zeros(
        (N, T), dtype=np.uint8)
    cdef np.uint8_t *pptr = <np.uint8_t *>presence.data

    # Hash edge_ids -> row index for O(1) lookup
    cdef dict eid_map = {}
    cdef i64 *eidptr = <i64 *>edge_ids.data
    for j in range(N):
        eid_map[eidptr[j]] = j

    # Scan every snapshot and mark presence
    cdef i64 key
    cdef i32 s_val, t_val
    cdef np.ndarray[i32, ndim=1] s_arr, t_arr
    cdef i32 *sptr
    cdef i32 *tptr_s  # renamed to avoid shadow
    cdef Py_ssize_t row_idx

    for t in range(T):
        s_arr = np.ascontiguousarray(snapshots[t][0], dtype=np.int32)
        t_arr = np.ascontiguousarray(snapshots[t][1], dtype=np.int32)
        nE = s_arr.shape[0]
        sptr = <i32 *>s_arr.data
        tptr_s = <i32 *>t_arr.data

        for j in range(nE):
            s_val = sptr[j]
            t_val = tptr_s[j]
            if directed:
                key = _encode_directed(s_val, t_val)
            else:
                key = _encode_undirected(s_val, t_val)

            if key in eid_map:
                row_idx = <Py_ssize_t>eid_map[key]
                pptr[row_idx * T + t] = 1

    # Tag from presence (pure C inner loop)
    cdef np.ndarray[i32, ndim=2] tags = np.full((N, T), TAG_O, dtype=np.int32)
    cdef i32 *tags_ptr = <i32 *>tags.data
    cdef np.ndarray[i32, ndim=1] n_spans_arr = np.zeros(N, dtype=np.int32)
    cdef i32 *ns_ptr = <i32 *>n_spans_arr.data

    with nogil:
        for j in range(N):
            _tag_gapped(pptr + j * T,
                        tags_ptr + j * T,
                        ns_ptr + j,
                        <i32>T)

    return tags, n_spans_arr


# vertex_lifecycle

def vertex_lifecycle(list snapshots, bint directed=False):
    """Per-vertex birth and death times across snapshots.

    Returns
    -------
    vertex_ids : i32[M]
    birth      : i32[M]
    death      : i32[M]   (-1 = alive at end)
    """
    cdef Py_ssize_t T = len(snapshots), t, j, nE
    cdef dict first_seen = {}
    cdef dict last_seen = {}
    cdef i32 v

    for t in range(T):
        src, tgt = snapshots[t]
        nE = src.shape[0]
        for j in range(nE):
            v = <i32>src[j]
            if v not in first_seen:
                first_seen[v] = t
            last_seen[v] = t
            v = <i32>tgt[j]
            if v not in first_seen:
                first_seen[v] = t
            last_seen[v] = t

    cdef Py_ssize_t n = len(first_seen)
    cdef np.ndarray[i32, ndim=1] vids  = np.empty(n, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] birth = np.empty(n, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] death = np.empty(n, dtype=np.int32)
    cdef i32 *vp = <i32 *>vids.data
    cdef i32 *bp = <i32 *>birth.data
    cdef i32 *dp = <i32 *>death.data

    cdef Py_ssize_t idx = 0
    for key in sorted(first_seen.keys()):
        vp[idx] = <i32>key
        bp[idx] = <i32>first_seen[key]
        ls = last_seen[key]
        dp[idx] = -1 if ls == T - 1 else <i32>(ls + 1)
        idx += 1

    return vids, birth, death


# cross_document_stats

def cross_document_stats(np.ndarray[i32, ndim=1] birth,
                         np.ndarray[i32, ndim=1] death,
                         np.ndarray[i32, ndim=1] doc_boundaries,
                         i32 T):
    """Summary statistics with document-boundary awareness.

    Parameters
    ----------
    doc_boundaries : i32[n_docs - 1]
        Chunk indices where each subsequent document begins.
    """
    cdef Py_ssize_t N = birth.shape[0]
    cdef Py_ssize_t n_bounds = doc_boundaries.shape[0]

    # Map chunk -> doc id
    cdef np.ndarray[i32, ndim=1] chunk_doc = np.zeros(T, dtype=np.int32)
    cdef i32 *cdp = <i32 *>chunk_doc.data
    cdef i32 *dbp = <i32 *>doc_boundaries.data
    cdef i32 cur_doc = 0
    cdef Py_ssize_t bi = 0
    cdef Py_ssize_t i

    for i in range(T):
        if bi < n_bounds and i >= dbp[bi]:
            cur_doc += 1
            bi += 1
        cdp[i] = cur_doc

    # Histogram + classify
    cdef np.ndarray[i32, ndim=1] hist = np.zeros(T + 1, dtype=np.int32)
    cdef i32 *hp = <i32 *>hist.data
    cdef i32 *bptr = <i32 *>birth.data
    cdef i32 *dptr = <i32 *>death.data
    cdef i32 b, d, span
    cdef i32 n_cross = 0, n_within = 0, n_single = 0

    # Histogram pass (nogil)
    with nogil:
        for i in range(N):
            d = dptr[i]
            if d < 0:
                d = T
            span = d - bptr[i]
            if span > 0 and span <= T:
                hp[span] += 1

    # Classification pass (needs list append)
    cdef list cross_lifespans = []
    for i in range(N):
        b = bptr[i]
        d = dptr[i]
        if d < 0:
            d = T
        span = d - b
        if span <= 0:
            continue
        if span == 1:
            n_single += 1
            n_within += 1
        else:
            if cdp[b] != cdp[min(d - 1, T - 1)]:
                n_cross += 1
                cross_lifespans.append(span)
            else:
                n_within += 1

    cdef f64 mean_ls = 0.0
    if N > 0:
        for i in range(N):
            d = dptr[i]
            if d < 0:
                d = T
            mean_ls += <f64>(d - bptr[i])
        mean_ls /= <f64>N

    return {
        "n_total": <int>N,
        "n_cross_doc": <int>n_cross,
        "n_within_doc": <int>n_within,
        "n_singleton": <int>n_single,
        "lifespan_histogram": hist,
        "cross_doc_lifespans": np.array(cross_lifespans, dtype=np.int32),
        "mean_lifespan": <float>mean_ls,
    }


# persistence_spectrum

def persistence_spectrum(np.ndarray[i32, ndim=1] birth,
                         np.ndarray[i32, ndim=1] death,
                         i32 T):
    """Sorted lifespans + birth-death pairs for persistence analysis."""
    cdef Py_ssize_t N = birth.shape[0], i
    cdef np.ndarray[f64, ndim=1] lifespans = np.empty(N, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] pairs     = np.empty((N, 2), dtype=np.float64)
    cdef f64 *lp = <f64 *>lifespans.data
    cdef f64 *pp = <f64 *>pairs.data
    cdef i32 *bp = <i32 *>birth.data
    cdef i32 *dp = <i32 *>death.data
    cdef i32 d

    with nogil:
        for i in range(N):
            d = dp[i]
            if d < 0:
                d = T
            lp[i]       = <f64>(d - bp[i])
            pp[2*i]     = <f64>bp[i]
            pp[2*i + 1] = <f64>d

    order = np.argsort(-lifespans)
    return lifespans[order], pairs[order]
