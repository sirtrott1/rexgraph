# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._temporal - Temporal bundle storage, BIOES phase detection,
and lifecycle tracking for temporal rexgraphs.

Delta-encoded snapshot storage with adaptive checkpoints. Sorted-merge
edge and face diffing between consecutive snapshots. BIOES phase
tagging on Betti timeseries (B=begin, I=inside, O=outside, E=end,
S=single). Face tracking via exact boundary match and Jaccard overlap.

Energy-ratio BIOES tagging detects phases from the E_kin/E_pot ratio
under the Relational Laplacian RL_1 = L_1 + alpha_G * L_O. Phases
correspond to kinetic-dominated (topological), crossover, and
potential-dominated (geometric) regimes.

Cascade event tracking records edge activation order during signal
propagation, enabling analysis of perturbation spread through the
chain complex.

General boundary variants handle branching edges, self-loops, and
witness edges alongside standard 2-endpoint edges.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memset, memcpy
from libc.math cimport fabs, log

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
)

np.import_array()

cdef enum:
    TAG_B = 0
    TAG_I = 1
    TAG_O = 2
    TAG_E = 3
    TAG_S = 4

cdef enum:
    FACE_PERSIST = 0
    FACE_BORN    = 1
    FACE_DIED    = 2
    FACE_SPLIT   = 3
    FACE_MERGE   = 4


# Edge encoding

cdef inline i64 _encode_edge_undirected_i32(i32 s, i32 t) noexcept nogil:
    if s <= t: return (<i64>s) * 2147483648LL + <i64>t
    return (<i64>t) * 2147483648LL + <i64>s

cdef inline i64 _encode_edge_directed_i32(i32 s, i32 t) noexcept nogil:
    return (<i64>s) * 2147483648LL + <i64>t

cdef inline i64 _encode_edge_undirected_i64(i64 s, i64 t) noexcept nogil:
    if s <= t: return s * 4294967296LL + t
    return t * 4294967296LL + s

cdef inline i64 _encode_edge_directed_i64(i64 s, i64 t) noexcept nogil:
    return s * 4294967296LL + t


# Temporal bundle: delta encoding

def encode_snapshot_delta_i32(np.ndarray[i32, ndim=1] prev_src,
                               np.ndarray[i32, ndim=1] prev_tgt,
                               np.ndarray[i32, ndim=1] curr_src,
                               np.ndarray[i32, ndim=1] curr_tgt,
                               bint directed=False):
    """
    Compute delta from previous snapshot to current.

    Encodes edges as canonical ints, sorts, does merge-diff.

    Returns
    -------
    born_src, born_tgt : int32[n_born]
    died_src, died_tgt : int32[n_died]
    """
    cdef Py_ssize_t nP = prev_src.shape[0], nC = curr_src.shape[0]
    cdef i32[::1] ps = prev_src, pt = prev_tgt, cs = curr_src, ct = curr_tgt
    cdef Py_ssize_t j

    cdef np.ndarray[i64, ndim=1] prev_enc = np.empty(nP, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] curr_enc = np.empty(nC, dtype=np.int64)
    cdef i64[::1] pe = prev_enc, ce = curr_enc
    if directed:
        for j in range(nP): pe[j] = _encode_edge_directed_i32(ps[j], pt[j])
        for j in range(nC): ce[j] = _encode_edge_directed_i32(cs[j], ct[j])
    else:
        for j in range(nP): pe[j] = _encode_edge_undirected_i32(ps[j], pt[j])
        for j in range(nC): ce[j] = _encode_edge_undirected_i32(cs[j], ct[j])

    prev_enc.sort()
    curr_enc.sort()

    cdef np.ndarray[i32, ndim=1] b_s = np.empty(nC, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] b_t = np.empty(nC, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] d_s = np.empty(nP, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] d_t = np.empty(nP, dtype=np.int32)
    cdef i32[::1] bsv = b_s, btv = b_t, dsv = d_s, dtv = d_t
    cdef Py_ssize_t ip = 0, ic = 0, nb = 0, nd = 0

    while ip < nP and ic < nC:
        if pe[ip] < ce[ic]:
            dsv[nd] = ps[ip]; dtv[nd] = pt[ip]; nd += 1; ip += 1
        elif pe[ip] > ce[ic]:
            bsv[nb] = cs[ic]; btv[nb] = ct[ic]; nb += 1; ic += 1
        else:
            ip += 1; ic += 1
    while ip < nP:
        dsv[nd] = ps[ip]; dtv[nd] = pt[ip]; nd += 1; ip += 1
    while ic < nC:
        bsv[nb] = cs[ic]; btv[nb] = ct[ic]; nb += 1; ic += 1

    return b_s[:nb], b_t[:nb], d_s[:nd], d_t[:nd]


def encode_snapshot_delta_i64(np.ndarray[i64, ndim=1] prev_src,
                               np.ndarray[i64, ndim=1] prev_tgt,
                               np.ndarray[i64, ndim=1] curr_src,
                               np.ndarray[i64, ndim=1] curr_tgt,
                               bint directed=False):
    """Compute delta from previous to current. int64 variant."""
    cdef Py_ssize_t nP = prev_src.shape[0], nC = curr_src.shape[0]
    cdef i64[::1] ps = prev_src, pt = prev_tgt, cs = curr_src, ct = curr_tgt
    cdef Py_ssize_t j

    cdef np.ndarray[i64, ndim=1] prev_enc = np.empty(nP, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] curr_enc = np.empty(nC, dtype=np.int64)
    cdef i64[::1] pe = prev_enc, ce = curr_enc
    if directed:
        for j in range(nP): pe[j] = _encode_edge_directed_i64(ps[j], pt[j])
        for j in range(nC): ce[j] = _encode_edge_directed_i64(cs[j], ct[j])
    else:
        for j in range(nP): pe[j] = _encode_edge_undirected_i64(ps[j], pt[j])
        for j in range(nC): ce[j] = _encode_edge_undirected_i64(cs[j], ct[j])

    prev_enc.sort()
    curr_enc.sort()

    cdef np.ndarray[i64, ndim=1] b_s = np.empty(nC, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] b_t = np.empty(nC, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] d_s = np.empty(nP, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] d_t = np.empty(nP, dtype=np.int64)
    cdef i64[::1] bsv = b_s, btv = b_t, dsv = d_s, dtv = d_t
    cdef Py_ssize_t ip = 0, ic = 0, nb = 0, nd = 0

    while ip < nP and ic < nC:
        if pe[ip] < ce[ic]:
            dsv[nd] = ps[ip]; dtv[nd] = pt[ip]; nd += 1; ip += 1
        elif pe[ip] > ce[ic]:
            bsv[nb] = cs[ic]; btv[nb] = ct[ic]; nb += 1; ic += 1
        else:
            ip += 1; ic += 1
    while ip < nP:
        dsv[nd] = ps[ip]; dtv[nd] = pt[ip]; nd += 1; ip += 1
    while ic < nC:
        bsv[nb] = cs[ic]; btv[nb] = ct[ic]; nb += 1; ic += 1

    return b_s[:nb], b_t[:nb], d_s[:nd], d_t[:nd]


def encode_snapshot_delta(prev_src, prev_tgt, curr_src, curr_tgt, directed=False):
    """Auto-dispatch by dtype."""
    if prev_src.dtype == np.int64:
        return encode_snapshot_delta_i64(prev_src, prev_tgt, curr_src, curr_tgt, directed)
    return encode_snapshot_delta_i32(prev_src, prev_tgt, curr_src, curr_tgt, directed)


# Temporal index

def build_temporal_index_i32(list snapshots, bint directed=False,
                              double checkpoint_threshold=0.5):
    """
    Build adaptive checkpoint index from a list of snapshots.

    Each snapshot is (sources_i32, targets_i32).
    Checkpoints are stored when cumulative delta exceeds
    checkpoint_threshold * current_edge_count.

    Returns
    -------
    checkpoints : list of (time, sources, targets)
    deltas : list of (time, born_src, born_tgt, died_src, died_tgt)
    checkpoint_times : int32[n_checkpoints]
    """
    cdef Py_ssize_t T = len(snapshots), t
    cdef Py_ssize_t cumulative_delta, nE_curr

    checkpoints = []
    deltas = []
    cp_times = []

    if T == 0:
        return checkpoints, deltas, np.array([], dtype=np.int32)

    s0, t0 = snapshots[0]
    checkpoints.append((0, s0.copy(), t0.copy()))
    cp_times.append(0)
    cumulative_delta = 0

    for t in range(1, T):
        sc, tc = snapshots[t]
        sp, tp = snapshots[t - 1]

        bs, bt, ds, dt = encode_snapshot_delta_i32(sp, tp, sc, tc, directed)
        n_born = bs.shape[0]
        n_died = ds.shape[0]
        cumulative_delta += n_born + n_died
        nE_curr = sc.shape[0]

        if nE_curr > 0 and <f64>cumulative_delta / <f64>nE_curr > checkpoint_threshold:
            checkpoints.append((t, sc.copy(), tc.copy()))
            cp_times.append(t)
            cumulative_delta = 0
            deltas.append((t, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
                           np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)))
        else:
            deltas.append((t, bs, bt, ds, dt))

    return checkpoints, deltas, np.array(cp_times, dtype=np.int32)


def build_temporal_index_i64(list snapshots, bint directed=False,
                              double checkpoint_threshold=0.5):
    """Build adaptive checkpoint index. int64 variant."""
    cdef Py_ssize_t T = len(snapshots), t
    cdef Py_ssize_t cumulative_delta, nE_curr

    checkpoints = []
    deltas = []
    cp_times = []

    if T == 0:
        return checkpoints, deltas, np.array([], dtype=np.int64)

    s0, t0 = snapshots[0]
    checkpoints.append((0, s0.copy(), t0.copy()))
    cp_times.append(0)
    cumulative_delta = 0

    for t in range(1, T):
        sc, tc = snapshots[t]
        sp, tp = snapshots[t - 1]

        bs, bt, ds, dt = encode_snapshot_delta_i64(sp, tp, sc, tc, directed)
        n_born = bs.shape[0]
        n_died = ds.shape[0]
        cumulative_delta += n_born + n_died
        nE_curr = sc.shape[0]

        if nE_curr > 0 and <f64>cumulative_delta / <f64>nE_curr > checkpoint_threshold:
            checkpoints.append((t, sc.copy(), tc.copy()))
            cp_times.append(t)
            cumulative_delta = 0
            deltas.append((t, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                           np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
        else:
            deltas.append((t, bs, bt, ds, dt))

    return checkpoints, deltas, np.array(cp_times, dtype=np.int64)


def build_temporal_index(snapshots, directed=False, checkpoint_threshold=0.5):
    """Auto-dispatch by dtype of first snapshot."""
    if len(snapshots) == 0:
        return [], [], np.array([], dtype=np.int32)
    if snapshots[0][0].dtype == np.int64:
        return build_temporal_index_i64(snapshots, directed, checkpoint_threshold)
    return build_temporal_index_i32(snapshots, directed, checkpoint_threshold)


# Edge lifecycle tracking

def edge_lifecycle_i32(list snapshots, bint directed=False):
    """
    Compute per-edge birth and death times across all snapshots.

    Returns
    -------
    edge_ids : int64[n_unique]
    birth    : int32[n_unique]
    death    : int32[n_unique]
        -1 if alive at end.
    """
    cdef Py_ssize_t T = len(snapshots), t, j, nE
    cdef dict first_seen = {}
    cdef dict last_seen = {}

    for t in range(T):
        src, tgt = snapshots[t]
        nE = src.shape[0]
        for j in range(nE):
            if directed:
                key = _encode_edge_directed_i32(src[j], tgt[j])
            else:
                key = _encode_edge_undirected_i32(src[j], tgt[j])
            if key not in first_seen:
                first_seen[key] = t
            last_seen[key] = t

    cdef Py_ssize_t n = len(first_seen)
    cdef np.ndarray[i64, ndim=1] eids = np.empty(n, dtype=np.int64)
    cdef np.ndarray[i32, ndim=1] birth = np.empty(n, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] death = np.empty(n, dtype=np.int32)
    cdef i64[::1] ev = eids
    cdef i32[::1] bv = birth, dv = death

    cdef Py_ssize_t idx = 0
    for key in sorted(first_seen.keys()):
        ev[idx] = key
        bv[idx] = <i32>first_seen[key]
        ls = last_seen[key]
        dv[idx] = -1 if ls == T - 1 else <i32>(ls + 1)
        idx += 1

    return eids, birth, death


def edge_lifecycle_i64(list snapshots, bint directed=False):
    """Per-edge birth/death times. int64 variant."""
    cdef Py_ssize_t T = len(snapshots), t, j, nE
    cdef dict first_seen = {}
    cdef dict last_seen = {}

    for t in range(T):
        src, tgt = snapshots[t]
        nE = src.shape[0]
        for j in range(nE):
            if directed:
                key = _encode_edge_directed_i64(src[j], tgt[j])
            else:
                key = _encode_edge_undirected_i64(src[j], tgt[j])
            if key not in first_seen:
                first_seen[key] = t
            last_seen[key] = t

    cdef Py_ssize_t n = len(first_seen)
    cdef np.ndarray[i64, ndim=1] eids = np.empty(n, dtype=np.int64)
    cdef np.ndarray[i32, ndim=1] birth = np.empty(n, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] death = np.empty(n, dtype=np.int32)

    cdef Py_ssize_t idx = 0
    for key in sorted(first_seen.keys()):
        eids[idx] = key
        birth[idx] = <i32>first_seen[key]
        ls = last_seen[key]
        death[idx] = -1 if ls == T - 1 else <i32>(ls + 1)
        idx += 1

    return eids, birth, death


def edge_lifecycle(snapshots, directed=False):
    """Auto-dispatch by dtype."""
    if len(snapshots) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    if snapshots[0][0].dtype == np.int64:
        return edge_lifecycle_i64(snapshots, directed)
    return edge_lifecycle_i32(snapshots, directed)


# BIOES phase detection

def compute_edge_metrics(list snapshots, bint directed=False):
    """
    Compute per-timestep edge metrics for BIOES tagging.

    Returns
    -------
    edge_counts : int32[T]
    born_counts : int32[T]
    died_counts : int32[T]
    """
    cdef Py_ssize_t T = len(snapshots), t
    cdef np.ndarray[i32, ndim=1] ec = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] bc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] dc = np.zeros(T, dtype=np.int32)
    cdef i32[::1] ecv = ec, bcv = bc, dcv = dc

    for t in range(T):
        src, tgt = snapshots[t]
        ecv[t] = <i32>src.shape[0]

    for t in range(1, T):
        sp, tp = snapshots[t - 1]
        sc, tc = snapshots[t]
        bs, bt, ds, dt = encode_snapshot_delta(sp, tp, sc, tc, directed)
        bcv[t] = <i32>bs.shape[0]
        dcv[t] = <i32>ds.shape[0]

    return ec, bc, dc


def detect_phases(np.ndarray[i64, ndim=1] beta0,
                  np.ndarray[i64, ndim=1] beta1,
                  double tol=0.0):
    """
    Detect structural phases from Betti number sequences.

    A phase is a maximal contiguous interval [a, b] where both
    beta_0 and beta_1 remain constant (within tol).

    Returns
    -------
    phase_start : int32[n_phases]
    phase_end   : int32[n_phases]  (inclusive)
    phase_b0    : int64[n_phases]  (beta_0 value during phase)
    phase_b1    : int64[n_phases]  (beta_1 value during phase)
    """
    cdef Py_ssize_t T = beta0.shape[0], t
    cdef i64[::1] b0 = beta0, b1 = beta1

    if T == 0:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    cdef list starts = []
    cdef list ends = []
    cdef list pb0 = []
    cdef list pb1 = []

    cdef Py_ssize_t phase_begin = 0
    cdef i64 cur_b0 = b0[0], cur_b1 = b1[0]

    for t in range(1, T):
        if fabs(<f64>(b0[t] - cur_b0)) > tol or fabs(<f64>(b1[t] - cur_b1)) > tol:
            starts.append(phase_begin)
            ends.append(t - 1)
            pb0.append(cur_b0)
            pb1.append(cur_b1)
            phase_begin = t
            cur_b0 = b0[t]; cur_b1 = b1[t]

    starts.append(phase_begin)
    ends.append(T - 1)
    pb0.append(cur_b0)
    pb1.append(cur_b1)

    return (np.array(starts, dtype=np.int32), np.array(ends, dtype=np.int32),
            np.array(pb0, dtype=np.int64), np.array(pb1, dtype=np.int64))


def assign_bioes_tags(Py_ssize_t T,
                      np.ndarray[i32, ndim=1] phase_start,
                      np.ndarray[i32, ndim=1] phase_end,
                      Py_ssize_t min_phase_len=2):
    """
    Assign BIOES tags to each timestep from detected phases.

    Phases shorter than min_phase_len are outside-tagged unless
    they span exactly 1 step (S-tagged).  Multi-step phases get
    B at start, E at end, I in between.

    Returns
    -------
    tags : int32[T]
        0=B, 1=I, 2=O, 3=E, 4=S.
    """
    cdef np.ndarray[i32, ndim=1] tags = np.full(T, TAG_O, dtype=np.int32)
    cdef i32[::1] tv = tags
    cdef i32[::1] ps = phase_start, pe = phase_end
    cdef Py_ssize_t nP = phase_start.shape[0], p, t
    cdef Py_ssize_t a, b, plen

    for p in range(nP):
        a = ps[p]; b = pe[p]; plen = b - a + 1
        if plen < min_phase_len:
            if plen == 1:
                tv[a] = TAG_S
            # else leave as O
            continue
        tv[a] = TAG_B
        tv[b] = TAG_E
        for t in range(a + 1, b):
            tv[t] = TAG_I

    return tags


def compute_bioes_full(list snapshots,
                       np.ndarray[i64, ndim=1] beta0,
                       np.ndarray[i64, ndim=1] beta1,
                       bint directed=False,
                       double phase_tol=0.0,
                       Py_ssize_t min_phase_len=2):
    """
    Full BIOES pipeline: edge metrics + phase detection + tagging.

    Parameters
    ----------
    snapshots : list of (sources, targets) per timestep
    beta0, beta1 : int64[T] precomputed Betti numbers
    directed : bool
    phase_tol : float
    min_phase_len : int

    Returns
    -------
    tags : int32[T]
    edge_counts : int32[T]
    born_counts : int32[T]
    died_counts : int32[T]
    phase_start : int32[n_phases]
    phase_end   : int32[n_phases]
    phase_b0    : int64[n_phases]
    phase_b1    : int64[n_phases]
    """
    cdef Py_ssize_t T = len(snapshots)

    ec, bc, dc = compute_edge_metrics(snapshots, directed)
    p_start, p_end, p_b0, p_b1 = detect_phases(beta0, beta1, phase_tol)
    tags = assign_bioes_tags(T, p_start, p_end, min_phase_len)

    return tags, ec, bc, dc, p_start, p_end, p_b0, p_b1


def detect_phases_kd(np.ndarray[i64, ndim=2] betti_matrix,
                     double tol=0.0):
    """
    Detect structural phases from multi-dimensional Betti sequences.

    A phase breaks when ANY tracked Betti number shifts beyond tol.
    This unifies edge-level (beta_0, beta_1) and face-level (beta_2)
    regime detection into a single pass.

    Parameters
    ----------
    betti_matrix : int64[T, K]
        Row t holds [beta_0(t), beta_1(t), ..., beta_{K-1}(t)].

    Returns
    -------
    phase_start : int32[n_phases]
    phase_end   : int32[n_phases]
    phase_betti : int64[n_phases, K]  (Betti values during each phase)
    """
    cdef Py_ssize_t T = betti_matrix.shape[0]
    cdef Py_ssize_t K = betti_matrix.shape[1]
    cdef i64[:, ::1] bm = betti_matrix
    cdef Py_ssize_t t, k
    cdef bint changed

    if T == 0:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                np.empty((0, K), dtype=np.int64))

    cdef list starts = []
    cdef list ends = []
    cdef list phase_vals = []

    cdef np.ndarray[i64, ndim=1] cur = betti_matrix[0].copy()
    cdef i64[::1] cv = cur
    cdef Py_ssize_t phase_begin = 0

    for t in range(1, T):
        changed = False
        for k in range(K):
            if fabs(<f64>(bm[t, k] - cv[k])) > tol:
                changed = True
                break
        if changed:
            starts.append(phase_begin)
            ends.append(t - 1)
            phase_vals.append(cur.copy())
            phase_begin = t
            for k in range(K): cv[k] = bm[t, k]

    starts.append(phase_begin)
    ends.append(T - 1)
    phase_vals.append(cur.copy())

    cdef Py_ssize_t nP = len(starts)
    cdef np.ndarray[i64, ndim=2] pv = np.empty((nP, K), dtype=np.int64)
    for p in range(nP):
        pv[p, :] = phase_vals[p]

    return (np.array(starts, dtype=np.int32),
            np.array(ends, dtype=np.int32), pv)


def detect_phases_with_events(np.ndarray[i64, ndim=2] betti_matrix,
                               np.ndarray[i32, ndim=1] face_born,
                               np.ndarray[i32, ndim=1] face_died,
                               np.ndarray[i32, ndim=1] face_split,
                               np.ndarray[i32, ndim=1] face_merge,
                               double betti_tol=0.0,
                               Py_ssize_t event_threshold=1):
    """
    Detect phases using both Betti stability AND face event quiescence.

    A phase breaks when:
      (a) any Betti number shifts, OR
      (b) face structural events (born+died+split+merge) exceed threshold.


    Parameters
    ----------
    betti_matrix : int64[T, K]
    face_born, face_died, face_split, face_merge : int32[T]
    event_threshold : int

    Returns
    -------
    phase_start, phase_end : int32[n_phases]
    phase_betti : int64[n_phases, K]
    break_reasons : int32[n_phases]
        0=start, 1=betti_shift, 2=face_event, 3=both
    """
    cdef Py_ssize_t T = betti_matrix.shape[0]
    cdef Py_ssize_t K = betti_matrix.shape[1]
    cdef i64[:, ::1] bm = betti_matrix
    cdef i32[::1] fb = face_born, fd = face_died, fs = face_split, fm = face_merge
    cdef Py_ssize_t t, k
    cdef bint betti_changed, event_exceeded
    cdef Py_ssize_t total_events

    if T == 0:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                np.empty((0, K), dtype=np.int64), np.array([], dtype=np.int32))

    cdef list starts = []
    cdef list ends = []
    cdef list phase_vals = []
    cdef list reasons = []

    cdef np.ndarray[i64, ndim=1] cur = betti_matrix[0].copy()
    cdef i64[::1] cv = cur
    cdef Py_ssize_t phase_begin = 0

    reasons.append(0)

    for t in range(1, T):
        betti_changed = False
        for k in range(K):
            if fabs(<f64>(bm[t, k] - cv[k])) > betti_tol:
                betti_changed = True
                break

        total_events = fb[t] + fd[t] + fs[t] + fm[t]
        event_exceeded = total_events >= event_threshold

        if betti_changed or event_exceeded:
            starts.append(phase_begin)
            ends.append(t - 1)
            phase_vals.append(cur.copy())
            phase_begin = t
            for k in range(K): cv[k] = bm[t, k]
            if betti_changed and event_exceeded:
                reasons.append(3)
            elif betti_changed:
                reasons.append(1)
            else:
                reasons.append(2)

    starts.append(phase_begin)
    ends.append(T - 1)
    phase_vals.append(cur.copy())

    cdef Py_ssize_t nP = len(starts)
    cdef np.ndarray[i64, ndim=2] pv = np.empty((nP, K), dtype=np.int64)
    for p in range(nP):
        pv[p, :] = phase_vals[p]

    return (np.array(starts, dtype=np.int32),
            np.array(ends, dtype=np.int32), pv,
            np.array(reasons, dtype=np.int32))


def assign_bioes_per_dimension(Py_ssize_t T,
                                np.ndarray[i64, ndim=2] betti_matrix,
                                double tol=0.0,
                                Py_ssize_t min_phase_len=2):
    """
    Assign independent BIOES tags per Betti dimension.

    Each Betti number gets its own phase detection and tagging.

    Parameters
    ----------
    T : int
    betti_matrix : int64[T, K]
    tol : float
    min_phase_len : int

    Returns
    -------
    tags : int32[T, K]
    """
    cdef Py_ssize_t K = betti_matrix.shape[1], k
    cdef np.ndarray[i32, ndim=2] tags = np.full((T, K), TAG_O, dtype=np.int32)

    for k in range(K):
        col = betti_matrix[:, k].copy()
        zero_col = np.zeros(T, dtype=np.int64)
        ps, pe, _, _ = detect_phases(col, zero_col, tol)
        dim_tags = assign_bioes_tags(T, ps, pe, min_phase_len)
        tags[:, k] = dim_tags

    return tags


def compute_bioes_unified(list edge_snapshots,
                          list face_snapshots,
                          np.ndarray[i64, ndim=2] betti_matrix,
                          bint directed=False,
                          double phase_tol=0.0,
                          Py_ssize_t min_phase_len=2,
                          Py_ssize_t face_event_threshold=1,
                          double jaccard_threshold=0.5):
    """
    Unified BIOES pipeline across all dimensions.


    Parameters
    ----------
    edge_snapshots : list of (sources, targets) per timestep
    face_snapshots : list of (B2_col_ptr, B2_row_idx) per timestep
        Empty list if no faces at any timestep.
    betti_matrix : int64[T, K]  (K=2 for 1-rex, K=3 for 2-rex)

    Returns
    -------
    unified_tags    : int32[T]
    per_dim_tags    : int32[T, K]
    edge_counts     : int32[T]
    edge_born       : int32[T]
    edge_died       : int32[T]
    face_counts     : int32[T]
    face_born       : int32[T]
    face_died       : int32[T]
    face_split      : int32[T]
    face_merge      : int32[T]
    phase_start     : int32[n_phases]
    phase_end       : int32[n_phases]
    phase_betti     : int64[n_phases, K]
    break_reasons   : int32[n_phases]
    """
    cdef Py_ssize_t T = len(edge_snapshots)

    ec, ebc, edc = compute_edge_metrics(edge_snapshots, directed)

    cdef np.ndarray[i32, ndim=1] f_counts = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] f_born = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] f_died = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] f_split = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] f_merge = np.zeros(T, dtype=np.int32)

    if len(face_snapshots) == T and T > 0:
        _, f_counts, _, f_born, f_died, f_split, f_merge = face_lifecycle(
            face_snapshots, edge_snapshots, directed, jaccard_threshold)

    p_start, p_end, p_betti, reasons = detect_phases_with_events(
        betti_matrix, f_born, f_died, f_split, f_merge,
        phase_tol, face_event_threshold)
    unified_tags = assign_bioes_tags(T, p_start, p_end, min_phase_len)

    per_dim_tags = assign_bioes_per_dimension(T, betti_matrix, phase_tol, min_phase_len)

    return (unified_tags, per_dim_tags,
            ec, ebc, edc,
            f_counts, f_born, f_died, f_split, f_merge,
            p_start, p_end, p_betti, reasons)


# Face tracking

def _encode_face_i32(np.ndarray[i32, ndim=1] B2_col_ptr,
                     np.ndarray[i32, ndim=1] B2_row_idx,
                     np.ndarray[i32, ndim=1] sources,
                     np.ndarray[i32, ndim=1] targets,
                     Py_ssize_t f, bint directed):
    """Encode face f as sorted tuple of canonical edge encodings."""
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef i32[::1] sv = sources, tv = targets
    cdef Py_ssize_t start = cp[f], end = cp[f + 1], n = end - start, k
    cdef np.ndarray[i64, ndim=1] enc = np.empty(n, dtype=np.int64)
    cdef i64[::1] ev = enc
    cdef i32 e
    for k in range(n):
        e = ri[start + k]
        if directed:
            ev[k] = _encode_edge_directed_i32(sv[e], tv[e])
        else:
            ev[k] = _encode_edge_undirected_i32(sv[e], tv[e])
    enc.sort()
    return enc


def _encode_face_i64(np.ndarray[i64, ndim=1] B2_col_ptr,
                     np.ndarray[i64, ndim=1] B2_row_idx,
                     np.ndarray[i64, ndim=1] sources,
                     np.ndarray[i64, ndim=1] targets,
                     Py_ssize_t f, bint directed):
    """Encode face. int64 variant."""
    cdef i64[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef i64[::1] sv = sources, tv = targets
    cdef Py_ssize_t start = <Py_ssize_t>cp[f], end = <Py_ssize_t>cp[f + 1]
    cdef Py_ssize_t n = end - start, k
    cdef np.ndarray[i64, ndim=1] enc = np.empty(n, dtype=np.int64)
    cdef i64[::1] ev = enc
    cdef i64 e
    for k in range(n):
        e = ri[start + k]
        if directed:
            ev[k] = _encode_edge_directed_i64(sv[<Py_ssize_t>e], tv[<Py_ssize_t>e])
        else:
            ev[k] = _encode_edge_undirected_i64(sv[<Py_ssize_t>e], tv[<Py_ssize_t>e])
    enc.sort()
    return enc


def track_faces_i32(np.ndarray[i32, ndim=1] B2_cp_prev,
                    np.ndarray[i32, ndim=1] B2_ri_prev,
                    np.ndarray[i32, ndim=1] src_prev,
                    np.ndarray[i32, ndim=1] tgt_prev,
                    np.ndarray[i32, ndim=1] B2_cp_curr,
                    np.ndarray[i32, ndim=1] B2_ri_curr,
                    np.ndarray[i32, ndim=1] src_curr,
                    np.ndarray[i32, ndim=1] tgt_curr,
                    bint directed=False,
                    double jaccard_threshold=0.5):
    """
    Track face correspondence between consecutive snapshots.

    Face identity = sorted canonical edge encoding of boundary.
    Exact match is PERSIST. Jaccard >= threshold is SPLIT/MERGE.

    Returns
    -------
    events : int32[max(nF_prev, nF_curr)]
    prev_to_curr : int32[nF_prev]
        -1 if died.
    curr_to_prev : int32[nF_curr]
        -1 if born.
    jaccard_scores : float64[nF_curr]
    """
    cdef Py_ssize_t nF_prev = B2_cp_prev.shape[0] - 1
    cdef Py_ssize_t nF_curr = B2_cp_curr.shape[0] - 1
    cdef Py_ssize_t fp, fc, k

    prev_faces = []
    for fp in range(nF_prev):
        prev_faces.append(_encode_face_i32(B2_cp_prev, B2_ri_prev,
                                            src_prev, tgt_prev, fp, directed))
    curr_faces = []
    for fc in range(nF_curr):
        curr_faces.append(_encode_face_i32(B2_cp_curr, B2_ri_curr,
                                            src_curr, tgt_curr, fc, directed))

    cdef dict prev_lookup = {}
    for fp in range(nF_prev):
        key = bytes(prev_faces[fp].data)
        prev_lookup[key] = fp

    cdef np.ndarray[i32, ndim=1] p2c = np.full(nF_prev, -1, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] c2p = np.full(nF_curr, -1, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] jacc = np.zeros(nF_curr, dtype=np.float64)
    cdef i32[::1] p2cv = p2c, c2pv = c2p
    cdef f64[::1] jv = jacc

    for fc in range(nF_curr):
        key = bytes(curr_faces[fc].data)
        if key in prev_lookup:
            fp = prev_lookup[key]
            c2pv[fc] = <i32>fp
            p2cv[fp] = <i32>fc
            jv[fc] = 1.0

    cdef f64 best_j, j_score
    cdef Py_ssize_t best_fp, inter_size, union_size
    for fc in range(nF_curr):
        if c2pv[fc] >= 0:
            continue
        curr_set = set(curr_faces[fc].tolist())
        best_j = 0.0; best_fp = -1
        for fp in range(nF_prev):
            if p2cv[fp] >= 0:
                continue
            prev_set = set(prev_faces[fp].tolist())
            inter_size = len(curr_set & prev_set)
            union_size = len(curr_set | prev_set)
            if union_size > 0:
                j_score = <f64>inter_size / <f64>union_size
                if j_score > best_j:
                    best_j = j_score; best_fp = fp
        jv[fc] = best_j
        if best_j >= jaccard_threshold and best_fp >= 0:
            c2pv[fc] = <i32>best_fp

    cdef Py_ssize_t max_nF = nF_prev if nF_prev > nF_curr else nF_curr
    cdef np.ndarray[i32, ndim=1] events_prev = np.full(nF_prev, FACE_DIED, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] events_curr = np.full(nF_curr, FACE_BORN, dtype=np.int32)
    cdef i32[::1] epv = events_prev, ecv = events_curr

    for fp in range(nF_prev):
        if p2cv[fp] >= 0:
            epv[fp] = FACE_PERSIST

    cdef dict prev_match_count = {}
    for fc in range(nF_curr):
        if c2pv[fc] >= 0:
            fp = c2pv[fc]
            if fp not in prev_match_count:
                prev_match_count[fp] = 0
            prev_match_count[fp] += 1

    for fp, count in prev_match_count.items():
        if count > 1:
            epv[fp] = FACE_SPLIT

    for fc in range(nF_curr):
        if c2pv[fc] >= 0:
            fp = c2pv[fc]
            if jv[fc] >= 1.0 - 1e-10:
                ecv[fc] = FACE_PERSIST
            elif prev_match_count.get(fp, 0) > 1:
                ecv[fc] = FACE_SPLIT
            else:
                ecv[fc] = FACE_MERGE

    return events_prev, events_curr, p2c, c2p, jacc


def track_faces_i64(np.ndarray[i64, ndim=1] B2_cp_prev,
                    np.ndarray[i64, ndim=1] B2_ri_prev,
                    np.ndarray[i64, ndim=1] src_prev,
                    np.ndarray[i64, ndim=1] tgt_prev,
                    np.ndarray[i64, ndim=1] B2_cp_curr,
                    np.ndarray[i64, ndim=1] B2_ri_curr,
                    np.ndarray[i64, ndim=1] src_curr,
                    np.ndarray[i64, ndim=1] tgt_curr,
                    bint directed=False,
                    double jaccard_threshold=0.5):
    """Track face correspondence. int64 variant."""
    cdef Py_ssize_t nF_prev = B2_cp_prev.shape[0] - 1
    cdef Py_ssize_t nF_curr = B2_cp_curr.shape[0] - 1
    cdef Py_ssize_t fp, fc

    prev_faces = []
    for fp in range(nF_prev):
        prev_faces.append(_encode_face_i64(B2_cp_prev, B2_ri_prev,
                                            src_prev, tgt_prev, fp, directed))
    curr_faces = []
    for fc in range(nF_curr):
        curr_faces.append(_encode_face_i64(B2_cp_curr, B2_ri_curr,
                                            src_curr, tgt_curr, fc, directed))

    cdef dict prev_lookup = {}
    for fp in range(nF_prev):
        prev_lookup[bytes(prev_faces[fp].data)] = fp

    cdef np.ndarray[i64, ndim=1] p2c = np.full(nF_prev, -1, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] c2p = np.full(nF_curr, -1, dtype=np.int64)
    cdef np.ndarray[f64, ndim=1] jacc = np.zeros(nF_curr, dtype=np.float64)
    cdef i64[::1] p2cv = p2c, c2pv = c2p
    cdef f64[::1] jv = jacc

    for fc in range(nF_curr):
        key = bytes(curr_faces[fc].data)
        if key in prev_lookup:
            fp = prev_lookup[key]
            c2pv[fc] = <i64>fp; p2cv[fp] = <i64>fc; jv[fc] = 1.0

    cdef f64 best_j, j_score
    cdef Py_ssize_t best_fp, inter_size, union_size
    for fc in range(nF_curr):
        if c2pv[fc] >= 0: continue
        curr_set = set(curr_faces[fc].tolist())
        best_j = 0.0; best_fp = -1
        for fp in range(nF_prev):
            if p2cv[fp] >= 0: continue
            prev_set = set(prev_faces[fp].tolist())
            inter_size = len(curr_set & prev_set)
            union_size = len(curr_set | prev_set)
            if union_size > 0:
                j_score = <f64>inter_size / <f64>union_size
                if j_score > best_j: best_j = j_score; best_fp = fp
        jv[fc] = best_j
        if best_j >= jaccard_threshold and best_fp >= 0:
            c2pv[fc] = <i64>best_fp

    cdef np.ndarray[i32, ndim=1] ep = np.full(nF_prev, FACE_DIED, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] ec = np.full(nF_curr, FACE_BORN, dtype=np.int32)
    cdef i32[::1] epv = ep, ecv = ec

    for fp in range(nF_prev):
        if p2cv[fp] >= 0: epv[fp] = FACE_PERSIST

    cdef dict pmc = {}
    for fc in range(nF_curr):
        if c2pv[fc] >= 0:
            fp = <Py_ssize_t>c2pv[fc]
            pmc[fp] = pmc.get(fp, 0) + 1

    for fp_key, count in pmc.items():
        if count > 1: epv[fp_key] = FACE_SPLIT

    for fc in range(nF_curr):
        if c2pv[fc] >= 0:
            fp = <Py_ssize_t>c2pv[fc]
            if jv[fc] >= 1.0 - 1e-10: ecv[fc] = FACE_PERSIST
            elif pmc.get(fp, 0) > 1: ecv[fc] = FACE_SPLIT
            else: ecv[fc] = FACE_MERGE

    return ep, ec, p2c, c2p, jacc


def track_faces(B2_cp_prev, B2_ri_prev, src_prev, tgt_prev,
                B2_cp_curr, B2_ri_curr, src_curr, tgt_curr,
                directed=False, jaccard_threshold=0.5):
    """Auto-dispatch by dtype."""
    if B2_cp_prev.dtype == np.int64:
        return track_faces_i64(B2_cp_prev, B2_ri_prev, src_prev, tgt_prev,
                               B2_cp_curr, B2_ri_curr, src_curr, tgt_curr,
                               directed, jaccard_threshold)
    return track_faces_i32(B2_cp_prev, B2_ri_prev, src_prev, tgt_prev,
                           B2_cp_curr, B2_ri_curr, src_curr, tgt_curr,
                           directed, jaccard_threshold)


def face_lifecycle_i32(list face_snapshots, list edge_snapshots,
                       bint directed=False, double jaccard_threshold=0.5):
    """
    Full face lifecycle across all timesteps.

    face_snapshots[t] = (B2_col_ptr, B2_row_idx) at time t
    edge_snapshots[t] = (sources, targets) at time t

    Returns
    -------
    events_per_step : list of (events_prev, events_curr, p2c, c2p, jaccard)
    face_counts     : int32[T]
    persist_counts  : int32[T]
    born_counts     : int32[T]
    died_counts     : int32[T]
    split_counts    : int32[T]
    merge_counts    : int32[T]
    """
    cdef Py_ssize_t T = len(face_snapshots), t
    cdef np.ndarray[i32, ndim=1] fc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] pc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] bc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] dc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] sc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] mc = np.zeros(T, dtype=np.int32)
    cdef i32[::1] fcv = fc, pcv = pc, bcv = bc, dcv = dc, scv = sc, mcv = mc
    cdef Py_ssize_t fi, nF_p, nF_c

    events_list = []

    for t in range(T):
        cp, ri = face_snapshots[t]
        nF = cp.shape[0] - 1
        fcv[t] = <i32>nF

    for t in range(1, T):
        cp_p, ri_p = face_snapshots[t - 1]
        cp_c, ri_c = face_snapshots[t]
        s_p, t_p = edge_snapshots[t - 1]
        s_c, t_c = edge_snapshots[t]

        ep, ec, p2c, c2p, jacc = track_faces_i32(
            cp_p, ri_p, s_p, t_p, cp_c, ri_c, s_c, t_c,
            directed, jaccard_threshold)
        events_list.append((ep, ec, p2c, c2p, jacc))

        nF_p = cp_p.shape[0] - 1
        nF_c = cp_c.shape[0] - 1
        for fi in range(nF_c):
            if ec[fi] == FACE_PERSIST: pcv[t] += 1
            elif ec[fi] == FACE_BORN: bcv[t] += 1
            elif ec[fi] == FACE_SPLIT: scv[t] += 1
            elif ec[fi] == FACE_MERGE: mcv[t] += 1
        for fi in range(nF_p):
            if ep[fi] == FACE_DIED: dcv[t] += 1

    return events_list, fc, pc, bc, dc, sc, mc


def face_lifecycle_i64(list face_snapshots, list edge_snapshots,
                       bint directed=False, double jaccard_threshold=0.5):
    """Full face lifecycle. int64 variant."""
    cdef Py_ssize_t T = len(face_snapshots), t
    cdef np.ndarray[i32, ndim=1] fc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] pc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] bc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] dc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] sc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] mc = np.zeros(T, dtype=np.int32)
    cdef i32[::1] fcv = fc, pcv = pc, bcv = bc, dcv = dc, scv = sc, mcv = mc
    cdef Py_ssize_t fi, nF_p, nF_c

    events_list = []

    for t in range(T):
        cp, ri = face_snapshots[t]
        fcv[t] = <i32>(cp.shape[0] - 1)

    for t in range(1, T):
        cp_p, ri_p = face_snapshots[t - 1]
        cp_c, ri_c = face_snapshots[t]
        s_p, t_p = edge_snapshots[t - 1]
        s_c, t_c = edge_snapshots[t]

        ep, ec, p2c, c2p, jacc = track_faces_i64(
            cp_p, ri_p, s_p, t_p, cp_c, ri_c, s_c, t_c,
            directed, jaccard_threshold)
        events_list.append((ep, ec, p2c, c2p, jacc))

        nF_c = cp_c.shape[0] - 1
        nF_p = cp_p.shape[0] - 1
        for fi in range(nF_c):
            if ec[fi] == FACE_PERSIST: pcv[t] += 1
            elif ec[fi] == FACE_BORN: bcv[t] += 1
            elif ec[fi] == FACE_SPLIT: scv[t] += 1
            elif ec[fi] == FACE_MERGE: mcv[t] += 1
        for fi in range(nF_p):
            if ep[fi] == FACE_DIED: dcv[t] += 1

    return events_list, fc, pc, bc, dc, sc, mc


def face_lifecycle(face_snapshots, edge_snapshots,
                   directed=False, jaccard_threshold=0.5):
    """Auto-dispatch by dtype of first edge snapshot."""
    if len(edge_snapshots) == 0:
        return [], np.array([], dtype=np.int32), np.array([], dtype=np.int32), \
               np.array([], dtype=np.int32), np.array([], dtype=np.int32), \
               np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    if edge_snapshots[0][0].dtype == np.int64:
        return face_lifecycle_i64(face_snapshots, edge_snapshots,
                                  directed, jaccard_threshold)
    return face_lifecycle_i32(face_snapshots, edge_snapshots,
                              directed, jaccard_threshold)


# Energy-ratio BIOES tagging

cdef enum:
    ENERGY_KINETIC   = 0
    ENERGY_CROSSOVER = 1
    ENERGY_POTENTIAL = 2


def detect_phases_energy_ratio(np.ndarray[f64, ndim=1] E_kin,
                                np.ndarray[f64, ndim=1] E_pot,
                                double ratio_tol=0.2,
                                double floor=1e-12):
    """Detect phases from the E_kin / E_pot energy ratio timeseries.

    The ratio r(t) = E_kin(t) / E_pot(t) under RL_1 diffusion
    classifies the signal regime:
        r >> 1 : kinetic (topological) dominated
        r ~ 1  : crossover
        r << 1 : potential (geometric) dominated

    A phase boundary occurs when the regime classification changes,
    where regimes are defined by log(r) crossing +/-ratio_tol around 0.

    Parameters
    ----------
    E_kin : f64[T] - topological energy <f|L_1|f> per timestep
    E_pot : f64[T] - geometric energy <f|L_O|f> per timestep
    ratio_tol : float - log-ratio threshold for crossover band
    floor : float - minimum energy to avoid division by zero

    Returns
    -------
    phase_start : int32[n_phases]
    phase_end : int32[n_phases] (inclusive)
    phase_regime : int32[n_phases] (ENERGY_KINETIC/CROSSOVER/POTENTIAL)
    log_ratios : f64[T]
    """
    cdef Py_ssize_t T = E_kin.shape[0], t
    cdef f64[::1] ek = E_kin, ep = E_pot
    cdef np.ndarray[f64, ndim=1] lr = np.empty(T, dtype=np.float64)
    cdef f64[::1] lrv = lr
    cdef f64 ek_t, ep_t

    # Compute log ratios
    for t in range(T):
        ek_t = ek[t] if ek[t] > floor else floor
        ep_t = ep[t] if ep[t] > floor else floor
        lrv[t] = log(ek_t / ep_t)

    if T == 0:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                np.array([], dtype=np.int32), lr)

    # Classify each timestep
    cdef np.ndarray[i32, ndim=1] regime = np.empty(T, dtype=np.int32)
    cdef i32[::1] rv = regime
    for t in range(T):
        if lrv[t] > ratio_tol:
            rv[t] = ENERGY_KINETIC
        elif lrv[t] < -ratio_tol:
            rv[t] = ENERGY_POTENTIAL
        else:
            rv[t] = ENERGY_CROSSOVER

    # Detect phase boundaries where regime changes
    cdef list starts = []
    cdef list ends = []
    cdef list regimes = []
    cdef Py_ssize_t phase_begin = 0
    cdef i32 cur_regime = rv[0]

    for t in range(1, T):
        if rv[t] != cur_regime:
            starts.append(phase_begin)
            ends.append(t - 1)
            regimes.append(cur_regime)
            phase_begin = t
            cur_regime = rv[t]

    starts.append(phase_begin)
    ends.append(T - 1)
    regimes.append(cur_regime)

    return (np.array(starts, dtype=np.int32),
            np.array(ends, dtype=np.int32),
            np.array(regimes, dtype=np.int32),
            lr)


def compute_bioes_energy(np.ndarray[f64, ndim=1] E_kin,
                         np.ndarray[f64, ndim=1] E_pot,
                         double ratio_tol=0.2,
                         Py_ssize_t min_phase_len=2,
                         double floor=1e-12):
    """Full BIOES pipeline from energy ratio timeseries.

    Combines energy-ratio phase detection with BIOES tag assignment.
    This is the energy-domain counterpart to the Betti-based
    compute_bioes_full.

    Parameters
    ----------
    E_kin : f64[T]
    E_pot : f64[T]
    ratio_tol : float
    min_phase_len : int
    floor : float

    Returns
    -------
    tags : int32[T] - BIOES tags (0=B, 1=I, 2=O, 3=E, 4=S)
    phase_start : int32[n_phases]
    phase_end : int32[n_phases]
    phase_regime : int32[n_phases]
    log_ratios : f64[T]
    crossover_times : int32[n_crossover]
        Timesteps where log_ratio crosses zero.
    """
    cdef Py_ssize_t T = E_kin.shape[0]

    p_start, p_end, p_regime, log_ratios = detect_phases_energy_ratio(
        E_kin, E_pot, ratio_tol, floor)
    tags = assign_bioes_tags(T, p_start, p_end, min_phase_len)

    # Find zero-crossings of log ratio
    cdef f64[::1] lrv = log_ratios
    cdef list crossings = []
    cdef Py_ssize_t t
    for t in range(1, T):
        if (lrv[t - 1] > 0 and lrv[t] <= 0) or (lrv[t - 1] < 0 and lrv[t] >= 0):
            crossings.append(t)

    return (tags, p_start, p_end, p_regime, log_ratios,
            np.array(crossings, dtype=np.int32))


def detect_phases_joint(np.ndarray[i64, ndim=2] betti_matrix,
                        np.ndarray[f64, ndim=1] E_kin,
                        np.ndarray[f64, ndim=1] E_pot,
                        double betti_tol=0.0,
                        double ratio_tol=0.2,
                        double floor=1e-12):
    """Detect phases using both Betti stability AND energy ratio regime.

    A phase breaks when:
      (a) any Betti number shifts, OR
      (b) the energy regime changes (kinetic/crossover/potential).

    This is the most comprehensive phase detector, combining topological
    invariant changes with continuous energy dynamics.

    Parameters
    ----------
    betti_matrix : int64[T, K]
    E_kin, E_pot : f64[T]
    betti_tol : float
    ratio_tol : float

    Returns
    -------
    phase_start, phase_end : int32[n_phases]
    phase_betti : int64[n_phases, K]
    phase_regime : int32[n_phases]
    break_reasons : int32[n_phases]
        0=start, 1=betti_shift, 2=energy_regime, 3=both
    log_ratios : f64[T]
    """
    cdef Py_ssize_t T = betti_matrix.shape[0]
    cdef Py_ssize_t K = betti_matrix.shape[1]
    cdef i64[:, ::1] bm = betti_matrix
    cdef f64[::1] ek = E_kin, ep = E_pot
    cdef Py_ssize_t t, k
    cdef bint betti_changed, regime_changed

    # Precompute log ratios and regimes
    cdef np.ndarray[f64, ndim=1] lr = np.empty(T, dtype=np.float64)
    cdef f64[::1] lrv = lr
    cdef np.ndarray[i32, ndim=1] regime = np.empty(T, dtype=np.int32)
    cdef i32[::1] rv = regime
    cdef f64 ek_t, ep_t

    for t in range(T):
        ek_t = ek[t] if ek[t] > floor else floor
        ep_t = ep[t] if ep[t] > floor else floor
        lrv[t] = log(ek_t / ep_t)
        if lrv[t] > ratio_tol:
            rv[t] = ENERGY_KINETIC
        elif lrv[t] < -ratio_tol:
            rv[t] = ENERGY_POTENTIAL
        else:
            rv[t] = ENERGY_CROSSOVER

    if T == 0:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                np.empty((0, K), dtype=np.int64), np.array([], dtype=np.int32),
                np.array([], dtype=np.int32), lr)

    cdef list starts = []
    cdef list ends = []
    cdef list phase_vals = []
    cdef list phase_regimes = []
    cdef list reasons = []

    cdef np.ndarray[i64, ndim=1] cur_betti = betti_matrix[0].copy()
    cdef i64[::1] cv = cur_betti
    cdef i32 cur_regime = rv[0]
    cdef Py_ssize_t phase_begin = 0

    reasons.append(0)  # first phase starts at t=0

    for t in range(1, T):
        betti_changed = False
        for k in range(K):
            if fabs(<f64>(bm[t, k] - cv[k])) > betti_tol:
                betti_changed = True
                break

        regime_changed = (rv[t] != cur_regime)

        if betti_changed or regime_changed:
            starts.append(phase_begin)
            ends.append(t - 1)
            phase_vals.append(cur_betti.copy())
            phase_regimes.append(cur_regime)
            phase_begin = t
            for k in range(K): cv[k] = bm[t, k]
            cur_regime = rv[t]
            if betti_changed and regime_changed:
                reasons.append(3)
            elif betti_changed:
                reasons.append(1)
            else:
                reasons.append(2)

    starts.append(phase_begin)
    ends.append(T - 1)
    phase_vals.append(cur_betti.copy())
    phase_regimes.append(cur_regime)

    cdef Py_ssize_t nP = len(starts)
    cdef np.ndarray[i64, ndim=2] pv = np.empty((nP, K), dtype=np.int64)
    for p in range(nP):
        pv[p, :] = phase_vals[p]

    return (np.array(starts, dtype=np.int32),
            np.array(ends, dtype=np.int32), pv,
            np.array(phase_regimes, dtype=np.int32),
            np.array(reasons, dtype=np.int32), lr)


# Cascade event tracking

def cascade_edge_activation(np.ndarray[f64, ndim=2] edge_signals,
                            double activation_threshold=0.01):
    """Record edge activation order during signal propagation.

    Given a timeseries of edge signals (from diffusion, wave, or
    Schrodinger evolution), determines the first timestep each edge
    exceeds the activation threshold.

    The signal is on the edge space because in the rex framework,
    edges are the primitive elements. Vertex activation can be derived
    via B_1 projection.

    Parameters
    ----------
    edge_signals : f64[T, nE]
        Edge signal magnitude at each timestep. For complex signals,
        pass np.abs(psi_E) or born_probabilities.
    activation_threshold : float
        Minimum signal magnitude to count as activated.

    Returns
    -------
    activation_time : int32[nE]
        First timestep edge exceeds threshold. -1 if never activated.
    activation_order : int32[n_activated]
        Edge indices sorted by activation time (earliest first).
    activation_rank : int32[nE]
        Rank of each edge in activation order. -1 if never activated.
    """
    cdef Py_ssize_t T = edge_signals.shape[0]
    cdef Py_ssize_t nE = edge_signals.shape[1]
    cdef f64[:, ::1] sv = edge_signals
    cdef Py_ssize_t t, e

    cdef np.ndarray[i32, ndim=1] act_time = np.full(nE, -1, dtype=np.int32)
    cdef i32[::1] atv = act_time

    for e in range(nE):
        for t in range(T):
            if sv[t, e] >= activation_threshold:
                atv[e] = <i32>t
                break

    # Build activation order using numpy argsort (avoids Python list sort)
    cdef np.ndarray[i32, ndim=1] mask_act = np.where(act_time >= 0, 1, 0).astype(np.int32)
    cdef Py_ssize_t n_act = 0
    for e in range(nE):
        if atv[e] >= 0:
            n_act += 1

    cdef np.ndarray[i32, ndim=1] order = np.empty(n_act, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] rank = np.full(nE, -1, dtype=np.int32)
    cdef i32[::1] ov = order, rkv = rank

    if n_act > 0:
        # Extract activated edge indices, sort by activation time
        act_edges = np.where(act_time >= 0)[0].astype(np.int32)
        act_times_sub = act_time[act_edges]
        sort_idx = np.argsort(act_times_sub, kind='mergesort')
        sorted_edges = act_edges[sort_idx]

        for r in range(n_act):
            e_idx = sorted_edges[r]
            ov[r] = <i32>e_idx
            rkv[e_idx] = <i32>r

    return act_time, order, rank


def cascade_wavefront(np.ndarray[f64, ndim=2] edge_signals,
                      np.ndarray[i32, ndim=1] edge_src,
                      np.ndarray[i32, ndim=1] edge_tgt,
                      double activation_threshold=0.01):
    """Track the wavefront of signal propagation through the edge space.

    At each timestep, identifies which edges are newly activated
    (crossed threshold since last step). This gives the expanding
    wavefront of signal spread.

    Parameters
    ----------
    edge_signals : f64[T, nE]
    edge_src, edge_tgt : int32[nE] - edge endpoints for vertex projection
    activation_threshold : float

    Returns
    -------
    wavefront : list of int32[] per timestep
        Edge indices newly activated at each step.
    cumulative : int32[T]
        Cumulative count of activated edges at each timestep.
    vertex_activation_time : int32[nV]
        First timestep any edge incident to vertex v activates. -1 if never.
        Derived from edge activation (V = boundary of E).
    """
    cdef Py_ssize_t T = edge_signals.shape[0]
    cdef Py_ssize_t nE = edge_signals.shape[1]
    cdef f64[:, ::1] sv = edge_signals
    cdef i32[::1] es = edge_src, et = edge_tgt
    cdef Py_ssize_t t, e

    # Determine nV from edge endpoints
    cdef i32 max_v = 0
    for e in range(nE):
        if es[e] > max_v: max_v = es[e]
        if et[e] > max_v: max_v = et[e]
    cdef Py_ssize_t nV = <Py_ssize_t>(max_v + 1)

    cdef np.ndarray[i32, ndim=1] cumul = np.zeros(T, dtype=np.int32)
    cdef i32[::1] cv = cumul
    cdef np.ndarray[i32, ndim=1] v_act = np.full(nV, -1, dtype=np.int32)
    cdef i32[::1] vav = v_act

    cdef bint* activated = <bint*>malloc(nE * sizeof(bint))
    cdef i32* wf_buf = <i32*>malloc(nE * sizeof(i32))
    if activated == NULL or wf_buf == NULL:
        if activated != NULL: free(activated)
        if wf_buf != NULL: free(wf_buf)
        raise MemoryError()

    cdef Py_ssize_t n_total = 0, wf_count
    cdef np.ndarray[i32, ndim=1] wf_arr
    try:
        for e in range(nE):
            activated[e] = False

        wavefront_list = []

        for t in range(T):
            wf_count = 0
            for e in range(nE):
                if not activated[e] and sv[t, e] >= activation_threshold:
                    activated[e] = True
                    wf_buf[wf_count] = <i32>e
                    wf_count += 1
                    n_total += 1
                    # V = boundary(E): vertex activates when its edge does
                    if vav[es[e]] < 0: vav[es[e]] = <i32>t
                    if vav[et[e]] < 0: vav[et[e]] = <i32>t

            wf_arr = np.empty(wf_count, dtype=np.int32)
            if wf_count > 0:
                memcpy(&wf_arr[0], wf_buf, wf_count * sizeof(i32))
            wavefront_list.append(wf_arr)
            cv[t] = <i32>n_total
    finally:
        free(activated)
        free(wf_buf)

    return wavefront_list, cumul, v_act


# General boundary temporal functions


def _edge_key_set(np.ndarray[i32, ndim=1] boundary_ptr,
                  np.ndarray[i32, ndim=1] boundary_idx):
    """
    Build set of canonical edge keys from general boundary.

    Returns
    -------
    keys : list of tuples
    key_set : set of tuples
    """
    cdef Py_ssize_t nE = boundary_ptr.shape[0] - 1, e
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx

    keys = []
    for e in range(nE):
        bverts = []
        for j in range(bp[e], bp[e + 1]):
            bverts.append(int(bi[j]))
        bverts.sort()
        keys.append(tuple(bverts))

    return keys, set(keys)


def encode_snapshot_delta_general(np.ndarray[i32, ndim=1] prev_bp,
                                  np.ndarray[i32, ndim=1] prev_bi,
                                  np.ndarray[i32, ndim=1] curr_bp,
                                  np.ndarray[i32, ndim=1] curr_bi):
    """
    Compute delta between two snapshots using general boundary representation.

    Edge identity = sorted tuple of boundary vertices.

    Parameters
    ----------
    prev_bp, prev_bi : boundary_ptr, boundary_idx for previous snapshot
    curr_bp, curr_bi : boundary_ptr, boundary_idx for current snapshot

    Returns
    -------
    born_keys : list of tuples
    died_keys : list of tuples
    n_born : int
    n_died : int
    """
    prev_keys, prev_set = _edge_key_set(prev_bp, prev_bi)
    curr_keys, curr_set = _edge_key_set(curr_bp, curr_bi)

    born_keys = [k for k in curr_keys if k not in prev_set]
    died_keys = [k for k in prev_keys if k not in curr_set]

    return born_keys, died_keys, len(born_keys), len(died_keys)


def build_temporal_index_general(list snapshots,
                                 double checkpoint_threshold=0.5):
    """
    Build adaptive checkpoint index from general boundary snapshots.

    Each snapshot is (boundary_ptr_i32, boundary_idx_i32).
    Checkpoints store full (bp, bi) arrays.

    Returns
    -------
    checkpoints : list of (time, boundary_ptr, boundary_idx)
    deltas : list of (time, born_keys, died_keys)
    checkpoint_times : int32[n_checkpoints]
    """
    cdef Py_ssize_t T = len(snapshots), t
    cdef Py_ssize_t cumulative_delta, nE_curr

    checkpoints = []
    deltas = []
    cp_times = []

    if T == 0:
        return checkpoints, deltas, np.array([], dtype=np.int32)

    bp0, bi0 = snapshots[0]
    checkpoints.append((0, bp0.copy(), bi0.copy()))
    cp_times.append(0)
    cumulative_delta = 0

    for t in range(1, T):
        bp_c, bi_c = snapshots[t]
        bp_p, bi_p = snapshots[t - 1]

        born_keys, died_keys, n_born, n_died = encode_snapshot_delta_general(
            bp_p, bi_p, bp_c, bi_c)
        cumulative_delta += n_born + n_died
        nE_curr = bp_c.shape[0] - 1

        if nE_curr > 0 and <f64>cumulative_delta / <f64>nE_curr > checkpoint_threshold:
            checkpoints.append((t, bp_c.copy(), bi_c.copy()))
            cp_times.append(t)
            cumulative_delta = 0
            deltas.append((t, [], []))
        else:
            deltas.append((t, born_keys, died_keys))

    return checkpoints, deltas, np.array(cp_times, dtype=np.int32)


def edge_lifecycle_general(list snapshots):
    """
    Compute per-edge birth and death times using general boundary.

    Edge identity = sorted tuple of all boundary vertices.

    Parameters
    ----------
    snapshots : list of (boundary_ptr, boundary_idx) per timestep

    Returns
    -------
    edge_keys  : list of tuples
    birth      : int32[n_unique]
    death      : int32[n_unique]
        -1 if alive at end.
    """
    cdef Py_ssize_t T = len(snapshots), t
    cdef dict first_seen = {}
    cdef dict last_seen = {}

    for t in range(T):
        bp, bi = snapshots[t]
        keys, _ = _edge_key_set(bp, bi)
        for key in keys:
            if key not in first_seen:
                first_seen[key] = t
            last_seen[key] = t

    cdef Py_ssize_t n = len(first_seen)
    sorted_keys = sorted(first_seen.keys())

    cdef np.ndarray[i32, ndim=1] birth = np.empty(n, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] death = np.empty(n, dtype=np.int32)
    cdef i32[::1] bv = birth, dv = death

    cdef Py_ssize_t idx = 0
    for key in sorted_keys:
        bv[idx] = <i32>first_seen[key]
        ls = last_seen[key]
        dv[idx] = -1 if ls == T - 1 else <i32>(ls + 1)
        idx += 1

    return sorted_keys, birth, death


def compute_edge_metrics_general(list snapshots):
    """
    Per-timestep edge counts and birth/death counts using general boundary.

    Returns
    -------
    edge_counts : int32[T]
    born_counts : int32[T]
    died_counts : int32[T]
    """
    cdef Py_ssize_t T = len(snapshots), t
    cdef np.ndarray[i32, ndim=1] ec = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] bc = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] dc = np.zeros(T, dtype=np.int32)
    cdef i32[::1] ecv = ec, bcv = bc, dcv = dc

    for t in range(T):
        bp, bi = snapshots[t]
        ecv[t] = <i32>(bp.shape[0] - 1)

    for t in range(1, T):
        bp_p, bi_p = snapshots[t - 1]
        bp_c, bi_c = snapshots[t]
        _, _, n_born, n_died = encode_snapshot_delta_general(bp_p, bi_p, bp_c, bi_c)
        bcv[t] = <i32>n_born
        dcv[t] = <i32>n_died

    return ec, bc, dc


def compute_bioes_general(list snapshots,
                          np.ndarray[i64, ndim=1] beta0,
                          np.ndarray[i64, ndim=1] beta1,
                          double phase_tol=0.0,
                          Py_ssize_t min_phase_len=2):
    """
    Full BIOES pipeline for general boundary snapshots.

    Parameters
    ----------
    snapshots : list of (boundary_ptr, boundary_idx) per timestep
    beta0, beta1 : int64[T] precomputed Betti numbers

    Returns
    -------
    Same as compute_bioes_full.
    """
    cdef Py_ssize_t T = len(snapshots)

    ec, bc, dc = compute_edge_metrics_general(snapshots)
    p_start, p_end, p_b0, p_b1 = detect_phases(beta0, beta1, phase_tol)
    tags = assign_bioes_tags(T, p_start, p_end, min_phase_len)

    return tags, ec, bc, dc, p_start, p_end, p_b0, p_b1


def _encode_face_general(np.ndarray[i32, ndim=1] B2_col_ptr,
                         np.ndarray[i32, ndim=1] B2_row_idx,
                         np.ndarray[i32, ndim=1] boundary_ptr,
                         np.ndarray[i32, ndim=1] boundary_idx):
    """
    Encode faces by their boundary edge keys (general boundary).

    Face identity = frozenset of edge keys, where each edge key is the
    sorted tuple of that edge's boundary vertices.
    """
    cdef Py_ssize_t nF = B2_col_ptr.shape[0] - 1
    cdef Py_ssize_t f, j, e
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx

    face_keys = []
    for f in range(nF):
        edge_keys = []
        for j in range(cp[f], cp[f + 1]):
            e = ri[j]
            bverts = []
            for k in range(bp[e], bp[e + 1]):
                bverts.append(int(bi[k]))
            bverts.sort()
            edge_keys.append(tuple(bverts))
        edge_keys.sort()
        face_keys.append(tuple(edge_keys))

    return face_keys


def track_faces_general(np.ndarray[i32, ndim=1] B2_cp_prev,
                        np.ndarray[i32, ndim=1] B2_ri_prev,
                        np.ndarray[i32, ndim=1] bp_prev,
                        np.ndarray[i32, ndim=1] bi_prev,
                        np.ndarray[i32, ndim=1] B2_cp_curr,
                        np.ndarray[i32, ndim=1] B2_ri_curr,
                        np.ndarray[i32, ndim=1] bp_curr,
                        np.ndarray[i32, ndim=1] bi_curr,
                        double jaccard_threshold=0.5):
    """
    Track faces across two snapshots using general boundary edge keys.

    Detects: persist, born, died, split, merge.

    Returns
    -------
    events : list of (event_type, prev_face_idx_or_-1, curr_face_idx_or_-1)
    n_persist, n_born, n_died, n_split, n_merge : int
    """
    prev_keys = _encode_face_general(B2_cp_prev, B2_ri_prev, bp_prev, bi_prev)
    curr_keys = _encode_face_general(B2_cp_curr, B2_ri_curr, bp_curr, bi_curr)

    prev_set_map = {}  # key -> face_idx
    for i, k in enumerate(prev_keys):
        prev_set_map[k] = i
    curr_set_map = {}
    for i, k in enumerate(curr_keys):
        curr_set_map[k] = i

    events = []
    n_persist = n_born = n_died = n_split = n_merge = 0

    matched_prev = set()
    matched_curr = set()

    for i, k in enumerate(curr_keys):
        if k in prev_set_map:
            events.append((FACE_PERSIST, prev_set_map[k], i))
            matched_prev.add(prev_set_map[k])
            matched_curr.add(i)
            n_persist += 1

    unmatched_prev = [i for i in range(len(prev_keys)) if i not in matched_prev]
    unmatched_curr = [i for i in range(len(curr_keys)) if i not in matched_curr]

    if unmatched_prev and unmatched_curr:
        prev_esets = [set(prev_keys[i]) for i in unmatched_prev]
        curr_esets = [set(curr_keys[i]) for i in unmatched_curr]

        used_curr = set()
        for pi_idx, pi in enumerate(unmatched_prev):
            children = []
            for ci_idx, ci in enumerate(unmatched_curr):
                if ci in used_curr:
                    continue
                inter = len(prev_esets[pi_idx] & curr_esets[ci_idx])
                union = len(prev_esets[pi_idx] | curr_esets[ci_idx])
                jacc = <f64>inter / <f64>union if union > 0 else 0.0
                if jacc >= jaccard_threshold:
                    children.append(ci)
                    used_curr.add(ci)
            if len(children) > 1:
                for ci in children:
                    events.append((FACE_SPLIT, pi, ci))
                    matched_curr.add(ci)
                n_split += 1
                matched_prev.add(pi)
            elif len(children) == 1:
                events.append((FACE_PERSIST, pi, children[0]))
                matched_prev.add(pi)
                matched_curr.add(children[0])
                n_persist += 1

        used_prev = set()
        for ci_idx, ci in enumerate(unmatched_curr):
            if ci in matched_curr:
                continue
            parents = []
            for pi_idx, pi in enumerate(unmatched_prev):
                if pi in matched_prev:
                    continue
                inter = len(prev_esets[pi_idx] & curr_esets[ci_idx])
                union = len(prev_esets[pi_idx] | curr_esets[ci_idx])
                jacc = <f64>inter / <f64>union if union > 0 else 0.0
                if jacc >= jaccard_threshold:
                    parents.append(pi)
            if len(parents) > 1:
                for pi in parents:
                    events.append((FACE_MERGE, pi, ci))
                    matched_prev.add(pi)
                n_merge += 1
                matched_curr.add(ci)

    for i in range(len(prev_keys)):
        if i not in matched_prev:
            events.append((FACE_DIED, i, -1))
            n_died += 1

    for i in range(len(curr_keys)):
        if i not in matched_curr:
            events.append((FACE_BORN, -1, i))
            n_born += 1

    return events, n_persist, n_born, n_died, n_split, n_merge


def compute_bioes_unified_general(list edge_snapshots,
                                  list face_snapshots,
                                  np.ndarray[i64, ndim=2] betti_matrix,
                                  double phase_tol=0.0,
                                  Py_ssize_t min_phase_len=2,
                                  Py_ssize_t face_event_threshold=1,
                                  double jaccard_threshold=0.5):
    """
    Unified BIOES pipeline for general boundary snapshots.

    Parameters
    ----------
    edge_snapshots : list of (boundary_ptr, boundary_idx) per timestep
    face_snapshots : list of (B2_col_ptr, B2_row_idx) per timestep
    betti_matrix : int64[T, K]

    Returns
    -------
    Same as compute_bioes_unified.
    """
    cdef Py_ssize_t T = len(edge_snapshots)

    ec, ebc, edc = compute_edge_metrics_general(edge_snapshots)

    cdef np.ndarray[i32, ndim=1] f_counts = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] f_born = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] f_died = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] f_split = np.zeros(T, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] f_merge = np.zeros(T, dtype=np.int32)

    if len(face_snapshots) == T and T > 1:
        for t in range(T):
            b2cp, b2ri = face_snapshots[t]
            f_counts[t] = <i32>(b2cp.shape[0] - 1) if b2cp.shape[0] > 1 else 0

        for t in range(1, T):
            b2cp_p, b2ri_p = face_snapshots[t - 1]
            bp_p, bi_p = edge_snapshots[t - 1]
            b2cp_c, b2ri_c = face_snapshots[t]
            bp_c, bi_c = edge_snapshots[t]
            evts, np_, nb, nd, ns, nm = track_faces_general(
                b2cp_p, b2ri_p, bp_p, bi_p,
                b2cp_c, b2ri_c, bp_c, bi_c,
                jaccard_threshold)
            f_born[t] = <i32>nb
            f_died[t] = <i32>nd
            f_split[t] = <i32>ns
            f_merge[t] = <i32>nm

    p_start, p_end, p_betti, reasons = detect_phases_with_events(
        betti_matrix, f_born, f_died, f_split, f_merge,
        phase_tol, face_event_threshold)
    unified_tags = assign_bioes_tags(T, p_start, p_end, min_phase_len)

    per_dim_tags = assign_bioes_per_dimension(T, betti_matrix, phase_tol, min_phase_len)

    return (unified_tags, per_dim_tags,
            ec, ebc, edc,
            f_counts, f_born, f_died, f_split, f_merge,
            p_start, p_end, p_betti, reasons)
