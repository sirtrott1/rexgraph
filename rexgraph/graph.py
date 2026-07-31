"""
rexgraph.graph: Orchestration layer for relational complexes.

A k-rex is a relational complex of top grade k: a finite sequence of signed
integer boundary maps (B_1, ..., B_k) with entries in {-1, 0, +1}, satisfying
the chain condition B_{d-1} B_d = 0. Edges are primitive and vertices are
derived from edge boundaries via the vertex lifecycle contract: a vertex
exists if and only if some edge contains it in its boundary.

RexGraph lazily composes the Cython modules in rexgraph.core through
@cached_property accessors. No Cython module imports another; all
inter-module composition happens here.

Computation is organized into cached bundles that call the Cython
builder functions in dependency order:

    _adjacency_bundle  - _cycles.build_symmetric_adjacency()
    _overlap_bundle    - _overlap.build_L_O()
    spectral_bundle    - _laplacians.build_all_laplacians(B1, B2_hodge, L_O)

Individual properties (L0, betti, coupling_constants, etc.) are thin
accessors into the bundle dicts with no additional computation.

TemporalRex wraps a sequence of snapshots sharing continuous identity,
with delta-encoded storage, BIOES phase detection, and lifecycle tracking.
"""

from __future__ import annotations

from collections import namedtuple
from functools import cached_property
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rexgraph import core as _core

# Per-module imports with graceful fallback.
# Each module is None if it failed to compile, so callers must guard
# usage behind the module being not-None (or behind _HAS_RCF for v2).
_boundary = getattr(_core, '_boundary', None)
_cycles = getattr(_core, '_cycles', None)
_faces = getattr(_core, '_faces', None)
_field = getattr(_core, '_field', None)
_hodge = getattr(_core, '_hodge', None)
_laplacians = getattr(_core, '_laplacians', None)
_overlap = getattr(_core, '_overlap', None)
_persistence = getattr(_core, '_persistence', None)
_quotient = getattr(_core, '_quotient', None)
_rex = getattr(_core, '_rex', None)
_signal = getattr(_core, '_signal', None)
_sparse = getattr(_core, '_sparse', None)
_spectral = getattr(_core, '_spectral', None)
_standard = getattr(_core, '_standard', None)
_state = getattr(_core, '_state', None)
_temporal = getattr(_core, '_temporal', None)
_transition = getattr(_core, '_transition', None)
_wave = getattr(_core, '_wave', None)

# RCF modules (new in v2)
_frustration = getattr(_core, '_frustration', None)
_relational = getattr(_core, '_relational', None)
_character = getattr(_core, '_character', None)
_void = getattr(_core, '_void', None)
_rcfe = getattr(_core, '_rcfe', None)
_joins = getattr(_core, '_joins', None)
_query = getattr(_core, '_query', None)
_fiber = getattr(_core, '_fiber', None)
_dirac = getattr(_core, '_dirac', None)
_hypermanifold = getattr(_core, '_hypermanifold', None)
_common = getattr(_core, '_common', None)

# v0.4 modules
_interfacing = getattr(_core, '_interfacing', None)
_channels = getattr(_core, '_channels', None)
_cross_complex = getattr(_core, '_cross_complex', None)

_HAS_RCF = all(m is not None for m in (
    _frustration, _relational, _character, _void, _rcfe, _joins, _query, _fiber,
))


# Helper types

_i32 = np.int32
_i64 = np.int64
_f64 = np.float64
_u8 = np.uint8
_c128 = np.complex128


def _asarray(x, dtype=_i32):
    """Coerce to contiguous ndarray."""
    return np.ascontiguousarray(x, dtype=dtype)


def _serialize_hodge_dict(d: dict) -> dict:
    """Convert a Hodge result dict to JSON-safe types.

    Values that are ndarrays are converted via .tolist(); scalars
    and nested dicts are passed through recursively.
    """
    if d is None:
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _serialize_hodge_dict(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# Dense materialization + dense-only linear algebra live in the modular
# rexgraph.dense_matrix; _ensure_dense is kept as the in-module alias so the many
# call sites here are unchanged while the dense path stays isolated in one file.
from rexgraph.dense_matrix import ensure_dense as _ensure_dense
from rexgraph.dense_matrix import spectral_distance as _spectral_distance


# RexGraph


def _edge_low_eig(L, n, k):
    """The smallest ``k`` eigenpairs (ascending) of a PSD edge operator, cheap and matrix-free.

    Dense ``eigh`` for small ``n`` (LAPACK is faster there); matrix-free ARPACK (``which='SM'``)
    otherwise, so no dense ``nE x nE`` is materialized. Returns ``(evals_ascending, evecs)`` with
    ``evecs[:, i]`` the eigenvector of ``evals[i]``. The caller reads the Fiedler by skipping the
    exactly-known kernel dimension - no float threshold.
    """
    import numpy as np
    import scipy.sparse as sp
    if n <= 1:
        return np.zeros(1, dtype=np.float64), np.ones((max(n, 1), 1), dtype=np.float64)
    k = max(1, min(k, n - 1))
    if n <= 2000:
        M = np.asarray(L.todense()) if sp.issparse(L) else np.asarray(L, dtype=np.float64)
        ev, V = np.linalg.eigh(M)
        idx = np.argsort(ev)[:k]
        return ev[idx], V[:, idx]
    from scipy.sparse.linalg import eigsh
    ev, V = eigsh(sp.csr_matrix(L).astype(np.float64), k=k, which='SM')
    idx = np.argsort(ev)
    return ev[idx], V[:, idx]


# Incremental mutation support: remap bookkeeping + tiered cache invalidation.
#
# Remap carries the index remaps produced by an incremental mutation (edge
# and/or face insertion/deletion) so downstream consumers can translate old
# indices to new ones without recomputing from scratch.
Remap = namedtuple("Remap", "edge_map vertex_map face_map")

# Dependency tiers for selective _invalidate(): each is a frozenset of
# cached_property names on RexGraph. A mutation names only the tiers it
# actually disturbs, so unrelated caches survive instead of being blown away.
_TIER_B1_ONLY = frozenset({
    "_B1_dual", "B1", "_v2e", "_vertex_info", "degree", "in_degree", "out_degree",
    "edge_types", "has_branching", "_is_standard_only", "_adjacency_bundle",
    "_overlap_bundle", "L_overlap", "overlap_gramian", "L0", "L0_sparse",
    "L1", "L1_sparse", "_sources", "_targets",
})
_TIER_B2_ONLY = frozenset({
    "_B2_dual", "_B2_hodge_dual", "B2", "B2_hodge", "nF_hodge",
    "self_loop_face_indices", "chain_valid", "_chain_col_maxabs",
    "L2", "L2_sparse",
})
_TIER_GLOBAL = frozenset({
    # edge->face CSR: size depends on nE and nF, invalidate on any structural change
    "_e2f",
    "spectral_bundle", "_dense_rcf_bundle", "betti", "edge_fiedler",
    "fiedler_val_L1", "fiedler_vec_L1", "eigenvalues_L0", "fiedler_vector_L0",
    "fiedler_overlap", "relational_laplacian", "RL", "_rl_eigen", "_green_cache",
    "_sparse_character", "_sparse_phi", "structural_character", "vertex_character",
    "star_character", "coherence", "local_coherence", "frustration_exact",
    "L_frustration", "L_coPC", "_edge_signs", "layout", "coupling_constants",
    "alpha_G", "field_coupling_psd", "_vertex_bundle", "nhats", "hat_names",
})


class RexGraph:
    """A relational complex (rex) with lazily computed derived properties.

    A k-rex is a relational complex of top grade k,

        C_k -> C_{k-1} -> ... -> C_0

    whose boundary maps carry entries in {-1, 0, +1} and satisfy the chain
    condition d_{k-1} . d_k = 0.  Edges are primitive; the vertex set is
    derived: V = union over e in E of supp(d_1(e)).  A column may carry any
    arity, so a branching hyperedge is a first-class cell rather than a clique
    expansion, which is why this is not a simplicial or CW complex.

    Computation is organized into cached bundles that call Cython
    builder functions. Individual properties are thin accessors
    into the bundle dicts.

    Parameters
    ----------
    boundary_ptr, boundary_idx : ndarray, optional
        CSR-format boundary map d_1.
    sources, targets : ndarray, optional
        Standard 2-endpoint edges. Mutually exclusive with
        boundary_ptr/boundary_idx.
    B2_col_ptr, B2_row_idx, B2_vals : ndarray, optional
        CSC representation of B_2.
    w_E : ndarray, optional
        Edge attribution w_E: E -> Attr.
    w_boundary : dict, optional
        Per-boundary-point attribution.
    directed : bool
        Whether edge orientations encode direction.
    """

    __slots__ = (
        "__dict__",
        "_boundary_ptr",
        "_boundary_idx",
        "_sources",
        "_targets",
        "_B2_col_ptr",
        "_B2_row_idx",
        "_B2_vals",
        "_w_E",
        "_w_boundary",
        "_directed",
        "_signs",
        "_nV",
        "_nE",
        "_nF",
        "_graded_duals",
        "_pending_edges",
        "_pending_hyperedges",
        "_pending_faces",
        "_live_edges",
        "_live_faces",
        "_dirty",
        "_last_remap",
    )

    # Construction

    def __init__(
        self,
        *,
        boundary_ptr: Optional[NDArray] = None,
        boundary_idx: Optional[NDArray] = None,
        sources: Optional[NDArray] = None,
        targets: Optional[NDArray] = None,
        B2_col_ptr: Optional[NDArray] = None,
        B2_row_idx: Optional[NDArray] = None,
        B2_vals: Optional[NDArray] = None,
        w_E: Optional[NDArray] = None,
        w_boundary: Optional[dict] = None,
        directed: bool = False,
        signs: Optional[NDArray] = None,
        g_channel: str = "raw",
    ):
        # G (overlap) channel form used by the RL4 character:
        #   "normalized" - L_O = I - D^{-1/2} K D^{-1/2} (float, degree-comparable; default)
        #   "raw"        - K = |B1|^T |B1| (exact integer co-incidence counts, canonical)
        # Both are always available via `L_overlap` / `overlap_gramian`; this only
        # selects which `g_channel_operator` (hence spectral_bundle/RL4) uses.
        if g_channel not in ("normalized", "raw"):
            raise ValueError("g_channel must be 'normalized' or 'raw', got %r" % g_channel)
        self._g_channel = g_channel

        # General boundary or src/tgt shorthand
        if boundary_ptr is not None:
            self._boundary_ptr = _asarray(boundary_ptr, _i32)
            self._boundary_idx = _asarray(boundary_idx, _i32)
            self._nE = self._boundary_ptr.shape[0] - 1
            self._sources = None
            self._targets = None
        elif sources is not None:
            src = _asarray(sources, _i32)
            tgt = _asarray(targets, _i32)
            self._sources = src
            self._targets = tgt
            self._nE = src.shape[0]
            bp = np.arange(0, 2 * self._nE + 1, 2, dtype=_i32)
            bi = np.empty(2 * self._nE, dtype=_i32)
            bi[0::2] = src
            bi[1::2] = tgt
            self._boundary_ptr = bp
            self._boundary_idx = bi
        else:
            raise ValueError("Provide boundary_ptr/boundary_idx or sources/targets.")

        # Derive vertex set
        if self._boundary_idx.shape[0] > 0:
            self._nV = int(self._boundary_idx.max()) + 1
        else:
            self._nV = 0

        # Face data
        if B2_col_ptr is not None:
            self._B2_col_ptr = _asarray(B2_col_ptr, _i32)
            self._B2_row_idx = _asarray(B2_row_idx, _i32)
            self._B2_vals = np.ascontiguousarray(B2_vals, dtype=_f64)
            self._nF = self._B2_col_ptr.shape[0] - 1
        else:
            self._B2_col_ptr = np.zeros(1, dtype=_i32)
            self._B2_row_idx = np.zeros(0, dtype=_i32)
            self._B2_vals = np.zeros(0, dtype=_f64)
            self._nF = 0

        # Attribution
        self._w_E = w_E
        self._w_boundary = w_boundary if w_boundary is not None else {}
        self._directed = directed
        self._signs = signs

        # Optional graded boundaries for grade >= 3 (B_3, B_4, ...) as a list of
        # scipy sparse matrices. Populated by from_cells; None for the classic
        # 1-rex / 2-rex constructors (which stay purely in the B1/B2 slots).
        self._graded_duals = None
        self._pending_edges = None
        self._pending_hyperedges = None
        self._pending_faces = None
        self._live_edges = None
        self._live_faces = None
        self._dirty = False
        self._last_remap = None

    def _invalidate(self, *tiers):
        """Drop the cached_property keys belonging to the named dependency tiers.

        Each tier is a frozenset of cached_property names (_TIER_B1_ONLY /
        _TIER_B2_ONLY / _TIER_GLOBAL). Only the named keys are removed from
        __dict__; unnamed caches survive. Replaces the old blunt clear-all.
        """
        d = self.__dict__
        for tier in tiers:
            for key in tier:
                d.pop(key, None)

    def _identity_remap(self):
        """The no-op Remap: every index maps to itself (nothing was tombstoned)."""
        return Remap(edge_map=np.arange(self._nE, dtype=_i32),
                     vertex_map=np.arange(self._nV, dtype=_i32),
                     face_map=np.arange(self._nF, dtype=_i32))

    def add_edges(self, sources, targets, *, w_E=None, signs=None, w_boundary=None):
        """Stage new standard edges for O(delta) append (materialized on next read).

        Records the columns in the pending buffer, bumps the logical _nE/_nV, and
        stages attribution. No array copy at call time. New edges land at the end.
        """
        ns = _asarray(sources, _i32)
        nt = _asarray(targets, _i32)
        if ns.shape[0] != nt.shape[0]:
            raise ValueError("sources and targets must have equal length")
        n_new = int(ns.shape[0])
        if self._pending_edges is None:
            self._pending_edges = {"src": [], "tgt": [], "w_E": [], "signs": []}
        self._pending_edges["src"].append(ns)
        self._pending_edges["tgt"].append(nt)
        self._pending_edges["w_E"].append(
            np.zeros(n_new, _f64) if w_E is None else _asarray(w_E, _f64))
        self._pending_edges["signs"].append(
            np.ones(n_new, _i32) if signs is None else _asarray(signs, _i32))
        if w_boundary:
            for (e_local, v), feat in w_boundary.items():
                self._w_boundary[self._nE + int(e_local)] = feat
        self._nE += n_new
        mx = int(max(int(ns.max()) if n_new else -1, int(nt.max()) if n_new else -1))
        if mx + 1 > self._nV:
            self._nV = mx + 1
        self._dirty = True
        # Drop any already-cached B1_ONLY/GLOBAL derived properties now (cheap
        # dict.pop, no array touch) so a stale cached_property (e.g. _e2f,
        # which depends on nE) can't be read back before the deferred raw-array
        # flush in _ensure_clean runs. The raw arrays themselves stay deferred.
        self._invalidate(_TIER_B1_ONLY, _TIER_GLOBAL)

    def add_hyperedges(self, columns, *, w_E=None, signs=None):
        """Stage new GENERAL-ARITY boundary cells for O(delta) append (materialized on
        next read). Each entry of `columns` is an int array of the vertex ids incident
        to one new cell (arity = its length; 2 recovers a plain edge). Complements
        add_edges (the 2-arity convenience); removal/compaction are already general.
        No array copy at call time. New cells land at the end on flush."""
        cols = [_asarray(c, _i32) for c in columns]
        n_new = len(cols)
        if self._pending_hyperedges is None:
            self._pending_hyperedges = {"cols": [], "w_E": [], "signs": []}
        self._pending_hyperedges["cols"].extend(cols)
        self._pending_hyperedges["w_E"].append(
            np.zeros(n_new, _f64) if w_E is None else _asarray(w_E, _f64))
        self._pending_hyperedges["signs"].append(
            np.ones(n_new, _i32) if signs is None else _asarray(signs, _i32))
        self._nE += n_new
        mx = -1
        for c in cols:
            if c.shape[0] and int(c.max()) > mx:
                mx = int(c.max())
        if mx + 1 > self._nV:
            self._nV = mx + 1
        self._dirty = True
        self._invalidate(_TIER_B1_ONLY, _TIER_GLOBAL)

    def add_faces(self, face_edges, face_signs):
        """Stage new faces (each a list of edge indices + matching +/-1 signs) for
        O(delta) append. Materialized on next read. Chain validity (B1.B2=0) is
        enforced by the existing _B2_hodge_dual filter, unchanged."""
        if len(face_edges) != len(face_signs):
            raise ValueError("face_edges and face_signs must have equal length")
        if self._pending_faces is None:
            self._pending_faces = {"edges": [], "signs": []}
        for fe, fs in zip(face_edges, face_signs):
            fe = _asarray(fe, _i32)
            fs = np.ascontiguousarray(fs, dtype=_f64)
            if fe.shape[0] != fs.shape[0]:
                raise ValueError("each face's edges and signs must have equal length")
            self._pending_faces["edges"].append(fe)
            self._pending_faces["signs"].append(fs)
            self._nF += 1
        self._dirty = True
        # Eager invalidate (cheap dict.pop, arrays stay deferred): a cached_property
        # is a non-data descriptor, so once cached a later read bypasses the getter
        # and its _ensure_clean guard; dropping the stale keys now forces the next
        # read to re-run the getter -> flush -> recompute. B1_ONLY caches survive a
        # face-only append (edge boundary unchanged).
        self._invalidate(_TIER_B2_ONLY, _TIER_GLOBAL)

    def _ensure_clean(self):
        """Materialize pending appends and tombstones, then selectively invalidate.

        Called at the top of every eager-array read. Returns the Remap produced (an
        identity remap when nothing was tombstoned).
        """
        if not self._dirty:
            return self._last_remap if self._last_remap is not None else self._identity_remap()
        touched = []
        # 1. flush pending edge appends (numpy concatenate)
        if self._pending_edges is not None:
            nE_before = int(self._boundary_ptr.shape[0] - 1)
            src_all = (np.concatenate(self._pending_edges["src"])
                       if self._pending_edges["src"] else np.zeros(0, _i32))
            tgt_all = (np.concatenate(self._pending_edges["tgt"])
                       if self._pending_edges["tgt"] else np.zeros(0, _i32))
            k = int(src_all.shape[0])
            new_idx = np.empty(2 * k, dtype=_i32)
            new_idx[0::2] = src_all
            new_idx[1::2] = tgt_all
            last = int(self._boundary_ptr[-1])
            add_ptr = last + np.arange(2, 2 * k + 1, 2, dtype=_i32)
            self._boundary_ptr = np.concatenate([self._boundary_ptr, add_ptr])
            self._boundary_idx = np.concatenate([self._boundary_idx, new_idx])
            self._sources = None
            self._targets = None
            # Attribution carry-through for the new edges: batches were already
            # staged at batch length in add_edges (defaulting to zeros/ones), so we
            # only need to concatenate. Skip allocating when nothing was ever set,
            # so w_E/signs stay None until real attribution shows up.
            w_e_batches = self._pending_edges["w_E"]
            signs_batches = self._pending_edges["signs"]
            if self._w_E is not None or any(np.any(b) for b in w_e_batches):
                base = (np.zeros(nE_before, dtype=_f64) if self._w_E is None
                        else np.asarray(self._w_E, dtype=_f64))
                self._w_E = np.concatenate([base, np.concatenate(w_e_batches)])
            if self._signs is not None or any(np.any(b != 1) for b in signs_batches):
                base = (np.ones(nE_before, dtype=_i32) if self._signs is None
                        else np.asarray(self._signs, dtype=_i32))
                self._signs = np.concatenate([base, np.concatenate(signs_batches)])
            self._pending_edges = None
            touched.append(_TIER_B1_ONLY)
            touched.append(_TIER_GLOBAL)
        # 1b. flush pending general-arity hyperedge appends (numpy concatenate)
        if self._pending_hyperedges is not None:
            nE_before = int(self._boundary_ptr.shape[0] - 1)
            cols = self._pending_hyperedges["cols"]
            sizes = (np.array([c.shape[0] for c in cols], dtype=_i32) if cols
                     else np.zeros(0, _i32))
            flat = np.concatenate(cols) if cols else np.zeros(0, _i32)
            last = int(self._boundary_ptr[-1])
            add_ptr = last + np.cumsum(sizes).astype(_i32)
            self._boundary_ptr = np.concatenate([self._boundary_ptr, add_ptr])
            self._boundary_idx = np.concatenate([self._boundary_idx, flat])
            self._sources = None
            self._targets = None
            we_b = self._pending_hyperedges["w_E"]
            sg_b = self._pending_hyperedges["signs"]
            if self._w_E is not None or any(np.any(b) for b in we_b):
                base = (np.zeros(nE_before, dtype=_f64) if self._w_E is None
                        else np.asarray(self._w_E, dtype=_f64))
                self._w_E = np.concatenate([base, np.concatenate(we_b)])
            if self._signs is not None or any(np.any(b != 1) for b in sg_b):
                base = (np.ones(nE_before, dtype=_i32) if self._signs is None
                        else np.asarray(self._signs, dtype=_i32))
                self._signs = np.concatenate([base, np.concatenate(sg_b)])
            self._pending_hyperedges = None
            touched.append(_TIER_B1_ONLY)
            touched.append(_TIER_GLOBAL)
        # 2. flush pending face appends (filled in by add_faces)
        if self._pending_faces is not None:
            self._flush_faces(touched)
        # 3. compaction (reconcile mask lengths first: same-batch appends may have
        # grown _nE/_nF after the tombstone masks were created; appended cells are live)
        remap = self._identity_remap()
        if self._live_edges is not None or self._live_faces is not None:
            if self._live_edges is not None and self._live_edges.shape[0] < self._nE:
                self._live_edges = np.concatenate([
                    self._live_edges,
                    np.ones(self._nE - self._live_edges.shape[0], dtype=bool)])
            if self._live_faces is not None and self._live_faces.shape[0] < self._nF:
                self._live_faces = np.concatenate([
                    self._live_faces,
                    np.ones(self._nF - self._live_faces.shape[0], dtype=bool)])
            remap = self._compact_now(touched)
        self._dirty = False
        self._last_remap = remap
        if touched:
            self._invalidate(*touched)
        return remap

    def _flush_faces(self, touched):
        """Concatenate staged faces onto the B2 CSC arrays. Called by _ensure_clean."""
        pend = self._pending_faces
        self._pending_faces = None
        if not pend["edges"]:
            return
        new_rows = np.concatenate(pend["edges"]) if pend["edges"] else np.zeros(0, _i32)
        new_vals = np.concatenate(pend["signs"]) if pend["signs"] else np.zeros(0, _f64)
        last = int(self._B2_col_ptr[-1])
        sizes = np.array([e.shape[0] for e in pend["edges"]], dtype=_i32)
        add_ptr = last + np.cumsum(sizes).astype(_i32)
        self._B2_col_ptr = np.concatenate([self._B2_col_ptr, add_ptr])
        self._B2_row_idx = np.concatenate([self._B2_row_idx, new_rows])
        self._B2_vals = np.concatenate([self._B2_vals, new_vals])
        touched.append(_TIER_B2_ONLY)
        touched.append(_TIER_GLOBAL)

    def remove_edges(self, mask):
        """Tombstone edges where mask is nonzero (O(delta)); compaction is deferred."""
        m = _asarray(mask, _i32)
        if m.shape[0] != self._nE:
            raise ValueError("mask length must equal nE (%d)" % self._nE)
        if self._live_edges is None:
            self._live_edges = np.ones(self._nE, dtype=bool)
        self._live_edges[m != 0] = False
        self._dirty = True
        # Eager invalidate now (cheap dict.pop; the compaction that renumbers arrays
        # is still deferred to _ensure_clean). A cached_property is a non-data
        # descriptor, so without this a later read would bypass the getter and return
        # a pre-removal cache. An edge removal renumbers edges (and can drop faces
        # touching a removed edge), so every tier is affected.
        self._invalidate(_TIER_B1_ONLY, _TIER_B2_ONLY, _TIER_GLOBAL)

    def remove_faces(self, mask):
        """Tombstone faces where mask is nonzero (O(delta)); compaction is deferred."""
        m = _asarray(mask, _i32)
        if m.shape[0] != self._nF:
            raise ValueError("mask length must equal nF (%d)" % self._nF)
        if self._live_faces is None:
            self._live_faces = np.ones(self._nF, dtype=bool)
        self._live_faces[m != 0] = False
        self._dirty = True
        # Eager invalidate (see remove_edges). A face-only removal leaves the edge
        # boundary intact, so B1_ONLY survives.
        self._invalidate(_TIER_B2_ONLY, _TIER_GLOBAL)

    def compact(self):
        """Force materialization and return the Remap for the renumbering applied
        since the last compact() (an identity remap if nothing was pending or
        tombstoned since then)."""
        self._ensure_clean()
        remap = self._last_remap if self._last_remap is not None else self._identity_remap()
        self._last_remap = None      # consumed: a later no-op compact() returns identity
        return remap

    def _compact_now(self, touched):
        """Drop tombstoned edges/faces, renumber, remap attribution and B2. Returns
        a Remap. Called by _ensure_clean when tombstones are present (after appends
        have already been flushed, so _boundary_* and _B2_* are up to date)."""
        from rexgraph.core import _rex
        # edges
        if self._live_edges is not None:
            live = self._live_edges.astype(_i32)
            new_ptr, new_idx, nV_new, v_map, e_map = _rex.compact_boundary(
                self._boundary_ptr, self._boundary_idx, live)
            self._boundary_ptr = new_ptr
            self._boundary_idx = new_idx
            self._sources = None
            self._targets = None
            keep = self._live_edges
            if self._w_E is not None:
                self._w_E = np.asarray(self._w_E)[keep]
            if self._signs is not None:
                self._signs = np.asarray(self._signs)[keep]
            if self._w_boundary:
                em = np.asarray(e_map)
                new_wb = {}
                for key, feat in self._w_boundary.items():
                    e_old = key[0] if isinstance(key, tuple) else key
                    e_new = int(em[e_old])
                    if e_new >= 0:
                        new_wb[(e_new, key[1]) if isinstance(key, tuple) else e_new] = feat
                self._w_boundary = new_wb
            self._nE = int(new_ptr.shape[0] - 1)
            self._nV = int(nV_new)
            self._live_edges = None
        else:
            e_map = np.arange(self._nE, dtype=_i32)
            v_map = np.arange(self._nV, dtype=_i32)
        # faces: one pass over the ORIGINAL faces that drops a column iff it is
        # tombstoned OR its boundary references a removed edge, and remaps the
        # surviving rows through e_map (identity when no edges were removed).
        self._compact_faces(e_map)
        self._live_faces = None
        touched.append(_TIER_B1_ONLY)
        touched.append(_TIER_B2_ONLY)
        touched.append(_TIER_GLOBAL)
        face_map = np.arange(self._nF, dtype=_i32)
        return Remap(edge_map=np.asarray(e_map), vertex_map=np.asarray(v_map),
                     face_map=face_map)

    def _compact_faces(self, e_map):
        """Rebuild the B2 CSC in one pass over the ORIGINAL faces: drop a face column
        iff it is tombstoned (_live_faces) OR its boundary references a removed edge
        (e_map == -1); for surviving faces remap the row (edge) indices through e_map.
        Doing both drops in a single pass avoids the id-shift bug of dropping in two
        stages. e_map is the identity arange when no edges were removed."""
        em = np.asarray(e_map)
        has_edge_remap = em.shape[0] > 0
        cp, ri, vals = self._B2_col_ptr, self._B2_row_idx, self._B2_vals
        live_f = self._live_faces
        out_ptr, out_rows, out_vals = [0], [], []
        for f in range(cp.shape[0] - 1):
            if live_f is not None and not live_f[f]:
                continue                       # tombstoned face
            rows = ri[cp[f]:cp[f + 1]]
            mapped = em[rows] if has_edge_remap else rows
            if has_edge_remap and np.any(mapped < 0):
                continue                       # references a removed edge
            out_rows.append(mapped)
            out_vals.append(vals[cp[f]:cp[f + 1]])
            out_ptr.append(out_ptr[-1] + int(rows.shape[0]))
        self._B2_col_ptr = np.asarray(out_ptr, dtype=_i32)
        self._B2_row_idx = (np.concatenate(out_rows).astype(_i32) if out_rows
                            else np.zeros(0, _i32))
        self._B2_vals = (np.concatenate(out_vals).astype(_f64) if out_vals
                         else np.zeros(0, _f64))
        self._nF = int(self._B2_col_ptr.shape[0] - 1)

    def set_cell_attrs(self, indices, *, w_E=None, signs=None):
        """Set per-cell attribution in place at the given current indices, then
        invalidate the character/global tier (signs/weights feed those). Used
        by apply_edge_delta to replay a delta's MODIFIED cells."""
        self._ensure_clean()
        idx = _asarray(indices, _i32)
        if w_E is not None:
            if self._w_E is None:
                self._w_E = np.zeros(self._nE, dtype=_f64)
            self._w_E = np.asarray(self._w_E, dtype=_f64).copy()
            self._w_E[idx] = np.asarray(w_E, dtype=_f64)
        if signs is not None:
            if self._signs is None:
                self._signs = np.ones(self._nE, dtype=_i32)
            self._signs = np.asarray(self._signs, dtype=_i32).copy()
            self._signs[idx] = np.asarray(signs, dtype=_i32)
        self._invalidate(_TIER_GLOBAL)

    # Factory constructors

    @classmethod
    def from_graph(
        cls,
        sources: ArrayLike,
        targets: ArrayLike,
        *,
        directed: bool = False,
        w_E: Optional[NDArray] = None,
        g_channel: str = "raw",
    ) -> RexGraph:
        """Embed a simple graph as a 1-rex. ``g_channel`` selects the overlap G
        form used by the character ('normalized' default, or 'raw' integer)."""
        return cls(
            sources=np.asarray(sources),
            targets=np.asarray(targets),
            directed=directed,
            w_E=w_E,
            g_channel=g_channel,
        )

    @classmethod
    def from_hypergraph(
        cls,
        hyperedge_ptr: ArrayLike,
        hyperedge_idx: ArrayLike,
        *,
        g_channel: str = "raw",
    ) -> RexGraph:
        """Embed a hypergraph as a 1-rex with branching edges."""
        return cls(
            boundary_ptr=np.asarray(hyperedge_ptr),
            boundary_idx=np.asarray(hyperedge_idx),
            g_channel=g_channel,
        )

    @classmethod
    def from_simplicial(
        cls,
        sources: ArrayLike,
        targets: ArrayLike,
        triangles: ArrayLike,
        *,
        g_channel: str = "raw",
    ) -> RexGraph:
        """Embed a simplicial 2-complex as a 2-rex.

        Parameters
        ----------
        sources, targets : array-like of int
            Edge endpoint arrays, shape (nE,).
        triangles : array-like of int, shape (nT, 3)
            Each row is (v0, v1, v2), the vertex indices of a triangle.
            Edges are looked up from (sources, targets); orientation signs
            are determined by vertex ordering.
        """
        src = _asarray(sources, _i32)
        tgt = _asarray(targets, _i32)
        tri = np.asarray(triangles, dtype=_i32)
        nV = int(max(src.max(), tgt.max())) + 1
        nE = src.shape[0]

        # Build edge lookup: (min_v, max_v) -> (edge_index, sign)
        # sign = +1 if src < tgt (matches canonical orientation), -1 otherwise
        edge_map = {}
        for e in range(nE):
            s, t = int(src[e]), int(tgt[e])
            key = (min(s, t), max(s, t))
            edge_map[key] = (e, +1.0 if s < t else -1.0)

        nT = tri.shape[0]
        tri_e0 = np.empty(nT, dtype=_i32)
        tri_e1 = np.empty(nT, dtype=_i32)
        tri_e2 = np.empty(nT, dtype=_i32)
        tri_s0 = np.empty(nT, dtype=_f64)
        tri_s1 = np.empty(nT, dtype=_f64)
        tri_s2 = np.empty(nT, dtype=_f64)

        for t_idx in range(nT):
            v0, v1, v2 = int(tri[t_idx, 0]), int(tri[t_idx, 1]), int(tri[t_idx, 2])
            # Triangle boundary: d(v0,v1,v2) = (v0,v1) - (v0,v2) + (v1,v2)
            # Edge 0: v0->v1, Edge 1: v0->v2, Edge 2: v1->v2
            for i, (va, vb, boundary_sign) in enumerate([
                (v0, v1, +1.0),
                (v0, v2, -1.0),
                (v1, v2, +1.0),
            ]):
                key = (min(va, vb), max(va, vb))
                if key not in edge_map:
                    raise ValueError(
                        f"Triangle ({v0},{v1},{v2}) references edge ({va},{vb}) "
                        f"not found in edge list."
                    )
                eidx, canon_sign = edge_map[key]
                # Actual sign: boundary_sign * canon_sign accounts for
                # whether edge orientation matches triangle orientation
                actual_sign = boundary_sign * canon_sign
                if va > vb:
                    actual_sign = -actual_sign
                if i == 0:
                    tri_e0[t_idx] = eidx
                    tri_s0[t_idx] = actual_sign
                elif i == 1:
                    tri_e1[t_idx] = eidx
                    tri_s1[t_idx] = actual_sign
                else:
                    tri_e2[t_idx] = eidx
                    tri_s2[t_idx] = actual_sign

        # Returns (B2_col_ptr, B2_row_idx, B2_vals)
        cp, ri, vv = _rex.from_simplicial_2complex(
            nV, src, tgt, tri_e0, tri_e1, tri_e2, tri_s0, tri_s1, tri_s2
        )
        return cls(
            sources=src, targets=tgt,
            B2_col_ptr=cp, B2_row_idx=ri, B2_vals=vv,
            g_channel=g_channel,
        )

    @classmethod
    def from_adjacency(cls, A: NDArray, *, directed: bool = False,
                       g_channel: str = "raw") -> RexGraph:
        """Construct a 1-rex from an adjacency matrix."""
        if directed:
            rows, cols = np.nonzero(A)
        else:
            rows, cols = np.nonzero(np.triu(A, k=1))
        weights = A[rows, cols]
        w_E = weights.reshape(-1, 1) if not np.allclose(weights, 1) else None
        return cls(
            sources=rows.astype(_i32),
            targets=cols.astype(_i32),
            directed=directed,
            w_E=w_E,
            g_channel=g_channel,
        )

    @classmethod
    def from_cells(cls, cells_by_grade, *, g_channel: str = "raw") -> RexGraph:
        """Build a rex of ARBITRARY top grade from a graded, mixed-arity cell list.

        Parameters
        ----------
        cells_by_grade : sequence
            ``cells_by_grade[0]`` is the vertex count (int); for ``d >= 1``,
            ``cells_by_grade[d]`` is a list of d-cells, each either a plain list of
            ``(d-1)``-cell indices (positional signs: first ``-1``, rest ``+1``) or an
            explicit ``[(index, sign), ...]`` list. Arity is per-cell and free;
            grade is unbounded. See :mod:`rexgraph.graded_boundary`.

        Notes
        -----
        Grades 1 and 2 are stored in the classic B1 (CSR) / B2 (CSC) slots so every
        existing accessor keeps working unchanged; grades ``>= 3`` are stored in the
        optional ``_graded_duals`` list. The full ``[B_1, B_2, B_3, ...]`` list is
        available via :meth:`graded_boundaries` (and hence the sparse Dirac).
        """
        from rexgraph.graded_boundary import build_graded_boundaries, _cell_entries

        n_verts = int(cells_by_grade[0])
        boundaries = build_graded_boundaries(cells_by_grade)

        if len(boundaries) == 0:
            raise ValueError("from_cells needs at least a grade-1 (edge) list")

        # --- B1 slot: general boundary CSR straight from the grade-1 cells, using
        #     the positional sign convention (first index is the source, rest are
        #     targets), which is exactly what the CSR storage encodes. ---
        edge_cells = cells_by_grade[1]
        counts = np.empty(len(edge_cells), dtype=_i32)
        idx_parts = []
        for e, cell in enumerate(edge_cells):
            idx, _sgn = _cell_entries(cell)
            counts[e] = idx.shape[0]
            idx_parts.append(idx.astype(_i32, copy=False))
        boundary_ptr = np.zeros(len(edge_cells) + 1, dtype=_i32)
        np.cumsum(counts, out=boundary_ptr[1:])
        boundary_idx = (np.concatenate(idx_parts).astype(_i32)
                        if idx_parts else np.zeros(0, dtype=_i32))

        # --- B2 slot: CSC triplet from the grade-2 boundary map (signed, any arity). ---
        b2_kwargs = {}
        if len(boundaries) >= 2:
            B2 = boundaries[1].tocsc()
            b2_kwargs = dict(
                B2_col_ptr=B2.indptr.astype(_i32),
                B2_row_idx=B2.indices.astype(_i32),
                B2_vals=B2.data.astype(_f64),
            )

        rex = cls(
            boundary_ptr=boundary_ptr,
            boundary_idx=boundary_idx,
            g_channel=g_channel,
            **b2_kwargs,
        )

        # Honor the declared vertex count (isolated vertices are not visible from
        # the edge supports alone).
        if n_verts > rex._nV:
            rex._nV = n_verts

        # --- Grades >= 3 live in the optional _graded_duals list (scipy CSR). ---
        if len(boundaries) >= 3:
            rex._graded_duals = [b.tocsr() for b in boundaries[2:]]

        return rex

    def graded_boundaries(self) -> list:
        """The full graded boundary list ``[B_1, B_2, B_3, ...]`` (scipy CSR).

        Delegates to :func:`rexgraph.graded_boundary.graded_boundaries_from_rex`,
        the single source of truth; this is the contract the sparse Dirac accessor
        reads. Nothing is densified.
        """
        from rexgraph.graded_boundary import graded_boundaries_from_rex
        return graded_boundaries_from_rex(self)

    def sparse_dirac(self):
        """The sparse, matrix-free graded Dirac ``D = d + d*`` over the full
        ``[B_1, B_2, B_3, ...]`` list (see :class:`rexgraph.dirac_propagator.SparseDirac`).
        Grade-general: an N-rex built via :meth:`from_cells` propagates through every
        grade, not just vertices/edges/faces."""
        from rexgraph.dirac_propagator import SparseDirac
        return SparseDirac(self.graded_boundaries())

    def dirac_light(self, t: float, psi0: NDArray = None, order: int = None):
        """Light/wave propagator ``e^{-itD}`` of a graded tensor state, returned as
        ``(re, im)``: ``re = cos(tD)`` is the in-grade (gradient) part and
        ``im = -sin(tD)`` is the grade-CROSSING (curl) part - amplitude the
        off-diagonal boundary blocks transport between grades. Sparse mat-vecs, any
        ``t``, no eigendecomposition. Supersedes the dense ``dirac_operator`` /
        ``dirac_eigenvalues`` eigenpath for propagation."""
        from rexgraph.dirac_propagator import dirac_light as _dl
        return _dl(self, float(t), psi0=psi0, order=order)

    def dirac_heat(self, t: float, psi0: NDArray = None):
        """Per-grade heat propagator ``e^{-tD^2} = e^{-tL}`` of a graded state
        (stable, in-grade diffusion) - the diffusive companion to
        :meth:`dirac_light`, whose imaginary part is the grade-crossing transport."""
        from rexgraph.dirac_propagator import dirac_heat as _dh
        return _dh(self, float(t), psi0=psi0)

    def field_heat(self, F: NDArray, t: float, order: int = None, W=None):
        """Heat evolution ``e^{-t·W⁻¹M} F`` of the coupled (edge, face) field on the
        graded space C₁⊕C₂ (M = [[RL1,-gB2],[-gB2ᵀ,L2]]) under the tensor metric ``W``,
        matrix-free via a Chebyshev polynomial of the SPARSE M - any t, no
        eigendecomposition. ``W`` defaults to the √w boundary-weight metric (identity
        when unweighted, so this is e^{-tM}); pass a 1D diagonal or full SPD metric to
        override. ``F`` may be an edge signal (nE), a graded state (nE+nF), or a
        block/tensor field (…, m). See :mod:`rexgraph.field_propagator`."""
        from rexgraph.field_propagator import field_heat as _fh
        return _fh(self, F, float(t), order=order, W=W)

    def field_wave(self, F: NDArray, t: float, order: int = None, W=None):
        """Wave evolution ``cos(t·√(W⁻¹M)) F`` of the graded field under the tensor
        metric ``W`` (oscillation at ωₖ=√λₖ of the metric generator), matrix-free
        Chebyshev, any t, no eigendecomposition. Same field-shape / metric contract as
        :meth:`field_heat`."""
        from rexgraph.field_propagator import field_wave as _fw
        return _fw(self, F, float(t), order=order, W=W)

    # Dimensions

    @property
    def nV(self) -> int:
        self._ensure_clean()
        return self._nV

    @property
    def nE(self) -> int:
        self._ensure_clean()
        return self._nE

    @property
    def nF(self) -> int:
        self._ensure_clean()
        return self._nF

    @property
    def dimension(self) -> int:
        self._ensure_clean()
        if self._nE == 0:
            return 0
        return 2 if self._nF > 0 else 1

    # Raw structural data

    @property
    def boundary_ptr(self) -> NDArray:
        self._ensure_clean()
        return self._boundary_ptr

    @property
    def boundary_idx(self) -> NDArray:
        self._ensure_clean()
        return self._boundary_idx

    @property
    def sources(self) -> Optional[NDArray]:
        self._ensure_clean()
        if self._sources is not None:
            return self._sources
        sizes = np.diff(self._boundary_ptr)
        if np.any(sizes < 2):
            return None
        self._sources = self._boundary_idx[self._boundary_ptr[:-1]]
        return self._sources

    @property
    def targets(self) -> Optional[NDArray]:
        self._ensure_clean()
        if self._targets is not None:
            return self._targets
        sizes = np.diff(self._boundary_ptr)
        if np.any(sizes < 2):
            return None
        self._targets = self._boundary_idx[self._boundary_ptr[:-1] + 1]
        return self._targets

    @cached_property
    def _is_standard_only(self) -> bool:
        self._ensure_clean()
        sizes = np.diff(self._boundary_ptr)
        return bool(np.all(sizes == 2))

    @property
    def w_E(self) -> Optional[NDArray]:
        return self._w_E

    @property
    def w_boundary(self) -> dict:
        return self._w_boundary

    def set_vertex_attribution(self, X: NDArray) -> None:
        """Set per-boundary-point attribution from vertex features."""
        self._ensure_clean()
        self._w_boundary = {}
        bp, bi = self._boundary_ptr, self._boundary_idx
        for e in range(self._nE):
            for j in range(bp[e], bp[e + 1]):
                self._w_boundary[(int(e), int(bi[j]))] = X[bi[j]]

    # Edge classification

    @cached_property
    def edge_types(self) -> NDArray:
        """Per-edge type classification: 0=standard, 1=self-loop, 2=branching, 3=witness."""
        if self._is_standard_only:
            return _rex.classify_edges_standard(
                self._nE, self.sources, self.targets
            )
        etypes, _ = _rex.classify_edges_general(
            self._nE, self._boundary_ptr, self._boundary_idx
        )
        return etypes

    @cached_property
    def has_branching(self) -> bool:
        return bool(np.any(self.edge_types == 2))

    # Vertex degree data (via derive_vertex_set)

    @cached_property
    def _vertex_info(self) -> Tuple[int, NDArray, NDArray, NDArray]:
        """(nV_derived, degree, in_degree, out_degree) from _rex.derive_vertex_set."""
        src, tgt = self._ensure_src_tgt()
        return _rex.derive_vertex_set(self._nE, src, tgt)

    @cached_property
    def degree(self) -> NDArray:
        """Per-vertex degree array."""
        return self._vertex_info[1]

    @cached_property
    def in_degree(self) -> NDArray:
        return self._vertex_info[2]

    @cached_property
    def out_degree(self) -> NDArray:
        return self._vertex_info[3]

    # Incidence CSR

    @cached_property
    def _v2e(self) -> Tuple[NDArray, NDArray]:
        """Vertex-to-edge CSR adjacency."""
        src, tgt = self._ensure_src_tgt()
        return _rex.build_vertex_to_edge_csr(self._nV, self._nE, src, tgt)

    @cached_property
    def _e2f(self) -> Tuple[NDArray, NDArray]:
        """Edge-to-face CSR adjacency."""
        self._ensure_clean()
        if self._nF == 0:
            return np.zeros(self._nE + 1, dtype=_i32), np.zeros(0, dtype=_i32)
        return _rex.build_edge_to_face_csr(
            self._nE, self._nF, self._B2_col_ptr, self._B2_row_idx
        )

    def _ensure_src_tgt(self) -> Tuple[NDArray, NDArray]:
        """Return (sources, targets) even for general boundary (first two vertices)."""
        src = self.sources
        tgt = self.targets
        if src is not None and tgt is not None:
            return src.astype(_i32, copy=False), tgt.astype(_i32, copy=False)
        # General boundary fallback: extract first two boundary vertices
        src = self._boundary_idx[self._boundary_ptr[:-1]].astype(_i32)
        tgt_offsets = np.minimum(
            self._boundary_ptr[:-1] + 1,
            self._boundary_ptr[1:] - 1,
        )
        tgt = self._boundary_idx[tgt_offsets].astype(_i32)
        return src, tgt

    # Boundary operators

    @cached_property
    def _B1_dual(self):
        """DualCSR representation of B1."""
        self._ensure_clean()
        if self._is_standard_only:
            return _boundary.build_B1(self._nV, self._nE, self.sources, self.targets)
        # General boundary: dense to DualCSR
        return _sparse.from_dense_f64(self._build_B1_general())

    def _build_B1_general(self) -> NDArray:
        """Build dense B1 from general boundary data."""
        B1 = np.zeros((self._nV, self._nE), dtype=_f64)
        bp, bi = self._boundary_ptr, self._boundary_idx
        for e in range(self._nE):
            start, end = bp[e], bp[e + 1]
            k = end - start
            if k == 0:
                continue
            elif k == 1:
                B1[bi[start], e] = 1.0
            elif k == 2 and bi[start] == bi[start + 1]:
                pass  # self-loop
            else:
                B1[bi[start], e] = -1.0
                for j in range(start + 1, end):
                    B1[bi[j], e] += 1.0
        return B1

    @cached_property
    def _B2_dual(self):
        """DualCSR representation of B2, assembled straight from the CSC triplet. The columns are
        already the per-face boundaries (edge rows, orientation signs), so we scatter them directly
        rather than materializing a dense nE x nF matrix and rescanning it for the few nonzeros."""
        self._ensure_clean()
        if self._nF == 0:
            return None
        cp, ri, vl = self._B2_col_ptr, self._B2_row_idx, self._B2_vals
        lengths = np.diff(cp).astype(np.int32)
        return _boundary.build_B2_from_cycles(self._nE, ri, vl, lengths)

    @cached_property
    def B1(self) -> NDArray:
        """Signed incidence matrix B_1, shape (nV, nE)."""
        return _sparse.to_dense_f64(self._B1_dual)

    @cached_property
    def B2(self) -> NDArray:
        """Face boundary matrix B_2, shape (nE, nF)."""
        if self._nF == 0:
            return np.zeros((max(self._nE, 1), 0), dtype=_f64)
        return _sparse.to_dense_f64(self._B2_dual)

    @cached_property
    def chain_valid(self) -> bool:
        """Verify B_1 B_2 = 0."""
        if self._nF == 0:
            return True
        ok, _ = _boundary.verify_chain_complex(self._B1_dual, self._B2_dual)
        return ok

    # Clique expansion

    @cached_property
    def clique_expansion(self) -> RexGraph:
        """Clique expansion of branching edges."""
        self._ensure_clean()
        new_src, new_tgt, new_weights, _ = _rex.clique_expand_branching(
            self._nE, self._boundary_ptr, self._boundary_idx, self.edge_types
        )
        return RexGraph(
            sources=new_src,
            targets=new_tgt,
            w_E=new_weights.reshape(-1, 1),
        )

    # BUNDLE 1: Symmetric adjacency CSR

    @cached_property
    def _adjacency_bundle(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Symmetric adjacency CSR: (adj_ptr, adj_idx, adj_edge)."""
        src, tgt = self._ensure_src_tgt()
        return _cycles.build_symmetric_adjacency(self._nV, self._nE, src, tgt)

    # BUNDLE 2: Overlap Laplacian and adjacency

    @cached_property
    def _overlap_bundle(self) -> dict:
        """Overlap Laplacian L_O and similarity S.

        Uses K = |B_1|^T |B_1| (unsigned Gramian) with row-sum
        normalization: L_O = I - D_ov^{-1/2} K D_ov^{-1/2}.
        L_O is PSD with eigenvalues in [0, 1] by construction.
        """
        if not self._is_standard_only:
            return self._overlap_bundle_general()

        src, tgt = self._ensure_src_tgt()
        nV, nE = self._nV, self._nE

        L_O = _overlap.build_L_O(nV, nE, src, tgt)
        S, d_ov = _overlap.build_overlap_adjacency(nV, nE, src, tgt)
        return {'L_O': L_O, 'S': S, 'd_ov': d_ov}

    def _overlap_bundle_general(self) -> dict:
        """Overlap computation for branching or witness edges.

        Uses the same K = |B_1|^T |B_1| Gramian + row-sum normalization
        as the standard path, but builds K from boundary sets.
        """
        from scipy import sparse as _sp
        bp = np.ascontiguousarray(self._boundary_ptr)
        bi = np.ascontiguousarray(self._boundary_idx)
        nE, nV = self._nE, self._nV

        # M = edge->vertex 0/1 incidence (nE x nV) directly from the boundary CSR;
        # K = |B1|^T |B1| = M M^T is the sparse shared-vertex Gramian (no O(nE^2)
        # Python loop). K_ij = # vertices shared by edges i and j.
        M = _sp.csr_matrix((np.ones(bi.shape[0], dtype=_f64), bi, bp), shape=(nE, nV))
        M.sum_duplicates()
        M.data[:] = 1.0
        K = (M @ M.T).tocsr()

        # Row-sum normalization: L_O = I - D^{-1/2} K D^{-1/2}
        d_ov = np.asarray(K.sum(axis=1)).ravel()
        inv_sqrt = np.zeros(nE, dtype=_f64)
        nz = d_ov > 1e-12
        inv_sqrt[nz] = 1.0 / np.sqrt(d_ov[nz])
        Dis = _sp.diags(inv_sqrt)
        # Consumer (dense build_all_laplacians / rl_pipeline) expects dense L_O/S.
        S = np.ascontiguousarray((Dis @ K @ Dis).toarray(), dtype=_f64)
        L_O = np.eye(nE, dtype=_f64) - S
        return {'L_O': L_O, 'S': S, 'd_ov': d_ov}

    @cached_property
    def L_overlap(self) -> NDArray:
        """Overlap Laplacian L_O = I - D_ov^{-1/2} K D_ov^{-1/2} (normalized G)."""
        return self._overlap_bundle['L_O']

    @cached_property
    def overlap_gramian(self) -> NDArray:
        """Raw overlap Gramian K = |B1|^T |B1| - the CANONICAL integer G channel
        (exact co-incidence counts; reference Part IX). The unnormalized alternate
        to `L_overlap`. Dense nE x nE, consistent with `L_overlap` (the RL4
        character consumer is dense; large graphs take the sparse spectral path,
        which does not form it)."""
        if self._is_standard_only:
            src, tgt = self._ensure_src_tgt()
            K = _overlap.build_overlap_gramian(self._nV, self._nE, src, tgt)
            return _ensure_dense(K)
        # general boundary: K = M M^T from the boundary CSR (0/1 incidence)
        from scipy import sparse as _sp
        bp = np.ascontiguousarray(self._boundary_ptr)
        bi = np.ascontiguousarray(self._boundary_idx)
        M = _sp.csr_matrix((np.ones(bi.shape[0], dtype=_f64), bi, bp),
                           shape=(self._nE, self._nV))
        M.sum_duplicates()
        M.data[:] = 1.0
        return _ensure_dense((M @ M.T).tocsr())

    @property
    def g_channel(self) -> str:
        """Selected G (overlap) channel form: 'normalized' (default) or 'raw'."""
        return getattr(self, "_g_channel", "normalized")

    @cached_property
    def g_channel_operator(self) -> NDArray:
        """The G-channel operator the RL4 character consumes: the raw integer
        Gramian when g_channel='raw' (canonical), else the normalized overlap
        Laplacian (default). Both forms stay available via `overlap_gramian` /
        `L_overlap`; this is the single point where the character selects one."""
        return self.overlap_gramian if self.g_channel == "raw" else self.L_overlap

    @cached_property
    def overlap_similarity(self) -> NDArray:
        """Normalized overlap similarity S, entries in [0, 1]."""
        return self._overlap_bundle['S']

    @cached_property
    def overlap_pairs(self) -> list:
        """Top-k edge pairs by overlap similarity."""
        if not self._is_standard_only:
            return []
        src, tgt = self._ensure_src_tgt()
        return _overlap.build_overlap_pairs(self._nV, self._nE, src, tgt)

    # BUNDLE 3: ALL Laplacians, eigenvalues, Betti, coupling

    @cached_property
    def _chain_col_maxabs(self):
        """Per-face max|.| of the SPARSE product B1 @ B2 - the chain condition
        B1 B2 = 0 tested column-by-column (a structural zero-pattern check).
        Never densifies B1@B2 (nV x nF): both operators are sparse and the
        product is a sparse matmul. Returns f64[nF]."""
        if self._nF == 0 or self._B2_dual is None:
            return np.zeros(0, dtype=_f64)
        B1s = _sparse.to_scipy_csr(self._B1_dual)   # nV x nE
        B2s = _sparse.to_scipy_csr(self._B2_dual)   # nE x nF
        product = (B1s @ B2s).tocsc()               # sparse nV x nF
        if product.nnz == 0:
            return np.zeros(self._nF, dtype=_f64)
        product.data = np.abs(product.data)
        colmax = np.asarray(product.max(axis=0).todense()).ravel()
        return np.ascontiguousarray(colmax, dtype=_f64)

    @cached_property
    def _B2_hodge_dual(self):
        """DualCSR of B2 with chain-violating faces filtered.

        Filters faces where B_1 B_2[:, f] != 0, which violate the chain
        complex axiom. Direct algebraic check via the sparse chain condition
        (`_chain_col_maxabs`) - no dense B1@B2.
        """
        if self._nF == 0 or self._B2_dual is None:
            return None

        colmax = self._chain_col_maxabs
        keep = [f for f in range(self._nF) if colmax[f] < 1e-10]

        if len(keep) == self._nF:
            return self._B2_dual

        if len(keep) == 0:
            return None

        # Slice the kept face columns sparsely and rebuild the DualCSR.
        B2s = _sparse.to_scipy_csr(self._B2_dual).tocsc()
        B2_filtered = B2s[:, keep].tocsr()
        return _sparse.from_scipy_csr(B2_filtered)

    @cached_property
    def B2_hodge(self) -> NDArray:
        """B_2 with self-loop faces filtered for exact Hodge decomposition.

        Excludes faces whose boundary edges span fewer than 3 distinct
        vertices, which would violate B_1 B_2 = 0.
        """
        if self._B2_hodge_dual is None:
            return np.zeros((max(self._nE, 1), 0), dtype=_f64)
        return _sparse.to_dense_f64(self._B2_hodge_dual)

    @cached_property
    def nF_hodge(self) -> int:
        """Number of faces in B2_hodge (excluding self-loop faces)."""
        return self.B2_hodge.shape[1]

    @cached_property
    def self_loop_face_indices(self) -> list:
        """Indices of faces excluded from B2_hodge (chain-violating faces)."""
        if self._nF == 0:
            return []
        colmax = self._chain_col_maxabs
        return [f for f in range(self._nF) if colmax[f] >= 1e-10]

    @cached_property
    def spectral_bundle(self) -> dict:
        """All Laplacians, spectral decompositions, and relational Laplacian.

        The vertex-space layer (L0, its spectrum, Betti via union-find, Fiedler via
        matrix-free ARPACK) plus the cheap+exact edge-space quantities (c^2 = alpha_G,
        the L1 Fiedler eigenpair). This is the SCALE-FREE path and it is now the
        universal default: there is no dense-vs-sparse size cutoff. The edge-space RL /
        hats / full eigenbases are never eagerly materialized here - the character,
        energy, propagation, and interfacing layers all read them matrix-free from the
        sparse RL4 (see sparse_character / sparse_interfacing). The full DENSE relational
        bundle is available ON DEMAND via `_dense_rcf_bundle` (used by the low-level
        dense Cython kernels and their unit tests), i.e. dense only when explicitly
        asked for, never as a fixed gate. `eigen_dense_limit` no longer gates this.
        """
        bundle = _laplacians.build_all_laplacians_sparse(
            self._B1_dual,
            self._B2_hodge_dual,
            self._nV, self._nE, self._nF,
        )
        self._fill_cheap_edge_spectra(bundle)   # alpha_G + L1 Fiedler eigenpair
        return bundle

    @cached_property
    def _dense_rcf_bundle(self) -> dict:
        """The full DENSE relational bundle (RL, hats, nhats, trace_values, hat_names,
        chi, K1, L_C, RL_1 + edge-space eigenbases) via the dense Cython builder.

        Materialized ON DEMAND only - the public scale-free API never touches it (it
        reads everything matrix-free from the sparse RL4). It backs the low-level dense
        kernels (`_character.hat_eigen`, `_query.explain_edge`, `_rcfe.coupling_tensor`,
        `_channels.primal_signal_character`, `_interfacing.build_interfacing_bundle`) and
        their unit tests, and the dense-oracle side of the parity tests. Building it is
        O(nE^2) memory / O(nE^3) eigencost, so a caller that reaches for it on a large
        complex is explicitly opting into the dense cost."""
        L_SG = None
        if _HAS_RCF:
            # F = T - G (integer/exact tower, Def 3.3) - the default frustration.
            L_SG = _ensure_dense(self.frustration_exact)
        return _laplacians.build_all_laplacians(
            self._B1_dual,
            self._B2_hodge_dual,
            self.g_channel_operator,   # raw Gramian G (default) or normalized L_O
            L_SG_in=L_SG,
            auto_alpha=True,
            k=-1,
        )

    def _fill_cheap_edge_spectra(self, bundle) -> None:
        """Fill only the O(nnz) edge-space coupling into the scale-free bundle:
            alpha_G = c^2 = G/T = tr((B2 B2^T)^2)/tr((B1^T B1)^2)   (exact integer traces, cheap)

        The L1 Fiedler eigenpair is DELIBERATELY not built here. It needs an ARPACK smallest-
        eigenvalue solve that costs O(seconds) on a large edge space, and the character / coherence /
        agent-monitor hot path (which rebuilds spectral_bundle constantly) never reads it. It is a
        lazy accessor instead (`edge_fiedler` / `fiedler_val_L1` / `fiedler_vec_L1`), computed on
        first demand. Building it eagerly here was the dominant cost of a large-hive monitor step.
        """
        nE = self._nE
        if nE == 0:
            return
        try:
            import scipy.sparse as sp
            from rexgraph.core._sparse import to_scipy_csr
            B1 = to_scipy_csr(self._B1_dual).astype(np.float64)          # nV x nE
            L1_down = sp.csr_matrix(B1.T @ B1)                           # gradient tier
            if self._nF > 0:
                B2 = to_scipy_csr(self._B2_hodge_dual).astype(np.float64)  # nE x nF
                L1_up = sp.csr_matrix(B2 @ B2.T)                         # curl tier
            else:
                L1_up = sp.csr_matrix((nE, nE))
            # alpha_G = c^2 = G/T = ||L1_up||_F^2 / ||L1_down||_F^2 (the DOWN/UP exchange rate,
            # = (k-2)/2 on K_k). Same value as the dense path, integer traces, no eigensolve.
            calT = float(L1_down.multiply(L1_down).sum())               # tr((B1^T B1)^2)
            calG = float(L1_up.multiply(L1_up).sum())                   # tr((B2 B2^T)^2)
            bundle['alpha_G'] = (calG / calT) if calT > 0.0 else 0.0
        except Exception:
            pass   # cheap path unavailable -> leave the builder's None/NaN slots
        # The sparse builder writes a placeholder fiedler_val_L1 = 0.0 that is NOT the real
        # value (the L1 Fiedler is the lazy `edge_fiedler` / `fiedler_val_L1` accessor). Null it
        # so a bundle-level reader gets an explicit None ("not in the bundle") instead of silently
        # trusting a wrong 0.0.
        bundle['fiedler_val_L1'] = None

    @cached_property
    def edge_fiedler(self) -> Tuple[float, NDArray]:
        """(fiedler_val_L1, fiedler_vec_L1): the smallest NONZERO eigenpair of the edge Laplacian
        L1 = B1^T B1 + B2 B2^T - the algebraic connectivity of the edge space. Computed ON DEMAND
        (ARPACK smallest-eigenvalue solve, expensive on a large edge space), skipping exactly the
        beta1 known kernel modes (no float threshold). Not built into spectral_bundle, which the
        character/coherence hot path rebuilds constantly and never needs the Fiedler for."""
        nE = self._nE
        if nE == 0:
            return 0.0, np.zeros(0, dtype=_f64)
        import scipy.sparse as sp
        from rexgraph.core._sparse import to_scipy_csr
        B1 = to_scipy_csr(self._B1_dual).astype(np.float64)
        L1 = sp.csr_matrix(B1.T @ B1)
        if self._nF > 0 and self._B2_hodge_dual is not None:
            B2 = to_scipy_csr(self._B2_hodge_dual).astype(np.float64)
            L1 = sp.csr_matrix(L1 + B2 @ B2.T)
        b1 = int(self.betti[1])
        ev1, V1 = _edge_low_eig(L1, nE, b1 + 4)
        val = float(ev1[b1]) if b1 < len(ev1) else 0.0
        vec = V1[:, b1].copy() if b1 < V1.shape[1] else np.zeros(nE, dtype=_f64)
        return val, vec

    @property
    def fiedler_val_L1(self) -> float:
        """Algebraic connectivity of the edge space (smallest nonzero L1 eigenvalue). Lazy."""
        return self.edge_fiedler[0]

    @property
    def fiedler_vec_L1(self) -> NDArray:
        """Fiedler vector of the edge Laplacian L1. Lazy."""
        return self.edge_fiedler[1]

    # Spectral accessors (thin dict lookups into spectral_bundle)

    @cached_property
    def L0(self) -> NDArray:
        """L_0 = B_1 B_1^T."""
        L0_val = self.spectral_bundle['L0']
        if L0_val is not None:
            return _ensure_dense(L0_val)
        # Sparse mode: build L0 on demand (nV x nV, usually feasible)
        return _ensure_dense(_laplacians.build_L0(self._B1_dual))

    @cached_property
    def L1(self) -> NDArray:
        """L_1 = B_1^T B_1 + B_2_hodge B_2_hodge^T (edge Hodge Laplacian)."""
        L1_full = self.spectral_bundle.get('L1_full')
        if L1_full is not None:
            return _ensure_dense(L1_full)
        # L1 not in bundle; build on demand using B2_hodge
        L1_down = _laplacians.build_L1_down(self._B1_dual)
        if self.nF_hodge > 0 and self._B2_hodge_dual is not None:
            L1_up = _laplacians.build_L1_up(self._B2_hodge_dual)
            return _ensure_dense(_laplacians.build_L1_full(L1_down, L1_up))
        return _ensure_dense(L1_down)

    @cached_property
    def L2(self) -> NDArray:
        """L_2 = B_2^T B_2 (face Laplacian). Built on demand in the scale-free path,
        where the bundle carries no dense edge/face operators."""
        L2 = self.spectral_bundle.get('L2')
        if L2 is not None:
            return _ensure_dense(L2)
        if self.nF_hodge == 0 or self._B2_hodge_dual is None:
            return np.zeros((self.nF_hodge, self.nF_hodge), dtype=_f64)
        return _ensure_dense(_laplacians.build_L2(self._B2_hodge_dual))

    # Scale-safe SPARSE Laplacian accessors
    # The public L0/L1/L2/L_overlap/overlap_gramian properties return dense
    # ndarrays: that is the documented API and what the dense character/RL kernel
    # and the io/viz consumers expect. Those densify nE x nE (nV x nV for L0) and
    # so OOM if accessed on a very large graph. These *_sparse accessors return the
    # SAME operators as scipy CSR (nnz ~ 2*nE) via the sparse core builders - no
    # densification - for callers that need the operator, not a dense matrix. (The
    # agent pipeline already hand-rolls this pattern; see pipeline._sparse_L0.)

    @cached_property
    def L0_sparse(self):
        """L_0 = B_1 B_1^T = D - A as a sparse scipy CSR (no densification)."""
        return _laplacians.build_L0_sparse(self._B1_dual)

    @cached_property
    def L1_sparse(self):
        """L_1 (edge Hodge Laplacian) as a sparse scipy CSR: the down part
        B_1^T B_1, plus the up part B_2 B_2^T when Hodge faces exist."""
        L1 = _laplacians.build_L1_down_sparse(self._B1_dual)
        if self.nF_hodge > 0 and self._B2_hodge_dual is not None:
            L1 = (L1 + _laplacians.build_L1_up_sparse(self._B2_hodge_dual)).tocsr()
        return L1

    @cached_property
    def L2_sparse(self):
        """L_2 = B_2^T B_2 as a sparse scipy CSR (0x0 when there are no faces)."""
        from scipy import sparse as _sp
        if self.nF_hodge == 0 or self._B2_hodge_dual is None:
            return _sp.csr_matrix((0, 0), dtype=_f64)
        return _laplacians.build_L2_sparse(self._B2_hodge_dual)

    @cached_property
    def overlap_gramian_sparse(self):
        """Raw overlap Gramian K = |B1|^T |B1| (canonical integer G channel) as a
        sparse scipy CSR - the non-densifying form of `overlap_gramian`."""
        from scipy import sparse as _sp
        if self._is_standard_only:
            src, tgt = self._ensure_src_tgt()
            K = _overlap.build_overlap_gramian(self._nV, self._nE, src, tgt)
            return K.tocsr() if _sp.issparse(K) else _sp.csr_matrix(K)
        bp = np.ascontiguousarray(self._boundary_ptr)
        bi = np.ascontiguousarray(self._boundary_idx)
        M = _sp.csr_matrix((np.ones(bi.shape[0], dtype=_f64), bi, bp),
                           shape=(self._nE, self._nV))
        M.sum_duplicates()
        M.data[:] = 1.0
        return (M @ M.T).tocsr()

    @cached_property
    def betti(self) -> Tuple[int, int, int]:
        """Betti numbers (beta_0, beta_1, beta_2) - EIGEN-FREE, from ranks/union-find,
        not from a spectrum: beta_0 by union-find over the components, rank(B_k) by
        exact rational column reduction (the canon's Z/Q-elimination, no SVD, no
        eigendecomposition). Equals the dense-spectrum betti exactly; the spectral
        bundle's spectrum-derived betti remains available as the oracle."""
        from rexgraph.graded_boundary import betti_numbers
        b = betti_numbers(self.graded_boundaries())
        b = (list(b) + [0, 0, 0])[:3]        # pad to the (beta0, beta1, beta2) contract
        return (int(b[0]), int(b[1]), int(b[2]))

    @cached_property
    def dirac_dimension(self) -> int:
        """Number of Dirac modes = dim of the Dirac operator D = nV + nE + nF_hodge.
        Exact - no dense (nV+nE+nF)² operator is materialized."""
        return int(self._nV + self._nE + self.nF_hodge)

    @cached_property
    def dirac_harmonic_count(self) -> int:
        """Number of zero (harmonic) Dirac modes = dim ker(D) = Σ Betti (β₀+β₁+β₂),
        the total homology dimension (ker D = ⊕ₖ ker Lₖ). An EXACT integer invariant
        - no dense Dirac, no eigendecomposition."""
        b = self.betti
        return int(b[0] + b[1] + b[2])

    @cached_property
    def field_coupling_psd(self) -> Tuple[float, bool]:
        """(coupling g, is_psd) for the coupled field operator
        M = [[RL1, -g·B2],[-g·B2ᵀ, L2]], computed WITHOUT a dense (nE+nF)² operator:
        g = 1/max(‖B2‖_F, 1) (cheap, matches _field), and is_psd = (smallest
        eigenvalue of the SPARSE block M ≥ -ε) via Lanczos (eigsh, k=1) - no full
        eigendecomposition. Scale-safe."""
        import scipy.sparse as _sp
        import scipy.sparse.linalg as _sla
        nF = int(self.nF_hodge)
        if nF == 0 or self._B2_hodge_dual is None:
            return 1.0, True                      # M = RL1 (a Laplacian) -> PSD
        from rexgraph.core._sparse import to_scipy_csr
        B2 = to_scipy_csr(self._B2_hodge_dual).tocsr()          # nE × nF
        b2_frob = float(np.sqrt(B2.multiply(B2).sum()))
        g = 1.0 / (b2_frob if b2_frob > 1.0 else 1.0)
        RL1 = self.relational_laplacian
        RL1 = (_sp.csr_matrix(np.asarray(RL1)) if RL1 is not None
               else self.L1_sparse.tocsr())       # RL_1 if built, else L1 (fallback)
        L2 = self.L2_sparse.tocsr()
        M = _sp.bmat([[RL1, (-g) * B2], [(-g) * B2.T, L2]], format='csr')
        n = M.shape[0]
        try:
            # Same EXACT quantity (smallest eigenvalue of M) either way - this is a
            # solver-capability boundary, not an accuracy-at-scale trade: ARPACK's
            # eigsh needs k < n and is unreliable at n≤3, so tiny M uses the direct
            # dense eig; both are exact.
            if n <= 3:
                lam_min = float(np.linalg.eigvalsh(M.toarray()).min())
            else:
                lam_min = float(_sla.eigsh(M, k=1, which='SA', return_eigenvectors=False,
                                           maxiter=n * 20, tol=1e-5)[0])
        except Exception:
            lam_min = float(np.linalg.eigvalsh(M.toarray()).min())
        return g, bool(lam_min >= -1e-9)

    @property
    def euler_characteristic(self) -> int:
        """Euler characteristic chi = n - m + f."""
        return self._nV - self._nE + self._nF

    @cached_property
    def eigenvalues_L0(self) -> NDArray:
        return self.spectral_bundle['evals_L0']

    @cached_property
    def fiedler_vector_L0(self) -> NDArray:
        return self.spectral_bundle['fiedler_vec_L0']

    @cached_property
    def fiedler_overlap(self) -> Tuple[float, NDArray]:
        """Fiedler value and vector of L_O."""
        sb = self.spectral_bundle
        val = sb.get('fiedler_L_O', 0.0)
        vec = sb.get('fiedler_vec_L_O')
        if vec is None:
            vec = np.zeros(self._nE, dtype=_f64)
        return float(val), vec

    @cached_property
    def coupling_constants(self) -> Tuple[float, float]:
        """Coupling constants (alpha_G, alpha_T)."""
        sb = self.spectral_bundle
        return (
            sb.get('alpha_G', float('nan')),
            sb.get('alpha_T', 0.0),
        )

    @cached_property
    def alpha_G(self) -> float:
        """Geometric coupling constant fiedler(L_1) / fiedler(L_O)."""
        return self.coupling_constants[0]

    @cached_property
    def relational_laplacian(self) -> Optional[NDArray]:
        """Relational Laplacian RL_1 = L1_down + alpha_G * L1_up (gradient + c^2*curl).

        Built on demand in the scale-free path (where the dense bundle skips it). None only when
        alpha_G is NaN.
        """
        RL = self.spectral_bundle.get('RL_1')
        if RL is not None:
            return _ensure_dense(RL)
        aG = self.spectral_bundle.get('alpha_G', float('nan'))
        if aG != aG:                                    # NaN -> no coupling available
            return None
        return _ensure_dense(self.L1_down) + float(aG) * _ensure_dense(self.L1_up)

    @cached_property
    def evals_RL1(self) -> Optional[NDArray]:
        """Eigenvalues of RL_1."""
        return self.spectral_bundle.get('evals_RL_1')

    @cached_property
    def evecs_RL1(self) -> Optional[NDArray]:
        """Eigenvectors of RL_1."""
        return self.spectral_bundle.get('evecs_RL_1')

    # Dense operator accessors (used by _state.RexState methods)

    @cached_property
    def B1_dense(self) -> NDArray:
        """Dense B_1, shape (nV, nE)."""
        return self.B1

    @cached_property
    def B2_dense(self) -> NDArray:
        """Dense B_2 (all faces), shape (nE, nF)."""
        return self.B2

    @cached_property
    def LO(self) -> NDArray:
        """Overlap Laplacian L_O (alias for L_overlap)."""
        return self.L_overlap

    @cached_property
    def L1_full(self) -> NDArray:
        """Full edge Hodge Laplacian (alias for L1)."""
        return self.L1

    @cached_property
    def L1_down(self) -> NDArray:
        """Down (gradient) edge Laplacian B_1^T B_1. Built on demand in the scale-free path."""
        v = self.spectral_bundle.get('L1_down')
        if v is not None:
            return _ensure_dense(v)
        return _ensure_dense(_laplacians.build_L1_down(self._B1_dual))

    @cached_property
    def L1_up(self) -> NDArray:
        """Up (curl) edge Laplacian B_2 B_2^T. Zero when there are no faces."""
        v = self.spectral_bundle.get('L1_up')
        if v is not None:
            return _ensure_dense(v)
        if self._nF == 0:
            return np.zeros((self._nE, self._nE), dtype=_f64)
        return _ensure_dense(_laplacians.build_L1_up(self._B2_hodge_dual))

    @cached_property
    def alpha_T(self) -> float:
        """Topological coupling constant beta_1 / nE."""
        return self.spectral_bundle.get('alpha_T', 0.0)

    @property
    def alpha0(self) -> float:
        """Vertex-tier diffusion rate (default 1.0)."""
        return getattr(self, '_alpha0', 1.0)

    @alpha0.setter
    def alpha0(self, value: float):
        self._alpha0 = float(value)

    @property
    def alpha2(self) -> float:
        """Face-tier diffusion rate (default 1.0)."""
        return getattr(self, '_alpha2', 1.0)

    @alpha2.setter
    def alpha2(self, value: float):
        self._alpha2 = float(value)

    @cached_property
    def rex_laplacian(self) -> Optional[NDArray]:
        """Alias for relational_laplacian (used by RexState)."""
        return self.relational_laplacian

    @cached_property
    def harmonic_space(self) -> NDArray:
        """Basis for ker(L_1).

        Rows are an orthonormal basis of harmonic edge signals.
        """
        sb = self.spectral_bundle
        evals = sb.get('evals_L1')
        evecs = sb.get('evecs_L1')
        if evals is not None and evecs is not None:
            mask = evals < 1e-10
            return evecs[:, mask].T
        # Eigensolve was deferred; compute on demand
        evals_all, evecs_all = np.linalg.eigh(self.L1)
        mask = evals_all < 1e-10
        return evecs_all[:, mask].T

    # RCF bundles and accessors

    @cached_property
    def _edge_signs(self) -> NDArray:
        """Edge signs for frustration Laplacian. +1/-1 per edge."""
        if self._signs is not None:
            return np.asarray(self._signs, dtype=_f64)
        return np.ones(self._nE, dtype=_f64)

    @cached_property
    def frustration_exact(self):
        """Doc-exact frustration channel F = T - G (INTEGER, Def 3.3), as a sparse
        scipy CSR. F[i,j] = T[i,j] - G[i,j] off-diagonal (0 same-orientation,
        -2 opposite at a shared vertex), F[i,i] = Σ_j|F[i,j]|. Pure integer - the
        exact/Hodge tower - built from T = B₁ᵀB₁ and G = |B₁|ᵀ|B₁| (no float weights,
        unlike the legacy signed-Gramian `L_frustration_weighted`)."""
        from scipy import sparse as _sp
        T = _laplacians.build_L1_down_sparse(self._B1_dual).tocsr()   # B₁ᵀB₁
        G = self.overlap_gramian_sparse.tocsr()                      # |B₁|ᵀ|B₁|
        Foff = (T - G).tocsr()
        Foff.setdiag(0.0); Foff.eliminate_zeros()
        d = np.asarray(np.abs(Foff).sum(axis=1)).ravel()
        return (Foff + _sp.diags(d)).tocsr()

    @cached_property
    def L_frustration(self) -> NDArray:
        """Frustration channel F = T - G (integer/exact tower; Def 3.3). Dense
        nE×nE view of `frustration_exact`."""
        if not _HAS_RCF:
            return np.zeros((self._nE, self._nE), dtype=_f64)
        return _ensure_dense(self.frustration_exact)

    @cached_property
    def L_frustration_weighted(self) -> NDArray:
        """Legacy inverse-log-degree *weighted* signed-Gramian frustration - the
        geometric/approximation-tower alternate (float weights). Kept for the
        explicit integer-vs-weighted distinction; not used in the default RL4."""
        if not _HAS_RCF:
            return np.zeros((self._nE, self._nE), dtype=_f64)
        src, tgt = self._ensure_src_tgt()
        return _frustration.build_L_SG(
            self._nV, self._nE, src, tgt, signs=self._edge_signs)

    @cached_property
    def L_coPC(self) -> Optional[NDArray]:
        """Copath complex Laplacian L_C (line-graph Hodge).

        None if the line graph has no edges or trace is zero.
        Read from spectral_bundle (computed once during build_all_laplacians).
        """
        return self.spectral_bundle.get('L_C')

    @cached_property
    def _rcf_bundle(self) -> dict:
        """Dense relational bundle: RL, hats, nhats, trace_values, hat_names, chi.

        Now backed by `_dense_rcf_bundle` (built ON DEMAND), NOT by spectral_bundle -
        the default spectral_bundle is scale-free and never carries the dense edge-space
        RL/hats. The public API reads character/energy/propagation matrix-free and never
        reaches this; it exists for the low-level dense Cython kernels and their tests.
        Empty dict if the dense build produced no active hats (nE == 0)."""
        bundle = self._dense_rcf_bundle
        if bundle.get('RL') is None or bundle.get('nhats', 0) == 0:
            return {}
        return bundle

    @cached_property
    def RL(self) -> NDArray:
        """Relational Laplacian RL = sum of trace-normalized typed Laplacians.

        tr(RL) = nhats. When L_coPC is available, nhats = 4 (RL4).
        Otherwise nhats = 3 (RL3).

        RL is inherently a dense nE x nE object (callers do np.trace / eigvalsh /
        RL.T on it). On the scale-free sparse path the dense bundle never built it,
        so materialize it ON DEMAND from the sparse RL4 - dense only when the dense
        accessor is actually touched, never as a fixed size gate. The matrix-free
        moment quantities use self._rl4_sparse and never reach this accessor.
        """
        if self._use_sparse_character:
            return np.ascontiguousarray(self._rl4_sparse.toarray(), dtype=_f64)
        rcf = self._rcf_bundle
        return rcf.get('RL', np.zeros((self._nE, self._nE), dtype=_f64))

    @cached_property
    def _use_sparse_character(self) -> bool:
        """True when the character/coherence quantities must come from the
        scale-free sparse path: the dense spectral bundle did not build RL (large
        graphs, nE > eigen_dense_limit) but the RCF core is available. Reads the
        bundle's 'RL' slot directly - it is None in sparse mode - so this never
        allocates the dense RL (which would OOM at scale)."""
        return (_HAS_RCF and self._nE > 0
                and self.spectral_bundle.get('RL') is None)

    @cached_property
    def _sparse_character(self) -> dict:
        """The O(nnz) character bundle {chi, chi_star, nhats, hat_names, RL, hats,
        rl_diag} - per-edge character and star-average from DIAGONALS only, no
        per-vertex solves. The per-vertex Green's phi/kappa is computed separately
        and lazily (``_sparse_phi``) so accessing chi never pays the nV solves."""
        from rexgraph.sparse_character import build_sparse_character_cheap
        return build_sparse_character_cheap(self)

    @cached_property
    def _sparse_phi(self) -> dict:
        """Per-vertex Green's character {phi, kappa} (nV block-CG solves) - computed
        lazily, only when vertex_character/coherence is actually accessed."""
        from rexgraph.sparse_character import compute_sparse_phi
        return compute_sparse_phi(self, self._sparse_character)

    @cached_property
    def nhats(self) -> int:
        """Number of active hat operators in the relational Laplacian."""
        if self._use_sparse_character:
            return self._sparse_character['nhats']
        return self._rcf_bundle.get('nhats', 3)

    @cached_property
    def hat_names(self) -> list:
        """Active channel names (['L1_down','L_O','L_SG','L_C'] order), hybrid-aware
        so callers get the right labels on both the dense and scale-free sparse
        paths (the dense `_rcf_bundle` is empty when the sparse path fired)."""
        if self._use_sparse_character:
            return list(self._sparse_character.get('hat_names', []))
        return list(self._rcf_bundle.get('hat_names', []))

    @cached_property
    def _rl_eigen(self) -> Tuple[NDArray, NDArray]:
        """Cached eigendecomposition of RL."""
        if not _HAS_RCF:
            return np.linalg.eigh(self.RL)
        return _relational.rl_eigen(self.RL)

    @cached_property
    def _green_cache(self) -> dict:
        """Green function cache: B1 @ RL^-1, S0.

        RL3/RL4 is full-rank SPD, so RL^+ = RL^-1: the primary path factors RL
        once (Cholesky) and solves for B1_RLp directly - no eigendecomposition and
        no dense nE x nE pseudoinverse. Falls back to the spectral pinv path only
        if RL is not numerically SPD (degenerate / empty).
        """
        if not _HAS_RCF:
            return {}
        gc = _relational.build_green_cache_spd(self.RL, self.B1)
        if gc is not None:
            return gc
        evals, evecs = self._rl_eigen
        return _relational.build_green_cache(
            self.RL, self.B1, evals, evecs)

    @cached_property
    def _vertex_bundle(self) -> dict:
        """Vertex character phi, star character chi*, coherence kappa.

        Keys: phi, chi_star, kappa
        """
        if not _HAS_RCF:
            return {}
        rcf = self._rcf_bundle
        if 'hats' not in rcf or 'nhats' not in rcf:   # sparse-character mode has no dense hats;
            return {}                                 # callers use .get() defaults (like structural_character)
        src, tgt = self._ensure_src_tgt()
        v2e_ptr, v2e_idx = self._v2e
        return _character.build_character_bundle(
            self.B1, self.RL, rcf['hats'], rcf['nhats'],
            self._nV, self._nE,
            np.asarray(v2e_ptr, dtype=_i32),
            np.asarray(v2e_idx, dtype=_i32),
            green_cache=self._green_cache,
        )

    @cached_property
    def structural_character(self) -> NDArray:
        """chi(sigma) in Delta^{nhats-1} per edge. Shape (nE, nhats)."""
        if self._use_sparse_character:
            return self._sparse_character['chi']
        return self._rcf_bundle.get('chi',
                                     np.zeros((self._nE, self.nhats), dtype=_f64))

    @cached_property
    def vertex_character(self) -> NDArray:
        """phi(v) in Delta^{nhats-1} per vertex. Shape (nV, nhats)."""
        if self._use_sparse_character:
            return self._sparse_phi['phi']
        return self._vertex_bundle.get('phi',
                                        np.zeros((self._nV, self.nhats), dtype=_f64))

    @cached_property
    def star_character(self) -> NDArray:
        """chi*(v) = mean of chi(e) over incident edges. Shape (nV, nhats)."""
        if self._use_sparse_character:
            return self._sparse_character['chi_star']
        return self._vertex_bundle.get('chi_star',
                                        np.zeros((self._nV, self.nhats), dtype=_f64))

    @cached_property
    def coherence(self) -> NDArray:
        """kappa(v) = 1 - 0.5 * ||phi(v) - chi*(v)||_1. Shape (nV,)."""
        if self._use_sparse_character:
            return self._sparse_phi['kappa']
        return self._vertex_bundle.get('kappa',
                                        np.zeros(self._nV, dtype=_f64))

    @cached_property
    def local_coherence(self) -> NDArray:
        """O(nnz) per-vertex coherence κ_loc(v) = 1 - 0.5·mean_{e∈star(v)}‖χ(e)-χ*(v)‖₁
        - how consistent a vertex's incident-edge characters are with their star
        average. Uses only the per-edge character χ and χ* (diagonals, no solves),
        so it is available at every scale. This is the LOCAL-tower coherence; the
        per-vertex Green's `coherence` (κ vs the global φ) is the exact but O(nV·solve)
        companion (see Part B: local energy and global Green's are two moments of one
        propagator). Shape (nV,)."""
        nhats = int(self.nhats)
        if nhats == 0 or self._nV == 0:
            return np.ones(self._nV, dtype=_f64)
        chi = np.asarray(self.structural_character, dtype=_f64)      # (nE, nhats)
        chi_star = np.asarray(self.star_character, dtype=_f64)       # (nV, nhats)
        v2e_ptr, v2e_idx = self._v2e
        v2e_ptr = np.asarray(v2e_ptr); v2e_idx = np.asarray(v2e_idx)
        chi_inc = chi[v2e_idx] if v2e_idx.size else chi[:0]
        kloc = np.ones(self._nV, dtype=_f64)
        for v in range(self._nV):
            lo, hi = int(v2e_ptr[v]), int(v2e_ptr[v + 1])
            if hi > lo:
                dev = np.abs(chi_inc[lo:hi] - chi_star[v]).sum(axis=1)  # per incident edge
                kloc[v] = 1.0 - 0.5 * float(dev.mean())
        return kloc

    @cached_property
    def vertex_energy_character(self) -> NDArray:
        """Per-vertex LOCAL energy character - the per-edge energy diag(RL4²)
        (row-norms, O(nnz)) aggregated over each vertex's star through the boundary
        B₁. This is the vertex propagator's local end via the boundary (Part B /
        script 14/15), NOT a Green's solve. Shape (nV,)."""
        ec = np.asarray(self.energy_character, dtype=_f64)              # per-edge O(nnz)
        v2e_ptr, v2e_idx = self._v2e
        v2e_ptr = np.asarray(v2e_ptr); v2e_idx = np.asarray(v2e_idx)
        out = np.zeros(self._nV, dtype=_f64)
        if v2e_idx.size:
            ec_inc = ec[v2e_idx]
            for v in range(self._nV):
                lo, hi = int(v2e_ptr[v]), int(v2e_ptr[v + 1])
                if hi > lo:
                    out[v] = float(ec_inc[lo:hi].mean())
        return out

    @cached_property
    def vertex_scale_profile(self) -> NDArray:
        """Local scale character (Part B / script 15): the closed-k-walk moments
        (L0^k)_vv per vertex for k = 0,1,2,3 - the heat kernel's LOCAL end, i.e. the
        star neighborhood's structure at each scale, via sparse matvecs / row-norms
        (no eigendecomposition). k≤2 are exact O(nnz) (1, deg, ‖L0[v,:]‖²); k=3 =
        closed 3-walks (clustering/triangles) via one sparse L0² pass. Two vertices
        of equal degree agree at k≤1 and DIVERGE at k≥2 by local clustering - the
        local<->global bridge. Shape (nV, 4): [1, deg, (L0²)_vv, (L0³)_vv]."""
        L0 = self.L0_sparse.tocsr()
        n = L0.shape[0]
        prof = np.zeros((n, 4), dtype=_f64)
        if n == 0:
            return prof
        prof[:, 0] = 1.0
        prof[:, 1] = L0.diagonal()                                      # (L0)_vv = deg
        prof[:, 2] = np.asarray(L0.multiply(L0).sum(axis=1)).ravel()    # (L0²)_vv = ‖row‖²
        L02 = (L0 @ L0).tocsr()
        prof[:, 3] = np.asarray(L0.multiply(L02).sum(axis=1)).ravel()   # (L0³)_vv
        return prof

    @cached_property
    def scale_bridge(self) -> dict:
        """Local<->global structure across the scale profile (Part B / script 15). The
        low-order closed-walk moments (L0^k)_vv are the LOCAL character; two vertices
        of equal degree agree at k≤1 and DIVERGE at k≥2 by their clustering - the
        thing the star neighborhood exposes. The clean per-vertex clustering signal is
        the local clustering coefficient C(v) = 2·triangles(v)/(deg(deg-1)), with
        triangles(v) = (A³)_vv/2 (A = adjacency = D - L0). All sparse matvecs, O(nnz)
        for bounded degree (one A² pass). Returns O(nnz) summaries + the per-vertex
        clustering coefficient (0 = star/path, 1 = fully clustered)."""
        import scipy.sparse as _sp
        prof = self.vertex_scale_profile
        deg = prof[:, 1]
        m2, m3 = prof[:, 2], prof[:, 3]
        L0 = self.L0_sparse.tocsr()
        n = L0.shape[0]
        clustering = np.zeros(n, dtype=_f64)
        if n > 0:
            A = (_sp.diags(deg) - L0).tocsr()                 # adjacency
            A2 = (A @ A).tocsr()
            tri = 0.5 * np.asarray(A.multiply(A2).sum(axis=1)).ravel()   # A³_vv/2
            with np.errstate(divide='ignore', invalid='ignore'):
                clustering = np.where(deg > 1, 2.0 * tri / (deg * (deg - 1.0)), 0.0)
        return {
            'scale2_mean': float(m2.mean()) if m2.size else 0.0,
            'scale3_mean': float(m3.mean()) if m3.size else 0.0,
            'clustering_per_vertex': clustering,
            'clustering_mean': float(clustering.mean()) if clustering.size else 0.0,
        }

    @cached_property
    def character_varentropy(self) -> dict:
        """The varentropy self-diagnostic (Part D.4 / script 19): the H₂-H₃ gap of
        RL4's normalized spectrum. H₂ = -log(tr RL4²/tr RL4)² is the default coherence
        (Rényi-2, O(nnz)); H₃ costs one extra sparse matmul (tr RL4³). Their gap is a
        CHEAP certificate of when H₂ is trustworthy: ~0 on flat/unweighted spectra
        (H₂ is exact), growing with weight-induced non-uniformity (the spectrum
        carries structure the 2nd moment alone misses). Rényi is non-increasing in
        order so gap ≥ 0. Returns {'H2','H3','gap'}."""
        from rexgraph import scale_propagator as _spg
        X = self._rl4_sparse
        H2 = float(_spg.renyi_entropy(X, 2))
        H3 = float(_spg.renyi_entropy(X, 3))
        return {'H2': round(H2, 6), 'H3': round(H3, 6),
                'gap': round(max(0.0, H2 - H3), 6)}

    def weighted_curvature_signature(self, w_e: NDArray = None) -> dict:
        """The weighted geometric signature (Part F / script 20): curvature is the
        weighted-chain residual R = B₁(W-I)B₂ = B₁WB₂ (using B₁B₂=0) - the deviation
        of the weighted state from the unweighted ∂²=0 ideal, zero iff W=cI. Sparse
        (no dense B1/B2/R), decomposed by group, all O(nnz):
          per-vertex  ‖R[v,:]‖  (which junction bends most),
          per-face    ‖R[:,f]‖  (which filled cycle is most strained),
          per-edge    |w_e-1|·‖B₁[:,e]‖·‖B₂[e,:]‖  (additive rank-1 contributions),
        plus weight concentration N_eff=(Σw)²/Σw² and curvature-per-weight. `w_e`
        defaults to the graph's edge weights (unit -> R=0, no curvature)."""
        import scipy.sparse as _sp
        from rexgraph.core._sparse import to_scipy_csr
        nE, nF = int(self._nE), int(self.nF_hodge)
        w = (np.abs(np.asarray(self.w_E, dtype=_f64)).ravel()
             if w_e is None and getattr(self, 'w_E', None) is not None
             else (np.ones(nE, dtype=_f64) if w_e is None
                   else np.abs(np.asarray(w_e, dtype=_f64)).ravel()))
        if w.size != nE:
            w = np.ones(nE, dtype=_f64)
        n_eff = float((w.sum() ** 2) / max(float((w * w).sum()), 1e-30))
        if nF == 0 or self._B2_hodge_dual is None:
            z = np.zeros(0, dtype=_f64)
            return {'total_curvature': 0.0, 'per_vertex': np.zeros(self._nV),
                    'per_face': z, 'per_edge': np.zeros(nE),
                    'n_eff': n_eff, 'curvature_per_weight': 0.0, 'weighted': False}
        B1 = to_scipy_csr(self._B1_dual).tocsr()             # nV × nE
        B2 = to_scipy_csr(self._B2_hodge_dual).tocsr()       # nE × nF
        R = (B1 @ (_sp.diags(w - 1.0) @ B2)).tocsr()         # nV × nF, sparse
        per_vertex = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel())
        per_face = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel())
        b1n = np.sqrt(np.asarray(B1.multiply(B1).sum(axis=0)).ravel())      # ‖B1[:,e]‖
        b2n = np.sqrt(np.asarray(B2.multiply(B2).sum(axis=1)).ravel())      # ‖B2[e,:]‖
        per_edge = np.abs(w - 1.0) * b1n * b2n
        total = float(np.sqrt(np.asarray(R.multiply(R).sum())))
        wdev = float(np.abs(w - 1.0).sum())
        return {'total_curvature': total, 'per_vertex': per_vertex,
                'per_face': per_face, 'per_edge': per_edge, 'n_eff': n_eff,
                'curvature_per_weight': (total / wdev) if wdev > 1e-12 else 0.0,
                'weighted': bool(wdev > 1e-12)}

    # The character as moments of f(RL4) (scale-propagator calculus)
    # Eigen-free, O(nnz) or matrix-free polynomial - no per-vertex solve, no
    # eigendecomposition (see rexgraph.scale_propagator; scripts 13-20).

    @cached_property
    def _rl4_sparse(self):
        """RL4 as a sparse scipy CSR (from the sparse path at scale, or the dense
        bundle sparsified). Backing operator for the moment quantities below."""
        import scipy.sparse as _sp
        if self._use_sparse_character:
            from rexgraph.sparse_character import build_sparse_rl
            return build_sparse_rl(self)[0].tocsr()
        RL = self.spectral_bundle.get('RL')
        if RL is None:
            return _sp.csr_matrix((self._nE, self._nE), dtype=_f64)
        return _sp.csr_matrix(np.asarray(RL))

    @cached_property
    def energy_character(self) -> NDArray:
        """Local per-edge energy character diag(RL4²)_e = ‖RL4[e,:]‖² (row-norms,
        O(nnz)) - the short-time moment of the heat propagator (Part C.3). Shape (nE,)."""
        from rexgraph import scale_propagator as _spg
        return _spg.energy_character(self._rl4_sparse)

    @cached_property
    def harmonic_entropy(self) -> float:
        """Harmonic log H₂(RL4) = -log(tr(RL4²)/tr(RL4)²) = eigen-free Rényi-2
        (collision) entropy of RL4's normalized spectrum (Part D.1)."""
        from rexgraph import scale_propagator as _spg
        return _spg.harmonic_entropy(self._rl4_sparse)

    @cached_property
    def character_reliability(self) -> dict:
        """Varentropy reliability flag {H2, H3, shannon_est, gap} - the cheap
        self-diagnostic certifying when the trace-norm (Rényi-2) character suffices;
        ~0 on flat/unweighted spectra, grows when weighted (Part D.4)."""
        from rexgraph import scale_propagator as _spg
        return _spg.reliability_gap(self._rl4_sparse)

    def heat_character(self, t: float, mode: str = 'exact') -> NDArray:
        """Scale-resolved edge-space character diag(e^{-t·RL4}) - the general-f heat
        propagator DIAGONAL. SUPERSEDED: a general matrix-function diagonal has no
        exact O(nnz) form and is blind to inter-grade transport. Prefer
        :meth:`dirac_light` (grade-crossing heat in the Dirac vector space) for the
        propagator, and energy_character (t->0 local star) + harmonic_entropy (t->∞
        global role) for the exact O(nnz) heat moments. Kept as a research accessor
        (mode='exact' dense-exact, 'stochastic' uniform Hutchinson). Shape (nE,).

        Calls the warning-free internal impl directly (the public
        ``_experimental.heat_propagator_diag`` wrapper is deprecation-warned)."""
        import scipy.sparse as _sp
        from rexgraph import _experimental as _exp
        R = self._rl4_sparse
        R = R.tocsr() if _sp.issparse(R) else _sp.csr_matrix(np.asarray(R, dtype=_f64))
        n = R.shape[0]
        if n == 0:
            return np.zeros(0, dtype=_f64)
        lam_max = float(np.asarray(np.abs(R).sum(axis=1)).ravel().max())
        return _exp._chebyshev_diag_impl(lambda P: R @ P, n,
                                         lambda l: np.exp(-float(t) * l),
                                         lam_max, lam_min=0.0, order=48, mode=mode)

    @cached_property
    def greens_diagonal_eigenfree(self) -> NDArray:
        """diag(RL4⁻¹) EXACT via block-CG solves of RL4·X = I to a fixed tolerance -
        one algorithm at every scale, no eigendecomposition, no size-gated
        approximation (Part A / script 11). Shape (nE,).

        RL4 is full-rank SPD so this is a plain inverse diagonal; for a SINGULAR edge
        operator (the edge Laplacian L1, individual channel hats) use
        greens_character_edge / greens_diagonal_singular, which deflate the harmonic
        kernel - a plain solve here would blow up on the null space."""
        from rexgraph import scale_propagator as _spg
        return _spg.greens_diagonal(self._rl4_sparse)

    @cached_property
    def greens_character_edge(self) -> NDArray:
        """diag(L1⁺) - the Green's character of the SINGULAR edge Laplacian L1 =
        B1ᵀB1 + B2B2ᵀ, eigen-free via harmonic-projector deflation L1⁺=(L1+P_H)⁻¹−P_H
        (oracle 09). P_H projects onto the harmonic space ker(L1) via the combinatorial
        harmonic basis (rexgraph.harmonic_sparse.harmonic_basis - fundamental cycles
        flux-projected onto ker(B2ᵀ), no eigendecomposition). This is the per-edge
        self-response through the harmonic-regularized edge propagator. Shape (nE,)."""
        from rexgraph import scale_propagator as _spg
        from rexgraph.harmonic_sparse import harmonic_basis
        from rexgraph.core._laplacians import build_L1_down_sparse, build_L1_up_sparse
        L1 = build_L1_down_sparse(self._B1_dual).tocsr()
        if self.nF_hodge > 0 and self._B2_hodge_dual is not None:
            L1 = (L1 + build_L1_up_sparse(self._B2_hodge_dual)).tocsr()
        return _spg.greens_diagonal_deflated(L1, harmonic_basis(self))

    def greens_diagonal_singular(self, L, H) -> NDArray:
        """diag(L⁺) for an arbitrary SINGULAR symmetric-PSD edge operator L with kernel
        basis H (nE × k), via harmonic-projector deflation. Thin pass-through to
        scale_propagator.greens_diagonal_deflated; H=None (full rank) gives diag(L⁻¹).
        Use e.g. with cycle_basis (ker B1ᵀB1) for the topology channel, or harmonic_basis
        (ker L1) for the edge Laplacian."""
        from rexgraph import scale_propagator as _spg
        return _spg.greens_diagonal_deflated(L, H)

    @cached_property
    def relaxation(self) -> dict:
        """Edge-centric relaxation via the MOMENT tower - the canonical relaxation object
        (canon: relaxation = moments of one propagator on the EDGE operators, not a vertex
        Fiedler value). One discoverable entry point onto quantities that already live in
        the tower, all eigen-free:

          effective_modes : e^{H2} - effective number of RL4 spectral modes (mode count)
          harmonic_log    : H2 = -log(tr(RL4^2)/tr(RL4)^2)  (Renyi-2 collision entropy)
          energy_character: diag(RL4^2) per edge  - the LOCAL short-time heat moment
          greens_edge     : diag(L1^+)  per edge  - the GLOBAL integrated self-response
          varentropy_gap  : H2 - H3, a cheap certificate of when the 2nd-moment summary
                            is trustworthy (~0 = exact)

        The per-channel spectral-GAP metric (lambda_2 / mixing times) is a SEPARATE scope
        - channel_spectral_gaps / per_channel_mixing_times. That is a metric; this is the
        relational relaxation."""
        if not _HAS_RCF or self._nE == 0:
            return {}
        H2 = float(self.harmonic_entropy)
        return {
            'effective_modes': float(np.exp(H2)),
            'harmonic_log': H2,
            'energy_character': self.energy_character,
            'greens_edge': self.greens_character_edge,
            'varentropy_gap': float(self.character_varentropy.get('gap', 0.0)),
        }

    @cached_property
    def _hat_eigen_bundle(self) -> list:
        """Per-hat eigendecompositions. List of (evals, evecs) per hat.

        Computed once via _character.hat_eigen_all and reused by
        per_channel_mixing_times and primal_signal_character.
        """
        if not _HAS_RCF:
            return []
        rcf = self._rcf_bundle
        hats = rcf.get('hats', [])
        nhats = rcf.get('nhats', 0)
        if nhats == 0:
            return []
        return _character.hat_eigen_all(hats, nhats, self._nE)

    @cached_property
    def inverse_centrality_ratio(self) -> NDArray:
        """mu(v) = median(degree) / degree(v) per vertex. Shape (nV,)."""
        deg = self.degree.astype(_f64)
        med = float(np.median(deg[deg > 0])) if np.any(deg > 0) else 1.0
        # np.where would still evaluate med/deg for the zero entries first and warn; a
        # vertex can have degree 0 when the edge list skips its index.
        return np.divide(med, deg, out=np.zeros_like(deg), where=deg > 0)

    @cached_property
    def channel_spectral_gaps(self) -> dict:
        """Exact per-channel spectral gap lambda_2 (smallest positive eigenvalue of each
        trace-normalized hat), a dict keyed by channel name ('L1_down','L_O','L_SG','L_C').

        A METRIC, not the relational relaxation object. T and G use the transpose duality
        (lambda_2 of the tiny nV x nV vertex-dual Laplacian, kernel beta_0) so the
        topological zeros collapse into the small vertex space and the gap is EXACT and
        cheap; C/F use the kernel-robust path. The edge-centric relaxation is the moment
        tower (energy_character / harmonic_entropy / greens_character_edge), separate."""
        if not _HAS_RCF or self._nE == 0:
            return {}
        from rexgraph.sparse_character import channel_spectral_gaps
        return channel_spectral_gaps(self)

    @cached_property
    def per_channel_mixing_times(self) -> NDArray:
        """Per-channel mixing-time METRIC mu_X = ln(nE) / lambda_2(hat_X). Shape (nhats,).
        lambda_2 is the exact channel_spectral_gaps (T/G exact via the transpose duality,
        C/F kernel-robust). This is a spectral-gap summary; the edge-centric relaxation
        is the moment tower, not this."""
        if not _HAS_RCF:
            return np.zeros(0, dtype=_f64)
        if self._use_sparse_character:
            from rexgraph.sparse_character import per_channel_mixing_times_sparse
            return per_channel_mixing_times_sparse(self)
        hat_eigen = self._hat_eigen_bundle
        if len(hat_eigen) == 0:
            return np.zeros(0, dtype=_f64)
        evals_list = [h[0] for h in hat_eigen]
        return _character.per_channel_mixing_times_from_evals(
            evals_list, self.nhats, self._nE)

    @cached_property
    def phi_similarity(self) -> NDArray:
        """Vertex character similarity S_phi[i,j] = 1 - 0.5*||phi_i - phi_j||_1.

        Shape (nV, nV), values in [0, 1]. Measures cross-dimensional
        coherence between vertex pairs.
        """
        if not _HAS_RCF:
            return np.eye(self._nV, dtype=_f64)
        return _fiber.phi_similarity_matrix(
            self.vertex_character, self._nV, self.nhats)

    @cached_property
    def fiber_similarity(self) -> NDArray:
        """Fiber bundle similarity S_fb[i,j] between vertices.

        S_fb[i,j] = max(cos(chi*_i, chi*_j), 0) * phi_sim(i,j).
        Shape (nV, nV), values in [0, 1]. Combines star character
        alignment (fiber cosine) with vertex character agreement.
        """
        if not _HAS_RCF:
            return np.eye(self._nV, dtype=_f64)
        return _fiber.sfb_similarity_matrix(
            self.star_character, self.vertex_character,
            self._nV, self.nhats)

    @cached_property
    def void_complex(self) -> dict:
        """Void spectral theory.

        Keys: Bvoid, Lvoid, n_voids, n_potential, eta, chi_void,
              fills_beta, void_strain
        """
        if not _HAS_RCF:
            return {'n_voids': 0, 'n_potential': 0}
        adj_ptr, adj_idx, adj_edge = self._adjacency_bundle
        # Scale-free path: pass no dense RL/hats (build_void_complex handles None -
        # it is called exactly this way at scale today). Never materialize the dense
        # _rcf_bundle here, which would OOM on large complexes.
        rcf = {} if self._use_sparse_character else self._rcf_bundle
        sb = self.spectral_bundle
        return _void.build_void_complex(
            self.B1, self.B2_hodge,
            adj_ptr, adj_idx, adj_edge,
            self._nV, self._nE,
            rcf.get('RL'), rcf.get('hats'), rcf.get('nhats', 0),
            sb.get('evals_L1'), sb.get('evecs_L1'),
        )

    @cached_property
    def rcfe_curvature(self) -> NDArray:
        """RCFE curvature C(sigma) per edge. Shape (nE,)."""
        if not _HAS_RCF:
            return np.zeros(self._nE, dtype=_f64)
        return _rcfe.compute_curvature(self.B2_hodge, self._nE, self.nF_hodge)

    @cached_property
    def rcfe_strain(self) -> float:
        """Total RCFE strain S = sum C(e) * RL[e,e]."""
        if not _HAS_RCF:
            return 0.0
        # RL[e,e] from the sparse RL4 diagonal - avoids materializing the dense RL
        # (which OOMs / is absent on the scale-free sparse path).
        rl_diag = np.ascontiguousarray(self._rl4_sparse.diagonal())
        return float(_rcfe.compute_strain(self.rcfe_curvature, rl_diag, self._nE))

    def attributed_curvature(self, w_e: NDArray = None,
                              a_v: NDArray = None) -> dict:
        """Attributed boundary curvature (Def 3.1-3.2): the residual R = B1^w @ B2^w and per-face
        kappa_f = ||R[:,f]||, where B1^w[v,e] = a_v * B1[v,e] * sqrt(w_e) and B2^w = sqrt(w_e) * B2.

        Each face boundary has only a few edges, so we contract B2 in its sparse (CSC) form -
        R[:,f] is the signed sum of the boundary edges' B1^w columns - instead of building the dense
        nE x nF matrix and a dense matmul. Identical to the dense kernel to 1e-16, and O(nV*nF + nnz)
        memory rather than O(nE*nF), so it scales to complexes where dense B2 would not fit."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        from scipy import sparse as _sp
        nF = self.nF_hodge
        if w_e is None:
            w_e = np.ones(self._nE, dtype=_f64)
        if a_v is None:
            a_v = np.ones(self._nV, dtype=_f64)
        w_e = np.ascontiguousarray(w_e, dtype=_f64)
        a_v = np.ascontiguousarray(a_v, dtype=_f64)
        sqw = np.sqrt(np.maximum(w_e, 0.0))
        B1w = (a_v[:, None] * self.B1) * sqw[None, :]        # nV x nE
        d = self._B2_hodge_dual
        if nF == 0 or d is None:
            return {'kappa_f': np.zeros(0, dtype=_f64), 'R': np.zeros((self._nV, 0), dtype=_f64),
                    'B1w': B1w, 'B2w': _sp.csc_matrix((self._nE, 0), dtype=_f64)}
        cp = np.asarray(d.col_ptr); ri = np.asarray(d.row_idx); vl = np.asarray(d.vals_csc)
        b2w = vl * sqw[ri]                                   # sqrt(w_e)-scaled boundary signs
        lengths = np.diff(cp)
        if lengths.size and np.all(lengths == 3):           # every face a triangle: fully vectorized
            E = ri.reshape(nF, 3); S = b2w.reshape(nF, 3)   # reshape valid only when all len == 3
            R = (B1w[:, E[:, 0]] * S[:, 0] + B1w[:, E[:, 1]] * S[:, 1] + B1w[:, E[:, 2]] * S[:, 2])
        else:                                               # general / mixed boundary length
            R = np.zeros((self._nV, nF), dtype=_f64)
            for f in range(nF):
                for k in range(cp[f], cp[f + 1]):
                    R[:, f] += B1w[:, ri[k]] * b2w[k]
        R = np.ascontiguousarray(R, dtype=_f64)
        kappa_f = np.ascontiguousarray(np.sqrt((R * R).sum(0)), dtype=_f64)
        B2w = _sp.csc_matrix((b2w, ri, cp), shape=(self._nE, nF))
        return {'kappa_f': kappa_f, 'R': R, 'B1w': B1w, 'B2w': B2w}

    def strain_equilibrium(self, born_face: NDArray = None,
                            t: float = 0.0,
                            vertex_idx: int = 0) -> dict:
        """Full dynamic strain equilibrium analysis."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        nF = self.nF_hodge
        if nF == 0:
            return {
                'alpha': 0.0, 'delta': np.zeros(0, dtype=_f64),
                'sigma': np.zeros(self._nE, dtype=_f64),
                'bianchi_ok': True, 'bianchi_residual': 0.0,
                'strain_norm': 0.0, 'kappa_f': np.zeros(0, dtype=_f64),
            }
        ac = self.attributed_curvature()
        kappa_f = ac['kappa_f']
        if born_face is None:
            if _dirac is not None:
                psi_re, psi_im = self.graded_state(t=t, vertex_idx=vertex_idx)
                per_cell, _ = self.born_graded(psi_re, psi_im)
                born_face = per_cell[self._nV + self._nE:]
            else:
                born_face = np.ones(nF, dtype=_f64) / nF
        born_face = np.ascontiguousarray(born_face, dtype=_f64)
        result = _rcfe.strain_equilibrium(
            self.B1, self.B2_hodge, kappa_f, born_face,
            self._nV, self._nE, nF)
        result['kappa_f'] = kappa_f
        return result

    # RCF methods

    def impute(self, observed_signal: NDArray, observed_mask: NDArray) -> dict:
        """Impute missing signal values via harmonic interpolation through RL."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        return _query.signal_impute(
            self.RL,
            np.asarray(observed_signal, dtype=_f64),
            np.asarray(observed_mask, dtype=_u8),
            self._nE,
        )

    def propagate_signal(self, vertex_signal: NDArray, mode: str = "heat",
                         t: float = 1.0, order: int = 40,
                         tol: float = 1e-8) -> NDArray:
        """Dynamic vertex propagator - the vertex propagator S₀ = B₁ RL⁺ B₁ᵀ applied
        to a SIGNAL by diffusion, not by enumerating vertices. A vertex signal s is
        lifted to its incident edge boundaries through the coboundary B₁ᵀ, diffused
        through the SPARSE relational operator RL4, and projected back to vertices
        through B₁ - a single diffusion of O(nnz·iters), localized to the edges the
        signal actually reaches. This is the demand-driven read of the propagator: a
        signal flows through the relevant edge boundaries to the relevant vertices,
        rather than the static O(nV·solve) per-vertex Green's enumeration.
        Scale-free (sparse RL, no per-vertex loop). Modes:
          'heat'   : e^{-t·RL4} - diffusive/local (small t -> tight star, large t ->
                     the global role; the script-15 scale bridge, one signal at a time)
          'greens' : RL4⁻¹ - the equilibrium/global Green's response, CG to `tol`.
        Returns the propagated vertex response (nV,), nonzero on the reached vertices.
        propagate_signal(e_v, 'greens') is column v of S₀ = B₁ RL⁺ B₁ᵀ (so the single
        vertex's global character is ONE diffusion, not nV)."""
        from rexgraph.core._sparse import to_scipy_csr
        s = np.asarray(vertex_signal, dtype=_f64).ravel()
        if s.size != self._nV:
            raise ValueError(f"vertex_signal must have length nV={self._nV}")
        if self._B1_dual is None or self._nE == 0:
            return np.zeros(self._nV, dtype=_f64)
        B1 = to_scipy_csr(self._B1_dual).tocsr()          # nV × nE
        RL = self._rl4_sparse                              # nE × nE sparse SPD
        b = np.asarray(B1.T @ s).ravel()                   # lift to incident edges
        if not np.any(np.abs(b) > 0):
            return np.zeros(self._nV, dtype=_f64)
        if mode == "greens":
            from rexgraph.sparse_character import _block_cg
            rl_diag = RL.diagonal()
            dinv = np.where(np.abs(rl_diag) > 1e-30, 1.0 / rl_diag, 1.0)
            x = _block_cg(lambda P: RL @ P, b[:, None], dinv, tol=tol)[:, 0]
        else:                                              # heat: e^{-t·RL} b, matvecs
            from rexgraph._experimental import _cheb_apply
            lam_max = (float(np.asarray(np.abs(RL).sum(axis=1)).ravel().max())
                       if RL.shape[0] else 1.0)
            x = _cheb_apply(lambda P: RL @ P, lambda l: np.exp(-t * l),
                            lam_max, 0.0, order, b[:, None])[:, 0]
        return np.asarray(B1 @ x).ravel()                  # project back to vertices

    def character_response(self, seed_vertices: NDArray,
                           tol: float = 1e-10) -> NDArray:
        """Demand-driven per-vertex character φ(v,k) at just a SEED set of vertices,
        by diffusion - the dynamic read of the character at the query vertices,
        O(|seed|·nhats·diffusion) instead of the static O(nV·solve) full enumeration.
        Each seed's incident-edge boundary b_v = B₁[v,:]ᵀ is diffused once through the
        sparse RL4 (block-CG, all seeds at once), then
          φ(v,k) = [b_v^T RL⁻¹ ĥ_k RL⁻¹ b_v] / [b_v^T RL⁻¹ b_v]
        is read at the seeds - identical to ``vertex_character[seed_vertices]`` but
        computed only where asked (you propagate a query, you don't enumerate the
        graph). Returns (len(seed), nhats)."""
        from rexgraph.core._sparse import to_scipy_csr
        from rexgraph.sparse_character import _block_cg
        seeds = np.asarray(seed_vertices, dtype=int).ravel()
        nhats = int(self.nhats)
        uniform = 1.0 / nhats if nhats > 0 else 0.0
        out = np.full((seeds.size, nhats), uniform, dtype=_f64)
        if nhats == 0 or self._nE == 0 or seeds.size == 0:
            return out
        cheap = self._sparse_character if self._use_sparse_character else None
        if cheap is None:                       # dense/small path: exact enumeration
            vc = np.asarray(self.vertex_character, dtype=_f64)
            return vc[seeds]
        RL = cheap['RL'].tocsr()
        hats, names, rl_diag = cheap['hats'], cheap['hat_names'], cheap['rl_diag']
        hat_by_name = dict(zip(names, hats))
        B1 = to_scipy_csr(self._B1_dual).tocsr()            # nV × nE
        Bc = np.ascontiguousarray(B1[seeds].toarray().T)    # nE × |seed| = b_v columns
        dinv = np.where(np.abs(rl_diag) > 1e-30, 1.0 / rl_diag, 1.0)
        X = _block_cg(lambda P: RL @ P, Bc, dinv, tol=tol)  # RL⁻¹ b_v (one block solve)
        s0 = np.einsum('ev,ev->v', Bc, X)                   # b_v · x_v (denominator)
        ok = np.abs(s0) > 1e-15
        denom = np.where(ok, s0, 1.0)
        for k, name in enumerate(names):
            num = np.einsum('ev,ev->v', X, hat_by_name[name] @ X)
            out[:, k] = np.where(ok, num / denom, uniform)
        return out

    def coherence_response(self, seed_vertices: NDArray) -> NDArray:
        """Demand-driven coherence κ at a SEED set of vertices, by diffusion - the
        dynamic read of coherence at just the query vertices,
        O(|seed|·nhats·diffusion) not O(nV·solve). κ(v)=1-0.5‖φ(v)-χ*(v)‖₁ with φ from
        ``character_response`` (Green's diffusion at the seeds) and χ* the star-average
        character (O(nnz)). Identical to ``coherence[seed_vertices]``, computed only
        where asked. Returns (len(seed),)."""
        seeds = np.asarray(seed_vertices, dtype=int).ravel()
        if seeds.size == 0:
            return np.zeros(0, dtype=_f64)
        phi = self.character_response(seeds)                          # diffusion
        chistar = np.asarray(self.star_character, dtype=_f64)[seeds]  # O(nnz)
        return 1.0 - 0.5 * np.abs(phi - chistar).sum(axis=1)

    def local_context(self, seed_vertices: NDArray, t: float = 1.0,
                      threshold: float = 1e-6,
                      max_vertices: Optional[int] = None) -> dict:
        """Bounded local context around query vertices, by heat diffusion - the star
        neighborhood and its reach, for CONTEXT and BOUNDARY ISOLATION without ever
        enumerating the whole graph. A seed indicator is diffused through e^{-t·RL4}
        (small t -> tight star, larger t -> wider role; the script-15 scale bridge), and
        the reached vertices (|response| > threshold) form the relevant sub-complex.
        Returns {seeds, reached, weights, n_reached, character} - the demand-driven
        'relevant subgraph' primitive the agent layer acts on: propagate from the
        query, work the bounded neighborhood, never pay for the combinatorial whole."""
        seeds = np.asarray(seed_vertices, dtype=int).ravel()
        if seeds.size == 0 or self._nV == 0:
            return {'seeds': seeds.tolist(), 'reached': [], 'weights': [],
                    'n_reached': 0, 'character': np.zeros((0, self.nhats))}
        seed_ind = np.zeros(self._nV, dtype=_f64)
        seed_ind[seeds[(seeds >= 0) & (seeds < self._nV)]] = 1.0
        resp = self.propagate_signal(seed_ind, mode='heat', t=t)
        reached = np.where(np.abs(resp) > threshold)[0]
        if max_vertices is not None and reached.size > int(max_vertices):
            reached = reached[np.argsort(-np.abs(resp[reached]))[:int(max_vertices)]]
        return {
            'seeds': seeds.tolist(),
            'reached': reached.tolist(),
            'weights': resp[reached].round(6).tolist(),
            'n_reached': int(reached.size),
            'character': self.character_response(reached),   # φ at the reached vertices
        }

    def _explain_vertex_dynamic(self, idx: int) -> dict:
        """Single-vertex diagnostic by DEMAND-DRIVEN diffusion - φ, κ, χ*, the channel
        discrepancy, degree, incident edges and neighbor vertices - all from the query
        vertex via one diffusion + sparse local reads, no full per-vertex enumeration
        and no dense B1/RL. Matches the _query.explain_vertex return contract."""
        from rexgraph.core._sparse import to_scipy_csr
        idx = int(idx)
        nhats = int(self.nhats)
        phi_v = np.asarray(self.character_response([idx])[0], dtype=_f64)   # diffusion
        chi_star_v = (np.asarray(self.star_character, dtype=_f64)[idx]
                      if nhats else np.zeros(0, dtype=_f64))
        gaps = np.abs(phi_v - chi_star_v) if nhats else np.zeros(0, dtype=_f64)
        kappa_v = float(1.0 - 0.5 * gaps.sum()) if nhats else 0.0
        disc_ch = int(np.argmax(gaps)) if nhats else 0
        dom_ch = int(np.argmax(phi_v)) if nhats else 0
        v2e_ptr, v2e_idx = self._v2e
        v2e_ptr = np.asarray(v2e_ptr); v2e_idx = np.asarray(v2e_idx)
        lo, hi = int(v2e_ptr[idx]), int(v2e_ptr[idx + 1])
        incident = [int(v2e_idx[j]) for j in range(lo, hi)]
        # neighbors from the sparse B1 columns (O(deg)), not a dense O(nV) scan
        neighbors = set()
        if self._B1_dual is not None and incident:
            B1c = to_scipy_csr(self._B1_dual).tocsc()
            for e in incident:
                for v in B1c.indices[B1c.indptr[e]:B1c.indptr[e + 1]]:
                    if int(v) != idx:
                        neighbors.add(int(v))
        return {
            'phi': phi_v, 'chi_star': chi_star_v, 'kappa': kappa_v,
            'discrepant_channel': disc_ch,
            'channel_gap': float(gaps[disc_ch]) if nhats else 0.0,
            'dominant_channel': dom_ch, 'degree': hi - lo,
            'incident_edges': np.array(incident, dtype=np.int32),
            'neighbor_vertices': np.array(sorted(neighbors), dtype=np.int32),
        }

    def _explain_edge_dynamic(self, idx: int) -> dict:
        """Single-edge (relation) diagnostic - its place in the Hodge tower plus its
        criticality - all from sparse local reads and ONE diffusion, no dense B1/B2/K1
        scans and no eigendecomposition. Matches the _query.explain_edge contract:
          below   = boundary vertices (endpoints)            - sparse B1 column, ∂/down
          above   = co-boundary faces containing the edge    - sparse B2 row, δ/up
          lateral = sibling edges sharing an endpoint (K1)   - endpoint stars (_v2e)
          chi     = channel character [T,G,F,C]              - diagonals, O(nnz)
          effective_resistance = RL4⁺[e,e] = (RL4⁻¹ e_e)[e]  - ONE demand-driven
                    diffusion (the edge's Green's self-response: high = bridge/critical,
                    low = redundant). The dense kernel used a full pinv and returned
                    NaN above its eigen limit; this is exact at every scale."""
        from rexgraph.core._sparse import to_scipy_csr
        idx = int(idx)
        nhats = int(self.nhats)
        # below: the two boundary vertices, from the sparse B1 column
        below = []
        if self._B1_dual is not None:
            B1c = to_scipy_csr(self._B1_dual).tocsc()
            below = sorted(int(v) for v in
                           B1c.indices[B1c.indptr[idx]:B1c.indptr[idx + 1]])
        # above: co-boundary faces, from the sparse B2 row
        above = []
        if self.nF_hodge > 0 and self._B2_hodge_dual is not None:
            B2 = to_scipy_csr(self._B2_hodge_dual).tocsr()      # nE × nF
            above = sorted(int(f) for f in B2.getrow(idx).indices)
        # lateral: edges sharing an endpoint (co-incident), from the endpoints' stars
        v2e_ptr, v2e_idx = self._v2e
        v2e_ptr = np.asarray(v2e_ptr); v2e_idx = np.asarray(v2e_idx)
        lateral = set()
        for v in below:
            for j in range(int(v2e_ptr[v]), int(v2e_ptr[v + 1])):
                e = int(v2e_idx[j])
                if e != idx:
                    lateral.add(e)
        # chi: channel character (diagonals)
        chi = (np.asarray(self.structural_character, dtype=_f64)[idx]
               if nhats else np.zeros(0, dtype=_f64))
        dominant = int(np.argmax(chi)) if nhats else 0
        # effective_resistance = the CLASSIC bridge measure b_eᵀ L0⁺ b_e (one L0 solve;
        # ->1 = bridge/critical, <1 = redundant). relational_self_response = RL4⁺[e,e]
        # (the relation's Green's self-energy in the full [T,G,F,C] operator - the value
        # the old dense kernel returned as "effective_resistance", kept but correctly
        # named, and now exact at scale where that kernel NaN'd).
        r_eff = self.effective_resistance(idx) if self._nE > 0 else float('nan')
        r_self = float('nan')
        if self._nE > 0:
            from rexgraph.sparse_character import _block_cg
            RL = self._rl4_sparse
            ee = np.zeros(self._nE, dtype=_f64); ee[idx] = 1.0
            rl_diag = RL.diagonal()
            dinv = np.where(np.abs(rl_diag) > 1e-30, 1.0 / rl_diag, 1.0)
            x = _block_cg(lambda P: RL @ P, ee[:, None], dinv, tol=1e-10)[:, 0]
            r_self = float(x[idx])
        return {
            'below': np.array(below, dtype=np.int32),
            'above': np.array(above, dtype=np.int32),
            'lateral': np.array(sorted(lateral), dtype=np.int32),
            'chi': chi, 'dominant_channel': dominant,
            'effective_resistance': r_eff,
            'relational_self_response': r_self,
            'n_incident_faces': len(above), 'degree': len(below),
        }

    def explain_context(self, vertices: NDArray = None, edges: NDArray = None,
                        t: float = 1.0, threshold: float = 1e-6,
                        max_cells: Optional[int] = None) -> dict:
        """The forged contextual picture around query ENTITIES (vertices) and RELATIONS
        (edges) - the unified view the LLM reads. Per-seed diagnostics (explain_vertex
        for entities, explain_edge for relations) PLUS the bounded relevant sub-complex
        reached by ONE heat diffusion seeded across both grades: a vertex seed injects
        its incident edge boundaries (B₁ᵀ e_v), an edge seed injects itself (e_e), the
        combined edge signal diffuses through RL4 (e^{-t·RL4}), and the reached edges
        (relations) and their projected vertices (entities) form the isolated
        neighborhood. One diffusion, localized to the signal's reach - no whole-graph
        enumeration, scale-free. Seed either grade or both; small t -> tight, larger t ->
        wider. Returns {seed_vertices, seed_edges, neighborhood:{vertices,
        vertex_weights, vertex_coherence, edges, edge_weights, edge_character}}."""
        from rexgraph.core._sparse import to_scipy_csr
        vseeds = np.asarray(vertices if vertices is not None else [], dtype=int).ravel()
        eseeds = np.asarray(edges if edges is not None else [], dtype=int).ravel()
        vseeds = vseeds[(vseeds >= 0) & (vseeds < self._nV)]
        eseeds = eseeds[(eseeds >= 0) & (eseeds < self._nE)]
        B1 = to_scipy_csr(self._B1_dual).tocsr() if self._B1_dual is not None else None
        # combined edge signal: vertex seeds via the coboundary, edge seeds directly
        b = np.zeros(self._nE, dtype=_f64)
        if vseeds.size and B1 is not None:
            vind = np.zeros(self._nV, dtype=_f64); vind[vseeds] = 1.0
            b += np.asarray(B1.T @ vind).ravel()
        if eseeds.size:
            b[eseeds] += 1.0
        # diffuse on the edge operator, then project the response back to vertices
        edge_resp = np.zeros(self._nE, dtype=_f64)
        if np.any(np.abs(b) > 0) and self._nE > 0:
            from rexgraph._experimental import _cheb_apply
            RL = self._rl4_sparse
            lam_max = float(np.asarray(np.abs(RL).sum(axis=1)).ravel().max())
            edge_resp = _cheb_apply(lambda P: RL @ P, lambda l: np.exp(-t * l),
                                    lam_max, 0.0, 40, b[:, None])[:, 0]
        vertex_resp = (np.asarray(B1 @ edge_resp).ravel() if B1 is not None
                       else np.zeros(self._nV, dtype=_f64))
        reached_e = np.where(np.abs(edge_resp) > threshold)[0]
        reached_v = np.where(np.abs(vertex_resp) > threshold)[0]
        if max_cells is not None:
            mc = int(max_cells)
            if reached_e.size > mc:
                reached_e = reached_e[np.argsort(-np.abs(edge_resp[reached_e]))[:mc]]
            if reached_v.size > mc:
                reached_v = reached_v[np.argsort(-np.abs(vertex_resp[reached_v]))[:mc]]
        chi_all = np.asarray(self.structural_character, dtype=_f64) if self.nhats else None
        return {
            'seed_vertices': [self._explain_vertex_dynamic(int(v)) for v in vseeds],
            'seed_edges': [self._explain_edge_dynamic(int(e)) for e in eseeds],
            'neighborhood': {
                'vertices': reached_v.tolist(),
                'vertex_weights': vertex_resp[reached_v].round(6).tolist(),
                'vertex_coherence': (np.asarray(self.coherence_response(reached_v)).round(4).tolist()
                                     if reached_v.size else []),
                'edges': reached_e.tolist(),
                'edge_weights': edge_resp[reached_e].round(6).tolist(),
                'edge_character': (chi_all[reached_e].round(4).tolist()
                                   if chi_all is not None and reached_e.size else []),
            },
        }

    def _effective_resistance_batch(self, edge_indices: NDArray) -> NDArray:
        """Classic effective resistance R_eff(e) = b_eᵀ L0⁺ b_e for a BATCH of edges
        (b_e = B1[:,e] = the edge's signed endpoints, L0 = the graph Laplacian) in ONE
        block-CG solve - the bridge measure: R_eff -> 1 for a true BRIDGE (removal
        disconnects), < 1 for a REDUNDANT edge in a cycle. L0 is singular but every b_e
        lies in its range (∈ im(B1)=im(L0)), so preconditioned CG from 0 converges to
        the pseudoinverse solution. Demand-driven, scale-free. Aligned to edge_indices."""
        edges = np.asarray(edge_indices, dtype=int).ravel()
        if edges.size == 0 or self._nE == 0:
            return np.zeros(edges.size, dtype=_f64)
        from rexgraph.core._sparse import to_scipy_csr
        from rexgraph import scale_propagator as _spg
        B1 = to_scipy_csr(self._B1_dual).tocsc()            # nV × nE
        Bc = np.ascontiguousarray(np.asarray(B1[:, edges].todense()))   # nV × |edges|
        L0 = self.L0_sparse.tocsr()
        d = L0.diagonal()
        dinv = np.where(d > 1e-30, 1.0 / d, 1.0)
        # block-CG L0⁺ b_e - CPU or GPU-resident (auto, size-gated) via block_cg_solve
        X = _spg.block_cg_solve(L0, Bc, dinv, tol=1e-10)
        return np.einsum('ve,ve->e', Bc, X)

    def effective_resistance(self, edge_idx: int) -> float:
        """Classic effective resistance R_eff(e) = b_eᵀ L0⁺ b_e for one edge (relation)
        via a single L0 solve - the load-bearing measure: R_eff -> 1 = a BRIDGE
        (critical/near-unique link, removal fragments the graph), lower = REDUNDANT
        (many parallel paths)."""
        return float(self._effective_resistance_batch(np.asarray([int(edge_idx)]))[0])

    def agentic_reading(self, vertices: NDArray = None, edges: NDArray = None,
                        t: float = 1.0, max_cells: Optional[int] = None,
                        top_k: int = 8) -> dict:
        """The decision-ready agentic reading over a query's ENTITIES and RELATIONS -
        the keystone the agent/LLM layer consumes. One forged diffusion
        (``explain_context``) reduced to what a turn needs:
          neighborhood : the bounded relevant sub-complex (entities + relations).
          load_bearing : relations ranked by effective_resistance - the BRIDGES
                         (high = critical/near-unique, removal fragments), top_k.
          frustrated   : entities whose coherence is a LOW outlier (data-adaptive lower
                         Tukey fence on the neighborhood's κ) - the incoherent/frustrated.
          context_size : |reached vertices| + |reached edges| - the bounded relevant
                         size, i.e. the real token/cost driver (how much context a
                         correct answer needs).
        All demand-driven and bounded - no whole-graph enumeration."""
        ctx = self.explain_context(vertices=vertices, edges=edges, t=t,
                                   max_cells=max_cells)
        nb = ctx['neighborhood']
        r_edges = np.asarray(nb['edges'], dtype=int)
        r_verts = np.asarray(nb['vertices'], dtype=int)
        kappa = np.asarray(nb['vertex_coherence'], dtype=_f64)
        # load-bearing: effective resistance at the reached relations (one block solve)
        load_bearing = []
        if r_edges.size:
            eff = self._effective_resistance_batch(r_edges)
            order = np.argsort(-eff)[:int(top_k)]
            load_bearing = [{'edge': int(r_edges[i]),
                             'effective_resistance': round(float(eff[i]), 4)}
                            for i in order]
        # frustrated: low-κ outliers via a data-adaptive lower Tukey fence
        frustrated = []
        if kappa.size >= 4:
            q1, q3 = np.percentile(kappa, [25.0, 75.0])
            fence = q1 - 1.5 * (q3 - q1)
            frustrated = [{'vertex': int(r_verts[i]), 'coherence': round(float(kappa[i]), 4)}
                          for i in range(kappa.size) if kappa[i] < fence]
        return {
            'seed_vertices': ctx['seed_vertices'],
            'seed_edges': ctx['seed_edges'],
            'neighborhood': nb,
            'load_bearing': load_bearing,
            'frustrated': frustrated,
            'context_size': int(r_verts.size + r_edges.size),
        }

    def explain(self, dim: int, idx: int) -> dict:
        """Full diagnostic for a cell (edge or vertex)."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        if dim == 1:
            # Demand-driven: below/above/lateral from the sparse boundary/coboundary/
            # endpoint-stars, χ from diagonals, and effective_resistance = RL4⁺[e,e]
            # from ONE diffusion - no dense B1/B2/K1 scans and no eigendecomposition
            # (the dense kernel returned NaN for effective_resistance at scale).
            return self._explain_edge_dynamic(idx)
        elif dim == 0:
            # Demand-driven: diffuse from the single query vertex (φ via
            # character_response, κ/χ* local reads, neighbors from the sparse B1) -
            # no full nV vertex_character/coherence enumeration and no dense B1/RL.
            return self._explain_vertex_dynamic(idx)
        else:
            raise ValueError(f"explain not supported for dim={dim}")

    def propagate(self, source: NDArray, target: NDArray) -> dict:
        """Spectral propagation score through RL."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        if self._use_sparse_character:
            # eigen-free: RL4⁻¹ source via block-CG + sparse hat matvecs, no rl_eigen.
            from rexgraph.sparse_character import spectral_propagate_sparse
            return spectral_propagate_sparse(
                self, np.asarray(source, dtype=_f64),
                np.asarray(target, dtype=_f64))
        rcf = self._rcf_bundle
        return _query.spectral_propagate(
            self.RL, rcf['hats'], rcf['nhats'],
            np.asarray(source, dtype=_f64),
            np.asarray(target, dtype=_f64),
            self._nE,
        )

    def inner_join(self, other: 'RexGraph', shared_vertices: NDArray) -> dict:
        """Inner join (intersection) with another RexGraph."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        return _joins.inner_join(
            self.B1, self.B2, self._nV, self._nE, self._nF,
            other.B1, other.B2, other._nV, other._nE, other._nF,
            np.asarray(shared_vertices, dtype=_i32),
        )

    def left_join(self, other: 'RexGraph', shared_vertices: NDArray) -> dict:
        """Left join: keep all of self, bring in other's shared edges."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        return _joins.left_join(
            self.B1, self.B2, self._nV, self._nE, self._nF,
            other.B1, other.B2, other._nV, other._nE, other._nF,
            np.asarray(shared_vertices, dtype=_i32),
        )

    def outer_join(self, other: 'RexGraph', shared_vertices: NDArray) -> dict:
        """Outer join (pushout) over shared vertices."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        return _joins.outer_join(
            self.B1, self.B2, self._nV, self._nE, self._nF,
            other.B1, other.B2, other._nV, other._nE, other._nF,
            np.asarray(shared_vertices, dtype=_i32),
        )

    def structural_summary(self) -> dict:
        """Aggregate structural statistics."""
        if not _HAS_RCF:
            return {}
        return _character.structural_summary(
            self.structural_character, self.vertex_character,
            self.coherence, self._nE, self._nV,
            self.nhats,
        )

    # Interfacing vector

    def interfacing_vector(
        self,
        target_indices: NDArray,
        target_weights: NDArray,
        target_signal: NDArray,
        *,
        vertex_weights: Optional[NDArray] = None,
    ) -> dict:
        """Full interfacing vector analysis for a source entity.

        Parameters
        ----------
        target_indices : i32 array
            Vertex indices of source targets.
        target_weights : f64 array
            Per-target weights.
        target_signal : f64[nE], or None
            Target/phenotype edge vector. None means score the induced flow psi
            against itself (the self-interfacing reading), which costs one L0^+
            solve rather than the two a caller needs to obtain psi first.
        vertex_weights : f64[nV], optional
            Per-vertex weights. Defaults to IDF: 1 / ln(degree + e).

        Returns
        -------
        dict with rho, psi, scores, schrodinger, iv, sphere_pos,
        signal_magnitude, coverage, efficiency, confidence.
        """
        if vertex_weights is None:
            deg = self.degree.astype(_f64)
            vertex_weights = 1.0 / np.log(deg + np.e)
        # Prefer the eigen-free sparse bundle at scale. The dense _interfacing
        # materializes S_T = B1^T L0^+ B1 (nE x nE) plus a dense L0^+ / RL
        # eigendecomposition, which OOMs above eigen_dense_limit. The sparse path
        # computes the SAME bundle matrix-free (LSQR L0^+ bilinear + sparse channel
        # matvecs; genuinely-spectral schrodinger/coverage via a bounded eigsh),
        # gated on the same condition as the rest of the scale-free character stack
        # (_use_sparse_character == "dense RL was not built"). Verified against the
        # dense oracle on small graphs in tests/test_interfacing_sparse.py.
        if _HAS_RCF and self._use_sparse_character:
            from rexgraph.sparse_interfacing import build_interfacing_bundle_sparse
            return build_interfacing_bundle_sparse(
                self, target_indices, target_weights, target_signal,
                vertex_weights=vertex_weights)
        if _interfacing is None:
            raise RuntimeError("_interfacing module not available.")
        if target_signal is None:
            # the dense kernel has no self-target mode, so resolve psi with a
            # throwaway pass and feed it back. Only the legacy dense path pays this.
            psi = self.interfacing_vector(
                target_indices, target_weights,
                np.zeros(self._nE, dtype=_f64),
                vertex_weights=vertex_weights)["psi"]
            return self.interfacing_vector(
                target_indices, target_weights,
                np.ascontiguousarray(psi, dtype=_f64),
                vertex_weights=vertex_weights)
        sb = self.spectral_bundle
        evals_rl, evecs_rl = self._rl_eigen
        return _interfacing.build_interfacing_bundle(
            _asarray(target_indices, _i32),
            np.ascontiguousarray(target_weights, dtype=_f64),
            np.ascontiguousarray(vertex_weights, dtype=_f64),
            self.B1,
            sb['evals_L0'],
            np.ascontiguousarray(sb['evecs_L0'], dtype=_f64),
            self.L_overlap,
            self.L_frustration,
            evals_rl, evecs_rl,
            np.ascontiguousarray(target_signal, dtype=_f64),
            self._nV, self._nE,
        )

    # Primal signal character

    def primal_signal_character(self, psi: NDArray) -> NDArray:
        """Energy decomposition of an edge signal across typed channels.

        E_X = psi^T hat_X^+ psi per channel, normalized to sum to 1.

        Parameters
        ----------
        psi : f64[nE]

        Returns
        -------
        f64[nhats]
        """
        # Eigen-free: E_X = psiᵀ hat_X⁺ psi via LSQR pseudoinverse quadratic forms on
        # the sparse channel hats (== the dense _channels.primal_signal_character to
        # ~1e-9, no per-channel eigendecomposition / hat_eigen bundle).
        from rexgraph.sparse_character import primal_signal_character_sparse
        return primal_signal_character_sparse(self, psi)

    # Spectral channel score

    def spectral_channel_score(self, source: NDArray, target: NDArray) -> float:
        """Spectral propagation score: source through RL eigenmodes onto target.

        Parameters
        ----------
        source : f64[nE]
        target : f64[nE]

        Returns
        -------
        float
        """
        # Eigen-free: sourceᵀ RL4⁺ target is one block-CG solve (RL4 is full-rank SPD,
        # so RL4⁺=RL4⁻¹) == the dense eigenmode sum to ~1e-9, no eigendecomposition.
        from rexgraph.sparse_character import spectral_channel_score_sparse
        return spectral_channel_score_sparse(self, source, target)

    # Face-void dipole

    def face_void_dipole(self, psi: NDArray) -> dict:
        """Face-void dipole of an edge signal.

        Projects psi onto the realized face basis (B2) and the void
        basis (Bvoid), returning face_affinity, void_affinity, and
        dipole_ratio in [-1, 1].

        Parameters
        ----------
        psi : f64[nE]

        Returns
        -------
        dict
        """
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        vc = self.void_complex
        Bvoid = vc.get('Bvoid')
        if hasattr(Bvoid, "toarray"):           # void_complex hands back a sparse Bvoid; the kernel
            Bvoid = np.ascontiguousarray(Bvoid.toarray(), dtype=_f64)  # needs a dense array (else ValueError)
        return _character.face_void_dipole(     # Bvoid stays None when there are no voids (kernel handles it)
            np.ascontiguousarray(psi, dtype=_f64),
            self.B2_hodge,
            Bvoid,
            self._nE, self.nF_hodge,
        )

    # Context face selection

    def context_face_selection(self, context_matrix: NDArray) -> 'RexGraph':
        """Build a new RexGraph with faces selected by context matrix.

        E = C^T |B1| > 0. Triangle included iff some context covers
        all three boundary edges.

        Parameters
        ----------
        context_matrix : uint8[n_contexts, nV]

        Returns
        -------
        RexGraph with selected faces. Also stores per_context_face_count
        and per_context_void_fraction as attributes.
        """
        adj_ptr, adj_idx, adj_edge = self._adjacency_bundle
        result = _faces.context_face_selection(
            self.B1,
            np.ascontiguousarray(context_matrix, dtype=_u8),
            adj_ptr, adj_idx, adj_edge,
            self._nV, self._nE,
        )
        nF = result['nF']
        if nF == 0:
            rex = RexGraph(
                boundary_ptr=self._boundary_ptr.copy(),
                boundary_idx=self._boundary_idx.copy(),
                w_E=self._w_E, directed=self._directed, signs=self._signs,
            )
        else:
            from rexgraph.core._boundary import build_B2_from_cycles
            B2_dual = build_B2_from_cycles(
                self._nE, result['cycle_edges'],
                result['cycle_signs'], result['cycle_lengths'])
            B2_dense = _sparse.to_dense_f64(B2_dual)
            from scipy import sparse as sp
            B2_sp = sp.csc_matrix(B2_dense)
            rex = RexGraph(
                boundary_ptr=self._boundary_ptr.copy(),
                boundary_idx=self._boundary_idx.copy(),
                B2_col_ptr=np.asarray(B2_sp.indptr, dtype=_i32),
                B2_row_idx=np.asarray(B2_sp.indices, dtype=_i32),
                B2_vals=np.asarray(B2_sp.data, dtype=_f64),
                w_E=self._w_E, directed=self._directed, signs=self._signs,
            )
        rex._context_face_result = result
        return rex

    # Typed face selection

    def typed_face_selection(self, edge_type_labels: NDArray) -> 'RexGraph':
        """Build a new RexGraph with faces from same-type triangles.

        A triangle is a face iff all three boundary edges share the
        same type label. Cross-type triangles become voids.

        Parameters
        ----------
        edge_type_labels : i32[nE]

        Returns
        -------
        RexGraph with realized faces. Also stores typed_face_result
        with void data as an attribute.
        """
        adj_ptr, adj_idx, adj_edge = self._adjacency_bundle
        n_types = int(np.max(edge_type_labels)) + 1
        result = _faces.typed_face_selection(
            _asarray(edge_type_labels, _i32),
            adj_ptr, adj_idx, adj_edge,
            self._nV, self._nE, n_types,
        )
        nF = result['nF_realized']
        if nF == 0:
            rex = RexGraph(
                boundary_ptr=self._boundary_ptr.copy(),
                boundary_idx=self._boundary_idx.copy(),
                w_E=self._w_E, directed=self._directed, signs=self._signs,
            )
        else:
            from rexgraph.core._boundary import build_B2_from_cycles
            cycle_lengths = np.full(nF, 3, dtype=_i32)
            B2_dual = build_B2_from_cycles(
                self._nE, result['realized_edges'],
                result['realized_signs'], cycle_lengths)
            B2_dense = _sparse.to_dense_f64(B2_dual)
            from scipy import sparse as sp
            B2_sp = sp.csc_matrix(B2_dense)
            rex = RexGraph(
                boundary_ptr=self._boundary_ptr.copy(),
                boundary_idx=self._boundary_idx.copy(),
                B2_col_ptr=np.asarray(B2_sp.indptr, dtype=_i32),
                B2_row_idx=np.asarray(B2_sp.indices, dtype=_i32),
                B2_vals=np.asarray(B2_sp.data, dtype=_f64),
                w_E=self._w_E, directed=self._directed, signs=self._signs,
            )
        rex._typed_face_result = result
        return rex

    # Quotient filtration

    def quotient_filtration(self, channel: int, *, n_steps: int = 20) -> dict:
        """Filtration by removing edges in order of decreasing chi[:, channel].

        Parameters
        ----------
        channel : int
        n_steps : int

        Returns
        -------
        dict with thresholds, beta0, beta1, beta2, n_edges_remaining,
        edges_removed_order, transition_index, transition_threshold.
        """
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        return _quotient.quotient_filtration_by_character(
            self.structural_character, channel, n_steps,
            self.B1, self.B2_hodge,
            self._nV, self._nE, self.nF_hodge,
        )

    # Linkage complex

    def linkage_complex(self, sfb_threshold: float = 0.85) -> 'RexGraph':
        """Build a new RexGraph from fiber bundle similarity S_fb.

        Thresholds the vertex-vertex S_fb matrix to produce edges,
        enumerates all triangles as faces, and builds boundary operators.

        Parameters
        ----------
        sfb_threshold : float

        Returns
        -------
        RexGraph
        """
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        result = _fiber.linkage_complex(
            self.fiber_similarity, sfb_threshold, self._nV)
        if result['n_edges'] == 0:
            return RexGraph(
                sources=np.zeros(0, dtype=_i32),
                targets=np.zeros(0, dtype=_i32),
            )
        B2 = result.get('B2')
        if B2 is not None and result['nF'] > 0:
            from scipy import sparse as sp
            B2_sp = sp.csc_matrix(np.asarray(B2, dtype=_f64))
            return RexGraph(
                sources=result['src'], targets=result['tgt'],
                B2_col_ptr=np.asarray(B2_sp.indptr, dtype=_i32),
                B2_row_idx=np.asarray(B2_sp.indices, dtype=_i32),
                B2_vals=np.asarray(B2_sp.data, dtype=_f64),
            )
        return RexGraph(sources=result['src'], targets=result['tgt'])

    # Layout

    @cached_property
    def layout(self) -> NDArray:
        """2D spectral layout with force-directed refinement.

        Uses compute_layout(), which handles spectral embedding and
        selects between O(n^2) naive and O(n log n) Barnes-Hut
        based on vertex count.
        """
        if self._nV == 0:
            return np.empty((0, 2), dtype=_f64)

        sb = self.spectral_bundle
        evecs = np.ascontiguousarray(sb['evecs_L0'], dtype=_f64)
        evals = sb['evals_L0']
        src, tgt = self._ensure_src_tgt()

        px, py = _spectral.compute_layout(
            evecs, self._nV, self._nE,
            np.ascontiguousarray(src, dtype=_i32),
            np.ascontiguousarray(tgt, dtype=_i32),
            evals_in=evals,
        )
        return np.column_stack([px, py])

    @cached_property
    def layout_3d(self) -> NDArray:
        layout_2d = self.layout
        sb = self.spectral_bundle
        evecs = sb['evecs_L0']
        if evecs.shape[1] >= 4:
            pz = np.ascontiguousarray(evecs[:, 3], dtype=_f64)
        else:
            pz = np.zeros(self._nV, dtype=_f64)
        return np.column_stack([layout_2d, pz])

    # Cycle and face operations

    @cached_property
    def cycle_basis(self) -> list:
        """Fundamental cycle basis from tree-cotree decomposition."""
        src, tgt = self._ensure_src_tgt()
        nV, nE = self._nV, self._nE
        if self.has_branching:
            expanded = self.clique_expansion
            result = _cycles.find_fundamental_cycles(
                expanded.nV, expanded.nE, expanded.sources, expanded.targets
            )
        else:
            result = _cycles.find_fundamental_cycles(nV, nE, src, tgt)

        edges, signs, lengths, nF, nc = result
        cycles = []
        offset = 0
        for i in range(nF):
            clen = int(lengths[i])
            c = np.zeros(self._nE, dtype=_f64)
            for k in range(clen):
                c[int(edges[offset + k])] = float(signs[offset + k])
            cycles.append(c)
            offset += clen
        return cycles

    def fill_cycle(self, cycle_edges: NDArray) -> RexGraph:
        """Adjoin a face whose boundary is the given cycle."""
        self._ensure_clean()
        c = np.asarray(cycle_edges, dtype=_f64)
        new_col = c.reshape(-1, 1)
        if self._nF == 0:
            B2_dense = new_col
        else:
            B2_dense = np.hstack([self.B2, new_col])
        from scipy import sparse as sp
        B2_sp = sp.csc_matrix(B2_dense)
        return RexGraph(
            boundary_ptr=self._boundary_ptr.copy(),
            boundary_idx=self._boundary_idx.copy(),
            B2_col_ptr=np.asarray(B2_sp.indptr, dtype=_i32),
            B2_row_idx=np.asarray(B2_sp.indices, dtype=_i32),
            B2_vals=np.asarray(B2_sp.data, dtype=_f64),
            w_E=self._w_E,
            w_boundary=self._w_boundary,
            directed=self._directed,
        )

    def promote(self) -> RexGraph:
        """Promote to a 2-rex with beta_1 = 0."""
        R = self
        for c in self.cycle_basis:
            R = R.fill_cycle(c)
        return R

    # Face data (for dashboard / analysis.py)

    def face_data(
        self,
        vertex_names: list,
        edge_names: list,
        rho: NDArray,
    ) -> dict:
        """Face analysis via _faces.build_face_data.

        Parameters
        ----------
        vertex_names : list[str]
        edge_names : list[str]
        rho : f64[nE]
            Per-edge harmonic fraction from Hodge decomposition.

        Returns
        -------
        dict
            Contains faces, vertex_face_count, and metrics.
        """
        if self._nF == 0 or self._B2_dual is None:
            return {'faces': [], 'vertex_face_count': np.zeros(self._nV, dtype=_i32), 'metrics': {}}
        src, tgt = self._ensure_src_tgt()
        return _faces.build_face_data(
            self._B2_dual, src, tgt, self._nV,
            vertex_names, edge_names,
            np.ascontiguousarray(rho, dtype=_f64),
        )

    # Hodge decomposition

    def hodge(self, g: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """Hodge decomposition: g = B1^T phi + B2 psi + eta.

        Uses B2_hodge (self-loop faces filtered) so that B_1 B_2 = 0
        holds exactly and the three components are orthogonal.
        """
        flow = np.ascontiguousarray(g, dtype=_f64)
        return _hodge.hodge_decomposition(self._B1_dual, self._B2_hodge_dual, flow)

    def hodge_full(self, g: NDArray) -> dict:
        """Full Hodge analysis with normalized components, rho, and divergence.

        Uses B2_hodge for exact orthogonality. Returns a dict with:
        grad, curl, harm, grad_norm, curl_norm, harm_norm, flow_norm,
        rho, pct_grad, pct_curl, pct_harm, divergence, div_norm,
        face_curl, orthogonality.
        """
        flow = np.ascontiguousarray(g, dtype=_f64)
        sb = self.spectral_bundle
        return _hodge.build_hodge(
            self._B1_dual, self._B2_hodge_dual, flow,
            L0=sb.get('L0'), L2=sb.get('L2'),
        )

    # Signal operations

    def signal(self, dim: int, values: ArrayLike) -> NDArray:
        v = np.asarray(values, dtype=_f64)
        expected = [self._nV, self._nE, self._nF][dim]
        if v.shape[0] != expected:
            raise ValueError(f"Expected {expected} values for dim {dim}, got {v.shape[0]}.")
        return v

    def signal_energy(self, g: NDArray, dim: int) -> float:
        """Rayleigh quotient <g|L|g> for signal g on dimension dim."""
        L = [self.L0, self.L1, self.L2][dim]
        g = np.ascontiguousarray(g, dtype=_f64)
        return float(g @ L @ g)

    def normalize(self, g: NDArray, norm: str = "l2") -> NDArray:
        if norm == "l1":
            return _state.normalize_l1(g)
        return _state.normalize_l2(g)

    # State construction

    def create_state(self, t: float = 0.0):
        """Create a RexState bound to this graph's dimensions."""
        return _state.RexState(self._nV, self._nE, self._nF, t)

    def energy_kin_pot(self, f_E: NDArray) -> Tuple[float, float, float]:
        """Kinetic/potential energy decomposition of an edge signal, Hodge-decomposed.

        Returns (E_kin, E_pot, ratio) where:
            E_kin = <f_E | L1_down | f_E>  = ||B_1 f||^2   (gradient / drain energy)
            E_pot = <f_E | L1_up   | f_E>  = ||B_2^T f||^2 (curl / rotation energy)
        so E_kin + alpha_G*E_pot = <f | RL_1 | f> is consistent with the relational Laplacian
        RL_1 = L1_down + alpha_G*L1_up. (Was L_1 and L_O, which did not match RL_1.)
        """
        f = np.ascontiguousarray(f_E, dtype=_f64)
        return _state.energy_kin_pot(f, self.L1_down, self.L1_up)

    def dirac_state(self, dim: int, idx: int) -> Tuple[NDArray, NDArray, NDArray]:
        """Dirac delta state: all zeros except 1.0 at (dim, idx)."""
        return _state.dirac_state(self._nV, self._nE, self._nF, dim, idx)

    def dirac_edge(self, edge_idx: int) -> Tuple[NDArray, NDArray]:
        """Dirac delta on a single edge (field state, no V).

        Returns (f_E, f_F) for perturbation analysis.
        """
        return _state.dirac_edge(self._nE, self._nF, edge_idx)

    def uniform_state(self, norm: str = "l1") -> Tuple[NDArray, NDArray, NDArray]:
        """Uniform state at all dimensions."""
        norm_type = 0 if norm == "l1" else 1
        return _state.uniform_state(self._nV, self._nE, self._nF, norm_type)

    # Perturbation constructors

    def edge_perturbation(self, edge_idx: int) -> Tuple[NDArray, NDArray]:
        """Dirac delta on a single edge for perturbation analysis."""
        return _signal.build_edge_perturbation(self._nE, self._nF, edge_idx)

    def vertex_perturbation(self, vertex_idx: int) -> Tuple[NDArray, NDArray]:
        """Perturbation at a vertex, spread to incident edges via B_1^T."""
        return _signal.build_vertex_perturbation(
            vertex_idx, self.B1, self._nE, self._nF)

    def multi_edge_perturbation(self, edge_indices: ArrayLike) -> Tuple[NDArray, NDArray]:
        """Uniform perturbation across multiple edges."""
        idx = np.asarray(edge_indices, dtype=_i32)
        return _signal.build_multi_edge_perturbation(self._nE, self._nF, idx)

    def spectral_perturbation(self, mode_idx: int = 1) -> Tuple[NDArray, NDArray]:
        """Perturbation from a single RL_1 eigenmode."""
        evecs = self.evecs_RL1
        if evecs is None:
            evecs = np.linalg.eigh(self.L1)[1]
        return _signal.build_spectral_perturbation(
            self._nE, self._nF, evecs, mode_idx)

    # Per-edge energy decomposition

    def per_edge_energy(self, f_E: NDArray) -> Tuple[NDArray, NDArray]:
        """Per-edge kinetic and potential energy contributions.

        Kinetic from L1_down (gradient), potential from L1_up (curl), consistent with
        `energy_kin_pot` and RL_1 = L1_down + alpha_G*L1_up; each sums to the corresponding total.
        """
        f = np.ascontiguousarray(f_E, dtype=_f64)
        return _quotient.per_edge_energy(f, self.L1_down, self.L1_up)

    def subcomplex_by_energy(
        self,
        f_E: NDArray,
        regime: int,
        *,
        ratio_tol: float = 0.2,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Subcomplex of edges in a specific energy regime.

        regime: 0=kinetic, 1=crossover, 2=potential.
        """
        self._ensure_clean()
        E_kin, E_pot = self.per_edge_energy(f_E)
        return _quotient.subcomplex_by_energy_regime(
            E_kin, E_pot, regime, ratio_tol,
            self._nV, self._boundary_ptr, self._boundary_idx,
        )

    # Hyperslice

    def hyperslice(self, dim: int, idx: int) -> Tuple:
        """Hyperslice through cell sigma in C_d.

        Returns
        -------
        For dim=0: (above_edges, lateral_vertices)              (2-tuple)
        For dim=1: (below_vertices, above_faces, lateral_edges)  (3-tuple)
        For dim=2: (below_edges, lateral_faces)              (2-tuple)
        """
        v2e_ptr, v2e_idx = self._v2e
        e2f_ptr, e2f_idx = self._e2f
        src, tgt = self._ensure_src_tgt()

        return _rex.hyperslice(
            dim, idx,
            v2e_ptr=v2e_ptr, v2e_idx=v2e_idx,
            sources=src, targets=tgt,
            e2f_ptr=e2f_ptr, e2f_idx=e2f_idx,
            nF=self._nF,
            B2_col_ptr=self._B2_col_ptr,
            B2_row_idx=self._B2_row_idx,
        )

    def hyperslice_telescope(self, dim: int, idx: int, depth: int = 1) -> dict:
        result = {"center": (dim, idx)}
        below_cells = [(dim, idx)]
        above_cells = [(dim, idx)]
        for step in range(1, depth + 1):
            new_below = []
            for d, i in below_cells:
                if d > 0:
                    hs = self.hyperslice(d, i)
                    if d == 1:
                        below, above, lateral = hs
                        new_below.extend((d - 1, int(c)) for c in below)
                    elif d == 2:
                        below, lateral = hs
                        new_below.extend((d - 1, int(c)) for c in below)
            result[f"below_{step}"] = new_below
            below_cells = new_below

            new_above = []
            for d, i in above_cells:
                if d < self.dimension:
                    hs = self.hyperslice(d, i)
                    if d == 0:
                        above, lateral = hs
                        new_above.extend((d + 1, int(c)) for c in above)
                    elif d == 1:
                        below, above, lateral = hs
                        new_above.extend((d + 1, int(c)) for c in above)
            result[f"above_{step}"] = new_above
            above_cells = new_above

            hs = self.hyperslice(dim, idx)
            if dim == 1:
                _, _, lat = hs
            else:
                _, lat = hs
            result[f"lateral_{step}"] = [(dim, int(c)) for c in lat]
        return result

    # Transition operators

    def evolve_markov(self, g: NDArray, dim: int, t: float) -> NDArray:
        """Markov continuous-time evolution via matrix exponential.

        Eigen-free: e^{-tL} g by matrix-free Chebyshev on the SPARSE Laplacian
        (== the dense expm markov_continuous_expm to ~1e-10), no O(n^3) expm.
        """
        from rexgraph import scale_propagator as _spg
        L_sp = [self.L0_sparse, self.L1_sparse, self.L2_sparse][dim]
        return _spg.heat_apply(L_sp, np.ascontiguousarray(g, dtype=_f64), float(t))

    def evolve_schrodinger(self, psi: NDArray, dim: int, t: float) -> Tuple[NDArray, NDArray]:
        """Schrodinger (unitary) evolution via spectral method.

        Returns (f_real, f_imag) components of exp(-i L_k t) psi.

        Eigen-free: e^{-iLt} = cos(tL) - i sin(tL) applied via one shared set of
        Chebyshev matvecs on the SPARSE L (== the dense mode-sum
        schrodinger_evolve_spectral to ~1e-10), no eigendecomposition.
        """
        from rexgraph import scale_propagator as _spg
        L_sp = [self.L0_sparse, self.L1_sparse, self.L2_sparse][dim]
        psi = np.ascontiguousarray(psi, dtype=_f64)
        lam_max = _spg._gershgorin_bound(L_sp) * 1.0001 + 1e-30
        order = int(max(24, min(400, 1.5 * float(t) * lam_max + 24)))
        f_re, f_im = _spg.matfunc_trajectory(
            L_sp, psi,
            [lambda l: np.cos(float(t) * l), lambda l: -np.sin(float(t) * l)],
            order, lam_max=lam_max,
        )
        return np.ascontiguousarray(f_re), np.ascontiguousarray(f_im)

    def evolve_coupled(
            self,
            state: NDArray,
            t: float,
            *,
            n_steps: int = 100,
            alpha0: float = 1.0,
            alpha1: float = 1.0,
            alpha2: float = 1.0,
        ) -> Tuple[NDArray, NDArray, NDArray]:
            """Coupled cross-dimensional diffusion via RK4 integration.

            Uses RL_1 = alpha1 * L_1 + alpha_G * L_O on the edge tier,
            and B2_hodge for the face coupling.

            Parameters
            ----------
            state : f64[nV + nE + nF]
                Packed state vector (f0, f1, f2).
            t : float
                Total integration time.
            n_steps : int
                Number of RK4 steps.

            Returns
            -------
            y_final : f64[nV + nE + nF]
            trajectory : f64[n_steps+1, nV+nE+nF]
            times : f64[n_steps+1]
            """
            sizes = np.array([self._nV, self._nE, self._nF], dtype=_i32)
            ag = self.alpha_G
            if ag != ag:  # NaN check
                ag = 0.0

            _L0 = self.L0
            _L1 = self.L1
            _L2 = self.L2 if self._nF > 0 else np.zeros((0, 0), dtype=_f64)
            _L_O = self.L_overlap
            _B1 = self.B1
            _B2 = self.B2_hodge

            def deriv(y, _t):
                return _transition.coupled_derivative(
                    y, sizes,
                    _L0, _L1, _L2, _L_O, _B1, _B2,
                    alpha0, alpha1, alpha2, ag,
                )

            return _transition.rk4_integrate(state, 0.0, t, n_steps, deriv)

    # Wave mechanics

    def wave_state(self, dim: int, amplitudes: ArrayLike = None) -> NDArray:
        if amplitudes is None:
            n = [self._nV, self._nE, self._nF][dim]
            psi = (np.random.randn(n) + 1j * np.random.randn(n)).astype(_c128)
        else:
            psi = np.asarray(amplitudes, dtype=_c128)
        _wave.normalize_c128(psi)
        return psi

    def measure(self, psi: NDArray, dim: int = None) -> Tuple[int, NDArray]:
        psi = psi.astype(_c128, copy=False)
        probs = _wave.born_probabilities(psi)
        outcome = np.random.choice(len(probs), p=probs / probs.sum())
        collapsed = np.zeros_like(psi)
        collapsed[outcome] = 1.0 + 0j
        return outcome, collapsed

    def born_probabilities(self, psi: NDArray) -> NDArray:
        return _wave.born_probabilities(psi.astype(_c128, copy=False))

    def entanglement_entropy(self, psi: NDArray, dim_A: int, dim_B: int = None) -> float:
        if dim_B is None:
            dim_B = len(psi) // dim_A
        return _wave.entanglement_entropy(psi, dim_A, dim_B)

    # Rex field wave evolution (complex amplitudes on (E, F))

    def evolve_field_wave(
        self,
        psi_E: NDArray,
        psi_F: NDArray,
        t: float,
    ) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Schrodinger evolution on the rex field (E, F).

        psi_E evolves under RL_1, psi_F under L_2. Vertex observables
        are derived via B_1 psi_E.

        Returns (psi_E_t, psi_F_t, psi_V_t).
        """
        import scipy.sparse as _sp
        from rexgraph import scale_propagator as _spg
        psi_E = np.asarray(psi_E, dtype=_c128)
        psi_F = np.asarray(psi_F, dtype=_c128)
        # Eigen-free: e^{-i RL1 t} psi_E and e^{-i L2 t} psi_F via one shared set of
        # Chebyshev matvecs on the SPARSE operators (== the dense mode-sum
        # field_schrodinger_evolve to ~1e-10, no eigh on RL1/L1/L2).
        RL1 = self.relational_laplacian
        RL1 = _sp.csr_matrix(np.asarray(RL1)) if RL1 is not None else self.L1_sparse
        psi_E_t = _spg.schrodinger_apply(RL1, psi_E, t)
        if self.nF_hodge > 0 and psi_F.shape[0] > 0:
            psi_F_t = _spg.schrodinger_apply(self.L2_sparse, psi_F, t)
        else:
            psi_F_t = psi_F.copy()
        psi_V_t = self.B1.dot(psi_E_t.real) + 1j * self.B1.dot(psi_E_t.imag)
        return psi_E_t, psi_F_t, psi_V_t

    def evolve_field_trajectory(
        self,
        psi_E: NDArray,
        psi_F: NDArray,
        times: NDArray,
    ) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Rex field Schrodinger evolution through multiple timepoints.

        Returns (traj_E, traj_F, traj_V) each shaped [nT, ...].
        """
        import scipy.sparse as _sp
        from rexgraph import scale_propagator as _spg
        psi_E = np.asarray(psi_E, dtype=_c128)
        psi_F = np.asarray(psi_F, dtype=_c128)
        times = np.ascontiguousarray(times, dtype=_f64)
        # Eigen-free trajectory: shared Chebyshev vectors across all timepoints
        # (== the dense mode-sum field_schrodinger_trajectory to ~1e-10, no eigh).
        RL1 = self.relational_laplacian
        RL1 = _sp.csr_matrix(np.asarray(RL1)) if RL1 is not None else self.L1_sparse
        traj_E = _spg.schrodinger_trajectory(RL1, psi_E, times)      # (nT, nE) complex
        if self.nF_hodge > 0 and psi_F.shape[0] > 0:
            traj_F = _spg.schrodinger_trajectory(self.L2_sparse, psi_F, times)
        else:
            traj_F = np.repeat(psi_F[None, :], times.shape[0], axis=0)
        traj_V = traj_E @ self.B1.T                                  # (nT, nV) complex
        return traj_E, traj_F, traj_V

    def measure_in_eigenbasis(self, psi: NDArray, dim: int = 1) -> Tuple:
        """Measure in the eigenbasis of the Laplacian for dimension dim.

        For dim=1 with RL_1 available, uses RL_1 eigenvectors.
        Returns (outcome, probability, collapsed_state).
        """
        psi = np.asarray(psi, dtype=_c128)
        sb = self.spectral_bundle
        if dim == 1 and sb.get('evecs_RL_1') is not None:
            evecs = sb['evecs_RL_1']
        elif dim == 0:
            evecs = sb['evecs_L0']
        elif dim == 2 and sb.get('evecs_L2') is not None:
            evecs = sb['evecs_L2']
        else:
            # on-demand FULL eigenbasis. dim=1 uses RL_1 (matches the fast path above), not L1.
            if dim == 1:
                L = self.relational_laplacian
                if L is None:
                    L = self.L1
            else:
                L = [self.L0, self.L1, self.L2][dim]
            _, evecs = np.linalg.eigh(np.ascontiguousarray(_ensure_dense(L), dtype=_f64))
        return _wave.measure_in_eigenbasis(psi, np.ascontiguousarray(evecs, dtype=_f64))

    # Field operator (coupled edge-face dynamics from _field)

    @cached_property
    def field_operator(self) -> Tuple[NDArray, float, bool]:
        """Coupled field operator M on (E, F) space.

        M = [[ RL_1,      -g * B_2     ],
             [-g * B_2^T,     L_2      ]]

        Returns (M, g_used, is_psd).
        """
        RL = self.relational_laplacian
        if RL is None:
            RL = _ensure_dense(self.L1)
        L2 = _ensure_dense(self.L2)
        B2h = self.B2_hodge
        return _field.build_field_operator(RL, L2, B2h)

    @cached_property
    def field_eigen(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Eigendecomposition of the field operator M.

        Returns (evals, evecs, freqs) where freqs = sqrt(evals).
        """
        M, _, _ = self.field_operator
        return _field.field_eigendecomposition(M)

    def field_diffuse(self, F0: NDArray, times: NDArray) -> NDArray:
        """First-order diffusion on (E, F) via the field operator.

        F(t) = sum_k exp(-lambda_k t) <v_k|F0> v_k.

        Parameters
        ----------
        F0 : f64[nE + nF] - packed initial field state
        times : f64[T] - timepoints

        Returns
        -------
        trajectory : f64[T, nE + nF]
        """
        F0 = np.ascontiguousarray(F0, dtype=_f64)
        times = np.ascontiguousarray(times, dtype=_f64)
        # Eigen-free: matrix-free Chebyshev e^{-tM} on the SPARSE field operator
        # (== the dense mode-sum field_diffusion_trajectory to ~1e-10, no eigh).
        from rexgraph import field_propagator as _fp
        return _fp.field_heat_trajectory(self, F0, times)

    def field_wave_evolve(
        self,
        F0: NDArray,
        dFdt0: NDArray,
        times: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Second-order wave equation on (E, F).

        d^2F/dt^2 = -M F. Returns (position_traj, velocity_traj).

        Parameters
        ----------
        F0 : f64[nE + nF] - initial position
        dFdt0 : f64[nE + nF] - initial velocity
        times : f64[T]

        Returns
        -------
        traj : f64[T, nE + nF]
        vel_traj : f64[T, nE + nF]
        """
        F0 = np.ascontiguousarray(F0, dtype=_f64)
        dFdt0 = np.ascontiguousarray(dFdt0, dtype=_f64)
        times = np.ascontiguousarray(times, dtype=_f64)
        # Eigen-free: matrix-free cos(t sqrt(M)) F0 and -sqrt(M) sin(t sqrt(M)) F0 on
        # the SPARSE field operator (== the dense mode-sum wave_evolve_trajectory to
        # ~1e-10, no eigh). dFdt0 is accepted for API parity but, as in the prior
        # dense path, the propagated wave uses the zero-initial-velocity solution.
        from rexgraph import field_propagator as _fp
        pos, vel = _fp.field_wave_full(self, F0, times)
        return pos, vel

    def classify_modes(self) -> dict:
        """Classify field eigenmodes as edge-dominated, face-dominated, or coupled.

        Returns (mode_type, edge_weight, face_weight, n_resonant).
        """
        evals, evecs, freqs = self.field_eigen
        return _field.classify_modes(evals, evecs, self._nE, int(self.nF_hodge))

    def derive_vertex_state(self, F: NDArray) -> NDArray:
        """Derive vertex observable from packed field state via B_1.

        f_V = B_1 @ F[:nE].
        """
        F = np.ascontiguousarray(F, dtype=_f64)
        return _field.derive_vertex_state(F, self.B1, self._nE)

    # Dirac operator and graded state

    @cached_property
    def dirac_operator(self) -> NDArray:
        """Dirac operator D = d + d* on R^(nV+nE+nF). Real symmetric.

        D^2 = blkdiag(L0, L1, L2) by the chain condition B1 B2 = 0.
        """
        if _dirac is None:
            raise RuntimeError("_dirac module not available")
        D, _ = _dirac.build_dirac_operator(self.B1, self.B2_hodge)
        return D

    @cached_property
    def _dirac_eigen(self) -> Tuple[NDArray, NDArray]:
        """Cached eigendecomposition of the Dirac operator."""
        if _dirac is None:
            raise RuntimeError("_dirac module not available")
        return _dirac.dirac_eigen(self.dirac_operator)

    @cached_property
    def dirac_eigenvalues(self) -> NDArray:
        """Eigenvalues of the Dirac operator (positive and negative)."""
        return self._dirac_eigen[0]

    def graded_state(self, t: float = 0.0,
                     psi0: NDArray = None,
                     vertex_idx: int = 0) -> Tuple[NDArray, NDArray]:
        """Evolve graded state: Psi(t) = exp(-iDt) Psi(0).

        If psi0 is None, uses canonical collapse at vertex_idx.
        Returns (psi_re, psi_im).
        """
        if _dirac is None:
            raise RuntimeError("_dirac module not available")
        evals, evecs = self._dirac_eigen
        if psi0 is None:
            psi0 = _dirac.canonical_collapse(
                self.B1, self._nV, self._nE, self.nF_hodge, vertex_idx)
        psi0 = np.ascontiguousarray(psi0, dtype=_f64)
        return _dirac.schrodinger_evolve(evals, evecs, psi0, t)

    def graded_trajectory(self, times: NDArray,
                          psi0: NDArray = None,
                          vertex_idx: int = 0) -> dict:
        """Evolve graded state at multiple timepoints.

        Returns dict with traj_re, traj_im, born.
        """
        if _dirac is None:
            raise RuntimeError("_dirac module not available")
        evals, evecs = self._dirac_eigen
        if psi0 is None:
            psi0 = _dirac.canonical_collapse(
                self.B1, self._nV, self._nE, self.nF_hodge, vertex_idx)
        psi0 = np.ascontiguousarray(psi0, dtype=_f64)
        times = np.ascontiguousarray(times, dtype=_f64)
        traj_re, traj_im, born = _dirac.schrodinger_trajectory(
            evals, evecs, psi0, times)
        return {
            'traj_re': traj_re, 'traj_im': traj_im, 'born': born,
            'times': times, 'nV': self._nV, 'nE': self._nE,
            'nF': self.nF_hodge,
        }

    def canonical_collapse(self, vertex_idx: int = 0) -> NDArray:
        """Canonical graded projection: (delta_v, B1^T delta_v, 0).

        Face component is exactly zero by the chain condition.
        """
        if _dirac is None:
            raise RuntimeError("_dirac module not available")
        return _dirac.canonical_collapse(
            self.B1, self._nV, self._nE, self.nF_hodge, vertex_idx)

    def born_graded(self, psi_re: NDArray, psi_im: NDArray) -> Tuple[NDArray, NDArray]:
        """Born probability per cell and per dimension.

        Returns (per_cell, per_dim) where per_dim = [P_V, P_E, P_F].
        """
        if _dirac is None:
            raise RuntimeError("_dirac module not available")
        return _dirac.born_graded(psi_re, psi_im,
                                   self._nV, self._nE, self.nF_hodge)

    def energy_partition(self, psi_re: NDArray, psi_im: NDArray) -> NDArray:
        """Fraction of energy in V, E, F sectors. Sums to 1."""
        if _dirac is None:
            raise RuntimeError("_dirac module not available")
        return _dirac.energy_partition(psi_re, psi_im,
                                        self._nV, self._nE, self.nF_hodge)

    # Hypermanifold

    @cached_property
    def hypermanifold(self) -> dict:
        """Filtered manifold sequence M1 < M2 < M3.

        Each level adds cells, DOF, Bianchi identities.
        """
        if _hypermanifold is None:
            raise RuntimeError("_hypermanifold module not available")
        # EIGEN-FREE: Betti from ranks/union-find (rex.betti), not eigenvalue nullity.
        b0, b1, b2 = self.betti
        return _hypermanifold.build_manifold_sequence_from_betti(
            int(b0), int(b1), int(b2), self._nV, self._nE, self.nF_hodge)

    @cached_property
    def harmonic_shadow(self) -> dict:
        """Cycles at d=1 that become boundaries at d=2.

        shadow_dim = beta_1(1) - beta_1(2) = rank(B2).
        """
        # EIGEN-FREE: shadow_dim = rank(B2) = beta_1(d=1) - beta_1(d=2), from the exact
        # rank / union-find path - no dense eigh(L1_down), no eigenvalue nullity.
        from rexgraph.graded_boundary import (
            graded_boundaries_from_rex, _sparse_rank)
        Bs = graded_boundaries_from_rex(self)
        nV, nE = int(self._nV), int(self._nE)
        b0, b1, _ = self.betti
        beta_1_at_d1 = nE - (nV - int(b0))            # cycle-space dim (no faces)
        rank_B2 = _sparse_rank(Bs[1]) if len(Bs) > 1 else 0
        beta_1_at_d2 = int(b1)                        # = beta_1_at_d1 - rank_B2
        return {
            'shadow_dim': int(rank_B2),
            'beta_1_at_d1': int(beta_1_at_d1),
            'beta_1_at_d2': int(beta_1_at_d2),
        }

    @cached_property
    def dimensional_subsumption(self) -> Tuple[bool, list]:
        """Verify beta_k(d+1) <= beta_k(d) (Theorem 8.1)."""
        if _hypermanifold is None:
            return True, []
        hm = self.hypermanifold
        betti_seq = [m['betti'] for m in hm['manifolds']]
        return _hypermanifold.dimensional_subsumption(betti_seq)

    # Signal analysis pipeline (from _signal)

    def analyze_perturbation(
        self,
        f_E: NDArray,
        f_F: Optional[NDArray] = None,
        *,
        times: Optional[NDArray] = None,
        n_steps: int = 50,
        t_max: float = 10.0,
    ) -> dict:
        """One-call perturbation analysis pipeline.

        Propagates f_E under RL_1 diffusion and computes energy
        trajectory, cascade activation, face emergence, BIOES phase
        tags, Hodge decomposition of initial/final states, and
        derived vertex observables.

        Parameters
        ----------
        f_E : f64[nE] - initial edge signal
        f_F : f64[nF] or None - initial face signal (default zeros)
        times : f64[T] or None - timepoints (auto-generated if None)
        n_steps : int - number of steps if times is None
        t_max : float - max time if times is None

        Returns
        -------
        dict with trajectory, E_kin, E_pot, ratio, cascade data,
        BIOES tags, Hodge decomposition, vertex observables.
        """
        f_E = np.ascontiguousarray(f_E, dtype=_f64)
        if f_F is None:
            f_F = np.zeros(self._nF, dtype=_f64)
        else:
            f_F = np.ascontiguousarray(f_F, dtype=_f64)
        if times is None:
            times = np.linspace(0, t_max, n_steps, dtype=_f64)
        else:
            times = np.ascontiguousarray(times, dtype=_f64)

        # EIGEN-FREE diffusion trajectory: e^{-t·op} f_E via Chebyshev sparse mat-vecs
        # (no eigendecomposition), sharing one set of Chebyshev vectors across all t.
        # `op` is the SAME operator the dense path propagated under: RL_1 when its
        # relational Laplacian is available (the intended operator), else L_1.
        from rexgraph import scale_propagator as _spg
        RL1_op = self.relational_laplacian
        if self.evals_RL1 is not None and RL1_op is not None:
            op = RL1_op
        else:
            op = self.L1_sparse
        trajectory = _spg.heat_trajectory(op, f_E, times)     # (T, nE), matrix-free

        sb = self.spectral_bundle
        src, tgt = self._ensure_src_tgt()
        ag = self.alpha_G
        if ag != ag:
            ag = 0.0

        return _signal.analyze_perturbation(
            f_E, f_F,
            self.L1_down, self.L1_up,                 # kinetic=gradient, potential=curl (matches RL_1)
            None, None,                               # spectrum unused: trajectory is eigen-free
            self.B1, self.B2_hodge,
            times,
            L0=sb.get('L0'),
            L2_op=sb.get('L2'),
            RL1=self.relational_laplacian,
            edge_src=np.ascontiguousarray(src, dtype=_i32),
            edge_tgt=np.ascontiguousarray(tgt, dtype=_i32),
            alpha_G=ag,
            precomputed_trajectory=trajectory,
        )

    def analyze_perturbation_field(
        self,
        f_E: NDArray,
        f_F: Optional[NDArray] = None,
        *,
        times: Optional[NDArray] = None,
        n_steps: int = 50,
        t_max: float = 10.0,
        mode: str = "diffusion",
    ) -> dict:
        """Perturbation analysis using the full (E, F) field operator.

        Propagates the packed field state under the coupled field operator
        M, then extracts per-dimension energy and cascade information.

        Parameters
        ----------
        f_E : f64[nE]
        f_F : f64[nF] or None
        times : f64[T] or None
        mode : 'diffusion' or 'wave'

        Returns
        -------
        dict with field_trajectory, edge_trajectory, face_trajectory,
        vertex_trajectory, E_kin, E_pot, norm_E, norm_F, and
        wave energy data (if mode='wave').
        """
        f_E = np.ascontiguousarray(f_E, dtype=_f64)
        if f_F is None:
            f_F = np.zeros(self.nF_hodge, dtype=_f64)
        else:
            f_F = np.ascontiguousarray(f_F, dtype=_f64)
        if times is None:
            times = np.linspace(0, t_max, n_steps, dtype=_f64)
        else:
            times = np.ascontiguousarray(times, dtype=_f64)

        # EIGEN-FREE for BOTH modes: matrix-free Chebyshev on the SPARSE graded field
        # operator (no dense (nE+nF)² eigendecomposition). diffusion -> e^{-tM}; wave ->
        # cos(t√M) positions + -√M sin(t√M) velocities.
        from rexgraph import field_propagator as _fp
        F0 = np.concatenate([f_E, f_F])
        Msp = _fp.assemble_field_operator(self)          # SPARSE M, O(nnz), built once
        evals = evecs = freqs = None
        precomputed_vel = None
        if mode == "wave":
            # wave_energy downstream needs the operator (sparse M.dot), not the spectrum
            precomputed, precomputed_vel = _fp.field_wave_full(self, F0, times, M=Msp)
            M = Msp
        else:
            precomputed = _fp.field_heat_trajectory(self, F0, times, M=Msp)  # (T, nE+nF)
            M = None                                     # diffusion doesn't use M downstream

        return _signal.analyze_perturbation_field(
            f_E, f_F, M, evals, evecs, freqs,
            self.L1_down, self.L1_up, self.B1,        # kinetic=gradient, potential=curl (matches RL_1)
            times, self._nE, self.nF_hodge, mode,
            precomputed_trajectory=precomputed,
            precomputed_velocity=precomputed_vel,
        )

    # Quotient complex

    def subcomplex(
        self,
        *,
        v_mask: Optional[NDArray] = None,
        e_mask: Optional[NDArray] = None,
        f_mask: Optional[NDArray] = None,
        edge_type: Optional[int] = None,
        signal: Optional[NDArray] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        self._ensure_clean()
        if edge_type is not None:
            return _quotient.subcomplex_by_edge_type(
                self.edge_types.astype(_u8, copy=False), _u8(edge_type),
                self._nV, self._boundary_ptr, self._boundary_idx,
            )
        if signal is not None and threshold is not None:
            return _quotient.subcomplex_by_threshold(
                signal, threshold, True,
                self._nV, self._boundary_ptr, self._boundary_idx,
            )
        if e_mask is not None:
            vm = v_mask if v_mask is not None else np.zeros(self._nV, dtype=_u8)
            fm = f_mask if f_mask is not None else np.zeros(self._nF, dtype=_u8)
            return _quotient.closure_of_faces_and_edges(
                vm, _asarray(e_mask, _u8), fm,
                self._nV, self._nE,
                self._boundary_ptr, self._boundary_idx,
                self._B2_col_ptr, self._B2_row_idx,
            )
        raise ValueError("Specify edge_type, signal+threshold, or explicit masks.")

    def quotient(self, v_mask: NDArray, e_mask: NDArray, f_mask: NDArray) -> dict:
        """Build quotient complex R/I with optional RL_1 on quotient edges."""
        self._ensure_clean()
        ag = self.alpha_G
        lo = self.L_overlap if ag == ag and ag != 0.0 else None
        return _quotient.build_quotient(
            self.B1, v_mask, e_mask, f_mask,
            self._B2_col_ptr.astype(_i32, copy=False),
            self._B2_row_idx.astype(_i32, copy=False),
            self._B2_vals,
            LO=lo,
            alpha_G=ag if lo is not None else 0.0,
        )

    def congruent(self, dim: int, a: int, b: int, mask: NDArray) -> bool:
        """Test whether cells a and b are congruent modulo subcomplex."""
        if dim == 1:
            ok, _ = _quotient.congruent_edges(
                a, b, self.B1, _asarray(mask, _u8),
            )
            return bool(ok)
        ok, _ = _quotient.congruent_faces(
            a, b, self.B2, _asarray(mask, _u8),
        )
        return bool(ok)

    # Star subcomplexes

    def star_of_vertex(self, v: int) -> Tuple[NDArray, NDArray, NDArray]:
        """Star of a vertex: incident edges, incident faces, closed downward.

        Returns
        -------
        v_mask, e_mask, f_mask : uint8 arrays
            Masks for the star subcomplex.
        """
        v2e_ptr, v2e_idx = self._v2e
        e2f_ptr, e2f_idx = self._e2f
        return _quotient.star_of_vertex(
            _i32(v), self._nV, self._nE, self._nF,
            self._boundary_ptr, self._boundary_idx,
            v2e_ptr, v2e_idx, e2f_ptr, e2f_idx,
            self._B2_col_ptr, self._B2_row_idx,
        )

    def star_of_edge(self, edge_idx: int) -> Tuple[NDArray, NDArray, NDArray]:
        """Star of an edge: overlap neighborhood, incident faces, closed.

        Returns
        -------
        v_mask, e_mask, f_mask : uint8 arrays
            Masks for the star subcomplex.
        """
        v2e_ptr, v2e_idx = self._v2e
        e2f_ptr, e2f_idx = self._e2f
        return _quotient.star_of_edge(
            _i32(edge_idx), self._nV, self._nE, self._nF,
            self._boundary_ptr, self._boundary_idx,
            v2e_ptr, v2e_idx, e2f_ptr, e2f_idx,
            self._B2_col_ptr, self._B2_row_idx,
        )

    def validate_subcomplex(
        self,
        v_mask: NDArray,
        e_mask: NDArray,
        f_mask: NDArray,
    ) -> Tuple[bool, list]:
        """Check that masks define a valid subcomplex.

        Verifies closure conditions: boundary vertices of selected edges
        are selected, boundary edges of selected faces are selected.

        Returns
        -------
        valid : bool
        violations : list of (kind, cell_idx, missing_idx) tuples
        """
        self._ensure_clean()
        return _quotient.validate_subcomplex(
            _asarray(v_mask, _u8),
            _asarray(e_mask, _u8),
            _asarray(f_mask, _u8),
            self._boundary_ptr, self._boundary_idx,
            self._B2_col_ptr, self._B2_row_idx,
        )

    def hyperslice_quotient(
        self, dim: int, cell_idx: int,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Form a subcomplex from the hyperslice around a cell.

        For vertex: incident edges + faces, closed downward.
        For edge: boundary vertices + overlap neighbors + faces, closed.
        For face: boundary edges, closed.

        Parameters
        ----------
        dim : int
            Cell dimension (0=vertex, 1=edge, 2=face).
        cell_idx : int
            Index of the cell.

        Returns
        -------
        v_mask, e_mask, f_mask : uint8 arrays
        """
        self._ensure_clean()
        v2e_ptr, v2e_idx = self._v2e
        e2f_ptr, e2f_idx = self._e2f
        return _quotient.hyperslice_quotient(
            dim, cell_idx,
            self._nV, self._nE, self._nF,
            self._boundary_ptr, self._boundary_idx,
            v2e_ptr, v2e_idx, e2f_ptr, e2f_idx,
            self._B2_col_ptr, self._B2_row_idx,
        )

    def edge_type_quotient(
        self, type_codes: Sequence[int],
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Build a subcomplex from edges matching any of the given type codes.

        Parameters
        ----------
        type_codes : list of int
            Edge type codes (0=standard, 1=self-loop, 2=branching, 3=witness).

        Returns
        -------
        v_mask, e_mask, f_mask : uint8 arrays
        """
        return _quotient.edge_type_quotient(
            self.edge_types.astype(_u8, copy=False),
            list(type_codes),
            self._nV,
            self._boundary_ptr, self._boundary_idx,
        )

    # Relative homology

    def relative_cycle_basis(
        self, Q: dict,
    ) -> NDArray:
        """Basis for relative 1-cycles H_1(R, I).

        Computes an orthonormal basis for the harmonic subspace of the
        quotient edge Laplacian L1_quot.

        Parameters
        ----------
        Q : dict
            Result of self.quotient().

        Returns
        -------
        basis : f64[nE_quot, beta1_rel]
            Each column is a relative cycle generator on quotient edges.
        """
        return _quotient.relative_cycle_basis(Q['B1_quot'], Q['B2_quot'])

    def connecting_homomorphism(
        self,
        Q: dict,
        relative_cycle: NDArray,
        v_mask: NDArray,
        e_mask: NDArray,
    ) -> NDArray:
        """Apply the connecting homomorphism delta: H_1(R,I) -> H_0(I).

        Lifts a relative 1-cycle to the full edge space, applies B_1,
        and restricts to vertices in the subcomplex.

        Parameters
        ----------
        Q : dict
            Result of self.quotient().
        relative_cycle : f64[nE_quot]
            Coefficients of a relative cycle on quotient edges.
        v_mask : uint8[nV]
            Vertex mask for the subcomplex.
        e_mask : uint8[nE]
            Edge mask for the subcomplex.

        Returns
        -------
        boundary_in_I : f64[nV_I]
            Boundary restricted to subcomplex vertices.
        """
        return _quotient.connecting_homomorphism(
            self.B1,
            _asarray(v_mask, _u8),
            _asarray(e_mask, _u8),
            np.ascontiguousarray(relative_cycle, dtype=_f64),
            Q['e_reindex'],
        )

    # Signal restriction and lifting

    def restrict_signal(
        self, signal: NDArray, mask: NDArray,
    ) -> NDArray:
        """Restrict a real signal from the full complex to the quotient.

        Drops cells in the subcomplex, compacts surviving entries.

        Parameters
        ----------
        signal : f64[n]
            Signal on k-cells of the full complex.
        mask : uint8[n]
            Subcomplex mask (1 = in I, to be dropped).

        Returns
        -------
        f64[n_quot]
        """
        return _quotient.restrict_signal(
            np.ascontiguousarray(signal, dtype=_f64),
            _asarray(mask, _u8),
        )

    def lift_signal(
        self,
        signal_quot: NDArray,
        mask: NDArray,
        fill_value: float = 0.0,
    ) -> NDArray:
        """Lift a signal from the quotient to the full complex.

        Fills subcomplex cells with fill_value, copies survivors.

        Parameters
        ----------
        signal_quot : f64[n_quot]
        mask : uint8[n]
            Subcomplex mask (1 = in I).
        fill_value : float

        Returns
        -------
        f64[n]
        """
        return _quotient.lift_signal(
            np.ascontiguousarray(signal_quot, dtype=_f64),
            _asarray(mask, _u8),
            fill_value,
        )

    def restrict_field_state(
        self,
        f_E: NDArray,
        f_F: NDArray,
        e_mask: NDArray,
        f_mask: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Restrict an (E, F) field state to the quotient.

        Returns
        -------
        f_E_quot : f64[nE_quot]
        f_F_quot : f64[nF_quot]
        """
        return _quotient.restrict_field_state(
            np.ascontiguousarray(f_E, dtype=_f64),
            np.ascontiguousarray(f_F, dtype=_f64),
            _asarray(e_mask, _u8),
            _asarray(f_mask, _u8),
        )

    def lift_field_state(
        self,
        f_E_quot: NDArray,
        f_F_quot: NDArray,
        e_mask: NDArray,
        f_mask: NDArray,
        fill_value: float = 0.0,
    ) -> Tuple[NDArray, NDArray]:
        """Lift an (E, F) field state from the quotient to the full complex.

        Returns
        -------
        f_E : f64[nE]
        f_F : f64[nF]
        """
        return _quotient.lift_field_state(
            np.ascontiguousarray(f_E_quot, dtype=_f64),
            np.ascontiguousarray(f_F_quot, dtype=_f64),
            _asarray(e_mask, _u8),
            _asarray(f_mask, _u8),
            fill_value,
        )

    # Congruence classes (bulk)

    def congruence_classes(
        self, mask: NDArray, dim: int = 1,
    ) -> Tuple[NDArray, int]:
        """Partition surviving cells into congruence classes modulo I.

        Parameters
        ----------
        mask : uint8[nE] or uint8[nF]
            For dim=1: edge mask for the subcomplex I (1 = in I).
            For dim=2: face mask for the subcomplex I (1 = in I).
        dim : int
            1 for edge classes, 2 for face classes.

        Returns
        -------
        labels : i32[n]
            Class label for each cell, -1 for cells in I.
        n_classes : int
            Number of equivalence classes among survivors.
        """
        m = _asarray(mask, _u8)
        if dim == 1:
            return _quotient.congruence_classes_edges(self.B1, m)
        return _quotient.congruence_classes_faces(self.B2, m)

    # Quotient Hodge decomposition

    def quotient_hodge(
        self,
        Q: dict,
        signal: NDArray,
        e_mask: NDArray,
    ) -> dict:
        """Hodge decomposition on the quotient complex R/I.

        Restricts the signal to surviving edges, builds the quotient
        Laplacians from Q['B1_quot'] and Q['B2_quot'], and runs the
        full Hodge analysis via the Cython hodge pipeline.

        Parameters
        ----------
        Q : dict
            Result of self.quotient(). Must contain B1_quot, B2_quot,
            and L1_quot.
        signal : f64[nE]
            Edge signal on the full complex.
        e_mask : uint8[nE]
            Edge mask for the subcomplex.

        Returns
        -------
        dict
            Hodge analysis on the quotient edge space (same keys as
            hodge_full: grad, curl, harm, pct_grad, pct_curl, pct_harm,
            rho, divergence, div_norm, face_curl, orthogonality).
        """
        sig_q = _quotient.restrict_signal(
            np.ascontiguousarray(signal, dtype=_f64),
            _asarray(e_mask, _u8),
        )
        if sig_q.shape[0] == 0:
            return {
                'grad': np.zeros(0), 'curl': np.zeros(0),
                'harm': np.zeros(0),
                'pct_grad': 0.0, 'pct_curl': 0.0, 'pct_harm': 0.0,
            }

        B1q = Q['B1_quot']
        B2q = Q['B2_quot']

        # Build L0q = B1q @ B1q.T, L2q = B2q.T @ B2q
        L0q = B1q @ B1q.T
        nFq = B2q.shape[1] if B2q.ndim == 2 else 0
        L2q = B2q.T @ B2q if nFq > 0 else None

        # Build DualCSR representations for the _hodge path
        from rexgraph.core import _sparse
        B1q_dual = _sparse.from_dense_f64(np.ascontiguousarray(B1q, dtype=_f64))
        B2q_dual = None
        if nFq > 0:
            B2q_dual = _sparse.from_dense_f64(
                np.ascontiguousarray(B2q, dtype=_f64))

        return _hodge.build_hodge(B1q_dual, B2q_dual, sig_q, L0=L0q, L2=L2q)

    # Full quotient analysis (dashboard-ready)

    def quotient_analysis(
        self,
        e_mask: NDArray,
        signal: Optional[NDArray] = None,
    ) -> dict:
        """Complete quotient analysis for dashboard consumption.

        Runs the full pipeline: closure -> quotient construction ->
        relative Betti -> Hodge on R and R/I -> congruence classes ->
        spectral comparison.

        Parameters
        ----------
        e_mask : uint8[nE] or bool[nE]
            Edge mask for the subcomplex I (1 = in I).
        signal : f64[nE], optional
            Edge signal for Hodge comparison. Defaults to unit flow.

        Returns
        -------
        dict
            Complete quotient analysis including:
            - dims: (nVq, nEq, nFq)
            - betti_rel: (b0, b1, b2) relative
            - chain_ok: bool
            - hodge_R: Hodge percentages on full R
            - hodge_RI: Hodge percentages on R/I
            - congruence_labels, n_congruence_classes
            - evals_L1q, evals_RL1q (if available)
            - fiedler_RL1q
            - energy: E_kin, E_pot, ratio on quotient signal
            - surviving_edges: indices of edges not in I
        """
        em = _asarray(e_mask, _u8)

        # 1. Closure
        v_mask, e_mask_closed, f_mask = self.subcomplex(e_mask=em)

        # 2. Quotient construction
        Q = self.quotient(v_mask, e_mask_closed, f_mask)
        nVq, nEq, nFq = Q['dims']

        # 3. Signal
        if signal is None:
            signal = np.ones(self._nE, dtype=_f64)
        signal = np.ascontiguousarray(signal, dtype=_f64)

        # 4. Hodge on full R
        H_R = self.hodge_full(signal)

        # 5. Hodge on quotient R/I
        H_RI = self.quotient_hodge(Q, signal, e_mask_closed)

        # 6. Congruence classes (edges)
        cong_labels, n_classes = self.congruence_classes(e_mask_closed, dim=1)

        # 6b. Face congruence classes
        f_cong_labels = np.array([], dtype=_i32)
        n_f_classes = 0
        if self._nF > 0 and int(np.sum(f_mask)) > 0:
            try:
                f_cong_labels, n_f_classes = self.congruence_classes(
                    f_mask, dim=2)
            except Exception:
                pass

        # 7. Surviving edge indices
        surv = np.where(~np.asarray(e_mask_closed, dtype=bool))[0]

        # 8. Spectral data on quotient
        evals_L1q = np.array([])
        evals_RL1q = np.array([])
        fiedler_RL1q = 0.0
        fiedler_L1q = 0.0

        if nEq > 0:
            L1q = Q['L1_quot']
            evals_L1q = np.sort(np.linalg.eigvalsh(L1q))
            nz_L1 = evals_L1q[evals_L1q > 1e-10]
            fiedler_L1q = float(nz_L1[0]) if len(nz_L1) > 0 else 0.0

            RL1q = Q.get('RL1_quot')
            if RL1q is not None:
                evals_RL1q = np.sort(np.linalg.eigvalsh(RL1q))
                nz = evals_RL1q[evals_RL1q > 1e-10]
                fiedler_RL1q = float(nz[0]) if len(nz) > 0 else 0.0

        # 9. Energy on quotient signal
        E_kin_q, E_pot_q, E_ratio_q = 0.0, 0.0, 1.0
        if nEq > 0 and 'LO_quot' in Q:
            sig_q = _quotient.restrict_signal(signal, e_mask_closed)
            E_kin_q, E_pot_q, E_ratio_q = _quotient.quotient_energy_kin_pot(
                sig_q, Q['L1_quot'], Q['LO_quot'])

        # 10. Relative cycle basis
        rel_cycle_dim = 0
        try:
            rel_basis = self.relative_cycle_basis(Q)
            rel_cycle_dim = rel_basis.shape[1] if rel_basis.ndim == 2 else 0
        except Exception:
            pass

        # 11. Hodge orthogonality on quotient
        hodge_orthogonal = True
        if H_RI.get('orthogonality'):
            hodge_orthogonal = H_RI['orthogonality'].get('orthogonal', True)

        return {
            'dims': (nVq, nEq, nFq),
            'betti_rel': Q['betti_rel'],
            'chain_ok': Q['chain_valid'],
            'chain_error': Q['chain_error'],
            'v_star': Q['v_star'],
            'hodge_R': {
                'pct_grad': float(H_R['pct_grad']),
                'pct_curl': float(H_R['pct_curl']),
                'pct_harm': float(H_R['pct_harm']),
            },
            'hodge_RI': {
                'pct_grad': float(H_RI.get('pct_grad', 0)),
                'pct_curl': float(H_RI.get('pct_curl', 0)),
                'pct_harm': float(H_RI.get('pct_harm', 0)),
                'orthogonal': hodge_orthogonal,
            },
            'congruence_labels': cong_labels.tolist(),
            'n_congruence_classes': n_classes,
            'face_congruence_labels': f_cong_labels.tolist(),
            'n_face_congruence_classes': n_f_classes,
            'rel_cycle_dim': rel_cycle_dim,
            'surviving_edges': surv.tolist(),
            'evals_L1q': evals_L1q.tolist(),
            'evals_RL1q': evals_RL1q.tolist(),
            'fiedler_L1q': fiedler_L1q,
            'fiedler_RL1q': fiedler_RL1q,
            'energy': {
                'E_kin': float(E_kin_q),
                'E_pot': float(E_pot_q),
                'ratio': float(E_ratio_q),
            },
        }

    # Subgraph extraction

    def subgraph(
        self, edge_mask: NDArray,
    ) -> Tuple["RexGraph", NDArray, NDArray]:
        """Extract induced subgraph keeping only masked edges.

        Vertices are reindexed. Faces are kept only if all boundary
        edges survive. Returns a new RexGraph with consistent B1, B2,
        plus mapping arrays.

        Parameters
        ----------
        edge_mask : uint8[nE] or bool[nE]
            1 for edges to KEEP.

        Returns
        -------
        sub : RexGraph
            New graph with reindexed vertices, edges, faces.
        v_map : i32[nV_sub]
            Maps new vertex index -> old vertex index.
        e_map : i32[nE_sub]
            Maps new edge index -> old edge index.
        """
        keep = np.asarray(edge_mask, dtype=bool)
        src, tgt = self._ensure_src_tgt()

        # Surviving edges
        e_indices = np.where(keep)[0].astype(_i32)
        nE_sub = e_indices.shape[0]
        if nE_sub == 0:
            empty = RexGraph(sources=np.zeros(0, dtype=_i32),
                             targets=np.zeros(0, dtype=_i32))
            return empty, np.zeros(0, dtype=_i32), np.zeros(0, dtype=_i32)

        sub_src = src[e_indices]
        sub_tgt = tgt[e_indices]

        # Reindex vertices
        v_used = np.unique(np.concatenate([sub_src, sub_tgt]))
        v_map = v_used.astype(_i32)
        v_remap = np.full(self._nV, -1, dtype=_i32)
        for new_i, old_i in enumerate(v_used):
            v_remap[old_i] = new_i

        new_src = v_remap[sub_src]
        new_tgt = v_remap[sub_tgt]

        # Surviving faces: keep only faces whose ALL boundary edges survive
        edge_kept_set = set(e_indices.tolist())
        surviving_faces = []
        face_B2_cols = []

        if self._nF > 0:
            B2_full = self.B2
            cp, ri, vl = self._B2_col_ptr, self._B2_row_idx, self._B2_vals
            for f in range(self._nF):
                boundary_edges = [int(ri[j]) for j in range(cp[f], cp[f + 1])]
                if all(e in edge_kept_set for e in boundary_edges):
                    surviving_faces.append(f)

        # Build B2 for subgraph
        if surviving_faces:
            B2_full = self.B2
            nF_sub = len(surviving_faces)
            # Remap edge indices for B2
            e_remap = np.full(self._nE, -1, dtype=_i32)
            for new_j, old_j in enumerate(e_indices):
                e_remap[old_j] = new_j

            B2_sub = np.zeros((nE_sub, nF_sub), dtype=_f64)
            for fi_new, fi_old in enumerate(surviving_faces):
                for e_old in range(self._nE):
                    val = B2_full[e_old, fi_old]
                    if abs(val) > 1e-15 and e_remap[e_old] >= 0:
                        B2_sub[e_remap[e_old], fi_new] = val

            from scipy import sparse as sp
            B2_sp = sp.csc_matrix(B2_sub)
            sub = RexGraph(
                sources=new_src, targets=new_tgt,
                B2_col_ptr=np.asarray(B2_sp.indptr, dtype=_i32),
                B2_row_idx=np.asarray(B2_sp.indices, dtype=_i32),
                B2_vals=np.asarray(B2_sp.data, dtype=_f64),
                directed=self._directed,
            )
        else:
            sub = RexGraph(
                sources=new_src, targets=new_tgt,
                directed=self._directed,
            )

        return sub, v_map, e_indices.astype(_i32)

    # Community-based graph partitioning

    def partition_communities(
        self, max_size: int = 500,
    ) -> list:
        """Recursively partition the graph into renderable chunks.

        Uses Louvain communities from _standard. Each partition is
        small enough to render in a browser (max_size edges).

        Parameters
        ----------
        max_size : int
            Maximum edges per partition.

        Returns
        -------
        list of (sub_rex, v_map, e_map) tuples
            Each tuple is a subgraph with its vertex and edge maps
            back to the original indices.
        """
        if self._nE <= max_size:
            return [(self,
                     np.arange(self._nV, dtype=_i32),
                     np.arange(self._nE, dtype=_i32))]

        # Compute Louvain communities
        adj_ptr, adj_idx, adj_edge = self._adjacency_bundle
        e_wt = np.ones(self._nE, dtype=_f64)
        adj_wt = _standard.build_adj_weights(adj_edge, e_wt)
        metrics = _standard.build_standard_metrics(
            adj_ptr, adj_idx, adj_edge, adj_wt, self._nV, self._nE)
        labels = metrics['community_labels']
        n_comm = metrics['n_communities']

        if n_comm <= 1:
            # Louvain didn't split - return whole graph
            return [(self,
                     np.arange(self._nV, dtype=_i32),
                     np.arange(self._nE, dtype=_i32))]

        src, tgt = self._ensure_src_tgt()
        pieces = []

        for c in range(n_comm):
            # Keep edges whose BOTH endpoints are in community c
            e_keep = np.array(
                [labels[src[e]] == c and labels[tgt[e]] == c
                 for e in range(self._nE)],
                dtype=bool,
            )
            if not np.any(e_keep):
                continue

            sub, v_map, e_map = self.subgraph(e_keep)
            if sub.nE > max_size:
                # Recurse into large communities
                sub_pieces = sub.partition_communities(max_size)
                for sub_sub, sv_map, se_map in sub_pieces:
                    pieces.append((
                        sub_sub,
                        v_map[sv_map],
                        e_map[se_map],
                    ))
            else:
                pieces.append((sub, v_map, e_map))

        return pieces

    # Signal dashboard precomputation

    def signal_dashboard_data(
        self,
        *,
        probe_edges: Optional[Sequence[int]] = None,
        times: Optional[NDArray] = None,
        n_steps: int = 50,
        t_max: float = 10.0,
    ) -> dict:
        """Precompute all data needed by the signal dashboard template.

        No JS math: all trajectories, Hodge decompositions, BIOES tags,
        mode classifications, and energy trajectories are computed here.

        Parameters
        ----------
        probe_edges : list of int, optional
            Edge indices to run perturbation from. Defaults to the
            highest-energy edge.
        times : f64[T], optional
            Timepoints. Auto-generated if None.
        n_steps, t_max : int, float
            Steps and max time for auto-generated timepoints.

        Returns
        -------
        dict
            Signal dashboard data contract with keys:
            - probes: dict of probe_edge -> perturbation results
            - field: field operator analysis
            - mode_classification: edge/face/coupled mode labels
            - evals_RL1, evecs_RL1: eigendecomposition
        """
        if times is None:
            times = np.linspace(0, t_max, n_steps, dtype=_f64)
        else:
            times = np.ascontiguousarray(times, dtype=_f64)

        # Default probes
        if probe_edges is None:
            flow = np.ones(self._nE, dtype=_f64)
            E_kin_per, E_pot_per = self.per_edge_energy(flow)
            total = E_kin_per + E_pot_per
            probe_edges = [int(np.argmax(total))]

        probes = {}
        for pe in probe_edges:
            f_E, f_F = self.edge_perturbation(pe)
            result = self.analyze_perturbation(f_E, f_F, times=times)

            # Extract trajectory summary (sparse: top-k per timestep)
            traj = result.get('trajectory')
            traj_summary = None
            if traj is not None:
                # Keep per-edge max signal over time
                max_signal = np.max(np.abs(traj), axis=0)
                top_edges = np.argsort(-max_signal)[:min(20, self._nE)]
                traj_summary = {
                    'top_edges': top_edges.tolist(),
                    'top_values': traj[:, top_edges].tolist()
                    if traj.shape[0] <= 100
                    else traj[::max(1, traj.shape[0] // 50), :][:, top_edges].tolist(),
                }

            probes[pe] = {
                'E_kin': result.get('E_kin', np.array([])).tolist(),
                'E_pot': result.get('E_pot', np.array([])).tolist(),
                'ratio': result.get('ratio', np.array([])).tolist(),
                'bioes_tags': result.get('bioes_tags', np.array([])).tolist(),
                'hodge_initial': _serialize_hodge_dict(
                    result.get('hodge_initial', {})),
                'hodge_final': _serialize_hodge_dict(
                    result.get('hodge_final', {})),
                'trajectory_summary': traj_summary,
                'cascade': {
                    'activation_time': result.get('activation_time', np.array([])).tolist(),
                    'activation_order': result.get('activation_order', np.array([])).tolist(),
                },
            }

        # Field operator data
        field_data = {}
        nF_hodge = self.nF_hodge
        if nF_hodge > 0:
            try:
                M, g_field, is_psd = self.field_operator
                f_evals, f_evecs, f_freqs = self.field_eigen
                mode_data = self.classify_modes()

                # Field diffusion from unit edge signal
                F0 = np.zeros(self._nE + nF_hodge, dtype=_f64)
                F0[:self._nE] = 1.0 / max(np.sqrt(self._nE), 1.0)
                diff_traj = self.field_diffuse(F0, times)

                norm_E = np.array([float(np.linalg.norm(diff_traj[t, :self._nE]))
                                   for t in range(len(times))])
                norm_F = np.array([float(np.linalg.norm(diff_traj[t, self._nE:]))
                                   for t in range(len(times))])

                field_data = {
                    'coupling_g': float(g_field),
                    'is_psd': bool(is_psd),
                    'evals': f_evals.tolist(),
                    'freqs': f_freqs.tolist(),
                    'mode_types': mode_data.get('mode_type', np.array([])).tolist(),
                    'edge_weights': mode_data.get('edge_weight', np.array([])).tolist(),
                    'face_weights': mode_data.get('face_weight', np.array([])).tolist(),
                    'diffusion_norm_E': norm_E.tolist(),
                    'diffusion_norm_F': norm_F.tolist(),
                }
            except Exception:
                pass

        # RL1 eigendata
        evals_rl = self.evals_RL1
        evecs_rl = self.evecs_RL1

        return {
            'probes': probes,
            'field': field_data,
            'times': times.tolist(),
            'evals_RL1': evals_rl.tolist() if evals_rl is not None else [],
            'alpha_G': float(self.alpha_G) if self.alpha_G == self.alpha_G else 0.0,
        }

    # Quotient dashboard precomputation

    def quotient_dashboard_data(
        self,
        *,
        vertex_labels: Optional[Sequence[str]] = None,
        edge_types_str: Optional[Sequence[str]] = None,
        signal: Optional[NDArray] = None,
        max_vertex_presets: int = 8,
    ) -> dict:
        """Precompute quotient presets for the dashboard template.

        Generates quotient analyses for common subcomplexes:
        star of top-degree vertices, by edge type, by energy regime.

        Parameters
        ----------
        vertex_labels : list of str, optional
            Vertex names for labelling presets.
        edge_types_str : list of str, optional
            String edge type labels for type-based presets.
        signal : f64[nE], optional
            Signal for Hodge comparison.
        max_vertex_presets : int
            Max number of vertex star presets.

        Returns
        -------
        dict
            Quotient dashboard data contract with keys:
            - presets: dict of preset_name -> quotient_analysis result
            - full_betti: (b0, b1, b2) of full complex
            - full_hodge: Hodge percentages of full complex
        """
        if signal is None:
            signal = np.ones(self._nE, dtype=_f64)
        signal = np.ascontiguousarray(signal, dtype=_f64)

        if vertex_labels is None:
            vertex_labels = [f"v{i}" for i in range(self._nV)]

        presets = {}

        # Star of top-degree vertices
        deg = self.degree
        top_verts = np.argsort(-deg)[:min(max_vertex_presets, self._nV)]
        src, tgt = self._ensure_src_tgt()

        for vi in top_verts:
            vi = int(vi)
            vm, em, fm = self.star_of_vertex(vi)
            try:
                qa = self.quotient_analysis(em, signal=signal)
                qa['label'] = f"Star({vertex_labels[vi]})"
                presets[f"star_v{vi}"] = qa
            except Exception:
                pass

        # By edge type (string labels)
        if edge_types_str is not None:
            unique_types = sorted(set(edge_types_str))
            for t in unique_types:
                em = np.array(
                    [1 if edge_types_str[j] == t else 0 for j in range(self._nE)],
                    dtype=_u8,
                )
                if np.sum(em) == 0 or np.sum(em) == self._nE:
                    continue
                try:
                    qa = self.quotient_analysis(em, signal=signal)
                    qa['label'] = f"Type: {t}"
                    presets[f"type_{t}"] = qa
                except Exception:
                    pass

        # Star of top-betweenness edges (max 4)
        try:
            E_kin_per, E_pot_per = self.per_edge_energy(signal)
            total_energy = E_kin_per + E_pot_per
            top_energy_edges = np.argsort(-total_energy)[
                :min(4, self._nE)]
            for ei in top_energy_edges:
                ei = int(ei)
                vm, em, fm = self.star_of_edge(ei)
                try:
                    qa = self.quotient_analysis(em, signal=signal)
                    qa['label'] = f"Star(e{ei + 1})"
                    presets[f"star_e{ei}"] = qa
                except Exception:
                    pass
        except Exception:
            pass

        # By energy regime
        try:
            E_kin_per, E_pot_per = self.per_edge_energy(signal)
            for regime, name in [(0, "kinetic"), (1, "crossover"), (2, "potential")]:
                vm, em, fm = self.subcomplex_by_energy(signal, regime)
                n_in = int(np.sum(em))
                if 0 < n_in < self._nE:
                    try:
                        qa = self.quotient_analysis(em, signal=signal)
                        qa['label'] = f"Energy: {name} ({n_in} edges)"
                        presets[f"energy_{name}"] = qa
                    except Exception:
                        pass
        except Exception:
            pass

        # Full complex reference data
        H_full = self.hodge_full(signal)

        return {
            'presets': presets,
            'full_betti': list(self.betti),
            'full_hodge': {
                'pct_grad': float(H_full['pct_grad']),
                'pct_curl': float(H_full['pct_curl']),
                'pct_harm': float(H_full['pct_harm']),
            },
            'alpha_G': float(self.alpha_G) if self.alpha_G == self.alpha_G else 0.0,
            'nV': self._nV,
            'nE': self._nE,
            'nF': self._nF,
        }

    # Persistent homology

    def filtration(
        self,
        kind: str,
        *,
        signal: Optional[NDArray] = None,
        positions: Optional[NDArray] = None,
        eigenvector: Optional[NDArray] = None,
        component: int = 2,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        v2e_ptr, v2e_idx = self._v2e
        bp, bi = self._boundary_ptr, self._boundary_idx
        b2cp, b2ri = self._B2_col_ptr, self._B2_row_idx

        dispatch = {
            "vertex_sublevel": lambda: _persistence.filtration_sublevel_vertex(signal, bp, bi, b2cp, b2ri),
            "edge_sublevel": lambda: _persistence.filtration_sublevel_edge(signal, self._nV, v2e_ptr, v2e_idx, b2cp, b2ri),
            "face_sublevel": lambda: _persistence.filtration_sublevel_face(signal, self._nV, self._nE, bp, bi, b2cp, b2ri),
            "spectral": lambda: _persistence.filtration_spectral(eigenvector, self._nV, bp, bi, v2e_ptr, v2e_idx, b2cp, b2ri),
            "rips": lambda: _persistence.filtration_rips(positions, bp, bi, b2cp, b2ri),
            "dimension": lambda: _persistence.filtration_dimension(self._nV, self._nE, self._nF),
        }
        if kind == "hodge":
            grad, curl, harm = self.hodge(signal)
            return _persistence.filtration_hodge_component(
                grad, curl, harm, self._nV, v2e_ptr, v2e_idx, b2cp, b2ri, component,
            )
        if kind in dispatch:
            return dispatch[kind]()
        raise ValueError(f"Unknown filtration kind: {kind}")

    def persistence(self, filt_v: NDArray, filt_e: NDArray, filt_f: NDArray) -> dict:
        self._ensure_clean()
        return _persistence.persistence_diagram(
            filt_v, filt_e, filt_f,
            self._boundary_ptr, self._boundary_idx,
            self._B2_col_ptr, self._B2_row_idx,
        )

    def persistence_barcodes(self, result: dict, dim: int = -1) -> NDArray:
        return _persistence.persistence_barcodes(result["pairs"], result["essential"], dim)

    def persistence_landscape(self, barcodes: NDArray, grid: NDArray, k_max: int = 5) -> NDArray:
        return _persistence.persistence_landscape(barcodes, grid, k_max)

    @staticmethod
    def persistence_distance(dgm1: NDArray, dgm2: NDArray, metric: str = "bottleneck", p: float = 2.0) -> float:
        if metric == "bottleneck":
            return _persistence.bottleneck_distance(dgm1, dgm2)
        return _persistence.wasserstein_distance(dgm1, dgm2, p)

    def persistence_entropy(self, barcodes: NDArray) -> float:
        return _persistence.persistence_entropy(barcodes)

    def enrich_persistence(self, result: dict) -> dict:
        pairs = result["pairs"]
        edge_ann = _persistence.enrich_pairs_edge_type(pairs, self.edge_types)
        grad, curl, harm = self.hodge(np.ones(self._nE))
        dom, frac = _persistence.enrich_pairs_hodge(pairs, grad ** 2, curl ** 2, harm ** 2)
        return {"edge_type_annotations": edge_ann, "hodge_dominant": dom, "hodge_fractions": frac}

    # Mutation (returns new RexGraph via Cython)

    def insert_edges(
        self,
        new_sources: ArrayLike,
        new_targets: ArrayLike,
    ) -> RexGraph:
        """Insert standard edges and return a new RexGraph.

        The vertex set is expanded per the lifecycle contract.
        """
        src, tgt = self._ensure_src_tgt()
        ns = _asarray(new_sources, _i32)
        nt = _asarray(new_targets, _i32)

        new_src, new_tgt, nV_new = _rex.insert_edges(
            self._nV, self._nE, src, tgt, ns, nt,
        )
        return RexGraph(
            sources=new_src,
            targets=new_tgt,
            w_E=self._w_E,  # note: does not extend w_E for new edges
            directed=self._directed,
        )

    def delete_edges(self, mask: NDArray) -> RexGraph:
        """Delete edges where mask is nonzero and return a new RexGraph.

        Vertices with no remaining incident edges are removed per the
        lifecycle contract. Returns remapped arrays.
        """
        src, tgt = self._ensure_src_tgt()
        delete_mask = _asarray(mask, _i32)

        new_src, new_tgt, nV_new, v_map, e_map = _rex.delete_edges(
            self._nV, self._nE, src, tgt, delete_mask,
        )
        return RexGraph(
            sources=new_src,
            targets=new_tgt,
            directed=self._directed,
        )

    # Serialization

    def to_json(self) -> dict:
        self._ensure_clean()
        d = {
            "nV": self._nV, "nE": self._nE, "nF": self._nF,
            "dimension": self.dimension,
            "boundary_ptr": self._boundary_ptr.tolist(),
            "boundary_idx": self._boundary_idx.tolist(),
            "edge_types": self.edge_types.tolist(),
            "betti": list(self.betti),
            "euler_characteristic": self.euler_characteristic,
            # Directedness is part of the signed complex, not a derived quantity.
            "directed": bool(self._directed),
        }
        if self._nV > 0 and self._nE > 0:
            d["layout"] = self.layout.tolist()
        if self._nF > 0:
            d["B2_col_ptr"] = self._B2_col_ptr.tolist()
            d["B2_row_idx"] = self._B2_row_idx.tolist()
            # Face orientation signs: without these the loader fabricates ones
            # and silently discards the signed 2-boundary.
            d["B2_vals"] = self._B2_vals.tolist()
        if self._w_E is not None:
            d["w_E"] = self._w_E.tolist()
        if self._signs is not None:
            d["signs"] = np.asarray(self._signs, dtype=_f64).tolist()
        if self._w_boundary:
            # Tuple keys cannot be JSON object keys; store as [key_list, value]
            # pairs so the full (edge, boundary-point) attribution survives.
            d["w_boundary"] = [
                [
                    list(k) if isinstance(k, tuple) else [k],
                    v.tolist() if isinstance(v, np.ndarray)
                    else (float(v) if isinstance(v, np.floating) else v),
                ]
                for k, v in self._w_boundary.items()
            ]
        return d

    def to_dict(self) -> dict:
        self._ensure_clean()
        return {
            "boundary_ptr": self._boundary_ptr,
            "boundary_idx": self._boundary_idx,
            "B2_col_ptr": self._B2_col_ptr,
            "B2_row_idx": self._B2_row_idx,
            "B2_vals": self._B2_vals,
            "w_E": self._w_E,
            "w_boundary": self._w_boundary,
            "directed": self._directed,
            "signs": self._signs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RexGraph:
        return cls(
            boundary_ptr=d["boundary_ptr"],
            boundary_idx=d["boundary_idx"],
            B2_col_ptr=d.get("B2_col_ptr"),
            B2_row_idx=d.get("B2_row_idx"),
            B2_vals=d.get("B2_vals"),
            w_E=d.get("w_E"),
            w_boundary=d.get("w_boundary", {}),
            directed=d.get("directed", False),
            signs=d.get("signs"),
        )

    # New additions: shape, metadata, signatures, operators, ontology

    def attach_metadata(self, dim: int, idx: int, key: str, value) -> None:
        """Attach arbitrary metadata to a cell."""
        if not hasattr(self, '_cell_metadata'):
            self._cell_metadata = {0: {}, 1: {}, 2: {}}
        self._cell_metadata.setdefault(dim, {}).setdefault(idx, {})[key] = value

    def get_metadata(self, dim: int, idx: int, key: str = None):
        """Retrieve cell metadata."""
        if not hasattr(self, '_cell_metadata'):
            return {} if key is None else None
        cell_md = self._cell_metadata.get(dim, {}).get(idx, {})
        return cell_md if key is None else cell_md.get(key)

    def edge_signature(self, edge_idx: int) -> tuple:
        """Hashable algebraic signature for an edge on Delta^2."""
        chi = self.structural_character[edge_idx]
        return (round(float(chi[0]), 6), round(float(chi[1]+chi[3]), 6), round(float(chi[2]), 6))

    def group_edges_by_signature(self) -> dict:
        """Group edges into equivalence classes by algebraic signature."""
        groups = {}
        for e in range(self.nE):
            sig = self.edge_signature(e)
            groups.setdefault(sig, []).append(e)
        return groups

    def subcomplex_by_criteria(self, criteria: dict) -> "RexGraph":
        """Build a subcomplex from criteria on cell metadata."""
        if not hasattr(self, '_cell_metadata'):
            return self
        edge_md = self._cell_metadata.get(1, {})
        keep = []
        for e in range(self.nE):
            md = edge_md.get(e, {})
            match = True
            for key, expected in criteria.items():
                if key not in md:
                    match = False; break
                actual = md[key]
                if isinstance(expected, dict):
                    if 'min' in expected and actual < expected['min']: match = False; break
                    if 'max' in expected and actual > expected['max']: match = False; break
                elif isinstance(expected, list):
                    if actual not in expected: match = False; break
                else:
                    if actual != expected: match = False; break
            if match:
                keep.append(e)
        if not keep:
            return RexGraph.from_simplicial(np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros((0,3), dtype=np.int32))
        e_mask = np.zeros(self.nE, dtype=bool)
        for e in keep: e_mask[e] = True
        v_mask, em, f_mask = self.subcomplex(e_mask=e_mask)
        # Rebuild from masked sources/targets
        kept_edges = np.where(em)[0]
        src_kept = self.sources[kept_edges]
        tgt_kept = self.targets[kept_edges]
        # Remap vertex indices to consecutive
        old_verts = np.where(v_mask)[0]
        vmap = {int(old_verts[i]): i for i in range(len(old_verts))}
        new_src = np.array([vmap[int(s)] for s in src_kept], dtype=np.int32)
        new_tgt = np.array([vmap[int(t)] for t in tgt_kept], dtype=np.int32)
        # Rebuild faces from kept edges
        kept_faces = np.where(f_mask)[0] if f_mask.any() else np.array([], dtype=np.int64)
        new_tri = np.zeros((0, 3), dtype=np.int32)
        if len(kept_faces) > 0:
            B2 = self.B2_dense
            face_list = []
            for f_idx in kept_faces:
                col = B2[:, f_idx]
                f_edges = np.where(col != 0)[0]
                verts = set()
                for fe in f_edges:
                    verts.add(int(self.sources[fe]))
                    verts.add(int(self.targets[fe]))
                if len(verts) == 3 and all(v in vmap for v in verts):
                    face_list.append(tuple(sorted(vmap[v] for v in verts)))
            if face_list:
                new_tri = np.array(face_list, dtype=np.int32)
        return RexGraph.from_simplicial(new_src, new_tgt, new_tri)

    def operator_distance(self, other: "RexGraph", metric: str = 'frobenius') -> float:
        """Distance between RL_4 operators of two complexes."""
        RL_a, RL_b = self.relational_laplacian, other.relational_laplacian
        if RL_a.shape != RL_b.shape:
            raise ValueError(f"Shape mismatch: {RL_a.shape} vs {RL_b.shape}")
        diff = RL_a - RL_b
        if metric == 'frobenius': return float(np.linalg.norm(diff, ord='fro'))
        elif metric == 'spectral':
            return _spectral_distance(_ensure_dense(RL_a), _ensure_dense(RL_b))
        raise ValueError(f"Unknown metric: {metric}")

    def modulated_channel(self, channel_idx: int, schedule_fn, t: float):
        """Return lambda(t) * hat_X for time-dependent channel modulation."""
        # hat_X is already available as the trace-normalized channel operator;
        # use it directly instead of a dense V diag(lambda) V^T reconstruction.
        # Scale-free path uses the sparse channel hats (densified for the caller).
        if self._use_sparse_character:
            hat_X = self._sparse_character['hats'][channel_idx]
        else:
            hat_X = self._rcf_bundle['hats'][channel_idx]
        return float(schedule_fn(t)) * _ensure_dense(hat_X)

    def cell_shape(self, dim: int, idx: int) -> dict:
        """Full algebraic shape of a cell."""
        result = {'dim': dim, 'idx': idx, 'betti': list(self.betti)}
        if dim == 1:
            below, above, lateral = self.hyperslice(dim, idx)
            result['below'] = below; result['above'] = above; result['lateral'] = lateral
            chi = self.structural_character[idx]
            result['chi'] = {'T': float(chi[0]), 'GC': float(chi[1]+chi[3]), 'F': float(chi[2])}
            result['signature'] = self.edge_signature(idx)
        elif dim == 0 and idx < self.nV:
            result['kappa'] = float(self.coherence[idx])
            result['phi'] = self.vertex_character[idx].tolist()
        elif dim == 2 and self.nF > 0 and idx < self.nF:
            curv = self.rcfe_curvature
            result['curvature'] = float(curv[idx]) if idx < len(curv) else 0.0
        return result

    def shape_tensor(self) -> dict:
        """Complex-level shape: the L_gb 4x4 tensor."""
        try: return self.l_gb_tensor
        except Exception: return {'error': 'L_gb not available'}

    @classmethod
    def from_ontology(cls, terms: list, subsumption: list) -> "RexGraph":
        """Build a rex from ontology terms (edges) and subsumption (faces)."""
        if not terms:
            return cls.from_simplicial(np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros((0,3), dtype=np.int32))
        src = np.array([t[0] for t in terms], dtype=np.int32)
        tgt = np.array([t[1] for t in terms], dtype=np.int32)
        tri = np.array(subsumption, dtype=np.int32) if subsumption else np.zeros((0,3), dtype=np.int32)
        return cls.from_simplicial(src, tgt, tri)

    def malaugh_extend(self, new_src, new_tgt, new_faces=None):
        """Extend the complex (Malaugh derivative). Returns (new_rex, edge_map)."""
        old_src = np.asarray(self.sources, dtype=np.int32)
        old_tgt = np.asarray(self.targets, dtype=np.int32)
        combined_src = np.concatenate([old_src, np.asarray(new_src, dtype=np.int32)])
        combined_tgt = np.concatenate([old_tgt, np.asarray(new_tgt, dtype=np.int32)])
        edge_map = {e: e for e in range(old_src.shape[0])}
        # Get existing faces from B2
        B2 = self.B2_dense
        old_face_list = []
        if self.nF > 0:
            for f in range(self.nF):
                col = B2[:, f]
                edges_in_face = np.where(col != 0)[0]
                if len(edges_in_face) == 3:
                    verts = set()
                    for e in edges_in_face:
                        verts.add(int(old_src[e]))
                        verts.add(int(old_tgt[e]))
                    if len(verts) == 3:
                        old_face_list.append(tuple(sorted(verts)))
        old_faces = np.array(old_face_list, dtype=np.int32) if old_face_list else np.zeros((0,3), dtype=np.int32)
        if new_faces is not None:
            nf = np.asarray(new_faces, dtype=np.int32).reshape(-1,3)
            combined_faces = np.concatenate([old_faces, nf]) if old_faces.shape[0]>0 else nf
        else:
            combined_faces = old_faces
        new_rex = RexGraph.from_simplicial(combined_src, combined_tgt, combined_faces)
        chain_err = np.max(np.abs(new_rex.B1_dense @ new_rex.B2_dense))
        if chain_err > 1e-10: raise ValueError(f"Chain condition violated: {chain_err}")
        return new_rex, edge_map

    def __repr__(self) -> str:
        return f"RexGraph(nV={self._nV}, nE={self._nE}, nF={self._nF}, dim={self.dimension})"


# construct via make_edge_delta() so `directed` is stamped correctly; do not build raw from the kernel's 8-tuple
TemporalDelta = namedtuple(
    "TemporalDelta",
    "born_cols born_offsets born_wE born_signs died_keys mod_keys mod_wE mod_signs directed",
    defaults=(False,),
)


def make_edge_delta(prev_ptr, prev_idx, prev_wE, prev_signs,
                    curr_ptr, curr_idx, curr_wE, curr_signs, directed=False):
    """Build a TemporalDelta from two cell-states, stamping `directed` so the record
    faithfully carries the key-encoding scheme its keys were computed with. Always
    construct edge deltas through this helper, never `TemporalDelta(*kernel_return)`
    directly (the kernel returns 8 arrays and does not carry `directed`, so a raw
    construction would default `directed` to False and mis-key a directed delta on
    replay)."""
    from rexgraph.core._temporal import encode_delta_full
    arrays = encode_delta_full(prev_ptr, prev_idx, prev_wE, prev_signs,
                               curr_ptr, curr_idx, curr_wE, curr_signs, directed)
    return TemporalDelta(*arrays, directed=directed)


def _cell_state(rex):
    """Read a RexGraph's boundary CSR + attribution (materializing pending mutations).
    Returns (boundary_ptr, boundary_idx, w_E_or_None, signs_or_None)."""
    rex._ensure_clean()
    return rex._boundary_ptr, rex._boundary_idx, rex._w_E, rex._signs


# construct via make_face_delta() so directed is stamped; do not build raw from the kernel tuple
# a face's identity is the order-independent hash of its constituent edges'
# canonical keys, not its raw boundary-vertex encoding (see track_faces/
# face_lifecycle for the exact/Jaccard boundary-vertex identity used there)
FaceDelta = namedtuple(
    "FaceDelta",
    "born_edge_keys born_offsets born_signs died_face_keys directed",
    defaults=(False,),
)


def _face_state(rex):
    """(B2_col_ptr, B2_row_idx, B2_vals, edge_keys) with edge_keys[e] the canonical key
    of edge e, so a face column's row indices map to stable edge identities."""
    rex._ensure_clean()
    from rexgraph.core._temporal import cell_keys_of
    keys = cell_keys_of(rex._boundary_ptr, rex._boundary_idx, rex._directed)
    return rex._B2_col_ptr, rex._B2_row_idx, rex._B2_vals, keys


def make_face_delta(prev_face_state, curr_face_state, directed=False):
    """Build a FaceDelta, stamping `directed` so the record carries the edge-key
    scheme its keys were computed with (needed by apply_face_delta on replay to
    recompute the live complex's edge keys with the matching scheme). Always
    construct face deltas through this helper, never `FaceDelta(*kernel_return)`."""
    from rexgraph.core._temporal import encode_face_delta
    arrays = encode_face_delta(prev_face_state, curr_face_state, directed)
    return FaceDelta(*arrays, directed=directed)


# FNV-1a constants mirroring _temporal.pyx's _face_key_from_buf, so face_key_of
# below hashes identically to encode_face_delta without needing a Cython rebuild.
_FNV_OFFSET_64 = 1469598103934665603
_FNV_PRIME_64 = 1099511628211
_MASK_64 = (1 << 64) - 1


def _to_signed_i64(u):
    """Reinterpret a uint64 bit pattern (Python int, already masked to 64 bits)
    as a signed int64, matching C's `<i64><unsigned long long>` cast."""
    return u - (1 << 64) if u >= (1 << 63) else u


def face_key_of_keys(edge_keys):
    """Order-independent int64 hash of a single face's constituent edge-keys
    (FNV-1a over the sorted i64 edge keys). Mirrors the arity != 2 branch of
    _cell_key_i32/_cell_key_i64 (_temporal.pyx's _face_key_from_buf) so a
    face's identity is computed with the same scheme as a cell's identity,
    just over already-resolved edge keys instead of raw vertex ids. Factored
    out of `face_key_of` so both the per-face-column path and a raw
    edge-key-array path (reconstruct_at's key-level replay) share one hash
    implementation instead of duplicating it."""
    keys = sorted(int(k) for k in edge_keys)
    h = _FNV_OFFSET_64
    for k in keys:
        h = (h ^ (k & _MASK_64)) & _MASK_64
        h = (h * _FNV_PRIME_64) & _MASK_64
    return _to_signed_i64(h)


def face_key_of(B2_col_ptr, B2_row_idx, edge_keys, directed=False):
    """Order-independent int64 hash of each face's constituent edge-keys.

    Pure-Python/numpy mirror of `_temporal._face_key_from_buf` (FNV-1a over the
    sorted i64 edge keys), so a live rex's face keys match the keys a FaceDelta
    was built with by `encode_face_delta`. No Cython counterpart is added in
    this task (avoids a rebuild); this is the canonical implementation.

    `directed` is accepted only for interface symmetry with `cell_keys_of` and
    `encode_face_delta`: `edge_keys` already carries the directedness it was
    computed with (from `cell_keys_of(..., directed)`), so it is not used
    again here.
    """
    ptr = np.asarray(B2_col_ptr)
    idx = np.asarray(B2_row_idx)
    ek = np.asarray(edge_keys)
    nF = int(ptr.shape[0] - 1)
    out = np.empty(nF, dtype=_i64)
    for f in range(nF):
        cols = idx[int(ptr[f]):int(ptr[f + 1])]
        out[f] = face_key_of_keys(ek[int(c)] for c in cols)
    return out


def apply_edge_delta(rex, delta):
    """Fold a TemporalDelta onto a live RexGraph via in-place mutators (O(delta)).

    `delta.directed` selects the same key-encoding scheme the delta's own keys
    were built with, so the live rex's recomputed keys line up with died_keys/
    mod_keys."""
    from rexgraph.core._temporal import cell_keys_of
    # 1. died: mask current cells whose key is in delta.died_keys
    if delta.died_keys.shape[0]:
        rex._ensure_clean()
        cur_keys = cell_keys_of(rex._boundary_ptr, rex._boundary_idx, delta.directed)
        died = np.isin(cur_keys, delta.died_keys)
        if died.any():
            rex.remove_edges(died.astype(_i32))
    # 2. born: split by arity; arity-2 via add_edges, else add_hyperedges
    n_born = int(delta.born_offsets.shape[0] - 1)
    if n_born:
        cols = [delta.born_cols[delta.born_offsets[i]:delta.born_offsets[i + 1]]
                for i in range(n_born)]
        arity2 = [i for i, c in enumerate(cols) if c.shape[0] == 2]
        other = [i for i in range(n_born) if cols[i].shape[0] != 2]
        if arity2:
            src = np.array([cols[i][0] for i in arity2], dtype=_i32)
            tgt = np.array([cols[i][1] for i in arity2], dtype=_i32)
            rex.add_edges(src, tgt,
                          w_E=np.asarray(delta.born_wE)[arity2],
                          signs=np.asarray(delta.born_signs)[arity2])
        if other:
            rex.add_hyperedges([cols[i] for i in other],
                               w_E=np.asarray(delta.born_wE)[other],
                               signs=np.asarray(delta.born_signs)[other])
    # 3. modified: resolve keys to current indices, set attrs. A mod_key is only
    # ever emitted for a cell present in BOTH prev and curr (a persisting cell),
    # so in a correctly sequenced replay every mod_key MUST resolve; fail loud
    # rather than silently reconstruct the wrong state.
    if delta.mod_keys.shape[0]:
        rex._ensure_clean()
        cur_keys = cell_keys_of(rex._boundary_ptr, rex._boundary_idx, delta.directed)
        pos = {int(k): i for i, k in enumerate(cur_keys)}
        try:
            idx = np.array([pos[int(k)] for k in delta.mod_keys], dtype=_i32)
        except KeyError as e:
            raise ValueError(
                "apply_edge_delta: modified-cell key %s not present in the live "
                "complex; a persisting cell must resolve, so the delta was applied "
                "out of order or onto the wrong base state" % e.args[0])
        rex.set_cell_attrs(idx,
                           w_E=np.asarray(delta.mod_wE),
                           signs=np.asarray(delta.mod_signs))


def apply_face_delta(rex, fdelta):
    """Fold a FaceDelta onto a live RexGraph. Resolve born-face edge-keys to
    current edge indices, add_faces; build the removal mask from
    died_face_keys. Apply AFTER apply_edge_delta for the same step: a face's
    edge-keys only resolve once the edge deltas for that step have landed
    (reconstruct_at guarantees this ordering)."""
    from rexgraph.core._temporal import cell_keys_of
    rex._ensure_clean()
    cur_keys = cell_keys_of(rex._boundary_ptr, rex._boundary_idx, fdelta.directed)
    pos = {int(k): i for i, k in enumerate(cur_keys)}
    # died faces: mask current faces whose face-key is in died_face_keys
    if fdelta.died_face_keys.shape[0] and rex._nF:
        cur_face_keys = face_key_of(rex._B2_col_ptr, rex._B2_row_idx, cur_keys, fdelta.directed)
        died = np.isin(cur_face_keys, fdelta.died_face_keys)
        if died.any():
            rex.remove_faces(died.astype(_i32))
    # born faces: resolve constituent edge-keys to current indices
    n_born = int(fdelta.born_offsets.shape[0] - 1)
    if n_born:
        face_edges, face_signs = [], []
        for i in range(n_born):
            ks = fdelta.born_edge_keys[fdelta.born_offsets[i]:fdelta.born_offsets[i + 1]]
            eidx = np.array([pos[int(k)] for k in ks], dtype=_i32)
            face_edges.append(eidx)
            face_signs.append(fdelta.born_signs[fdelta.born_offsets[i]:fdelta.born_offsets[i + 1]])
        rex.add_faces(face_edges, face_signs)


class TemporalRex:
    """A temporal rexgraph Gamma = (R(t_0), ..., R(t_T)).

    A sequence of rex snapshots sharing continuous identity.
    """

    __slots__ = (
        "__dict__",
        "_snapshots",
        "_face_snapshots",
        "_directed",
        "_general",
        "_T",
        "_index_checkpoints",
        "_index_deltas",
        "_index_face_deltas",
        "_index_cp_times",
        "_cumulative_delta",
        "_last_state",
        "_last_face_state",
        "_encoding",
        "_snapshots_materialized",
        "_checkpoint_threshold",
    )

    # adaptive checkpoint threshold: store a full checkpoint once cumulative
    # born+died / current edge count exceeds this, mirroring the rule in
    # `build_temporal_index` (_temporal.pyx ~211).
    _CHECKPOINT_THRESHOLD = 0.5

    def __init__(
        self,
        snapshots: list,
        *,
        face_snapshots: Optional[list] = None,
        directed: bool = False,
        general: bool = False,
    ):
        """Build a temporal store from a list of snapshots.

        `snapshots` is a list of connectivity tuples, either (sources, targets)
        or, when `general=True`, (boundary_ptr, boundary_idx). This constructor
        is connectivity only: it carries no w_E or signs, so every reconstructed
        snapshot has w_E=None and signs=None, even if `face_snapshots` supplies
        signed face (B2) data.

        To preserve edge attribution (w_E, signs) and time varying weights,
        build the store by appending full RexGraph snapshots instead: start
        with `TemporalRex([])` and call `append_snapshot(rex)` for each full
        RexGraph. That path round trips w_E, signs, and B2 signs through
        serialize/load/reconstruct.
        """
        self._snapshots = snapshots
        self._face_snapshots = face_snapshots or []
        self._directed = directed
        self._general = general
        self._T = len(snapshots)
        # Wall-clock (or experiment-clock) per step. Absent, the step index IS the
        # time: the identity bridge, so a store built without timestamps behaves
        # exactly as it always did.
        self._times = [float(i) for i in range(self._T)]

        # snapshots-backed construction: full snapshots already held in
        # `_snapshots`, so `at(t)` is authoritative and the incremental
        # checkpoint/delta index is built lazily (see `_ensure_index`).
        self._snapshots_materialized = True
        self._encoding = "general" if general else ("directed" if directed else "undirected")
        self._index_checkpoints = None
        self._index_deltas = None
        self._index_face_deltas = None
        self._index_cp_times = None
        self._cumulative_delta = 0
        self._last_state = None
        self._last_face_state = None
        self._checkpoint_threshold = self._CHECKPOINT_THRESHOLD

    @property
    def T(self) -> int:
        return self._T

    @property
    def times(self) -> NDArray:
        """The clock each step was taken on. Defaults to the step index."""
        return np.asarray(self._times, dtype=_f64)

    def time_at(self, t: int) -> float:
        """The moment step `t` was taken."""
        return float(self._times[t])

    def step_at(self, when: float) -> Optional[int]:
        """The step current at `when`: the latest one taken at or before it.

        Between measurements the complex is whatever the last measurement said, so
        this is a step function, not an interpolation. None if `when` precedes the
        first measurement -- there is no complex to report, and reporting step 0
        would invent one.
        """
        times = self._times
        if not times or when < times[0]:
            return None
        lo, hi = 0, len(times) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if times[mid] <= when:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def reconstruct_at_time(self, when: float) -> Optional[RexGraph]:
        """The complex as it stood at `when`. The bridge between this store's step
        index and any clock the rest of the system records in (the RCDB's tx and
        validity times, an instrument's timestamps, an experiment's passages)."""
        t = self.step_at(when)
        return None if t is None else self.reconstruct_at(t)

    def at(self, t: int) -> RexGraph:
        snap = self._snapshots[t]
        if self._general:
            bp, bi = snap
            kwargs = dict(boundary_ptr=bp, boundary_idx=bi)
        else:
            src, tgt = snap
            kwargs = dict(sources=src, targets=tgt)
        kwargs["directed"] = self._directed

        if self._face_snapshots and t < len(self._face_snapshots):
            fsnap = self._face_snapshots[t]
            b2cp, b2ri = fsnap[0], fsnap[1]
            b2v = fsnap[2] if len(fsnap) > 2 else None
            kwargs["B2_col_ptr"] = b2cp
            kwargs["B2_row_idx"] = b2ri
            # a legacy 2 tuple face snapshot (col_ptr, row_idx) carries no sign
            # information at all, so ones is the only honest default there; a 3
            # tuple (col_ptr, row_idx, vals) carries the real signed orientation
            # and must be round tripped, not overwritten.
            kwargs["B2_vals"] = (
                np.ascontiguousarray(b2v, dtype=_f64) if b2v is not None
                else np.ones(b2ri.shape[0], dtype=_f64)
            )

        return RexGraph(**kwargs)

    def _seed_rex(self, checkpoint) -> RexGraph:
        """Build a fresh RexGraph from a full checkpoint tuple
        (time, boundary_ptr, boundary_idx, w_E, signs, B2_col_ptr, B2_row_idx, B2_vals).
        Only the pieces actually present in the checkpoint are passed through, so
        a checkpoint with no attribution/faces seeds a bare connectivity rex."""
        _, bp, bi, wE, signs, b2cp, b2ri, b2v = checkpoint
        kw = dict(boundary_ptr=bp, boundary_idx=bi, directed=self._directed)
        if wE is not None:
            kw["w_E"] = wE
        if signs is not None:
            kw["signs"] = signs
        if b2cp is not None and b2cp.shape[0] > 1:
            kw["B2_col_ptr"] = b2cp
            kw["B2_row_idx"] = b2ri
            kw["B2_vals"] = b2v
        return RexGraph(**kw)

    def _full_checkpoint(self, t: int, rex: RexGraph) -> Tuple:
        """Read a full-fidelity checkpoint (connectivity + attribution + faces)
        off an already-built snapshot rex, via the connectivity/attribution
        and face state readers."""
        bp, bi, wE, signs = _cell_state(rex)
        b2cp, b2ri, b2v, _ = _face_state(rex)
        return (t, bp.copy(), bi.copy(), wE, signs, b2cp.copy(), b2ri.copy(), b2v.copy())

    def _checkpoint_of(self, rex: RexGraph, t: int) -> Tuple:
        """Full-state checkpoint tuple for `rex` at time `t` (thin alias over
        `_full_checkpoint`, argument order matched to how `_append_index_entry`
        and `append_snapshot` call it)."""
        return self._full_checkpoint(t, rex)

    def _append_index_entry(self, rex: RexGraph, *, face: bool = True,
                             record_snapshot: bool = True) -> int:
        """Diff `self._last_state`/`self._last_face_state` against `rex` and push
        ONE new index entry (an edge delta + face delta, or a full checkpoint once
        cumulative churn crosses `_checkpoint_threshold` of the current edge count),
        then advance `_T`. This is the single incremental step (O(delta), never a
        walk of prior history) shared by both `append_snapshot` (streaming growth)
        and `_ensure_index` (batch build over already-materialized snapshots), so
        the two paths produce a byte-identical index.

        `record_snapshot=False` is how `_ensure_index` replays snapshots that are
        already sitting in `self._snapshots` (placed there at construction time)
        without appending duplicates onto that list.
        """
        t = self._T
        cp, ci, cw, cs = _cell_state(rex)
        nE = int(cp.shape[0] - 1)
        cw = np.zeros(nE, _f64) if cw is None else np.asarray(cw, _f64)
        cs = np.ones(nE, _i32) if cs is None else np.asarray(cs, _i32)
        curr_face_state = _face_state(rex)

        if self._last_state is None:
            # nothing to diff against yet: this is the very first snapshot the
            # index has ever seen, so it is always a full checkpoint.
            self._index_checkpoints[t] = self._checkpoint_of(rex, t)
            self._index_cp_times = np.append(self._index_cp_times, t).astype(np.int64)
            self._index_deltas.append(None)
            self._index_face_deltas.append(None)
            self._cumulative_delta = 0
        else:
            pp, pi, pw, ps = self._last_state
            d = make_edge_delta(pp, pi, pw, ps, cp, ci, cw, cs, self._directed)
            fd = make_face_delta(self._last_face_state, curr_face_state, self._directed) if face else None

            n_born = int(d.born_offsets.shape[0] - 1)
            n_died = int(d.died_keys.shape[0])
            # Modifications count too. Existence and orientation are independent
            # conditions of the composite binary, so a cell reversing orientation is
            # a real change to replay, not a weaker form of one appearing. Counting
            # only born+died meant a history made entirely of reversals -- the normal
            # case wherever direction carries the signal -- registered zero churn,
            # never checkpointed, and left reconstruct_at replaying the whole chain.
            n_mod = int(d.mod_keys.shape[0])
            self._cumulative_delta += n_born + n_died + n_mod

            if nE > 0 and (self._cumulative_delta / nE) > self._checkpoint_threshold:
                self._index_checkpoints[t] = self._checkpoint_of(rex, t)
                self._index_cp_times = np.append(self._index_cp_times, t).astype(np.int64)
                self._index_deltas.append(None)
                self._index_face_deltas.append(None)
                self._cumulative_delta = 0
            else:
                self._index_deltas.append(d)
                self._index_face_deltas.append(fd)

        if record_snapshot and self._snapshots_materialized:
            self._snapshots.append((cp, ci) if self._general else (rex.sources, rex.targets))

        self._last_state = (cp, ci, cw, cs)
        self._last_face_state = curr_face_state
        self._T = t + 1
        while len(self._times) < self._T:
            self._times.append(float(len(self._times)))

        for name in ("temporal_index", "edge_lifecycle", "edge_metrics", "face_lifecycle_data"):
            self.__dict__.pop(name, None)

        return t

    def append_snapshot(self, rex: RexGraph, *, face: bool = True,
                        at: Optional[float] = None) -> int:
        """Append one new snapshot to a live temporal store, maintaining the
        checkpoint/delta index INCREMENTALLY: one edge diff, one face diff, and an
        int comparison against `_checkpoint_threshold`, O(delta), never a
        from-scratch rebuild of the whole history.

        If the index has not been built yet (a store just constructed from a
        full snapshot list, `append_snapshot` called before any analysis or
        `reconstruct_at` touched it), `_ensure_index` runs once first to seed
        it from the snapshots already on hand; every call after that is O(delta).

        Returns the new snapshot's time index.
        """
        self._ensure_index()
        t = self._append_index_entry(rex, face=face, record_snapshot=True)
        if at is not None:
            if t and float(at) < self._times[t - 1]:
                raise ValueError(
                    f"timestamp {at!r} precedes step {t - 1}'s {self._times[t - 1]!r}; "
                    "an out-of-order clock makes step_at ambiguous")
            self._times[t] = float(at)
        return t

    def _ensure_index(self) -> None:
        """Build the checkpoint/delta index used by `reconstruct_at`, if not
        already built. Replays `at(k)` for every snapshot already on hand
        through `_append_index_entry`, the exact same incremental step
        `append_snapshot` uses for streaming growth, so a store built from a
        full snapshot list and a store grown one snapshot at a time end up with
        an IDENTICAL index. `record_snapshot=False` keeps this replay from
        appending duplicates onto `self._snapshots`, which already holds every
        snapshot from construction.

        Checkpoint 0 is always full (the first call into `_append_index_entry`
        has nothing to diff against). Full incremental maintenance thereafter
        (`_checkpoint_threshold`, the same adaptive rule `build_temporal_index`
        uses in the Cython index, _temporal.pyx ~211) is entirely delegated to
        `_append_index_entry`.
        """
        if self._index_cp_times is not None:
            return

        self._index_checkpoints = {}
        self._index_deltas = []
        self._index_face_deltas = []
        self._index_cp_times = np.zeros(0, dtype=np.int64)
        self._last_state = None
        self._last_face_state = None
        self._cumulative_delta = 0

        T = self._T
        self._T = 0
        try:
            for k in range(T):
                self._append_index_entry(self.at(k), face=True, record_snapshot=False)
        except Exception:
            # roll back to the unbuilt state so a retry rebuilds cleanly and the
            # "is not None" guard above does not later no op against a half
            # built index
            self._index_checkpoints = None
            self._index_deltas = None
            self._index_face_deltas = None
            self._index_cp_times = None
            self._cumulative_delta = 0
            self._last_state = None
            self._last_face_state = None
            self._T = T
            raise

    def reconstruct_at(self, t: int) -> RexGraph:
        """Rebuild the snapshot at time `t` by seeding from the nearest full
        checkpoint at or before `t`, then replaying the intervening edge/face
        deltas at the KEY LEVEL (never mutating a live rex, never renumbering).

        `apply_edge_delta`/`apply_face_delta` fold a delta onto a live rex via
        the in-place mutators (`remove_edges`/`add_edges`/`compact`), which
        is fine for single-step use, but is wrong here: compaction
        renumbers vertices to a contiguous range whenever an edge death orphans
        one, while every delta's died/mod keys were computed by `_ensure_index`
        against the ORIGINAL, stable vertex-id scheme. Chaining deltas through
        in-place mutation lets an early death's renumbering desync every later
        delta's keys from the live complex, so a later death/mod silently fails
        to resolve (its key no longer matches anything) and either a stale
        edge persists past its death or the wrong cell gets modified.

        Instead, accumulate the live cell set (and live face set) as plain
        dicts keyed by canonical key, with born columns carrying the ORIGINAL
        vertex ids straight from the delta, apply died/born/modified purely at
        the key level, then build exactly ONE RexGraph at the end from the
        accumulated cells. No live rex is ever mutated mid-replay, so there is
        nothing to renumber and every key stays valid for the whole chain."""
        self._ensure_index()
        from rexgraph.core._temporal import cell_keys_of
        cts = self._index_cp_times
        c = int(cts[np.searchsorted(cts, t, side="right") - 1])
        _, bp, bi, wE, signs, b2cp, b2ri, b2v = self._index_checkpoints[c]
        directed = self._directed

        # live cells: canonical key -> [column (original vertex ids), w_E, sign]
        cells = {}
        seed_keys = cell_keys_of(np.asarray(bp), np.asarray(bi), directed)
        for j in range(len(bp) - 1):
            col = np.asarray(bi[bp[j]:bp[j + 1]]).copy()
            cells[int(seed_keys[j])] = [
                col,
                float(wE[j]) if wE is not None else 0.0,
                int(signs[j]) if signs is not None else 1,
            ]

        # live faces: face key -> (edge_keys array, sign array), edge_keys in
        # terms of the SEED checkpoint's own edge keys (born faces below
        # replace their entry wholesale with the delta's own edge keys)
        faces = {}
        if b2cp is not None and len(b2cp) > 1:
            for f in range(len(b2cp) - 1):
                rows = np.asarray(b2ri[b2cp[f]:b2cp[f + 1]])
                eks = seed_keys[rows]
                faces[int(face_key_of_keys(eks))] = (
                    eks.copy(), np.asarray(b2v[b2cp[f]:b2cp[f + 1]]).copy())

        for k in range(c + 1, t + 1):
            d = self._index_deltas[k]
            if d is not None:
                for key in d.died_keys:
                    cells.pop(int(key), None)
                nb = int(d.born_offsets.shape[0] - 1)
                for i in range(nb):
                    col = np.asarray(d.born_cols[d.born_offsets[i]:d.born_offsets[i + 1]]).copy()
                    bk = int(cell_keys_of(np.array([0, len(col)], dtype=col.dtype),
                                          col, d.directed)[0])
                    cells[bk] = [col, float(d.born_wE[i]), int(d.born_signs[i])]
                for i in range(len(d.mod_keys)):
                    mk = int(d.mod_keys[i])
                    if mk not in cells:
                        raise ValueError(
                            "reconstruct_at: modified-cell key %d absent from the "
                            "live cell set at step %d; a persisting cell must "
                            "resolve, so the index was built out of order" % (mk, k))
                    cells[mk][1] = float(d.mod_wE[i])
                    cells[mk][2] = int(d.mod_signs[i])
            fd = self._index_face_deltas[k] if self._index_face_deltas else None
            if fd is not None:
                for fk in fd.died_face_keys:
                    faces.pop(int(fk), None)
                nbf = int(fd.born_offsets.shape[0] - 1)
                for i in range(nbf):
                    eks = np.asarray(
                        fd.born_edge_keys[fd.born_offsets[i]:fd.born_offsets[i + 1]]).copy()
                    fsg = np.asarray(
                        fd.born_signs[fd.born_offsets[i]:fd.born_offsets[i + 1]]).copy()
                    faces[int(face_key_of_keys(eks))] = (eks, fsg)

        # build ONE RexGraph, preserving original vertex ids (no renumber)
        ordered = list(cells.items())
        key_to_pos = {key: p for p, (key, _cell) in enumerate(ordered)}
        ptr = [0]
        idx = []
        wl = []
        sl = []
        for _key, (col, w, s) in ordered:
            idx.extend(int(v) for v in col)
            ptr.append(len(idx))
            wl.append(w)
            sl.append(s)
        idx_dtype = bi.dtype if len(bi) else _i32
        kw = dict(
            boundary_ptr=np.array(ptr, dtype=idx_dtype),
            boundary_idx=(np.array(idx, dtype=idx_dtype) if idx
                         else np.zeros(0, dtype=idx_dtype)),
            directed=directed,
        )
        if any(w != 0.0 for w in wl):
            kw["w_E"] = np.array(wl, dtype=_f64)
        if any(s != 1 for s in sl):
            kw["signs"] = np.array(sl, dtype=_i32)
        if faces:
            fcp = [0]
            fri = []
            fv = []
            for _fk, (eks, fsg) in faces.items():
                # a face whose edge died mid-replay was already popped from
                # `faces` by died_face_keys/died_keys upstream in the normal
                # case; guard anyway so a stale face never silently resolves
                # to the wrong (reused) column position.
                for ek in eks:
                    ek_i = int(ek)
                    if ek_i not in key_to_pos:
                        raise ValueError(
                            "reconstruct_at: face references edge key %d not in "
                            "the live cell set; a face delta must be replayed "
                            "AFTER its edges' delta for the same step" % ek_i)
                    fri.append(key_to_pos[ek_i])
                fcp.append(len(fri))
                fv.extend(float(x) for x in fsg)
            kw.update(
                B2_col_ptr=np.array(fcp, dtype=_i32),
                B2_row_idx=np.array(fri, dtype=_i32),
                B2_vals=np.array(fv, dtype=_f64),
            )
        return RexGraph(**kw)

    def _snapshot_at(self, t: int) -> RexGraph:
        """Return the snapshot at `t`, from materialized storage if available,
        else reconstructed from the checkpoint/delta index."""
        if self._snapshots_materialized:
            return self.at(t)
        return self.reconstruct_at(t)

    def _all_snapshots(self) -> list:
        """Return all T snapshots: the raw materialized `(src, tgt)`/`(bp, bi)`
        tuples when this store was built from full snapshots, or reconstructed
        RexGraph instances (via the checkpoint/delta index) otherwise."""
        if self._snapshots_materialized:
            return self._snapshots
        return [self.reconstruct_at(t) for t in range(self._T)]

    def _snapshot_pairs(self) -> list:
        """Normalize `_all_snapshots()` into the raw-tuple shape the temporal
        kernels expect: `(src, tgt)` per timestep in standard mode, `(bp, bi)`
        in general mode. When snapshots are materialized, `_all_snapshots()`
        already returns those tuples untouched; when delta-backed, each
        element is a reconstructed RexGraph and this reads the equivalent
        arrays off it via `_cell_state`."""
        snaps = self._all_snapshots()
        if self._snapshots_materialized:
            return snaps
        if self._general:
            return [_cell_state(snap)[:2] for snap in snaps]
        return [(snap.sources, snap.targets) for snap in snaps]

    def _face_snapshot_pairs(self) -> list:
        """`_face_snapshots` trimmed to the plain `(B2_col_ptr, B2_row_idx)`
        shape the temporal Cython kernels expect. Each stored entry may be a
        legacy 2 tuple or a 3 tuple that also carries `B2_vals`; the
        kernels below only ever consume the CSR structure, never the signs,
        so this strips a third element down to the 2 tuple form without
        touching `self._face_snapshots` itself."""
        return [(fsnap[0], fsnap[1]) for fsnap in self._face_snapshots]

    @cached_property
    def temporal_index(self) -> Tuple:
        snaps = self._snapshot_pairs()
        if self._general:
            return _temporal.build_temporal_index_general(snaps)
        return _temporal.build_temporal_index(snaps, self._directed)

    @cached_property
    def edge_lifecycle(self) -> Tuple:
        snaps = self._snapshot_pairs()
        if self._general:
            return _temporal.edge_lifecycle_general(snaps)
        return _temporal.edge_lifecycle(snaps, self._directed)

    def bioes(
        self,
        betti_matrix: NDArray,
        *,
        phase_tol: float = 0.0,
        min_phase_len: int = 2,
        face_event_threshold: int = 1,
        min_shared: int = 1,
    ) -> Tuple:
        snaps = self._snapshot_pairs()
        face_snaps = self._face_snapshot_pairs()
        if self._general:
            return _temporal.compute_bioes_unified_general(
                snaps, face_snaps, betti_matrix,
                phase_tol, min_phase_len, face_event_threshold, min_shared,
            )
        return _temporal.compute_bioes_unified(
            snaps, face_snaps, betti_matrix,
            self._directed, phase_tol, min_phase_len,
            face_event_threshold, min_shared,
        )

    def mutations(self) -> dict:
        """Paired death and birth at one moment: a cell whose existence changed into
        another structure.

        A cell dying as another is born, on overlapping boundary, is a topology
        mutating. Read as an unrelated death plus an unrelated birth it looks like
        churn; read as one event it is the thing worth knowing. Betti numbers cannot
        see it at all -- the pair can leave every one of them where it was.

        Pairing is by SHARED BOUNDARY VERTICES, exactly, and the count of them is the
        magnitude: the same currency the face correspondence uses, with no similarity
        score and no cutoff. A swap that keeps most of its boundary is a small
        mutation; one that keeps a single vertex is nearly an unrelated death and
        birth. The count is reported and the policy is the caller's.

        Greedy on largest overlap, and each cell is paired at most once: one death
        cannot be the origin of two births, or the count of what happened stops
        meaning anything.

        Returns t / when / died_key / born_key / shared.
        """
        self._ensure_index()
        d = self.delta_tensor()
        if len(d["t"]) == 0:
            return {k: np.asarray([], dtype=dt) for k, dt in
                    (("t", np.int64), ("when", _f64), ("died_key", np.int64),
                     ("born_key", np.int64), ("shared", np.int32))}

        # boundary vertex sets, per step, for the cells that appeared or vanished
        from rexgraph.core._temporal import cell_keys_of
        verts = {}
        for t in sorted({int(x) for x in d["t"]} | {int(x) - 1 for x in d["t"]}):
            if t < 0:
                continue
            rex = self.reconstruct_at(t)
            rex._ensure_clean()
            bp, bi = rex._boundary_ptr, rex._boundary_idx
            keys = np.asarray(cell_keys_of(bp, bi, self._directed), dtype=np.int64)
            for j, k in enumerate(keys):
                verts[(t, int(k))] = set(np.asarray(bi[bp[j]:bp[j + 1]]).tolist())

        t_out, w_out, dk_out, bk_out, sh_out = [], [], [], [], []
        for t in sorted({int(x) for x in d["t"]}):
            at = d["t"] == t
            died = [int(k) for k, e in zip(d["key"][at], d["existence"][at]) if e < 0]
            born = [int(k) for k, e in zip(d["key"][at], d["existence"][at]) if e > 0]
            if not died or not born:
                continue
            cand = []
            for dk in died:
                dv = verts.get((t - 1, dk), set())
                for bk in born:
                    overlap = len(dv & verts.get((t, bk), set()))
                    if overlap:
                        cand.append((overlap, dk, bk))
            cand.sort(key=lambda c: (-c[0], c[1], c[2]))
            used_d, used_b = set(), set()
            for overlap, dk, bk in cand:
                if dk in used_d or bk in used_b:
                    continue
                used_d.add(dk); used_b.add(bk)
                t_out.append(t); w_out.append(self._times[t])
                dk_out.append(dk); bk_out.append(bk); sh_out.append(overlap)

        return {"t": np.asarray(t_out, dtype=np.int64),
                "when": np.asarray(w_out, dtype=_f64),
                "died_key": np.asarray(dk_out, dtype=np.int64),
                "born_key": np.asarray(bk_out, dtype=np.int64),
                "shared": np.asarray(sh_out, dtype=np.int32)}

    #: BIOES tag codes, matching rexgraph.core._temporal
    TAG_B, TAG_I, TAG_O, TAG_E, TAG_S = 0, 1, 2, 3, 4

    def bioes_grid(self) -> dict:
        """BIOES per cell per moment: cells on one axis, time on the other.

        O is the 0 of the existence condition; B/I/E/S presuppose existence=1 and say
        where in a contiguous life you are. So this is the lifetime-position reading
        of the existence channel, not a separate scheme laid over it. Tagging
        TIMESTEPS by phase can never use O, because phases partition the timeline and
        nothing is outside them.

        A row is what the whole complex is doing at one moment; a column is one
        cell's life. Orientation rides alongside as its own channel rather than
        inside the tag: a cell that reverses has persisted, and folding the reversal
        into the tag would collapse two independent conditions back together.

        Returns
        -------
        keys : int64[nCells]         stable cell identities, sorted (the cell axis)
        tags : int8[T, nCells]       B/I/O/E/S per cell per moment
        orientation : int8[T, nCells]  the sign each cell carries, 0 where absent
        moment : int32[T, 5]         per-moment counts of each letter
        """
        from rexgraph.core._temporal import cell_keys_of

        self._ensure_index()
        T = self._T
        present, orient = [], []
        for t in range(T):
            rex = self.reconstruct_at(t)
            rex._ensure_clean()
            keys = np.asarray(cell_keys_of(rex._boundary_ptr, rex._boundary_idx,
                                           self._directed), dtype=np.int64)
            signs = rex._signs
            signs = (np.ones(keys.shape[0], _i32) if signs is None
                     else np.asarray(signs, _i32).ravel())
            present.append({int(k): int(sg) for k, sg in zip(keys, signs)})
            orient.append(None)

        axis = sorted({k for step in present for k in step})
        keys = np.asarray(axis, dtype=np.int64)
        n = len(axis)
        tags = np.full((T, n), self.TAG_O, dtype=np.int8)
        orientation = np.zeros((T, n), dtype=np.int8)
        col_of = {k: c for c, k in enumerate(axis)}

        for t, step in enumerate(present):
            for k, sg in step.items():
                orientation[t, col_of[k]] = sg

        # walk each cell's presence trace and bound every contiguous run. A run of
        # one is S; otherwise its ends are B and E and its interior is I. A cell that
        # flickers therefore gets one bounded span per window rather than one life
        # spanning the gap, which is what actually happened.
        for k in axis:
            c = col_of[k]
            here = [k in step for step in present]
            t = 0
            while t < T:
                if not here[t]:
                    t += 1
                    continue
                start = t
                while t + 1 < T and here[t + 1]:
                    t += 1
                if start == t:
                    tags[start, c] = self.TAG_S
                else:
                    tags[start, c] = self.TAG_B
                    tags[t, c] = self.TAG_E
                    for m in range(start + 1, t):
                        tags[m, c] = self.TAG_I
                t += 1

        moment = np.zeros((T, 5), dtype=np.int32)
        for t in range(T):
            for tag in range(5):
                moment[t, tag] = int((tags[t] == tag).sum())

        return {"keys": keys, "tags": tags, "orientation": orientation,
                "moment": moment, "times": self.times}

    def delta_tensor(self, *, dense: bool = False):
        """The temporal delta tensor: per step, the change in each of the composite
        binary's two independent conditions.

        A relational complex's entries carry an existence condition in {0,1} and an
        orientation in {+1,-1}. They vary independently -- a cell can persist while
        its orientation reverses -- so differencing each separately gives a history
        with two channels rather than one churn count:

            existence   -1 the cell died, +1 it was born, 0 it persisted
            orientation -1/+1 its orientation reversed, 0 it held

        A born or died cell scores 0 in the orientation channel: it has no previous
        orientation to have changed. That is what keeps the channels independent
        rather than one shadowing the other.

        Cells are identified by their canonical key, so a cell that reverses and
        reverses back is one identity across the history, not three.

        Returns a sparse event view -- t / key / existence / orientation, one row per
        cell that changed -- or, with ``dense=True``, the (T, n_cells, 2) array plus
        the key axis. Step 0 has no predecessor and is all zeros.
        """
        from rexgraph.core._temporal import cell_keys_of

        self._ensure_index()
        t_out, k_out, e_out, o_out = [], [], [], []
        prev = {}
        for t in range(self._T):
            rex = self.reconstruct_at(t)
            rex._ensure_clean()
            keys = np.asarray(cell_keys_of(rex._boundary_ptr, rex._boundary_idx,
                                           self._directed), dtype=np.int64)
            signs = rex._signs
            # signs=None is the all-positive orientation, not an absent one
            signs = (np.ones(keys.shape[0], _i32) if signs is None
                     else np.asarray(signs, _i32).ravel())
            curr = {int(k): int(sg) for k, sg in zip(keys, signs)}
            if t:
                for key, sg in curr.items():
                    was = prev.get(key)
                    if was is None:
                        t_out.append(t); k_out.append(key)
                        e_out.append(1); o_out.append(0)
                    elif was != sg:
                        # {-2, +2} halved: the orientation channel is a delta of the
                        # condition, on the same {-1, 0, +1} scale as existence.
                        t_out.append(t); k_out.append(key)
                        e_out.append(0); o_out.append((sg - was) // 2)
                for key in prev:
                    if key not in curr:
                        t_out.append(t); k_out.append(key)
                        e_out.append(-1); o_out.append(0)
            prev = curr

        out = {
            "t": np.asarray(t_out, dtype=np.int64),
            "when": np.asarray([self._times[i] for i in t_out], dtype=_f64),
            "key": np.asarray(k_out, dtype=np.int64),
            "existence": np.asarray(e_out, dtype=np.int8),
            "orientation": np.asarray(o_out, dtype=np.int8),
        }
        if not dense:
            return out
        axis = np.unique(out["key"]) if out["key"].size else np.zeros(0, np.int64)
        tensor = np.zeros((self._T, axis.shape[0], 2), dtype=np.int8)
        if axis.shape[0]:
            col = np.searchsorted(axis, out["key"])
            tensor[out["t"], col, 0] = out["existence"]
            tensor[out["t"], col, 1] = out["orientation"]
        return tensor, axis

    def temporal_persistence(self, final_rex: Optional[RexGraph] = None) -> dict:
        R = final_rex or self._snapshot_at(self._T - 1)
        snaps = self._snapshot_pairs()
        if self._general:
            filt = _persistence.filtration_temporal_general(
                snaps, R.nV, R.nE,
                R.boundary_ptr, R.boundary_idx,
                R._B2_col_ptr, R._B2_row_idx,
            )
        else:
            snap_src = [s[0] for s in snaps]
            snap_tgt = [s[1] for s in snaps]
            filt = _persistence.filtration_temporal(
                snap_src, snap_tgt, R.nV, R.nE,
                R.sources, R.targets,
                R._B2_col_ptr, R._B2_row_idx,
            )
        return R.persistence(*filt)

    # Energy-domain temporal analysis

    @cached_property
    def edge_metrics(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Per-timestep edge counts, births, and deaths.

        Returns (edge_counts, edge_born, edge_died) each int32[T].
        """
        snaps = self._snapshot_pairs()
        if self._general:
            return _temporal.compute_edge_metrics_general(snaps)
        return _temporal.compute_edge_metrics(snaps, self._directed)

    @cached_property
    def face_lifecycle_data(self) -> Optional[Tuple]:
        """Face lifecycle tracking across all timesteps.

        Returns None if no face snapshots are available.
        """
        if not self._face_snapshots or len(self._face_snapshots) != self._T:
            return None
        if self._general:
            return None  # general face lifecycle not yet supported
        return _temporal.face_lifecycle(
            self._face_snapshot_pairs(), self._snapshot_pairs(), self._directed)

    def bioes_energy(
        self,
        E_kin: NDArray,
        E_pot: NDArray,
        *,
        ratio_tol: float = 0.2,
        min_phase_len: int = 2,
    ) -> Tuple:
        """Energy-domain BIOES from kinetic/potential timeseries.

        Classifies temporal phases by E_kin/E_pot ratio regime
        (kinetic / crossover / potential) and assigns BIOES tags.

        Parameters
        ----------
        E_kin : f64[T] - topological energy per timestep
        E_pot : f64[T] - geometric energy per timestep

        Returns
        -------
        tags, phase_start, phase_end, phase_regime, log_ratios,
        crossover_times.
        """
        E_kin = np.ascontiguousarray(E_kin, dtype=np.float64)
        E_pot = np.ascontiguousarray(E_pot, dtype=np.float64)
        return _temporal.compute_bioes_energy(
            E_kin, E_pot, ratio_tol, min_phase_len)

    def bioes_joint(
        self,
        betti_matrix: NDArray,
        E_kin: NDArray,
        E_pot: NDArray,
        *,
        betti_tol: float = 0.0,
        ratio_tol: float = 0.2,
    ) -> Tuple:
        """Joint Betti + energy phase detection.

        A phase breaks when any Betti number shifts OR the energy
        regime (kinetic/crossover/potential) changes.

        Returns
        -------
        phase_start, phase_end, phase_betti, phase_regime,
        break_reasons, log_ratios.
        """
        betti_matrix = np.ascontiguousarray(betti_matrix, dtype=np.int64)
        E_kin = np.ascontiguousarray(E_kin, dtype=np.float64)
        E_pot = np.ascontiguousarray(E_pot, dtype=np.float64)
        return _temporal.detect_phases_joint(
            betti_matrix, E_kin, E_pot, betti_tol, ratio_tol)

    def cascade_activation(
        self,
        edge_signals: NDArray,
        threshold: float = 0.01,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Edge activation order during signal propagation.

        Parameters
        ----------
        edge_signals : f64[T, nE] - signal magnitude per timestep

        Returns
        -------
        activation_time : i32[nE] (-1 if never activated)
        activation_order : i32[n_activated]
        activation_rank : i32[nE] (-1 if never activated)
        """
        signals = np.ascontiguousarray(edge_signals, dtype=np.float64)
        return _temporal.cascade_edge_activation(signals, threshold)

    def cascade_wavefront(
        self,
        edge_signals: NDArray,
        threshold: float = 0.01,
    ) -> dict:
        """Wavefront tracking with spatial propagation analysis.

        Requires standard (non-general) snapshots for edge endpoints.
        """
        signals = np.ascontiguousarray(edge_signals, dtype=np.float64)
        if self._general:
            raise ValueError("cascade_wavefront requires standard snapshots")
        snap0 = self._snapshot_at(0)
        src = np.ascontiguousarray(snap0.sources, dtype=np.int32)
        tgt = np.ascontiguousarray(snap0.targets, dtype=np.int32)
        return _temporal.cascade_wavefront(signals, src, tgt, threshold)

    def __repr__(self) -> str:
        fmt = "general" if self._general else "standard"
        return f"TemporalRex(T={self._T}, format={fmt})"


def cross_complex_bridge(rex_A, rex_B, labels_A, labels_B,
                         channel_scores_A=None, channel_scores_B=None):
    """Graph-level cross-complex bridge between two RexGraphs.

    Aligns the two complexes by vertex label, then combines kappa
    correlation, void-fraction comparison, and (optionally) channel-score
    correlation over the shared vertices. Thin wrapper around
    ``_cross_complex.cross_complex_bridge`` that extracts the required
    arrays from each RexGraph.

    Parameters
    ----------
    rex_A, rex_B : RexGraph
    labels_A, labels_B : sequence of str
        Vertex labels; ``labels_X[i]`` is the label of vertex ``i`` in X.
    channel_scores_A, channel_scores_B : array-like, optional
        Per-group channel scores; if both given, channel correlation is
        included under the ``'channel'`` key.

    Returns
    -------
    dict
        ``{'kappa', 'void', 'n_shared'[, 'channel']}`` (see the kernel).
    """
    if _cross_complex is None:
        raise RuntimeError("rexgraph.core._cross_complex is not available")

    shared, idx_A, idx_B = _cross_complex.align_by_labels(
        list(labels_A), list(labels_B))

    kappa_A = np.ascontiguousarray(rex_A.coherence, dtype=_f64)
    kappa_B = np.ascontiguousarray(rex_B.coherence, dtype=_f64)

    vc_A = rex_A.void_complex
    vc_B = rex_B.void_complex

    return _cross_complex.cross_complex_bridge(
        kappa_A, kappa_B,
        np.ascontiguousarray(idx_A, dtype=_i32),
        np.ascontiguousarray(idx_B, dtype=_i32),
        int(vc_A['n_voids']), int(vc_A['n_potential']),
        int(vc_B['n_voids']), int(vc_B['n_potential']),
        channel_scores_A=channel_scores_A,
        channel_scores_B=channel_scores_B,
    )
