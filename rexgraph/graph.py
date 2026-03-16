"""
rexgraph.graph - Orchestration layer for relational complexes.

A k-rex is a finite chain complex in which edges are primitive and
vertices are derived from edge boundaries via the vertex lifecycle
contract: a vertex exists if and only if some edge contains it in
its boundary.

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

import functools
import json
from functools import cached_property
from typing import Optional, Sequence, Tuple, Union

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


def _ensure_dense(M):
    """Convert matrix to dense ndarray if not already."""
    if isinstance(M, np.ndarray):
        return M
    if M is None:
        return None
    return np.asarray(M.toarray(), dtype=_f64)


# RexGraph


class RexGraph:
    """A relational complex (rex) with lazily computed derived properties.

    A k-rex is a finite chain complex

        C_k -> C_{k-1} -> ... -> C_0

    satisfying d_{k-1} . d_k = 0.  Edges are primitive; the vertex set
    is derived: V = union over e in E of supp(d_1(e)).

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
    ):
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

    def _invalidate(self):
        """Clear all cached properties after mutation."""
        for attr in list(self.__dict__):
            delattr(self, attr)

    # Factory constructors

    @classmethod
    def from_graph(
        cls,
        sources: ArrayLike,
        targets: ArrayLike,
        *,
        directed: bool = False,
        w_E: Optional[NDArray] = None,
    ) -> RexGraph:
        """Embed a simple graph as a 1-rex."""
        return cls(
            sources=np.asarray(sources),
            targets=np.asarray(targets),
            directed=directed,
            w_E=w_E,
        )

    @classmethod
    def from_hypergraph(
        cls,
        hyperedge_ptr: ArrayLike,
        hyperedge_idx: ArrayLike,
    ) -> RexGraph:
        """Embed a hypergraph as a 1-rex with branching edges."""
        return cls(
            boundary_ptr=np.asarray(hyperedge_ptr),
            boundary_idx=np.asarray(hyperedge_idx),
        )

    @classmethod
    def from_simplicial(
        cls,
        sources: ArrayLike,
        targets: ArrayLike,
        triangles: ArrayLike,
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
        )

    @classmethod
    def from_adjacency(cls, A: NDArray, *, directed: bool = False) -> RexGraph:
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
        )

    # Dimensions

    @property
    def nV(self) -> int:
        return self._nV

    @property
    def nE(self) -> int:
        return self._nE

    @property
    def nF(self) -> int:
        return self._nF

    @property
    def dimension(self) -> int:
        if self._nE == 0:
            return 0
        return 2 if self._nF > 0 else 1

    # Raw structural data

    @property
    def boundary_ptr(self) -> NDArray:
        return self._boundary_ptr

    @property
    def boundary_idx(self) -> NDArray:
        return self._boundary_idx

    @property
    def sources(self) -> Optional[NDArray]:
        if self._sources is not None:
            return self._sources
        sizes = np.diff(self._boundary_ptr)
        if np.any(sizes < 2):
            return None
        self._sources = self._boundary_idx[self._boundary_ptr[:-1]]
        return self._sources

    @property
    def targets(self) -> Optional[NDArray]:
        if self._targets is not None:
            return self._targets
        sizes = np.diff(self._boundary_ptr)
        if np.any(sizes < 2):
            return None
        self._targets = self._boundary_idx[self._boundary_ptr[:-1] + 1]
        return self._targets

    @cached_property
    def _is_standard_only(self) -> bool:
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
        """DualCSR representation of B2."""
        if self._nF == 0:
            return None
        # Build dense B2 from CSC data, then convert
        B2_dense = np.zeros((self._nE, self._nF), dtype=_f64)
        cp, ri, vl = self._B2_col_ptr, self._B2_row_idx, self._B2_vals
        for f in range(self._nF):
            for k in range(cp[f], cp[f + 1]):
                B2_dense[ri[k], f] = vl[k]
        return _boundary.build_B2_from_dense(self._nE, self._nF, B2_dense)

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
        This guarantees L_O is PSD with eigenvalues in [0, 1].
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
        bp, bi = self._boundary_ptr, self._boundary_idx
        nE = self._nE
        K = np.zeros((nE, nE), dtype=_f64)
        bsets = []
        for e in range(nE):
            bsets.append(set(bi[bp[e]:bp[e + 1]].tolist()))

        # K_ij = number of vertices shared by edges i and j (unsigned Gramian)
        for i in range(nE):
            for j in range(i, nE):
                shared = len(bsets[i] & bsets[j])
                K[i, j] = shared
                K[j, i] = shared

        # Row-sum normalization: L_O = I - D^{-1/2} K D^{-1/2}
        d_ov = K.sum(axis=1)
        inv_sqrt = np.zeros(nE, dtype=_f64)
        for i in range(nE):
            if d_ov[i] > 1e-12:
                inv_sqrt[i] = 1.0 / np.sqrt(d_ov[i])
        D_inv_sqrt = np.diag(inv_sqrt)
        S = D_inv_sqrt @ K @ D_inv_sqrt
        L_O = np.eye(nE, dtype=_f64) - S
        return {'L_O': L_O, 'S': S, 'd_ov': d_ov}

    @cached_property
    def L_overlap(self) -> NDArray:
        """Overlap Laplacian L_O = I - D_ov^{-1/2} K D_ov^{-1/2}."""
        return self._overlap_bundle['L_O']

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
    def _B2_hodge_dual(self):
        """DualCSR of B2 with chain-violating faces filtered.

        Filters faces where B_1 B_2[:, f] != 0, which violate the chain
        complex axiom. This is a direct algebraic check rather than the
        heuristic vertex-count approach, ensuring no valid faces are
        wrongly excluded and no invalid faces slip through.
        """
        if self._nF == 0 or self._B2_dual is None:
            return None

        B1_dense = self.B1
        B2_dense = self.B2

        # Direct check: B1 @ B2[:, f] should be zero for Hodge faces
        product = B1_dense @ B2_dense  # (nV, nF)
        keep = []
        for f in range(self._nF):
            col_norm = float(np.max(np.abs(product[:, f])))
            if col_norm < 1e-10:
                keep.append(f)

        if len(keep) == self._nF:
            return self._B2_dual

        if len(keep) == 0:
            return None

        # Build filtered B2 and convert to DualCSR
        B2_filtered = B2_dense[:, keep].copy()
        return _boundary.build_B2_from_dense(
            self._nE, len(keep), np.ascontiguousarray(B2_filtered, dtype=_f64)
        )

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
        B1_dense = self.B1
        B2_dense = self.B2
        product = B1_dense @ B2_dense  # (nV, nF)
        excluded = []
        for f in range(self._nF):
            if float(np.max(np.abs(product[:, f]))) >= 1e-10:
                excluded.append(f)
        return excluded

    @cached_property
    def spectral_bundle(self) -> dict:
        """All Laplacians and spectral decompositions.

        Single call producing L0, L1, L2, eigenvalues, eigenvectors,
        Betti numbers, coupling constants, RL_1, Lambda, and diagnostics.

        Uses B2_hodge (self-loop faces filtered) for correct L_1 and
        Betti numbers.
        """
        return _laplacians.build_all_laplacians(
            self._B1_dual,
            self._B2_hodge_dual,
            self.L_overlap,
            auto_alpha=True,
            k=-1,
        )

    # Spectral accessors (thin dict lookups into spectral_bundle)

    @cached_property
    def L0(self) -> NDArray:
        """L_0 = B_1 B_1^T."""
        return _ensure_dense(self.spectral_bundle['L0'])

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
        """L_2 = B_2^T B_2."""
        L2 = self.spectral_bundle.get('L2')
        if L2 is not None:
            return _ensure_dense(L2)
        return np.zeros((0, 0), dtype=_f64)

    @cached_property
    def betti(self) -> Tuple[int, int, int]:
        """Betti numbers (beta_0, beta_1, beta_2)."""
        sb = self.spectral_bundle
        return (sb['beta0'], sb['beta1'], sb['beta2'])

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
        """Relational Laplacian RL_1 = L_1 + alpha_G * L_O.

        None when L_O is absent or alpha_G is NaN/zero.
        """
        RL = self.spectral_bundle.get('RL_1')
        return _ensure_dense(RL) if RL is not None else None

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
    def L_frustration(self) -> NDArray:
        """Frustration Laplacian L_SG."""
        if not _HAS_RCF:
            return np.zeros((self._nE, self._nE), dtype=_f64)
        src, tgt = self._ensure_src_tgt()
        return _frustration.build_L_SG(
            self._nV, self._nE, src, tgt, signs=self._edge_signs)

    @cached_property
    def L_coPC(self) -> Optional[NDArray]:
        """Copath complex Laplacian L_C (line-graph Hodge).

        None if the line graph has no edges (e.g. star graphs).
        """
        if not _HAS_RCF:
            return None
        K1 = self.spectral_bundle.get('K1')
        if K1 is None:
            return None
        lg = _relational.build_line_graph(
            np.asarray(K1, dtype=_f64), self._nE)
        if lg['nE_L'] == 0:
            return None
        return _relational.build_L_coPC(lg)

    @cached_property
    def _rcf_bundle(self) -> dict:
        """Relational Laplacian, hat operators, structural character.

        Assembles RL from all available typed Laplacians.
        Uses 3 hats (RL3) when L_coPC is unavailable, 4 hats (RL4) otherwise.

        Keys: RL, hats, nhats, trace_values, hat_names, chi
        """
        if not _HAS_RCF:
            return {}
        laplacians = [self.L1, self.L_overlap, self.L_frustration]
        names = ['L1_down', 'L_O', 'L_SG']
        L_C = self.L_coPC
        if L_C is not None:
            laplacians.append(L_C)
            names.append('L_C')
        result = _relational.build_RL(laplacians, names)
        chi = _character.compute_chi(
            result['RL'], result['hats'], result['nhats'], self._nE)
        result['chi'] = chi
        return result

    @cached_property
    def RL(self) -> NDArray:
        """Relational Laplacian RL = sum of trace-normalized typed Laplacians.

        tr(RL) = nhats. When L_coPC is available, nhats = 4 (RL4).
        Otherwise nhats = 3 (RL3).
        """
        rcf = self._rcf_bundle
        return rcf.get('RL', np.zeros((self._nE, self._nE), dtype=_f64))

    @cached_property
    def nhats(self) -> int:
        """Number of active hat operators in the relational Laplacian."""
        return self._rcf_bundle.get('nhats', 3)

    @cached_property
    def _rl_eigen(self) -> Tuple[NDArray, NDArray]:
        """Cached eigendecomposition of RL."""
        if not _HAS_RCF:
            return np.linalg.eigh(self.RL)
        return _relational.rl_eigen(self.RL)

    @cached_property
    def _green_cache(self) -> dict:
        """Green function cache: RL^+, B1 @ RL^+, S0."""
        if not _HAS_RCF:
            return {}
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
        return self._rcf_bundle.get('chi',
                                     np.zeros((self._nE, self.nhats), dtype=_f64))

    @cached_property
    def vertex_character(self) -> NDArray:
        """phi(v) in Delta^{nhats-1} per vertex. Shape (nV, nhats)."""
        return self._vertex_bundle.get('phi',
                                        np.zeros((self._nV, self.nhats), dtype=_f64))

    @cached_property
    def star_character(self) -> NDArray:
        """chi*(v) = mean of chi(e) over incident edges. Shape (nV, nhats)."""
        return self._vertex_bundle.get('chi_star',
                                        np.zeros((self._nV, self.nhats), dtype=_f64))

    @cached_property
    def coherence(self) -> NDArray:
        """kappa(v) = 1 - 0.5 * ||phi(v) - chi*(v)||_1. Shape (nV,)."""
        return self._vertex_bundle.get('kappa',
                                        np.zeros(self._nV, dtype=_f64))

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
        rcf = self._rcf_bundle
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
        rl_diag = np.ascontiguousarray(np.diag(self.RL))
        return float(_rcfe.compute_strain(self.rcfe_curvature, rl_diag, self._nE))

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

    def explain(self, dim: int, idx: int) -> dict:
        """Full diagnostic for a cell (edge or vertex)."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
        rcf = self._rcf_bundle
        if dim == 1:
            K1 = np.abs(self.B1).T @ np.abs(self.B1)
            return _query.explain_edge(
                self.B1, self.B2_hodge, K1, self.RL,
                rcf['hats'], rcf['nhats'],
                idx, self._nV, self._nE, self.nF_hodge,
            )
        elif dim == 0:
            v2e_ptr, v2e_idx = self._v2e
            return _query.explain_vertex(
                self.B1, self.RL, rcf['hats'], rcf['nhats'],
                self.vertex_character, self.coherence,
                self.structural_character,
                v2e_ptr, v2e_idx,
                idx, self._nV, self._nE,
            )
        else:
            raise ValueError(f"explain not supported for dim={dim}")

    def propagate(self, source: NDArray, target: NDArray) -> dict:
        """Spectral propagation score through RL."""
        if not _HAS_RCF:
            raise RuntimeError("RCF modules not available.")
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
        """Kinetic/potential energy decomposition of an edge signal.

        Returns (E_kin, E_pot, ratio) where:
            E_kin = <f_E | L_1 | f_E>  (topological)
            E_pot = <f_E | L_O | f_E>  (geometric)
        """
        f = np.ascontiguousarray(f_E, dtype=_f64)
        return _state.energy_kin_pot(f, self.L1, self.L_overlap)

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

        Returns (E_kin_per_edge, E_pot_per_edge) where each sums
        to the total <f|L|f>.
        """
        f = np.ascontiguousarray(f_E, dtype=_f64)
        return _quotient.per_edge_energy(f, self.L1, self.L_overlap)

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
        """Markov continuous-time evolution via matrix exponential."""
        L_dense = _ensure_dense([self.L0, self.L1, self.L2][dim])
        return _transition.markov_continuous_expm(
            np.ascontiguousarray(g, dtype=_f64), L_dense, t
        )

    def evolve_schrodinger(self, psi: NDArray, dim: int, t: float) -> Tuple[NDArray, NDArray]:
        """Schrodinger (unitary) evolution via spectral method.

        Returns (f_real, f_imag) components of exp(-i L_k t) psi.
        """
        L_dense = _ensure_dense([self.L0, self.L1, self.L2][dim])
        evals, evecs = np.linalg.eigh(L_dense)
        return _transition.schrodinger_evolve_spectral(
            np.ascontiguousarray(psi, dtype=_f64), evals, evecs, t
        )

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

        deriv = functools.partial(
            _transition.coupled_derivative,
            sizes=sizes,
            L0=self.L0,
            L1=self.L1,
            L2=self.L2 if self._nF > 0 else np.zeros((0, 0), dtype=_f64),
            L_O=self.L_overlap,
            B1_dense=self.B1,
            B2_dense=self.B2_hodge,
            alpha0=alpha0,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha_G=ag,
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
        psi_E = np.asarray(psi_E, dtype=_c128)
        psi_F = np.asarray(psi_F, dtype=_c128)
        sb = self.spectral_bundle
        evals_rl = self.evals_RL1
        evecs_rl = self.evecs_RL1
        if evals_rl is None:
            evals_rl, evecs_rl = np.linalg.eigh(self.L1)
        evals_l2 = sb.get('evals_L2')
        evecs_l2 = sb.get('evecs_L2')
        return _wave.field_schrodinger_evolve(
            psi_E, psi_F,
            evals_rl, evecs_rl,
            evals_l2, evecs_l2,
            t, B1=self.B1,
        )

    def evolve_field_trajectory(
        self,
        psi_E: NDArray,
        psi_F: NDArray,
        times: NDArray,
    ) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Rex field Schrodinger evolution through multiple timepoints.

        Returns (traj_E, traj_F, traj_V) each shaped [nT, ...].
        """
        psi_E = np.asarray(psi_E, dtype=_c128)
        psi_F = np.asarray(psi_F, dtype=_c128)
        times = np.ascontiguousarray(times, dtype=_f64)
        sb = self.spectral_bundle
        evals_rl = self.evals_RL1
        evecs_rl = self.evecs_RL1
        if evals_rl is None:
            evals_rl, evecs_rl = np.linalg.eigh(self.L1)
        evals_l2 = sb.get('evals_L2')
        evecs_l2 = sb.get('evecs_L2')
        return _wave.field_schrodinger_trajectory(
            psi_E, psi_F,
            evals_rl, evecs_rl,
            evals_l2, evecs_l2,
            times, B1=self.B1,
        )

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
            L = [self.L0, self.L1, self.L2][dim]
            _, evecs = np.linalg.eigh(_ensure_dense(L))
        return _wave.measure_in_eigenbasis(psi, evecs)

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
        evals, evecs, _ = self.field_eigen
        return _field.field_diffusion_trajectory(F0, evals, evecs, times)

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
        evals, evecs, freqs = self.field_eigen
        return _field.wave_evolve_trajectory(F0, evals, evecs, freqs, times)

    def classify_modes(self) -> dict:
        """Classify field eigenmodes as edge-dominated, face-dominated, or coupled.

        Returns dict with mode_type (int array), edge_weight, face_weight,
        and frequency for each eigenmode.
        """
        evals, evecs, freqs = self.field_eigen
        return _field.classify_modes(evals, evecs, self._nE, int(self.nF_hodge))

    def derive_vertex_state(self, F: NDArray) -> NDArray:
        """Derive vertex observable from packed field state via B_1.

        f_V = B_1 @ F[:nE].
        """
        F = np.ascontiguousarray(F, dtype=_f64)
        return _field.derive_vertex_state(F, self.B1, self._nE)

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

        evals_rl = self.evals_RL1
        evecs_rl = self.evecs_RL1
        if evals_rl is None:
            evals_rl, evecs_rl = np.linalg.eigh(self.L1)

        sb = self.spectral_bundle
        src, tgt = self._ensure_src_tgt()
        ag = self.alpha_G
        if ag != ag:
            ag = 0.0

        return _signal.analyze_perturbation(
            f_E, f_F,
            self.L1, self.L_overlap,
            evals_rl, evecs_rl,
            self.B1, self.B2_hodge,
            times,
            L0=sb.get('L0'),
            L2_op=sb.get('L2'),
            RL1=self.relational_laplacian,
            edge_src=np.ascontiguousarray(src, dtype=_i32),
            edge_tgt=np.ascontiguousarray(tgt, dtype=_i32),
            alpha_G=ag,
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

        M, g_used, _ = self.field_operator
        evals, evecs, freqs = self.field_eigen

        return _signal.analyze_perturbation_field(
            f_E, f_F, M, evals, evecs, freqs,
            self.L1, self.L_overlap, self.B1,
            times, self._nE, self.nF_hodge, mode,
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
        d = {
            "nV": self._nV, "nE": self._nE, "nF": self._nF,
            "dimension": self.dimension,
            "boundary_ptr": self._boundary_ptr.tolist(),
            "boundary_idx": self._boundary_idx.tolist(),
            "edge_types": self.edge_types.tolist(),
            "betti": list(self.betti),
            "euler_characteristic": self.euler_characteristic,
        }
        if self._nV > 0 and self._nE > 0:
            d["layout"] = self.layout.tolist()
        if self._nF > 0:
            d["B2_col_ptr"] = self._B2_col_ptr.tolist()
            d["B2_row_idx"] = self._B2_row_idx.tolist()
        if self._w_E is not None:
            d["w_E"] = self._w_E.tolist()
        return d

    def to_dict(self) -> dict:
        return {
            "boundary_ptr": self._boundary_ptr,
            "boundary_idx": self._boundary_idx,
            "B2_col_ptr": self._B2_col_ptr,
            "B2_row_idx": self._B2_row_idx,
            "B2_vals": self._B2_vals,
            "w_E": self._w_E,
            "w_boundary": self._w_boundary,
            "directed": self._directed,
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
        )

    def __repr__(self) -> str:
        return f"RexGraph(nV={self._nV}, nE={self._nE}, nF={self._nF}, dim={self.dimension})"


# TemporalRex


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
    )

    def __init__(
        self,
        snapshots: list,
        *,
        face_snapshots: Optional[list] = None,
        directed: bool = False,
        general: bool = False,
    ):
        self._snapshots = snapshots
        self._face_snapshots = face_snapshots or []
        self._directed = directed
        self._general = general
        self._T = len(snapshots)

    @property
    def T(self) -> int:
        return self._T

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
            b2cp, b2ri = self._face_snapshots[t]
            kwargs["B2_col_ptr"] = b2cp
            kwargs["B2_row_idx"] = b2ri
            kwargs["B2_vals"] = np.ones(b2ri.shape[0], dtype=_f64)

        return RexGraph(**kwargs)

    @cached_property
    def temporal_index(self) -> Tuple:
        if self._general:
            return _temporal.build_temporal_index_general(self._snapshots)
        return _temporal.build_temporal_index(self._snapshots, self._directed)

    @cached_property
    def edge_lifecycle(self) -> Tuple:
        if self._general:
            return _temporal.edge_lifecycle_general(self._snapshots)
        return _temporal.edge_lifecycle(self._snapshots, self._directed)

    def bioes(
        self,
        betti_matrix: NDArray,
        *,
        phase_tol: float = 0.0,
        min_phase_len: int = 2,
        face_event_threshold: int = 1,
        jaccard_threshold: float = 0.5,
    ) -> Tuple:
        if self._general:
            return _temporal.compute_bioes_unified_general(
                self._snapshots, self._face_snapshots, betti_matrix,
                phase_tol, min_phase_len, face_event_threshold, jaccard_threshold,
            )
        return _temporal.compute_bioes_unified(
            self._snapshots, self._face_snapshots, betti_matrix,
            self._directed, phase_tol, min_phase_len,
            face_event_threshold, jaccard_threshold,
        )

    def temporal_persistence(self, final_rex: Optional[RexGraph] = None) -> dict:
        R = final_rex or self.at(self._T - 1)
        if self._general:
            filt = _persistence.filtration_temporal_general(
                self._snapshots, R.nV, R.nE,
                R.boundary_ptr, R.boundary_idx,
                R._B2_col_ptr, R._B2_row_idx,
            )
        else:
            snap_src = [s[0] for s in self._snapshots]
            snap_tgt = [s[1] for s in self._snapshots]
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
        if self._general:
            return _temporal.compute_edge_metrics_general(self._snapshots)
        return _temporal.compute_edge_metrics(self._snapshots, self._directed)

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
            self._face_snapshots, self._snapshots, self._directed)

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
        snap = self._snapshots[0]
        if self._general:
            raise ValueError("cascade_wavefront requires standard snapshots")
        src = np.ascontiguousarray(snap[0], dtype=np.int32)
        tgt = np.ascontiguousarray(snap[1], dtype=np.int32)
        return _temporal.cascade_wavefront(signals, src, tgt, threshold)

    def __repr__(self) -> str:
        fmt = "general" if self._general else "standard"
        return f"TemporalRex(T={self._T}, format={fmt})"
