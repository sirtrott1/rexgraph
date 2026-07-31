"""rexgraph.dirac_propagator: the graded Dirac operator as a SPARSE, matrix-free
operator, and propagation of graded TENSOR STATES through it.

The Dirac operator ``D = d + d*`` acts on the whole graded space
``C_0 ⊕ C_1 ⊕ ... ⊕ C_G`` with the boundary maps ``B_d : C_d -> C_{d-1}`` sitting
in its OFF-diagonal blocks:

    D = [[ 0,      B_1,    0,    ... ],
         [ B_1^T,  0,      B_2,  ... ],
         [ 0,      B_2^T,  0,    ... ],
         [ ...                      ]]

``D`` is real symmetric and ``D^2 = blkdiag(L_0, L_1, ..., L_G)`` (the off-diagonal
blocks of ``D^2`` vanish because ``B_d B_{d+1} = 0``).

Why this module exists (the fix for the quarantined heat propagator): the old path
chased ``diag(e^{-tL})`` on the EDGE space alone - a diagonal of a general matrix
function, which has no exact O(nnz) form and is blind to inter-grade transport. The
information actually lives in the OFF-diagonal blocks of ``D``: applying a function
of ``D`` to a graded state VECTOR propagates amplitude ACROSS grades
(vertex<->edge<->face), and that is a sparse-matvec Chebyshev evaluation - O(nnz.K),
any parameter, no eigendecomposition.

Odd/even structure (why the light propagator is the grade-mixing one): odd powers of
``D`` (``D, D^3, ...``) are the off-diagonal / grade-CROSSING terms; even powers
(``D^2, D^4, ...``) are block-diagonal / in-grade. ``D`` is indefinite, so ``e^{-tD}``
is unbounded on the negative branch; the BOUNDED grade-crossing operator is
``sin(tD)`` - the imaginary part of the light/wave propagator ``e^{-itD}``::

    e^{-itD} = cos(tD)  -  i sin(tD)
               ^in-grade    ^grade-crossing (curl)   [gradient]^

This supersedes ``core._dirac.schrodinger_evolve`` (dense eigendecomposition) and the
edge-space ``_experimental.heat_propagator_diag``. Grade-general: it consumes a list
of sparse boundary maps, so it works for a 2-rex today and an N-rex the moment the
higher boundaries exist.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

_f64 = np.float64

# Block matvec goes parallel only when the state block is large enough that the
# per-tile sparse mat-vecs (which release the GIL) outweigh thread-pool overhead.
# Tiny inputs stay on the plain serial path. Measured in state entries (N * k).
_PARALLEL_MIN_ELEMS = 1 << 15
_PARALLEL_MAX_THREADS = 8

__all__ = [
    "SparseDirac",
    "dirac_from_rex",
    "dirac_light",
    "dirac_heat",
]


def _boundaries_from_rex(rex):
    """The sparse boundary maps ``[B_1, B_2, ...]`` of a RexGraph, each
    ``B_d : C_d -> C_{d-1}`` as scipy CSR. Reads the rex's own signed incidence, so
    whatever arity/sign convention the complex was built with (witness / pairwise /
    branching edges; triangle / n-gon faces) is carried through unchanged.

    Grade-general first: if the rex exposes ``graded_boundaries`` (a property another
    workstream provides that returns the full ``[B_1, B_2, B_3, ...]`` sparse list),
    use it verbatim so an N-rex propagates the moment its higher boundaries exist.
    Otherwise fall back to the vertex/edge (+ face) construction from the rex's own
    signed incidence."""
    graded = getattr(rex, "graded_boundaries", None)
    if callable(graded):                       # RexGraph.graded_boundaries is a method
        try:
            graded = graded()
        except Exception:
            graded = None
    if graded is not None:
        try:
            boundaries = list(graded)
        except TypeError:
            boundaries = None
        if boundaries:
            return [b.tocsr() for b in boundaries]

    from rexgraph.sparse_character import _b1_csr
    from rexgraph.core._sparse import to_scipy_csr

    boundaries = [_b1_csr(rex).tocsr()]                     # B_1 : nV x nE
    if int(getattr(rex, "nF", 0)) > 0 and rex._B2_hodge_dual is not None:
        boundaries.append(to_scipy_csr(rex._B2_hodge_dual).tocsr())   # B_2 : nE x nF
    return boundaries


class SparseDirac:
    """The graded Dirac operator ``D = d + d*`` as a sparse, matrix-free operator.

    Parameters
    ----------
    boundaries : list of sparse matrices
        ``boundaries[d]`` is ``B_{d+1} : C_{d+1} -> C_d`` with shape
        ``(n_d, n_{d+1})``. ``boundaries[0]`` is ``B_1`` (nV x nE), ``boundaries[1]``
        is ``B_2`` (nE x nF), etc. Any column arity / signing is respected.
    """

    def __init__(self, boundaries):
        if not boundaries:
            raise ValueError("need at least B_1")
        self.B = [b.tocsr().astype(_f64) for b in boundaries]
        self.Bt = [b.T.tocsr() for b in self.B]
        # grade sizes n_0, n_1, ..., n_G  (n_0 = rows of B_1; n_{d+1} = cols of B_{d+1})
        self.sizes = [self.B[0].shape[0]] + [b.shape[1] for b in self.B]
        self.offsets = np.cumsum([0] + self.sizes).astype(np.int64)
        self.N = int(self.offsets[-1])
        self.n_grades = len(self.sizes)

    # -- structure ---------------------------------------------------------------

    def grade_slice(self, d):
        """The index slice of grade ``d`` in the stacked state vector."""
        return slice(int(self.offsets[d]), int(self.offsets[d + 1]))

    def _matvec_serial(self, psi):
        """Serial core of :meth:`matvec` - one pass of the graded sparse mat-vecs.

        ``(D psi)_d = B_{d+1} psi_{d+1} + B_d^T psi_{d-1}`` - the down-map from the
        grade above plus the up-map from the grade below."""
        out = np.zeros_like(psi)
        off = self.offsets
        G = self.n_grades
        for d in range(G):
            block = out[off[d]:off[d + 1]]
            if d >= 1:                                  # B_d^T psi_{d-1}  (up)
                block += self.Bt[d - 1] @ psi[off[d - 1]:off[d]]
            if d + 1 < G:                               # B_{d+1} psi_{d+1}  (down)
                block += self.B[d] @ psi[off[d + 1]:off[d + 2]]
        return out

    def matvec(self, psi):
        """Apply ``D`` to a graded state vector or block of states.

        A single vector (or a 1-column block) takes the plain serial path. A wider
        block (``N x k``, ``k > 1``) that is large enough tiles its COLUMNS across a
        thread pool - each tile is an independent set of GIL-releasing sparse
        mat-vecs, exactly the column-tiling pattern of
        ``scale_propagator.greens_diagonal``. Results are bit-identical to serial; the
        gate (:data:`_PARALLEL_MIN_ELEMS`) keeps tiny inputs off the thread pool.
        """
        psi = np.asarray(psi, dtype=_f64)
        if psi.ndim < 2 or psi.shape[1] <= 1 or psi.size < _PARALLEL_MIN_ELEMS:
            return self._matvec_serial(psi)

        k = psi.shape[1]
        from rexgraph import compute as _compute
        nthreads = _compute.get_threads() or (os.cpu_count() or 1)
        nthreads = min(_PARALLEL_MAX_THREADS, nthreads, k)
        if nthreads <= 1:
            return self._matvec_serial(psi)

        step = max(1, -(-k // nthreads))                # ceil(k / nthreads) columns/tile
        bounds = [(s, min(s + step, k)) for s in range(0, k, step)]
        if len(bounds) <= 1:
            return self._matvec_serial(psi)

        def _tile(b):                                   # one column tile, independent matvec
            start, stop = b
            return start, self._matvec_serial(psi[:, start:stop])

        results = _compute.parallel_map(_tile, bounds, threads=min(_PARALLEL_MAX_THREADS,
                                                                    len(bounds)))
        out = np.empty_like(psi)
        for start, block in results:
            out[:, start:start + block.shape[1]] = block
        return out

    def aslinearoperator(self):
        return sp.linalg.LinearOperator((self.N, self.N), matvec=self.matvec,
                                        rmatvec=self.matvec, dtype=_f64)

    def to_scipy(self):
        """Assemble ``D`` as an explicit sparse CSR (for verification / tiny inputs)."""
        G, off = self.n_grades, self.offsets
        blocks = [[None] * G for _ in range(G)]
        for d in range(G - 1):
            blocks[d][d + 1] = self.B[d]
            blocks[d + 1][d] = self.Bt[d]
        return sp.bmat(blocks, format="csr")

    def spectral_radius(self, tol=1e-3, maxiter=None):
        """Largest ``|eigenvalue|`` of ``D`` via a few Lanczos mat-vecs (not a full
        eigensolve). Gershgorin bound as a floor / fallback."""
        gersh = 0.0
        for b in self.B:
            rs = np.asarray(np.abs(b).sum(axis=1)).ravel()
            cs = np.asarray(np.abs(b).sum(axis=0)).ravel()
            gersh = max(gersh, rs.max(initial=0.0), cs.max(initial=0.0))
        if self.N <= 3:
            return max(float(np.abs(np.linalg.eigvalsh(self.to_scipy().toarray())).max()),
                       1e-12)
        try:
            lm = sp.linalg.eigsh(self.aslinearoperator(), k=1, which="LM",
                                 return_eigenvectors=False,
                                 maxiter=maxiter or self.N * 10, tol=tol)
            return max(float(abs(lm[0])), gersh, 1e-12)
        except Exception:
            return max(gersh, 1e-12)

    # -- propagation of tensor states -------------------------------------------

    def _cheb_apply(self, func, psi, lam_max, order):
        """Apply ``func(D)`` to a state block ``psi`` (n x k) by a Chebyshev
        polynomial of ``D`` on ``[-lam_max, lam_max]`` - sparse mat-vecs only, no
        eigendecomposition."""
        j = np.arange(order)
        nodes = np.cos(np.pi * (j + 0.5) / order)          # Chebyshev nodes in [-1,1]
        lam = nodes * lam_max                              # mapped to D's spectrum
        fvals = func(lam)
        c = np.array([(2.0 / order) * np.sum(fvals * np.cos(np.pi * k * (j + 0.5) / order))
                      for k in range(order)])
        c[0] /= 2.0
        scale = 1.0 / lam_max                               # rescale D to [-1,1]

        def Ds(x):
            return scale * self.matvec(x)

        tkm1 = psi
        tk = Ds(psi)
        acc = c[0] * tkm1 + c[1] * tk
        for k in range(2, order):
            tkp1 = 2.0 * Ds(tk) - tkm1
            acc = acc + c[k] * tkp1
            tkm1, tk = tk, tkp1
        return acc

    def light(self, psi0, t, order=None, lam_max=None):
        """The light / wave propagator ``e^{-itD} psi0`` on a graded tensor state.

        Returns ``(re, im)`` where ``re = cos(tD) psi0`` is the in-grade (gradient)
        part and ``im = -sin(tD) psi0`` is the grade-CROSSING (curl) part - the
        amplitude that the off-diagonal boundary blocks transport between grades. All
        sparse mat-vecs, arbitrary ``t``, no eigendecomposition.
        """
        psi0 = np.asarray(psi0, dtype=_f64)
        if lam_max is None:
            lam_max = self.spectral_radius() * 1.02 + 1e-9
        if order is None:                                  # scale work to t*lam_max
            order = int(max(24, min(400, 1.5 * t * lam_max + 24)))
        re = self._cheb_apply(lambda l: np.cos(t * l), psi0, lam_max, order)
        im = self._cheb_apply(lambda l: -np.sin(t * l), psi0, lam_max, order)
        return re, im

    def heat_squared(self, psi0, t, order=None, lam_max=None):
        """The (stable, per-grade) heat propagator ``e^{-tD^2} psi0 = e^{-tL} psi0``.

        ``D^2`` is block-diagonal, so this diffuses WITHIN each grade - it does not
        cross grades. Provided as the diffusive companion to :meth:`light` (whose
        imaginary part is the grade-crossing transport). Uses ``func(l)=e^{-t l^2}``
        applied through ``D``, so it stays matrix-free on the same operator.
        """
        psi0 = np.asarray(psi0, dtype=_f64)
        if lam_max is None:
            lam_max = self.spectral_radius() * 1.02 + 1e-9
        if order is None:
            order = int(max(24, min(400, 1.5 * t * lam_max * lam_max + 24)))
        return self._cheb_apply(lambda l: np.exp(-t * l * l), psi0, lam_max, order)

    def grade_energy(self, psi):
        """Per-grade energy ``||psi_d||^2`` of a (real or stacked re/im) state - the
        readout that shows amplitude crossing grades under :meth:`light`."""
        psi = np.asarray(psi, dtype=_f64)
        return np.array([float(np.sum(psi[self.grade_slice(d)] ** 2))
                         for d in range(self.n_grades)])

    def trajectory(self, psi0, times, order=None, lam_max=None):
        """Propagate the light state ``e^{-itD} psi0`` at multiple ``times`` and read
        off the per-grade Born energy at each - the sparse/matvec companion to
        ``core._dirac.schrodinger_trajectory`` (no eigendecomposition).

        Returns a dict with:

        - ``times``   : ``float64[T]`` the requested times.
        - ``energy``  : ``float64[T, n_grades]`` Born energy ``||re_d||^2 + ||im_d||^2``
          on each grade ``d`` at each time - shows amplitude flowing between grades.
        - ``total``   : ``float64[T]`` total energy ``||psi(t)||^2``; constant under the
          unitary ``e^{-itD}`` (a conservation check).

        ``lam_max`` is computed once and reused across all times so the spectral bound
        is a single Lanczos pass, not one per timepoint.
        """
        psi0 = np.asarray(psi0, dtype=_f64)
        times = np.atleast_1d(np.asarray(times, dtype=_f64))
        if lam_max is None:
            lam_max = self.spectral_radius() * 1.02 + 1e-9
        T = times.shape[0]
        energy = np.zeros((T, self.n_grades), dtype=_f64)
        total = np.zeros(T, dtype=_f64)
        for i, t in enumerate(times):
            re, im = self.light(psi0, float(t), order=order, lam_max=lam_max)
            e = self.grade_energy(re) + self.grade_energy(im)
            energy[i] = e
            total[i] = float(e.sum())
        return {"times": times, "energy": energy, "total": total}


def dirac_from_rex(rex):
    """Build the :class:`SparseDirac` of a RexGraph from its own signed boundaries."""
    return SparseDirac(_boundaries_from_rex(rex))


def _default_seed(sd):
    """Default propagation seed: unit amplitude on a single grade-0 (vertex) cell, zero
    elsewhere. A concrete, reproducible localized starting state whose gradient is
    non-zero, so the off-diagonal boundary blocks carry amplitude UP the grades under
    the propagator (unlike the constant grade-0 vector, which is harmonic and does not
    move on a regular complex)."""
    psi0 = np.zeros(sd.N, dtype=_f64)
    psi0[0] = 1.0                                   # first vertex; grade 0 starts at index 0
    return psi0


def dirac_light(rex, t, psi0=None, order=None):
    """Light / wave propagator ``e^{-itD} psi0`` on a rex, built from its own signed
    boundaries. Returns ``(re, im)`` = ``(cos(tD) psi0, -sin(tD) psi0)`` - the in-grade
    (gradient) and grade-crossing (curl) parts. ``psi0`` defaults to unit amplitude on
    grade 0 (see :func:`_default_seed`). Matrix-free, arbitrary ``t``, no
    eigendecomposition."""
    sd = dirac_from_rex(rex)
    if psi0 is None:
        psi0 = _default_seed(sd)
    return sd.light(psi0, float(t), order=order)


def dirac_heat(rex, t, psi0=None):
    """Per-grade heat propagator ``e^{-tD^2} psi0 = e^{-tL} psi0`` on a rex. ``D^2`` is
    block-diagonal, so this diffuses WITHIN each grade (the diffusive companion to
    :func:`dirac_light`, whose imaginary part crosses grades). ``psi0`` defaults to unit
    amplitude on grade 0 (see :func:`_default_seed`)."""
    sd = dirac_from_rex(rex)
    if psi0 is None:
        psi0 = _default_seed(sd)
    return sd.heat_squared(psi0, float(t))
