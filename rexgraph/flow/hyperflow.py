"""The flow complex: both grades, closed by hyperfaces, three channels live.

A relational flow model needs somewhere for signal to flow. That is not automatic, and
the usual construction does not provide it: building the complex from the GROUPS alone,
one branching column per group, gives a forest of stars. Nothing closes, beta_1 = 0, and
both the curl and harmonic tiers are empty, so every signal is pure gradient and a "flow
layer" over it is ordinary message passing with no structural content. That is what a
plain `from_hypergraph` gives and why a model built on it has nothing to offer over an
MLP on the same features.

The construction here carries BOTH grades in one complex:

    the GROUP        a branching relation of arity k over the entity and its partners.
                     This is what OPENS the cycle.
    each MEASUREMENT a 2-ary relation over the same entities. These are the legs.
    auto_hyperface   closes each group against the measurements that span its boundary,
                     with the face coefficients SOLVED from B1 c_f = 0 rather than
                     declared.

Neither half works alone. Without the group there is no cycle; without the measurements
there is nothing for a face to close. Together the group opens the hole and the face fills
it, which is exactly `curl_dim = cycle_count - dim_H`: the cycle count does not change,
what changes is whether those cycles are holes or boundaries.

Only then is the Hodge decomposition non-trivial in all three parts,

    R^nE  =  im(B1^T)  (+)  im(B2)  (+)  ker(L1)
             gradient        curl        harmonic

with all three parts non-trivial, which is what gives a flow model separately
addressable channels. Propagation between grades is the graded Dirac. Equiweight
(Gamma D + D Gamma = 0) makes D odd with respect to the grading, so it never leaves a
signal in the grade it started in.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from rexgraph.faces import auto_hyperface
from rexgraph.graph import RexGraph

__all__ = ["FlowComplex", "build_flow_complex"]


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M, dtype=float)


def _b1_csr(rex):
    """B1 as sparse CSR, from the dual rather than through `rex.B1`.

    `rex.B1` is a cached_property returning `to_dense_f64(self._B1_dual)`, so
    reading it materialises an nV x nE float array AND keeps it: 1.5 GB at
    nV 8,000 against 1.5 MB for the same operator in CSR. The dual is the sparse
    source both are built from.
    """
    from rexgraph.core._sparse import to_scipy_csr
    rex._ensure_clean()
    return to_scipy_csr(rex._B1_dual).tocsr()


def _b2_csr(rex):
    """B2 as sparse CSR, from the dual. See `_b1_csr` for why not `rex.B2`."""
    from rexgraph.core._sparse import to_scipy_csr
    rex._ensure_clean()
    B2 = rex._B2_hodge_dual if rex._B2_hodge_dual is not None else rex._B2_dual
    if B2 is None or B2.ncol == 0:
        return sp.csr_matrix((int(rex.nE), 0), dtype=float)
    return to_scipy_csr(B2).tocsr()


def build_flow_complex(groups: Sequence[Sequence[int]], *,
                       include_groups: bool = True, measurements: bool = True,
                       close: bool = True,
                       weights: Sequence[float] | None = None) -> FlowComplex:
    """Build the flow complex from a list of groups.

    Each group is a list of entity ids: the first is the distinguished one (the entity
    the group is *of*), the rest are its partners. `measurements=False` omits the 2-ary
    legs and `include_groups=False` omits the branching columns, so the degenerate
    constructions stay reachable for comparison rather than being asserted away.

    `weights` is one metric weight per relation, groups first then measurements, in the
    order they are built. It enters the T and G channels; C is deliberately unweighted.
    """
    bp, bi = [0], []

    def add(vs):
        bi.extend(int(v) for v in vs)
        bp.append(len(bi))

    n_groups = 0
    if include_groups:
        for g in groups:
            if len(g) >= 2:
                add(g)
                n_groups += 1
    n_meas = 0
    if measurements:
        for g in groups:
            head = g[0]
            for other in g[1:]:
                add([head, other])
                n_meas += 1

    rex = RexGraph.from_hypergraph(np.asarray(bp, np.int32), np.asarray(bi, np.int32))
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != int(rex.nE):
            raise ValueError(
                f"weights must be one per relation ({int(rex.nE)}), got {w.shape[0]}")
        rex._w_E = w
    if close:
        auto_hyperface(rex)
    return FlowComplex(rex, n_groups=n_groups, n_measurements=n_meas)


class FlowComplex:
    """A relational complex with all three edge channels available, plus the graded
    propagation that moves signal between grades."""

    def __init__(self, rex: RexGraph, *, n_groups: int = 0, n_measurements: int = 0):
        self.rex = rex
        self.n_groups = int(n_groups)
        self.n_measurements = int(n_measurements)

    #### shape
    @property
    def n_faces(self) -> int:
        return int(self.rex.nF_hodge)

    @property
    def _B1(self):
        """Dense B1. No caller left: every reader moved to `_b1_csr`, which reads
        the dual instead of materialising and caching an nV x nE float array.
        Kept rather than removed; remove when that is decided."""
        return _dense(self.rex.B1)

    @property
    def _B2(self):
        """Dense B2. No caller left; see `_B1`."""
        nE = int(self.rex.nE)
        return _dense(self.rex.B2) if self.n_faces else np.zeros((nE, 0))

    @property
    def gradient_dim(self) -> int:
        """dim im(B1^T): the part of a signal explained by a vertex potential.

        Read through `graded_boundary._sparse_rank`, which settles the rank over the
        rationals and only falls to a float estimate when neither exact path applies.
        A boundary column carries the share 1/(k-1), so it looks like a float matrix
        while having an exact integer representative: scaling by (k-1) clears the
        denominator and the rank is invariant under that. Taking a float SVD rank
        here instead would put an integer answer on a tolerance, and the operator
        would have to be densified to ask.
        """
        if int(self.rex.nE) == 0:
            return 0
        from rexgraph.graded_boundary import _sparse_rank
        # rank(B1^T) == rank(B1), and B1 is the cheap side to reduce: its columns
        # carry the arity of one relation, where a column of the transpose carries
        # the degree of one vertex. Same integer, far less fill.
        return _sparse_rank(_b1_csr(self.rex))

    @property
    def curl_dim(self) -> int:
        """dim im(B2): the cycles that BOUND. Zero without faces, so this is the
        dimension a hyperface adds. Same exact rank as `gradient_dim`."""
        if not self.n_faces:
            return 0
        from rexgraph.graded_boundary import _sparse_rank
        return _sparse_rank(_b2_csr(self.rex))

    @property
    def harmonic_dim(self) -> int:
        """dim ker(L1): the cycles that are still HOLES. Equals betti_1."""
        return int(self.rex.betti[1])

    @property
    def cycle_count(self) -> int:
        """curl + harmonic: every cycle, whether or not it bounds. Invariant under
        attaching a face, which is what `curl_dim = cycle_count - dim_H` says."""
        return self.curl_dim + self.harmonic_dim

    @property
    def chain_residual(self) -> float:
        """max |B1 B2|, which the chain condition requires to be 0. Adjudicated exactly.

        This was a float max over the densified operators, which is what core refuses in
        its own docstring, and it returned 0.0 for every arity it was tried at for a
        reason that is not the mathematics: a column is (-1, 1/(k-1), ..., 1/(k-1)) and
        `(k-1) * fl(1/(k-1))` happens to round back to exactly 1 for k = 3..12. It does
        not in general. Scanning k = 3..4000, 483 arities leave a nonzero float column sum
        (the first is k = 50), so a structurally perfect complex would have reported a
        failure there, and a genuinely broken one can report success anywhere the error
        lands under the noise.

        `rex.chain_valid` is the same predicate over the rationals, so this defers to it
        and returns 0.0 exactly when the condition holds. The float magnitude stays
        available as `chain_residual_float` for anyone who wants the numerical size rather
        than the answer.
        """
        if not self.n_faces:
            return 0.0
        return 0.0 if self.rex.chain_valid else self.chain_residual_float

    @property
    def chain_residual_float(self) -> float:
        """The float magnitude of B1 B2, for scale rather than for the verdict.

        Sparse: the verdict is `chain_residual` and this only reports how large the
        numerical violation is, which needs no dense operator to say.
        """
        if not self.n_faces:
            return 0.0
        prod = (_b1_csr(self.rex) @ _b2_csr(self.rex)).tocoo()
        return float(np.abs(prod.data).max()) if prod.nnz else 0.0

    def summary(self) -> dict:
        return {
            "nV": int(self.rex.nV), "nE": int(self.rex.nE), "n_faces": self.n_faces,
            "n_groups": self.n_groups, "n_measurements": self.n_measurements,
            "gradient": self.gradient_dim, "curl": self.curl_dim,
            "harmonic": self.harmonic_dim, "betti": tuple(int(b) for b in self.rex.betti),
            "chain_residual": self.chain_residual,
            "c0_squared": self.rex.c0_squared,
        }

    #### the Hodge decomposition
    def decompose(self, f) -> dict:
        """Split an edge signal into gradient, curl and harmonic, through the library.

        The recovery is driven by the E equation: energy is ADDITIVE across the two
        towers because the chain condition makes the boundary pairing vanish,
        <B1^T a, B2 b> = a^T (B1 B2) b = 0. Gradient and curl are therefore orthogonal
        by construction, not by numerical accident, so recovering two parts recovers the
        third by subtraction and no cross term has to be estimated.

        The harmonic part comes from `harmonic_sparse.harmonic_projection`, which applies
        P_H = H (H^T H)^-1 H^T LOW-RANK against a SPARSE Gram and never forms the dense
        nE x nE projector. The gradient part is a sparse SPD solve through
        `scale_propagator.block_cg_solve`, the same matrix-free solver the Green's
        diagonal uses. Curl is what is left.
        """
        from rexgraph.harmonic_sparse import cycle_basis as _cycles
        from rexgraph.harmonic_sparse import harmonic_basis, harmonic_projection

        f = np.asarray(f, dtype=float).ravel()
        nE = int(self.rex.nE)
        if f.shape[0] != nE:
            raise ValueError(f"signal must have one entry per relation ({nE})")

        # ker(B1) is the CYCLE space and that is curl (+) harmonic, not harmonic alone.
        # So one low-rank projection off the cycle basis gives the gradient complement,
        # and a second off the harmonic basis splits what is left. Both go through the
        # same projector, which uses the sparse Gram and never forms nE x nE.
        Z = _cycles(self.rex)
        cyc = harmonic_projection(Z, f) if Z is not None and Z.shape[1] else np.zeros(nE)
        H = harmonic_basis(self.rex)
        harm = harmonic_projection(H, f) if H is not None and H.shape[1] else np.zeros(nE)
        grad = f - cyc
        return {"gradient": grad, "curl": cyc - harm, "harmonic": harm}

    def energy_split(self, f) -> dict:
        """The E equation as a check: E = E_gradient + E_curl + E_harmonic, exactly.

        Additivity is what the chain condition buys, so a nonzero cross term means the
        decomposition is wrong rather than that the data is awkward.
        """
        p = self.decompose(f)
        e = {k: float(v @ v) for k, v in p.items()}
        e["total"] = float(f @ f)
        e["cross_residual"] = e["total"] - (e["gradient"] + e["curl"] + e["harmonic"])
        return e

    #### graded propagation
    def grade_slice(self, d: int) -> slice:
        nV, nE = int(self.rex.nV), int(self.rex.nE)
        return (slice(0, nV) if d == 0 else
                slice(nV, nV + nE) if d == 1 else
                slice(nV + nE, nV + nE + self.n_faces))

    def grade_energy(self, psi) -> list:
        psi = np.asarray(psi, dtype=float).ravel()
        return [float(psi[self.grade_slice(d)] @ psi[self.grade_slice(d)]) for d in (0, 1, 2)]

    def step(self, psi):
        """One application of the graded Dirac: D psi.

        D is supported only between consecutive grades, so this always MOVES the signal
        across a grade and never leaves it where it was. That is equiweight
        (Gamma D + D Gamma = 0) as an operation rather than an identity on paper.
        """
        psi = np.asarray(psi, dtype=float).ravel()
        return np.asarray(self.rex.dirac_operator, dtype=float) @ psi

    def propagate(self, psi, t: float, order: int = 60):
        """The unitary (wave) propagator e^{-itD}, returning (real, imaginary).

        Norm-conserving, so the grades exchange amplitude without any being lost. That is
        what makes it a flow rather than a diffusion; use `rexgraph.dirac_propagator`'s
        heat form when dissipation is what is wanted.
        """
        from rexgraph.dirac_propagator import dirac_from_rex

        sd = dirac_from_rex(self.rex)
        psi = np.asarray(psi, dtype=float).ravel()
        if psi.shape[0] != sd.N:
            raise ValueError(f"state must have one entry per cell ({sd.N})")
        return sd.light(psi, t=t, order=order)


def flow_adjacency(rex, *, alpha=1.0):
    """The propagation operator that READS BOTH GRADES: L1_down + alpha * L1_up.

    `coparticipation_adjacency` is built from abs(B1) alone and never touches B2, so a
    learner using it is blind to every face in the complex: attaching hyperfaces leaves
    its operator bit-identical. Measured on a three-group fixture, the co-participation
    block over the measurement relations is unchanged by closing the complex, which is
    why an ablation over that learner reports the same accuracy for an open and a closed
    complex. The curl tier exists and the model cannot see it.

    This is the operator a flow model needs: the down-Laplacian carries the gradient tier
    and the up-Laplacian carries the curl tier, so signal propagates through both. alpha
    is the exchange rate between them; `rex.c0_squared` is the principled choice and is
    exact rational, which is why it is available separately rather than baked in here.

    Returned as a torch sparse tensor, normalised the way GreensCochain expects.
    """
    import scipy.sparse as _sp
    import torch as _torch

    from rexgraph.core._sparse import to_scipy_csr as _tsc

    B1 = _tsc(rex._B1_dual).tocsr()
    L = (B1.T @ B1).tocsr()                                  # L1_down: gradient tier
    if int(rex.nF_hodge) > 0 and alpha:
        B2 = _tsc(rex._B2_hodge_dual).tocsr()
        L = (L + float(alpha) * (B2 @ B2.T)).tocsr()         # + curl tier
    # a Laplacian is PSD with a kernel; GreensCochain wants a low-pass ADJACENCY, so take
    # the off-diagonal magnitude and renormalise exactly as coparticipation_adjacency does
    A = abs(L).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    n = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel() + 1.0
    dinv = 1.0 / np.sqrt(deg)
    S = (_sp.diags(dinv) @ (A + _sp.eye(n)) @ _sp.diags(dinv)).tocoo()
    idx = _torch.tensor(np.vstack([S.row, S.col]), dtype=_torch.long)
    val = _torch.tensor(S.data, dtype=_torch.float64)
    return _torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
