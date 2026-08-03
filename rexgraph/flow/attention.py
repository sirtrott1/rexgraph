"""rexgraph.flow.attention: matrix-free co-participation attention.

Two edges co-participate iff they share an incident vertex (most often a
high-arity branching vertex, but the same structural test applies to any
shared endpoint). A masked edge's signal is predicted from its observed
co-participants, each weighted by a learned compatibility over the two
edges' inside tensors. Setting gamma=0 collapses the weighting to uniform,
which is the zero-parameter settle: no learned compatibility at all, just
"average what the co-participants say".

Everything here is matrix-free and sparse. The co-participation neighborhood
is read directly off the signed incidence B1 (nV x nE): `absB1.T @ absB1` is
the (unweighted, by-shared-vertex-count) edge-to-edge adjacency, since entry
(e, e') is nonzero exactly when edges e and e' share an incident vertex.
Using abs(B1) rather than the signed B1 avoids a spurious zero from sign
cancellation on a shared vertex; the diagonal (every edge co-participates
with itself) is dropped before the CSR is handed back. No dense operator,
no eigendecomposition.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from rexgraph.core._sparse import to_scipy_csr

__all__ = ["coparticipation_neighbors", "coparticipation_attention", "CoParticipationAttention"]


def coparticipation_neighbors(rex) -> tuple[NDArray, NDArray]:
    """CSR of each edge's co-participants (edges sharing an incident vertex).

    Built matrix-free from the signed incidence `rex._B1_dual` (nV x nE):
    for each vertex, the edges incident to it are mutual co-participants, so
    `abs(B1).T @ abs(B1)` is exactly the (self-inclusive) co-participation
    adjacency over edges. The diagonal (self co-participation) is dropped.

    Returns `(nbr_ptr, nbr_idx)`, both int32: `nbr_idx[nbr_ptr[e]:nbr_ptr[e+1]]`
    is edge e's co-participants, self excluded.
    """
    abs_b1 = abs(to_scipy_csr(rex._B1_dual))  # nV x nE
    adjacency = (abs_b1.T @ abs_b1).tocsr()  # nE x nE, structural (shared-vertex) adjacency
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    nbr_ptr = adjacency.indptr.astype(np.int32)
    nbr_idx = adjacency.indices.astype(np.int32)
    return nbr_ptr, nbr_idx


def coparticipation_attention(
    nbr_ptr: NDArray,
    nbr_idx: NDArray,
    inside: NDArray,
    signal: NDArray,
    obs_mask: NDArray,
    proj: NDArray | None = None,
    gamma: float = 0.0,
) -> NDArray:
    """Predict each edge's signal from its OBSERVED co-participants.

    For edge e, aggregate over its observed co-participants e' with weight
    `exp(-gamma * ||z_e - z_e'||^2)`, where `z = inside @ proj` (or
    `z = inside` when proj is None): `pred_e = sum(w * signal_e') / sum(w)`.
    gamma=0 makes every weight 1, i.e. the uniform zero-parameter settle.
    An edge with no observed co-participant falls back to the global mean
    of the observed signal (or 0.0 if nothing at all is observed).
    """
    signal = np.asarray(signal, dtype=np.float64)
    obs_mask = np.asarray(obs_mask, dtype=bool)
    n_edges = signal.shape[0]

    z = None
    if gamma != 0.0:
        inside = np.asarray(inside, dtype=np.float64)
        if inside.ndim == 1:
            inside = inside.reshape(-1, 1)
        z = inside @ proj if proj is not None else inside

    global_mean = float(signal[obs_mask].mean()) if obs_mask.any() else 0.0

    pred = np.empty(n_edges, dtype=np.float64)
    for e in range(n_edges):
        nbrs = nbr_idx[nbr_ptr[e]:nbr_ptr[e + 1]]
        obs_nbrs = nbrs[obs_mask[nbrs]]
        if obs_nbrs.size == 0:
            pred[e] = global_mean
            continue
        if gamma == 0.0:
            pred[e] = float(signal[obs_nbrs].mean())
            continue
        diff = z[obs_nbrs] - z[e]
        sq_dist = np.einsum("ij,ij->i", diff, diff)
        w = np.exp(-gamma * sq_dist)
        w_sum = w.sum()
        pred[e] = float(np.dot(w, signal[obs_nbrs]) / w_sum) if w_sum > 0 else global_mean
    return pred


class CoParticipationAttention:
    """Co-participation attention whose compatibility is LEARNED, not hand-set.

    The `coparticipation_attention` function needs a `proj`/`gamma` handed to it;
    this class fits those two tiny parameters from the data itself, by
    self-supervised masked reconstruction, so the data supervises the fit
    rather than a hand-picked constant. The recipe:

    1. Split the OBSERVED edges into an inner-train and an inner-val subset
       (keyed on `seed`, sized by `mask_frac`).
    2. Predict inner-val's signal from inner-train ALONE, through the same
       matrix-free `coparticipation_attention` kernel.
    3. Minimize that inner-val reconstruction MSE over (proj, gamma) with a
       gradient-free optimizer (Nelder-Mead over the flattened, tiny
       parameter vector: a handful of numbers, not a network).

    The fitted params are then used by `predict` against the TRUE held-out
    set. Nothing here is a dense solve or an eigendecomposition: fitting is
    just repeated calls into the matrix-free kernel over a shrinking
    training/validation split of the observed edges.
    """

    def __init__(self, inside_dim: int, proj_dim: int = 2):
        self.inside_dim = int(inside_dim)
        self.proj_dim = int(proj_dim)
        # identity-like init: as close to "no rotation" as a possibly
        # non-square (inside_dim x proj_dim) matrix allows.
        proj = np.zeros((self.inside_dim, self.proj_dim), dtype=np.float64)
        for i in range(min(self.inside_dim, self.proj_dim)):
            proj[i, i] = 1.0
        self.proj = proj
        self.gamma = 0.0

    def fit_self_supervised(
        self,
        rex,
        inside: NDArray,
        signal: NDArray,
        obs_mask: NDArray | None = None,
        mask_frac: float = 0.2,
        seed: int = 0,
        steps: int = 300,
    ) -> CoParticipationAttention:
        """Fit (proj, gamma) by self-supervised masked reconstruction.

        `signal` is assumed zeroed at any edge not actually observed (the
        same sentinel convention `coparticipation_attention` already uses
        for its no-co-participant fallback); pass `obs_mask` explicitly to
        override that inference instead of relying on it. The observed
        edges are split (by `seed`) into an inner-train set and an
        inner-val set of size roughly `mask_frac` of the observed edges;
        the compatibility is optimized so that predicting inner-val from
        inner-train ALONE (via `coparticipation_attention`) minimizes the
        inner-val reconstruction MSE. The winning (proj, gamma) are stored
        on `self` and this instance is returned.
        """
        inside = np.asarray(inside, dtype=np.float64)
        if inside.ndim == 1:
            inside = inside.reshape(-1, 1)
        signal = np.asarray(signal, dtype=np.float64)
        n_edges = signal.shape[0]

        if obs_mask is not None:
            obs = np.asarray(obs_mask, dtype=bool)
        else:
            # sentinel inference: a genuinely observed edge whose signal is exactly
            # 0.0 would be misread as unobserved, so warn (pass obs_mask to be safe).
            import warnings
            warnings.warn(
                "CoParticipationAttention.fit_self_supervised: obs_mask not given; "
                "inferring observed edges as signal != 0.0, which misclassifies any "
                "truly-observed edge whose signal is exactly 0.0. Pass obs_mask explicitly.",
                stacklevel=2,
            )
            obs = signal != 0.0

        nbr_ptr, nbr_idx = coparticipation_neighbors(rex)

        rng = np.random.RandomState(seed)
        draw = rng.rand(n_edges)
        inner_val = obs & (draw < mask_frac)
        inner_train = obs & ~inner_val

        inside_dim, proj_dim = self.inside_dim, self.proj_dim
        n_proj = inside_dim * proj_dim

        def objective(x: NDArray) -> float:
            proj = x[:n_proj].reshape(inside_dim, proj_dim)
            gamma = float(x[n_proj]) ** 2  # square keeps gamma >= 0 without a hard bound
            pred = coparticipation_attention(
                nbr_ptr, nbr_idx, inside, signal, inner_train, proj=proj, gamma=gamma
            )
            resid = pred[inner_val] - signal[inner_val]
            return float(np.mean(resid * resid)) if resid.size else 0.0

        x0 = np.concatenate([self.proj.ravel(), [1.0]])  # start gamma at 1.0, not stuck at 0
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxfev": steps, "xatol": 1e-3, "fatol": 1e-6},
        )
        x_best = np.asarray(result.x, dtype=np.float64)

        self.proj = x_best[:n_proj].reshape(inside_dim, proj_dim).copy()
        self.gamma = float(x_best[n_proj]) ** 2
        return self

    def predict(
        self,
        rex_or_nbrs: object | tuple[NDArray, NDArray],
        inside: NDArray,
        signal: NDArray,
        obs_mask: NDArray,
    ) -> NDArray:
        """Predict every edge's signal from its OBSERVED co-participants.

        `rex_or_nbrs` accepts either a rex (co-participation neighbors are
        rebuilt) or an already-built `(nbr_ptr, nbr_idx)` pair (skips the
        rebuild), so a caller that already has the neighbors from `fit_self_
        supervised`'s rex does not have to recompute them.
        """
        if (
            isinstance(rex_or_nbrs, tuple)
            and len(rex_or_nbrs) == 2
            and all(isinstance(a, np.ndarray) for a in rex_or_nbrs)
        ):
            nbr_ptr, nbr_idx = rex_or_nbrs
        else:
            nbr_ptr, nbr_idx = coparticipation_neighbors(rex_or_nbrs)
        return coparticipation_attention(
            nbr_ptr, nbr_idx, inside, signal, obs_mask, proj=self.proj, gamma=self.gamma
        )
