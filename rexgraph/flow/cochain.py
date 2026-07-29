"""rexgraph.flow.cochain: an edge-cochain classifier trained THROUGH the co-participation channel.

The complex IS the model. The parameters are a bare cochain ``Z[nE, C]`` (forward = Z, one class
logit vector per edge, no features and no embeddings); the class is carried to the masked/unlabelled
edges not by the forward pass but by the OPTIMIZER, whose gradient is preconditioned by the complex's
unsigned co-participation Green's function. Two edges co-participate iff they share an incident vertex
(``abs(B1).T @ abs(B1)``, unsigned so a shared vertex never cancels); a BRANCHING vertex of arity K
(a target bound by K ligands) makes all K edges mutual co-participants, which is the arity>2 structure
a pairwise k-hop view cannot express. On such structure the co-participation channel takes masked-edge
classification from chance (where plain Adam is stuck, having sent no gradient to a masked edge) to
strong generalization: the optimizer itself propagates the training signal across the hyperedges.

The model exposes ``greens_groups()`` so ``make_optimizer("auto")`` routes it to ``GreensCochain``
automatically. Everything here is matrix-free and sparse: no signs, no dense operator, no eigensolve.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from rexgraph.core._sparse import to_scipy_csr

try:  # torch is an optional dependency (as elsewhere in rexgraph.nn)
    import torch as _torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover (env without torch)
    _HAS_TORCH = False

__all__ = ["coparticipation_adjacency"]


def coparticipation_adjacency(rex, restrict_vertices: Optional[NDArray] = None):
    """Normalised unsigned co-participation adjacency over the complex's edges, as a torch sparse
    tensor ``[nE, nE]`` ready to hand to ``GreensCochain`` as ``green_adj``.

    Built matrix-free from the signed incidence ``rex._B1_dual`` (nV x nE): ``abs(B1).T @ abs(B1)``
    is the (self-inclusive) co-participation adjacency over edges, since entry ``(e, e')`` is nonzero
    exactly when edges e and e' share an incident vertex. abs(B1) (not the signed B1) avoids a
    spurious zero from sign cancellation on a shared vertex; the diagonal is dropped, then the
    operator is renormalised ``D^-1/2 (C + I) D^-1/2`` so it is the low-pass operator GreensCochain
    expects. ``restrict_vertices`` (a boolean mask over vertices, or an array of vertex ids to keep)
    keeps only those vertices as connectors (e.g. the target side alone) for ablations. A branching
    vertex of arity K contributes a K-clique here, so arity>2 hyperedges are represented natively.
    """
    if not _HAS_TORCH:  # pragma: no cover (env without torch)
        raise ImportError("coparticipation_adjacency requires PyTorch (an optional dependency).")
    abs_b1 = abs(to_scipy_csr(rex._B1_dual)).tocsr()  # nV x nE, unsigned incidence
    if restrict_vertices is not None:
        rv = np.asarray(restrict_vertices)
        if rv.dtype != bool:
            keep = np.zeros(abs_b1.shape[0], dtype=bool)
            keep[rv.astype(np.int64)] = True
        else:
            keep = rv
        abs_b1 = abs_b1.multiply(keep.reshape(-1, 1)).tocsr()
    coparticip = (abs_b1.T @ abs_b1).tocsr()  # nE x nE, shared-vertex-count adjacency
    coparticip.setdiag(0)
    coparticip.eliminate_zeros()
    renorm = (coparticip + sp.eye(coparticip.shape[0])).tocoo()
    deg = np.asarray(coparticip.sum(1)).ravel() + 1.0  # self-loop-inclusive degree
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    a_hat = (sp.diags(dinv) @ renorm @ sp.diags(dinv)).tocoo()
    idx = np.vstack([a_hat.row, a_hat.col])
    return _torch.sparse_coo_tensor(
        idx, _torch.tensor(a_hat.data, dtype=_torch.float64), a_hat.shape
    ).coalesce()


if _HAS_TORCH:

    class CoParticipationCochain(_torch.nn.Module):
        """Edge-cochain classifier over a relational complex, trained through the co-participation
        Green's channel (see the module docstring). ``forward()`` returns the bare cochain ``Z``;
        ``greens_groups()`` hands GreensCochain the co-participation ``green_adj`` for that cochain, so
        ``make_optimizer("auto")`` routes training through the complex automatically.

        Args:
            rex: the relational complex (its ``_B1_dual`` defines co-participation).
            n_classes: number of edge classes.
            green_lam / green_iters / green_channel: the Green's-preconditioning knobs (``"low"`` is
                the co-participation smoother; the k-hop channels remain available for heterophily).
            restrict_vertices: optional connector-vertex restriction, for ablations.
        """

        def __init__(self, rex, n_classes, *, green_lam=4.0, green_iters=20,
                     green_channel="low", restrict_vertices=None, dtype=None):
            super().__init__()
            dtype = _torch.float64 if dtype is None else dtype
            self._rex = rex  # kept so the complex (and thus the operator) can re-serialize
            self._restrict = None if restrict_vertices is None else np.asarray(restrict_vertices)
            self._adj = coparticipation_adjacency(rex, restrict_vertices)
            n_edges = int(self._adj.shape[0])
            self.Z = _torch.nn.Parameter(_torch.zeros(n_edges, int(n_classes), dtype=dtype))
            self.green_lam = float(green_lam)
            self.green_iters = int(green_iters)
            self.green_channel = str(green_channel)

        def forward(self):
            return self.Z

        def greens_groups(self) -> List[dict]:
            return [{
                "params": [self.Z],
                "green_adj": self._adj,
                "green_channel": self.green_channel,
                "green_lam": self.green_lam,
                "green_iters": self.green_iters,
            }]

        def fit(self, labels, obs_mask, *, epochs=400, lr=0.3):
            """Train the cochain on the OBSERVED edges only; the optimizer propagates the class to the
            masked edges. Uses ``make_optimizer("auto")`` -> GreensCochain via ``greens_groups()``."""
            from rexgraph.nn.factory import make_optimizer

            labels_t = _torch.as_tensor(np.asarray(labels), dtype=_torch.long)
            obs_t = _torch.as_tensor(np.asarray(obs_mask), dtype=_torch.bool)
            opt, _label = make_optimizer("auto", self, self.parameters(), lr=lr)
            for _ in range(int(epochs)):
                opt.zero_grad()
                loss = _torch.nn.functional.cross_entropy(self.Z[obs_t], labels_t[obs_t])
                loss.backward()
                opt.step()
            return self

        @_torch.no_grad()
        def predict(self) -> NDArray:
            return self.Z.argmax(1).cpu().numpy()

        def save_safetensors(self, path):
            """Persist the model to ONE `.safetensors` file: the complex through the canonical
            rex-state serializer, the trained cochain and any connector restriction as namespaced
            extra tensors, and the Green's knobs as extra metadata. Reload with
            :meth:`load_safetensors`. Returns the written path."""
            from rexgraph.io.safetensors_bridge import rex_to_safetensors

            extra = {"cochain/Z": self.Z.detach().cpu().numpy()}
            if self._restrict is not None:
                # normalise to a full-length bool mask stored as uint8, so reload is unambiguous
                # (safetensors demotes bool->uint8, which coparticipation_adjacency would otherwise
                # misread as an index array).
                n_vertices = abs(to_scipy_csr(self._rex._B1_dual)).shape[0]
                r = self._restrict
                if r.dtype == bool:
                    mask = r
                else:
                    mask = np.zeros(n_vertices, dtype=bool)
                    mask[r.astype(np.int64)] = True
                extra["cochain/restrict"] = mask.astype(np.uint8)
            meta = {
                "kind": "CoParticipationCochain",
                "n_classes": int(self.Z.shape[1]),
                "green_lam": self.green_lam,
                "green_iters": self.green_iters,
                "green_channel": self.green_channel,
                "has_restrict": self._restrict is not None,
            }
            return rex_to_safetensors(self._rex, path, extra_tensors=extra, extra_meta=meta)

        @classmethod
        def load_safetensors(cls, path):
            """Rebuild a model saved by :meth:`save_safetensors`: reconstruct the complex, rebuild the
            co-participation operator (identical, since the complex round-trips losslessly), and load
            the trained cochain. The reloaded model predicts identically to the saved one."""
            from rexgraph.io.safetensors_bridge import load_safetensors, load_extra

            full = load_safetensors(path)
            meta = load_extra(path)
            if meta.get("kind") != "CoParticipationCochain":
                raise TypeError(
                    f"{path}: not a CoParticipationCochain (kind={meta.get('kind')!r})")
            tensors = full["tensors"]
            # stored as a uint8 bool mask; cast back to bool so it is read as a mask, not indices
            restrict = tensors["cochain/restrict"].astype(bool) if meta.get("has_restrict") else None
            model = cls(
                full["object"], meta["n_classes"],
                green_lam=meta["green_lam"], green_iters=meta["green_iters"],
                green_channel=meta["green_channel"], restrict_vertices=restrict,
            )
            z = np.asarray(tensors["cochain/Z"])
            model.Z.data = _torch.as_tensor(z, dtype=model.Z.dtype)
            return model

    __all__.append("CoParticipationCochain")
