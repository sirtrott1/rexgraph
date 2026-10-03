"""Edge cochain classification through the co participation operator.

The parameters are class logits Z[nE, C], one row per relation. forward returns Z.
greens_groups supplies the adjacency to GreensCochain, so observed labels update
the cochain through the Green preconditioner. The default adjacency uses Core's
share reading; an explicit connector restriction uses the unsigned incidence Gram.
Both paths normalize the sparse operator after replacing its diagonal with self loops.
The operator reads B1 and does not read faces in B2.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from rexgraph.core._sparse import to_scipy_csr

try:  # torch is an optional dependency (as elsewhere in rexgraph.nn)
    import torch as _torch

    _HAS_TORCH = True
    from rexgraph.compute import sparse_coo_tensor as _sparse_coo_tensor
except Exception:  # pragma: no cover (env without torch)
    _HAS_TORCH = False

__all__ = ["coparticipation_adjacency"]


def coparticipation_adjacency(rex, restrict_vertices: NDArray | None = None):
    """Return a normalized sparse Torch adjacency over relations, of shape [nE, nE].

    Without restrict_vertices, use rex.overlap_share_sparse. With a boolean mask or
    vertex index array, restrict abs(B1) to the selected connectors and form its Gram.
    Drop the Gram diagonal, add self loops, and return D^-1/2 (C + I) D^-1/2.
    This operator depends on B1 only. flow_adjacency supplies the separate B2 reading.
    """
    if not _HAS_TORCH:  # pragma: no cover (env without torch)
        raise ImportError("coparticipation_adjacency requires PyTorch (an optional dependency).")
    if restrict_vertices is None:
        # Use the share operator independently of the complex's c_channel selector.
        coparticip = rex.overlap_share_sparse.tocsr().copy()
    else:
        abs_b1 = abs(to_scipy_csr(rex._B1_dual)).tocsr()  # nV x nE, unsigned incidence
        rv = np.asarray(restrict_vertices)
        if rv.dtype != bool:
            keep = np.zeros(abs_b1.shape[0], dtype=bool)
            keep[rv.astype(np.int64)] = True
        else:
            keep = rv
        # restricting the CONNECTORS is not a sub operator of the full Gramian, so this
        # branch has to form its own
        abs_b1 = abs_b1.multiply(keep.reshape(-1, 1)).tocsr()
        coparticip = (abs_b1.T @ abs_b1).tocsr()
    coparticip.setdiag(0)
    coparticip.eliminate_zeros()
    renorm = (coparticip + sp.eye(coparticip.shape[0])).tocoo()
    deg = np.asarray(coparticip.sum(1)).ravel() + 1.0  # self loop inclusive degree
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    a_hat = (sp.diags(dinv) @ renorm @ sp.diags(dinv)).tocoo()
    idx = np.vstack([a_hat.row, a_hat.col])
    return _sparse_coo_tensor(
        idx, _torch.tensor(a_hat.data, dtype=_torch.float64), a_hat.shape
    ).coalesce()


if _HAS_TORCH:

    class CoParticipationCochain(_torch.nn.Module):
        """Edge cochain classifier over a relational complex, trained through the co participation
        Green's channel (see the module docstring). ``forward()`` returns the bare cochain ``Z``;
        ``greens_groups()`` hands GreensCochain the co participation ``green_adj`` for that cochain, so
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
            self._rex = rex  # kept so the complex (and thus the operator) can re serialize
            self._restrict = None if restrict_vertices is None else np.asarray(restrict_vertices)
            self.register_buffer("_adj", coparticipation_adjacency(rex, restrict_vertices).to(dtype=dtype),
                                 persistent=False)
            n_edges = int(self._adj.shape[0])
            self.Z = _torch.nn.Parameter(_torch.zeros(n_edges, int(n_classes), dtype=dtype))
            self.green_lam = float(green_lam)
            self.green_iters = int(green_iters)
            self.green_channel = str(green_channel)

        def forward(self):
            return self.Z

        def greens_groups(self) -> list[dict]:
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

            labels_t = _torch.as_tensor(labels, device=self.Z.device)
            obs_t = _torch.as_tensor(obs_mask, device=self.Z.device)
            if (labels_t.shape != (self.Z.shape[0],) or labels_t.dtype not in
                    (_torch.int8, _torch.int16, _torch.int32, _torch.int64, _torch.uint8)):
                raise ValueError("cochain labels must be one integer class index per edge")
            if obs_t.shape != labels_t.shape or obs_t.dtype != _torch.bool or not obs_t.any():
                raise ValueError("cochain training requires a nonempty boolean observation mask")
            labels_t = labels_t.long()
            if ((labels_t[obs_t] < 0) | (labels_t[obs_t] >= self.Z.shape[1])).any():
                raise ValueError("observed cochain labels lie outside the declared classes")
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

        def checkpoint(self, *, optimizer=None, optimizer_spec=None, axes=None, reference=None, step=0):
            """Capture native source, parameters and optional training continuation."""
            from rexgraph.model_contract import capture_model_checkpoint
            config = {"n_classes": int(self.Z.shape[1]), "green_lam": self.green_lam,
                      "green_iters": self.green_iters, "green_channel": self.green_channel,
                      "restrict_vertices": self._restrict, "dtype": str(self.Z.dtype).split(".")[-1]}
            return capture_model_checkpoint(self, self._rex, configuration=config, optimizer=optimizer,
                optimizer_spec=optimizer_spec, axes=axes, reference=reference, step=step)

        def save_safetensors(self, path):
            """Persist the model to ONE `.safetensors` file: the complex through the canonical
            rex state serializer, the trained cochain and any connector restriction as namespaced
            extra tensors, and the Green's knobs as extra metadata. Reload with
            :meth:`load_safetensors`. Returns the written path."""
            from rexgraph.artifacts import save_rex_artifact

            extra = {"cochain/Z": self.Z.detach().cpu().numpy()}
            if self._restrict is not None:
                # normalise to a full length bool mask stored as uint8, so reload is unambiguous
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
            return save_rex_artifact(self._rex, path, tensors=extra, metadata=meta, format="safetensors")

        @classmethod
        def load_safetensors(cls, path):
            """Rebuild a model saved by :meth:`save_safetensors`: reconstruct the complex, rebuild the
            co participation operator (identical, since the complex round trips losslessly), and load
            the trained cochain. The reloaded model predicts identically to the saved one."""
            from rexgraph.artifacts import load_rex_artifact

            artifact = load_rex_artifact(path, format="safetensors")
            meta = dict(artifact.metadata)
            if meta.get("kind") != "CoParticipationCochain":
                raise TypeError(
                    f"{path}: not a CoParticipationCochain (kind={meta.get('kind')!r})")
            tensors = artifact.tensors
            # stored as a uint8 bool mask; cast back to bool so it is read as a mask, not indices
            restrict = tensors["cochain/restrict"].astype(bool) if meta.get("has_restrict") else None
            z = _torch.as_tensor(np.asarray(tensors["cochain/Z"]))
            model = cls(
                artifact.object, meta["n_classes"],
                green_lam=meta["green_lam"], green_iters=meta["green_iters"],
                green_channel=meta["green_channel"], restrict_vertices=restrict, dtype=z.dtype,
            )
            if z.shape != model.Z.shape or not z.is_floating_point() or not _torch.isfinite(z).all():
                raise ValueError("stored cochain values do not match the declared model")
            with _torch.no_grad():
                model.Z.copy_(z)
            return model

    __all__.append("CoParticipationCochain")
