"""rexgraph.flow.online: native online predict-then-observe field, no external
deep-learning framework dependency.

GreensCochainField is the owner's Green's-cochain math applied ONLINE as one
propagate (predict) plus one relational correction (update) per event, over the
co-participation Laplacian L_C. State is a cochain phi over edges keyed by the
canonical cell key so it survives index shifts. No epochs, no learning rate, no
embeddings, no argmax classifier: the update is a matrix-free Green's solve via
_block_cg and one preconditioned residual step, pure numpy/scipy.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from rexgraph.core._temporal import cell_keys_of
from rexgraph.sparse_character import _block_cg, build_sparse_channels

__all__ = ["GreensCochainField", "edge_persistence"]


def _keys_of(rex) -> NDArray:
    rex._ensure_clean()
    return cell_keys_of(rex._boundary_ptr, rex._boundary_idx, rex._directed)


def _sparse_L_C(rex):
    """The co-participation line-graph Laplacian as scipy CSR, or None if the
    line graph has no edges (trace-zero channel dropped). `rex.L_coPC` only
    reads the DENSE bundle (opt-in, O(nE^2)); the scale-free accessor is the
    per-channel sparse builder, matching rexgraph.tests.test_relational's own
    `dict(build_sparse_channels(g)).get('L_C')` idiom."""
    return dict(build_sparse_channels(rex)).get('L_C')


def edge_persistence(region, rex, rex_next) -> NDArray:
    """Structural, domain-agnostic default observe target: for each edge index in
    `region` (in `rex`), 1.0 if its canonical cell key is still present in
    `rex_next`, else 0.0. Supervised entirely by the stream's own evolution."""
    k = _keys_of(rex)
    next_keys = set(_keys_of(rex_next).tolist())
    region = np.asarray(region, dtype=np.int64)
    return np.asarray([1.0 if int(k[i]) in next_keys else 0.0 for i in region.tolist()],
                      dtype=np.float64)


class GreensCochainField:
    """A cochain field phi over edges, propagated and corrected by the Green's
    function over L_C. One propagate + one correction per event, matrix-free."""

    def __init__(self, *, green_lam: float = 4.0, green_iters: int = 20,
                 observe: Optional[Callable] = None):
        self.green_lam = float(green_lam)
        self.green_iters = int(green_iters)
        self.observe = observe if observe is not None else edge_persistence
        self.phi: Dict[int, float] = {}
        self._pending = None                          # (region_indices, rex_at_predict)

    def _sparse_L_C(self, rex):
        """Instance-level bounded cache over the module-level `_sparse_L_C(rex)`
        builder. Keyed by id(rex); the cache entry holds a STRONG reference to
        `rex` alongside its L_C, so while cached the id cannot be reused by another
        object and the identity check (`hit[0] is rex`) is collision-free. Bounded
        to 4 entries (evict oldest) so it never grows unbounded across a long run."""
        from collections import OrderedDict
        cache = self.__dict__.setdefault("_lc_cache", OrderedDict())   # id(rex) -> (rex, L_C)
        key = id(rex)
        hit = cache.get(key)
        if hit is not None and hit[0] is rex:                          # identity-verified hit
            cache.move_to_end(key)
            return hit[1]
        L = _sparse_L_C(rex)                                           # the existing module-level accessor
        cache[key] = (rex, L)
        cache.move_to_end(key)
        while len(cache) > 4:                                          # bounded
            cache.popitem(last=False)
        return L

    def _phi_vec(self, keys) -> NDArray:
        return np.array([self.phi.get(int(k), 0.0) for k in keys], dtype=np.float64)

    def _write_back(self, keys, vec) -> None:
        for k, x in zip(keys.tolist(), vec.tolist()):
            self.phi[int(k)] = float(x)

    def _greens_apply(self, L, seed) -> NDArray:
        """Solve (I + green_lam * L) x = seed via the native matrix-free block CG."""
        lam = self.green_lam
        apply_A = lambda P: P + lam * (L @ P)
        dinv = 1.0 / (1.0 + lam * L.diagonal())
        return _block_cg(apply_A, seed[:, None], dinv, maxit=self.green_iters)[:, 0]

    def predict(self, rex, region) -> NDArray:
        """Green's-propagate the settled field onto the region and record it BEFORE
        observation. Writes the propagated field back into phi by key; returns the
        predicted values at `region`."""
        keys = _keys_of(rex)
        phi_vec = self._phi_vec(keys)
        region = np.asarray(region, dtype=np.int64)
        L = self._sparse_L_C(rex)
        if L is None or region.size == 0:
            propagated = phi_vec                      # no line graph -> propagation is identity
        else:
            propagated = self._greens_apply(L, phi_vec)
        self._write_back(keys, propagated)
        return propagated[region] if region.size else np.zeros(0, dtype=np.float64)

    def correct(self, rex, region, target) -> Dict[str, object]:
        """One Green's-preconditioned relational correction of phi toward `target`
        over `region`. Reports pred/target/error(updated)."""
        keys = _keys_of(rex)
        phi_vec = self._phi_vec(keys)
        region = np.asarray(region, dtype=np.int64)
        target = np.asarray(target, dtype=np.float64)
        pred = phi_vec[region] if region.size else np.zeros(0, dtype=np.float64)
        residual = target - pred
        res_full = np.zeros(phi_vec.shape[0], dtype=np.float64)
        if region.size:
            res_full[region] = residual
        L = self._sparse_L_C(rex)
        if L is None or region.size == 0:
            step = res_full
        else:
            step = self._greens_apply(L, res_full)
        phi_vec = phi_vec + step
        self._write_back(keys, phi_vec)
        err = float(np.abs(residual).mean()) if region.size else 0.0
        return {"pred": pred, "target": target, "error": err, "updated": True}

    def predict_then_observe(self, t, change, rex) -> Dict[str, object]:
        """Predict at t (recorded before observation), then observe + correct the
        PREVIOUS step's pending region against the realized snapshot at t. Keyed by
        canonical cell key so a renumber-free index shift never mis-pairs. `rex` is
        the caller's already-materialized snapshot at t (no at() call here)."""
        if change is None:
            added = np.arange(rex.nE, dtype=np.int64)
        else:
            added = np.asarray(change.added, dtype=np.int64)
        pred = self.predict(rex, added)
        result = {"pred": pred, "target": None, "error": None,
                  "updated": False, "t": t, "region": added}
        if self._pending is not None:
            prev_region, rex_prev = self._pending
            target = self.observe(prev_region, rex_prev, rex)
            corr = self.correct(rex_prev, prev_region, target)
            result["target"] = corr["target"]
            result["error"] = corr["error"]
            result["updated"] = corr["updated"]
        self._pending = (added, rex)
        return result
