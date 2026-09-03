"""The cache layout every array backend writes, and the code that walks it.

`RexHDF5Format` and `RexZarrFormat` held byte-identical copies of this: the same
thirteen cache groups and the same thirteen methods for writing them, 76 statements
duplicated across two files. Two copies of one rule is how a fix lands in one of them,
which is the whole reason this is one file now.

What is genuinely per-backend stays per-backend. These methods decide WHAT is written
and in what shape; the subclass decides HOW bytes reach the store, through the seam:

    _store(group, name, arr)          write one array
    _store_chunked(group, name, arr)  the same, for arrays past `large_threshold`
    _load(group, name)                read one array back
    _has(group, name)                 is it there
    _get_or_create(group, name)       a subgroup, made if absent
    large_threshold                   nE past which chunking is used

plus four writers whose payloads differ by backend and so are NOT here:
`_write_spectral_cache`, `_write_faces_cache`, `_write_field_cache`,
`_write_standard_metrics_cache`, and `_write_temporal_cache`.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

import numpy as np

from rexgraph.io._compat import as_str, dumps, to_native

if TYPE_CHECKING:
    from rexgraph.graph import RexGraph


_CACHE_GROUPS: dict[str, list[str]] = {
    "algebra": [
        "B1", "B2", "L0", "L1", "L2",
        "L1_down", "L1_up",
        "L_overlap",
    ],
    "spectral": [
        "spectral_bundle",
        "eigenvalues_L0", "fiedler_vector_L0",
        "evals_L1", "evecs_L1",
        "evals_L2",
        "evals_L_O", "evecs_L_O",
        "diag_L1_down", "diag_L1_up",
        "fiedler_overlap",
        "layout", "layout_3d",
    ],
    "relational": [
        "relational_laplacian",
        "evals_RL1", "evecs_RL1",
        "alpha_G", "alpha_T",
        "L1_alpha", "evals_L1a", "evecs_L1a",
        "Lambda", "evals_Lambda", "evecs_Lambda",
    ],
    "topology": [
        "betti", "euler_characteristic", "chain_valid",
        "edge_types", "cycle_basis", "harmonic_space",
        "nF_hodge", "self_loop_face_indices", "B2_hodge",
    ],
    "hodge": [
        "hodge_decomposition",
        "hodge_rho",
    ],
    "faces": [
        "detected_faces", "face_metrics",
    ],
    "field": [
        "field_operator", "field_eigen", "mode_classification",
    ],
    "signal": [
        "perturbation_result", "field_perturbation_result",
    ],
    "temporal": [
        "edge_lifecycle", "edge_metrics",
        "face_lifecycle", "bioes_result",
        "betti_matrix",
    ],
    "standard_metrics": [
        "standard_metrics",
    ],
}

_ALL_CACHEABLE: set[str] = set()
for _entries in _CACHE_GROUPS.values():
    _ALL_CACHEABLE.update(_entries)


class CacheLayoutMixin:
    """The backend-independent half of an array-format writer."""

    def _is_large(self, rex) -> bool:
        return rex.nE >= self.large_threshold

    def _resolve_cache_names(self, cache) -> set[str]:
        """Expand cache spec into individual property names."""
        if cache is None:
            return set()
        if isinstance(cache, str):
            if cache == "all":
                return set(_ALL_CACHEABLE)
            if cache in _CACHE_GROUPS:
                return set(_CACHE_GROUPS[cache])
            return {cache}
        out: set[str] = set()
        for c in cache:
            if c == "all":
                return set(_ALL_CACHEABLE)
            if c in _CACHE_GROUPS:
                out.update(_CACHE_GROUPS[c])
            else:
                out.add(c)
        return out

    def _write_rex_graph(self, g, rex, *, cache=None) -> None:
        """Serialize a RexGraph to an HDF5 group via the canonical rex state.

        Every tensor goes through the one rex-state encoder (`to_state`), so the on-disk
        reconstruction contract cannot drift from `.rex`, arrow, and safetensors. Dataset names
        are `fname_encode`d because h5py treats '/' as a group separator, and nested-rex tensor
        names legitimately contain '/'.
        """
        from .rex_state import fname_encode, to_state

        large = self._is_large(rex)
        st = to_state(rex)
        for name, arr in st.tensors.items():
            store_fn = self._store_chunked if large else self._store
            store_fn(g, fname_encode(name), np.asarray(arr))
        g.attrs["rex_state_header"] = dumps(st.header)
        g.attrs["tensor_names"] = json.dumps(list(st.tensors.keys()))

        if cache:
            self._write_cache(g, rex, cache, large)

    def _read_rex_graph(self, g) -> RexGraph:
        """Reconstruct a RexGraph from an HDF5 group via the canonical rex state."""
        from .rex_state import RexState, fname_encode, from_state

        hdr = json.loads(as_str(g.attrs["rex_state_header"]))
        names = json.loads(as_str(g.attrs["tensor_names"]))
        tensors = {name: self._load(g, fname_encode(name)) for name in names}
        return from_state(RexState(tensors, hdr))

    def _write_temporal_rex(self, g, trex, *, cache=None) -> None:
        """Serialize a TemporalRex with all snapshots and optional cache."""
        T = trex.T
        g.attrs["T"] = T
        g.attrs["directed"] = bool(trex._directed)
        g.attrs["general"] = bool(trex._general)
        g.attrs["relation_id_snapshots"] = dumps([
            value is not None for value in getattr(trex, "_snapshot_relation_ids", ())
        ])

        sg = g.create_group("snapshots")
        for t in range(T):
            tg = sg.create_group(str(t))
            snap = trex._snapshots[t]
            if trex._general:
                self._store(tg, "boundary_ptr", snap[0])
                self._store(tg, "boundary_idx", snap[1])
            else:
                self._store(tg, "sources", snap[0])
                self._store(tg, "targets", snap[1])
            relation_ids = trex._snapshot_relation_ids[t]
            if relation_ids is not None:
                self._store(tg, "relation_ids", relation_ids)

        if trex._face_snapshots:
            fg = g.create_group("face_snapshots")
            for t, fsnap in enumerate(trex._face_snapshots):
                ftg = fg.create_group(str(t))
                self._store(ftg, "B2_col_ptr", fsnap[0])
                self._store(ftg, "B2_row_idx", fsnap[1])

        if cache:
            rex_final = trex.at(T - 1)
            self._write_cache(g, rex_final, cache, self._is_large(rex_final))
            self._write_temporal_cache(g, trex, cache)

    def _write_cache(self, g, rex, cache, large: bool) -> None:
        """Write precomputed properties into subgroups."""
        names = self._resolve_cache_names(cache)
        if not names:
            return

        store_fn = self._store_chunked if large else self._store

        self._write_algebra_cache(g, rex, names, store_fn)
        self._write_spectral_cache(g, rex, names, store_fn)
        self._write_relational_cache(g, rex, names, store_fn)
        self._write_topology_cache(g, rex, names, store_fn)
        self._write_hodge_cache(g, rex, names)
        self._write_faces_cache(g, rex, names)
        self._write_field_cache(g, rex, names, store_fn)
        self._write_signal_cache(g, rex, names)
        self._write_standard_metrics_cache(g, rex, names)

    def _write_algebra_cache(self, g, rex, names, store_fn) -> None:
        algebra_props = {"B1", "B2", "L0", "L1", "L2",
                         "L1_down", "L1_up",
                         "L_overlap"}
        if not (names & (algebra_props | {"algebra"})):
            return
        ag = self._get_or_create(g, "algebra")
        for prop in algebra_props:
            if prop in names or "algebra" in names:
                try:
                    arr = getattr(rex, prop)
                    if arr is not None:
                        store_fn(ag, prop, arr)
                except Exception:
                    pass

    def _write_relational_cache(self, g, rex, names, store_fn) -> None:
        rel_props = {"relational_laplacian",
                     "evals_RL1", "evecs_RL1",
                     "alpha_G", "alpha_T",
                     "L1_alpha", "evals_L1a", "evecs_L1a",
                     "Lambda", "evals_Lambda", "evecs_Lambda"}
        if not (names & (rel_props | {"relational"})):
            return
        rg = self._get_or_create(g, "relational")

        if "relational_laplacian" in names or "relational" in names:
            try:
                rl = rex.rex_laplacian
                if rl is not None:
                    store_fn(rg, "RL_1", rl)
            except Exception:
                pass

        sb_keys = {
            "evals_RL1": "evals_RL_1",
            "evecs_RL1": "evecs_RL_1",
            "L1_alpha": "L1_alpha",
            "evals_L1a": "evals_L1a",
            "evecs_L1a": "evecs_L1a",
            "Lambda": "Lambda",
            "evals_Lambda": "evals_Lambda",
            "evecs_Lambda": "evecs_Lambda",
        }
        for cache_name, sb_key in sb_keys.items():
            if cache_name in names or "relational" in names:
                try:
                    val = rex.spectral_bundle.get(sb_key)
                    if val is not None:
                        store_fn(rg, cache_name, val)
                except Exception:
                    pass

        if "alpha_G" in names or "relational" in names:
            try:
                sb = rex.spectral_bundle
                rg.attrs["alpha_G"] = to_native(sb.get("alpha_G", float("nan")))
                rg.attrs["alpha_T"] = to_native(sb.get("alpha_T", 0.0))
                rg.attrs["alpha_used"] = to_native(sb.get("alpha_used", float("nan")))
            except Exception:
                pass

    def _write_topology_cache(self, g, rex, names, store_fn) -> None:
        topo_props = {"betti", "euler_characteristic", "chain_valid",
                      "edge_types", "cycle_basis", "harmonic_space",
                      "nF_hodge", "self_loop_face_indices", "B2_hodge"}
        if not (names & (topo_props | {"topology"})):
            return
        tg = self._get_or_create(g, "topology")

        if "betti" in names or "topology" in names:
            try:
                b0, b1, b2 = rex.betti
                tg.attrs["betti"] = json.dumps([b0, b1, b2])
            except Exception:
                pass

        if "euler_characteristic" in names or "topology" in names:
            with contextlib.suppress(Exception):
                tg.attrs["euler_characteristic"] = int(rex.euler_characteristic)

        if "chain_valid" in names or "topology" in names:
            with contextlib.suppress(Exception):
                tg.attrs["chain_valid"] = bool(rex.chain_valid)

        if "edge_types" in names or "topology" in names:
            with contextlib.suppress(Exception):
                self._store(tg, "edge_types", rex.edge_types)

        if "cycle_basis" in names or "topology" in names:
            try:
                cycles = rex.cycle_basis
                cbg = self._get_or_create(tg, "cycle_basis")
                cbg.attrs["n_cycles"] = len(cycles)
                for i, cyc in enumerate(cycles):
                    self._store(cbg, f"c{i}", np.asarray(cyc, dtype=np.int32))
            except Exception:
                pass

        if "harmonic_space" in names or "topology" in names:
            with contextlib.suppress(Exception):
                store_fn(tg, "harmonic_space", rex.harmonic_space)

        if "nF_hodge" in names or "topology" in names:
            with contextlib.suppress(Exception):
                tg.attrs["nF_hodge"] = int(rex.nF_hodge)

        if "self_loop_face_indices" in names or "topology" in names:
            try:
                indices = rex.self_loop_face_indices
                tg.attrs["self_loop_face_indices"] = json.dumps(
                    [int(i) for i in indices]
                )
            except Exception:
                pass

        if "B2_hodge" in names or "topology" in names:
            with contextlib.suppress(Exception):
                store_fn(tg, "B2_hodge", rex.B2_hodge)

    def _write_hodge_cache(self, g, rex, names) -> None:
        hodge_props = {"hodge_decomposition", "hodge_rho"}
        if not (names & (hodge_props | {"hodge"})):
            return

        if "hodge_decomposition" in names or "hodge" in names:
            try:
                w = rex.w_E if rex.w_E is not None else np.ones(rex.nE)
                grad, curl, harm = rex.hodge(w)
                hg = self._get_or_create(g, "hodge")
                self._store(hg, "gradient", grad)
                self._store(hg, "curl", curl)
                self._store(hg, "harmonic", harm)
                total = np.dot(w, w)
                if total > 0:
                    hg.attrs["pct_gradient"] = float(np.dot(grad, grad) / total)
                    hg.attrs["pct_curl"] = float(np.dot(curl, curl) / total)
                    hg.attrs["pct_harmonic"] = float(np.dot(harm, harm) / total)

                try:
                    analysis = rex.hodge_full(w)
                    if isinstance(analysis, dict) and "rho" in analysis:
                        self._store(hg, "rho", analysis["rho"])
                    elif hasattr(analysis, "rho"):
                        self._store(hg, "rho", analysis.rho)
                except Exception:
                    pass
            except Exception:
                pass

        if "hodge_rho" in names and not self._has(g, "hodge"):
            try:
                w = rex.w_E if rex.w_E is not None else np.ones(rex.nE)
                analysis = rex.hodge_full(w)
                hg = self._get_or_create(g, "hodge")
                if isinstance(analysis, dict) and "rho" in analysis:
                    self._store(hg, "rho", analysis["rho"])
                elif hasattr(analysis, "rho"):
                    self._store(hg, "rho", analysis.rho)
            except Exception:
                pass

    def _write_signal_cache(self, g, rex, names) -> None:
        signal_props = {"perturbation_result", "field_perturbation_result"}
        if not (names & (signal_props | {"signal"})):
            return
        self._get_or_create(g, "signal")
