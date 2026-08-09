"""
Analysis pipeline with progressive stage reporting.

Breaks the analyze() computation into stages so intermediate results
can be streamed to the frontend as they complete. Each stage triggers
a set of @cached_property accesses on the RexGraph.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import numpy as np


def _sparse_L0(rex):
    """Graph Laplacian L0 = B1 @ B1^T as a SPARSE scipy matrix, built directly
    from edge endpoints (L0 = D - A).

    We deliberately DO NOT use ``rex.L0`` or the core ``build_L0``: both allocate
    a dense array (nV x nV, and internally nV x nE for B1), which OOMs on large
    graphs - even though L0 itself is sparse (nnz ~ 2*nE). Building D - A directly
    keeps it sparse and is identical to B1 B1^T for the standard incidence.
    """
    try:
        from scipy import sparse
    except Exception:
        return None
    nV = int(getattr(rex, "nV", 0) or 0)
    src = getattr(rex, "sources", None)
    tgt = getattr(rex, "targets", None)
    if nV >= 1 and src is not None and tgt is not None:
        src = np.asarray(src).ravel()
        tgt = np.asarray(tgt).ravel()
        if src.size and tgt.size == src.size:
            w = getattr(rex, "w_E", None)
            if w is not None and np.size(w) == src.size:
                w = np.abs(np.asarray(w, dtype=float)).ravel()
            else:
                w = np.ones(src.size, dtype=float)
            A = sparse.coo_matrix(
                (np.concatenate([w, w]),
                 (np.concatenate([src, tgt]), np.concatenate([tgt, src]))),
                shape=(nV, nV)).tocsr()
            deg = np.asarray(A.sum(axis=1)).ravel()
            return (sparse.diags(deg) - A).tocsr()
    # Fallback: an already-sparse L0 supplied on the object (e.g. tests, or
    # callers holding a sparse Laplacian). Use it ONLY if it's already sparse so
    # we never trigger the dense nV x nV materialization of rex.L0.
    L0 = getattr(rex, "L0", None)
    if L0 is not None and sparse.issparse(L0):
        return sparse.csr_matrix(L0)
    return None


def _smallest_eigenvalues_L0(rex, k: int = 20) -> np.ndarray | None:
    """Return the k smallest eigenvalues of the graph Laplacian L0 (0, then the
    Fiedler value, ...), computed with a SPARSE truncated solver on a sparse L0.

    Never forms a dense Laplacian. Work is bounded (capped iterations); if the
    solver can't converge cheaply on a very large graph it returns ``None`` and
    the caller treats the (optional) spectral indicators as unavailable, rather
    than OOM-ing or hanging.
    """
    try:
        from scipy.sparse.linalg import eigsh
    except Exception:
        return None
    L = _sparse_L0(rex)
    if L is None:
        return None
    n = L.shape[0]
    k = int(max(1, min(k, n - 1)))
    if n <= 3 or k >= n - 1:
        try:
            ev = np.linalg.eigvalsh(L.toarray())  # tiny graphs only
            return np.sort(ev)[:k] if k < len(ev) else np.sort(ev)
        except Exception:
            return None
    maxiter = min(20000, max(2000, 50 * k))
    attempts = []
    if n <= 3000:
        # shift-invert converges fastest for the low end and its LU is cheap for
        # small graphs (the common case: documents/schemas). Tiny negative shift
        # keeps L0 (which is singular) factorizable. Above this size the LU fill-in
        # can be heavy, so we go straight to the matrix-free solver instead.
        attempts.append({"sigma": -1e-6, "which": "LM"})
    attempts.append({"which": "SA"})  # matrix-free, bounded memory, for big graphs
    for kwargs in attempts:
        try:
            ev = eigsh(L, k=k, return_eigenvectors=False, maxiter=maxiter,
                       tol=1e-4, **kwargs)
            return np.sort(np.real(ev))
        except Exception:
            continue
    return None


def _strain_equilibrium_sparse(rex, kappa_f, born_face):
    """strain_equilibrium (Def 5.2-5.3) via SPARSE B1/B2 matvecs - no dense boundary
    operators, O(nnz), scale-free. Mirrors `_rcfe.strain_equilibrium`:
      alpha = <B2 κ, B2 pF> / ||B2 pF||²,  δ = face_deficit(κ, alpha, pF),
      σ = B2 δ  (relational strain),  Bianchi: B1 σ = 0."""
    from rexgraph.core._rcfe import face_deficit
    from rexgraph.core._sparse import to_scipy_csr
    nF = int(rex.nF_hodge)
    kappa_f = np.asarray(kappa_f, dtype=np.float64)
    born_face = np.asarray(born_face, dtype=np.float64)
    if nF == 0 or rex._B2_hodge_dual is None:
        z = np.zeros(int(rex.nE), dtype=np.float64)
        return {'alpha': 0.0, 'delta': np.zeros(0), 'sigma': z,
                'bianchi_ok': True, 'bianchi_residual': 0.0, 'strain_norm': 0.0}
    B2 = to_scipy_csr(rex._B2_hodge_dual).tocsr()          # nE × nF (sparse)
    B1 = to_scipy_csr(rex._B1_dual).tocsr()                # nV × nE (sparse)
    B2k = np.asarray(B2 @ kappa_f).ravel()
    B2p = np.asarray(B2 @ born_face).ravel()
    denom = float(B2p @ B2p)
    alpha = float((B2k @ B2p) / denom) if denom > 1e-15 else 0.0
    delta = np.asarray(face_deficit(kappa_f, alpha, born_face, nF), dtype=np.float64)
    sigma = np.asarray(B2 @ delta).ravel()                 # σ = B2 δ
    residual = np.asarray(B1 @ sigma).ravel()              # B1 σ (Bianchi)
    max_res = float(np.max(np.abs(residual))) if residual.size else 0.0
    return {
        'alpha': alpha, 'delta': delta, 'sigma': sigma,
        'bianchi_ok': max_res < 1e-10, 'bianchi_residual': max_res,
        'strain_norm': float(np.sqrt(sigma @ sigma)),
    }


def _attributed_kappa_sparse(rex, w_e=None):
    """Per-face attributed curvature κ_f = ||R[:,f]|| with R = B1·diag(w)·B2 - the
    weighted chain residual (Part F; = 0 when w is uniform, since B1B2=0), via sparse
    matvecs, O(nnz). Matches rex.attributed_curvature()['kappa_f'] (vertex
    amplitudes a_v = 1). No dense B1w/B2w/R."""
    import scipy.sparse as sp
    from rexgraph.core._sparse import to_scipy_csr
    nE, nF = int(rex.nE), int(rex.nF_hodge)
    if nF == 0:
        return np.zeros(0, dtype=np.float64)
    w = (np.ones(nE, dtype=np.float64) if w_e is None
         else np.asarray(w_e, dtype=np.float64).ravel())
    B1 = to_scipy_csr(rex._B1_dual).tocsr()               # nV × nE
    B2 = to_scipy_csr(rex._B2_hodge_dual).tocsr()         # nE × nF
    R = (B1 @ (sp.diags(w) @ B2)).tocsc()                 # nV × nF (sparse)
    return np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel())


class AnalysisPipeline:
    """Progressive analysis pipeline for a RexGraph.

    Stages:
        1. construction: nV, nE, nF, edge types, chain_valid
        2. topology: betti, euler, dimension
        3. spectral: eigenvalues, fiedler, coupling_constants
        4. relational: RL, structural_character, coherence
        5. hodge - gradient/curl/harmonic fractions
        6. void: void complex, eta, fills_beta, void_strain
        7. epsilon: chain condition, vertex excess, equiweight violation
        8. advanced: persistence, field, dirac (optional)

    Usage:
        pipe = AnalysisPipeline(rex)
        pipe.on_stage(lambda name, data: print(f"{name}: {data}"))
        results = pipe.run(depth='standard')
    """

    #: `drawing` is first, and in every depth, because a picture of what was constructed
    #: is the cheapest thing the pipeline can say and the one a reader wants before any
    #: number. It reads the same payload `rexgraph_render` does, so what the pipeline
    #: draws and what a tool draws cannot differ.
    STAGES_QUICK = ("construction", "drawing", "topology", "spectral")
    STAGES_STANDARD = STAGES_QUICK + ("relational", "hodge", "void", "epsilon")
    STAGES_FULL = STAGES_STANDARD + (
        "advanced", "rcfe", "sigma_sweep", "ricci_flow", "continuum_limit",
    )

    def __init__(self, rex, *, draw: bool = True, draw_limit: int = 400):
        self.rex = rex
        self.results: dict[str, Any] = {}
        self.callbacks: list[Callable] = []
        self.current_stage = ""
        self.completed_stages: list[str] = []
        self.draw = bool(draw)
        self.draw_limit = int(draw_limit)

    def on_stage(self, callback: Callable[[str, dict], None]):
        """Register a callback for progressive stage reporting."""
        self.callbacks.append(callback)

    def run(self, depth: str = "standard") -> dict[str, Any]:
        """Run the pipeline to the specified depth.

        Parameters
        ----------
        depth : 'quick', 'standard', or 'full'

        Returns
        -------
        dict with all stage results merged
        """
        if depth == "quick":
            stages = self.STAGES_QUICK
        elif depth in ("full", "deep"):
            stages = self.STAGES_FULL
        else:
            stages = self.STAGES_STANDARD

        # Size guard. The deep stages (betti/Hodge/curvature) build matrices whose
        # cost grows fast with edges and faces; large or dense graphs can exhaust
        # memory and take down the worker. Beyond a configurable ceiling, keep only
        # the cheap construction stage (nV/nE/nF/chain_valid) and report clearly,
        # instead of grinding for many seconds and then OOM-ing. Tune with
        # REXGRAPH_MAX_ANALYSIS_NODES / _EDGES (0 disables the guard).
        import os
        nV = int(getattr(self.rex, "nV", 0) or 0)
        nE = int(getattr(self.rex, "nE", 0) or 0)
        try:
            max_nodes = int(os.environ.get("REXGRAPH_MAX_ANALYSIS_NODES", "200000"))
            max_edges = int(os.environ.get("REXGRAPH_MAX_ANALYSIS_EDGES", "1000000"))
        except ValueError:
            max_nodes, max_edges = 200000, 1000000
        if depth != "quick" and (
                (max_nodes > 0 and nV > max_nodes) or (max_edges > 0 and nE > max_edges)):
            stages = ("construction",)
            self.results["size_limited"] = {
                "nV": nV, "nE": nE,
                "limit_nodes": max_nodes, "limit_edges": max_edges,
                "message": (
                    "Graph too large for deep topological analysis "
                    f"({nV} nodes / {nE} edges exceeds the "
                    f"{max_nodes}-node / {max_edges}-edge limit). Returned basic "
                    "structure only. Raise REXGRAPH_MAX_ANALYSIS_NODES / _EDGES to "
                    "override, or reduce/sparsify the graph."),
            }

        for stage_name in stages:
            self.current_stage = stage_name
            method = getattr(self, f"_stage_{stage_name}", None)
            if method is None:
                continue
            try:
                data = method()
                self.results[stage_name] = data
                self.completed_stages.append(stage_name)
                self._emit(stage_name, data)
            except Exception as e:
                error_data = {"error": str(e), "stage": stage_name}
                self.results[stage_name] = error_data
                self.completed_stages.append(stage_name)
                self._emit(stage_name, error_data)

        # kappa fallback: for text co-occurrence graphs the
        # structural coherence can be NaN, leaving kappa blank in the UI.
        # When that happens but a Hodge decomposition exists, use the
        # gradient fraction as a coherence proxy so the value is
        # informative rather than "-".
        rel = self.results.get("relational")
        hodge = self.results.get("hodge")
        if isinstance(rel, dict) and isinstance(hodge, dict):
            if rel.get("kappa_mean") is None and "pct_gradient" in hodge:
                try:
                    g = float(hodge["pct_gradient"])
                    # pct_grad may be a fraction (0-1) or a percentage
                    # (0-100) depending on the kernel; normalise to [0,1].
                    proxy = g / 100.0 if g > 1.0 else g
                    proxy = round(max(0.0, min(1.0, proxy)), 4)
                    rel["kappa_mean"] = proxy
                    rel["kappa_is_proxy"] = True
                except Exception:
                    pass

        return self.results

    def _emit(self, stage_name: str, data: dict):
        for cb in self.callbacks:
            with contextlib.suppress(Exception):
                cb(stage_name, data)

    # Stage implementations

    def _stage_construction(self) -> dict:
        rex = self.rex
        meta = getattr(rex, "_agent_meta", {})
        result = {
            "nV": rex.nV,
            "nE": rex.nE,
            "nF": rex.nF,
            "dimension": rex.dimension,
            "chain_valid": rex.chain_valid,
        }
        if meta:
            result["input_type"] = meta.get("input_type", "unknown")
            result["n_types"] = meta.get("n_types", 0)
            result["type_names"] = meta.get("type_names", [])
            result["vertex_labels"] = meta.get("vertex_labels", [])
        return result

    def _stage_drawing(self) -> dict:
        """A picture of the complex, and what it left out.

        No threshold decides whether to draw. `draw_limit` bounds how many cells go into
        the document and the result REPORTS what was drawn against what exists, so a
        truncated picture says it is truncated instead of a rule deciding silently that
        this complex is too big to look at. Set `draw=False` to skip it.

        Failure is reported, not raised: a pipeline that cannot draw has still analysed
        the complex, and losing the analysis because the picture failed would be the
        wrong trade.
        """
        if not self.draw:
            return {"drawn": False, "reason": "drawing is off for this run"}
        try:
            from agent.graph_view import render_payload
            from agent.render_svg import render_svg

            labels = (getattr(self.rex, "_agent_meta", {}) or {}).get("vertex_labels")
            payload = render_payload(self.rex, labels=labels, limit=self.draw_limit)
            drawn = {r["index"] for r in payload.get("relations", [])}
            # a face whose relations were not all drawn is not drawn either, so count the
            # ones that made it rather than the ones in the payload
            faces = sum(1 for f in payload.get("faces", [])
                        if set(f["relations"]) <= drawn)
            return {
                "drawn": True,
                "svg": render_svg(payload),
                "cells_drawn": len(drawn),
                "cells_total": int(self.rex.nE),
                "truncated": len(drawn) < int(self.rex.nE),
                "faces_drawn": faces,
                "faces_total": int(self.rex.nF_hodge),
                "view": "structural",
                "reading": ("positions are the adjacency layout, so cells sit near what "
                            "they are connected to rather than near what they resemble; "
                            "length is the quadrance, so it carries arity; colour is the "
                            "character through K7's spectrum. `view='plane'` is the "
                            "exact-rational layout, which is the reading rather than the "
                            "drawing"),
            }
        except Exception as exc:
            return {"drawn": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _stage_topology(self) -> dict:
        rex = self.rex
        return {
            "betti": list(rex.betti),
            "euler_characteristic": rex.euler_characteristic,
        }

    def _stage_spectral(self) -> dict:
        rex = self.rex
        result = {}

        # For large complexes a full dense eigendecomposition of L0 is
        # wasteful: we only need the low end of the spectrum (Fiedler
        # value, spectral gap, a handful of modes). Use a sparse
        # truncated solver instead. Small graphs keep the
        # exact dense path via spectral_bundle.
        use_sparse = getattr(rex, "nE", 0) > 100 or getattr(rex, "nV", 0) > 200
        sparse_ok = False
        if use_sparse:
            try:
                evals = _smallest_eigenvalues_L0(rex, k=min(20, rex.nV - 1))
                if evals is not None and len(evals):
                    result["eigenvalues_L0"] = evals.tolist()
                    result["eigenvalues_truncated"] = True
                    result["n_eigenvalues"] = int(len(evals))
                    # Fiedler value = smallest strictly positive eigenvalue, so which
                    # ones are zero has to be decided. dim ker(L0) is beta_0, an integer
                    # the rank tower gives exactly, so skip that many rather than cut at
                    # a magnitude: a graph with a genuinely tiny gap (a near-disconnected
                    # component, which is exactly what the Fiedler value is for) is
                    # indistinguishable from a numerical zero to a threshold.
                    try:
                        n_zero = int(rex.betti[0])
                    except Exception:                # noqa: BLE001
                        n_zero = int((evals <= 0.0).sum())
                    pos = np.sort(evals)[n_zero:]
                    result["fiedler_value"] = float(pos[0]) if len(pos) else 0.0
                    if len(pos) >= 2:
                        result["spectral_gap"] = float(pos[1] - pos[0])
                    sparse_ok = True
            except Exception:
                sparse_ok = False

        if not sparse_ok:
            try:
                evals_L0 = rex.eigenvalues_L0
                result["eigenvalues_L0"] = (
                    evals_L0.tolist() if evals_L0 is not None else None
                )
                # Fiedler value is the smallest positive eigenvalue of L0
                sb = rex.spectral_bundle
                result["fiedler_value"] = float(sb.get('fiedler_val_L0', 0.0))
            except Exception:
                result["eigenvalues_L0"] = None

        try:
            alpha_G, alpha_T = rex.coupling_constants
            result["alpha_G"] = float(alpha_G)
            result["alpha_T"] = float(alpha_T)
        except Exception:
            pass

        return result

    def _stage_relational(self) -> dict:
        rex = self.rex
        result = {}
        import os

        # THE CHARACTER: O(nnz), the reference's default. Everything here is a
        #    diagonal, row-norm, star aggregation, sparse matvec, or trace: no
        #    per-vertex inverse solve, no eigendecomposition (SCALE_PROPAGATOR_CALCULUS
        #    Parts A-D; scripts 13-20). The per-vertex Green's φ (the GLOBAL moment) is
        # an optional refinement at the end.

        # (1) Per-EDGE character χ = ĥ_k[e,e]/RL[e,e] - the base character (diagonals).
        try:
            chi = rex.structural_character
            result["nhats"] = int(rex.nhats)
            if chi is not None and chi.ndim == 2:
                result["chi_mean"] = chi.mean(axis=0).tolist()
                result["chi_per_edge"] = chi.tolist()
        except Exception:
            pass

        # (2) Local ENERGY character diag(RL4²) = ‖RL4[e,:]‖² (row-norms, the short-time
        #     heat moment; Part C.3 / script 14), and its per-vertex value via the
        #     BOUNDARY/star aggregation: the vertex propagator's local end, no solve.
        try:
            ec = np.asarray(rex.energy_character, dtype=float)
            if ec.size:
                result["energy_character_mean"] = round(float(ec.mean()), 6)
                result["energy_character_std"] = round(float(ec.std()), 6)
            ve = np.asarray(rex.vertex_energy_character, dtype=float)
            if ve.size:
                result["vertex_energy_mean"] = round(float(ve.mean()), 6)
                result["vertex_energy_per_vertex"] = ve.round(6).tolist()
        except Exception:
            pass

        # (3) Per-VERTEX character via the boundary: χ*(v) = star-average of χ over
        #     incident edges (B₁ aggregation, O(nnz)) - the default vertex character.
        try:
            chistar = np.asarray(rex.star_character, dtype=float)
            if chistar.ndim == 2 and chistar.shape[0] > 0:
                channels = ["T", "G", "F", "C"][:chistar.shape[1]]
                result["star_character_mean"] = chistar.mean(axis=0).tolist()
                dominant = [channels[int(np.argmax(chistar[v, :len(channels)]))]
                            for v in range(chistar.shape[0])]
                dom_counts = {}
                for c in dominant:
                    dom_counts[c] = dom_counts.get(c, 0) + 1
                result["dominant_channel_distribution"] = dom_counts
                result["dominant_channel_per_vertex"] = dominant
        except Exception:
            pass

        # (4) SCALE BRIDGE (local<->global; Part B / script 15): the closed-k-walk
        #     moments (L0^k)_vv per vertex - the star neighborhood's structure at each
        #     scale, plus the clustering signal that separates locally-clustered from
        #     unclustered members at equal degree. All sparse matvecs, O(nnz).
        try:
            sb = rex.scale_bridge
            result["scale_bridge"] = {
                "scale2_mean": round(float(sb["scale2_mean"]), 4),
                "scale3_mean": round(float(sb["scale3_mean"]), 4),
                "clustering_mean": round(float(sb["clustering_mean"]), 4),
            }
        except Exception:
            pass

        # (5) COHERENCE. Default (Part D.4: "default to H₂"): the global harmonic-log
        #     H₂ = -log(tr RL4²/tr RL4)² (Rényi-2, O(nnz) trace) as the graph-level
        #     coherence, and the per-vertex LOCAL coherence κ_loc (star-consistency of
        #     χ, O(nnz)). No solves, available at every scale.
        try:
            result["harmonic_entropy_H2"] = round(float(rex.harmonic_entropy), 6)
            # Varentropy self-diagnostic (script 19): the H₂-H₃ gap certifies when the
            # cheap H₂ coherence is trustworthy - ~0 on flat/unweighted spectra, grows
            # with weight-induced non-uniformity. One extra sparse matmul.
            ve = rex.character_varentropy
            result["coherence_varentropy_gap"] = ve["gap"]
            result["coherence_trustworthy"] = bool(ve["gap"] < 0.05)
        except Exception:
            pass
        try:
            kloc = np.asarray(rex.local_coherence, dtype=float)
            if kloc.size and not np.all(np.isnan(kloc)):
                kc = np.where(np.isnan(kloc), 0.0, kloc)
                result["kappa_mean"] = round(float(kc.mean()), 4)
                result["kappa_std"] = round(float(kc.std()), 4)
                result["kappa_min"] = round(float(kc.min()), 4)
                result["kappa_max"] = round(float(kc.max()), 4)
                result["kappa_per_vertex"] = [round(float(k), 4) for k in kloc]
                result["coherence_method"] = "harmonic_log_local"
        except Exception:
            pass

        # OPTIONAL GLOBAL REFINEMENT: the per-vertex Green's character φ and
        #    coherence κ_greens (the GLOBAL moment, t->∞: diag of B₁ RL4⁺ ĥ RL4⁺ B₁ᵀ).
        #    This is the one quantity that genuinely needs nV solves (its sandwiched
        #    two-inverse form resists selected inversion), so it is a bounded add-on,
        #    NOT the default character. Budget: REXGRAPH_VERTEX_CHARACTER_MAX_NODES
        # (default 1500; 0 = always). The character above is complete without it.
        try:
            budget = int(os.environ.get("REXGRAPH_VERTEX_CHARACTER_MAX_NODES", "1500"))
        except ValueError:
            budget = 1500
        if (budget <= 0) or (int(rex.nV) <= budget):
            try:
                phi = rex.vertex_character
                if phi is not None and phi.ndim == 2:
                    result["phi_mean"] = phi.mean(axis=0).tolist()
                    result["phi_per_vertex"] = phi.tolist()
                kg = np.asarray(rex.coherence, dtype=float)          # Green's κ
                if kg.size and not np.all(np.isnan(kg)):
                    kgc = np.where(np.isnan(kg), 0.0, kg)
                    result["kappa_greens_mean"] = round(float(kgc.mean()), 4)
                    result["kappa_greens_per_vertex"] = [round(float(k), 4) for k in kg]
            except Exception:
                pass
            try:
                times = rex.per_channel_mixing_times
                if times is not None:
                    result["mixing_times"] = times.tolist()
            except Exception:
                pass

        return result

    def _stage_hodge(self) -> dict:
        rex = self.rex
        result = {}
        try:
            flow = np.ones(rex.nE, dtype=np.float64)
            hodge_data = rex.hodge_full(flow)
            result["pct_gradient"] = float(hodge_data.get("pct_grad", 0))
            result["pct_curl"] = float(hodge_data.get("pct_curl", 0))
            result["pct_harmonic"] = float(hodge_data.get("pct_harm", 0))

            # Orthogonality verification
            orth = hodge_data.get("orthogonality")
            if orth:
                result["orthogonal"] = bool(orth.get("orthogonal", True))
                result["max_inner_product"] = float(
                    orth.get("max_inner", 0)
                )

            # Per-edge component norms (how much of the signal at
            # each edge is gradient vs curl vs harmonic)
            grad_norm = hodge_data.get("grad_norm")
            curl_norm = hodge_data.get("curl_norm")
            harm_norm = hodge_data.get("harm_norm")

            if grad_norm is not None:
                result["grad_norm_per_edge"] = grad_norm.tolist()
            if curl_norm is not None:
                result["curl_norm_per_edge"] = curl_norm.tolist()
            if harm_norm is not None:
                result["harm_norm_per_edge"] = harm_norm.tolist()

            # Face curl (per-face circulation)
            face_curl = hodge_data.get("face_curl")
            if face_curl is not None and len(face_curl) > 0:
                result["face_curl"] = face_curl.tolist()
                result["max_face_curl"] = round(
                    float(np.max(np.abs(face_curl))), 6
                )
                result["mean_face_curl"] = round(
                    float(np.mean(np.abs(face_curl))), 6
                )

            # Divergence at vertices
            div_data = hodge_data.get("divergence")
            if div_data is not None:
                result["divergence_per_vertex"] = div_data.tolist()
                result["max_divergence"] = round(
                    float(np.max(np.abs(div_data))), 6
                )

            # Harmonic mode analysis: combinatorial + low-rank, scale-free.
            # H = spanning-tree fundamental-cycle basis projected onto ker(B2^T)
            # (rexgraph.harmonic_sparse); P_harm applied low-rank as
            # H (H^T H)^-1 H^T flow, never the dense nE×nE projector, no eigensolve.
            # Channel diagonals hat_k[e,e] = structural_character · RL[e,e] (sparse,
            # hybrid) instead of a per-hat eigendecomposition + V diag(λ) V^T rebuild.
            # The dimension of the harmonic (oscillatory) space is β₁ = betti[1], an
            # EXACT integer: free, no basis to build. The per-mode harmonic SIGNAL
            # (harmonic_projection = a β₁×β₁ HᵀH solve) is optional detail, gated to
            # the same latency budget as the per-vertex Green's character; the Hodge
            # FRACTIONS + per-edge grad/curl/harm norms above (hodge_full, O(nnz)) are
            # the primary output and are always present.
            import os as _os
            dim_H = int(rex.betti[1])
            result["dim_H"] = dim_H
            result["n_oscillatory_modes"] = dim_H
            try:
                _hbudget = int(_os.environ.get(
                    "REXGRAPH_VERTEX_CHARACTER_MAX_NODES", "1500"))
            except ValueError:
                _hbudget = 1500
            _harm_detail = (_hbudget <= 0) or (int(rex.nV) <= _hbudget)
            try:
                if _harm_detail and dim_H > 0:
                    from rexgraph import harmonic_sparse as _hsp
                    H = _hsp.harmonic_basis(rex)
                    harm_signal = _hsp.harmonic_projection(H, flow)

                    total_sq = flow ** 2
                    harm_sq = harm_signal ** 2
                    safe_total = np.where(total_sq > 1e-30, total_sq, 1.0)
                    result["harm_frac_per_edge"] = (harm_sq / safe_total).tolist()

                    nh = int(rex.nhats)
                    if nh >= 3:
                        # hat_k[e,e] = structural_character[e,k] * RL[e,e]  (O(nnz))
                        chi = (np.asarray(rex.structural_character)
                               * np.asarray(rex._rl4_sparse.diagonal())[:, None])
                        T_ch, G_ch, F_ch = chi[:, 0], chi[:, 1], chi[:, 2]

                        frustration = np.abs(harm_signal) * T_ch
                        coparticipation = np.abs(harm_signal) * G_ch
                        result["frustration_per_edge"] = frustration.tolist()
                        result["coparticipation_per_edge"] = coparticipation.tolist()
                        fsum = float(np.sum(frustration))
                        csum = float(np.sum(coparticipation))
                        result["frustration_total"] = round(fsum, 6)
                        result["coparticipation_total"] = round(csum, 6)
                        result["health_ratio"] = (round(fsum / csum, 6)
                                                  if csum > 1e-10 else None)

                        denom = T_ch + F_ch
                        sigma = np.where(denom > 1e-10,
                                         (T_ch - F_ch) / np.where(denom > 1e-10, denom, 1.0),
                                         0.0)
                        result["sigma_asymmetry_per_edge"] = sigma.tolist()

                    # Harmonic mode summary (top edges per fundamental mode)
                    Hc = H.tocsc()
                    mode_summaries = []
                    for hi in range(min(dim_H, 10)):
                        h = np.asarray(Hc[:, hi].toarray()).ravel()
                        top_idx = np.argsort(np.abs(h))[-3:][::-1]
                        mode_summaries.append({
                            "mode": hi,
                            "top_edges": top_idx.tolist(),
                            "top_magnitudes": np.abs(h[top_idx]).tolist(),
                        })
                    result["harmonic_modes"] = mode_summaries

            except Exception:
                pass

        except Exception:
            pass
        return result

    def decompose_signal(
        self,
        signal: np.ndarray,
        signal_name: str = "signal",
    ) -> dict:
        """Decompose an arbitrary edge signal into Hodge components.

        This is the method that the case studies use to analyze
        survival correlations, drug propagation, query relevance,
        or any domain-specific edge signal.

        The decomposition reveals how much of the signal is:
        - **Gradient** (im B1^T): explainable by vertex-level data.
          Accessible to standard graph methods (Laplacian, PageRank).
        - **Curl** (im B2): face-level interactions, three-way
          relationships.  Requires face structure to detect.
        - **Harmonic** (ker L1): topological residual in the kernel
          of the Hodge Laplacian.  Requires the full complex.

        In the lung cancer study, only 9.6% of the survival signal
        was gradient.  90.4% was invisible to pairwise methods.

        Parameters
        ----------
        signal : f64[nE]
            An edge signal to decompose.
        signal_name : str
            Label for the signal in the output.

        Returns
        -------
        dict with Hodge decomposition, channel character, face/void
        dipole, per-edge components, and orthogonality verification.
        """
        rex = self.rex
        psi = np.ascontiguousarray(signal, dtype=np.float64)

        if len(psi) != rex.nE:
            raise ValueError(
                f"Signal length {len(psi)} does not match nE={rex.nE}"
            )

        result = {"signal_name": signal_name, "nE": rex.nE}

        # Hodge decomposition
        try:
            h = rex.hodge_full(psi)
            result["pct_gradient"] = round(
                float(h.get("pct_grad", 0)), 6
            )
            result["pct_curl"] = round(
                float(h.get("pct_curl", 0)), 6
            )
            result["pct_harmonic"] = round(
                float(h.get("pct_harm", 0)), 6
            )

            # What fraction is invisible to standard methods?
            result["pct_beyond_pairwise"] = round(
                float(h.get("pct_curl", 0))
                + float(h.get("pct_harm", 0)),
                6,
            )

            # Per-edge components
            for comp in ["grad", "curl", "harm"]:
                arr = h.get(comp)
                if arr is not None:
                    result[f"{comp}_component"] = arr.tolist()

            # Normalized components
            for comp in ["grad_norm", "curl_norm", "harm_norm"]:
                arr = h.get(comp)
                if arr is not None:
                    result[comp] = arr.tolist()

            # Face curl
            fc = h.get("face_curl")
            if fc is not None and len(fc) > 0:
                result["face_curl"] = fc.tolist()

            # Divergence
            div_d = h.get("divergence")
            if div_d is not None:
                result["divergence"] = div_d.tolist()

            # Orthogonality
            orth = h.get("orthogonality")
            if orth:
                result["orthogonal"] = bool(
                    orth.get("orthogonal", True)
                )
        except Exception as e:
            result["hodge_error"] = str(e)

        # Channel character of the signal
        try:
            psc = rex.primal_signal_character(psi)
            channels = ["T", "G", "F", "C"]
            result["channel_character"] = {
                channels[i]: round(float(psc[i]), 4)
                for i in range(min(len(psc), 4))
            }
            result["dominant_channel"] = channels[int(np.argmax(psc[:4]))]
        except Exception:
            pass

        # Face/void dipole
        try:
            fvd = rex.face_void_dipole(psi)
            result["face_affinity"] = round(
                float(fvd.get("face_affinity", 0)), 4
            )
            result["void_affinity"] = round(
                float(fvd.get("void_affinity", 0)), 4
            )
            result["dipole_ratio"] = round(
                float(fvd.get("dipole_ratio", 0)), 4
            )
        except Exception:
            pass

        return result

    def _stage_void(self) -> dict:
        rex = self.rex
        result = {}
        try:
            # Brute-force potential-triangle enumeration explodes on dense
            # typed multigraphs (e.g. L-R interaction complexes: few cell-
            # type vertices, hundreds of parallel pathway edges), producing
            # tens of thousands of degenerate triangles that mix distinct
            # edge types and cost minutes. When that's the case, use the
            # optimized spectral / congruence-quotient characterization
            # instead: the homologically correct and O(spectral) path.
            if self._void_bruteforce_intractable(rex):
                return self._stage_void_spectral(rex)

            vc = rex.void_complex
            n_voids = int(vc.get("n_voids", 0))
            n_potential = int(vc.get("n_potential", 0))
            result["method"] = "exact"

            result["n_voids"] = n_voids
            result["n_potential"] = n_potential
            result["void_strain"] = float(vc.get("void_strain", 0))

            # Void fraction: what proportion of potential triangles
            # are unrealized
            if n_potential > 0:
                result["void_fraction"] = round(n_voids / n_potential, 4)
            else:
                result["void_fraction"] = 0.0

            # Number of realized faces for comparison
            result["n_faces"] = n_potential - n_voids

            # Per-void harmonic content eta
            eta = vc.get("eta")
            if eta is not None and len(eta) > 0:
                eta_arr = np.asarray(eta, dtype=np.float64)
                result["mean_eta"] = round(float(np.mean(eta_arr)), 6)
                result["max_eta"] = round(float(np.max(eta_arr)), 6)
                result["min_eta"] = round(float(np.min(eta_arr)), 6)
                # Count of topologically nontrivial voids (eta > 0)
                result["n_nontrivial_voids"] = int(np.sum(eta_arr > 1e-10))
                result["eta"] = eta_arr.tolist()

            # Per-void fills_beta: would filling this void reduce
            # beta_1?
            fills = vc.get("fills_beta")
            if fills is not None:
                fills_arr = np.asarray(fills)
                result["fills_beta_count"] = int(np.sum(fills_arr))
                result["fills_beta"] = fills_arr.tolist()

            # Per-void structural character chi_void (n_voids x 4)
            chi_void = vc.get("chi_void")
            if chi_void is not None and len(chi_void) > 0:
                chi_arr = np.asarray(chi_void, dtype=np.float64)
                channels = ["T", "G", "F", "C"]

                # Mean structural character across voids
                if chi_arr.ndim == 2 and chi_arr.shape[0] > 0:
                    mean_chi = chi_arr.mean(axis=0)
                    result["void_chi_mean"] = {
                        channels[i]: round(float(mean_chi[i]), 4)
                        for i in range(min(len(channels), len(mean_chi)))
                    }

                    # Dominant channel across voids
                    dominant_counts = np.zeros(4, dtype=int)
                    for row in chi_arr:
                        dom = int(np.argmax(row[:4]))
                        dominant_counts[dom] += 1
                    result["void_dominant_channel"] = {
                        channels[i]: int(dominant_counts[i])
                        for i in range(4)
                    }

                    # Per-void character (stored for per-entity
                    # downstream analysis)
                    result["chi_void"] = chi_arr.tolist()

            # Kernel dimension of the void Laplacian Lvoid = Bvoid·Bvoidᵀ (nE×nE).
            # This is an EXACT integer nullity, not a count of near-zero eigenvalues:
            #   dim ker(Lvoid) = nE - rank(Bvoid)   (since rank(Bvoid Bvoidᵀ)=rank(Bvoid)).
            # Computed on the small sparse Bvoid (nE × n_voids) - the nE×nE Lvoid is
            # never materialized, and there is no eigendecomposition or magic threshold.
            Bvoid = vc.get("Bvoid")
            if Bvoid is not None:
                result["Bvoid_shape"] = list(Bvoid.shape)
                Bd = np.asarray(Bvoid.toarray() if hasattr(Bvoid, "toarray")
                                else Bvoid, dtype=np.float64)
                if Bd.size > 0 and Bd.shape[1] > 0:
                    # numpy's default is the canonical SVD rank tolerance (machine-eps
                    # scaled), not an arbitrary constant; Bvoid is an integer ±1 matrix.
                    result["void_kernel_dim"] = int(Bd.shape[0]
                                                     - np.linalg.matrix_rank(Bd))
                else:
                    result["void_kernel_dim"] = int(rex.nE)

            # Void indices (which potential triangles are voids)
            void_idx = vc.get("void_indices")
            if void_idx is not None:
                result["void_indices"] = np.asarray(void_idx).tolist()

        except Exception:
            pass
        return result

    # Void tractability + spectral/quotient fallback

    # Above this many estimated potential triangles, exhaustive
    # enumeration is both too slow and dominated by degenerate
    # parallel-edge combinations; switch to the spectral path.
    _VOID_TRIANGLE_CAP = 40000

    def _void_bruteforce_intractable(self, rex) -> bool:
        """Cheap upper-bound estimate of the potential-triangle count.

        find_potential_triangles is ~O(sum_v C(deg(v), 2)); for a dense
        multigraph the degrees carry parallel edges, so this proxy blows
        past the cap exactly when enumeration would.
        """
        try:
            deg = np.asarray(rex.degree, dtype=np.float64)
            if deg.size == 0:
                return False
            est = 0.5 * float(np.sum(deg * deg))
            return est > self._VOID_TRIANGLE_CAP
        except Exception:
            return False

    def _stage_void_spectral(self, rex) -> dict:
        """Void / higher-cell characterization via spectra + quotient.

        Returns the homologically meaningful invariants without
        enumerating triangles:
          - beta_1              independent 1-cycles (candidate voids)
          - shadow_dim          1-cycles filled by faces (= rank B2)
          - n_voids             open cycles after faces (= beta_1 of R)
          - congruence_classes  parallel typed edges collapsed to classes
          - hypermanifold betti per dimension level
        """
        result = {"method": "spectral_quotient"}
        try:
            # EXACT integer invariants (no dense nE×nE eigendecomposition of L1):
            #   cycle-space dim (1-skeleton β₁) = nE - rank(B1) = nE - nV + β₀
            #   shadow_dim (cycles filled by faces) = rank(B2) = nF_hodge - β₂
            #   open cycles (true homological holes) = β₁ = rex.betti[1]
            b0, b1_h, b2 = (int(x) for x in rex.betti)
            b1 = int(rex.nE - rex.nV + b0)          # candidate cycles (β₁ at grade 1)
            open_cycles = int(b1_h)                 # unfilled cycles (harmonic)
            result["beta_1"] = b1
            result["shadow_dim"] = int(rex.nF_hodge - b2)   # rank(B2)
            result["n_voids"] = open_cycles
            result["n_potential"] = b1
            result["n_faces"] = b1 - open_cycles            # = shadow_dim
            result["void_fraction"] = round(open_cycles / b1, 4) if b1 else 0.0
        except Exception:
            pass
        try:
            # Congruence classes modulo the empty subcomplex = edges grouped by
            # boundary signature (parallel edges collapse). O(nE) via hashing the
            # SPARSE B1 columns, not the dense O(nE²) pairwise `congruence_classes`.
            from rexgraph.core._sparse import to_scipy_csr
            B1c = to_scipy_csr(rex._B1_dual).tocsc()
            ip, idx, dat = B1c.indptr, B1c.indices, B1c.data
            sigs = {tuple(sorted((int(idx[k]), float(dat[k]))
                                 for k in range(ip[e], ip[e + 1])))
                    for e in range(rex.nE)}
            ncls = len(sigs)
            result["congruence_classes"] = int(ncls)
            if ncls > 0:
                result["parallel_multiplicity"] = round(rex.nE / ncls, 3)
        except Exception:
            pass
        try:
            hm = rex.hypermanifold
            result["hypermanifold_betti"] = [
                list(m.get("betti", [])) for m in hm.get("manifolds", [])
            ]
        except Exception:
            pass
        return result

    def _stage_epsilon(self) -> dict:
        rex = self.rex
        result = {}

        # ε₁: chain condition (should be 0 for a valid complex)
        result["eps1_chain"] = 0.0 if rex.chain_valid else 1.0

        # ε₃: equiweight ‖ΓD + DΓ‖. The derived axiom, read from the complex rather
        # than asserted here: RexGraph.equiweight_residual settles it from D's block
        # structure, so it costs nothing and no dense Dirac is formed. See
        # rexgraph.dirac_propagator.equiweight_residual for the version that takes a
        # foreign operator, where the residual is a distance from being a graded Dirac.
        result["eps3_equiweight"] = float(rex.equiweight_residual)

        return result

    def _stage_advanced(self) -> dict:
        rex = self.rex
        result = {}

        # Dirac spectrum diagnostics: EXACT integer invariants, no dense
        # (nV+nE+nF)² operator and no eigendecomposition: the mode count is the
        # operator dimension, and the harmonic (zero) mode count is dim ker(D) =
        # Σ Betti (total homology). O(1) from the Betti numbers.
        try:
            result["dirac_n_modes"] = rex.dirac_dimension
            result["dirac_n_harmonic"] = rex.dirac_harmonic_count
        except Exception:
            pass

        # Coupled field operator: coupling from ‖B₂‖_F (cheap) and PSD from the
        # SMALLEST eigenvalue of the SPARSE block operator (Lanczos, k=1), no dense
        # (nE+nF)² matrix and no full eigendecomposition.
        try:
            g, is_psd = rex.field_coupling_psd
            result["field_coupling"] = float(g)
            result["field_psd"] = bool(is_psd)
        except Exception:
            pass

        return result

    def _stage_rcfe(self) -> dict:
        """RCFE strain field equation analysis.

        Computes attributed curvature (how much dynamical weights
        violate the chain condition at each face), the optimal
        coupling constant alpha, the per-face deficit delta, the
        relational strain sigma = B2 * delta, and verifies the
        Bianchi identity B1 * sigma = 0.

        The strain lives in im(B2), the curl subspace.  It has zero
        gradient and zero harmonic component by construction.

        Requires faces (nF > 0) and the RCF Cython modules.
        """
        rex = self.rex
        result = {}

        nF = getattr(rex, "nF_hodge", 0)
        if nF == 0:
            result["has_faces"] = False
            result["reason"] = "No faces: RCFE requires nF > 0"
            return result

        result["has_faces"] = True

        # Attributed curvature with the complex's own edge weights (sparse)
        try:
            w_E = getattr(rex, "w_E", None)
            w_use = w_E if (w_E is not None and not np.allclose(w_E, 1.0)) else None
            kappa_f = _attributed_kappa_sparse(rex, w_e=w_use)
            result["kappa_f"] = kappa_f.tolist()
            result["total_curvature"] = round(
                float(np.linalg.norm(kappa_f)), 6
            )
            result["mean_curvature"] = round(float(np.mean(kappa_f)), 6)
            result["max_curvature"] = round(float(np.max(kappa_f)), 6)
            result["relational_integrity"] = round(
                1.0 / (1.0 + float(np.linalg.norm(kappa_f))), 6
            )
        except Exception as e:
            result["curvature_error"] = str(e)
            return result

        # The weighted geometric signature (script 20): curvature R = B₁(W-I)B₂ =
        # deviation from the unweighted ∂²=0 ideal, decomposed by group + weight
        # concentration. Per-face ‖R[:,f]‖ is kappa_f above; this adds the per-VERTEX
        # curvature (which junction bends most), the per-edge rank-1 contributions,
        # weight concentration N_eff, and curvature-per-weight, all sparse, O(nnz).
        try:
            sig = rex.weighted_curvature_signature()
            result["geometric_signature"] = {
                "weighted": bool(sig["weighted"]),
                "n_eff": round(float(sig["n_eff"]), 4),
                "curvature_per_weight": round(float(sig["curvature_per_weight"]), 6),
                "curvature_per_vertex": np.asarray(sig["per_vertex"]).round(6).tolist(),
                "max_vertex_curvature": round(float(np.max(sig["per_vertex"]))
                                              if sig["per_vertex"].size else 0.0, 6),
            }
        except Exception:
            pass

        # Strain analysis: two regimes
        #
        # 1. Curvature-only (born_face = 0): pure topological strain
        #    from the chain condition violation. Delta = kappa.
        #    This is the "static" strain: how much the current
        #    attribution departs from the chain condition.
        #
        # 2. Uniform enforcement (born_face = 1/nF): strain when
        #    dynamical content is uniformly distributed across faces.
        #    This measures the imbalance between curvature and
        #    equal enforcement.
        #
        # Both satisfy B1*sigma = 0 (Bianchi identity).
        se_curv = None
        try:
            nF = rex.nF_hodge

            # Curvature-only strain (delta = kappa, born = 0) - sparse B1/B2 matvecs
            born_zero = np.zeros(nF, dtype=np.float64)
            se_curv = _strain_equilibrium_sparse(rex, kappa_f, born_zero)
            result["curvature_strain"] = {
                "alpha": round(float(se_curv["alpha"]), 6),
                "strain_norm": round(
                    float(se_curv["strain_norm"]), 6
                ),
                "bianchi_ok": bool(se_curv["bianchi_ok"]),
                "bianchi_residual": float(
                    se_curv["bianchi_residual"]
                ),
            }

            sigma_curv = se_curv.get("sigma")
            if sigma_curv is not None and np.max(np.abs(sigma_curv)) > 1e-15:
                n_pos = int(np.sum(sigma_curv > 1e-10))
                n_neg = int(np.sum(sigma_curv < -1e-10))
                result["curvature_strain"]["n_under_realized"] = n_pos
                result["curvature_strain"]["n_over_realized"] = n_neg

            # Uniform enforcement strain
            born_uniform = np.ones(nF, dtype=np.float64) / nF
            se_uniform = _strain_equilibrium_sparse(rex, kappa_f, born_uniform)
            result["uniform_strain"] = {
                "alpha": round(float(se_uniform["alpha"]), 6),
                "strain_norm": round(
                    float(se_uniform["strain_norm"]), 6
                ),
                "bianchi_ok": bool(se_uniform["bianchi_ok"]),
                "bianchi_residual": float(
                    se_uniform["bianchi_residual"]
                ),
            }

            delta_u = se_uniform.get("delta")
            if delta_u is not None and len(delta_u) > 0:
                result["uniform_strain"]["max_deficit"] = round(
                    float(np.max(np.abs(delta_u))), 6
                )

            sigma_u = se_uniform.get("sigma")
            if sigma_u is not None and np.max(np.abs(sigma_u)) > 1e-15:
                n_pos = int(np.sum(sigma_u > 1e-10))
                n_neg = int(np.sum(sigma_u < -1e-10))
                result["uniform_strain"]["n_under_realized"] = n_pos
                result["uniform_strain"]["n_over_realized"] = n_neg

            # Use curvature-only as the primary strain report
            result["alpha"] = result["curvature_strain"]["alpha"]
            result["strain_norm"] = result["curvature_strain"][
                "strain_norm"
            ]
            result["bianchi_ok"] = result["curvature_strain"][
                "bianchi_ok"
            ]
            result["bianchi_residual"] = result["curvature_strain"][
                "bianchi_residual"
            ]

        except Exception as e:
            result["equilibrium_error"] = str(e)

        # Per-edge strain, read off the curvature-only equilibrium solved above
        try:
            sigma = se_curv.get("sigma") if se_curv is not None else None
            if sigma is not None:
                result["sigma_per_edge"] = [
                    round(float(s), 6) for s in sigma
                ]

                # Hodge decomposition of the strain itself
                # Strain = B2 * delta is in im(B2) by construction,
                # so it should be pure curl with zero gradient and
                # zero harmonic components
                try:
                    h = rex.hodge_full(sigma)
                    result["strain_pct_gradient"] = round(
                        float(h.get("pct_grad", 0)), 6
                    )
                    result["strain_pct_curl"] = round(
                        float(h.get("pct_curl", 0)), 6
                    )
                    result["strain_pct_harmonic"] = round(
                        float(h.get("pct_harm", 0)), 6
                    )
                except Exception:
                    pass

            delta = se_curv.get("delta") if se_curv is not None else None
            if delta is not None:
                result["delta_per_face"] = [
                    round(float(d), 6) for d in delta
                ]
        except Exception:
            pass

        # RCFE curvature C(sigma) per edge (from the topology)
        try:
            rc = rex.rcfe_curvature
            if rc is not None:
                result["rcfe_curvature_per_edge"] = [
                    round(float(c), 4) for c in rc
                ]
                result["rcfe_strain_total"] = round(
                    float(rex.rcfe_strain), 6
                )
        except Exception:
            pass

        return result

    def _stage_sigma_sweep(self) -> dict:
        """Sweep the relational strain across the enforcement parameter.

        Reconstructs the manual workflow's sigma sweep using the RCFE
        strain-equilibrium kernel.  We interpolate the per-face "born"
        target from the curvature-only regime (t=0) to uniform
        enforcement (t=1) and record how the equilibrium strain norm and
        optimal coupling alpha respond.  This is a genuine one-parameter
        sweep computed from the compiled kernel, not a placeholder.

        Requires faces (nF > 0) and the RCF module.
        """
        rex = self.rex
        result = {}
        nF = getattr(rex, "nF_hodge", 0)
        if nF == 0:
            return {"available": False, "reason": "sigma sweep requires nF > 0"}
        try:
            w_E = getattr(rex, "w_E", None)
            w_use = w_E if (w_E is not None and not np.allclose(w_E, 1.0)) else None
            kappa_f = _attributed_kappa_sparse(rex, w_e=w_use)

            ts = np.linspace(0.0, 1.0, 11)
            sweep = []
            uniform = np.ones(nF, dtype=np.float64) / nF
            for t in ts:
                born = t * uniform
                se = _strain_equilibrium_sparse(rex, kappa_f, born)
                sweep.append({
                    "t": round(float(t), 3),
                    "alpha": round(float(se["alpha"]), 6),
                    "strain_norm": round(float(se["strain_norm"]), 6),
                    "bianchi_ok": bool(se["bianchi_ok"]),
                })
            result["available"] = True
            result["parameter"] = "born_face_uniformity"
            result["sweep"] = sweep
            norms = [s["strain_norm"] for s in sweep]
            result["strain_norm_min"] = round(float(np.min(norms)), 6)
            result["strain_norm_max"] = round(float(np.max(norms)), 6)
            # t at which strain is minimised (most "relaxed" enforcement).
            result["t_min_strain"] = sweep[int(np.argmin(norms))]["t"]
        except Exception as e:
            result = {"available": False, "reason": str(e)}
        return result

    def _stage_ricci_flow(self) -> dict:
        """Discrete Ricci-flow analysis (optional, capability-gated).

        The manual workflow ran Ricci flow through Spore.  That external
        solver is not part of this package, so this stage checks for a
        Ricci-curvature capability on the complex and, when present,
        reports the attributed (relational) curvature as the t=0 state of
        the flow.  When no curvature kernel is available it returns a
        clean "unavailable" marker instead of fabricating a trajectory.
        """
        rex = self.rex
        if getattr(rex, "nF_hodge", 0) == 0:
            return {"available": False, "reason": "Ricci flow requires nF > 0"}
        try:
            ac = rex.attributed_curvature()
            kappa_f = np.asarray(ac["kappa_f"], dtype=float)
        except Exception as e:
            return {"available": False, "reason": f"no curvature kernel: {e}"}
        return {
            "available": True,
            "method": "attributed_curvature (t=0 state)",
            "note": (
                "Full multi-step Ricci flow was run externally (Spore) in "
                "the reference workflow; this reports the initial curvature "
                "field. Wire a flow solver here to extend to t>0."
            ),
            "curvature_norm": round(float(np.linalg.norm(kappa_f)), 6),
            "curvature_mean": round(float(np.mean(kappa_f)), 6),
            "curvature_max": round(float(np.max(np.abs(kappa_f))), 6),
        }

    def _stage_continuum_limit(self) -> dict:
        """Continuum-limit indicators (optional, capability-gated).

        Reports spectral-density indicators that track how the discrete
        complex approaches a continuum operator as it refines: the
        low-end eigenvalue spacing of L0 and the harmonic dimension.
        Returns an "unavailable" marker if a Laplacian can't be formed.
        """
        rex = self.rex
        ev = _smallest_eigenvalues_L0(rex, k=min(30, max(2, rex.nV - 1)))
        if ev is None or len(ev) < 3:
            return {
                "available": False,
                "reason": "insufficient spectrum for continuum indicators",
            }
        pos = ev[ev > 1e-9]
        result = {
            "available": True,
            "n_eigenvalues": int(len(ev)),
            "lambda_1": float(pos[0]) if len(pos) else 0.0,
        }
        if len(pos) >= 2:
            spacings = np.diff(pos)
            result["mean_spacing"] = round(float(np.mean(spacings)), 6)
            result["spacing_ratio"] = round(
                float(np.mean(spacings[1:] / (spacings[:-1] + 1e-12))), 6
            ) if len(spacings) > 1 else None
        try:
            betti = list(rex.betti)
            result["harmonic_dim_H1"] = int(betti[1]) if len(betti) > 1 else 0
        except Exception:
            pass
        return result

    @property
    def total_stages(self) -> int:
        return len(self.STAGES_STANDARD)

    @property
    def stage(self) -> int:
        return len(self.completed_stages)
