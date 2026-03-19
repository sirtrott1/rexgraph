"""
rexgraph.analysis - Structural analysis for the visualization dashboard.

Connects RexGraph to the dashboard JSON contract. Accepts a rex with
optional vertex labels and edge attributes, and returns a dict whose
keys match what the dashboard templates read.

Functions:
    analyze        - Full structural analysis for the graph dashboard.
    analyze_signal - Signal/perturbation/field data for the signal dashboard.
    analyze_quotient - Quotient presets for the quotient dashboard.
    analyze_all    - Combined analysis for all three dashboards in one call.

Cached bundles:

    spectral_bundle   - Laplacians, eigenvalues, Betti numbers, coupling
                        constants, relational Laplacian RL with hat
                        operators, structural character chi.
    _adjacency_bundle - Symmetric CSR for standard graph algorithms.
    _overlap_bundle   - L_O for overlap analysis.
    _rcf_bundle       - RL, hats, nhats, hat_names (from spectral_bundle).
    _vertex_bundle    - phi, chi_star, kappa from _character.

Methods called:

    rex.hodge_full()                   - Full Hodge decomposition.
    rex.face_data()                    - Face extraction, counts, metrics.
    rex.overlap_pairs                  - Top-k overlap pairs by Jaccard.
    rex.energy_kin_pot()               - E_kin / E_pot decomposition.
    rex.per_edge_energy()              - Per-edge kinetic/potential energy.
    rex.field_operator / field_eigen   - Coupled (E,F) field dynamics.
    rex.classify_modes()               - Edge/face/coupled mode labels.
    rex.analyze_perturbation()         - Signal propagation pipeline.
    rex.signal_dashboard_data()        - Precomputed signal trajectories.
    rex.quotient_dashboard_data()      - Precomputed quotient presets.
    rex.partition_communities()        - Hierarchical graph partitioning.
    rex.structural_character           - Per-edge chi on the simplex.
    rex.vertex_character / coherence   - Per-vertex phi and kappa.
    rex.phi_similarity / fiber_similarity - Fiber bundle similarity.
    rex.rcfe_curvature / rcfe_strain   - RCFE curvature and strain.
    rex.void_complex                   - Void spectral theory.
    rex.dirac_eigenvalues              - Dirac spectrum.
    rex.graded_state / born_graded     - Cross-dimensional Born probs.
    rex.hypermanifold / harmonic_shadow - Manifold sequence.
    rex.attributed_curvature           - Attributed boundary curvature.
    rex.strain_equilibrium             - Dynamic strain equilibrium.
    _standard.build_standard_metrics() - PageRank, betweenness, etc.

Usage:

    from rexgraph.analysis import analyze, analyze_signal, analyze_quotient
    from rexgraph.graph import RexGraph

    rex = RexGraph.from_graph(sources, targets)
    data = analyze(rex, vertex_labels=names, edge_attrs={"type": types})

    sig_data = analyze_signal(rex, vertex_labels=names, edge_attrs={"type": types})

    quot_data = analyze_quotient(rex, vertex_labels=names, edge_attrs={"type": types})
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from rexgraph.core import _standard

from .graph import RexGraph, _ensure_dense

__all__ = ["analyze", "analyze_signal", "analyze_quotient", "analyze_all"]


# Helpers

_f64 = np.float64
_i32 = np.int32


def _round(v, n=4):
    """Round a scalar to *n* decimal places. Returns 0.0 for NaN or Inf."""
    fv = float(v)
    if fv != fv or fv == float("inf") or fv == float("-inf"):
        return 0.0
    return round(fv, n)


def _partition_from_fiedler(vec: NDArray) -> list:
    """Return Fiedler bipartition labels: 'A' if >= 0, 'B' otherwise."""
    return ["A" if v >= 0 else "B" for v in vec]


def _vertex_roles(
    in_deg: NDArray,
    out_deg: NDArray,
    v_face_count: NDArray,
    nV: int,
) -> list:
    """Assign each vertex a structural role based on degree and face participation."""
    roles = []
    for i in range(nV):
        if in_deg[i] == 0:
            roles.append("source")
        elif out_deg[i] == 0:
            roles.append("sink")
        elif v_face_count[i] >= 5:
            roles.append("hub")
        elif v_face_count[i] >= 2:
            roles.append("mediator")
        else:
            roles.append("peripheral")
    return roles


# Flow construction

_NEGATIVE_STEMS = frozenset([
    "inhibit", "block", "suppress", "degrad", "ubiquitin", "cleav",
    "deactiv", "repress", "negat", "down", "decreas", "reduc",
    "remov", "destroy", "kill", "delet", "attenu", "antagoni",
    "downregulat", "down_regulat", "dephosphorylat",
])


def _build_flow(
    rex: RexGraph,
    edge_attrs: Optional[Dict[str, Any]],
    negative_types: Optional[List[str]],
    nE: int,
) -> NDArray:
    """Construct a signed flow signal from edge attributes.

    Magnitude is taken from w_E or edge_attrs['weight'].
    Sign is determined by matching edge_attrs['type'] against negative_types.
    """
    flow = np.ones(nE, dtype=_f64)

    # Magnitude from explicit weights
    if rex.w_E is not None:
        w = np.asarray(rex.w_E, dtype=_f64).ravel()
        if w.shape[0] == nE:
            flow = np.abs(w)

    # Override with edge_attrs weight if present
    if edge_attrs is not None:
        weight_vals = edge_attrs.get("weight")
        if weight_vals is not None and len(weight_vals) == nE:
            for j in range(nE):
                try:
                    flow[j] = abs(float(weight_vals[j]))
                except (ValueError, TypeError):
                    pass

    # Infer negative types from stem matching if not provided
    neg_set = set(negative_types) if negative_types else set()
    if not neg_set and edge_attrs is not None:
        type_vals = edge_attrs.get("type")
        if type_vals is not None:
            unique_types = set(type_vals)
            for val in unique_types:
                low = val.lower() if isinstance(val, str) else ""
                if any(stem in low for stem in _NEGATIVE_STEMS):
                    neg_set.add(val)

    # Apply sign
    if neg_set and edge_attrs is not None:
        type_vals = edge_attrs.get("type", [])
        for j in range(min(nE, len(type_vals))):
            if type_vals[j] in neg_set:
                flow[j] = -abs(flow[j])

    return flow


# Attribute classification for the dashboard metadata block

def _classify_attributes(
    edge_attrs: Optional[Dict[str, Any]],
    nE: int,
) -> list:
    """Build attribute metadata in the format expected by the dashboard."""
    if not edge_attrs:
        return []

    attributes = []
    for col_name, values in edge_attrs.items():
        vals = [str(v).strip() for v in values if str(v).strip()] if values else []
        if not vals:
            continue
        # Attempt numeric parse
        try:
            floats = [float(v) for v in vals]
            attributes.append({
                "name": col_name, "kind": "numeric",
                "min": _round(min(floats)), "max": _round(max(floats)),
                "mean": _round(sum(floats) / len(floats)),
                "count": len(vals),
            })
        except (ValueError, TypeError):
            unique = sorted(set(vals))
            counts = {k: v for k, v in Counter(vals).most_common()}
            attributes.append({
                "name": col_name, "kind": "categorical",
                "values": unique,
                "counts": counts,
                "count": len(vals),
            })
    return attributes


# Main analysis function

def analyze(
    rex: RexGraph,
    *,
    vertex_labels: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Dict[str, Any]] = None,
    negative_types: Optional[List[str]] = None,
    svg_w: int = 700,
    svg_h: int = 500,
    run_perturbation: bool = True,
    perturbation_steps: int = 50,
    perturbation_t_max: float = 10.0,
    diffusion_steps: int = 30,
    diffusion_t_max: float = 5.0,
) -> dict:
    """Compute the full analysis data contract for the dashboard.

    Runs all Cython computations through the rex's cached bundles and
    assembles the result into the JSON structure that
    rex_dashboard_template.jsx expects.

    Parameters
    ----------
    rex : RexGraph
        Relational complex to analyze.
    vertex_labels : sequence of str, optional
        Vertex names. Defaults to "v0", "v1", ...
    edge_attrs : dict, optional
        Named edge attributes ({"type": list, "weight": list, ...}).
    negative_types : list of str, optional
        Edge type values representing inhibition or negative flow.
    svg_w, svg_h : int
        Canvas dimensions for layout.
    run_perturbation : bool
        If True, run perturbation analysis from the highest-energy edge.
    perturbation_steps, perturbation_t_max : int, float
        Timepoints and max time for perturbation propagation.
    diffusion_steps, diffusion_t_max : int, float
        Timepoints and max time for field diffusion trajectories.

    Returns
    -------
    dict
        Complete data contract for the dashboard template.
    """
    nV, nE, nF = rex.nV, rex.nE, rex.nF

    # Labels
    if vertex_labels is not None:
        v_names = list(vertex_labels)
    else:
        v_names = [f"v{i}" for i in range(nV)]
    e_names = [f"e{j + 1}" for j in range(nE)]

    # Degree arrays
    src, tgt = rex._ensure_src_tgt()
    v_degree = rex.degree
    v_in = rex.in_degree
    v_out = rex.out_degree

    # Spectral bundle
    sb = rex.spectral_bundle

    eL0 = sb['evals_L0']
    evL0 = sb['evecs_L0']

    # L1 eigenvalues (present when L_O is provided, else None)
    eL1_arr = sb.get('evals_L1')
    eL1_full = eL1_arr if eL1_arr is not None else np.array([])

    # L1_down eigenvalues: not in the bundle directly; compute from matrix
    L1_down_mat = sb.get('L1_down')
    if L1_down_mat is not None:
        eL1_down = np.sort(np.linalg.eigvalsh(_ensure_dense(L1_down_mat)))
    else:
        eL1_down = np.array([])

    eL2_arr = sb.get('evals_L2')
    eL2 = eL2_arr if eL2_arr is not None else np.array([])

    eL_O_arr = sb.get('evals_L_O')
    eL_O = eL_O_arr if eL_O_arr is not None else np.array([])

    eL1a_arr = sb.get('evals_RL_1')
    eL1a = eL1a_arr if eL1a_arr is not None else np.array([])

    # RL eigenvalues (full relational Laplacian)
    evals_RL_arr = sb.get('RL')
    if evals_RL_arr is not None and evals_RL_arr.shape[0] > 0:
        from rexgraph.core._linalg import eigh as _eigh_rl
        evals_RL_full = _eigh_rl(evals_RL_arr)[0]
    else:
        evals_RL_full = np.array([])

    b0, b1, b2 = rex.betti
    rank_B1 = nV - b0
    rank_B2 = nF - b2
    alpha_G, alpha_T = rex.coupling_constants
    fiedler_LO_val, fiedler_LO_vec = rex.fiedler_overlap

    # Fiedler value of L1
    fiedler_L1 = float(sb.get('fiedler_val_L1', 0.0))

    chain_ok = rex.chain_valid

    # L1 diagonal values (compute from matrices)
    L1_down_mat_diag = sb.get('L1_down')
    L1_up_mat_diag = sb.get('L1_up')
    L1_down_diag = np.diag(L1_down_mat_diag).astype(_f64) if L1_down_mat_diag is not None else np.zeros(nE, dtype=_f64)
    L1_up_diag = np.diag(L1_up_mat_diag).astype(_f64) if L1_up_mat_diag is not None else np.zeros(nE, dtype=_f64)

    # Layout, scaled to SVG canvas
    layout = rex.layout  # (nV, 2)
    if nV > 0 and layout.shape[0] == nV:
        margin_frac = 0.10
        for dim_idx, dim_size in enumerate([svg_w, svg_h]):
            col = layout[:, dim_idx]
            mn, mx = col.min(), col.max()
            rng = mx - mn
            margin = dim_size * margin_frac
            if rng > 1e-10:
                layout[:, dim_idx] = margin + (col - mn) / rng * (dim_size - 2 * margin)
            else:
                layout[:, dim_idx] = dim_size / 2.0
        px = layout[:, 0]
        py = layout[:, 1]
    else:
        px = np.zeros(nV, dtype=_f64)
        py = np.zeros(nV, dtype=_f64)

    # Flow signal and Hodge decomposition
    flow = _build_flow(rex, edge_attrs, negative_types, nE)
    hodge_result = rex.hodge_full(flow)

    grad_comp = hodge_result['grad']
    curl_comp = hodge_result['curl']
    harm_comp = hodge_result['harm']
    grad_norm = hodge_result['grad_norm']
    curl_norm = hodge_result['curl_norm']
    harm_norm = hodge_result['harm_norm']
    flow_norm = hodge_result['flow_norm']
    rho = hodge_result['rho']
    divergence = hodge_result['divergence']
    div_norm = hodge_result['div_norm']
    pct_grad = hodge_result['pct_grad']
    pct_curl = hodge_result['pct_curl']
    pct_harm = hodge_result['pct_harm']

    # When harmonic energy is negligible, self-normalization amplifies
    # floating-point noise to +/-1. Zero it out.
    if pct_harm < 1e-4:
        harm_norm = np.zeros_like(harm_norm)
        rho = np.zeros_like(rho)

    # Energy decomposition: E_kin (topological) vs E_pot (geometric)
    E_kin, E_pot, E_ratio = rex.energy_kin_pot(flow)
    E_kin_per, E_pot_per = rex.per_edge_energy(flow)

    if E_ratio > 1.2:
        energy_regime = "kinetic"
    elif E_ratio < 0.8:
        energy_regime = "potential"
    else:
        energy_regime = "crossover"

    # RL_1 eigenvalues
    evals_RL1_arr = sb.get('evals_RL_1')
    evals_RL1 = evals_RL1_arr if evals_RL1_arr is not None else np.array([])

    if evals_RL1_arr is not None and len(evals_RL1_arr) > 1:
        sorted_eRL = np.sort(evals_RL1_arr)
        first_nz = sorted_eRL[sorted_eRL > 1e-10]
        fiedler_RL1 = float(first_nz[0]) if len(first_nz) > 0 else 0.0
    else:
        fiedler_RL1 = 0.0

    # Field operator (coupled edge-face dynamics)
    nF_hodge = rex.nF_hodge
    has_field = nF_hodge > 0

    field_data = {}
    field_evals = np.array([])
    mode_data = {}
    if has_field:
        try:
            M, g_field, is_psd = rex.field_operator
            f_evals, f_evecs, f_freqs = rex.field_eigen
            field_evals = f_evals
            mode_data = rex.classify_modes()
            n_edge_modes = int(np.sum(mode_data.get('mode_type', []) == 0))
            n_face_modes = int(np.sum(mode_data.get('mode_type', []) == 1))
            n_coupled = int(np.sum(mode_data.get('mode_type', []) == 2))
            field_data = {
                "coupling_g": _round(g_field),
                "is_psd": bool(is_psd),
                "dim_E": nE,
                "dim_F": nF_hodge,
                "dim_total": nE + nF_hodge,
                "n_edge_modes": n_edge_modes,
                "n_face_modes": n_face_modes,
                "n_coupled_modes": n_coupled,
                "top_freqs": [_round(f, 6) for f in f_freqs[:8]],
            }
        except Exception:
            has_field = False

    # Perturbation analysis
    perturbation_data = {}
    if run_perturbation and nE > 0:
        try:
            # Perturb from highest-energy edge
            total_per_edge = E_kin_per + E_pot_per
            probe_edge = int(np.argmax(total_per_edge))
            f_E, f_F = rex.edge_perturbation(probe_edge)
            times = np.linspace(0, perturbation_t_max, perturbation_steps, dtype=_f64)

            pert_result = rex.analyze_perturbation(f_E, f_F, times=times)

            # Extract cascade summary
            cascade_act = pert_result.get('cascade_activation')
            if cascade_act is not None:
                act_time, act_order, act_rank = cascade_act
                n_activated = int(np.sum(act_time >= 0))
                cascade_depth = int(np.max(act_rank)) + 1 if n_activated > 0 else 0
            else:
                n_activated = 0
                cascade_depth = 0

            # Energy trajectory summary
            E_kin_traj = pert_result.get('E_kin', np.array([]))
            E_pot_traj = pert_result.get('E_pot', np.array([]))
            ratio_traj = pert_result.get('ratio', np.array([]))

            # Hodge decomposition of final state
            hodge_final = pert_result.get('hodge_final', {})

            perturbation_data = {
                "probe_edge": e_names[probe_edge],
                "probe_source": v_names[src[probe_edge]],
                "probe_target": v_names[tgt[probe_edge]],
                "n_activated": n_activated,
                "cascade_depth": cascade_depth,
                "activation_pct": _round(n_activated / max(nE, 1) * 100, 1),
                "E_kin_trajectory": [_round(v, 6) for v in E_kin_traj],
                "E_pot_trajectory": [_round(v, 6) for v in E_pot_traj],
                "ratio_trajectory": [_round(v, 4) for v in ratio_traj],
                "times": [_round(t, 4) for t in times],
                "hodge_final_grad_pct": _round(hodge_final.get('pct_grad', 0) * 100, 1),
                "hodge_final_curl_pct": _round(hodge_final.get('pct_curl', 0) * 100, 1),
                "hodge_final_harm_pct": _round(hodge_final.get('pct_harm', 0) * 100, 1),
            }

            # Per-edge cascade activation for top activated edges
            if cascade_act is not None and n_activated > 0:
                sorted_by_act = np.argsort(act_time)
                first_activated = [
                    int(j) for j in sorted_by_act
                    if act_time[j] >= 0
                ][:10]
                perturbation_data["first_activated"] = [
                    {
                        "id": e_names[j],
                        "source": v_names[src[j]],
                        "target": v_names[tgt[j]],
                        "time_idx": int(act_time[j]),
                        "rank": int(act_rank[j]),
                    }
                    for j in first_activated
                ]
        except Exception:
            pass

    # Field diffusion trajectory (compact summary)
    diffusion_data = {}
    if has_field and nE > 0:
        try:
            F0 = np.zeros(nE + nF_hodge, dtype=_f64)
            F0[:nE] = np.abs(flow) / max(float(np.linalg.norm(flow)), 1e-15)
            diff_times = np.linspace(0, diffusion_t_max, diffusion_steps, dtype=_f64)
            diff_traj = rex.field_diffuse(F0, diff_times)

            # Track norm decay per dimension
            norm_E = np.array([float(np.linalg.norm(diff_traj[t, :nE]))
                               for t in range(len(diff_times))])
            norm_F = np.array([float(np.linalg.norm(diff_traj[t, nE:]))
                               for t in range(len(diff_times))])

            diffusion_data = {
                "times": [_round(t, 4) for t in diff_times],
                "norm_E": [_round(v, 6) for v in norm_E],
                "norm_F": [_round(v, 6) for v in norm_F],
                "equilibrium_idx": int(np.argmin(np.abs(np.diff(norm_E)))) + 1
                    if len(norm_E) > 1 else 0,
            }

            # Vertex observables at final time
            F_final = diff_traj[-1]
            v_state = rex.derive_vertex_state(F_final)
            diffusion_data["vertex_final"] = [_round(v, 6) for v in v_state]
        except Exception:
            pass

    # Fiedler partitions
    # Vertex partition from L0 Fiedler vector
    fiedler_v = rex.fiedler_vector_L0
    v_part_L0 = _partition_from_fiedler(fiedler_v)

    # Edge partition from L_O Fiedler vector
    e_part_LO = _partition_from_fiedler(fiedler_LO_vec)

    # Vertex partition induced by L_O edge partition
    v_part_LO = []
    for i in range(nV):
        inc = [j for j in range(nE) if src[j] == i or tgt[j] == i]
        if not inc:
            v_part_LO.append("A")
            continue
        a_count = sum(1 for j in inc if e_part_LO[j] == "A")
        v_part_LO.append("A" if a_count >= len(inc) - a_count else "B")

    # Edge partition from RL_1 Fiedler vector
    evecs_RL1_part = sb.get('evecs_RL_1')
    evals_RL1_part = sb.get('evals_RL_1')
    if evecs_RL1_part is not None and evals_RL1_part is not None and evals_RL1_part.shape[0] > 1:
        sorted_idx = np.argsort(evals_RL1_part)
        fiedler_RL1_vec = evecs_RL1_part[:, sorted_idx[1]] if sorted_idx.shape[0] > 1 else np.zeros(nE)
        e_part_L1a = _partition_from_fiedler(fiedler_RL1_vec)
    else:
        fiedler_RL1_vec = np.zeros(nE, dtype=_f64)
        e_part_L1a = ["A"] * nE

    # Standard graph metrics
    adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle

    # Use edge weights when available
    if rex.w_E is not None:
        e_wt_raw = np.abs(np.asarray(rex.w_E, dtype=_f64).ravel())
        if e_wt_raw.shape[0] != nE:
            e_wt_raw = np.ones(nE, dtype=_f64)
    else:
        e_wt_raw = np.ones(nE, dtype=_f64)
    adj_wt = _standard.build_adj_weights(adj_edge, e_wt_raw)

    std_metrics = _standard.build_standard_metrics(
        adj_ptr, adj_idx, adj_edge, adj_wt,
        nV, nE,
    )

    v_pr = std_metrics['pagerank']
    v_betw = std_metrics['betweenness_v']
    e_betw = std_metrics['betweenness_e']
    v_btw_norm = std_metrics['btw_norm_v']
    e_btw_norm = std_metrics['btw_norm_e']
    v_clust = std_metrics['clustering']
    louvain = std_metrics['community_labels']
    n_communities = std_metrics['n_communities']
    Q_mod = std_metrics['modularity']

    # Face data
    face_result = rex.face_data(v_names, e_names, rho)

    faces = face_result['faces']
    v_face_count = face_result['vertex_face_count']
    fm = face_result['metrics']

    v_avg_contrib = fm.get('v_avg_contrib', np.zeros(nV, dtype=_f64))
    v_total_contrib = fm.get('v_total_contrib', np.zeros(nV, dtype=_f64))
    v_avg_face_size = fm.get('v_avg_face_size', np.zeros(nV, dtype=_f64))
    e_avg_contrib = fm.get('e_avg_contrib', np.zeros(nE, dtype=_f64))
    e_total_contrib = fm.get('e_total_contrib', np.zeros(nE, dtype=_f64))
    e_avg_face_size = fm.get('e_avg_face_size', np.zeros(nE, dtype=_f64))
    e_bnd_asym = fm.get('e_bnd_asym', np.zeros(nE, dtype=_f64))
    f_concentration = fm.get('f_concentration', np.empty(0))
    v_tc_sum = fm.get('v_tc_sum', 0.0)
    e_tc_sum = fm.get('e_tc_sum', 0.0)
    asym_rho_corr = fm.get('asym_rho_corr', 0.0)

    # Attach curl and concentration to face dicts
    face_curl = hodge_result.get('face_curl', np.zeros(nF, dtype=_f64))
    for fi in range(nF):
        faces[fi]['curl'] = _round(float(face_curl[fi]) if fi < len(face_curl) else 0.0, 2)
        faces[fi]['conc'] = _round(float(f_concentration[fi]) if fi < len(f_concentration) else 0.0, 4)

    # Vertex roles
    v_role = _vertex_roles(v_in, v_out, v_face_count, nV)

    # Edge types from attributes
    if edge_attrs and "type" in edge_attrs:
        e_types_str = [str(v) for v in edge_attrs["type"]]
    else:
        e_types_str = ["edge"] * nE

    # Edge weights from attributes
    e_weight = np.ones(nE, dtype=_f64)
    if edge_attrs and "weight" in edge_attrs:
        for j in range(min(nE, len(edge_attrs["weight"]))):
            try:
                e_weight[j] = float(edge_attrs["weight"][j])
            except (ValueError, TypeError):
                pass

    # Overlap pairs
    overlap_raw = rex.overlap_pairs
    overlap_pairs = []
    for pair in overlap_raw:
        ei, ej = pair['edge_i'], pair['edge_j']
        shared_verts = set()
        if src[ei] == src[ej] or src[ei] == tgt[ej]:
            shared_verts.add(v_names[src[ei]])
        if tgt[ei] == src[ej] or tgt[ei] == tgt[ej]:
            shared_verts.add(v_names[tgt[ei]])
        overlap_pairs.append({
            "a": e_names[ei], "b": e_names[ej],
            "shared": sorted(shared_verts),
            "jaccard": _round(pair['similarity']),
        })

    # Attribute metadata
    attributes = _classify_attributes(edge_attrs, nE)
    type_attr = next((a for a in attributes if a["kind"] == "categorical"), None)
    weight_attr = next((a for a in attributes if a["kind"] == "numeric"), None)

    # Negative types for export
    neg_types_export = sorted(
        set(negative_types) if negative_types
        else {t for t in set(e_types_str) if any(
            stem in t.lower() for stem in _NEGATIVE_STEMS
        )}
    )

    # Cross-metric correlations
    v_deg_f64 = v_degree.astype(_f64)
    v_fc_f64 = v_face_count.astype(_f64)

    pr_deg_corr = _standard.safe_correlation(v_pr, v_deg_f64)
    btw_deg_corr = _standard.safe_correlation(v_betw, v_deg_f64)
    clust_fc_corr = _standard.safe_correlation(v_clust, v_fc_f64)
    ebtw_rho_corr = _standard.safe_correlation(e_betw, rho)
    ebtw_l1up_corr = _standard.safe_correlation(e_betw, L1_up_diag)

    # Louvain vs Fiedler splits
    fiedler_splits = 0
    for c in range(n_communities):
        members = [i for i in range(nV) if louvain[i] == c]
        if len(members) < 2:
            continue
        parts = set(v_part_L0[i] for i in members)
        if len(parts) > 1:
            fiedler_splits += 1

    # Partition type composition
    role_counts = Counter(v_role)
    L0_A = sum(1 for p in v_part_L0 if p == "A")
    LO_A = sum(1 for p in e_part_LO if p == "A")
    L1a_A = sum(1 for p in e_part_L1a if p == "A")
    partition_agree_v = sum(1 for i in range(nV) if v_part_L0[i] == v_part_LO[i])

    LO_A_types = Counter(e_types_str[j] for j in range(nE) if e_part_LO[j] == "A")
    LO_B_types = Counter(e_types_str[j] for j in range(nE) if e_part_LO[j] == "B")

    # Ranked edge lists
    rho_ranked = sorted(range(nE), key=lambda j: -rho[j])
    up_ranked = sorted(range(nE), key=lambda j: -float(L1_up_diag[j]))

    # Structural analysis block

    analysis = {
        "topology": {
            "components": int(b0),
            "cycles_total": int(nE - rank_B1) if rank_B1 else int(b1 + rank_B2),
            "cycles_filled": int(nF),
            "cycles_unfilled": int(b1),
            "voids": int(b2),
            "euler": int(nV - nE + nF),
            "fill_rate": _round(nF / max(nE - rank_B1, 1) * 100, 1),
        },
        "hodge": {
            "gradient_pct": _round(pct_grad * 100, 1),
            "curl_pct": _round(pct_curl * 100, 1),
            "harmonic_pct": _round(pct_harm * 100, 1),
            "dominant": (
                "gradient" if pct_grad > pct_curl and pct_grad > pct_harm
                else ("curl" if pct_curl > pct_harm else "harmonic")
            ),
        },
        "coupling": {
            "alpha_G": _round(alpha_G),
            "alpha_T": _round(alpha_T),
            "fiedler_RL1": _round(fiedler_RL1),
            "interpretation": (
                "geometry stronger" if alpha_G > 1
                else ("balanced" if alpha_G > 0.5 else "topology stronger")
            ),
        },
        "energy": {
            "E_kin": _round(E_kin, 6),
            "E_pot": _round(E_pot, 6),
            "ratio": _round(E_ratio, 4),
            "regime": energy_regime,
            "top_kinetic": [
                {"id": e_names[j], "source": v_names[src[j]],
                 "target": v_names[tgt[j]],
                 "Ekin": _round(E_kin_per[j], 4)}
                for j in sorted(range(nE), key=lambda x: -E_kin_per[x])[:5]
            ],
            "top_potential": [
                {"id": e_names[j], "source": v_names[src[j]],
                 "target": v_names[tgt[j]],
                 "Epot": _round(E_pot_per[j], 4)}
                for j in sorted(range(nE), key=lambda x: -E_pot_per[x])[:5]
            ],
        },
        "field": field_data if has_field else None,
        "perturbation": perturbation_data if perturbation_data else None,
        "diffusion": diffusion_data if diffusion_data else None,
        "structure": {
            "roles": {k: v for k, v in role_counts.most_common()},
            "hubs": [
                {"id": v_names[i], "degree": int(v_degree[i]),
                 "faces": int(v_face_count[i])}
                for i in sorted(range(nV), key=lambda x: -v_degree[x])[:5]
            ],
            "sources": [v_names[i] for i in range(nV) if v_in[i] == 0],
            "sinks": [v_names[i] for i in range(nV) if v_out[i] == 0],
            "high_rho": [
                {"id": e_names[j], "source": v_names[src[j]],
                 "target": v_names[tgt[j]], "rho": _round(rho[j])}
                for j in rho_ranked[:5]
            ],
            "most_cyclic": [
                {"id": e_names[j], "source": v_names[src[j]],
                 "target": v_names[tgt[j]],
                 "up": _round(L1_up_diag[j], 1)}
                for j in up_ranked[:5] if L1_up_diag[j] > 0
            ],
        },
        "partitions": {
            "L0_split": [int(L0_A), int(nV - L0_A)],
            "LO_split": [int(LO_A), int(nE - LO_A)],
            "L1a_split": [int(L1a_A), int(nE - L1a_A)],
            "vertex_agreement": int(partition_agree_v),
            "LO_A_types": {k: v for k, v in LO_A_types.most_common()},
            "LO_B_types": {k: v for k, v in LO_B_types.most_common()},
        },
        "negative_types": neg_types_export,
        "standard_metrics": {
            "n_communities": int(n_communities),
            "modularity": _round(Q_mod),
            "pr_sum": _round(float(np.sum(v_pr)), 6),
            "pr_deg_corr": _round(pr_deg_corr),
            "btw_deg_corr": _round(btw_deg_corr),
            "clust_fc_corr": _round(clust_fc_corr),
            "ebtw_rho_corr": _round(ebtw_rho_corr),
            "ebtw_l1up_corr": _round(ebtw_l1up_corr),
            "fiedler_splits": int(fiedler_splits),
            "communities": {
                str(c): sorted([v_names[i] for i in range(nV) if louvain[i] == c])
                for c in range(n_communities)
            },
            "top_pagerank": [
                {"id": v_names[i], "pr": _round(v_pr[i], 6),
                 "degree": int(v_degree[i]), "role": v_role[i]}
                for i in sorted(range(nV), key=lambda x: -v_pr[x])[:8]
            ],
            "top_betweenness": [
                {"id": v_names[i], "btw": _round(v_betw[i], 2),
                 "btwNorm": _round(v_btw_norm[i]),
                 "degree": int(v_degree[i])}
                for i in sorted(range(nV), key=lambda x: -v_betw[x])[:8]
            ],
            "top_clustering": [
                {"id": v_names[i], "cc": _round(v_clust[i]),
                 "degree": int(v_degree[i])}
                for i in sorted(range(nV), key=lambda x: -v_clust[x])[:8]
            ],
            "top_edge_betw": [
                {"id": e_names[j], "source": v_names[src[j]],
                 "target": v_names[tgt[j]],
                 "ebtw": _round(e_betw[j], 2),
                 "ebtwNorm": _round(e_btw_norm[j]),
                 "rho": _round(rho[j])}
                for j in sorted(range(nE), key=lambda x: -e_betw[x])[:8]
            ],
        },
        "face_structure": {
            "v_tc_sum": _round(v_tc_sum),
            "e_tc_sum": _round(e_tc_sum),
            "asym_rho_corr": _round(asym_rho_corr),
            "mean_asym": _round(float(np.mean(e_bnd_asym)) if nE > 0 else 0),
            "mean_face_size": _round(
                float(np.mean([f["size"] for f in faces])) if nF > 0 else 0, 2
            ),
            "top_v_by_total_contrib": [
                {"id": v_names[i],
                 "faceTotC": _round(v_total_contrib[i], 3),
                 "faceCount": int(v_face_count[i]),
                 "avgContrib": _round(v_avg_contrib[i], 3),
                 "avgFaceSize": _round(v_avg_face_size[i], 1)}
                for i in sorted(range(nV), key=lambda x: -v_total_contrib[x])[:5]
                if v_face_count[i] > 0
            ],
            "top_e_by_total_contrib": [
                {"id": e_names[j], "source": v_names[src[j]],
                 "target": v_names[tgt[j]],
                 "faceTotC": _round(e_total_contrib[j], 3),
                 "faceCount": int(L1_up_diag[j]),
                 "avgContrib": _round(e_avg_contrib[j], 3),
                 "avgFaceSize": _round(e_avg_face_size[j], 1)}
                for j in sorted(range(nE), key=lambda x: -e_total_contrib[x])[:5]
                if L1_up_diag[j] > 0
            ],
            "most_balanced": [
                {"id": e_names[j], "source": v_names[src[j]],
                 "target": v_names[tgt[j]],
                 "asym": _round(e_bnd_asym[j], 3),
                 "rho": _round(rho[j]),
                 "srcFaces": int(v_face_count[src[j]]),
                 "tgtFaces": int(v_face_count[tgt[j]])}
                for j in sorted(range(nE), key=lambda x: e_bnd_asym[x])[:5]
                if L1_up_diag[j] > 0
            ],
            "most_asymmetric": [
                {"id": e_names[j], "source": v_names[src[j]],
                 "target": v_names[tgt[j]],
                 "asym": _round(e_bnd_asym[j], 3),
                 "rho": _round(rho[j]),
                 "srcFaces": int(v_face_count[src[j]]),
                 "tgtFaces": int(v_face_count[tgt[j]])}
                for j in sorted(range(nE), key=lambda x: -e_bnd_asym[x])[:5]
            ],
            "most_concentrated_faces": [
                {"id": faces[fi]["id"], "size": faces[fi]["size"],
                 "conc": _round(f_concentration[fi], 3),
                 "vertices": faces[fi]["vertices"]}
                for fi in sorted(range(nF), key=lambda x: -f_concentration[x])[:5]
            ] if nF > 0 else [],
            "most_uniform_faces": [
                {"id": faces[fi]["id"], "size": faces[fi]["size"],
                 "conc": _round(f_concentration[fi], 3),
                 "vertices": faces[fi]["vertices"]}
                for fi in sorted(range(nF), key=lambda x: f_concentration[x])[:5]
            ] if nF > 0 else [],
        },
    }

    # Export data contract

    n_spec = 15

    export = {
        "meta": {
            "nV": nV, "nE": nE, "nF": nF,
            "svgW": svg_w, "svgH": svg_h,
            "attributes": attributes,
            "typeCol": type_attr["name"] if type_attr else None,
            "weightCol": weight_attr["name"] if weight_attr else None,
        },
        "vertices": [
            {
                "id": v_names[i],
                "x": _round(px[i], 1), "y": _round(py[i], 1),
                "degree": int(v_degree[i]),
                "inDeg": int(v_in[i]), "outDeg": int(v_out[i]),
                "faceCount": int(v_face_count[i]),
                "role": v_role[i],
                "partL0": v_part_L0[i], "partLO": v_part_LO[i],
                "fiedler": _round(fiedler_v[i]),
                "divergence": _round(divergence[i]),
                "divNorm": _round(div_norm[i]),
                "faceAvgC": _round(v_avg_contrib[i]),
                "faceTotC": _round(v_total_contrib[i]),
                "faceAvgSz": _round(v_avg_face_size[i], 1),
                "pagerank": _round(v_pr[i], 6),
                "betweenness": _round(v_betw[i], 2),
                "btwNorm": _round(v_btw_norm[i]),
                "clustering": _round(v_clust[i]),
                "community": int(louvain[i]),
            }
            for i in range(nV)
        ],
        "edges": [
            {
                "id": e_names[j],
                "source": v_names[src[j]], "target": v_names[tgt[j]],
                "type": e_types_str[j] if j < len(e_types_str) else "edge",
                "w": _round(e_weight[j]),
                "partLO": e_part_LO[j], "partL1a": e_part_L1a[j],
                "fiedlerLO": _round(fiedler_LO_vec[j]),
                "flow": _round(flow[j]),
                "gradRaw": _round(grad_comp[j]),
                "curlRaw": _round(curl_comp[j]),
                "harmRaw": _round(harm_comp[j]),
                "gradN": _round(grad_norm[j]),
                "curlN": _round(curl_norm[j]),
                "harmN": _round(harm_norm[j]),
                "flowN": _round(flow_norm[j]),
                "rho": _round(rho[j]),
                "L1down": _round(L1_down_diag[j]),
                "L1up": _round(L1_up_diag[j]),
                "faceAvgC": _round(e_avg_contrib[j]),
                "faceTotC": _round(e_total_contrib[j]),
                "faceAvgSz": _round(e_avg_face_size[j], 1),
                "bndAsym": _round(e_bnd_asym[j]),
                "eBetw": _round(e_betw[j], 2),
                "eBtwNorm": _round(e_btw_norm[j]),
                "Ekin": _round(E_kin_per[j], 4),
                "Epot": _round(E_pot_per[j], 4),
            }
            for j in range(nE)
        ],
        "faces": faces,
        "topology": {
            "chainOk": bool(chain_ok),
            "rankB1": int(rank_B1), "rankB2": int(rank_B2),
            "b0": int(b0),
            "b1_raw": int(nE - rank_B1) if rank_B1 else int(b1 + rank_B2),
            "b1_filled": int(b1),
            "b2": int(b2),
            "euler": int(nV - nE + nF),
            "nF_hodge": int(nF_hodge),
            "self_loop_faces": len(rex.self_loop_face_indices),
        },
        "coupling": {
            "alpha_G": _round(alpha_G, 6),
            "alpha_T": _round(alpha_T, 6),
            "fiedler_L1": _round(fiedler_L1),
            "fiedler_LO": _round(fiedler_LO_val),
            "fiedler_RL1": _round(fiedler_RL1),
        },
        "spectra": {
            "L0": [_round(v, 6) for v in eL0],
            "L1_down": [_round(v, 6) for v in eL1_down],
            "L1_full": [_round(v, 6) for v in eL1_full],
            "L2": [_round(v, 6) for v in eL2],
            "LO": [_round(v, 6) for v in eL_O],
            "RL1": [_round(v, 6) for v in evals_RL1],
            "RL": [_round(v, 6) for v in evals_RL_full],
            "field_M": [_round(v, 6) for v in field_evals],
            "fiedler_L0": _round(eL0[np.argsort(eL0)[1]]) if nV > 1 else 0,
            "fiedler_LO": _round(fiedler_LO_val),
            "fiedler_RL1": _round(fiedler_RL1),
        },
        "hodge": {
            "gradNorm": _round(float(np.linalg.norm(grad_comp))),
            "curlNorm": _round(float(np.linalg.norm(curl_comp))),
            "harmNorm": _round(float(np.linalg.norm(harm_comp))),
            "totalNorm": _round(float(np.linalg.norm(flow))),
            "gradPct": _round(pct_grad * 100, 1),
            "curlPct": _round(pct_curl * 100, 1),
            "harmPct": _round(pct_harm * 100, 1),
        },
        "energy": {
            "E_kin": _round(E_kin, 6),
            "E_pot": _round(E_pot, 6),
            "ratio": _round(E_ratio, 4),
            "regime": energy_regime,
        },
        "analysis": analysis,
        "overlap": overlap_pairs,
        # Private keys for reuse by analyze_signal/analyze_quotient.
        # These are stripped before final JSON serialization.
        "_flow": flow,
        "_edge_types_str": e_types_str,
        "_v_names": v_names,
        "_e_names": e_names,
    }

    # RCF sections (only if RCF modules are available)
    try:
        from rexgraph.core import _character, _rcfe, _void

        # Structural character
        chi = rex.structural_character
        phi = rex.vertex_character
        kappa = rex.coherence
        nhats = rex._rcf_bundle.get('nhats', 0)
        hat_names = rex._rcf_bundle.get('hat_names', [])

        chi_section = {
            "per_edge": [
                {hat_names[k] if k < len(hat_names) else f"ch{k}":
                 _round(float(chi[e, k]), 4) for k in range(nhats)}
                for e in range(nE)
            ],
            "per_vertex": [
                dict(
                    **{hat_names[k] if k < len(hat_names) else f"ch{k}":
                       _round(float(phi[v, k]), 4) for k in range(nhats)},
                    kappa=_round(float(kappa[v]), 4),
                )
                for v in range(nV)
            ],
        }

        summary = rex.structural_summary() if hasattr(rex, 'structural_summary') else {}
        chi_section["aggregate"] = {
            k: _round(v, 4) if isinstance(v, float) else v
            for k, v in summary.items()
        }

        # Per-channel mixing times
        try:
            ch_mix = rex.per_channel_mixing_times
            if ch_mix is not None and len(ch_mix) > 0:
                chi_section["per_channel_mixing_times"] = [
                    _round(float(t), 4) if t != float('inf') else None
                    for t in ch_mix
                ]
                aniso = _character.mixing_time_anisotropy(ch_mix, nhats)
                chi_section["mixing_time_anisotropy"] = {
                    "dominant_channel": int(aniso['dominant_channel']),
                    "slowest_channel": int(aniso['slowest_channel']),
                    "anisotropy": _round(float(aniso['anisotropy']), 4)
                    if aniso['anisotropy'] != float('inf') else None,
                }
        except Exception:
            pass

        # Inverse centrality ratio
        try:
            mu = rex.inverse_centrality_ratio
            chi_section["inverse_centrality_ratio"] = [
                _round(float(mu[v]), 4) for v in range(nV)
            ]
        except Exception:
            pass

        export["structural_character"] = chi_section

        # RCFE curvature
        curv = rex.rcfe_curvature
        strain = rex.rcfe_strain
        bianchi_ok, bianchi_res = _rcfe.verify_bianchi(
            rex.B1, rex.B2_hodge, curv, nE, rex.nF_hodge)

        export["rcfe"] = {
            "curvature": [_round(float(c), 6) for c in curv],
            "strain": _round(float(strain), 6),
            "bianchi_ok": bool(bianchi_ok),
            "bianchi_residual": _round(float(bianchi_res), 8),
        }

        # Void complex
        vc = rex.void_complex
        export["void_complex"] = {
            "n_voids": int(vc.get('n_voids', 0)),
            "n_potential": int(vc.get('n_potential', 0)),
            "void_strain": _round(float(vc.get('void_strain', 0)), 6),
            "mean_eta": _round(float(np.mean(vc['eta'])), 4) if vc.get('n_voids', 0) > 0 else 0.0,
            "n_fills_beta": int(np.sum(vc.get('fills_beta', []))) if vc.get('n_voids', 0) > 0 else 0,
            "realization_rate": _round(
                1.0 - vc.get('n_voids', 0) / max(vc.get('n_potential', 1), 1), 4),
        }

        # Fiber bundle similarity
        try:
            from rexgraph.core import _fiber
            phi_sim = rex.phi_similarity
            fb_sim = rex.fiber_similarity

            # Top vertex pairs by fiber similarity
            top_pairs = []
            for i in range(min(nV, 50)):
                for j in range(i + 1, min(nV, 50)):
                    if fb_sim[i, j] > 0.5:
                        top_pairs.append({
                            "a": v_names[i], "b": v_names[j],
                            "phi_sim": _round(float(phi_sim[i, j]), 4),
                            "fb_sim": _round(float(fb_sim[i, j]), 4),
                        })
            top_pairs.sort(key=lambda p: -p["fb_sim"])

            export["fiber_bundle"] = {
                "mean_phi_sim": _round(float(np.mean(phi_sim)), 4),
                "mean_fb_sim": _round(float(np.mean(fb_sim)), 4),
                "top_pairs": top_pairs[:10],
            }
        except Exception:
            pass

        # Interfacing (if flow signal and _interfacing are available)
        try:
            from rexgraph.core import _interfacing, _channels
            if flow is not None and nE > 0:
                # Primal signal character of the flow
                psc = rex.primal_signal_character(flow)
                export["channels"] = {
                    "primal_signal_character": [
                        _round(float(psc[k]), 4) for k in range(nhats)
                    ],
                }

                # Face-void dipole of the flow
                try:
                    fvd = rex.face_void_dipole(flow)
                    export["channels"]["face_void_dipole"] = {
                        "face_affinity": _round(float(fvd['face_affinity']), 4),
                        "void_affinity": _round(float(fvd['void_affinity']), 4),
                        "dipole_ratio": _round(float(fvd['dipole_ratio']), 4),
                        "total_projection": _round(float(fvd['total_projection']), 4),
                    }
                except Exception:
                    pass
        except (ImportError, AttributeError):
            pass

        # Dirac spectrum and graded state
        try:
            from rexgraph.core import _dirac
            d_evals = rex.dirac_eigenvalues
            total_dim = nV + nE + nF

            dirac_section = {
                "spectrum": [_round(float(v), 6) for v in d_evals[:20]],
                "total_dim": total_dim,
                "n_zero": int(np.sum(np.abs(d_evals) < 1e-8)),
            }

            # Graded state from vertex 0
            try:
                psi_re, psi_im = rex.graded_state(t=0.0, vertex_idx=0)
                per_cell, per_dim = rex.born_graded(psi_re, psi_im)
                e_part = rex.energy_partition(psi_re, psi_im)
                dirac_section["born_per_dim"] = [_round(float(p), 4) for p in per_dim]
                dirac_section["energy_partition"] = [_round(float(e), 4) for e in e_part]
            except Exception:
                pass

            export["dirac"] = dirac_section
        except Exception:
            pass

        # Hypermanifold sequence
        try:
            from rexgraph.core import _hypermanifold
            hm = rex.hypermanifold()
            manifolds = hm.get('manifolds', [])

            export["hypermanifold"] = {
                "n_levels": len(manifolds),
                "manifolds": [
                    {
                        "level": m.get('level', i),
                        "nV": int(m.get('nV', 0)),
                        "nE": int(m.get('nE', 0)),
                        "nF": int(m.get('nF', 0)),
                        "betti": list(m.get('betti', (0, 0, 0))),
                        "dof": int(m.get('dof', 0)),
                    }
                    for i, m in enumerate(manifolds)
                ],
            }

            # Harmonic shadow
            hs = rex.harmonic_shadow
            export["hypermanifold"]["harmonic_shadow"] = {
                "shadow_dim": int(hs.get('shadow_dim', 0)),
                "beta_1_at_d1": int(hs.get('beta_1_at_d1', 0)),
                "beta_1_at_d2": int(hs.get('beta_1_at_d2', 0)),
            }

            # Dimensional subsumption
            ok, violations = rex.dimensional_subsumption
            export["hypermanifold"]["subsumption_ok"] = bool(ok)
            export["hypermanifold"]["n_violations"] = len(violations)
        except Exception:
            pass

        # Extended RCFE: attributed curvature and strain equilibrium
        try:
            ac = rex.attributed_curvature()
            kappa_f = ac.get('kappa_f', np.zeros(0))
            export["rcfe"]["attributed_curvature"] = {
                "n_faces": len(kappa_f),
                "mean_kappa_f": _round(float(np.mean(kappa_f)), 6) if len(kappa_f) > 0 else 0.0,
                "max_kappa_f": _round(float(np.max(kappa_f)), 6) if len(kappa_f) > 0 else 0.0,
            }

            se = rex.strain_equilibrium()
            export["rcfe"]["strain_equilibrium"] = {
                "alpha": _round(float(se.get('alpha', 0)), 6),
                "strain_norm": _round(float(se.get('strain_norm', 0)), 6),
                "bianchi_ok": bool(se.get('bianchi_ok', True)),
                "bianchi_residual": _round(float(se.get('bianchi_residual', 0)), 8),
            }
        except Exception:
            pass

    except (ImportError, AttributeError, KeyError):
        pass

    return export


# Helpers for stripping private keys

def _strip_private(d: dict) -> dict:
    """Remove keys starting with '_' from the top level of a dict."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# Signal dashboard analysis


def analyze_signal(
    rex: RexGraph,
    *,
    vertex_labels: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Dict[str, Any]] = None,
    negative_types: Optional[List[str]] = None,
    probe_edges: Optional[List[int]] = None,
    n_steps: int = 50,
    t_max: float = 10.0,
    svg_w: int = 700,
    svg_h: int = 500,
) -> dict:
    """Compute the full data contract for the signal dashboard.

    Calls analyze() for base structural data, then appends signal-specific
    data from rex.signal_dashboard_data(): perturbation trajectories,
    field diffusion, mode classification, BIOES tags, and cascade
    activation - all precomputed in Python/Cython with zero JS math.

    Parameters
    ----------
    rex : RexGraph
        Relational complex to analyze.
    vertex_labels : sequence of str, optional
        Vertex names.
    edge_attrs : dict, optional
        Named edge attributes.
    negative_types : list of str, optional
        Edge type values representing inhibition.
    probe_edges : list of int, optional
        Edge indices to run perturbation from. Defaults to highest-energy edge.
    n_steps : int
        Number of timesteps for trajectories.
    t_max : float
        Maximum time for trajectories.
    svg_w, svg_h : int
        Canvas dimensions.

    Returns
    -------
    dict
        Combined graph + signal dashboard data contract.
    """
    # Base analysis (with perturbation disabled since we do our own)
    base = analyze(
        rex,
        vertex_labels=vertex_labels,
        edge_attrs=edge_attrs,
        negative_types=negative_types,
        svg_w=svg_w, svg_h=svg_h,
        run_perturbation=False,
    )

    # Use the flow signal that analyze() already computed
    flow = base.get("_flow")

    # Signal dashboard data
    times = np.linspace(0, t_max, n_steps, dtype=np.float64)

    # If no explicit probes, pick the highest-energy edge based on flow
    if probe_edges is None and flow is not None:
        try:
            E_kin_per, E_pot_per = rex.per_edge_energy(flow)
            total = E_kin_per + E_pot_per
            probe_edges = [int(np.argmax(total))]
        except Exception:
            probe_edges = [0] if rex.nE > 0 else []

    signal_data = rex.signal_dashboard_data(
        probe_edges=probe_edges,
        times=times,
    )

    # Strip private keys from base, merge signal data
    result = _strip_private(base)
    result["signal"] = signal_data
    result["meta"]["mode"] = "signal"

    return result


# Quotient dashboard analysis


def analyze_quotient(
    rex: RexGraph,
    *,
    vertex_labels: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Dict[str, Any]] = None,
    negative_types: Optional[List[str]] = None,
    max_vertex_presets: int = 8,
    svg_w: int = 700,
    svg_h: int = 500,
) -> dict:
    """Compute the full data contract for the quotient dashboard.

    Calls analyze() for base structural data, then appends quotient-specific
    data from rex.quotient_dashboard_data(): precomputed quotient analyses
    for vertex stars, edge type subcomplexes, and energy regime subcomplexes
    - all with Hodge decomposition, congruence classes, relative Betti
    numbers, and spectral comparisons computed in Python/Cython.

    Parameters
    ----------
    rex : RexGraph
        Relational complex to analyze.
    vertex_labels : sequence of str, optional
        Vertex names.
    edge_attrs : dict, optional
        Named edge attributes.
    negative_types : list of str, optional
        Edge type values representing inhibition.
    max_vertex_presets : int
        Maximum vertex star presets to precompute.
    svg_w, svg_h : int
        Canvas dimensions.

    Returns
    -------
    dict
        Combined graph + quotient dashboard data contract.
    """
    base = analyze(
        rex,
        vertex_labels=vertex_labels,
        edge_attrs=edge_attrs,
        negative_types=negative_types,
        svg_w=svg_w, svg_h=svg_h,
        run_perturbation=False,
    )

    # Use the flow signal and edge type labels from analyze()
    flow = base.get("_flow")
    e_types_str = base.get("_edge_types_str")
    v_names = base.get("_v_names")

    quotient_data = rex.quotient_dashboard_data(
        vertex_labels=v_names,
        edge_types_str=e_types_str,
        signal=flow,
        max_vertex_presets=max_vertex_presets,
    )

    result = _strip_private(base)
    result["quotient"] = quotient_data
    result["meta"]["mode"] = "quotient"

    return result


# Combined analysis for all dashboards


def analyze_all(
    rex: RexGraph,
    *,
    vertex_labels: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Dict[str, Any]] = None,
    negative_types: Optional[List[str]] = None,
    probe_edges: Optional[List[int]] = None,
    signal_steps: int = 50,
    signal_t_max: float = 10.0,
    max_vertex_presets: int = 8,
    svg_w: int = 700,
    svg_h: int = 500,
) -> dict:
    """Compute data contracts for all three dashboards in a single call.

    Runs analyze() once, then appends both signal and quotient data.
    More efficient than calling analyze_signal + analyze_quotient
    separately because the base analysis is shared.

    Parameters
    ----------
    rex : RexGraph
        Relational complex to analyze.
    vertex_labels, edge_attrs, negative_types : optional
        As for analyze().
    probe_edges : list of int, optional
        Edge indices for signal perturbation probes.
    signal_steps, signal_t_max : int, float
        Trajectory parameters for signal dashboard.
    max_vertex_presets : int
        Max vertex star presets for quotient dashboard.
    svg_w, svg_h : int
        Canvas dimensions.

    Returns
    -------
    dict
        Combined data contract with keys for all three dashboards:
        - All standard analyze() keys (graph dashboard)
        - "signal": signal dashboard data
        - "quotient": quotient dashboard data
    """
    base = analyze(
        rex,
        vertex_labels=vertex_labels,
        edge_attrs=edge_attrs,
        negative_types=negative_types,
        svg_w=svg_w, svg_h=svg_h,
        run_perturbation=True,
    )

    flow = base.get("_flow")
    e_types_str = base.get("_edge_types_str")
    v_names = base.get("_v_names")

    # Signal data
    times = np.linspace(0, signal_t_max, signal_steps, dtype=np.float64)

    if probe_edges is None and flow is not None:
        try:
            E_kin_per, E_pot_per = rex.per_edge_energy(flow)
            total = E_kin_per + E_pot_per
            probe_edges = [int(np.argmax(total))]
        except Exception:
            probe_edges = [0] if rex.nE > 0 else []

    signal_data = rex.signal_dashboard_data(
        probe_edges=probe_edges,
        times=times,
    )

    # Quotient data
    quotient_data = rex.quotient_dashboard_data(
        vertex_labels=v_names,
        edge_types_str=e_types_str,
        signal=flow,
        max_vertex_presets=max_vertex_presets,
    )

    result = _strip_private(base)
    result["signal"] = signal_data
    result["quotient"] = quotient_data
    result["meta"]["mode"] = "all"

    return result
