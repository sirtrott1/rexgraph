"""
agent.overview: one call that says what a complex is.

`rexgraph.analysis.analyze` was this, for the old dashboard. What it produced was
partly a picture of where a linear cut fell and partly a table of comparison baselines,
and getting it cost a dense eigendecomposition. This assembles the same kind of answer
from the readings that are exact and eigen-free, and adds the thing a section-by-section
report cannot have: the sections checked against each other.

    shape        counts, and the ARITY distribution. A graph-shaped summary cannot show
                 this, and it is the first thing worth knowing about a relational
                 complex: how many relations are pairwise and how many branch.
    homology     Betti and the rank tower, where each rank is the curl of the grade
                 below and the gradient of the grade above.
    integrity    whether every declared face bounds, naming the ones that do not.
    character    what the complex is made of, per channel, by name.
    flow         a signal split into gradient, curl and harmonic; without one, the
                 DIMENSIONS those parts would have.
    curvature    where the strain sits, per face, with the conservation residual.
    consistency  the cross-checks. Euler from the cell counts against Euler from Betti,
                 and the harmonic dimensions against Betti. Both are identities, so a
                 disagreement is a defect and not a tolerance.

Nothing here reports a Fiedler value, a partition derived from one, or PageRank,
betweenness, clustering or community. The first describes a cut rather than the cells;
the rest are the comparison column, still available through
`analyze(..., standard_metrics=True)` where comparison is the point.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from agent.metrics import coherence_kappa

__all__ = ["overview", "shape_of", "consistency_of"]


def shape_of(rex) -> dict:
    """Counts, and how the relations are distributed over arity.

    The arity histogram is the reading a graph summary has no room for: a complex whose
    relations are all pairwise and one carrying 4-ary relations are different objects,
    and `nE` alone does not distinguish them.
    """
    rex._ensure_clean()
    bp = rex._boundary_ptr
    widths = ([2] * int(rex.nE) if bp is None
              else [int(w) for w in np.diff(np.asarray(bp))])
    hist = Counter(widths)
    declared = int(getattr(rex, "_nF", 0) or 0)
    surviving = int(rex.nF_hodge)
    return {
        "nV": int(rex.nV), "nE": int(rex.nE),
        "nF_declared": declared, "nF": surviving,
        # a face that arrived and does not bound is dropped from the homology, so the
        # gap between what was declared and what survives is worth stating rather than
        # absorbing
        "faces_dropped": declared - surviving,
        "arity": {str(k): int(v) for k, v in sorted(hist.items())},
        "has_branching": bool(any(w > 2 for w in widths)),
        "n_branching": int(sum(v for k, v in hist.items() if k > 2)),
        "max_arity": int(max(widths)) if widths else 0,
    }


def consistency_of(rex, tower: dict) -> dict:
    """The identities that must hold, checked across the sections.

    Euler is `sum (-1)^k n_k` and also `sum (-1)^k beta_k`; the harmonic dimension at
    grade k is `beta_k`. Both are identities over the integers, so a disagreement is a
    defect rather than a number to interpret, and reporting them separately without
    comparing them is how one goes unnoticed.
    """
    betti = [int(b) for b in rex.betti]
    grades = tower.get("grades") or []
    harmonic = [int(g.get("harmonic", 0)) for g in grades]
    euler_counts = int(tower.get("euler", 0))
    euler_betti = int(tower.get("euler_from_betti", 0))
    matched = harmonic[:len(betti)] == betti[:len(harmonic)]
    unbounded = list(rex.self_loop_face_indices)
    return {
        "euler_from_counts": euler_counts,
        "euler_from_betti": euler_betti,
        "euler_agrees": euler_counts == euler_betti,
        "harmonic_equals_betti": bool(matched),
        "chain_valid": not unbounded,
        "unbounded_faces": unbounded[:16],
        "ok": bool(euler_counts == euler_betti and matched and not unbounded),
    }


def _character_summary(rex) -> dict:
    """What the complex is made of, per channel, by name.

    Means over the simplex, so the shares still sum to one and the dominant channel is
    readable. Named because `L1_down` and `L_O` coincide on an unweighted complex and
    positional indices make that look accidental.
    """
    names = list(getattr(rex, "hat_names", None) or [])
    out = {"channels": names}
    for key, attr in (("relations", "structural_character"),
                      ("entities", "vertex_character")):
        try:
            values = np.asarray(getattr(rex, attr), dtype=float)
        except Exception:                        # noqa: BLE001
            continue
        if values.ndim != 2 or values.size == 0:
            continue
        mean = values.mean(axis=0)
        keyed = {str(n): round(float(v), 6)
                 for n, v in zip(names or range(mean.shape[0]), mean, strict=False)}
        out[key] = {"mean": keyed,
                    "dominant": max(keyed, key=keyed.get) if keyed else None}
    try:
        kappa = coherence_kappa(rex)
        out["coherence_mean"] = round(float(kappa.mean()), 6) if kappa.size else None
    except Exception:                            # noqa: BLE001
        pass
    return out


def _flow(rex, signal) -> dict:
    """A signal's Hodge split, or the dimensions those parts would have.

    With no signal there is still an answer: how many independent directions each part
    HAS. That is structural and exact, where the split of a particular signal is a fact
    about that signal.
    """
    dims = rex.hodge_dimensions(1)
    out = {"dimensions": {k: dims[k] for k in ("gradient", "curl", "harmonic")},
           "n_relations": int(dims.get("n_cells", rex.nE))}
    if signal is None:
        out["reading"] = ("dimensions only; pass a signal to split one into its parts")
        return out
    sig = np.asarray(signal, dtype=float).ravel()
    if sig.shape[0] != int(rex.nE):
        raise ValueError(
            f"the signal has {sig.shape[0]} entries for {int(rex.nE)} relations")
    grad, curl, harm = rex.hodge(np.ascontiguousarray(sig))
    energies = {k: float(np.dot(v, v)) for k, v in
                (("gradient", grad), ("curl", curl), ("harmonic", harm))}
    total = sum(energies.values())
    out["energy"] = {k: round(v, 6) for k, v in energies.items()}
    out["share"] = ({k: round(v / total, 6) for k, v in energies.items()}
                    if total > 0 else dict.fromkeys(energies, 0.0))
    # additive because the chain condition makes the cross terms vanish, so a residual
    # here would mean the decomposition is wrong rather than that the data is awkward
    out["cross_residual"] = round(float(np.dot(sig, sig)) - total, 9)
    return out


def _curvature(rex) -> dict:
    if int(rex.nF_hodge) == 0:
        return {"n_faces": 0, "total": 0.0, "peak": None, "bianchi_residual": 0.0}
    ac = rex.attributed_curvature()
    kappa = np.asarray(ac["kappa_f"], dtype=float)
    eq = rex.strain_equilibrium()
    peak = int(np.argmax(kappa)) if kappa.size else None
    return {
        "n_faces": int(rex.nF_hodge),
        "total": round(float(kappa.sum()), 6),
        "peak": ({"face": peak, "kappa": round(float(kappa[peak]), 6)}
                 if peak is not None else None),
        "bianchi_residual": round(float(eq.get("bianchi_residual", 0.0)), 9),
        "strain_norm": round(float(eq.get("strain_norm", 0.0)), 6),
    }


def overview(rex, *, labels=None, signal=None, cells: bool = True,
             limit: int = 200, positions: bool = True) -> dict:
    """Everything worth saying about a complex, in one call."""
    tower = rex.rank_tower()
    out = {
        "shape": shape_of(rex),
        "homology": {
            "betti": [int(b) for b in rex.betti],
            "ranks": tower.get("ranks"),
            "grades": tower.get("grades"),
        },
        "character": _character_summary(rex),
        "flow": _flow(rex, signal),
        "curvature": _curvature(rex),
        "consistency": consistency_of(rex, tower),
    }
    try:
        from rexgraph.tower import closure_at, tower_law
        law = tower_law(rex)
        out["tower"] = {
            "mass": law["mass"], "trace": law["trace"], "moments": law["moments"],
            # tr(L_k) = ||B_k||^2 + ||B_k+1||^2 is an identity, so a failure is a defect
            "law_holds": law["holds"], "law_residual": law["residual"],
            "closure_at_2": closure_at(rex, 2),
        }
    except Exception:                            # noqa: BLE001 - optional reading
        pass
    try:
        from rexgraph.mesh_health import harmonic_health
        health = harmonic_health(rex, signal)
        out["circulation"] = {
            "dim_H": int(health.get("dim_H", 0)),
            "frustration_total": health.get("frustration_total"),
            "coparticipation_total": health.get("coparticipation_total"),
            "health_ratio": health.get("health_ratio"),
        }
    except Exception:                            # noqa: BLE001 - optional reading
        pass
    if cells:
        from agent.cell_view import cells as _cells
        out["cells"] = _cells(rex, labels=labels, signal=signal, limit=limit,
                              positions=positions)
    return out
