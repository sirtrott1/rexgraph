"""Every analysis depth, on every shape of complex.

`AnalysisPipeline` dispatches stages by name (`getattr(self, f"_stage_{name}")`), so
a stage is reachable without any caller naming it. Nothing tested past "standard",
which left five stages shipping in the UI unexercised: advanced, rcfe, sigma_sweep,
ricci_flow and continuum_limit.

The contract asserted here is deliberately weak: every stage the depth requests
appears in the result, and nothing it returns is a NaN or an infinity dressed as a
measurement. Anything stronger belongs with the stage that owns it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from agent.auto import FACE_RULE, auto_rex, build_rex_from_edges
from agent.pipeline import AnalysisPipeline

from rexgraph.graph import RexGraph

TEXT = ("Alpha connects beta. Beta connects gamma. Gamma connects alpha. "
        "Delta connects alpha. Alpha connects epsilon. Epsilon connects delta.")


def _pairwise(seed=0, nv=14, ne=34):
    rng = np.random.RandomState(seed)
    src = rng.randint(0, nv, ne).astype(np.int32)
    tgt = ((src + 1 + rng.randint(0, 4, ne)) % nv).astype(np.int32)
    return RexGraph(sources=src, targets=tgt)


def _shapes():
    """One complex of each shape the stages have to survive."""
    from agent.adapters import EdgeConstruction

    tri = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32))
    square = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                      targets=np.array([1, 2, 3, 0], np.int32))
    path = RexGraph(sources=np.array([0, 1, 2], np.int32),
                    targets=np.array([1, 2, 3], np.int32))
    ec = EdgeConstruction(
        sources=np.array([0, 0, 1, 1, 2, 2], np.int32),
        targets=np.array([3, 4, 3, 4, 3, 4], np.int32),
        weights=np.ones(6), signs=np.ones(6),
        type_labels=np.zeros(6, np.int32),
        vertex_labels=[f"v{i}" for i in range(5)],
        n_types=1, type_names=["e"])
    bipartite = build_rex_from_edges(ec, face_selection=FACE_RULE)
    return {
        "text (faced)": auto_rex(TEXT, face_selection=FACE_RULE),
        "text (1-rex)": auto_rex(TEXT, face_selection="none"),
        "triangle": tri,
        "square (4-gon)": square,
        "path (acyclic)": path,
        "bipartite": bipartite,
        "random pairwise": _pairwise(),
    }


SHAPES = _shapes()
DEPTHS = {"quick": AnalysisPipeline.STAGES_QUICK,
          "standard": AnalysisPipeline.STAGES_STANDARD,
          "full": AnalysisPipeline.STAGES_FULL}


def _finite(obj, path="") -> list[str]:
    """Every float a stage reports has to be a number. Returns the offending paths."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += _finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:200]):
            bad += _finite(v, f"{path}[{i}]")
    elif isinstance(obj, (float, np.floating)) and (math.isnan(obj)
                                                    or math.isinf(obj)):
        bad.append(f"{path} = {obj}")
    return bad


@pytest.mark.parametrize("depth", list(DEPTHS))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_a_depth_runs_on_a_shape(depth, shape):
    """The stages a depth declares are the stages that come back."""
    out = AnalysisPipeline(SHAPES[shape]).run(depth=depth)
    assert isinstance(out, dict) and out, f"{depth}/{shape} returned nothing"
    missing = [s for s in DEPTHS[depth] if s not in out]
    assert not missing, f"{depth}/{shape} did not produce: {', '.join(missing)}"


@pytest.mark.parametrize("depth", list(DEPTHS))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_a_depth_reports_only_numbers(depth, shape):
    """No NaN or infinity dressed up as a measurement."""
    out = AnalysisPipeline(SHAPES[shape]).run(depth=depth)
    bad = _finite(out)
    assert not bad, f"{depth}/{shape} non-finite: " + "; ".join(bad[:6])


@pytest.mark.parametrize("shape", list(SHAPES))
def test_depth_is_cumulative(shape):
    """quick is a prefix of standard is a prefix of full, and a stage computed at
    two depths gives the same answer at both."""
    rex = SHAPES[shape]
    q = AnalysisPipeline(rex).run(depth="quick")
    s = AnalysisPipeline(rex).run(depth="standard")
    f = AnalysisPipeline(rex).run(depth="full")
    assert set(q) <= set(s) <= set(f), "the depths are not nested"
    for stage in q:
        assert repr(q[stage]) == repr(s[stage]) == repr(f[stage]), (
            f"stage {stage!r} differs between depths on {shape}")


@pytest.mark.parametrize("stage", ["advanced", "rcfe", "sigma_sweep",
                                   "ricci_flow", "continuum_limit"])
def test_each_full_only_stage_produces_something(stage):
    """The five stages that only depth='full' reaches. These shipped untested."""
    out = AnalysisPipeline(SHAPES["text (faced)"]).run(depth="full")
    assert stage in out, f"{stage} missing from depth=full"
    assert out[stage] is not None, f"{stage} returned None"


def test_an_empty_complex_does_not_crash_any_depth():
    """A complex with no edges is a real input: a document that produced nothing."""
    empty = RexGraph(sources=np.array([], np.int32), targets=np.array([], np.int32))
    for depth in DEPTHS:
        out = AnalysisPipeline(empty).run(depth=depth)
        assert isinstance(out, dict), f"{depth} on an empty complex returned {out!r}"


def test_a_single_edge_does_not_crash_any_depth():
    one = RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32))
    for depth in DEPTHS:
        assert isinstance(AnalysisPipeline(one).run(depth=depth), dict)
