"""The decision engine: what it plans, what it runs, and what survives a save.

`engine.py` is 886 lines and 216 of them never ran under test. It is the layer that
inspects an input and decides how to analyse it, recording each choice with its
rationale. Two things are worth holding it to:

- a plan is a claim about what will happen, so running has to do what the plan said
- every decision carries a rationale, because the point of the log is that a user can
  read why a choice was made and override it

`EngineResult.save`/`load` writes the complex through rexgraph.io plus JSON sidecars.
A result that loads back without its plan or its analysis is a result that cannot be
reopened, so the round-trip is checked per format.
"""
from __future__ import annotations

import numpy as np
import pytest
from agent.engine import AnalysisPlan, DecisionEngine, EngineResult

TEXT = ("Alpha connects beta. Beta connects gamma. Gamma connects alpha. "
        "Delta relates to alpha. Epsilon relates to delta. Alpha relates to "
        "epsilon. The alpha pathway regulates the beta response.")


def _features(seed=0, n=24, d=6):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d)


def _edge_csv(tmp_path):
    p = tmp_path / "edges.csv"
    p.write_text("source,target,weight\n"
                 "alpha,beta,1.0\nbeta,gamma,1.0\ngamma,alpha,1.0\n"
                 "delta,alpha,0.5\nepsilon,delta,0.5\nalpha,epsilon,0.5\n")
    return str(p)


#### planning


def test_a_plan_is_produced_without_running_anything():
    plan = DecisionEngine().plan(TEXT)
    assert isinstance(plan, AnalysisPlan)
    assert plan.input_type, "the plan did not decide an input type"
    assert plan.adapter, "the plan did not choose an adapter"


def test_every_decision_carries_a_rationale():
    """The decision log exists so a choice can be read and overridden. A decision
    with an empty rationale is a choice the user cannot evaluate."""
    plan = DecisionEngine().plan(TEXT)
    assert plan.decisions, "the plan recorded no decisions"
    bare = [d for d in plan.decisions if not (d.rationale or "").strip()]
    assert not bare, f"decisions with no rationale: {[d.key for d in bare]}"


def test_every_decision_names_its_stage_and_key():
    for d in DecisionEngine().plan(TEXT).decisions:
        assert d.stage and d.key, f"decision {d!r} is unlabelled"


def test_the_plan_covers_every_stage():
    stages = {d.stage for d in DecisionEngine().plan(TEXT).decisions}
    for required in ("input", "faces", "depth"):
        assert required in stages, f"nothing was decided at stage {required!r}"


def test_text_is_planned_as_text():
    plan = DecisionEngine().plan(TEXT)
    assert plan.input_type == "text"
    assert plan.adapter == "TextAdapter"


def test_a_feature_matrix_is_planned_as_one():
    plan = DecisionEngine().plan(_features())
    assert plan.input_type in ("feature_matrix", "correlation")
    assert plan.adapter in ("FeatureMatrixAdapter", "CorrelationAdapter")


def test_an_edge_csv_is_planned_from_its_columns(tmp_path):
    plan = DecisionEngine().plan(_edge_csv(tmp_path))
    assert plan.input_type in ("edge_csv", "feature_csv")


def test_planning_is_deterministic():
    a = DecisionEngine().plan(TEXT)
    b = DecisionEngine().plan(TEXT)
    assert a.input_type == b.input_type
    assert a.depth == b.depth
    assert a.face_selection == b.face_selection
    assert [d.key for d in a.decisions] == [d.key for d in b.decisions]


def test_a_supplied_signal_is_recorded_in_the_plan():
    plan = DecisionEngine().plan(TEXT, signal=np.arange(4.0))
    assert plan.has_signal, "a supplied signal was not registered"


def test_contexts_are_recorded_in_the_plan():
    plan = DecisionEngine().plan(
        TEXT, contexts={"pathway": ["alpha", "beta"], "response": ["beta", "gamma"]})
    assert plan.has_context
    assert plan.n_contexts == 2


#### running


def test_running_produces_the_complex_the_plan_described():
    res = DecisionEngine().run(TEXT)
    assert isinstance(res, EngineResult)
    assert res.rex is not None and res.rex.nE > 0
    assert res.plan.input_type == "text"


def test_running_produces_an_analysis():
    res = DecisionEngine().run(TEXT)
    assert res.analysis, "the run returned no analysis"
    assert "construction" in res.analysis or "topology" in res.analysis


def test_the_depth_the_plan_chose_is_the_depth_that_ran():
    res = DecisionEngine().run(TEXT)
    stages = set(res.analysis)
    from agent.pipeline import AnalysisPipeline
    expected = {"quick": AnalysisPipeline.STAGES_QUICK,
                "standard": AnalysisPipeline.STAGES_STANDARD,
                "full": AnalysisPipeline.STAGES_FULL}[res.plan.depth]
    missing = [s for s in expected if s not in stages]
    assert not missing, (
        f"the plan said depth={res.plan.depth} but these stages did not run: "
        f"{', '.join(missing)}")


def test_an_explicit_depth_overrides_the_decision():
    res = DecisionEngine().run(TEXT, depth="quick")
    assert res.plan.depth == "quick", "an explicit depth was overridden by the engine"


def test_running_a_feature_matrix_builds_a_complex():
    res = DecisionEngine().run(_features())
    assert res.rex is not None and res.rex.nE > 0


def test_running_an_edge_csv_builds_a_complex(tmp_path):
    res = DecisionEngine().run(_edge_csv(tmp_path))
    assert res.rex is not None and res.rex.nE > 0


def test_a_run_reports_only_numbers():
    """No NaN or infinity anywhere in what the engine hands back."""
    import math
    res = DecisionEngine().run(TEXT)

    def bad(o, path=""):
        out = []
        if isinstance(o, dict):
            for k, v in o.items():
                out += bad(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o[:200]):
                out += bad(v, f"{path}[{i}]")
        elif isinstance(o, (float, np.floating)) and (math.isnan(o) or math.isinf(o)):
            out.append(f"{path}={o}")
        return out

    offending = bad(res.analysis, "analysis") + bad(res.interpretation or {}, "interp")
    assert not offending, "non-finite: " + "; ".join(offending[:6])


def test_a_signal_is_decomposed_when_one_is_given():
    res = DecisionEngine().run(TEXT)
    rng = np.random.RandomState(0)
    res2 = DecisionEngine().run(TEXT, signal=rng.randn(res.rex.nE))
    assert res2.signal_decomposition, "a supplied signal was not decomposed"
    d = res2.signal_decomposition
    parts = [v for k, v in d.items()
             if isinstance(v, (int, float)) and k in
             ("gradient", "curl", "harmonic")]
    if len(parts) == 3:
        assert abs(sum(parts) - 1.0) < 1e-6, f"the Hodge split does not sum to 1: {d}"


#### result round-trip


@pytest.mark.parametrize("suffix", [".rex", ".h5", ".zarr"])
def test_a_result_round_trips_through_a_file(tmp_path, suffix):
    pytest.importorskip({".rex": "numpy", ".h5": "h5py", ".zarr": "zarr"}[suffix])
    res = DecisionEngine().run(TEXT)
    path = str(tmp_path / f"result{suffix}")
    res.save(path)
    back = EngineResult.load(path)
    assert back.rex.nV == res.rex.nV and back.rex.nE == res.rex.nE
    assert tuple(back.rex.betti) == tuple(res.rex.betti), \
        "the reloaded complex has different topology"


def test_a_reloaded_result_keeps_its_plan(tmp_path):
    res = DecisionEngine().run(TEXT)
    path = str(tmp_path / "result.rex")
    res.save(path)
    back = EngineResult.load(path)
    assert back.plan is not None, "the plan did not survive the save"
    assert back.plan.input_type == res.plan.input_type
    assert back.plan.depth == res.plan.depth


def test_a_reloaded_result_keeps_its_decision_log(tmp_path):
    """The rationale is the reason to save the plan at all."""
    res = DecisionEngine().run(TEXT)
    path = str(tmp_path / "result.rex")
    res.save(path)
    back = EngineResult.load(path)
    assert back.plan.decisions, "the decision log did not survive the save"
    assert len(back.plan.decisions) == len(res.plan.decisions)
    assert all((d.rationale or "").strip() for d in back.plan.decisions)


def test_a_reloaded_result_keeps_its_analysis(tmp_path):
    res = DecisionEngine().run(TEXT)
    path = str(tmp_path / "result.rex")
    res.save(path)
    back = EngineResult.load(path)
    assert back.analysis, "the analysis did not survive the save"
    assert set(back.analysis) == set(res.analysis)


def test_a_reloaded_result_keeps_its_interpretation(tmp_path):
    res = DecisionEngine().run(TEXT)
    path = str(tmp_path / "result.rex")
    res.save(path)
    back = EngineResult.load(path)
    if res.interpretation:
        assert back.interpretation, "the interpretation did not survive the save"
