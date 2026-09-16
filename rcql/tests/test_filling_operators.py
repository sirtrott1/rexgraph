"""RCQL delegates attachment and harmonic shadow to Core."""
from dataclasses import replace

import numpy as np
import pytest

from rexgraph.cochain import Chain, Cochain
from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rcql import BoundSource, Executor, SourcePolicy, parse


def fixture():
    return RexGraph.from_cells([4, [[0, 1, 2], [0, 1], [0, 2]]], relation_ids=[3, 7, 11])


def test_core_parity_and_owned_rebinding():
    rex = fixture()
    cycle = Chain(1, np.array([2, -1, -1]), source=rex)
    engine = Executor(sources={"r": rex}, params={"c": cycle})
    before = object_digest(rex)
    result = engine.execute(parse('FROM $r RETURN FILL(cycle=$c), HARMONIC_SHADOW()'))
    assert object_digest(result.values[0]) == object_digest(rex.fill_cycle(cycle.values))
    assert result.values[1] == rex.harmonic_shadow and object_digest(rex) == before
    after = Executor(sources={"r": result.values[0]}).execute(parse(
        'FROM $r RETURN BETTI(1), HARMONIC_SHADOW(), STATE_HASH()'))
    assert after.values[:2] == (0, {"shadow_dim": 1, "beta_1_at_d1": 1, "beta_1_at_d2": 0})


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("mode", ["cochain", "foreign", "block", "grade", "array"])
def test_bad_basis_variance_and_shape_refused_before_adapter(mode, explain, monkeypatch):
    import rcql.executor
    rex = fixture()
    value = {"cochain": Cochain(1, np.ones(3), source=rex),
             "foreign": Chain(1, np.ones(3), source=fixture()),
             "block": Chain(1, np.ones((3, 2)), source=rex),
             "grade": Chain(0, np.ones(4), source=rex), "array": np.ones(3)}[mode]
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": rex}, params={"c": value}).execute(replace(
            parse('FROM $r RETURN FILL($c)'), explain=explain))


def test_explain_defers_attachment_and_rank_work(monkeypatch):
    rex = fixture()
    monkeypatch.setattr(RexGraph, "fill_cycle", lambda *a: pytest.fail("attached"))
    monkeypatch.setattr(RexGraph, "harmonic_shadow", property(lambda self: pytest.fail("rank run")))
    out = Executor(sources={"r": rex}, params={"c": Chain(1, np.array([2, -1, -1]), source=rex)}).execute(
        parse('EXPLAIN FROM $r RETURN FILL($c), HARMONIC_SHADOW()'))
    assert out.execution == ()


def test_fill_requires_identity_but_never_a_store_write():
    rex = fixture()
    c = Chain(1, np.array([2, -1, -1]), source=rex)
    policy = SourcePolicy.allow("read")
    engine = Executor(sources={"r": BoundSource(rex, policy)}, params={"c": c})
    with pytest.raises(PermissionError):
        engine.execute(parse('FROM $r RETURN FILL($c)'))
    assert engine.execute(parse('FROM $r RETURN HARMONIC_SHADOW()')).values[0] == rex.harmonic_shadow


def test_persist_filled_state_and_reopen(tmp_path):
    import rcdb
    path = f"rex://{tmp_path / 'db'}"
    rex = fixture()
    store = rcdb.open_store(path)
    store.put("r", rex)
    engine = Executor(sources={"r": rex}, params={"c": Chain(1, np.array([2, -1, -1]), source=rex)})
    child = engine.execute(parse('FROM $r RETURN FILL($c)')).values[0]
    Executor(sources={"db": store}, params={"child": child}).execute(parse(
        'FROM $db MUTATE "filled" SET state=$child, actor="Art" COMMIT'))
    store.close()
    store = rcdb.open_store(path)
    try:
        engine = Executor(sources={"db": store})
        out = engine.execute(parse('FROM RCDB_GET($db,"filled") RETURN BETTI(1), HARMONIC_SHADOW(), STATE_HASH()'))
        assert out.values[0] == 0 and out.values[2] == object_digest(child)
        assert object_digest(store.get("r")) == object_digest(rex)
    finally:
        store.close()
