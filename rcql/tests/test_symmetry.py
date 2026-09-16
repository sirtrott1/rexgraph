"""Typed symmetry groups keep the declared source, exact maps and lazy products."""
from dataclasses import replace

import pytest

from rexgraph import RexGraph
from rexgraph.chain_map import CoordinateComplex, GradedMap, SymmetryGroup
from rcql import BoundSource, Executor, SourcePolicy, parse


def setup():
    r = RexGraph.from_cells([3, [[0, 1], [1, 2]]], relation_ids=[17, 91])
    c = CoordinateComplex.from_rex(r)
    p = GradedMap(c, c, (((2, 0, 1), (1, 1, 1), (0, 2, 1)), ((1, 0, -1), (0, 1, -1))))
    return r, p, Executor(sources={"r": r}, params={"p": p, "generators": [p]})


@pytest.mark.parametrize("expression", ["SYMMETRY([$p],[1])", "SYMMETRY($generators,word=[1])"])
def test_text_and_parameter_generators_members_and_materialized_map(expression):
    _, p, engine = setup()
    out = engine.execute(parse(f'FROM $r LET g={expression} RETURN g,g.map,g.word,g.inverse.map,'
                                'g.identity.map,g.sizes,g.generator_count'))
    assert isinstance(out.values[0], SymmetryGroup)
    assert out.values[1].declaration.components == p.components
    assert out.values[2] == (1,) and out.values[3].declaration.components == p.components
    assert out.values[5:] == ((3, 2), 1)
    engine.params["map"] = out.values[1]
    assert engine.execute(parse('FROM $r RETURN CHAIN_MAP($map)')).values[0].commutation_residuals == (0,)
    assert "core-exact-chain-symmetry" in str(out.execution)


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("text", ["SYMMETRY([])", "SYMMETRY([CELL(1,0)])", "SYMMETRY([$p],[0])",
    "SYMMETRY([$p],[2])", "SYMMETRY([$p],[true])", "SYMMETRY([$p],[1.0])", "SYMMETRY([$p]).generators",
    "CHAIN_MAP(SYMMETRY([$p],[1]).map)"])
def test_invalid_arguments_before_adapter(explain, text, monkeypatch):
    import rcql.executor
    _, _, engine = setup()
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((ValueError, TypeError)):
        engine.execute(replace(parse('FROM $r RETURN '+text), explain=explain))


def test_explain_certifies_generators_without_selected_product(monkeypatch):
    _, _, engine = setup()
    monkeypatch.setattr(SymmetryGroup, "__post_init__", lambda *a: pytest.fail("group constructed"))
    monkeypatch.setattr(GradedMap, "then", lambda *a: pytest.fail("product materialized"))
    result = engine.execute(parse('EXPLAIN FROM $r RETURN SYMMETRY([$p],[1]).inverse.map'))
    assert result.execution == () and "euclidean_chain_symmetries" in str(result.values)


@pytest.mark.parametrize("explain", [False, True])
def test_reject_nonorthogonal_and_foreign_generators(explain, monkeypatch):
    import rcql.executor
    _, p, engine = setup()
    _, foreign, _ = setup()
    scale = GradedMap(p.domain, p.domain,
        tuple(tuple((i, i, 2) for i in range(n)) for n in p.domain.sizes))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    for value in (foreign, scale):
        engine.params["p"] = value
        with pytest.raises((ValueError, TypeError)):
            engine.execute(replace(parse('FROM $r RETURN SYMMETRY([$p])'), explain=explain))


@pytest.mark.parametrize("explain", [False, True])
def test_read_permission_required(explain, monkeypatch):
    import rcql.executor
    r, _, engine = setup()
    engine.sources["r"] = BoundSource(r, SourcePolicy.allow())
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises(PermissionError):
        engine.execute(replace(parse('FROM $r RETURN SYMMETRY([$p])'), explain=explain))


def test_bound_group_and_stale_result():
    r, p, engine = setup()
    engine.params["g"] = SymmetryGroup([p], [1])
    assert engine.execute(parse('FROM $r RETURN $g.inverse.word')).values == ((-1,),)
    r.add_edges([0], [2], relation_ids=[101])
    for prefix in ("", "EXPLAIN "):
        with pytest.raises(ValueError, match="changed"):
            engine.execute(parse(prefix+'FROM $r RETURN $g.map'))


def test_rcdb_reopen_uses_stored_boundaries_without_mutation(tmp_path):
    import rcdb
    from contextlib import closing
    from rexgraph.io.catalog import object_digest
    r, _, _ = setup()
    digest = object_digest(r)
    path = f"rex://{tmp_path / 'db'}"
    with closing(rcdb.open_store(path)) as db:
        db.put("r", r)
    with closing(rcdb.open_store(path)) as db:
        r = db.get("r")
        c = CoordinateComplex.from_rex(r)
        p = GradedMap(c, c, tuple(tuple((i, i, -1) for i in range(n)) for n in c.sizes))
        result = Executor(sources={"r": r}, params={"p": p}).execute(parse('FROM $r RETURN SYMMETRY([$p],[1]).map'))
        assert result.values[0].commutation_residuals == (0,)
        assert object_digest(db.get("r")) == digest and db.read_record("r").record.version == 1
