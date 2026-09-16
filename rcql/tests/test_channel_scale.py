"""Channel/star/scale readings against tiny exact reference oracles.

Dense tables occur only here, never in the native action or diagonal paths.
"""
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp
from rexgraph.channel_operator import channel_operator
from rexgraph.cochain import Cochain
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator
from rexgraph.rational_trig import exact_character

from rcql import BoundSource, Executor, SourcePolicy, call, param, parse, query, source

CELLS = [(0,), (0, 1), (1, 0, 2), (2, 0, 1, 3), (3, 4, 1, 2, 0)]


def make(cells=CELLS, weights=None, **kw):
    return RexGraph(boundary_ptr=np.array([0, *np.cumsum([len(c) for c in cells])], np.int32),
                    boundary_idx=np.array([v for c in cells for v in c], np.int32),
                    w_E=None if weights is None else np.array(weights, dtype=object), **kw)


def reference(cells, weights, c_channel, normalized):
    """Independent rational full matrices of the tiny fixture; normalized G diagonal only."""
    cols = []
    for cell in cells:
        cols.append({v: Q(1) if len(cell) == 1 else Q(-1) if i == 0 else Q(1, len(cell)-1)
                     for i, v in enumerate(cell)})
    n = len(cells)
    weights = [Q(w) for w in weights]
    T, G, C = [np.full((n, n), Q(0), dtype=object) for _ in range(3)]
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            T[i, j] = weights[i] * weights[j] * sum((a[v] * b[v] for v in a.keys() & b.keys()), Q(0))
            G[i, j] = weights[i] * weights[j] * sum((abs(a[v] * b[v]) for v in a.keys() & b.keys()), Q(0))
            if i != j:
                C[i, j] = -sum((Q(1) if c_channel == "count" else abs(a[v]*b[v])
                                for v in a.keys() & b.keys()), Q(0))
    F = T - G
    for i in range(n):
        F[i, i] = sum((abs(F[i, j]) for j in range(n) if j != i), Q(0))
        C[i, i] = -sum(C[i], Q(0))
    diags = {name: np.diag(mat).copy() for name, mat in zip("TGFC", (T, G, F, C), strict=True)}
    if normalized:
        rows = [sum(row, Q(0)) for row in G]
        diags["G"] = np.array([Q(1) - G[i, i] / d if d else Q(1) for i, d in enumerate(rows)], dtype=object)
        inv = np.array([1/np.sqrt(float(d)) if d else 0 for d in rows])
        G = np.eye(n) - inv[:, None] * np.array(G, float) * inv
    return dict(zip("TGFC", (T, G, F, C), strict=True)), diags


def run(rex, *exprs, explain=False, **params):
    return Executor(sources={"r": rex}, params=params).execute(query(source("r"), *exprs, explain=explain))


@pytest.mark.parametrize("c_channel", ["share", "count"])
@pytest.mark.parametrize("normalized,weights", [
    (False, [Q(2, 3), -2, 0, Q(4, 7), 3]), (True, [Q(2, 3), 2, 0, Q(4, 7), 3]),
    (False, [0]*5), (True, [0]*5),
])
@pytest.mark.parametrize("name", list("TGFC"))
def test_channel_actions_and_exact_diagonals_match_independent_oracles(c_channel, normalized, weights, name, monkeypatch):
    rex = make(weights=weights, c_channel=c_channel, g_channel="normalized" if normalized else "raw")
    matrices, diagonals = reference(CELLS, weights, c_channel, normalized)
    vector = Cochain(1, np.arange(10, dtype=float).reshape(5, 2)-2, source=rex)
    def forbidden(*args, **kwargs):
        pytest.fail("native action/diagonal assembled a matrix or diagonalized")
    monkeypatch.setattr(RexOperator, "as_scipy", forbidden)
    monkeypatch.setattr(RexGraph, "overlap_gramian_sparse", property(forbidden))
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    action = call("CHANNEL", name)
    result = run(rex, call("APPLY", action, param("x")),
                 call("SCALE_MOMENT", action, 1, True, True),
                 call("SCALE_MOMENT", action, 1, False, True), x=vector)
    np.testing.assert_allclose(result.values[0].values, np.array(matrices[name], float) @ vector.values, atol=1e-12)
    assert result.values[1].values.tolist() == diagonals[name].tolist()
    assert result.values[2] == sum(diagonals[name], Q(0))
    assert [value.value for value in result.exactness] == ["approximate", "rational", "rational"]
    descriptor = next(n for n in result.native_plan["nodes"] if n.get("operator") == "CHANNEL")["result"]["operator"]
    assert descriptor["shape"] == [5, 5]
    assert dict(descriptor["parameters"])["trace_normalized"] is False
    assert next(event for event in result.execution if event["operator"] == "APPLY")["methods"][0]["method"] == "factored-channel-action"


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_star_character_is_existing_incident_edge_mean_not_green_phi(normalized, exact, monkeypatch):
    rex = make([(0, 1, 3), (1, 0)], [2, 3],
               g_channel="normalized" if normalized else "raw", c_channel="count")
    chi, _ = exact_character(rex)
    def forbidden(*args, **kwargs):
        pytest.fail("star mean reached a solve or assembled channels")
    monkeypatch.setattr(RexGraph, "vertex_character", property(forbidden))
    monkeypatch.setattr(RexGraph, "overlap_gramian_sparse", property(forbidden))
    result = run(rex, call("STAR_CHARACTER", call("CELL", 0, 0), exact),
                 call("STAR_CHARACTER", call("CELL", 0, 2), exact))
    expected = [sum((row[k] for row in chi), Q(0))/2 for k in range(4)]
    if exact:
        assert result.values[0]["values"].tolist() == expected
        assert result.values[1]["values"].tolist() == [Q(1, 4)]*4
    else:
        np.testing.assert_allclose(result.values[0]["values"], np.array(expected, float))
        np.testing.assert_allclose(result.values[1]["values"], .25)
    assert result.provenance[0]["result_type"]["shape"] == [4]
    assert result.values[0]["channels"] == ("L1_down", "L_O", "L_SG", "L_C")


@pytest.mark.parametrize("kind", ["T", "G", "F", "C", "hodge"])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
def test_local_and_global_moments_are_complementary_views(kind, order):
    rex = make([(0, 1, 2), (1, 0), (2,)], [2, 3, 1])
    action = call("HODGE_OPERATOR", 1) if kind == "hodge" else call("CHANNEL", kind)
    matrix = run(rex, action).values[0].as_scipy().toarray()
    expected = np.linalg.matrix_power(matrix, order)
    result = run(rex, call("SCALE_MOMENT", action, order, True),
                 call("SCALE_MOMENT", action, order), call("CHARACTER_ENERGY", action))
    np.testing.assert_allclose(result.values[0].values, np.diag(expected), atol=1e-10)
    assert result.values[1] == pytest.approx(float(np.trace(expected)))
    assert sum(result.values[0].values) == pytest.approx(result.values[1])
    np.testing.assert_allclose(result.values[2].values, np.diag(matrix @ matrix))


@pytest.mark.parametrize("name", list("TGFC"))
def test_empty_channel_moments_keep_exact_empty_and_zero(name):
    rex = RexGraph.from_graph([], [])
    action = call("CHANNEL", name)
    result = run(rex, call("SCALE_MOMENT", action, 1, True, True),
                 call("SCALE_MOMENT", action, 1, False, True),
                 call("SCALE_MOMENT", action, 0, False, True),
                 call("CHARACTER_ENERGY", action))
    assert result.values[0].values.shape == result.values[3].values.shape == (0,)
    assert result.values[1] == result.values[2] == Q(0)
    assert result.exactness[0].value == "rational"


@pytest.mark.parametrize("weight", [Q(1, 10**200), Q(10**200)])
def test_exact_first_moment_does_not_require_float_channel_mass(weight):
    rex = make([(0, 1), (1, 2)], [weight, weight])
    result = run(rex, call("SCALE_MOMENT", call("CHANNEL", "T"), 1, False, True))
    assert result.values[0] == 4*weight*weight
    with pytest.raises(FloatingPointError):
        run(rex, call("APPLY", call("CHANNEL", "T"), call("ZERO", 1)))


@pytest.mark.parametrize("text", [
    'CHANNEL("bad")', 'CHANNEL(1)', 'STAR_CHARACTER(CELL(1, 0))',
    'SCALE_MOMENT(CHANNEL("G"), -1)', 'SCALE_MOMENT(CHANNEL("G"), true)',
    'SCALE_MOMENT(CHANNEL("G"), 2, true, true)', 'SCALE_MOMENT(GREEN(), 1)',
    'SCALE_MOMENT(BOUNDARY(1), 1)', 'CHARACTER_ENERGY(CELL(1, 0))',
])
@pytest.mark.parametrize("explain", [False, True])
def test_invalid_contracts_refuse_before_adapters(text, explain, monkeypatch):
    import rcql.executor
    rex = make()
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter ran"))
    with pytest.raises((ValueError, TypeError, SyntaxError)):
        Executor(sources={"r": rex}).execute(parse(("EXPLAIN " if explain else "") + "FROM $r RETURN " + text))


def test_foreign_channel_and_denied_read_are_refused():
    rex, other = make(), make()
    with pytest.raises(TypeError):
        run(rex, call("SCALE_MOMENT", param("op"), 1), op=channel_operator(other, "T"))
    with pytest.raises(PermissionError):
        Executor(sources={"r": BoundSource(rex, SourcePolicy.allow())}).execute(
            query(source("r"), call("CHANNEL", "T")))


@pytest.mark.parametrize("options", [{"g_channel": "normalized", "w_E": [-1]*5}, {"w_V": [2]*5}])
def test_unsupported_source_channel_contracts_refuse_during_explain(options):
    rex = make(weights=options.get("w_E"), g_channel=options.get("g_channel", "raw"))
    for name, value in options.items():
        if name not in {"g_channel", "w_E"}:
            setattr(rex, name, np.array(value))
    with pytest.raises(ValueError):
        run(rex, call("CHANNEL", "F"), explain=True)


def test_repeated_participants_do_not_silently_select_another_channel_convention():
    rex = make([(0, 0), (0, 1)])
    with pytest.raises(ValueError, match="distinct participants"):
        run(rex, call("CHANNEL", "T"), explain=True)


def test_custom_declared_symmetry_is_checked_against_materialization():
    rex = make([(0, 1), (1, 2)])
    op = RexOperator("not symmetric", (2, 2), 1, 1, lambda x: x,
                     matrix=sp.csr_matrix([[1., 2.], [0., 1.]]), source=rex, symmetric=True)
    with pytest.raises(ValueError, match="symmetric matrix"):
        run(rex, call("SCALE_MOMENT", param("op"), 2), op=op)


def test_literal_channel_keeps_descriptor_and_exact_diagonal():
    rex = make()
    op = channel_operator(rex, "F")
    result = run(rex, call("SCALE_MOMENT", param("op"), 1, True, True), op=op)
    assert result.values[0].values.tolist() == op.diagonal(exact=True).tolist()
    assert result.native_plan["nodes"][0]["value"]["type"]["operator"]["construction"] == "channel"


def test_channel_name_does_not_turn_a_custom_operator_into_a_certified_channel():
    rex = make()
    op = RexOperator("CHANNEL_T", (5, 5), 1, 1, lambda x: x, source=rex, symmetric=True)
    with pytest.raises(ValueError, match="exact SCALE_MOMENT"):
        run(rex, call("SCALE_MOMENT", param("op"), 1, True, True), op=op)


def test_trace_normalization_is_not_implicit():
    rex = make([(0, 1), (1, 2)], [3, 3])
    action = run(rex, call("CHANNEL", "T")).values[0]
    np.testing.assert_allclose(action.diagonal(), [18, 18])
    assert run(rex, call("SCALE_MOMENT", call("CHANNEL", "T"), 1)).values[0] == 36


def test_exact_identity_moment_on_hodge_and_stale_channel_selection():
    rex = make()
    result = run(rex, call("SCALE_MOMENT", call("HODGE_OPERATOR", 1), 0, True, True))
    assert result.values[0].values.tolist() == [Q(1)]*5
    action = channel_operator(rex, "G")
    # Deliberately invalidate the read only constructor selection behind the handle.
    rex._g_channel = "normalized"
    with pytest.raises(ValueError, match="changed"):
        action.diagonal(exact=True)
    with pytest.raises(ValueError, match="changed"):
        action.apply(np.zeros(5))
    with pytest.raises(ValueError, match="changed"):
        run(rex, call("SCALE_MOMENT", param("op"), 1), explain=True, op=action)


@pytest.mark.parametrize("exact", [False, True])
def test_negative_moment_builder_is_refused_before_dispatch(exact, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter ran"))
    with pytest.raises(ValueError, match="nonnegative"):
        run(make(), call("SCALE_MOMENT", call("CHANNEL", "T"), -1, True, exact))
