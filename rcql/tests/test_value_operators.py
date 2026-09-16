"""Exact value operations, namespace integrity and direct versus planned parity."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from rexgraph.cells import Cell
from rexgraph.cochain import Chain, Cochain, Field
from rexgraph.graph import RexGraph
from rexgraph.type_accession import TypeAccession

from rcql import Executor, call, catalogued, operator_inventory, query, source
from rcql.names import canonical_name
from rcql.operators import _REGISTRY, get_operator, register
from rcql.signatures import _CATALOGUE, register as register_signature
from rcql.value_contracts import ARGUMENTS


@pytest.fixture
def rex():
    return RexGraph.from_hypergraph(np.array([0, 4, 6, 8, 9]), np.array([0, 1, 2, 3, 1, 0, 4, 4, 5]))


def run(rex, name, *args, explain=False):
    return Executor(sources={"r": rex}).execute(replace(query(source("r"), call(name, *args)), explain=explain))


@pytest.mark.parametrize("name", sorted(catalogued() | {"REX"}))
def test_registration_cannot_replace_an_existing_name(name):
    signature, adapter = _CATALOGUE.get(name), _REGISTRY[name]
    for spelling in (name, name.lower(), name.title()):
        with pytest.raises(ValueError, match="already registered"):
            register(spelling)(lambda *args: None)
        if signature is not None:
            with pytest.raises(ValueError, match="already registered"):
                register_signature(replace(signature, name=spelling))
    assert _CATALOGUE.get(name) is signature
    assert _REGISTRY[name] is adapter


@pytest.mark.parametrize("name", ["", "x y", "x-y", "x.y", "1NAME", "é", "ſUM", "ＨＥＡＤ", None, 1])
def test_operator_names_are_ascii_and_unambiguous(name):
    with pytest.raises(ValueError, match="ASCII"):
        canonical_name(name)


@pytest.mark.parametrize("name", ["NOT", "True", "FALSE", "none"])
def test_literal_and_boolean_syntax_names_cannot_be_shadowed(name):
    with pytest.raises(ValueError, match="expression syntax"):
        register(name)(lambda source: None)


@pytest.mark.parametrize("name", ["ſUM", "SıGNING", "ＳＵＭ", "SUM ", "not"])
def test_every_operator_entry_path_rejects_ambiguous_names(name):
    from rcql import Call, lookup
    from rcql.builder import source_call
    for factory in (call, source_call, Call, lookup, get_operator):
        with pytest.raises(ValueError):
            factory(name)


def test_text_calls_validate_before_uppercase_normalization():
    from rcql import parse
    with pytest.raises(SyntaxError, match="ASCII"):
        parse("FROM $r RETURN SıGNING()")


@pytest.mark.parametrize("imports", [
    "import rcql.value_operators; import rcql.operators",
    "import rcql.value_contracts; import rcql.arguments; import rcql.operators",
    "import rcql.arguments; import rcql.value_operators; import rcql.signatures",
])
def test_new_modules_can_be_imported_independently(tmp_path, imports):
    import pathlib
    import subprocess
    import sys
    import rcql
    root = str(pathlib.Path(rcql.__file__).resolve().parent.parent)
    code = (
        f"import sys; sys.path.insert(0, {root!r}); {imports}; "
        "from rcql.value_operators import rate; from fractions import Fraction; "
        "assert rate(None, 1, 3) == Fraction(1, 3); "
        "from rcql.operators import _REGISTRY; from rcql import catalogued; "
        "assert set(_REGISTRY) == catalogued() | {'REX'}"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_all_new_names_are_current_and_have_matching_argument_contracts():
    from rcql.arguments import EXPRESSION_ARGUMENTS
    assert set(ARGUMENTS) <= catalogued()
    for name, contract in ARGUMENTS.items():
        assert EXPRESSION_ARGUMENTS[name] == contract


def test_inventory_names_are_canonical():
    assert all(row["name"] == canonical_name(row["name"]) for row in operator_inventory())


def test_primary_support_and_degree_keep_branching_witnesses_and_loops(rex):
    assert run(rex, "SUPPORT", Cell(rex, 1, 0)).values[0].indices == (0, 1, 2, 3)
    assert run(rex, "SUPPORT", Cell(rex, 1, 2)).values[0].indices == (4,)
    assert run(rex, "SUPPORT", Cell(rex, 1, 3)).values[0].indices == (5,)
    assert run(rex, "SHARED_BOUNDARY", Cell(rex, 1, 0), Cell(rex, 1, 1)).values[0].indices == (0, 1)
    np.testing.assert_array_equal(run(rex, "DEGREE", 0).values[0].values, [2, 2, 1, 1, 1, 1])
    assert run(rex, "DEGREE", Cell(rex, 0, 4)).values == (1,)
    np.testing.assert_array_equal(run(rex, "DEGREE", 1).values[0].values, [0, 0, 0, 0])
    cancelled = Chain(1, np.array([0, 0, 1, 0]), source=rex)
    assert run(rex, "SUPPORT", cancelled).values[0].indices == ()


@pytest.mark.parametrize("explain", [False, True])
def test_bad_grades_sources_variance_and_basis_are_refused(rex, explain):
    other = RexGraph.from_graph(sources=[0], targets=[1])
    values = Cochain(1, np.ones(rex.nE, dtype=int), source=rex)
    bad = [
        ("SUPPORT", Cell(rex, 0, 0)), ("SUPPORT", values),
        ("DEGREE", -1), ("DEGREE", True),
        ("FIELD", values, 0), ("FIELD", Chain(1, values.values, source=rex)),
        ("MASS", values, 0), ("GRAM", [values, Cochain(1, np.ones(1), source=other)]),
        ("GRAM", [values, Cochain(1, values.values, cell_keys=(3, 2, 1, 0), source=rex)]),
        ("ARGMIN", [Cell(rex, 1, 0), Cell(other, 1, 0)], [1, 2]),
        ("ORIENTED_MOMENT", values, values),
    ]
    for name, *args in bad:
        with pytest.raises((TypeError, ValueError)):
            run(rex, name, *args, explain=explain)


def test_gram_rank_retains_large_integer_and_rational_coefficients(rex, monkeypatch):
    scale = 2**90
    a = Cochain(1, np.array([scale, 1, 0, 0], dtype=object), source=rex)
    b = a.with_values(np.array([scale, 2, 0, 0], dtype=object))
    matrix = run(rex, "GRAM", [a, b]).values[0]
    assert matrix.tolist() == [[Fraction(scale**2 + 1), Fraction(scale**2 + 2)],
                               [Fraction(scale**2 + 2), Fraction(scale**2 + 4)]]
    assert run(rex, "GRAM_RANK", [a, b]).values == (2,)
    from rcql import value_operators
    monkeypatch.setattr(value_operators, "gram", lambda *a, **k: pytest.fail("rank assembled a Gram matrix"))
    assert run(rex, "GRAM_RANK", [a, b, a]).values == (2,)
    assert run(rex, "GRAM_RANK", []).values == (0,)
    assert run(rex, "GRAM", []).values[0].shape == (0, 0)
    from rexgraph.graded_metric import DiagonalMetric
    metric = DiagonalMetric(rex, 1, (1.1, 2.2, 3.3, 4.4))
    assert run(rex, "GRAM_RANK", [a, b], metric).values == (2,)


def test_local_incidence_readings_do_not_expand_a_coefficient_vector(rex, monkeypatch):
    from rcql import operators
    _ = rex._v2e                       # warm the cache before the patches below
    monkeypatch.setattr(operators, "boundary", lambda *a, **k: pytest.fail("SUPPORT built a coefficient field"))
    monkeypatch.setattr(RexGraph, "relation_supports", lambda *a, **k: pytest.fail("DEGREE scanned all relations"))
    assert run(rex, "SUPPORT", Cell(rex, 1, 0)).values[0].indices == (0, 1, 2, 3)
    assert run(rex, "DEGREE", Cell(rex, 0, 0)).values == (2,)


def test_complex_gram_uses_the_hermitian_pairing(rex):
    a = Cochain(1, np.array([1j, 1, 0, 0]), source=rex)
    b = a.with_values(np.array([1, 1j, 0, 0]))
    matrix = run(rex, "GRAM", [a, b], None, False).values[0]
    np.testing.assert_array_equal(matrix, np.stack([a.values, b.values]).conj() @ np.stack([a.values, b.values]).T)
    with pytest.raises(TypeError):
        run(rex, "GRAM", [a, b])


def test_boundary_moments_use_rational_shares_and_empty_upper_zero(rex):
    a = Chain(1, np.array([1, 0, 0, 0]), source=rex)
    b = Chain(1, np.array([0, 1, 0, 0]), source=rex)
    assert run(rex, "ORIENTED_MOMENT", a, a).values == (Fraction(4, 3),)
    assert run(rex, "FIELD_QUOTIENT", b, a).values == (Fraction(-1),)
    assert run(rex, "COBOUNDARY_MOMENT", a, a).values == (Fraction(0),)
    with pytest.raises(ZeroDivisionError):
        run(rex, "COFIELD_QUOTIENT", a, a)


@pytest.mark.parametrize("name", ["RATE", "TEMPORAL_RATE", "MOMENT_RATE"])
def test_finite_rates_preserve_exact_blocks_and_large_integers(rex, name):
    big = 2**100
    assert run(rex, name, big, 3).values == (Fraction(big, 3),)
    a = Cochain(1, np.array([[big, 1], [2, 3], [4, 5], [6, 7]], dtype=object), source=rex)
    result = run(rex, name, a, Fraction(2, 3)).values[0]
    assert result.source is rex and result.grade == 1
    assert result.values[0, 0] == Fraction(big * 3, 2)
    for interval in (True, 0, float("inf"), float("nan")):
        with pytest.raises((TypeError, ValueError, ZeroDivisionError)):
            run(rex, name, 1, interval)


def test_mass_handles_signed_integer_minimum(rex):
    value = Cochain(1, np.array([-(2**63), 1, -2, 3], dtype=np.int64), source=rex)
    assert run(rex, "MASS", value).values == (2**63 + 6,)


def test_extrema_keep_exact_order_and_first_tie(rex):
    cells = [Cell(rex, 1, 0), Cell(rex, 1, 1)]
    assert run(rex, "ARGMIN", cells, [2**100 + 1, 2**100]).values[0].index == 1
    assert run(rex, "ARGMAX", cells, [Fraction(1, 3), Fraction(1, 3)]).values[0].index == 0
    for values, measures in [([], []), (cells, [1]), (cells, [float("nan"), 1]), ([1, "x"], [1, 2])]:
        with pytest.raises((TypeError, ValueError)):
            run(rex, "ARGMIN", values, measures)


def test_field_and_type_view_do_not_infer_maps(rex):
    value = Cochain(1, np.array([1, 2, 3, 4]), source=rex)
    made = run(rex, "FIELD", value).values[0]
    assert isinstance(made, Field) and made.cochain is value
    accession = TypeAccession(rex, 1, "selected", ((0, 0, 1),))
    a = run(rex, "TYPE_VIEW", value, accession, True).values[0]
    b = get_operator("ACCESS").fn(rex, value, accession, True)
    np.testing.assert_array_equal(a.values, b.values)
    with pytest.raises(TypeError):
        run(rex, "TYPE_VIEW", value, "selected")


def test_weights_signing_and_orientation_are_separate(rex):
    rex.set_cell_attrs([0, 1, 2, 3], w_E=np.array([Fraction(2, 3), 2**80, 1, 1], dtype=object),
                       signs=np.array([-1, 1, -1, 1]))
    a, b = Cell(rex, 1, 0), Cell(rex, 1, 1)
    assert run(rex, "WEIGHT", a).values == (Fraction(2, 3),)
    assert run(rex, "WEIGHT", b).values == (Fraction(2**80),)
    assert run(rex, "SIGNING", a).values == (1,)
    assert run(rex, "SIGNING", b).values == (0,)
    assert run(rex, "PARITY", [a, b, a]).values == (1,)
    np.testing.assert_array_equal(run(rex, "ORIENTATION", a).values[0].values, [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(run(rex, "ORIENTATION", b).values[0].values, [0, 1, 0, 0, 0, 0])


def test_chain_valid_checks_declared_faces_before_the_hodge_filter():
    r = RexGraph.from_graph(sources=[0, 1, 2], targets=[1, 2, 0])
    assert run(r, "CHAIN_VALID").values == (True,)
    r.add_faces([np.array([0, 1])], [np.array([1.0, 1.0])])
    assert run(r, "CHAIN_VALID").values == (False,)


def test_green_operator_field_and_gram_match_the_declared_linear_system(rex):
    cells = [Cell(rex, 1, i) for i in range(rex.nE)]
    unit = np.eye(rex.nE)
    # This small dense object is the test oracle, not the query execution path.
    from rexgraph.native_sparse import native_boundaries
    boundary = native_boundaries(rex)[0].apply(unit)
    expected = np.linalg.solve(unit + boundary.T @ boundary, unit)
    executor = Executor(sources={"r": rex})
    result = executor.execute(query(source("r"), call("GREEN_GRAM", cells, call("GREEN_OPERATOR", 1))))
    np.testing.assert_allclose(result.values[0], expected, atol=1e-10)
    value = run(rex, "GREEN_FIELD", cells[0]).values[0]
    np.testing.assert_allclose(value.values, expected[:, 0], atol=1e-10)
    q = 1 - expected[0, 1] ** 2 / (expected[0, 0] * expected[1, 1])
    assert run(rex, "GREEN_SPREAD", *cells[:2]).values[0] == pytest.approx(q)
    zero = Cochain(1, np.zeros(rex.nE), source=rex)
    assert run(rex, "GREEN_SPREAD", zero, zero).values == (None,)


@pytest.mark.parametrize("explain", [False, True])
def test_green_parameters_and_spaces_are_checked_before_execution(rex, explain):
    for name, args in [
        ("GREEN_OPERATOR", (True,)), ("GREEN_OPERATOR", (-1,)),
        ("GREEN_OPERATOR", (1, -1)), ("GREEN_OPERATOR", (1, 1, 0)),
        ("GREEN_FIELD", (Cell(rex, 1, 0), 1, 1e-10, 0)),
        ("GREEN_GRAM", ([],)),
        ("GREEN_GRAM", ([Cell(rex, 0, 0), Cell(rex, 1, 0)],)),
    ]:
        with pytest.raises((ValueError, TypeError)):
            run(rex, name, *args, explain=explain)


@pytest.mark.parametrize("backend", ["memory", "rex"])
def test_new_readings_execute_on_rcdb_records(rex, tmp_path, backend):
    rcdb = pytest.importorskip("rcdb")
    from rcql import parse
    store = rcdb.MemoryStore() if backend == "memory" else rcdb.open_store(f"rex://{tmp_path / 'store'}")
    try:
        store.put("branching", rex, analytics=False)
        result = Executor(sources={"db": store}).execute(parse(
            'FROM RCDB_GET($db,"branching") LET x=INDICATOR(CELL(1,0)) '
            'RETURN GRAM_RANK([x,x]), GRAM([x,x]), RATE(x,3), '
            'SUPPORT(CELL(1,0)), DEGREE(CELL(0,0)), GREEN_FIELD(x), '
            'CHAIN_VALID(), WEIGHT(CELL(1,0))'))
        assert result.values[0] == 1
        assert result.values[1].tolist() == [[Fraction(1), Fraction(1)], [Fraction(1), Fraction(1)]]
        assert result.values[2].values.tolist() == [Fraction(1, 3), 0, 0, 0]
        assert result.values[3].indices == (0, 1, 2, 3)
        assert result.values[4] == 2
        assert np.all(np.isfinite(result.values[5].values))
        assert result.values[6:] == (True, Fraction(1))
    finally:
        store.close()
