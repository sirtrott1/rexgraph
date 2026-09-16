"""Sparse overlapping measurements, not partitions or invented chain maps."""
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import diagonal_metric
from rexgraph.graph import RexGraph
from rexgraph.type_accession import (
    AccessionFamily,
    TypeAccession,
    TypedFamily,
    co_relate,
    moment_tensor,
)


def make():
    return RexGraph.from_graph([0, 1], [1, 2])


def co(rex, values, **kw):
    return Cochain(1, np.asarray(values), source=rex, **kw)


def maps(rex, **kw):
    return AccessionFamily((TypeAccession(rex, 1, "mixed", ((0, 0, 1), (0, 1, 1), (1, 1, 1)), **kw),
                            TypeAccession(rex, 1, "shared", ((0, 0, 1), (1, 0, 1)), **kw)))


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("block", [False, True])
def test_overlap_nonprojector_maps_and_signed_tensor_oracle(exact, block):
    rex = make()
    data = np.array([[1, 2], [2, -1]]) if block else np.array([1, 2])
    original = rex._boundary_idx.copy()
    family = maps(rex).apply(co(rex, data), exact=exact)
    metric = diagonal_metric(rex, 1, co(rex, [2, 3]))
    result = moment_tensor(family, metric, exact=exact)
    # Tiny dense reference matrices exist only in this test.
    p, q, m = np.array([[1, 1], [0, 1]]), np.array([[1, 0], [1, 0]]), np.diag([2, 3])
    expected_fields = [p @ data, q @ data]
    expected = np.array([[np.sum(u*(m @ v)) for v in expected_fields] for u in expected_fields])
    np.testing.assert_array_equal(result.values, expected)
    assert family.names == result.names == ("mixed", "shared")
    np.testing.assert_array_equal(family["mixed"].values, expected_fields[0])
    np.testing.assert_array_equal(rex._boundary_idx, original)
    assert not np.array_equal(p @ p, p)  # the view is not required to be idempotent
    assert np.any(p @ q)  # shared visibility is retained
    assert result.values[0, 1] == co_relate(family["mixed"], family["shared"], metric, exact=exact)
    if not block:
        assert result.values.tolist() == [[30, 12], [12, 5]]
    if exact:
        assert all(isinstance(v, Q) for v in result.values.flat)


def test_cross_moment_preserves_orientation_and_hermitian_order():
    rex = make()
    a = TypeAccession(rex, 1, "first", ((0, 0, 1), (1, 1, 1)))
    b = TypeAccession(rex, 1, "second", ((0, 0, -1), (1, 1, -1)))
    family = AccessionFamily((a, b)).apply(co(rex, [1, 2]), exact=True)
    assert moment_tensor(family, exact=True).values.tolist() == [[5, -5], [-5, 5]]
    u, v = a.apply(co(rex, [1j, 1])), b.apply(co(rex, [-1, -1j]))
    metric = diagonal_metric(rex, 1, co(rex, [2, 3]))
    assert co_relate(u, v, metric) == 1j
    assert co_relate(v, u, metric) == -1j
    assert moment_tensor(TypedFamily((u, v)), metric).values.tolist() == [[5, 1j], [-1j, 5]]


def test_exact_sparse_coefficients_are_copied_coalesced_and_canonically_hashed():
    rex = make()
    entries = [[0, 1, Q(1, 3)], [1, 0, 2], [0, 1, Q(1, 6)]]
    a = TypeAccession(rex, 1, "a", entries)
    entries[0][2] = 99
    b = TypeAccession(rex, 1, "different display label", ((1, 0, 2), (0, 1, Q(1, 2))))
    assert a.coefficient_digest == b.coefficient_digest
    assert a.entries == ((0, 1, Q(1, 2)), (1, 0, Q(2)))
    assert a.apply(co(rex, [1, 3]), exact=True).values.tolist() == [Q(3, 2), Q(2)]


@pytest.mark.parametrize("entries", [((True, 0, 1),), ((-1, 0, 1),), ((2, 0, 1),),
    ((0, 2, 1),), ((.5, 0, 1),), ((0, 0, True),), ((0, 0, 1j),),
    ((0, 0, float("nan")),), ((0, 0, float("inf")),), ((0, 0, 1e308), (0, 0, 1e308))])
def test_invalid_sparse_maps_refuse(entries):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        TypeAccession(make(), 1, "bad", entries)


@pytest.mark.parametrize("weights", [Q(10**400), Q(1, 10**400)])
def test_extreme_exact_accession_is_not_cast_through_double(weights):
    rex = make()
    a = TypeAccession(rex, 1, "extreme", ((1, 0, weights),))
    assert a.apply(co(rex, [2, 0]), exact=True).values.tolist() == [Q(0), 2*weights]
    with pytest.raises(FloatingPointError, match="representable"):
        a.apply(co(rex, [2, 0]))


def test_zero_float_map_does_not_acquire_an_exact_certificate():
    rex = make()
    a = TypeAccession(rex, 1, "cancelled", ((0, 0, 1.), (0, 0, -1.)))
    assert a.entries == () and not a.exact
    with pytest.raises(TypeError, match="integer/rational accession"):
        a.apply(co(rex, [1, 2]), exact=True)
    np.testing.assert_array_equal(a.apply(co(rex, [1, 2])).values, 0.)


def test_empty_cell_space_is_not_an_empty_type_family():
    rex = RexGraph.from_graph([], [])
    family = AccessionFamily((TypeAccession(rex, 1, "a", ()), TypeAccession(rex, 1, "b", ())))
    for exact in (False, True):
        views = family.apply(co(rex, np.empty((0, 2), dtype=int)), exact=exact)
        tensor = moment_tensor(views, exact=exact)
        assert tensor.values.shape == (2, 2)
        assert tensor.values.tolist() == [[0, 0], [0, 0]]
    with pytest.raises(TypeError, match="one or more"):
        AccessionFamily(())


def test_source_grade_basis_variance_and_family_names_are_checked():
    rex, other = make(), make()
    a = maps(rex).accessions[0]
    for value in (co(other, [1, 2]), co(rex, [1, 2], cell_keys=("b", "a")),
                  Cochain(0, np.ones(3), source=rex), co(rex, np.ones((2, 1, 1)))):
        with pytest.raises(ValueError):
            a.apply(value)
    with pytest.raises(ValueError, match="unique"):
        AccessionFamily((a, a))
    with pytest.raises(ValueError, match="same source"):
        AccessionFamily((a, maps(other).accessions[1]))
    b = maps(rex).accessions[1]
    u, v = a.apply(co(rex, [1, 2])), b.apply(Chain(1, np.array([1, 2]), source=rex))
    with pytest.raises(ValueError, match="variance"):
        co_relate(u, v)
    with pytest.raises(ValueError, match="variance"):
        TypedFamily((u, v))


def test_population_change_invalidates_preexisting_map_and_views():
    rex = make()
    family = maps(rex)
    values = co(rex, [1, 2])
    views = family.apply(values, exact=True)
    rex.add_edges(np.array([2], np.int32), np.array([0], np.int32))
    with pytest.raises(ValueError, match="population changed"):
        family.apply(values)
    with pytest.raises(ValueError, match="population changed"):
        moment_tensor(views, exact=True)


def test_large_sparse_map_never_assembles_an_ambient_dense_matrix(monkeypatch):
    n = 4096
    rex = RexGraph.from_graph(np.zeros(n, dtype=int), np.arange(1, n+1))
    a = TypeAccession(rex, 1, "shift", tuple((i, (i+1) % n, Q(1, 3)) for i in range(n)))
    x = co(rex, np.arange(n, dtype=int))
    def forbidden(*args, **kw):
        pytest.fail("accession action requested dense/eigen materialization")
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(np, "diag", forbidden)
    exact = a.apply(x, exact=True)
    assert len(a.entries) == n
    assert exact.values[0] == Q(1, 3) and exact.values[-1] == 0
    np.testing.assert_allclose(a.apply(x).values, np.array(exact.values, float))


def test_requested_tensor_checks_type_axes_and_copies_values():
    from rexgraph.type_accession import TypedMomentTensor
    rex = make()
    family = maps(rex).apply(co(rex, [1, 2]), exact=True)
    metric = diagonal_metric(rex, 1)
    data = np.array([[13., 5.], [5., 2.]])
    result = TypedMomentTensor(data, family, metric)
    data[0, 0] = 100
    assert result.values[0, 0] == 13 and not result.values.flags.writeable
    with pytest.raises(ValueError, match="ordered type family"):
        TypedMomentTensor(np.zeros((3, 3)), family, metric)


@pytest.mark.parametrize("value", [1., Q(1, 3), Q(3)])
def test_branching_source_primary_incidence_is_not_a_type_map(value):
    rex = RexGraph(boundary_ptr=np.array([0, 3, 5], np.int32),
        boundary_idx=np.array([0, 2, 1, 1, 0], np.int32), w_E=np.array([2., -3.]))
    ptr, idx = rex._boundary_ptr.copy(), rex._boundary_idx.copy()
    a = TypeAccession(rex, 1, "measure", ((0, 1, value),))
    x = co(rex, [2, -3])
    result = a.apply(x, exact=isinstance(value, Q))
    assert result.values[0] == -3*value
    np.testing.assert_array_equal(rex._boundary_ptr, ptr)
    np.testing.assert_array_equal(rex._boundary_idx, idx)
