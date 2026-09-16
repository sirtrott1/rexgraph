"""Exact chain map proofs against independent small dense rational oracles."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graph import RexGraph
from rexgraph.type_accession import CoordinateSpace, TypeAccession


def space(name, n):
    return CoordinateSpace(name, tuple(str(i) for i in range(n)))


def identity(c, factor=1):
    return GradedMap(c, c, tuple(tuple((i, i, factor) for i in range(n)) for n in c.sizes))


def dense(entries, shape):
    result = np.full(shape, Q(0), object)
    for i, j, v in entries:
        result[i, j] += v
    return result


def quotient():
    rex = RexGraph.from_graph([0, 1], [1, 2])
    c = CoordinateComplex.from_rex(rex)
    d = CoordinateComplex((space("quotient-vertices", 2), space("quotient-relations", 1)),
                          (((0, 0, -1), (1, 0, 1)),))
    p = GradedMap(c, d, (((0, 0, 1), (0, 1, 1), (1, 2, 1)), ((0, 1, 1),)))
    return rex, p


@pytest.mark.parametrize("fixture", ["path", "branch", "witness", "self-loop", "empty", "grade3", "empty-middle"])
def test_complete_identity_towers_are_exact(fixture):
    if fixture == "grade3":
        rex = RexGraph.from_cells(solid_octahedron_3rex())
    elif fixture == "empty-middle":
        import scipy.sparse as sp
        rex = RexGraph.from_cells([1, []])
        # Explicit supported stored tower: B2 (0,0), B3 (0,1).
        # from_cells' support list importer intentionally refuses an empty cell.
        rex._graded_duals = [sp.csr_matrix((0, 1))]
    elif fixture == "empty":
        rex = RexGraph.from_graph([], [])
    else:
        cells = {"path": [3, [[0, 1], [1, 2]]], "branch": [4, [[0, 1, 2, 3]]],
                 "witness": [2, [[0], [0, 1]]], "self-loop": [1, [[0, 0]]]}[fixture]
        rex = RexGraph.from_cells(cells)
    c = CoordinateComplex.from_rex(rex)
    result = identity(c).verify()
    assert result.source_residual == result.target_residual == 0
    assert result.commutation_residuals == (Q(0),) * (len(c.sizes)-1)
    assert all(isinstance(v, Q) for entries in c.boundaries for _, _, v in entries)
    if fixture == "branch":
        assert dict((i, v) for i, _, v in c.boundaries[0]) == {0: -1, 1: Q(1, 3), 2: Q(1, 3), 3: Q(1, 3)}
    if fixture == "self-loop":
        assert c.boundaries == ((),) and c.sizes == (1, 1)
    if fixture == "empty-middle":
        assert c.sizes == (1, 0, 0, 1)


def test_rectangular_quotient_has_verified_square_without_injection():
    _, p = quotient()
    certificate = p.verify()
    p0, p1 = (dense(e, s) for e, s in zip(p.components, p.shapes, strict=True))
    b = dense(p.domain.boundaries[0], (3, 2))
    d = dense(p.codomain.boundaries[0], (2, 1))
    assert (p0 @ b).tolist() == (d @ p1).tolist() == [[0, -1], [0, 1]]
    assert certificate.commutation_residuals == (0,)
    assert p.shapes == ((2, 3), (1, 2))


@pytest.mark.parametrize("which", ["source", "target"])
def test_zero_map_does_not_certify_a_noncomplex(which):
    spaces = tuple(space(f"grade{k}", 1) for k in range(3))
    good = CoordinateComplex(spaces, ((), ()))
    bad = CoordinateComplex(spaces, (((0, 0, Q(1, 3)),), ((0, 0, 2),)))
    # Every square of the zero attribution commutes, but B1 B2 = 2/3.
    c, d = (bad, good) if which == "source" else (good, bad)
    with pytest.raises(ValueError, match=f"{which} residual 2/3"):
        GradedMap(c, d, ((), (), ())).verify()


@pytest.mark.parametrize("grade", [1, 2, 3])
def test_every_square_is_checked_not_just_c1(grade):
    c = CoordinateComplex.from_rex(RexGraph.from_cells(solid_octahedron_3rex()))
    p = identity(c)
    components = list(p.components)
    components[grade] = tuple((i, j, 2*v) for i, j, v in components[grade])
    with pytest.raises(ValueError, match=f"square failed at grade {grade}"):
        replace(p, components=components).verify()


@pytest.mark.parametrize("factor", [Q(2, 3), Q(-7, 11), Q(10**400), Q(1, 10**400), Q(0)])
def test_signed_extreme_rationals_never_enter_a_float_carrier(factor):
    _, p = quotient()
    after = identity(p.codomain, factor)
    composed = p.verify().then(after.verify())
    expected = tuple(tuple((i, j, factor*v) for i, j, v in e if factor*v) for e in p.components)
    assert composed.declaration.components == expected
    assert composed.commutation_residuals == (0,)


def test_general_sparse_composition_order_associativity_and_cancellation():
    rng = np.random.default_rng(904)
    spaces = [CoordinateComplex((space(str(k), n),), ()) for k, n in enumerate((4, 3, 5, 2))]
    arrays = [np.array([[Q(int(x), 3) for x in row] for row in rng.integers(-2, 3, (m, n))], object)
              for n, m in ((4, 3), (3, 5), (5, 2))]
    maps = [GradedMap(a, b, (tuple((i, j, v) for (i, j), v in np.ndenumerate(M)),))
            for a, b, M in zip(spaces[:-1], spaces[1:], arrays, strict=True)]
    p, q, r = maps
    left = p.then(q).then(r)
    right = p.then(q.then(r))
    assert left.components == right.components
    assert dense(left.components[0], left.shapes[0]).tolist() == (arrays[2] @ arrays[1] @ arrays[0]).tolist()
    c = CoordinateComplex((space("cancel", 2),), ())
    a = GradedMap(c, c, (((0, 0, 1), (1, 0, 1)),))
    b = GradedMap(c, c, (((0, 0, 1), (0, 1, -1)),))
    assert a.then(b).components == ((),)
    assert b.then(a).components != ((),)


def test_composition_refuses_equal_dimensions_or_even_reconstructed_equal_complex():
    _, p = quotient()
    other = CoordinateComplex(p.codomain.spaces, p.codomain.boundaries)
    assert other.coefficient_digest == p.codomain.coefficient_digest
    with pytest.raises(ValueError, match="identical declared middle"):
        p.then(identity(other))
    with pytest.raises(TypeError, match="another ChainMap"):
        p.verify().then(identity(p.codomain))


def test_accessions_require_explicit_target_not_an_induced_inverse():
    rex, p = quotient()
    accessions = tuple(TypeAccession(rex, k, "quotient", e, coordinates=p.codomain.spaces[k])
                       for k, e in enumerate(p.components))
    assembled = GradedMap.from_accessions(accessions, p.codomain)
    assert assembled.verify().commutation_residuals == (0,)
    assert assembled.components == p.components
    assert accessions[1].shape == (1, 2)


@pytest.mark.parametrize("defect", ["basis", "name", "grade", "source", "coordinates", "order", "float-zero", "missing", "target-grades"])
def test_accession_tower_mismatches_are_refused(defect):
    rex, p = quotient()
    accessions = [TypeAccession(rex, k, "q", e, coordinates=p.codomain.spaces[k]) for k, e in enumerate(p.components)]
    target = p.codomain
    a = accessions[1]
    if defect == "basis":
        accessions[1] = replace(a, cell_keys=("0", "1"))
    elif defect == "name":
        accessions[1] = replace(a, name="other")
    elif defect == "grade":
        accessions = accessions[::-1]
    elif defect == "source":
        accessions[1] = replace(a, source=RexGraph.from_graph([0, 1], [1, 2]))
    elif defect in {"coordinates", "order"}:
        accessions[0] = replace(accessions[0], coordinates=CoordinateSpace(
            "other" if defect == "coordinates" else target.spaces[0].name, ("1", "0")))
    elif defect == "float-zero":
        accessions[1] = replace(a, entries=((0, 0, 0.0),))
    elif defect == "missing":
        accessions.pop()
    else:
        target = CoordinateComplex(target.spaces[:1], ())
    with pytest.raises((ValueError, TypeError)):
        GradedMap.from_accessions(accessions, target)


@pytest.mark.parametrize("value", [1.0, 0.0, Q(1, 2)+0.0, True, 1j, float("nan"), float("inf")])
def test_exact_declarations_reject_floats_even_if_zero_or_integral(value):
    c = CoordinateComplex((space("s", 1), space("t", 1)), ((),))
    with pytest.raises((ValueError, TypeError)):
        GradedMap(c, c, (((0, 0, value),), ()))
    with pytest.raises((ValueError, TypeError)):
        CoordinateComplex(c.spaces, (((0, 0, value),),))


@pytest.mark.parametrize("defect", ["missing-boundary", "missing-component", "index", "negative-index", "grade-range"])
def test_malformed_axes_refuse(defect):
    _, p = quotient()
    with pytest.raises((ValueError, TypeError)):
        if defect == "missing-boundary":
            CoordinateComplex(p.domain.spaces, ())
        elif defect == "missing-component":
            replace(p, components=(p.components[0],))
        elif defect == "grade-range":
            replace(p, codomain=CoordinateComplex(p.codomain.spaces[:1], ()))
        else:
            replace(p, components=(((2 if defect == "index" else -1, 0, 1),), p.components[1]))


def test_inputs_are_copied_and_coalesced_into_immutable_declarations():
    c = CoordinateComplex((space("s", 1),), ())
    entries = [[(0, 0, 3), (0, 0, Q(-5, 2))]]
    p = GradedMap(c, c, entries)
    entries[0].clear()
    assert p.components == (((0, 0, Q(1, 2)),),)
    assert p.coefficient_digest == GradedMap(c, c, (((0, 0, Q(1, 2)),),)).coefficient_digest
    with pytest.raises(FrozenInstanceError):
        p.components = ()


@pytest.mark.parametrize("endpoint", ["domain", "codomain"])
def test_same_population_boundary_change_invalidates_certificate(endpoint):
    rex = RexGraph.from_graph([0], [1])
    live = CoordinateComplex.from_rex(rex)
    fixed = CoordinateComplex(live.spaces, live.boundaries)
    p = GradedMap(live, fixed, identity(live).components) if endpoint == "domain" else GradedMap(fixed, live, identity(live).components)
    result = p.verify()
    rex.remove_edges([1])
    rex._ensure_clean()
    rex.add_edges([1], [0])
    rex._ensure_clean()
    assert (rex.nV, rex.nE) == live.sizes
    with pytest.raises(ValueError, match="boundary state changed"):
        result.check_state()
    with pytest.raises(ValueError, match="boundary state changed"):
        p.verify()


def test_integral_higher_boundary_contract_is_not_rationalized(monkeypatch):
    import rexgraph.chain_map as cm
    from rexgraph.core import _sparse
    from rexgraph.native_sparse import NativeSparse
    rex = RexGraph.from_graph([0], [1])
    original = cm.raw_boundary_carriers
    fractional = NativeSparse(_sparse.dual_from_coo([0], [0], [0.5], 1, 1))
    monkeypatch.setattr(cm, "raw_boundary_carriers", lambda r: [*original(r), fractional])
    with pytest.raises(ValueError, match="integral stored higher"):
        CoordinateComplex.from_rex(rex)


def test_primary_boundary_disagreement_is_not_certified(monkeypatch):
    rex = RexGraph.from_graph([0], [1])
    monkeypatch.setattr(rex, "relation_supports", lambda: [[1, 0]])
    with pytest.raises(ValueError, match="primary incidence and stored B1 disagree"):
        CoordinateComplex.from_rex(rex)


def test_equal_parallel_columns_do_not_hide_changed_relation_identities():
    rex = RexGraph.from_cells([2, [[0, 1], [0, 1]]], relation_ids=np.array([10, 20]))
    c = CoordinateComplex.from_rex(rex)
    proof = identity(c).verify()
    rex.relation_ids[:] = [20, 10]
    assert CoordinateComplex.from_rex(rex).boundaries == c.boundaries
    with pytest.raises(ValueError, match="boundary state changed"):
        proof.check_state()


def test_large_sparse_composition_and_verification_forbid_dense_or_spectral_paths(monkeypatch):
    import scipy.linalg as la
    import scipy.sparse as sp
    n = 4096
    c = CoordinateComplex.from_rex(RexGraph.from_graph(np.arange(n), np.arange(1, n+1)))
    def forbidden(*a, **kw):
        pytest.fail("dense/oracle/solver path was called")
    for cls in (sp.csr_matrix, sp.csc_matrix):
        monkeypatch.setattr(cls, "toarray", forbidden)
        monkeypatch.setattr(cls, "todense", forbidden)
    for module in (np.linalg, la):
        for name in ("eigh", "eigvalsh", "svd", "pinv", "solve"):
            monkeypatch.setattr(module, name, forbidden)
    p = identity(c)
    q = identity(c, Q(-2, 3))
    result = p.verify().then(q.verify())
    assert sum(map(len, result.declaration.components)) == 2*n+1
    assert result.commutation_residuals == (0,)
