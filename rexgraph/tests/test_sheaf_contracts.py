"""Independent rectangular map, source state and incidence cost contracts."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.sheaf import ExactSheaf, Sheaf, UndeclaredRestrictionError


def branch(n=3):
    return RexGraph.from_hypergraph([0, n], list(range(n)))


def test_rectangular_stalks_glue_in_the_declared_mediator_space():
    sh = ExactSheaf(branch(), grade=0, stalk_dims=(2, 1, 3), mediator_dims=(2,))
    sh.assign(0, [Q(1, 3), 2])
    sh.assign(1, [Q(1, 3)])
    sh.assign(2, [Q(1, 3), 1, 1])
    sh.restrict(0, [[1, 0], [0, 1]], mediator=0)
    sh.restrict(1, [[1], [6]], mediator=0)
    sh.restrict(2, [[1, 0, 0], [0, 1, 1]], mediator=0)
    assert sh.glue().glued == 3
    check = sh.check_section()
    assert check.compatible
    assert (check.incidence_count, check.comparison_count) == (3, 2)
    sh.assign(2, [Q(1, 2), 1, 1])
    check = sh.check_section()
    assert not check.compatible
    assert [(x.left_cell, x.right_cell, x.residual) for x in check.obstructions] == [
        (0, 2, (Q(-1, 6), Q(0)))]
    assert sh.glue().obstruction_count == 2  # full pair diagnostics, not the anchor count


@pytest.mark.parametrize("grade", [0, 1])
@pytest.mark.parametrize("seed", range(12))
def test_random_rectangular_maps_against_independent_all_pair_oracle(grade, seed):
    rng = np.random.default_rng(seed)
    supports = [(0, 1, 2), (2, 0, 3), (0, 3), (1, 3, 4)]
    ptr = [0]
    for support in supports:
        ptr.append(ptr[-1] + len(support))
    rex = RexGraph.from_hypergraph(ptr, [v for s in supports for v in s])
    nc, nm = (5, 4) if grade == 0 else (4, 5)
    dims = tuple(int(v) for v in rng.integers(0, 4, nc))
    mdims = tuple(int(v) for v in rng.integers(0, 4, nm))
    sh = ExactSheaf(rex, grade=grade, stalk_dims=dims, mediator_dims=mdims,
                    require_declared_restrictions=True)
    inc = {c: ([m for m, support in enumerate(supports) if c in support]
               if grade == 0 else list(supports[c])) for c in range(nc)}
    transported = {}
    for c in range(nc):
        values = [Q(int(v), 3) for v in rng.integers(-2, 3, dims[c])]
        sh.assign(c, values)
        for m in inc[c]:
            matrix = [[Q(int(v), 2) for v in rng.integers(-2, 3, dims[c])]
                      for _ in range(mdims[m])]
            sh.restrict(c, matrix, mediator=m)
            transported[c, m] = tuple(sum((x*y for x, y in zip(row, values, strict=True)), Q(0))
                                       for row in matrix)
    expected = []
    gluable = glued = 0
    for a, b in combinations(range(nc), 2):
        shared = sorted(set(inc[a]) & set(inc[b]))
        if not shared:
            continue
        gluable += 1
        failed = False
        for m in shared:
            residual = tuple(x-y for x, y in zip(transported[a, m], transported[b, m], strict=True))
            if any(residual):
                expected.append((a, b, m, residual))
                failed = True
        glued += not failed
    full, compact = sh.glue(), sh.check_section()
    assert (full.gluable, full.glued) == (gluable, glued)
    assert [(o.left_cell, o.right_cell, o.mediator, o.residual) for o in full.obstructions] == expected
    assert compact.compatible == (not expected)
    assert compact.incidence_count == sum(map(len, inc.values()))
    assert compact.comparison_count == sum(max(0, sum(m in inc[c] for c in inc) - 1) for m in range(nm))


def test_mixed_dimensions_do_not_inherit_a_rectangular_identity():
    sh = ExactSheaf(branch(2), grade=0, stalk_dims=(2, 1), mediator_dims=(1,))
    for reader in (sh.glue, sh.check_section):
        with pytest.raises(UndeclaredRestrictionError) as error:
            reader()
        assert error.value.missing == ((0, 0),)
    with pytest.raises(ValueError, match="equal incidence dimensions"):
        sh.bind_boundary()
    assert sh._R == {}


def test_zero_dimensional_stalks_and_zero_maps_are_explicit():
    sh = ExactSheaf(branch(2), grade=0, stalk_dims=(0, 2), mediator_dims=(0,))
    sh.assign(0, [])
    sh.assign(1, [Q(1, 3), 7])
    sh.restrict(1, [], mediator=0)  # declared unique map Q^2 -> Q^0
    assert sh.check_section().compatible
    assert sh.glue().ratio == 1
    sh = ExactSheaf(branch(2), grade=0, stalk_dims=(0, 0), mediator_dims=(2,))
    sh.restrict(0, [[], []], mediator=0)
    sh.restrict(1, [[], []], mediator=0)
    assert sh.check_section().compatible


def test_cell_wide_restriction_is_atomic_across_different_mediator_dimensions():
    sh = ExactSheaf(branch(), stalk_dims=(2,), mediator_dims=(1, 2, 1))
    with pytest.raises(ValueError, match="2 by 2"):
        sh.restrict(0, [[1, 0]])
    assert sh._R == {}


@pytest.mark.parametrize("bad", [True, np.bool_(True), 1.2, "1"])
@pytest.mark.parametrize("argument", ["grade", "stalk_dim", "stalk_dims", "mediator_dims"])
def test_dimensions_are_integers_not_coerced_truncated_or_boolean(bad, argument):
    value = [bad] if argument == "stalk_dims" else [bad]*3 if argument == "mediator_dims" else bad
    with pytest.raises(TypeError, match="integer"):
        ExactSheaf(branch(), **{argument: value})


@pytest.mark.parametrize("bad", [1.0, True, np.bool_(True), complex(1), "1/3"])
def test_exact_stalk_and_map_coefficients_refuse_approximate_or_ambiguous_values(bad):
    sh = ExactSheaf(branch(2), grade=0)
    with pytest.raises(TypeError):
        sh.assign(0, [bad])
    with pytest.raises(TypeError):
        sh.restrict(0, [[bad]], mediator=0)


@pytest.mark.parametrize("reading", ["glue", "check_section", "bind_boundary"])
def test_source_basis_drift_is_rejected_before_stale_incidence_reading(reading):
    rex = branch()
    sh = ExactSheaf(rex)
    rex._boundary_idx[0], rex._boundary_idx[1] = rex._boundary_idx[1], rex._boundary_idx[0]
    with pytest.raises(ValueError, match="source incidence or basis changed"):
        getattr(sh, reading)()


def test_compact_check_is_incidence_sized_even_for_one_wide_relation(monkeypatch):
    n = 8192
    rex = branch(n)
    sh = ExactSheaf(rex, grade=0)
    for i in range(n):
        sh.assign(i, [Q(10**400, 3)])
    monkeypatch.setattr(sh, "meets", lambda: pytest.fail("pair graph constructed"))
    monkeypatch.setattr(np.linalg, "eigh", lambda *a, **k: pytest.fail("eigenbasis constructed"))
    calls = []
    transport = sh._transport
    def count(c, m):
        calls.append((c, m))
        return transport(c, m)
    monkeypatch.setattr(sh, "_transport", count)
    check = sh.check_section()
    assert check.compatible
    assert check.incidence_count == len(calls) == n
    assert check.comparison_count == n - 1
    assert rex.nE == 1  # no clique/star expansion of the primary relation


def test_detailed_glue_transports_each_incidence_only_once(monkeypatch):
    sh = ExactSheaf(branch(64), grade=0)
    calls = []
    transport = sh._transport
    def count(c, m):
        calls.append((c, m))
        return transport(c, m)
    monkeypatch.setattr(sh, "_transport", count)
    assert sh.glue().gluable == 64*63//2
    assert len(calls) == 64


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_numerical_stalks_and_restrictions_cannot_claim_gluing(bad):
    sh = Sheaf(branch(2), grade=0)
    with pytest.raises(ValueError, match="finite"):
        sh.assign(0, [bad])
    with pytest.raises(ValueError, match="finite"):
        sh.restrict(0, [[bad]])
    sh.stalks[0, 0] = bad  # public legacy storage can be edited directly
    for reader in (sh.glue, sh.sections):
        with pytest.raises(ValueError, match="finite"):
            reader()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_bad_numerical_tolerance_is_not_success(bad):
    sh = Sheaf(branch(2), grade=0)
    with pytest.raises((TypeError, ValueError), match="tolerance"):
        sh.glue(tol=bad)


def test_large_finite_numerical_disagreement_does_not_turn_inf_into_success():
    sh = Sheaf(branch(2), grade=0)
    sh.assign(0, [1e308])
    sh.assign(1, [-1e308])
    assert sh.glue()["glued"] == 0
    assert len(sh.sections()) == 2
    sh.assign(1, [1e308])
    assert sh.glue()["glued"] == 1


def test_boundary_maps_reuse_canonical_shares_and_witness_self_loop_rules():
    rex = RexGraph.from_hypergraph([0, 4, 5, 7], [0, 1, 2, 3, 2, 1, 1])
    sh = ExactSheaf(rex)
    sh.bind_boundary()
    assert sh._R[0, 1] == ((Q(1, 3),),)
    assert sh._R[1, 2] == ((Q(1),),)
    assert sh._R[2, 1] == ((Q(0),),)


@pytest.mark.parametrize("carrier", [Sheaf, ExactSheaf])
def test_grade_two_uses_the_carried_face_basis_not_discarded_face_records(carrier):
    rex = RexGraph.from_graph([0, 1, 2, 3, 0], [1, 2, 3, 0, 2])
    rex.add_faces([np.array([0, 1, 4], np.int32), np.array([2, 3, 4], np.int32)],
                  [np.array([1., 1., -1.]), np.array([1., 1., -1.])])
    rex._ensure_clean()
    assert rex.nF == 2 and rex.nF_hodge == 1
    sh = carrier(rex, grade=2)
    assert sh.n_cells == 1
    assert sh._inc == [[0, 1, 4]]
    sh.assign(0, [1])
    result = sh.glue()
    assert (result.h0 if carrier is ExactSheaf else result["H0"]) == 1


def test_agreement_connectivity_is_not_a_global_section_certificate():
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    sh = ExactSheaf(rex, grade=0)
    for cell in range(3):
        sh.assign(cell, [1])
    sh.restrict(0, [[2]], mediator=2)
    assert sh.glue().h0 == 1  # 0--1--2 agrees, but 0--2 does not
    assert sh.glue().obstruction_count == 1
    assert not sh.check_section().compatible


@pytest.mark.parametrize("carrier", [Sheaf, ExactSheaf])
def test_nonincident_restriction_cannot_be_silently_ignored(carrier):
    sh = carrier(RexGraph.from_graph([0, 1], [1, 2]))
    with pytest.raises(ValueError, match="not incident"):
        sh.restrict(0, [[1]], mediator=2)


@pytest.mark.parametrize("carrier", [Sheaf, ExactSheaf])
def test_boundary_binding_at_grade_two_is_not_a_misindexed_b1(carrier):
    sh = carrier(branch(), grade=2)
    with pytest.raises(ValueError, match="grades 0 and 1"):
        sh.bind_boundary()
