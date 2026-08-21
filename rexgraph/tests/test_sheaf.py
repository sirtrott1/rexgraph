"""Sheaf: stalks, restrictions, gluing and holonomy."""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.sheaf import Sheaf


def path4():
    return RexGraph(sources=np.array([0, 1, 2], np.int32),
                    targets=np.array([1, 2, 3], np.int32))


def cycle4():
    return RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                    targets=np.array([1, 2, 3, 0], np.int32))


def r(g):
    return (g["ratio"], g["H0"], g["H1"], g["gluable"], g["glued"])


#### meets
def test_meets_count_matches_spore_at_grade_1():
    assert len(Sheaf(path4()).meets()) == 2
    assert len(Sheaf(cycle4()).meets()) == 4
    both = RexGraph(sources=np.array([0, 1, 2, 4, 5, 6, 7], np.int32),
                    targets=np.array([1, 2, 3, 5, 6, 7, 4], np.int32))
    assert len(Sheaf(both).meets()) == 6


def test_a_pair_sharing_several_mediators_is_ONE_structural_edge():
    hx = RexGraph.from_hypergraph(np.array([0, 3, 6], np.int64),
                                  np.array([0, 1, 2, 0, 1, 3], np.int64))
    m = Sheaf(hx, stalk_dim=1).meets()
    assert len(m) == 1
    assert m[0][2] == [0, 1]                      # tested at BOTH mediators


def test_grade_selects_what_a_cell_is():
    rex = cycle4()
    assert Sheaf(rex, grade=1).n_cells == int(rex.nE)
    assert Sheaf(rex, grade=0).n_cells == int(rex.nV)
    with pytest.raises(ValueError):
        Sheaf(rex, grade=3)


def test_grade_0_mediates_through_the_edge_containing_both():
    sh = Sheaf(cycle4(), grade=0)
    assert len(sh.meets()) == 4                   # the 4 edges of the cycle
    for _a, _b, meds in sh.meets():
        assert len(meds) == 1


#### glue
def test_path_gluing_matches_spore():
    sh = Sheaf(path4(), stalk_dim=2)
    for e in range(3):
        sh.assign(e, [1, 0])
    assert r(sh.glue()) == (1.0, 1, 0, 2, 2)
    sh.assign(1, [5, 0])
    assert r(sh.glue()) == (0.0, 3, 2, 2, 0)


def test_cycle_gluing_matches_spore():
    sh = Sheaf(cycle4(), stalk_dim=2)
    for e in range(4):
        sh.assign(e, [1, 0])
    assert r(sh.glue()) == (1.0, 1, 0, 4, 4)
    sh.assign(0, [5, 0])
    assert r(sh.glue()) == (0.5, 2, 2, 4, 2)      # H0=2, NOT 3
    sh.assign(2, [5, 0])
    assert r(sh.glue()) == (0.0, 4, 4, 4, 0)
    for e in range(4):
        sh.assign(e, [5, 0])
    assert r(sh.glue()) == (1.0, 1, 0, 4, 4)      # uniform again: the value is irrelevant


#### restrictions on incidences
def test_a_per_cell_restriction_cannot_express_holonomy():
    sh = Sheaf(cycle4(), stalk_dim=2)
    for e in range(4):
        sh.assign(e, [1, 0])
    sh.restrict(0, np.diag([2.0, 2.0]))
    assert r(sh.glue()) == (0.5, 2, 2, 4, 2)
    sh.assign(0, [0.5, 0])
    assert r(sh.glue()) == (1.0, 1, 0, 4, 4)     # fully repaired: not an obstruction


def test_naming_a_mediator_distinguishes_the_ends():
    sh = Sheaf(cycle4(), stalk_dim=1)
    sh.restrict(0, [[2.0]], mediator=sh._inc[0][0])
    assert len(sh._R) == 1                        # one incidence, not the whole cell
    sh.restrict(1, [[2.0]])
    assert len(sh._R) == 1 + len(sh._inc[1])      # the cell form touches every incidence


def test_an_unassigned_sheaf_glues_because_nothing_disagrees():
    g = Sheaf(cycle4(), stalk_dim=2).glue()
    assert g["ratio"] == 1.0 and g["H1"] == 0 and g["gluable"] == 4


#### holonomy
def fan5_with_hyperface():
    from rexgraph.faces import auto_hyperface
    ptr, idx = [0], list(range(5))
    ptr.append(5)
    for i in range(1, 5):
        idx += [0, i]
        ptr.append(len(idx))
    rex = RexGraph.from_hypergraph(np.asarray(ptr, np.int64), np.asarray(idx, np.int64))
    auto_hyperface(rex)
    rex._ensure_clean()
    return rex


@pytest.mark.parametrize("theta,want", [
    ([0, 0, 0, 0, 0], 0.0),          # flat by construction
    ([1, 0, 0, 0, 0], 4.0),          # H alone: its face sign is k-1, NOT +/-1
    ([0, 1, 0, 0, 0], -1.0),         # one leg
    ([0, 1, 1, 1, 1], -4.0),         # every leg
    ([4, -1, -1, -1, -1], 20.0),     # the face column itself: <c, c>
])
def test_holonomy_matches_spore_on_a_hyperface(theta, want):
    sh = Sheaf(fan5_with_hyperface(), stalk_dim=2)
    assert float(sh.holonomy(theta)[0]) == pytest.approx(want)


def test_a_gradient_angle_field_is_flat_at_any_arity():
    sh = Sheaf(fan5_with_hyperface(), stalk_dim=2)
    theta = sh.gradient_angles([0, 1, 2, 3, 4])
    assert float(theta[0]) == pytest.approx(2.5)
    assert float(sh.holonomy(theta)[0]) == 0.0
    assert sh.is_flat(theta)


def test_holonomy_is_vacuous_without_faces():
    sh = Sheaf(cycle4(), stalk_dim=2)
    assert sh.holonomy(np.ones(4)).size == 0
    assert sh.is_flat(np.ones(4))


def test_binding_a_connection_puts_OPPOSITE_transports_at_the_two_ends():
    rex = cycle4()
    sh = Sheaf(rex, stalk_dim=2, grade=1)
    sh.bind_connection([0.7, 0.0, 0.0, 0.0])
    meds = sh._inc[0]
    a = sh._R[(0, meds[0])]
    b = sh._R[(0, meds[1])]
    assert not np.allclose(a, b)                       # the ends differ
    assert np.allclose(a @ b, np.eye(2), atol=1e-12)   # and are inverse: R(t) R(-t) = I


def test_a_d1_stalk_cannot_see_holonomy():
    sh = Sheaf(cycle4(), stalk_dim=1)
    sh.bind_connection([0.7, 0.0, 0.0, 0.0])
    R = sh._R[(0, sh._inc[0][0])]
    assert R.shape == (1, 1)
    assert float(R[0, 0]) == pytest.approx(np.cos(0.7))


#### sections
def test_sections_are_the_components_that_must_share_an_assignment():
    sh = Sheaf(path4(), stalk_dim=3)
    assert sh.sections() == [[0, 1, 2]]              # all glue: one free value
    sh.assign(1, [5, 0, 0])
    assert sh.sections() == [[0], [1], [2]]          # both meets fail: three
    assert len(sh.sections()) == sh.glue()["H0"]     # H0 counts exactly these


def test_an_indicator_restriction_keeps_only_the_labels_a_cell_admits():
    sh = Sheaf(path4(), stalk_dim=3)
    for e in range(3):
        sh.assign(e, [1, 1, 1])
    m = sh.mediators(0, 1)[0]
    sh.select(0, m, [1, 0, 0])
    sh.select(1, m, [0, 1, 0])
    assert not sh.admits(0, 1)                        # disjoint labels
    sh.select(1, m, [1, 1, 0])
    assert sh.admits(0, 1)                            # they overlap again


def test_one_cell_can_admit_DIFFERENT_labels_at_DIFFERENT_mediators():
    sh = Sheaf(path4(), stalk_dim=3)
    for e in range(3):
        sh.assign(e, [1, 1, 1])
    m01 = sh.mediators(0, 1)[0]
    m12 = sh.mediators(1, 2)[0]
    assert m01 != m12
    sh.select(1, m01, [1, 1, 0])
    sh.select(1, m12, [0, 0, 1])
    sh.select(0, m01, [1, 0, 0])
    sh.select(2, m12, [1, 0, 0])
    assert sh.admits(0, 1)                            # cell 1 admits label 0 here
    assert not sh.admits(1, 2)                        # and only label 2 there


def test_select_refuses_a_mask_of_the_wrong_width():
    sh = Sheaf(path4(), stalk_dim=3)
    with pytest.raises(ValueError, match="mask has"):
        sh.select(0, sh._inc[0][0], [1, 0])


#### grade
def two_triangles():
    from rexgraph.faces import autoface
    rex = RexGraph(sources=np.array([0, 1, 2, 1, 3], np.int32),
                   targets=np.array([1, 2, 0, 3, 2], np.int32))
    autoface(rex)
    rex._ensure_clean()
    return rex


def test_the_same_complex_offers_a_different_question_at_each_grade():
    rex = two_triangles()
    counts = {g: len(Sheaf(rex, stalk_dim=2, grade=g).meets()) for g in (0, 1, 2)}
    assert counts[0] == 5                        # one meet per edge
    assert counts[1] == 8
    assert counts[0] != counts[1]                # the point


def test_grades_0_and_1_can_coincide_and_that_is_the_duality_not_an_identity():
    from rexgraph.faces import auto_hyperface
    ptr, idx = [0], list(range(5))
    ptr.append(5)
    for i in range(1, 5):
        idx += [0, i]
        ptr.append(len(idx))
    fan = RexGraph.from_hypergraph(np.asarray(ptr, np.int64), np.asarray(idx, np.int64))
    auto_hyperface(fan)
    fan._ensure_clean()
    assert len(Sheaf(fan, grade=0).meets()) == len(Sheaf(fan, grade=1).meets()) == 10

    tri = two_triangles()
    assert len(Sheaf(tri, grade=0).meets()) != len(Sheaf(tri, grade=1).meets())


def test_a_grade_with_no_meets_glues_vacuously_and_says_so():
    sh = Sheaf(two_triangles(), stalk_dim=2, grade=2)
    g = sh.glue()
    assert g["gluable"] == 0 and g["ratio"] == 1.0 and g["H1"] == 0


def test_assign_is_grade_aware_which_sporeS_LANGUAGE_layer_is_not():
    rex = two_triangles()
    sh = Sheaf(rex, stalk_dim=2, grade=0)
    assert sh.n_cells == int(rex.nV)
    for c in range(sh.n_cells):
        sh.assign(c, [1.0, 0.0])
    sh.assign(0, [5.0, 0.0])
    g = sh.glue()
    assert g["H1"] > 0                            # the assign LANDED


#### boundary restrictions
def branching_with_legs():
    ptr = np.array([0, 4, 6, 8], np.int64)
    idx = np.array([0, 1, 2, 3, 0, 1, 0, 2], np.int64)
    rex = RexGraph.from_hypergraph(ptr, idx)
    rex._ensure_clean()
    return rex


def test_bind_boundary_reads_all_three_pillars_off_B1():
    sh = Sheaf(branching_with_legs(), stalk_dim=1)
    sh.bind_boundary()
    r = {k: float(v[0, 0]) for k, v in sh._R.items()}

    assert r[(0, 0)] == pytest.approx(-1.0)          # orientation: the head of the k=4
    for m in (1, 2, 3):
        assert r[(0, m)] == pytest.approx(1.0 / 3)   # share: 1/(k-1), k=4
    assert r[(1, 0)] == pytest.approx(-1.0)          # the leg's head
    assert r[(1, 1)] == pytest.approx(1.0)           # share: 1/(2-1)


def test_a_constant_section_glues_exactly_where_the_boundary_entries_agree():
    sh = Sheaf(branching_with_legs(), stalk_dim=1)
    sh.bind_boundary()
    for e in range(sh.n_cells):
        sh.assign(e, [1.0])
    g = sh.glue()
    assert (g["gluable"], g["glued"]) == (3, 1)
    assert set(g["failed"]) == {(0, 1), (0, 2)}      # branching against each leg


def test_equal_arity_relations_glue_under_a_constant_section():
    ptr = np.array([0, 2, 4], np.int64)
    idx = np.array([0, 1, 0, 2], np.int64)
    rex = RexGraph.from_hypergraph(ptr, idx)
    rex._ensure_clean()
    sh = Sheaf(rex, stalk_dim=1)
    sh.bind_boundary()
    for e in range(sh.n_cells):
        sh.assign(e, [1.0])
    assert sh.glue()["ratio"] == 1.0


def test_an_indicator_carries_existence_ONLY():
    rex = branching_with_legs()

    masked = Sheaf(rex, stalk_dim=1)
    for cell, meds in enumerate(masked._inc):
        for m in meds:
            masked.select(cell, m, [1.0])            # "present", nothing more
        masked.assign(cell, [1.0])
    assert masked.glue()["ratio"] == 1.0             # blind: everything agrees

    read = Sheaf(rex, stalk_dim=1)
    read.bind_boundary()
    for cell in range(read.n_cells):
        read.assign(cell, [1.0])
    assert read.glue()["ratio"] < 1.0                # the pillars separate them
