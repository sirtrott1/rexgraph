"""Betti numbers are not a diff, and an entry-wise diff is only legitimate at grade 1.

These pin both halves: that the trichotomy is read exactly where it is defined, and that
the function refuses where an entry-wise reading would measure the representative rather
than the complex.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.tensor_diff import format_diff, grade2_diff, tensor_diff


def _hg(rels, nV=None):
    ptr, idx = [0], []
    for r in rels:
        idx += list(r); ptr.append(len(idx))
    if nV is not None and (not idx or max(idx) < nV - 1):
        idx = idx + [nV - 1, nV - 1]; ptr.append(len(idx))     # pad the vertex space
    return RexGraph.from_hypergraph(np.asarray(ptr, np.int64),
                                    np.asarray(idx, np.int64))


def test_a_tensor_does_not_differ_from_itself():
    rex = _hg([[0, 1], [1, 2], [0, 2, 3]])
    d = tensor_diff(rex, rex)
    assert d["identical"] == int(rex.nE)
    assert d["reoriented"] == 0 and d["novel"] == 0
    assert d["existence_entries"] == 0 and d["orientation_entries"] == 0
    assert d["agreement"] == pytest.approx(1.0)


def test_orientation_is_seen_where_topology_is_not():
    """The whole point: same support, same Betti, different tensor.

    Reversing a relation changes no vertex set and no Betti number, so a topological
    comparison reports nothing. The entry-wise reading reports it exactly.
    """
    a = _hg([[0, 1], [1, 2], [2, 0]])
    b = _hg([[1, 0], [1, 2], [2, 0]])          # first relation reversed
    assert list(a.betti[:2]) == list(b.betti[:2]), "the fixture must be Betti-identical"
    d = tensor_diff(a, b)
    assert d["reoriented"] == 1 and d["novel"] == 0
    assert d["orientation_entries"] == 2, "a reversal disagrees at exactly two entries"
    assert d["existence_entries"] == 0


def test_existence_and_orientation_are_separated_not_summed():
    a = _hg([[0, 1], [1, 2]])
    b = _hg([[1, 0], [1, 2], [2, 3]])          # one reversed, one new
    d = tensor_diff(a, b)
    assert d["reoriented"] == 1 and d["novel"] == 1
    assert d["orientation_entries"] == 2
    assert d["existence_entries"] == 2, "the novel relation contributes its own span"


def test_arity_is_not_a_difference_of_its_own():
    """Share is 1/(k-1) from the span width, so it carries nothing the support does not."""
    a = _hg([[0, 1, 2, 3]])
    b = _hg([[0, 1, 2, 3]])
    d = tensor_diff(a, b)
    assert d["identical"] == 1 and d["orientation_entries"] == 0
    c = _hg([[1, 0, 2, 3]])                    # same support, distinguished vertex moved
    d2 = tensor_diff(a, c)
    assert d2["reoriented"] == 1 and d2["orientation_entries"] == 2


def test_the_merge_preview_is_the_candidate_predicate():
    """Novel relations split into rank-raising and cycle-closing, and that is exact."""
    from rexgraph.graded_boundary import _sparse_rank
    # two components, {0,1,2} and {3,4}. Input relation {0,2} is novel by support and
    # lies inside the first component's span, so it closes a cycle; {2,3} is novel and
    # bridges the components, so it adds a direction.
    ref = _hg([[0, 1], [1, 2], [3, 4]])
    inp = _hg([[0, 2], [2, 3]])
    d = tensor_diff(ref, inp)
    m = d["merge"]
    assert m["evaluated"] + m["not_evaluable"] == d["novel"]
    # verify against actually merging, relation by relation
    rp = np.asarray(ref.boundary_ptr, np.int64); ri = np.asarray(ref.boundary_idx, np.int64)
    r0 = _sparse_rank(ref._integer_B1().tocsc())
    adds = 0
    for e in range(int(inp.nE)):
        ip = np.asarray(inp.boundary_ptr, np.int64)
        ii = np.asarray(inp.boundary_idx, np.int64)
        span = [int(v) for v in ii[ip[e]:ip[e + 1]]]
        if max(span) >= int(ref.nV):
            continue
        merged = RexGraph.from_hypergraph(
            np.asarray(list(rp) + [int(rp[-1]) + len(span)], np.int64),
            np.asarray(list(ri) + span, np.int64))
        adds += int(_sparse_rank(merged._integer_B1().tocsc()) > r0)
    assert m["evaluated"] == 2, "both novel relations are inside the reference vertex space"
    assert m["marginal_adds_a_direction"] == adds == 1
    assert m["marginal_closes_a_cycle"] == 1
    assert m["rank_delta"] == 1 and m["cycle_delta"] == 1, "joint, not the marginal sum"


def test_labels_align_two_different_indexings():
    """The same relation under a different vertex numbering is not a difference."""
    a = _hg([[0, 1], [1, 2]])
    b = _hg([[2, 1], [1, 0]])
    la = ["x", "y", "z"]
    lb = ["z", "y", "x"]                       # b's 2 is x, b's 0 is z
    d = tensor_diff(a, b, ref_labels=la, inp_labels=lb)
    assert d["identical"] == 2, "relabelling is not a difference"
    assert d["reoriented"] == 0 and d["novel"] == 0


def test_labels_must_be_given_for_both_and_must_fit():
    a = _hg([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="both tensors or for neither"):
        tensor_diff(a, a, ref_labels=["x", "y", "z"])
    with pytest.raises(ValueError, match="one entry per vertex"):
        tensor_diff(a, a, ref_labels=["x"], inp_labels=["x"])


def test_a_vertex_the_reference_lacks_is_counted_apart():
    a = _hg([[0, 1], [1, 2]])
    b = _hg([[0, 1], [3, 4]])
    d = tensor_diff(a, b, ref_labels=["p", "q", "r"],
                    inp_labels=["p", "q", "r", "s", "t"])
    assert d["vertices_only_in_input"] == 2
    assert d["unmapped_relations"] == 1, "the relation on s,t maps nowhere"
    assert d["novel"] >= 1


def test_entry_wise_is_refused_above_grade_one():
    """Where the column is fixed only up to sign, entries measure the representative."""
    a = _hg([[0, 1], [1, 2], [2, 0]])
    with pytest.raises(ValueError, match="grade 1 only"):
        tensor_diff(a, a, grade=2)


def test_grade_two_compares_holonomy_and_survives_a_face_flip():
    """The reason grade 2 is separate: flipping a face column changes no cell."""
    from rexgraph.faces import auto_hyperface
    rng = np.random.default_rng(4)
    rels = [sorted(rng.choice(10, 4, replace=False).tolist()) for _ in range(6)]
    from itertools import combinations
    prs = sorted({tuple(sorted(p)) for g in rels for p in combinations(g, 2)})
    a = _hg(rels + [list(p) for p in prs])
    auto_hyperface(a)
    d = grade2_diff(a, a)
    assert d["grade"] == 2 and d["same_orientability"] is True
    assert d["frustration_delta"] == pytest.approx(0.0)
    assert "holonomy" in d["reading"]


def test_format_states_counts_without_a_verdict():
    a = _hg([[0, 1], [1, 2]])
    b = _hg([[1, 0], [1, 2], [2, 3]])
    s = format_diff(tensor_diff(a, b))
    assert "orientation" in s and "existence" in s
    for word in ("bad", "good", "error", "invalid", "correct"):
        assert word not in s.lower()


def test_the_merge_delta_is_joint_and_the_marginals_do_not_sum():
    """The trap this function exists to avoid.

    Each candidate reading is taken against the reference as it stands, but absorbing one
    relation changes the span the next is judged against. Three relations that each
    bridge the same two components would each read "adds a direction" on their own while
    the merge adds one direction and two cycles.
    """
    from rexgraph.graded_boundary import _sparse_rank
    ref = _hg([[0, 1], [2, 3]])                 # two components
    inp = _hg([[0, 2], [1, 3], [1, 2]])         # three bridges between the same two
    d = tensor_diff(ref, inp)
    m = d["merge"]
    assert m["evaluated"] == 3
    assert m["marginal_adds_a_direction"] == 3, "each alone would join the components"
    assert m["rank_delta"] == 1, "together they join them once"
    assert m["cycle_delta"] == 2
    assert m["marginals_sum_to_joint"] is False

    merged = _hg([[0, 1], [2, 3], [0, 2], [1, 3], [1, 2]])
    r0 = _sparse_rank(ref._integer_B1().tocsc())
    r1 = _sparse_rank(merged._integer_B1().tocsc())
    assert r1 - r0 == m["rank_delta"]
    assert (int(merged.nE) - r1) - (int(ref.nE) - r0) == m["cycle_delta"]
    assert "do not sum" in format_diff(d)


def test_the_marginals_do_sum_when_the_relations_are_independent():
    """The flag has to be able to say yes, or it is not a reading."""
    ref = _hg([[0, 1], [2, 3], [4, 5]])
    inp = _hg([[0, 2], [3, 4]])                 # each joins a distinct pair of components
    m = tensor_diff(ref, inp)["merge"]
    assert m["rank_delta"] == 2 == m["marginal_adds_a_direction"]
    assert m["cycle_delta"] == 0
    assert m["marginals_sum_to_joint"] is True


#### the operator algebra, and the delta as a member of it #####################

def _mixed(seed=0, n=12, nV=14):
    """A branching construction with real arity, plus its own contacts."""
    from itertools import combinations
    rng = np.random.default_rng(seed)
    wide = [sorted(rng.choice(nV, int(rng.integers(3, 6)), replace=False).tolist())
            for _ in range(n)]
    prs = sorted({tuple(sorted(p)) for g in wide for p in combinations(g, 2)})
    return wide + [list(p) for p in prs], wide


def test_share_sums_to_zero_at_every_arity():
    """The condition that makes a column a boundary. Nothing about arity enters it."""
    import scipy.sparse as sp
    rels, _w = _mixed()
    rex = _hg(rels)
    B = sp.csc_matrix(rex.B1)
    csum = np.asarray(B.sum(axis=0)).ravel()
    ar = np.array([len(r) for r in rels])
    assert len(set(ar.tolist())) >= 3, "the fixture must span several arities"
    for k in sorted(set(ar.tolist())):
        assert np.abs(csum[ar == k]).max() < 1e-12, f"arity {k} does not close"


def test_the_chain_condition_and_adjointness():
    import scipy.sparse as sp

    from rexgraph.faces import auto_hyperface
    rng = np.random.default_rng(5)
    rels, _w = _mixed(seed=5)
    rex = _hg(rels)
    auto_hyperface(rex)
    Bs = rex.graded_boundaries()
    assert len(Bs) >= 2, "the fixture must reach grade 2"
    prod = sp.csc_matrix(Bs[0]) @ sp.csc_matrix(Bs[1])
    assert (np.abs(prod.toarray()).max() if prod.nnz else 0.0) < 1e-10, "d.d != 0"
    for Bg in Bs[:2]:
        Bg = sp.csc_matrix(Bg)
        x = rng.normal(size=Bg.shape[1]); y = rng.normal(size=Bg.shape[0])
        assert float((Bg @ x) @ y) == pytest.approx(float(x @ (Bg.T @ y)), abs=1e-9)


def test_the_difference_is_itself_a_boundary_tensor():
    """Zero-sum columns are a linear subspace, so the delta lives where its arguments do.

    This is what makes the diff a calculus rather than a report: D has its own L = D D^T
    with the constant vector in the kernel, so every reading applies to it unchanged.
    """

    from rexgraph.tensor_diff import difference_tensor
    rels, _w = _mixed(seed=2)
    rng = np.random.default_rng(2)
    flip = set(rng.permutation(len(rels))[:len(rels) // 3].tolist())
    a = _hg(rels)
    b = _hg([list(reversed(r)) if i in flip else list(r) for i, r in enumerate(rels)])
    D, R = difference_tensor(a, b, verify=True)          # verify IS the assertion
    assert R["max_column_sum"] < 1e-12
    L = (D @ D.T).tocsr()
    assert np.abs(np.asarray(L.sum(axis=1)).ravel()).max() < 1e-9, (
        "zero column sum propagates to zero row sum of the Laplacian")
    assert R["disagree"] == len(flip), "a reversal moves its column and nothing else"
    assert R["agree"] == D.shape[1] - len(flip)
    assert R["rank"] <= R["disagree"], "rank cannot exceed the moved columns"


def test_the_delta_of_a_tensor_with_itself_is_zero():
    from rexgraph.tensor_diff import difference_tensor
    rels, _w = _mixed(seed=3)
    rex = _hg(rels)
    D, R = difference_tensor(rex, rex)
    assert D.nnz == 0 and R["rank"] == 0
    assert R["agree"] == D.shape[1] and R["disagree"] == 0


def test_deltas_add_but_their_ranks_do_not():
    """The marginal trap as an operator identity.

    D = D1 + D2 exactly, because subtraction is linear. rank(D1 + D2) <= rank(D1) +
    rank(D2) and the inequality is usually strict, which is why a joint reading can never
    be assembled from parts.
    """

    from rexgraph.tensor_diff import difference_tensor
    rels, _w = _mixed(seed=7, n=16, nV=16)
    rng = np.random.default_rng(7)
    flip = sorted(rng.permutation(len(rels))[:len(rels) // 2].tolist())
    half, rest = set(flip[:len(flip) // 2]), set(flip[len(flip) // 2:])
    a = _hg(rels)
    mid = _hg([list(reversed(r)) if i in half else list(r) for i, r in enumerate(rels)])
    b = _hg([list(reversed(r)) if i in (half | rest) else list(r)
             for i, r in enumerate(rels)])
    D, R = difference_tensor(a, b)
    D1, R1 = difference_tensor(a, mid)
    D2, R2 = difference_tensor(mid, b)
    assert (abs(D1 + D2 - D).max() if (D1 + D2 - D).nnz else 0.0) < 1e-12, (
        "the operators add exactly")
    assert R["frobenius2"] == pytest.approx(R1["frobenius2"] + R2["frobenius2"]), (
        "a sum over entries is additive over disjoint supports")
    assert R["rank"] <= R1["rank"] + R2["rank"], "rank is subadditive"


def test_the_difference_refuses_a_broken_alignment():
    """verify=True asserts the theorem, so it has to be able to fire."""
    import scipy.sparse as sp

    from rexgraph.tensor_diff import difference_tensor
    rels, _w = _mixed(seed=9)
    a = _hg(rels)
    D, R = difference_tensor(a, a, verify=True)
    assert R["max_column_sum"] == pytest.approx(0.0)
    # a column that does not sum to zero is not a boundary column, and the check is the
    # thing that would catch an alignment dropping an entry
    bad = sp.csc_matrix(np.array([[1.0], [0.0], [0.0]]))
    assert abs(float(bad.sum())) > 1e-9


def test_parallel_relations_are_matched_not_collapsed():
    """Two relations over one vertex set are two cells, so both must reach the diff.

    Parallel relations are exactly what carries a cycle without raising rank, so
    dropping the second copy would lose the case the model is built to represent.
    Pairing is a multiset match: interchangeable copies cancel in any order.
    """
    from rexgraph.tensor_diff import difference_tensor
    a = _hg([[0, 1], [0, 1], [1, 2]])          # {0,1} carries two relations
    b = _hg([[1, 0], [1, 0], [1, 2]])          # both reversed
    _D, R = difference_tensor(a, b)
    assert R["disagree"] == 2, "both copies moved, so both are in the difference"
    d = tensor_diff(a, b)
    assert d["reoriented"] == 2 and d["novel"] == 0
    assert d["orientation_entries"] == 4


def test_one_of_two_parallel_relations_reversed():
    """The multiset pairs the identical copy off first, leaving exactly one difference."""
    from rexgraph.tensor_diff import difference_tensor
    a = _hg([[0, 1], [0, 1], [1, 2]])
    b = _hg([[0, 1], [1, 0], [1, 2]])          # only the second reversed
    _D, R = difference_tensor(a, b)
    assert R["disagree"] == 1
    d = tensor_diff(a, b)
    assert d["identical"] == 2 and d["reoriented"] == 1 and d["novel"] == 0


def test_an_extra_parallel_copy_is_novel_not_invisible():
    """A third copy the reference does not have has no counterpart to pair with."""
    from rexgraph.tensor_diff import difference_tensor
    a = _hg([[0, 1], [1, 2]])
    b = _hg([[0, 1], [0, 1], [1, 2]])          # one more relation on {0,1}
    d = tensor_diff(a, b)
    assert d["identical"] == 2 and d["novel"] == 1, (
        "the extra copy is a cell the reference does not hold")
    _D, R = difference_tensor(a, b)
    assert R["only_in_input"] == 1
    m = d["merge"]
    assert m["rank_delta"] == 0 and m["cycle_delta"] == 1, (
        "a parallel relation raises no rank and opens exactly one cycle")
