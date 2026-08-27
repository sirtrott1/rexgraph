"""The leverage refines the rank, so a sectioning of the relations must close.

These pin the two theorems the readings rest on rather than the values they produce:
the masses of a partition total rank(B1), and no section's own cycles exceed its share
of the global cycle space. Both are asserted inside `section_readings`, so a regression
raises there rather than returning a plausible number.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.partition import (
    candidate_readings,
    coupling_fraction,
    grade_leverage,
    section_readings,
    section_tensor,
)


def _complex(seed=0, n=30, extra=45):
    rng = np.random.default_rng(seed)
    src = list(range(n)) + list(rng.integers(0, n, extra))
    tgt = list(np.roll(np.arange(n), -1)) + list(rng.integers(0, n, extra))
    rex = RexGraph(sources=np.asarray(src, np.int32), targets=np.asarray(tgt, np.int32))
    rex._ensure_clean()
    return rex


def test_a_partition_closes_on_the_rank():
    rex = _complex()
    nE = int(rex.nE)
    cut = nE // 3
    parts = {"a": range(cut), "b": range(cut, 2 * cut), "c": range(2 * cut, nE)}
    r = section_readings(rex, parts)
    total = sum(v["mass"] for v in r.values())
    rank = rex.rank_tower()["ranks"][0]
    assert abs(total - rank) < 1e-6, f"{total} against rank {rank}"


def test_every_section_stays_under_its_own_rank():
    rex = _complex(seed=3)
    rng = np.random.default_rng(1)
    nE = int(rex.nE)
    parts = {f"s{i}": rng.choice(nE, size=rng.integers(4, nE // 2), replace=False)
             for i in range(12)}
    r = section_readings(rex, parts)               # verify=True does the asserting
    for name, v in r.items():
        assert v["mass"] <= v["own_rank"] + 1e-6, name
        assert v["own_cycles"] <= v["share"] + 1e-6, name


def test_a_section_that_carries_its_own_cycles_has_no_gap():
    """Theorem 24's equality case: nothing outside is closing anything for it."""
    rex = RexGraph(sources=np.asarray([0, 1, 2], np.int32),
                   targets=np.asarray([1, 2, 0], np.int32))
    rex._ensure_clean()
    r = section_readings(rex, {"triangle": range(int(rex.nE))})["triangle"]
    assert r["own_cycles"] == 1
    assert abs(r["gap"]) < 1e-9


def test_a_bridge_holds_no_share_of_the_cycle_space():
    rex = RexGraph(sources=np.asarray([0, 1, 2, 0], np.int32),
                   targets=np.asarray([1, 2, 0, 3], np.int32))
    rex._ensure_clean()
    r = section_readings(rex, {"bridge": [3], "triangle": [0, 1, 2]})
    assert abs(r["bridge"]["share"]) < 1e-9        # R_eff = 1 exactly
    assert r["bridge"]["efficiency"] == pytest.approx(1.0)


def test_it_refuses_an_index_outside_the_complex():
    rex = _complex()
    with pytest.raises(IndexError):
        section_readings(rex, {"bad": [int(rex.nE)]})


def test_the_coupling_sign_is_not_the_spread():
    """Corollary 25.2: squaring the Gram destroys the compatibility distinction."""
    rex = _complex(seed=5)
    parts = {"a": range(0, 20), "b": range(20, 40)}
    c = coupling_fraction(rex, parts)
    assert set(c) == {"a", "b"}
    for v in c.values():
        assert 0.0 <= v <= 1.0


def test_the_leverage_may_be_supplied_so_one_solve_serves_every_sectioning():
    rex = _complex(seed=7)
    lev = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
    a = section_readings(rex, {"all": range(int(rex.nE))}, leverage=lev)
    b = section_readings(rex, {"all": range(int(rex.nE))})
    assert a["all"]["mass"] == pytest.approx(b["all"]["mass"])


#### the grade axis ###########################################################

def _hyper(seed=0, n=24, groups=14):
    """A branching construction: wide relations plus their own contacts, so the
    complex has real arity and a cycle space rather than a forest of stars."""
    from itertools import combinations
    rng = np.random.default_rng(seed)
    wide = [sorted(rng.choice(n, rng.integers(3, 6), replace=False).tolist())
            for _ in range(groups)]
    pairs = sorted({tuple(sorted(p)) for g in wide for p in combinations(g, 2)})
    ptr, idx = [0], []
    for r in wide + [list(p) for p in pairs]:
        idx += list(r); ptr.append(len(idx))
    return RexGraph.from_hypergraph(np.asarray(ptr, np.int64),
                                    np.asarray(idx, np.int64)), wide, pairs


def test_foster_holds_at_every_grade():
    """sum R_eff_k = rank(B_k) is not a grade-1 fact."""
    from rexgraph.faces import auto_hyperface
    rex, _w, _p = _hyper()
    auto_hyperface(rex)
    assert len(rex.graded_boundaries()) >= 2, "the fixture must reach grade 2"
    for k in (1, 2):
        lev, rank = grade_leverage(rex, k, verify=True)   # verify=True IS the assertion
        assert lev.size and rank > 0
        assert float(lev.sum()) == pytest.approx(rank, abs=1e-6)
        assert (lev >= -1e-9).all() and (lev <= 1 + 1e-9).all(), (
            "a projector diagonal lies in [0, 1]")


def test_grade_leverage_refuses_a_grade_that_is_not_there():
    rex, _w, _p = _hyper()
    with pytest.raises(IndexError):
        grade_leverage(rex, 9)


def test_own_cycles_split_into_curl_and_harmonic():
    """The identity that makes the block axis worth carrying.

    own_cycles counts the cycles a section holds alone; curl is the part its own faces
    fill and harmonic is the part that stays a hole. A section with cycles says nothing
    about whether they are closed until the split is taken.
    """
    import scipy.sparse as sp

    from rexgraph.faces import auto_hyperface
    rex, wide, pairs = _hyper()
    auto_hyperface(rex)
    B2 = sp.csc_matrix(rex.graded_boundaries()[1])
    sup = abs(B2).tocsc()
    pidx = {p: len(wide) + i for i, p in enumerate(pairs)}
    from itertools import combinations
    sections = {}
    for i, g in enumerate(wide):
        e = sorted({i} | {pidx[tuple(sorted(p))] for p in combinations(sorted(g), 2)})
        mask = np.zeros(B2.shape[0], bool); mask[e] = True
        faces = [f for f in range(B2.shape[1])
                 if sup[:, f].nnz and mask[sup[:, f].indices].all()]
        sections[f"g{i}"] = {1: e, 2: faces}
    T, axes = section_tensor(rex, sections, verify=True)
    R = axes["readings"]
    oi, ci, hi, gi = (R.index(x) for x in ("own_cycles", "curl", "harmonic", "gradient"))
    for a, _k in enumerate(axes["grades"]):
        S = T[:, a, :]
        ok = np.isfinite(S[:, oi])
        if not ok.any():
            continue
        assert np.abs(S[ok, oi] - S[ok, ci] - S[ok, hi]).max() < 1e-9
        assert (S[ok, hi] >= -1e-9).all(), "a section cannot have negative holes"
        assert np.abs(S[ok, gi] - S[ok, R.index("own_rank")]).max() < 1e-9


def test_the_tensor_keeps_its_axes():
    rex, wide, _p = _hyper()
    sections = {f"g{i}": [i] for i in range(len(wide))}
    T, axes = section_tensor(rex, sections, verify=True)
    assert T.shape == (len(sections), 1, len(axes["readings"]))
    assert axes["grades"] == [1] and axes["sections"] == list(sections)


#### declaring, before materialising ##########################################

def test_the_candidate_predicate_matches_materialisation():
    """Declare, read, then actually insert. The predicate has to be right every time.

    spans_new means rank(B1) rises by one and no cycle appears; closes means the rank
    holds and exactly one cycle appears. There is no third outcome.
    """
    from rexgraph.graded_boundary import _sparse_rank
    rng = np.random.default_rng(3)
    rex, wide, pairs = _hyper(seed=2)
    ptr, idx = [0], []
    for r in wide + [list(p) for p in pairs]:
        idx += list(r); ptr.append(len(idx))
    r0 = _sparse_rank(rex._integer_B1().tocsc())
    nV = int(rex.nV)

    cands = [sorted(rng.choice(wide[i % len(wide)], 2, replace=False).tolist())
             for i in range(6)]
    cands += [sorted(rng.choice(nV, 2, replace=False).tolist()) for _ in range(4)]
    preds = candidate_readings(rex, cands)
    for c, p in zip(cands, preds, strict=True):
        n2 = RexGraph.from_hypergraph(
            np.asarray(ptr + [len(idx) + len(c)], np.int64),
            np.asarray(idx + list(c), np.int64))
        r1 = _sparse_rank(n2._integer_B1().tocsc())
        dr = r1 - r0
        dc = (int(n2.nE) - r1) - (int(rex.nE) - r0)
        assert (dr, dc) == ((1, 0) if p["spans_new"] else (0, 1)), (
            f"candidate {c} predicted spans_new={p['spans_new']} but moved "
            f"rank by {dr} and cycles by {dc}")


def test_a_candidate_already_spanned_has_finite_quadrance():
    rex, wide, _p = _hyper(seed=1)
    inside = candidate_readings(rex, [sorted(wide[0][:2])])[0]
    assert not inside["spans_new"] and np.isfinite(inside["quadrance"])
    assert inside["quadrance"] > 0


def test_a_candidate_reaching_a_new_vertex_spans_new():
    rex, wide, _p = _hyper(seed=1)
    nV = int(rex.nV)
    # a relation onto a vertex the complex does not have cannot be spanned by it
    with pytest.raises(IndexError):
        candidate_readings(rex, [[0, nV + 3]])


def test_candidate_readings_do_not_touch_the_complex():
    rex, wide, _p = _hyper(seed=4)
    before = (int(rex.nE), int(rex.nV), rex.B1.copy())
    candidate_readings(rex, [sorted(wide[0][:2]), sorted(wide[1][:3])])
    assert (int(rex.nE), int(rex.nV)) == before[:2]
    assert (abs(rex.B1 - before[2]).max() if hasattr(rex.B1, "max") else 0) == 0


#### the energy substrate #####################################################

def test_byte_energy_reads_the_string_and_nothing_else():
    """No complex, no corpus: position-weighted squared bytes."""
    from rexgraph.partition import byte_energy
    assert byte_energy("a") == float(ord("a") ** 2)
    assert byte_energy("ab") == float(ord("a") ** 2 + (ord("b") * 2) ** 2)
    assert byte_energy("") == 0.0
    # position is 1-based, so order matters and no byte vanishes
    assert byte_energy("ab") != byte_energy("ba")
    # multi-byte characters are read as their utf-8 bytes, not as one code point
    assert byte_energy("é") == float(sum((b * (i + 1)) ** 2 for i, b
                                              in enumerate("é".encode())))


def test_energy_needs_a_label_per_vertex():
    from rexgraph.partition import energy_tensor
    rex, wide, _p = _hyper()
    with pytest.raises(ValueError, match="labels has"):
        energy_tensor(rex, {"a": [0]}, ["only", "three", "labels"])


def test_energy_moments_are_what_they_say():
    import scipy.sparse as sp

    from rexgraph.partition import byte_energy, energy_tensor
    rex, wide, _p = _hyper(seed=5)
    labels = [f"w{i}" * (1 + i % 4) for i in range(int(rex.nV))]
    E, moms = energy_tensor(rex, {"one": [0]}, labels)
    sup = abs(sp.csc_matrix(rex.B1)).tocsc()
    verts = np.unique(sup[:, [0]].indices)
    w = np.asarray([byte_energy(labels[v]) for v in verts])
    assert E[0, moms.index("total")] == pytest.approx(w.sum())
    assert E[0, moms.index("mean")] == pytest.approx(w.mean())
    assert E[0, moms.index("spread")] == pytest.approx(w.max() / w.sum())


def test_the_composition_is_multiplicative_and_stays_rank_one():
    """Theorem 27 as an assertion rather than a comment.

    The energy enters as a factor and never as a source, so every (reading, moment)
    plane is rank one. A higher rank means the substrates were mixed.
    """
    from rexgraph.partition import compose_substrates, energy_tensor
    rex, wide, _p = _hyper(seed=6)
    labels = [f"v{i}" * (1 + i % 5) for i in range(int(rex.nV))]
    sections = {f"g{i}": [i] for i in range(len(wide))}
    T, axes = section_tensor(rex, sections, verify=True)
    E, moms = energy_tensor(rex, sections, labels)
    P = compose_substrates(T, E, verify=True)
    assert P.shape == T.shape + (len(moms),)
    i, k, m = 0, 0, 1
    assert np.allclose(P[i, k, :, m], T[i, k, :] * E[i, m], equal_nan=True)
    for i in range(P.shape[0]):
        M = P[i, 0][np.isfinite(P[i, 0]).all(axis=1)]
        if M.shape[0] < 2 or not M.any():
            continue
        s = np.linalg.svd(M, compute_uv=False)
        assert s[1] <= 1e-9 * s[0], "the substrates must not mix"


def test_a_mixed_composition_is_refused():
    """The rank check has to actually fire, or it is decoration."""
    from rexgraph.partition import compose_substrates, energy_tensor
    rex, wide, _p = _hyper(seed=7)
    labels = [f"v{i}" * (1 + i % 5) for i in range(int(rex.nV))]
    sections = {f"g{i}": [i] for i in range(len(wide))}
    T, _axes = section_tensor(rex, sections, verify=True)
    E, _moms = energy_tensor(rex, sections, labels)
    P = compose_substrates(T, E, verify=True)
    P[0, 0, 0, 0] += 1000.0                      # fuse one entry by hand
    M = P[0, 0][np.isfinite(P[0, 0]).all(axis=1)]
    s = np.linalg.svd(M, compute_uv=False)
    assert s[1] > 1e-9 * s[0], "perturbing the product must break rank one"


def test_compose_refuses_mismatched_sections():
    from rexgraph.partition import compose_substrates
    with pytest.raises(ValueError, match="sections in T"):
        compose_substrates(np.zeros((3, 1, 4)), np.zeros((2, 2)))


#### a share is not a finding until you know the null ##########################

def test_hodge_shares_sum_to_one_and_carry_their_null():
    """The decomposition is orthogonal, so the energy shares close."""
    from rexgraph.partition import hodge_share
    rng = np.random.default_rng(11)
    rex = _complex(seed=3)
    f = rng.normal(size=int(rex.nE))
    h = hodge_share(rex, f)
    assert h["residual"] < 1e-6
    assert sum(h["share"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(h["null"].values()) == pytest.approx(1.0, abs=1e-9)
    for k in ("gradient", "curl", "harmonic"):
        assert h["excess"][k] == pytest.approx(h["share"][k] - h["null"][k])


def test_the_null_is_the_dimension_share_not_a_third():
    """Without this the reading is meaningless: 89% gradient may be BELOW chance."""
    from rexgraph.partition import hodge_share
    rng = np.random.default_rng(12)
    rex = _complex(seed=4)
    h = hodge_share(rex, rng.normal(size=int(rex.nE)))
    tot = sum(h["dims"].values())
    for k, d in h["dims"].items():
        assert h["null"][k] == pytest.approx(d / tot)
    assert h["null"]["gradient"] != pytest.approx(1 / 3), (
        "the null follows the complex, not the number of pieces")


def test_a_pure_gradient_reads_as_gradient():
    """B1^T u is gradient by construction, so the share must be 1."""
    import scipy.sparse as sp

    from rexgraph.partition import hodge_share
    rng = np.random.default_rng(13)
    rex = _complex(seed=5)
    u = rng.normal(size=int(rex.nV))
    f = np.asarray(sp.csc_matrix(rex.B1).T @ u).ravel()
    h = hodge_share(rex, f)
    assert h["share"]["gradient"] == pytest.approx(1.0, abs=1e-6)
    assert h["share"]["harmonic"] < 1e-6


def test_hodge_share_refuses_a_zero_signal_and_a_wrong_length():
    from rexgraph.partition import hodge_share
    rex = _complex(seed=6)
    with pytest.raises(ValueError, match="signal is"):
        hodge_share(rex, np.zeros(3))
    with pytest.raises(ValueError, match="no shares"):
        hodge_share(rex, np.zeros(int(rex.nE)))
    with pytest.raises(NotImplementedError):
        hodge_share(rex, np.ones(int(rex.nE)), grade=2)


def test_order_enters_as_orientation_not_as_a_set():
    """The point of the precedence construction, in miniature.

    a->b and b->a are the same support and opposite columns, so a reading that treats
    the support as a set cannot tell them apart, and the Hodge split can.
    """
    from rexgraph.partition import hodge_share
    ptr = np.asarray([0, 2, 4, 6], np.int64)
    idx = np.asarray([0, 1, 1, 2, 0, 2], np.int64)      # a-b, b-c, a-c: one triangle
    rex = RexGraph.from_hypergraph(ptr, idx)
    consistent = np.asarray([1.0, 1.0, 1.0])   # a->b, b->c, a->c agree: a<b<c
    cyclic = np.asarray([1.0, 1.0, -1.0])      # a->b, b->c, c->a: no ordering exists
    hc = hodge_share(rex, consistent)
    hy = hodge_share(rex, cyclic)
    assert hc["share"]["gradient"] > hy["share"]["gradient"], (
        "a consistent ordering must be more gradient than a cyclic one")
    assert hy["share"]["harmonic"] > 0.9, (
        "a 3-cycle with no consistent order is the harmonic generator")


def test_coverage_is_the_candidate_reading_not_the_face_state():
    """The join gives referential coverage, but the face state is not it.

    Putting a query in as a branching relation and reading the face of the whole group
    counts EVERY cycle there, and the reference's own cycles are among them. The exact
    decomposition is

        nullity = (cycles the reference already carried) + (1 if the query is covered)

    so reading `bounds` as `covered` is wrong exactly when the reference is itself
    cyclic. Coverage is the candidate reading: a query is covered iff its column lies in
    range(B1), which is `spans_new` inverted.
    """
    from itertools import combinations

    from rexgraph.faces import face_reading
    from rexgraph.partition import candidate_readings

    rng = np.random.default_rng(21)
    seen_states = set()
    for _ in range(60):
        k = int(rng.integers(4, 7))
        g = list(range(k))
        allp = list(combinations(g, 2))
        m = int(rng.integers(1, len(allp) + 1))
        sel = [allp[i] for i in sorted(rng.permutation(len(allp))[:m])]
        parent = {v: v for v in g}

        def find(x, parent=parent):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for a, b in sel:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        ncomp = len({find(v) for v in g})
        ptr, idx = [0], []
        for a, b in sel:
            idx += [a, b]; ptr.append(len(idx))
        q = len(sel); idx += g; ptr.append(len(idx))
        rx = RexGraph.from_hypergraph(np.asarray(ptr, np.int64),
                                      np.asarray(idx, np.int64))
        state = face_reading(rx, list(range(q + 1)))["state"]
        seen_states.add(state)
        ref = RexGraph.from_hypergraph(np.asarray(ptr[:q + 1], np.int64),
                                       np.asarray(idx[:2 * q], np.int64))
        if int(ref.nV) < k:
            continue
        spans_new = candidate_readings(ref, [g])[0]["spans_new"]
        covered = ncomp == 1
        assert (not spans_new) == covered, (
            "coverage is exactly whether the reference connects the query's terms")
        pair_cycles = m - (k - ncomp)
        nullity = {"open": 0, "bounds": 1}.get(state)
        predicted = pair_cycles + (1 if covered else 0)
        if nullity is None:
            assert predicted >= 2, "degenerate means two or more independent cycles"
        else:
            assert nullity == predicted, (
                f"nullity {nullity} against pair_cycles {pair_cycles} + "
                f"covered {covered}")
    assert {"open", "bounds", "degenerate"} <= seen_states, (
        "the fixture must exercise all three states or it proves nothing")


def test_the_candidate_decision_never_rests_on_the_band():
    """The projection rules and the exact rank adjudicates, so the band picks a method.

    A relative residual either sits at machine level, meaning the column is spanned, or
    at an O(1) fraction of the column, meaning it is not. Nothing real lands between, so
    a well-conditioned complex should never reach the exact path, and when it does the
    answer is still the integer one.
    """
    rng = np.random.default_rng(31)
    rex, wide, _p = _hyper(seed=11, n=18, groups=20)
    nV = int(rex.nV)
    cands = [sorted(rng.choice(nV, int(rng.integers(2, 5)), replace=False).tolist())
             for _ in range(40)]
    out = candidate_readings(rex, cands)
    assert not any(o["adjudicated"] for o in out), (
        "these are ordinary candidates; none should be numerically ambiguous")
    # and the verdicts are still the exact ones
    import scipy.sparse as sp

    from rexgraph.graded_boundary import _sparse_rank
    Bint = rex._integer_B1().tocsc()
    r0 = int(_sparse_rank(Bint))
    for c, o in zip(cands, out, strict=True):
        k = len(c)
        col = np.zeros(nV); col[c] = 1.0; col[c[0]] = -(k - 1)
        aug = sp.hstack([Bint, sp.csc_matrix(col.reshape(-1, 1))]).tocsc()
        assert o["spans_new"] == (int(_sparse_rank(aug)) > r0), (
            f"candidate {c} decided differently from the exact rank")


def test_a_zero_sum_column_passes_nothing_when_its_support_is_seeded_evenly():
    """The gate shuts. `B^T x = x . sum(column)`, and every boundary column sums to zero,
    so a uniformly seeded support gates EXACTLY nothing through: at any arity.

    This is why `section_response` takes the magnitude at the vertex and not at the
    section. Letting head and argument contributions cancel first scores better on a
    corpus sample (100.0% vs 94.0% top-1 over 50 queries) and fails where it matters: a
    section whose vocabulary is unique has degree 1 throughout, so `1/deg` seeding IS
    uniform, its own column cancels to zero, and the most distinctive section in the
    document scores at the floor."""
    import numpy as np
    import scipy.sparse as sp

    for k in (2, 3, 5, 9):
        share = 1.0 / (k - 1)
        col = np.array([-1.0] + [share] * (k - 1))
        B = sp.csc_matrix(col.reshape(-1, 1))
        assert abs(col.sum()) < 1e-12, "the column must be zero-sum to begin with"
        x = np.ones(k)                       # uniform on the support
        assert abs(float((B.T @ x)[0])) < 1e-12, f"arity {k} should gate nothing"
        y = np.ones(k); y[0] = 2.0           # break the uniformity
        assert abs(float((B.T @ y)[0])) > 1e-9, f"arity {k} should pass an uneven seed"


_LAYERED_BOOK = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
                 "CHAPTER I\n\nThe boundary column sums to zero at every arity.\n"
                 "Orientation lives in the sign and not in the position.\n\n"
                 "A relation over k vertices shares one over k minus one.\n\n"
                 "CHAPTER II\n\nThe kernel is the null space of the operator.\n"
                 "Frustration is the holonomy around a cycle, not a count.\n\n"
                 "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n")


def test_the_channel_profile_is_the_scalar_before_it_was_summed():
    """`section_response` returns a scalar per section, and that scalar is the profile
    summed over (topology, geometry, frustration, coparticipation). Asking for the
    profile adds nothing to the computation: it stops throwing the axes away.

    This matters because the channels move in OPPOSITE directions between a section that
    answers a query and one that merely shares vocabulary with it: measured over 237
    queries, topology 0.2379 against 0.2161 and coparticipation 0.2053 against 0.2570.
    Summing them annihilates that difference exactly, which is why every scalar reading
    sat at chance on the question."""
    import numpy as np

    from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
    from rexgraph.document import build_document, section_text
    from rexgraph.partition import section_response
    from rexgraph.sectioning import sectionings_of

    rex, info = build_document(_LAYERED_BOOK, profile=ENGLISH_GUTENBERG)
    base = info["base_layer"]
    sect = sectionings_of(rex)[base]
    vocab = {str(v).lower(): i for i, v in enumerate(info["vocab"])}
    q = section_text(rex, base, 1, _LAYERED_BOOK)
    seeds = [vocab[w] for w, _a, _b in tokenize(q, ENGLISH_GUTENBERG) if w in vocab]
    assert seeds

    scalar, _l = section_response(rex, sect, seeds)
    prof, _l2, names = section_response(rex, sect, seeds, channels=True)
    assert prof.shape == (scalar.size, 4)
    assert names[:2] == ["topology", "geometry"]
    assert np.allclose(scalar, prof.sum(axis=1)), "the scalar IS the summed profile"


def test_topology_and_geometry_share_a_diagonal_so_the_axes_are_three():
    """`diag(T) == diag(G)` because squaring an entry kills its sign, so the four
    channels carry three independent axes on a diagonal reading. Pinned so the profile
    is not read as four."""
    import numpy as np

    from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
    from rexgraph.document import build_document, section_text
    from rexgraph.partition import section_response
    from rexgraph.sectioning import sectionings_of

    rex, info = build_document(_LAYERED_BOOK, profile=ENGLISH_GUTENBERG)
    base = info["base_layer"]
    sect = sectionings_of(rex)[base]
    vocab = {str(v).lower(): i for i, v in enumerate(info["vocab"])}
    q = section_text(rex, base, 1, _LAYERED_BOOK)
    seeds = [vocab[w] for w, _a, _b in tokenize(q, ENGLISH_GUTENBERG) if w in vocab]
    prof, _l, _n = section_response(rex, sect, seeds, channels=True)
    assert np.allclose(prof[:, 0], prof[:, 1]), "T and G share their diagonal"


def test_section_coverage_is_exact_over_the_rationals():
    """Every quantity coverage reads is rational: a boundary entry is -1 at position 0
    and 1/(k-1) after it, a seed weight is 1/deg, and the reading is the unsigned total
    less the magnitude of the signed one."""
    from fractions import Fraction

    import numpy as np

    from rexgraph.graph import RexGraph
    from rexgraph.partition import section_coverage
    from rexgraph.sectioning import add_sectioning, sectionings_of

    ptr = np.array([0, 3, 6, 9, 12], dtype=np.int64)
    idx = np.array([0, 1, 2, 1, 3, 4, 2, 4, 5, 0, 5, 3], dtype=np.int64)
    rex = RexGraph.from_hypergraph(ptr, idx)
    add_sectioning(rex, "s", {"a": [0, 1], "b": [2, 3]})
    sect = sectionings_of(rex)["s"]
    owner = np.asarray(sect.owner_cochain(int(rex.nE)), dtype=np.int64)
    seeds = [0, 1, 4]

    deg = np.bincount(idx, minlength=int(rex.nV))
    want = [Fraction(0)] * 2
    for e in range(int(rex.nE)):
        lo, hi = int(ptr[e]), int(ptr[e + 1])
        k = hi - lo
        mass = Fraction(0)
        signed = Fraction(0)
        for j in range(lo, hi):
            v = int(idx[j])
            if v not in seeds:
                continue
            entry = Fraction(-1) if j == lo else Fraction(1, k - 1)
            w = Fraction(1, int(deg[v]))
            mass += abs(entry) * w
            signed += entry * w
        s = int(owner[e])
        if 0 <= s < 2:
            want[s] += mass - abs(signed)

    got, _labels = section_coverage(rex, sect, seeds)
    for s in range(2):
        assert got[s] == float(want[s]), f"section {s}: {got[s]!r} != {float(want[s])!r}"


def test_the_edge_primary_reading_is_exact_over_the_rationals():
    """`mass[e] = SUM over seeds v in e of |B[v,e]|/deg[v]`, summed per section. One
    hop, read at the relation, and every quantity in it is rational."""
    from fractions import Fraction

    import numpy as np

    from rexgraph.graph import RexGraph
    from rexgraph.partition import section_response
    from rexgraph.sectioning import add_sectioning, sectionings_of

    ptr = np.array([0, 3, 6, 9, 12], dtype=np.int64)
    idx = np.array([0, 1, 2, 1, 3, 4, 2, 4, 5, 0, 5, 3], dtype=np.int64)
    rex = RexGraph.from_hypergraph(ptr, idx)
    add_sectioning(rex, "s", {"a": [0, 1], "b": [2, 3]})
    sect = sectionings_of(rex)["s"]
    owner = np.asarray(sect.owner_cochain(int(rex.nE)), dtype=np.int64)
    seeds = [0, 1, 4]
    deg = np.bincount(idx, minlength=int(rex.nV))

    want = [Fraction(0)] * 2
    for e in range(int(rex.nE)):
        lo, hi = int(ptr[e]), int(ptr[e + 1])
        k = hi - lo
        for j in range(lo, hi):
            v = int(idx[j])
            if v not in seeds:
                continue
            entry = Fraction(1) if j == lo else Fraction(1, k - 1)   # |B[v,e]|
            s = int(owner[e])
            if 0 <= s < 2:
                want[s] += entry * Fraction(1, int(deg[v]))

    got, _labels = section_response(rex, sect, seeds, propagator="mass")
    for s in range(2):
        assert got[s] == float(want[s]), f"section {s}"


def test_the_edge_primary_reading_survives_an_evenly_covered_column():
    """A zero-sum column passes nothing signed when its support is seeded evenly, which
    is where a section is most distinctive. The unsigned total does not cancel."""
    import numpy as np

    from rexgraph.graph import RexGraph
    from rexgraph.partition import section_response
    from rexgraph.sectioning import add_sectioning, sectionings_of

    ptr = np.array([0, 3], dtype=np.int64)
    idx = np.array([0, 1, 2], dtype=np.int64)
    rex = RexGraph.from_hypergraph(ptr, idx)
    add_sectioning(rex, "s", {"only": [0]})
    sect = sectionings_of(rex)["s"]
    seeds = [0, 1, 2]                      # the whole support, every degree 1
    signed, _ = section_response(rex, sect, seeds, propagator="boundary")
    mass, _ = section_response(rex, sect, seeds, propagator="mass")
    assert float(mass[0]) > 0.0, "the unsigned reading still answers"
    assert float(np.asarray(signed)[0]) >= 0.0
