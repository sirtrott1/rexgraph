"""The composite-binary cochain: a {-1,0,1} cochain over cells, held as bitplanes.

The cochain is where a relational-native model is actually dense: `Z[nE, C]` has no
sparsity to exploit and no embedding beside it, so it is the one operator in the flow
layer that a packed ternary form fits. The adjacency is weighted and the boundary is
already stored without values, and neither is packed here.
"""
import numpy as np
import pytest

from rexgraph.flow.ternary_cochain import (
    PACKING_DENSITY,
    TernaryCochain,
    packed_bytes,
    packing_pays,
    residual_tower,
    ternary_reduce,
)


def _rand(nc, cc, seed=0):
    return np.random.default_rng(seed).choice(
        np.array([-1, 0, 1], np.int8), size=(nc, cc), p=[.25, .5, .25])


def test_the_score_is_exact_against_the_dense_product():
    """A count difference, so there is nothing to round."""
    a = _rand(500, 64)
    tc = TernaryCochain(a)
    q = np.random.default_rng(1).choice(np.array([-1, 1]), size=64)
    assert np.array_equal(tc.score(q), a.astype(np.int64) @ q)


def test_every_lane_scores_the_same():
    from rexgraph import ternary as tn
    a = _rand(300, 128)
    tc = TernaryCochain(a)
    q = np.random.default_rng(2).choice(np.array([-1, 1]), size=128)
    want = a.astype(np.int64) @ q
    for be in tn.backends_for("ternary_matvec_pm1"):
        assert np.array_equal(tc.score(q, prefer=be), want), be


def test_support_is_the_class_count_per_cell():
    a = _rand(200, 64)
    assert np.array_equal(TernaryCochain(a).support(), (a != 0).sum(1))


def test_reduction_keeps_the_sign_structure():
    """Two entries dominate, so the code of least spread is the two of them.

    The sweep decides it: support 1 scores 25, support 2 scores 50, support 3 scores
    33.4. Nothing was cut at a magnitude and the answer does not move with the scale.
    """
    z = np.array([[-5.0, -0.01, 0.01, 5.0]])
    q = TernaryCochain.from_float(z).dense()
    assert q.tolist() == [[-1, 0, 0, 1]]
    assert np.array_equal(TernaryCochain.from_float(z * 1000).dense(), q)


def test_a_uniform_cochain_is_exactly_composite_binary():
    """`ones` IS a composite-binary field at scale 1, so it reduces to itself.

    This is the case a magnitude cutoff gets exactly backwards: every entry sits at the
    mean, so a deadzone there zeroes the whole cochain and retains none of its mass,
    while the deviation from the exact composite binary is 0.
    """
    z = np.ones((4, 8))
    c = TernaryCochain.from_float(z)
    assert (c.dense() == 1).all()
    assert float(c.deviation.max()) == 0.0
    assert np.allclose(c.field(), z)


def test_the_deviation_is_the_spread_and_the_residual_mass():
    """`spread(x, q)` is not a proxy for the reduction error, it IS the error.

    With the scale that minimises the residual, `Q(x - s q) = Q(x) * spread(x, q)`, so
    the deviation from the exact composite binary is a quantity `rational_trig` already
    computes and needs no separate definition.
    """
    from rexgraph.rational_trig import spread
    a = np.random.default_rng(4).standard_normal((6, 32))
    q, s, dev = ternary_reduce(a)
    r = a - s[:, None] * q
    assert np.allclose(np.einsum("ij,ij->i", r, r),
                       np.einsum("ij,ij->i", a, a) * dev)
    for i in range(a.shape[0]):
        assert abs(dev[i] - spread(a[i], q[i])) < 1e-12


def test_an_exact_composite_binary_field_recovers_exactly():
    """A field already in {-1,0,1} up to one scale reduces to itself at deviation 0."""
    t = np.random.default_rng(5).integers(-1, 2, (6, 32)).astype(float) * 7.5
    q, s, dev = ternary_reduce(t)
    assert float(dev.max()) == 0.0
    assert np.allclose(s[:, None] * q, t)


def test_the_tower_carries_what_one_level_leaves():
    """Each level reduces the residual of the one above, so mass falls monotonically
    and `bits_per_entry` is two per level. Measured on real weights: one level retains
    about 0.79 of a 256-block's mass, two about 0.95, three about 0.98."""
    a = np.random.default_rng(6).standard_normal((32, 256))
    t = residual_tower(a, max_levels=4)
    masses = t["masses"]
    assert all(b < a_ for a_, b in zip(masses[:-1], masses[1:], strict=True))
    assert t["bits_per_entry"] == 2 * len(t["levels"])
    field = sum(s[:, None] * q for q, s in t["levels"])
    assert np.einsum("ij,ij->i", a - field, a - field).sum() == pytest.approx(masses[-1])


def test_packed_bytes_counts_the_word_padding():
    """A row pads to a whole 64-bit word, so a narrow cochain wastes most of it. The
    idealised 'two bits an entry' is wrong below 64 classes and this is why."""
    assert packed_bytes((1000, 64)) == 1000 * 1 * 8 * 2
    assert packed_bytes((1000, 4)) == packed_bytes((1000, 64))      # same cost, 16x the waste
    assert packed_bytes((1000, 65)) == 1000 * 2 * 8 * 2


def test_packing_is_refused_when_it_would_not_pay():
    """Narrow loses to dense, sparse loses to CSR. Neither is a tuned constant."""
    assert not packing_pays(int(0.3 * 1000 * 2), (1000, 2))         # too narrow
    assert not packing_pays(int(0.001 * 1000 * 256), (1000, 256))   # too sparse
    assert packing_pays(int(0.3 * 1000 * 256), (1000, 256))
    assert 0.0 < PACKING_DENSITY < 1.0


def test_the_query_length_has_to_match():
    tc = TernaryCochain(_rand(8, 64))
    with pytest.raises(ValueError, match="length 64"):
        tc.score(np.ones(63, dtype=np.int64))


def test_a_non_ternary_cochain_is_refused_rather_than_rounded():
    with pytest.raises(ValueError, match="not ternary"):
        TernaryCochain(np.array([[0, 2], [1, 0]]))


def test_predict_returns_one_class_per_cell():
    a = _rand(120, 64)
    p = TernaryCochain(a).predict()
    assert p.shape == (120,)
    assert np.array_equal(p, a.astype(np.float64).argmax(1)) or p.max() < 64


def test_quantising_a_trained_cochain_keeps_most_of_its_signal():
    """The measurement that decides whether the primitive is usable at all. On a
    co-participation task at 4 classes the float cochain held 0.4933 against a 0.25
    floor and the ternary one 0.4760, so 93% of the signal above chance survives and
    87% of the predictions are identical. Reproduced small here."""
    pytest.importorskip("torch")
    from rexgraph.flow.cochain import CoParticipationCochain
    from rexgraph.flow.ternary_cochain import from_cochain_model
    from rexgraph.graph import RexGraph

    rng = np.random.default_rng(0)
    n_ent, C = 200, 4
    truth = rng.integers(0, C, n_ent)
    src = rng.integers(0, n_ent, 800).astype(np.int32)
    tgt = rng.integers(0, n_ent, 800).astype(np.int32)
    keep = src != tgt
    src, tgt = src[keep], tgt[keep]
    labels = truth[src].astype(np.int64)
    obs = rng.random(len(src)) < 0.4
    m = CoParticipationCochain(RexGraph(sources=src, targets=tgt), C)
    m.fit(labels, obs, epochs=120, lr=0.3)
    acc_f = float((m.predict()[~obs] == labels[~obs]).mean())
    acc_t = float((from_cochain_model(m).predict()[~obs] == labels[~obs]).mean())
    chance = 1.0 / C
    assert acc_f > chance
    # the quantised model keeps a real share of what the float one found above chance
    assert (acc_t - chance) > 0.5 * (acc_f - chance), (acc_f, acc_t)
