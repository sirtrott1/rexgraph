"""A ternary operator held as two bitplanes, and the lanes that multiply it.

The encoding is the point: two bits carry a {-1,0,1} entry exactly, so the same product
moves 16x fewer bytes than float32 and every unit here is bandwidth bound long before
it is instruction bound. What is checked below is that the packing loses nothing and
that every registered backend returns the same answer, since a lane that is merely fast
is worthless if it disagrees.
"""
import numpy as np
import pytest

from rexgraph import compute
from rexgraph import ternary as tn


def _rand(nr, nc, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 0, 1], np.int8), size=(nr, nc), p=[.2, .6, .2])


@pytest.mark.parametrize("nc", [1, 63, 64, 65, 500])
def test_packing_is_lossless_at_and_across_the_word_boundary(nc):
    """64 is where a word ends, so it is where an off-by-one would live."""
    a = _rand(7, nc, seed=nc)
    assert np.array_equal(tn.pack(a).dense(), a)


def test_the_planes_are_sixteen_times_smaller_than_float32():
    a = _rand(256, 1024)
    op = tn.pack(a)
    assert op.nbytes * 16 == a.astype(np.float32).nbytes


def test_arity_is_the_support_size_so_the_share_is_recoverable():
    """Share is 1/(k-1) and is deliberately not stored: k is a popcount of presence,
    so a boundary column survives packing with nothing lost."""
    a = _rand(64, 300)
    assert np.array_equal(tn.pack(a).arity(), (a != 0).sum(1))


def test_pack_refuses_a_value_outside_the_set():
    """Rounding a 2 would return a DIFFERENT operator that still looks well formed."""
    with pytest.raises(ValueError, match="not ternary"):
        tn.pack(np.array([[0, 2], [1, 0]]))
    with pytest.raises(ValueError, match="2-D"):
        tn.pack(np.zeros((2, 2, 2)))


def test_the_pm1_product_is_exact_on_every_lane():
    """It is a difference of two counts, so it is an integer and nothing rounds."""
    a = _rand(96, 400)
    op = tn.pack(a)
    rng = np.random.default_rng(3)
    x = rng.choice(np.array([-1, 1]), size=400)
    want = a.astype(np.int64) @ x
    lanes = tn.backends_for("ternary_matvec_pm1")
    assert "cpu" in lanes
    for be in lanes:
        assert np.array_equal(tn.matvec(op, x, prefer=be), want), be


def test_the_float_product_agrees_on_every_lane():
    a = _rand(96, 400)
    op = tn.pack(a)
    v = np.random.default_rng(4).standard_normal(400)
    want = a.astype(np.float64) @ v
    for be in tn.backends_for("ternary_matvec_f64"):
        assert np.allclose(tn.matvec(op, v, prefer=be), want), be


def test_threading_does_not_change_the_answer():
    """The rows are independent, so a thread count is a scheduling choice and must not
    be a numerical one."""
    a = _rand(300, 512)
    op = tn.pack(a)
    x = np.random.default_rng(5).choice(np.array([-1, 1]), size=512)
    want = a.astype(np.int64) @ x
    for be in ("cpu", "openmp"):
        assert np.array_equal(tn.matvec(op, x, prefer=be), want), be


def test_a_new_architecture_registers_without_touching_this_module():
    """The extension point. A backend is added through compute, not by editing here."""
    calls = []

    def _fake(op, v):
        calls.append(op.shape)
        return op.dense().astype(np.int64) @ v

    compute.register_op("ternary_matvec_pm1", "acme_npu", _fake)
    try:
        assert "acme_npu" in tn.backends_for("ternary_matvec_pm1")
        a = _rand(8, 128)
        x = np.random.default_rng(6).choice(np.array([-1, 1]), size=128)
        got = tn.matvec(tn.pack(a), x, prefer="acme_npu")
        assert calls and np.array_equal(got, a.astype(np.int64) @ x)
    finally:
        compute._OPS["ternary_matvec_pm1"].pop("acme_npu", None)


def test_the_vector_length_has_to_match():
    op = tn.pack(_rand(4, 64))
    with pytest.raises(ValueError, match="length 64"):
        tn.matvec(op, np.ones(63, dtype=np.int64))


def test_an_all_zero_operator_is_the_zero_map():
    op = tn.pack(np.zeros((5, 128), np.int8))
    assert np.array_equal(op.arity(), np.zeros(5, np.int64))
    assert np.array_equal(tn.matvec(op, np.ones(128, dtype=np.int64)), np.zeros(5))


def test_the_arity_identity_agrees_with_counting_both_sides():
    """agree - disagree == k - 2*disagree, which is why one popcount suffices."""
    import rexgraph.core._ternary as k
    a = _rand(64, 256)
    P, S, nc = k.pack(a)
    x = np.random.default_rng(7).choice(np.array([-1, 1], np.int8), size=256)
    X = k.pack_vector(x)
    both = k.arity(P) - 2 * np.array(
        [bin(int(w)).count("1") for w in (P & (S ^ X)).ravel()]
    ).reshape(P.shape).sum(1)
    assert np.array_equal(k.matvec_pm1(P, S, X, k.arity(P)), both)


def test_a_supplied_arity_matches_a_derived_one():
    """The operator carries k, so passing it must not change the answer."""
    import rexgraph.core._ternary as k
    a = _rand(48, 320)
    P, S, nc = k.pack(a)
    x = np.random.default_rng(8).choice(np.array([-1, 1], np.int8), size=320)
    X = k.pack_vector(x)
    assert np.array_equal(k.matvec_pm1(P, S, X, None), k.matvec_pm1(P, S, X, k.arity(P)))


def test_a_device_resident_operator_gives_the_same_answer():
    """Residency is a placement decision and must not be a numerical one."""
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no device to hold it")
    a = _rand(64, 512)
    op = tn.pack(a)
    x = np.random.default_rng(9).choice(np.array([-1, 1], np.int8), size=512)
    want = a.astype(np.int64) @ x.astype(np.int64)
    assert np.array_equal(op.to("cuda").matvec(x), want)


def test_a_device_operator_checks_the_vector_length_too():
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no device to hold it")
    op = tn.pack(_rand(4, 128)).to("cuda")
    with pytest.raises(ValueError, match="length 128"):
        op.matvec(np.ones(127, dtype=np.int64))


#### the native device lane, where it is built
def _hip():
    from rexgraph import hip_ternary
    if not hip_ternary.available():
        pytest.skip("the HIP ternary lane is not built on this machine")
    return hip_ternary


def test_the_hip_lane_agrees_with_the_cpu_kernel():
    H = _hip()
    a = _rand(256, 2048)
    op = tn.pack(a)
    x = np.random.default_rng(11).choice(np.array([-1, 1], np.int8), size=2048)
    want = a.astype(np.int64) @ x.astype(np.int64)
    with H.resident(op) as r:
        assert np.array_equal(r.matvec(x), want)


def test_the_hip_lane_handles_a_vector_too_wide_for_lds():
    """Past a point the vector cannot be held in shared memory and the kernel reads it
    from global instead. Same answer, different path, so it is checked."""
    H = _hip()
    a = _rand(8, 70000)                      # nw well past the LDS budget
    op = tn.pack(a)
    x = np.random.default_rng(12).choice(np.array([-1, 1], np.int8), size=70000)
    with H.resident(op) as r:
        assert np.array_equal(r.matvec(x), a.astype(np.int64) @ x.astype(np.int64))


def test_the_hip_lane_is_reachable_through_dispatch():
    _hip()
    assert "hip" in tn.backends_for("ternary_matvec_pm1")
    a = _rand(32, 256)
    x = np.random.default_rng(13).choice(np.array([-1, 1], np.int8), size=256)
    assert np.array_equal(tn.matvec(tn.pack(a), x, prefer="hip"),
                          a.astype(np.int64) @ x.astype(np.int64))


def test_closing_a_resident_operator_twice_is_safe():
    H = _hip()
    r = H.resident(tn.pack(_rand(8, 128)))
    r.close()
    r.close()


def test_the_hip_lane_checks_the_vector_length():
    H = _hip()
    with H.resident(tn.pack(_rand(4, 128))) as r:
        with pytest.raises(ValueError, match="length 128"):
            r.matvec(np.ones(127, dtype=np.int64))


def test_the_float_path_resolves_a_vector_implementation():
    import rexgraph.core._ternary as k
    assert k.float_path() in ("generic", "avx512-masked")
    assert k.bitcount_path() in ("generic", "popcnt", "avx512-vpopcntdq")


@pytest.mark.parametrize("nr", [1, 2, 3, 4, 5, 7, 8, 9])
def test_row_blocking_handles_every_remainder(nr):
    """Rows are processed four at a time, so the tail is where a blocking bug lives."""
    a = _rand(nr, 300, seed=nr)
    v = np.random.default_rng(nr).standard_normal(300)
    assert np.allclose(tn.matvec(tn.pack(a), v), a.astype(np.float64) @ v)


def test_the_hip_float_path_agrees_with_the_cpu():
    H = _hip()
    a = _rand(300, 4096)
    op = tn.pack(a)
    v = np.random.default_rng(21).standard_normal(4096)
    want = a.astype(np.float64) @ v
    with H.resident(op) as r:
        assert np.allclose(r.matvec_f64(v), want)
    assert np.allclose(tn.matvec(op, v, prefer="hip"), want)
