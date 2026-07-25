"""Multi-GPU column tiling == single-GPU/CPU (the multi-device wiring).

Every GPU propagator/solver in the tower applies ONE shared sparse operator to a BLOCK
of RHS columns (state nE x ncols). Splitting the COLUMN block across GPUs is exact and
independent: the operator is replicated (identical on every device), only the columns are
partitioned, and the tiles concatenate back. This machine has a single GPU, so the
multi-GPU path can't be hardware-validated; instead we (a) prove the partition/concat math
is exact WITHOUT 2 GPUs by driving `_tile_columns_across_gpus` with a forced tile count >=2
over a CPU reference, (b) where a GPU exists, force >=2 tiles over the SAME physical device
and pin the multi wrappers to the single-device call, and (c) assert the public API still
matches the CPU oracle on this 1-GPU host. The device-count gate MUST degrade the multi-GPU
dispatch to the existing single-device path exactly when fewer than 2 GPUs are usable.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph import compute
from rexgraph import scale_propagator as spg
from rexgraph.graph import RexGraph


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _has_gpu(), reason="no GPU (torch.cuda) on this host")
multi_gpu_only = pytest.mark.skipif(
    compute.gpu_count() < 2, reason="fewer than 2 GPUs on this host")


@pytest.fixture(scope="module")
def big_L():
    n = 1024
    g = RexGraph.from_graph(np.arange(n), (np.arange(n) + 1) % n)
    return g.L1_sparse.tocsr()


@pytest.fixture(scope="module")
def spd_A():
    n = 400
    A = sp.random(n, n, density=0.02, format="csr", random_state=0)
    A = A + A.T
    return (A + sp.diags(np.abs(A).sum(1).A1 + 1.0)).tocsr()   # SPD, well-conditioned


# --- GPU enumeration + gates (mostly device-count-agnostic) -------------------------------------

def test_gpu_devices_matches_count():
    """gpu_devices() enumerates [0 .. gpu_count()-1]; consistent on any host (0, 1, or many GPUs)."""
    assert compute.gpu_devices() == list(range(compute.gpu_count()))


def test_max_gpus_env_caps(monkeypatch):
    """REXGRAPH_MAX_GPUS caps the usable GPU count; =0 forces the single-device path everywhere
    (device-count-agnostic: proves the cap independent of how many GPUs are physically present)."""
    monkeypatch.setenv("REXGRAPH_MAX_GPUS", "0")
    assert compute.gpu_count() == 0
    assert compute.gpu_devices() == []
    monkeypatch.setenv("REXGRAPH_MAX_GPUS", "1")
    assert compute.gpu_count() <= 1


def test_multi_gpu_min_work_env_and_default(monkeypatch):
    """The multi-GPU gate is larger than the single-GPU gate (replication overhead) and is
    overridable at runtime via REXGRAPH_MULTI_GPU_MIN_WORK."""
    assert compute.multi_gpu_min_work() > spg._GPU_MIN_WORK
    monkeypatch.setenv("REXGRAPH_MULTI_GPU_MIN_WORK", "12345")
    assert compute.multi_gpu_min_work() == 12345


def test_multi_gpu_plan_gates(monkeypatch):
    """_multi_gpu_plan returns device indices ONLY with >1 column, >=2 GPUs, and work over the
    multi gate; otherwise None (the single-device fall-through). On a <2-GPU host it is always
    None regardless of work - the property that keeps this machine on the existing path."""
    assert spg._multi_gpu_plan(1 << 40, 1) is None           # single column never tiles
    if compute.gpu_count() < 2:
        assert spg._multi_gpu_plan(1 << 40, 16) is None      # <2 GPUs -> single device
    else:
        assert spg._multi_gpu_plan(1 << 40, 16) == compute.gpu_devices()
        assert spg._multi_gpu_plan(0, 16) is None            # below the multi gate


def test_partition_columns_is_exact_contiguous_cover():
    """The column partition is contiguous, order-preserving, and covers range(ncols) exactly."""
    for ncols, nparts in [(10, 3), (8, 4), (5, 8), (1, 4), (17, 5)]:
        parts = spg._partition_columns(ncols, nparts)
        assert parts[0][0] == 0 and parts[-1][1] == ncols
        assert all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))
        assert all(0 < e - s for s, e in parts)              # no empty tile
        assert len(parts) == max(1, min(nparts, ncols))


# --- the column-tiling math is EXACT without 2 physical GPUs ------------------------------------

def test_tile_columns_equivalence_cpu_reference(big_L):
    """THE device-count-agnostic equivalence test: drive _tile_columns_across_gpus with a forced
    tile count >=2 (devices=[0,1,2]) over a CPU-reference kernel and assert the tiled+concatenated
    result equals the untiled single call to ~1e-12. This proves the partition-then-concatenate
    logic is exact even though only one physical GPU exists - the kernel is device-agnostic, so the
    same helper drives real GPUs in production."""
    rng = np.random.default_rng(0)
    F = rng.standard_normal((big_L.shape[0], 9))
    lm = spg._gershgorin_bound(big_L) * 1.0001 + 1e-30
    fn = lambda l: np.exp(-0.4 * l)
    ref = spg.matfunc_apply(big_L, F, fn, 64, lam_max=lm, backend="cpu")   # untiled

    def kernel(dev, s, e):                                    # CPU kernel: a column slice
        return spg.matfunc_apply(big_L, F[:, s:e], fn, 64, lam_max=lm, backend="cpu")

    tiled = spg._tile_columns_across_gpus(kernel, F.shape[1], [0, 1, 2], axis=1)
    assert tiled.shape == ref.shape
    np.testing.assert_allclose(tiled, ref, atol=1e-12)


@gpu_only
def test_matfunc_multi_over_same_device_matches_single(big_L):
    """With a GPU present, force >=2 column tiles over the SAME physical device ([0, 0]) through
    the real GPU-resident kernel and pin the multi wrapper to the single-device call. The Chebyshev
    recurrence is a fixed per-column polynomial, so this is bit-identical (~1e-12)."""
    rng = np.random.default_rng(1)
    F = rng.standard_normal((big_L.shape[0], 8))
    lm = spg._gershgorin_bound(big_L) * 1.0001 + 1e-30
    fn = lambda l: np.cos(0.7 * np.sqrt(np.maximum(l, 0.0)))
    lam, cos_kj = spg._cheb_basis(lm, 100)
    c = spg._cheb_coeffs(fn(lam), cos_kj, 100)
    single = spg._matfunc_gpu(big_L, F, c, lm, 100)
    multi = spg._matfunc_gpu_multi(big_L, F, c, lm, 100, [0, 0])
    np.testing.assert_allclose(multi, single, atol=1e-12)


@gpu_only
def test_block_cg_multi_over_same_device_matches_single(spd_A):
    """block-CG column tiling over the same device ([0, 0]) equals the single-device solve
    (block-CG tolerance, ~1e-9)."""
    R = spg._csr(spd_A)
    dinv = np.where(np.abs(R.diagonal()) > 1e-30, 1.0 / R.diagonal(), 1.0)
    rng = np.random.default_rng(2)
    B = rng.standard_normal((spd_A.shape[0], 6))
    import torch
    dev = spg._torch_device(None)
    Rt = spg._torch_csr(R, dev)
    single = spg._block_cg_gpu(Rt, torch.as_tensor(B, dtype=torch.float64, device=dev),
                               torch.as_tensor(dinv, dtype=torch.float64, device=dev),
                               tol=1e-10).cpu().numpy()
    multi = spg._block_cg_gpu_multi(R, B, dinv, [0, 0], 1e-10, 1000)
    np.testing.assert_allclose(multi, single, atol=1e-9)


@gpu_only
def test_greens_multi_over_same_device_matches_single(spd_A):
    """diag(A^-1) identity-column tiling across (the same) device ([0, 0]) equals the single-device
    diagonal (~1e-9), and both match the exact dense diag(inv)."""
    R = spg._csr(spd_A)
    n = R.shape[0]
    dinv = np.where(np.abs(R.diagonal()) > 1e-30, 1.0 / R.diagonal(), 1.0)
    single = spg._greens_diagonal_gpu(R, dinv, n, 128, 1e-10)
    multi = spg._greens_diagonal_multi(R, dinv, n, 128, 1e-10, [0, 0])
    exact = np.diag(np.linalg.inv(spd_A.toarray()))
    np.testing.assert_allclose(multi, single, atol=1e-9)
    np.testing.assert_allclose(multi, exact, atol=1e-9)


# --- the public API is unchanged on this (1-GPU) host -------------------------------------------

def test_public_matfunc_matches_cpu_oracle(big_L, monkeypatch):
    """matfunc_apply(backend='gpu') matches the CPU oracle on this host to ~1e-12 (GPU if present,
    else the CPU fallback). Gate forced low so the GPU/multi dispatch actually runs; on a 1-GPU
    host the plan is None, so the single-device path is taken - unchanged."""
    monkeypatch.setattr(spg, "_GPU_MIN_WORK", 0)
    rng = np.random.default_rng(3)
    F = rng.standard_normal((big_L.shape[0], 8))
    fn = lambda l: np.exp(-0.5 * l)
    cpu = spg.matfunc_apply(big_L, F, fn, 80, backend="cpu")
    out = spg.matfunc_apply(big_L, F, fn, 80, backend="gpu")
    np.testing.assert_allclose(out, cpu, atol=1e-12)


def test_public_block_cg_matches_cpu_oracle(spd_A, monkeypatch):
    """block_cg_solve(backend='gpu') matches the CPU solve on this host (~1e-9)."""
    monkeypatch.setattr(spg, "_GPU_MIN_WORK", 0)
    R = spg._csr(spd_A)
    dinv = np.where(np.abs(R.diagonal()) > 1e-30, 1.0 / R.diagonal(), 1.0)
    rng = np.random.default_rng(4)
    B = rng.standard_normal((spd_A.shape[0], 6))
    cpu = spg.block_cg_solve(spd_A, B, dinv, backend="cpu")
    out = spg.block_cg_solve(spd_A, B, dinv, backend="gpu")
    np.testing.assert_allclose(out, cpu, atol=1e-9)


def test_public_greens_matches_cpu_oracle(spd_A, monkeypatch):
    """greens_diagonal(backend='gpu') matches the CPU tiling and the exact dense diag(inv) on this
    host (~1e-9)."""
    monkeypatch.setattr(spg, "_GPU_MIN_WORK", 0)
    cpu = spg.greens_diagonal(spd_A, backend="cpu")
    out = spg.greens_diagonal(spd_A, backend="gpu")
    exact = np.diag(np.linalg.inv(spd_A.toarray()))
    np.testing.assert_allclose(out, cpu, atol=1e-9)
    np.testing.assert_allclose(out, exact, atol=1e-9)


@multi_gpu_only
def test_public_matfunc_multi_gpu_matches_cpu(big_L, monkeypatch):
    """On a genuine multi-GPU host, force the multi-GPU gate low so the column block is tiled across
    devices, and assert the public result still matches the CPU oracle (~1e-12). Skips on this
    single-GPU machine."""
    monkeypatch.setattr(spg, "_GPU_MIN_WORK", 0)
    monkeypatch.setenv("REXGRAPH_MULTI_GPU_MIN_WORK", "0")
    rng = np.random.default_rng(5)
    F = rng.standard_normal((big_L.shape[0], 16))
    fn = lambda l: np.exp(-0.3 * l)
    cpu = spg.matfunc_apply(big_L, F, fn, 80, backend="cpu")
    out = spg.matfunc_apply(big_L, F, fn, 80, backend="gpu")
    np.testing.assert_allclose(out, cpu, atol=1e-12)
