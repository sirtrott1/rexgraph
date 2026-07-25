"""GPU-resident Chebyshev == CPU (the parallel/GPU wiring).

Every eigen-free propagator in the tower is a Chebyshev polynomial of a sparse
operator applied to a state, so the SAME computation runs on CPU (scipy) or, when a
GPU backend is active, entirely on-device (operator + Chebyshev vectors stay on the
GPU across the whole recurrence). These tests pin GPU == CPU to machine precision.
They skip cleanly on a CPU-only host, so the suite passes everywhere.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph import scale_propagator as spg


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _has_gpu(), reason="no GPU (torch.cuda) on this host")


@pytest.fixture(scope="module")
def big_L():
    n = 2048
    g = RexGraph.from_graph(np.arange(n), (np.arange(n) + 1) % n)
    return g.L1_sparse.tocsr()


@gpu_only
@pytest.mark.parametrize("t", [0.2, 1.0])
def test_heat_gpu_matches_cpu(big_L, t):
    rng = np.random.default_rng(0)
    F = rng.standard_normal((big_L.shape[0], 16))
    cpu = spg.heat_apply(big_L, F, t, backend="cpu")
    gpu = spg.heat_apply(big_L, F, t, backend="gpu")
    np.testing.assert_allclose(gpu, cpu, atol=1e-11)


@gpu_only
def test_matfunc_wave_gpu_matches_cpu(big_L):
    rng = np.random.default_rng(1)
    F = rng.standard_normal((big_L.shape[0], 8))
    fn = lambda l: np.cos(0.7 * np.sqrt(np.maximum(l, 0.0)))
    cpu = spg.matfunc_apply(big_L, F, fn, 120, backend="cpu")
    gpu = spg.matfunc_apply(big_L, F, fn, 120, backend="gpu")
    np.testing.assert_allclose(gpu, cpu, atol=1e-11)


@gpu_only
def test_gpu_handles_1d_and_block_shapes(big_L):
    rng = np.random.default_rng(2)
    for shape in [(big_L.shape[0],), (big_L.shape[0], 1), (big_L.shape[0], 4)]:
        F = rng.standard_normal(shape)
        cpu = spg.heat_apply(big_L, F, 0.5, backend="cpu")
        gpu = spg.heat_apply(big_L, F, 0.5, backend="gpu")
        assert gpu.shape == cpu.shape
        np.testing.assert_allclose(gpu, cpu, atol=1e-11)


@gpu_only
def test_compute_default_backend_routes_to_gpu(big_L):
    """Setting the compute default backend to a GPU one makes heat_apply(backend=None)
    run on-device automatically - the dispatch seam callers rely on."""
    from rexgraph import compute
    rng = np.random.default_rng(3)
    F = rng.standard_normal((big_L.shape[0], 8))
    cpu = spg.heat_apply(big_L, F, 0.5, backend="cpu")
    prev = compute.get_default_backend()
    try:
        compute.set_default_backend("rocm")
        auto = spg.heat_apply(big_L, F, 0.5)     # backend=None -> compute default -> GPU
    finally:
        compute.set_default_backend(prev)
    np.testing.assert_allclose(auto, cpu, atol=1e-11)


def test_cpu_only_fallback_never_breaks(big_L):
    """A GPU request on a host without a usable GPU path falls back to CPU rather than
    raising (runs on every host, GPU or not)."""
    rng = np.random.default_rng(4)
    F = rng.standard_normal((big_L.shape[0], 4))
    ref = spg.heat_apply(big_L, F, 0.5, backend="cpu")
    out = spg.heat_apply(big_L, F, 0.5, backend="gpu")   # gpu if available, else cpu
    np.testing.assert_allclose(out, ref, atol=1e-11)


@gpu_only
def test_greens_diagonal_gpu_matches_cpu(monkeypatch):
    """diag(RL4^{-1}) via GPU-resident block-CG equals the CPU tiling (and the exact
    dense diag(inv)). The gate is forced low so the GPU path actually runs."""
    import scipy.sparse as sp
    monkeypatch.setattr(spg, "_GPU_MIN_WORK", 0)
    n = 512
    A = sp.random(n, n, density=0.02, format="csr", random_state=0)
    A = (A + A.T)
    A = (A + sp.diags(np.abs(A).sum(1).A1 + 1.0)).tocsr()   # SPD, well-conditioned
    cpu = spg.greens_diagonal(A, backend="cpu")
    gpu = spg.greens_diagonal(A, backend="gpu")
    exact = np.diag(np.linalg.inv(A.toarray()))
    np.testing.assert_allclose(cpu, exact, atol=1e-9)
    np.testing.assert_allclose(gpu, cpu, atol=1e-9)


@gpu_only
def test_sparse_phi_gpu_matches_cpu(monkeypatch):
    """The agent's coherence/character hot path (per-vertex block-CG Green's phi) runs
    GPU-resident and equals the CPU tiling. Gate forced low so the GPU path runs."""
    from rexgraph import sparse_character as sc
    from rexgraph.graph import RexGraph
    monkeypatch.setattr(spg, "_GPU_MIN_WORK", 0)
    rng = np.random.default_rng(0)
    n, src, tgt = 200, [], []
    for i in range(n):
        for j in rng.choice(n, size=4, replace=False):
            if i < int(j):
                src.append(i); tgt.append(int(j))
    g = RexGraph.from_graph(np.array(src), np.array(tgt))
    cheap = sc.build_sparse_character_cheap(g)
    cpu = sc.compute_sparse_phi(g, cheap, backend="cpu")
    gpu = sc.compute_sparse_phi(g, cheap, backend="gpu")
    np.testing.assert_allclose(gpu["phi"], cpu["phi"], atol=1e-10)
    np.testing.assert_allclose(gpu["kappa"], cpu["kappa"], atol=1e-10)
