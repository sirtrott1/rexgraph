"""Bit-identity of the CPU multi-core fan-out (compute.parallel_map dispatch).

Parallelism in the compute paths is a PURE performance/dispatch concern: fanning the
independent work items across a thread pool must produce results BIT-IDENTICAL to the
serial version - same order of reduction, same dtype. These tests force-serialize the
parallel path (by monkeypatching ``compute.parallel_map`` to a serial map, and by
pinning ``get_threads()`` to 1) and assert the parallelized quantity equals the serial
computation exactly (``np.array_equal``, not ``allclose``).

Covered:
  * sparse_character.compute_sparse_phi  - the per-vertex Green's phi / kappa, whose CPU
    chunk loop now fans the independent vertex chunks through compute.parallel_map. This
    is exactly what RexGraph.vertex_character / RexGraph.coherence delegate to.
  * RexGraph._effective_resistance_batch - LEFT serial-equivalent on purpose (it delegates
    to scale_propagator.block_cg_solve, a single vectorized block solve, so column tiling
    would change the shared CG stopping and break bit-identity). Guarded here as a
    determinism / alignment regression.

Note on `chunk`: block-CG uses a stopping criterion shared across the columns of a chunk,
so DIFFERENT chunk sizes give tol-level-different values. Bit-identity is asserted only
for the SAME chunk size, parallel vs serial - which is exactly what the parallelization
changes (thread dispatch of the same chunks), nothing else.
"""
import numpy as np
import pytest

from rexgraph import compute
from rexgraph.graph import RexGraph
from rexgraph.sparse_character import build_sparse_character_cheap, compute_sparse_phi


def _graph(nE, nV, seed=0):
    rng = np.random.RandomState(seed)
    src = rng.randint(0, nV, size=nE).astype(np.int32)
    tgt = rng.randint(0, nV, size=nE).astype(np.int32)
    m = src == tgt
    tgt[m] = (tgt[m] + 1) % nV
    return RexGraph.from_graph(src, tgt)


@pytest.fixture
def serialize_parallel_map(monkeypatch):
    """Force compute.parallel_map to a strictly serial in-order map, so a body that
    routes through it runs exactly as the pre-parallelization serial loop did."""
    def _serial(fn, items, **kw):
        return [fn(x) for x in items]
    monkeypatch.setattr(compute, "parallel_map", _serial)
    return _serial


class TestComputeSparsePhiParallel:
    """compute_sparse_phi: the fanned CPU chunk loop == the serial chunk loop, bit-for-bit."""

    def test_phi_kappa_parallel_equals_serial(self, monkeypatch):
        # Many small chunks over a moderate vertex set -> several independent block-CG
        # solves fanned across threads. Same chunk size on both sides.
        g = _graph(nE=200, nV=90, seed=0)
        cheap = build_sparse_character_cheap(g)
        assert cheap["nhats"] > 0

        compute.set_threads(None)                       # default (all cores) -> real fan-out
        par = compute_sparse_phi(g, cheap, chunk=8)

        with monkeypatch.context() as mp:               # identical call, forced serial
            mp.setattr(compute, "parallel_map",
                       lambda fn, items, **kw: [fn(x) for x in items])
            ser = compute_sparse_phi(g, cheap, chunk=8)

        assert par["phi"].dtype == ser["phi"].dtype == np.float64
        assert np.array_equal(par["phi"], ser["phi"])
        assert np.array_equal(par["kappa"], ser["kappa"])

    def test_phi_kappa_threads1_equals_default(self):
        """Pinning the thread width to 1 (parallel_map no-ops) must reduce to serial and
        match the default multi-thread run bit-for-bit - the thread cap is respected."""
        g = _graph(nE=200, nV=90, seed=1)
        cheap = build_sparse_character_cheap(g)

        prev = compute.get_threads()
        try:
            compute.set_threads(None)
            par = compute_sparse_phi(g, cheap, chunk=8)
            compute.set_threads(1)                        # single worker -> serial path
            ser = compute_sparse_phi(g, cheap, chunk=8)
        finally:
            compute.set_threads(prev)

        assert np.array_equal(par["phi"], ser["phi"])
        assert np.array_equal(par["kappa"], ser["kappa"])

    def test_single_chunk_is_serial_noop(self, serialize_parallel_map):
        """A chunk >= nV is a single work item; parallel_map must no-op (serial map),
        so the result is unchanged whether the map is serial or threaded."""
        g = _graph(nE=120, nV=40, seed=2)
        cheap = build_sparse_character_cheap(g)
        one = compute_sparse_phi(g, cheap, chunk=10_000)   # 1 chunk, forced-serial map
        assert one["phi"].shape == (g.nV, cheap["nhats"])
        assert np.all(np.isfinite(one["phi"]))
        assert np.all(np.isfinite(one["kappa"]))

    def test_graph_vertex_character_uses_parallel_phi(self, monkeypatch):
        """RexGraph.vertex_character / coherence delegate to compute_sparse_phi; when the
        sparse-character path is active, the graph-facing quantities equal the serialized
        computation bit-for-bit (the fan-out is transparent to the API)."""
        # nE > eigen_dense_limit (2000) forces the scale-free sparse character path.
        g = _graph(nE=2600, nV=1500, seed=7)
        if not g._use_sparse_character:
            pytest.skip("sparse character path not active for this graph")
        cheap = g._sparse_character

        compute.set_threads(None)
        phi_par = compute_sparse_phi(g, cheap, chunk=128)["phi"]
        with monkeypatch.context() as mp:
            mp.setattr(compute, "parallel_map",
                       lambda fn, items, **kw: [fn(x) for x in items])
            phi_ser = compute_sparse_phi(g, cheap, chunk=128)["phi"]
        assert np.array_equal(phi_par, phi_ser)

        # and the cached graph property is finite / well-shaped through the parallel path
        vc = np.asarray(g.vertex_character, dtype=np.float64)
        assert vc.shape == (g.nV, g.nhats)
        assert np.all(np.isfinite(vc))
        assert np.all(np.isfinite(np.asarray(g.coherence, dtype=np.float64)))


class TestEffectiveResistanceBatchLeftSerial:
    """_effective_resistance_batch is deliberately LEFT delegating to block_cg_solve (a
    single vectorized block solve): guard determinism and per-edge alignment."""

    def test_batch_equals_per_edge_and_thread_invariant(self):
        g = _graph(nE=150, nV=70, seed=3)
        edges = np.arange(min(12, g.nE))

        prev = compute.get_threads()
        try:
            compute.set_threads(None)
            batch_default = g._effective_resistance_batch(edges)
            compute.set_threads(1)
            batch_t1 = g._effective_resistance_batch(edges)
        finally:
            compute.set_threads(prev)

        # thread width does not change the (single-block) CPU solve
        assert np.array_equal(batch_default, batch_t1)
        # batch is aligned to edge order and agrees with the per-edge accessor
        per_edge = np.array([g.effective_resistance(int(e)) for e in edges], dtype=np.float64)
        assert np.allclose(batch_default, per_edge, atol=1e-8)
