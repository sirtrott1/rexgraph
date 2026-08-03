import numpy as np

from rexgraph.flow import FieldNavigator, flow_step
from rexgraph.graph import RexGraph, TemporalRex


def _snaps():
    S = [
        ([0, 0, 1], [1, 2, 3]),
        ([0, 0, 1, 2], [1, 2, 3, 4]),
        ([0, 0, 1, 2, 3], [1, 2, 3, 4, 5]),
        ([0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]),
        ([0, 0, 1, 2, 3, 4, 4], [1, 2, 3, 4, 5, 6, 0]),         # cycle-close (surprise)
        ([0, 0, 1, 2, 3, 4, 4, 5], [1, 2, 3, 4, 5, 6, 0, 7]),
    ]
    return TemporalRex([(np.asarray(s, np.int32), np.asarray(t, np.int32)) for s, t in S])


def test_navigator_flows_only_on_events():
    nav = FieldNavigator()
    log = nav.run(_snaps())
    n_events = sum(1 for e in log if e["event"])
    assert nav.flow_calls == n_events           # flow ran exactly once per event, never on idle steps
    assert 1 <= n_events < len(log)             # some events, but NOT every step (laziness is real)
    for e in log:
        if e["event"]:
            assert e["region"].size >= 1        # an event localizes to at least one changed edge


def _rex(src, tgt):
    return RexGraph(sources=np.asarray(src, np.int32), targets=np.asarray(tgt, np.int32))


def test_cycle_flow_is_circulating_not_draining():
    r = _rex([0, 1, 2], [1, 2, 0])                     # triangle cycle
    out = flow_step(r, np.arange(r.nE))
    assert np.linalg.norm(out["circulating"]) > 1e3 * (np.linalg.norm(out["draining"]) + 1e-12)


def test_tree_flow_is_draining_not_circulating():
    r = _rex([0, 0, 1], [1, 2, 3])                     # pure tree
    out = flow_step(r, np.arange(r.nE))
    assert np.linalg.norm(out["circulating"]) < 1e-6
    assert np.linalg.norm(out["draining"]) > 1e-3


def test_boundary_block_lands_on_incident_vertices_only():
    r = _rex([0, 1, 2, 3], [1, 2, 3, 4])               # path; edge 0 = (0,1)
    out = flow_step(r, np.array([0]))                   # seed only edge 0
    vr = np.abs(out["vertex_response"])
    assert vr[0] > 0.5 and vr[1] > 0.5                  # exactly the endpoints of edge 0
    assert np.allclose(vr[2:], 0.0)                     # nothing else lights up


def test_flow_path_is_matrix_free(monkeypatch):
    # spy the real dense/eigen solvers the flow path could touch, plus the matrix-free one,
    # so this test cannot go green just because nothing on the path happened to call numpy.linalg.
    import numpy.linalg as nla
    import scipy.sparse.linalg as ssla

    import rexgraph.core._linalg as _linalg
    import rexgraph.core._sparse as _sparse
    dense_calls, mf_calls = [], []

    def spy(mod, name, bucket):
        if hasattr(mod, name):
            orig = getattr(mod, name)
            monkeypatch.setattr(mod, name, lambda *a, _n=name, _o=orig, **k: (bucket.append(_n), _o(*a, **k))[1])

    for n in ("lstsq",):
        spy(_linalg, n, dense_calls)
    for n in ("spmm_AAt_dense_f64", "spmm_AtA_dense_f64"):
        spy(_sparse, n, dense_calls)
    for n in ("eigsh", "svds"):
        spy(ssla, n, dense_calls)
    for n in ("eig", "eigh", "eigvals", "eigvalsh", "svd", "lstsq", "pinv"):
        spy(nla, n, dense_calls)
    for n in ("lsqr", "cg"):
        spy(ssla, n, mf_calls)

    FieldNavigator().run(_snaps())
    assert dense_calls == [], f"flow path used a dense/eigensolver: {dense_calls}"
    assert mf_calls, "expected the matrix-free iterative solver (lsqr/cg) to run on the flow path"


def test_idle_steps_do_no_flow_work():
    nav = FieldNavigator()
    log = nav.run(_snaps())
    assert nav.flow_calls < len(log)          # laziness: not every step flows
