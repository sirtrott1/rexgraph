import numpy as np
import pytest

from rexgraph.flow import FieldNavigator, flow_step
from rexgraph.graph import RexGraph, TemporalRex


def _snaps():
    S = [
        ([0, 0, 1], [1, 2, 3]),
        ([0, 0, 1, 2], [1, 2, 3, 4]),
        ([0, 0, 1, 2, 3], [1, 2, 3, 4, 5]),
        ([0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]),
        ([0, 0, 1, 2, 3, 4, 4], [1, 2, 3, 4, 5, 6, 0]),         # cycle close (surprise)
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
    # Observe the native solve and reject dense, spectral or compatibility solves.
    # An unused spy alone must not establish that the flow path is native.
    import numpy.linalg as nla
    import scipy.sparse.linalg as ssla

    import rexgraph.core._hodge as _hodge
    import rexgraph.core._linalg as _linalg
    import rexgraph.core._sparse as _sparse
    forbidden_calls, native_calls = [], []

    def spy(mod, name, bucket):
        if hasattr(mod, name):
            orig = getattr(mod, name)
            monkeypatch.setattr(mod, name, lambda *a, _n=name, _o=orig, **k: (bucket.append(_n), _o(*a, **k))[1])

    for n in ("lstsq",):
        spy(_linalg, n, forbidden_calls)
    for n in ("spmm_AAt_dense_f64", "spmm_AtA_dense_f64"):
        spy(_sparse, n, forbidden_calls)
    for n in ("eigsh", "svds", "lsqr", "cg"):
        spy(ssla, n, forbidden_calls)
    for n in ("eig", "eigh", "eigvals", "eigvalsh", "svd", "lstsq", "pinv"):
        spy(nla, n, forbidden_calls)
    spy(_hodge, "least_squares", native_calls)

    FieldNavigator().run(_snaps())
    assert forbidden_calls == [], f"flow path used a forbidden operation: {forbidden_calls}"
    assert native_calls, "expected the native factor solver to run on the flow path"


@pytest.mark.parametrize("kind", ["cycle", "face", "branching"])
def test_flow_step_does_not_assemble_gram_matrices(monkeypatch, kind):
    from rexgraph.native_sparse import NativeSparse

    if kind == "face":
        rex = RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    elif kind == "branching":
        rex = RexGraph.from_hypergraph([0, 3, 5, 7], [0, 1, 2, 0, 1, 1, 2])
    else:
        rex = _rex([0, 1, 2], [1, 2, 0])
    rex._ensure_clean()

    def refuse_product(*args, **kwargs):
        raise AssertionError("flow_step assembled a Gram matrix")

    monkeypatch.setattr(NativeSparse, "product", refuse_product)
    result = flow_step(rex, np.arange(rex.nE))
    np.testing.assert_allclose(result["draining"] + result["circulating"], 1)
    assert abs(np.dot(result["draining"], result["circulating"])) < 1e-10


def test_idle_steps_do_no_flow_work():
    nav = FieldNavigator()
    log = nav.run(_snaps())
    assert nav.flow_calls < len(log)          # laziness: not every step flows


def test_flow_compatibility_exports_keep_their_original_objects():
    import rexgraph.flow as flow
    from rexgraph.flow.attention import CoParticipationAttention
    from rexgraph.flow.hyperflow import FlowComplex
    from rexgraph.flow.ternary_cochain import TernaryCochain

    assert flow.CoParticipationAttention is CoParticipationAttention
    assert flow.FlowComplex is FlowComplex
    assert flow.TernaryCochain is TernaryCochain
    assert set(flow.__all__) <= set(dir(flow))
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(flow, "not_a_flow_export")
