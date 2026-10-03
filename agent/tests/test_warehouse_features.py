"""Feature calculations retain Core's meaning and refuse invented zero results."""
from fractions import Fraction
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

from agent.warehouse.source import _diffused, _hodge_amplitudes, _numeric_signal, edge_features
from rexgraph import Absent, RexGraph
from rexgraph.object_identity import object_digest


def edge():
    return RexGraph.from_graph([0], [1])


def test_hodge_features_are_absolute_amplitudes_and_not_squared_energy():
    amplitudes = _hodge_amplitudes(edge(), np.array([-2.]))
    assert len(amplitudes) == 3
    assert np.allclose(amplitudes[0], [2.])
    assert np.allclose(amplitudes[1], [0.]) and np.allclose(amplitudes[2], [0.])


def test_single_edge_down_heat_agrees_with_its_analytic_exponential():
    values, names = _diffused(edge(), np.array([3.]), (0., .25, 1.))
    assert values.shape == (1,4) and names[-1] == "dirac_diffus"
    assert np.allclose(values[0,:3], 3*np.exp(-2*np.array([0.,.25,1.])), atol=1e-10)
    assert np.allclose(values[0,-1], 3*np.exp(-2), atol=1e-10)


def test_heat_walk_is_shared_for_all_requested_scales(monkeypatch):
    import rexgraph.scale_propagator as spg
    calls = []
    original = spg._cheb_vectors
    def vectors(*args, **kwargs):
        calls.append(args[1].shape)
        return original(*args, **kwargs)
    monkeypatch.setattr(spg, "_cheb_vectors", vectors)
    _diffused(edge(), np.array([1.]), (.1, .5, 1., 2.))
    assert calls == [(1,)]


def test_dirac_heat_seed_includes_higher_grades(monkeypatch):
    from rexgraph.dirac_propagator import SparseDirac
    from rexgraph.graded_boundary import solid_octahedron_3rex
    graph = RexGraph.from_cells(solid_octahedron_3rex())
    signal = np.arange(1, graph.nE+1, dtype=float)
    original = SparseDirac.heat_squared
    seeds = []
    def heat(self, state, time, **kwargs):
        assert list(self.sizes) == [6,12,8,1]
        assert state.shape == (27,)
        assert np.array_equal(state[self.grade_slice(1)], signal)
        assert np.count_nonzero(state[:6]) == np.count_nonzero(state[18:]) == 0
        seeds.append(state.copy())
        return original(self, state, time, **kwargs)
    monkeypatch.setattr(SparseDirac, "heat_squared", heat)
    before = object_digest(graph)
    features, _ = _diffused(graph, signal, (0.,))
    assert np.allclose(features[:,0], signal) and np.allclose(features[:,1], signal)
    assert len(seeds) == 1 and object_digest(graph) == before


@pytest.mark.parametrize("operation", ["hodge", "heat"])
def test_core_failures_surface_instead_of_becoming_zero_features(monkeypatch, operation):
    def fail(*args, **kwargs): raise RuntimeError("injected core failure")
    if operation == "hodge":
        monkeypatch.setattr(RexGraph, "hodge", fail)
        action = lambda: _hodge_amplitudes(edge(), np.array([1.]))
    else:
        from rexgraph.dirac_propagator import SparseDirac
        monkeypatch.setattr(SparseDirac, "heat_squared", fail)
        action = lambda: _diffused(edge(), np.array([1.]), (.5,))
    with pytest.raises(RuntimeError, match="injected core failure"): action()


@pytest.mark.parametrize("parts", [(np.zeros(1),), (np.zeros(2),)*3, (np.array([np.nan]),)*3])
def test_invalid_core_hodge_outputs_are_refused(monkeypatch, parts):
    monkeypatch.setattr(RexGraph, "hodge", lambda *a, **k: parts)
    with pytest.raises(ValueError, match="warehouse"):
        _hodge_amplitudes(edge(), np.array([1.]))


@pytest.mark.parametrize("signal", [np.array([[1.]]), np.array([True]), np.array([1j]),
    np.array([np.nan]), np.array([np.inf]), np.array([Absent], dtype=object),
    np.array([None], dtype=object), np.ma.array([1.], mask=[True]), np.array(["1"])])
def test_invalid_or_missing_ml_signals_are_explicitly_refused(signal):
    with pytest.raises(ValueError, match="warehouse signal"):
        _numeric_signal(signal, 1)


@pytest.mark.parametrize("times", [(), (-1.,), (np.nan,), (np.inf,), (True,), ("1",), 1.])
def test_invalid_heat_scales_are_refused(times):
    with pytest.raises(ValueError, match="warehouse heat scales"):
        _diffused(edge(), np.array([1.]), times)


def test_feature_names_and_float32_conversion_preserve_primary_exact_state():
    graph = edge()
    before = object_digest(graph)
    values, names = edge_features(graph, np.array([Fraction(1,7)],dtype=object), np.array([0]))
    assert values.dtype == np.float32 and values.shape == (1,len(names))
    assert names[5:8] == ["hodge_grad_abs", "hodge_curl_abs", "hodge_harm_abs"]
    assert values[0,5] == np.float32(1/7) and np.all(np.isfinite(values))
    assert object_digest(graph) == before


@pytest.mark.parametrize("mask", [np.array([True]), np.array([-1]), np.array([1]), np.array([.5])])
def test_feature_addresses_refuse_before_reading_core_fields(mask):
    graph = SimpleNamespace(nE=1)
    with pytest.raises(ValueError, match="tier indices"):
        edge_features(graph, np.array([1.]), mask)


def test_feature_float32_overflow_is_refused(monkeypatch):
    from agent.warehouse import source
    monkeypatch.setattr(source, "_chi_canonical", lambda r: np.zeros((1,4)))
    monkeypatch.setattr(source, "_hodge_amplitudes", lambda r,s: (s,s,s))
    monkeypatch.setattr(source, "_diffused", lambda r,s,t: (s.reshape(1,1), ["heat"]))
    graph = SimpleNamespace(nE=1, rcfe_curvature=np.zeros(1))
    with pytest.raises(ValueError, match="float32"):
        edge_features(graph, np.array([1e100]), np.array([0]))


def test_assembly_shares_full_features_and_publishes_the_feature_contract(monkeypatch, tmp_path):
    from agent.warehouse import source
    from agent.warehouse.assemble import assemble
    import agent.coordinator_adapter as adapter
    import rexgraph.coordinator as coordinator
    from rcdb import MemoryStore
    ed = source.EdgeData(np.repeat(np.arange(4),2), np.array([4,5,4,5,6,7,6,7]),
                         np.arange(1,9,dtype=float), 4,4)
    masks = source.tier_split(ed,2)
    expected, names = source.edge_features(source.edge_complex(ed), ed, np.arange(8))
    original = source.edge_features
    calls, specifications = [], []
    def features(*args, **kwargs):
        calls.append(tuple(args[2]))
        return original(*args, **kwargs)
    class Pools:
        def __init__(self, *args): pass
        def shutdown(self): pass
        def run(self, units, placement, **kwargs):
            results = {}
            for unit in units:
                spec = unit["fn"].args[0]
                specifications.append(spec)
                results[unit["id"]] = {"tier":spec["tier"], "saved":spec["save_path"],
                    "metric":1., "archetype":spec["archetype"], "device":"cpu", "config_id":unit["id"]}
            return results
    class Coordinator:
        def __init__(self, pools): self.pools=pools; self.cost=None
        def plan(self, units): return None
    monkeypatch.setattr(source, "load_edges", lambda *a, **k: ed)
    monkeypatch.setattr(source, "edge_features", features)
    monkeypatch.setattr(source, "hypergraph_bundle", lambda *a: SimpleNamespace(extra={"he_ptr":np.array([0]), "he_idx":np.array([],dtype=int)}))
    monkeypatch.setattr(coordinator, "LanePools", Pools)
    monkeypatch.setattr(coordinator, "Coordinator", Coordinator)
    monkeypatch.setattr(adapter, "work_units", lambda tasks: tasks)
    hive = SimpleNamespace(add_model=lambda *a, **k: None)
    with closing(MemoryStore()) as store:
        report = assemble("trusted-input", store=store, hive=hive, n_tiers=2,
                          sweep=[{"archetype":"hgnn"}], save_dir=str(tmp_path))
        assert calls == [tuple(range(8))] and len(report["tiers"]) == 2
        for spec, mask in zip(specifications,masks,strict=True):
            assert np.array_equal(spec["X"], expected[mask])
            record = store.get_record("tier-"+str(spec["tier"]))
            assert record.meta["feature_channels"] == names
            assert record.meta["feature_contract"] == {"version":2, "numeric":"float32",
                "hodge":"absolute-amplitude", "heat":"down-sector", "dirac_heat":"in-grade"}
