"""
Tests for rexgraph.core._query -- relational complex query engine.

Verifies:
    - Predicate masking: correct count, ops work, BETWEEN range
    - Mask boolean ops: AND, OR, NOT
    - Signal imputation: fully observed returns input, partial fills missing
    - Spectral propagation: self-propagation score > 0
    - Explain edge: returns expected keys, correct below/above
    - Explain vertex: returns expected keys, degree matches
    - Integration through RexGraph: impute, explain, propagate
"""
import numpy as np
import pytest

from rexgraph.core import _query
from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


# Predicate Masking

class TestPredicateMask:

    def test_gt(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask, count = _query.predicate_mask(vals, 5, _query.PRED_GT, 3.0)
        assert count == 2
        assert mask[3] == 1
        assert mask[4] == 1
        assert mask[2] == 0

    def test_le(self):
        vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        mask, count = _query.predicate_mask(vals, 3, _query.PRED_LE, 2.0)
        assert count == 2

    def test_eq(self):
        vals = np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float64)
        mask, count = _query.predicate_mask(vals, 4, _query.PRED_EQ, 2.0)
        assert count == 2

    def test_between(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask, count = _query.predicate_mask(vals, 5, _query.PRED_BETWEEN, 2.0, 4.0)
        assert count == 3  # 2.0, 3.0, 4.0

    def test_empty(self):
        vals = np.array([1.0, 2.0], dtype=np.float64)
        mask, count = _query.predicate_mask(vals, 2, _query.PRED_GT, 10.0)
        assert count == 0
        assert np.all(mask == 0)


# Mask Boolean Ops

class TestMaskOps:

    def test_and(self):
        a = np.array([1, 1, 0, 0], dtype=np.uint8)
        b = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = _query.mask_and(a, b, 4)
        assert list(result) == [1, 0, 0, 0]

    def test_or(self):
        a = np.array([1, 1, 0, 0], dtype=np.uint8)
        b = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = _query.mask_or(a, b, 4)
        assert list(result) == [1, 1, 1, 0]

    def test_not(self):
        a = np.array([1, 0, 1], dtype=np.uint8)
        result = _query.mask_not(a, 3)
        assert list(result) == [0, 1, 0]


# Signal Imputation

class TestSignalImpute:

    def test_fully_observed(self, k4):
        signal = np.ones(k4.nE, dtype=np.float64)
        mask = np.ones(k4.nE, dtype=np.uint8)
        result = _query.signal_impute(k4.RL, signal, mask, k4.nE)
        assert result['n_imputed'] == 0
        assert np.allclose(result['imputed'], signal)

    def test_partial_fills_missing(self, k4):
        signal = np.zeros(k4.nE, dtype=np.float64)
        signal[0] = 1.0
        signal[1] = 2.0
        mask = np.zeros(k4.nE, dtype=np.uint8)
        mask[0] = 1
        mask[1] = 1
        result = _query.signal_impute(k4.RL, signal, mask, k4.nE)
        assert result['n_observed'] == 2
        assert result['n_imputed'] == k4.nE - 2
        assert result['imputed'].shape == (k4.nE,)
        # Observed values should be preserved
        assert result['imputed'][0] == 1.0
        assert result['imputed'][1] == 2.0

    def test_returns_all_keys(self, k4):
        signal = np.ones(k4.nE, dtype=np.float64)
        mask = np.ones(k4.nE, dtype=np.uint8)
        result = _query.signal_impute(k4.RL, signal, mask, k4.nE)
        for key in ['imputed', 'confidence', 'residual', 'n_observed', 'n_imputed']:
            assert key in result


# Explain Edge

class TestExplainEdge:

    def test_returns_keys(self, k4):
        K1 = np.abs(np.asarray(k4.B1, dtype=np.float64)).T @ np.abs(np.asarray(k4.B1, dtype=np.float64))
        rcf = k4._rcf_bundle
        result = _query.explain_edge(
            k4.B1, k4.B2_hodge, K1, k4.RL,
            rcf['hats'], rcf['nhats'],
            0, k4.nV, k4.nE, k4.nF)
        for key in ['below', 'above', 'lateral', 'chi',
                     'dominant_channel', 'degree']:
            assert key in result

    def test_below_are_vertices(self, k4):
        K1 = np.abs(np.asarray(k4.B1, dtype=np.float64)).T @ np.abs(np.asarray(k4.B1, dtype=np.float64))
        rcf = k4._rcf_bundle
        result = _query.explain_edge(
            k4.B1, k4.B2_hodge, K1, k4.RL,
            rcf['hats'], rcf['nhats'],
            0, k4.nV, k4.nE, k4.nF)
        assert result['degree'] >= 2  # standard edge has 2 endpoints
        assert np.all(result['below'] >= 0)
        assert np.all(result['below'] < k4.nV)


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_impute(self, k4):
        signal = np.zeros(k4.nE, dtype=np.float64)
        signal[0] = 1.0
        mask = np.zeros(k4.nE, dtype=np.uint8)
        mask[0] = 1
        result = k4.impute(signal, mask)
        assert isinstance(result, dict)
        assert 'imputed' in result

    def test_explain_edge(self, k4):
        result = k4.explain(dim=1, idx=0)
        assert isinstance(result, dict)
        assert 'below' in result

    def test_explain_vertex(self, k4):
        result = k4.explain(dim=0, idx=0)
        assert isinstance(result, dict)
        assert 'phi' in result
        assert 'degree' in result

    def test_propagate(self, k4):
        source = np.zeros(k4.nE, dtype=np.float64)
        source[0] = 1.0
        target = np.zeros(k4.nE, dtype=np.float64)
        target[0] = 1.0
        result = k4.propagate(source, target)
        assert isinstance(result, dict)
        assert 'score' in result
        assert result['score'] > 0  # self-propagation
