"""A structural query rejects keys it does not implement.

`_matches` checks a fixed list of predicates. A key outside that list matches every
record, so `query(nE=4)` reads as a filter and returns the whole store: a wrong answer
shaped like a right one. The bound is spelled `max_nE`, and the error says so.
"""

import numpy as np
import pytest

import agent.rcdb as R
from rexgraph.faces import autoface
from rexgraph.graph import RexGraph


def _rex(extra=0):
    bp, bi = [0, 3, 5, 7, 9], [0, 1, 2, 0, 1, 1, 2, 2, 3]
    for i in range(extra):
        bp.append(bp[-1] + 2)
        bi += [3, 4 + i]
    r = RexGraph.from_hypergraph(np.asarray(bp, np.int32), np.asarray(bi, np.int32))
    autoface(r, 3)
    return r


@pytest.fixture
def store():
    s = R.MemoryStore()
    for i in range(4):
        s.put(f"r{i}", _rex(i))          # nE = 4, 5, 6, 7
    return s


def test_documented_bounds_filter(store):
    assert sorted(r.id for r in store.query(max_nE=4)) == ["r0"]
    assert sorted(r.id for r in store.query(min_nE=6)) == ["r2", "r3"]


def test_an_unsupported_key_raises(store):
    with pytest.raises(TypeError, match="unsupported query key"):
        store.query(nE=4)


def test_the_error_names_the_supported_keys(store):
    with pytest.raises(TypeError, match="max_nE"):
        store.query(nE=4)


def test_several_unknown_keys_are_reported_together(store):
    with pytest.raises(TypeError, match="bogus.*zzz|zzz.*bogus"):
        store.query(bogus=1, zzz=2)


def test_a_known_key_beside_an_unknown_one_still_raises(store):
    with pytest.raises(TypeError, match="unsupported query key"):
        store.query(min_nE=6, nE=4)


def test_an_empty_query_returns_everything(store):
    assert len(store.query()) == 4
