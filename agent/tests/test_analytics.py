"""Columnar queries over signatures, against every backend.

Every store answers a structural predicate at about the same speed because they all
walk their signatures one at a time. What none of them can answer is an aggregate --
how kappa distributes by source, which betti values are over-represented -- because
a signature is a document and aggregating documents means writing the loop yourself.

The view is derived, not another copy: it projects out of whatever store holds the
records and hands back ids that store can be asked for.
"""

import numpy as np
import pytest

from agent import rcdb
from rexgraph.graph import RexGraph

analytics = pytest.importorskip("agent.analytics")
pytest.importorskip("duckdb")


def _rex(labels, n_edges):
    r = RexGraph(sources=np.arange(n_edges, dtype=np.int32),
                 targets=np.arange(1, n_edges + 1, dtype=np.int32))
    r._agent_meta = {"vertex_labels": labels}
    return r


@pytest.fixture(params=["memory", "file", "sql", "rex"])
def store(request, tmp_path):
    kind = request.param
    if kind == "memory":
        st = rcdb.MemoryStore()
    elif kind == "file":
        st = rcdb.FileStore(str(tmp_path / "fs"))
    elif kind == "sql":
        st = rcdb.SQLStore(f"sqlite:///{tmp_path / 'rc.sqlite'}")
    else:
        st = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    for k in range(12):
        labels = [f"l{k}_{i}" for i in range(4)]
        st.put(f"r{k:02d}", _rex(labels, 3 + k),
               meta={"doc_id": f"r{k:02d}", "vertex_labels": labels,
                     "source": "even" if k % 2 == 0 else "odd"})
    return st


def test_a_view_projects_every_backend(store):
    view = analytics.signature_view(store)
    assert len(view) == 12
    assert "kappa_mean" in view.columns()


def test_a_predicate_returns_ids_the_store_can_resolve(store):
    view = analytics.signature_view(store)
    ids = view.ids("nE >= 10")
    assert ids
    for rid in ids:
        assert store.get(rid) is not None


def test_the_predicate_agrees_with_the_store_s_own_answer(store):
    """A faster answer that differs is not an answer."""
    view = analytics.signature_view(store)
    from_view = set(view.ids("nE >= 10"))
    from_store = {r.id for r in store.query(min_nE=10, limit=10 ** 9)}
    assert from_view == from_store


def test_aggregation_is_the_thing_no_store_could_do(store):
    view = analytics.signature_view(store)
    rows = view.sql(
        "SELECT source, count(*), avg(nE) FROM signatures GROUP BY source ORDER BY source")
    assert [r[0] for r in rows] == ["even", "odd"]
    assert sum(r[1] for r in rows) == 12


def test_complexes_come_back_from_the_store_not_the_view(store):
    """The view holds scalars. The payload has exactly one home."""
    view = analytics.signature_view(store)
    pairs = view.complexes("nE >= 12", limit=3)
    assert pairs
    for _rid, rex in pairs:
        assert rex is not None and int(rex.nE) >= 12


def test_a_view_is_derived_and_refreshed_rather_than_kept_in_step(store):
    """A view that silently lags what it describes is worse than one you refresh."""
    view = analytics.signature_view(store)
    before = len(view)
    labels = ["new0", "new1", "new2", "new3"]
    store.put("later", _rex(labels, 30),
              meta={"doc_id": "later", "vertex_labels": labels, "source": "even"})
    assert len(view) == before, "the view changed without being refreshed"
    view.refresh()
    assert len(view) == before + 1
    assert "later" in view.ids("nE >= 30")


def test_describe_summarises_every_column(store):
    view = analytics.signature_view(store)
    assert view.describe()


def test_the_view_survives_an_empty_store(tmp_path):
    view = analytics.signature_view(rcdb.MemoryStore())
    assert len(view) == 0
    assert view.ids("nE > 0") == []


def test_arrow_export_needs_no_dataframe_dependency(store):
    view = analytics.signature_view(store)
    table = view.to_arrow()
    assert table.num_rows == 12


def test_polars_can_read_the_arrow_export(store):
    """Whoever prefers a dataframe gets one without this module depending on it."""
    pl = pytest.importorskip("polars")
    view = analytics.signature_view(store)
    df = pl.from_arrow(view.to_arrow())
    assert df.height == 12
    assert "kappa_mean" in df.columns


def test_history_is_available_when_asked_for(tmp_path):
    st = rcdb.open_store(f"rex://{tmp_path / 'rx'}")
    labels = ["a", "b", "c", "d"]
    st.put("x", _rex(labels, 3), meta={"vertex_labels": labels})
    st.put("x", _rex(labels, 9), meta={"vertex_labels": labels})
    assert len(analytics.signature_view(st)) == 1
    assert len(analytics.signature_view(st, include_history=True)) == 2


def test_a_missing_duckdb_says_so(monkeypatch):
    import builtins
    real = builtins.__import__

    def _fail(name, *a, **kw):
        if name == "duckdb":
            raise ImportError("no duckdb")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fail)
    with pytest.raises(ImportError) as ei:
        analytics.signature_view(rcdb.MemoryStore())
    assert "duckdb" in str(ei.value)
