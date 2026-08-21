"""Baselines trained on features read back out of the store.

This pins the WIRING, not a method. The question is whether a supervised task can
be run end to end off the RCDB: ingest, persist across a process boundary, project
signatures into a feature matrix, fit, and read the store as it stood at a time,
without anything being smuggled in from the in-memory objects that produced it.

Run against real binding data (651 protein complexes, sqlite on a local file) the
same path gives dummy -0.003, ridge -0.000, randomforest 0.032, lightgbm 0.114,
xgboost -0.084. That LightGBM number is not evidence of anything chemical: the
features are seven structural scalars of a protein-ligand incidence, which carries
how many ligands were measured and almost nothing about binding. It is reported
here so nobody later mistakes a passing pipeline for a result.
"""

import subprocess
import sys

import numpy as np
import pytest

from agent import rcdb
from rexgraph.graph import RexGraph

sklearn = pytest.importorskip("sklearn")
analytics = pytest.importorskip("agent.analytics")
pytest.importorskip("duckdb")


def _corpus(store, n=120, seed=0):
    """Complexes whose size carries a signal, so a fitted model has something real
    to find and a broken pipeline cannot fake it."""
    rng = np.random.default_rng(seed)
    truth = {}
    for k in range(n):
        size = int(rng.integers(4, 40))
        labels = [f"hub{k}"] + [f"leaf{k}_{i}" for i in range(size)]
        rex = RexGraph(sources=np.zeros(size, np.int32),
                       targets=np.arange(1, size + 1, dtype=np.int32))
        rex._agent_meta = {"vertex_labels": labels}
        rid = f"r{k:04d}"
        store.put(rid, rex, meta={"doc_id": rid, "vertex_labels": labels,
                                  "source": "even" if k % 2 == 0 else "odd"})
        truth[rid] = float(np.log1p(size)) + float(rng.normal(0, 0.05))
    return truth


@pytest.fixture
def sqlite_store(tmp_path):
    """A local file endpoint: no server, no network, just a path."""
    return rcdb.open_store(f"sqlite:///{tmp_path / 'rcdb.sqlite'}")


def test_a_local_sqlite_file_is_a_working_endpoint(sqlite_store, tmp_path):
    _corpus(sqlite_store, n=20)
    assert (tmp_path / "rcdb.sqlite").exists()
    assert len(sqlite_store.list(limit=99)) == 20


def test_the_store_is_readable_from_another_process(sqlite_store, tmp_path):
    """The real persistence question. Everything else could be one process's memory."""
    _corpus(sqlite_store, n=15)
    uri = f"sqlite:///{tmp_path / 'rcdb.sqlite'}"
    code = (f"from agent import rcdb;s=rcdb.open_store({uri!r});"
            f"print(len(s.list(limit=999)), int(s.get('r0005').nE))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-800:]
    n_records, n_edges = out.stdout.split()
    assert int(n_records) == 15 and int(n_edges) > 0


def test_features_come_from_the_store_not_the_objects(sqlite_store, tmp_path):
    """Reopened from disk, so nothing can arrive via the complexes still in memory."""
    truth = _corpus(sqlite_store, n=60)
    reopened = rcdb.open_store(f"sqlite:///{tmp_path / 'rcdb.sqlite'}")
    view = analytics.signature_view(reopened)
    rows = view.sql("SELECT id, nV, nE, betti0, betti1, kappa_mean FROM signatures")
    assert len(rows) == 60
    assert all(r[0] in truth for r in rows)
    assert {r[2] for r in rows} != {0}, "every edge count came back zero"


def test_a_baseline_fits_on_what_the_store_returned(sqlite_store, tmp_path):
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    truth = _corpus(sqlite_store, n=150)
    reopened = rcdb.open_store(f"sqlite:///{tmp_path / 'rcdb.sqlite'}")
    view = analytics.signature_view(reopened)
    rows = view.sql("SELECT id, nV, nE, betti0, betti1, kappa_mean FROM signatures")
    X = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], float)
    y = np.array([truth[r[0]] for r in rows], float)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = {}
    for name, model in (("dummy", DummyRegressor(strategy="mean")),
                        ("ridge", Ridge(alpha=1.0))):
        pred = np.zeros_like(y)
        for tr, te in kf.split(X):
            m = model.__class__(**model.get_params())
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        scores[name] = r2_score(y, pred)
    assert scores["ridge"] > scores["dummy"], scores
    assert scores["ridge"] > 0.5, f"the signal did not survive the round trip: {scores}"


@pytest.mark.parametrize("lib", ["lightgbm", "xgboost"])
def test_the_gradient_boosted_baselines_run_on_store_features(sqlite_store, tmp_path, lib):
    mod = pytest.importorskip(lib)
    truth = _corpus(sqlite_store, n=120)
    reopened = rcdb.open_store(f"sqlite:///{tmp_path / 'rcdb.sqlite'}")
    view = analytics.signature_view(reopened)
    rows = view.sql("SELECT id, nV, nE, betti0, betti1, kappa_mean FROM signatures")
    X = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], float)
    y = np.array([truth[r[0]] for r in rows], float)

    if lib == "lightgbm":
        model = mod.LGBMRegressor(n_estimators=40, verbose=-1, random_state=0)
    else:
        model = mod.XGBRegressor(n_estimators=40, verbosity=0, random_state=0)
    model.fit(X[:90], y[:90])
    pred = model.predict(X[90:])
    assert pred.shape == (len(y) - 90,)
    assert np.all(np.isfinite(pred))


def test_a_model_can_be_trained_on_the_corpus_as_it_stood(sqlite_store, tmp_path):
    """The temporal claim, made concrete: a feature matrix at a past time excludes
    what had not been ingested yet, so a model trained on it cannot see the future."""
    import time

    _corpus(sqlite_store, n=40, seed=1)
    time.sleep(0.05)
    cutoff = time.time()
    time.sleep(0.05)
    rng = np.random.default_rng(9)
    for k in range(40, 60):
        size = int(rng.integers(4, 40))
        labels = [f"late{k}"] + [f"leaf{k}_{i}" for i in range(size)]
        rex = RexGraph(sources=np.zeros(size, np.int32),
                       targets=np.arange(1, size + 1, dtype=np.int32))
        rex._agent_meta = {"vertex_labels": labels}
        sqlite_store.put(f"r{k:04d}", rex,
                         meta={"doc_id": f"r{k:04d}", "vertex_labels": labels})

    now = analytics.signature_view(sqlite_store)
    then = analytics.signature_view(sqlite_store, as_of=cutoff)
    assert len(now) == 60
    assert len(then) == 40, "the past view saw records that did not exist yet"


def test_a_revised_record_trains_on_its_old_shape_when_asked(sqlite_store, tmp_path):
    import time

    _corpus(sqlite_store, n=20, seed=2)
    before = int(sqlite_store.get("r0003").nE)
    time.sleep(0.05)
    cutoff = time.time()
    time.sleep(0.05)
    rex = RexGraph(sources=np.zeros(3, np.int32), targets=np.arange(1, 4, dtype=np.int32))
    rex._agent_meta = {"vertex_labels": ["hub3", "a", "b", "c"]}
    sqlite_store.put("r0003", rex, meta={"doc_id": "r0003",
                                         "vertex_labels": ["hub3", "a", "b", "c"]})

    now = analytics.signature_view(sqlite_store)
    then = analytics.signature_view(sqlite_store, as_of=cutoff)
    assert now.sql("SELECT nE FROM signatures WHERE id='r0003'")[0][0] == 3
    assert then.sql("SELECT nE FROM signatures WHERE id='r0003'")[0][0] == before
