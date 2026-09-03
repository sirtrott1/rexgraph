"""One engine per connection string, and a way to let them go.

get_engine cached in-memory engines but built a NEW engine for every file-backed call.
An engine owns a connection pool, so a caller that asks per operation, as the workspace
persistence layer does, opened one pool per save and abandoned every one of them. The
cost was not only the leak: rebuilding the engine each time made a 25-test round trip
take 7.05s that now takes 0.41s.

The cache is process-wide and long lived by design, which is what makes calling
get_engine per operation correct. dispose_engines is how a process that is finished with
a database actually releases them.
"""
from __future__ import annotations

import gc
import warnings

import pytest

sa = pytest.importorskip("sqlalchemy")

from rexgraph.io.sql_bridge import dispose_engines, get_engine  # noqa: E402


def test_the_same_file_uri_returns_the_same_engine(tmp_path):
    uri = f"sqlite:///{tmp_path / 'a.db'}"
    try:
        assert get_engine(uri) is get_engine(uri), "a new engine per call is a new pool"
    finally:
        dispose_engines()


def test_different_uris_get_different_engines(tmp_path):
    try:
        a = get_engine(f"sqlite:///{tmp_path / 'a.db'}")
        b = get_engine(f"sqlite:///{tmp_path / 'b.db'}")
        assert a is not b, "two databases must not share one engine"
    finally:
        dispose_engines()


def test_dispose_releases_the_pools_and_empties_the_cache(tmp_path):
    uri = f"sqlite:///{tmp_path / 'c.db'}"
    engine = get_engine(uri)
    with engine.connect() as conn:
        conn.execute(sa.text("SELECT 1"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        dispose_engines()
        gc.collect()            # an undisposed pool reports here

    assert get_engine(uri) is not engine, "dispose must also forget, not just release"
    dispose_engines()


def test_dispose_is_safe_on_an_empty_cache():
    dispose_engines()
    dispose_engines()           # idempotent: shutdown may run it after a caller already did


def test_a_broken_engine_does_not_block_the_rest(tmp_path, monkeypatch):
    """One engine that fails to dispose must not strand the others."""
    from rexgraph.io import sql_bridge

    good = get_engine(f"sqlite:///{tmp_path / 'good.db'}")

    class Stubborn:
        def dispose(self):
            raise RuntimeError("no")

    sql_bridge._ENGINE_CACHE["stubborn://x"] = Stubborn()
    dispose_engines()

    assert not sql_bridge._ENGINE_CACHE, "the cache must be emptied even so"
    assert get_engine(f"sqlite:///{tmp_path / 'good.db'}") is not good
    dispose_engines()


def test_an_in_memory_uri_is_still_shared(tmp_path):
    """The branch that already cached must keep doing so."""
    try:
        assert get_engine("sqlite:///:memory:") is get_engine("sqlite:///:memory:")
    finally:
        dispose_engines()


def test_the_server_lifespan_disposes_the_cache(tmp_path, monkeypatch):
    """The shutdown hook has to actually run, which depends on how it is invoked.

    The disposal lives in the app's lifespan. A TestClient that is constructed but never
    entered skips startup and shutdown entirely, so the hook does not fire and the pools
    outlive the caller. That is not a hypothetical: it is how one suite was using it.
    """
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    pytest.importorskip("agent.server.app")

    from agent.server.app import app

    from rexgraph.io import sql_bridge

    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    get_engine(f"sqlite:///{tmp_path / 'lifespan.db'}")
    assert sql_bridge._ENGINE_CACHE, "the cache needs an entry for this to prove anything"

    with fastapi_testclient.TestClient(app):
        pass                     # entering and leaving runs startup and shutdown

    assert not sql_bridge._ENGINE_CACHE, "the lifespan did not dispose the cached engines"
