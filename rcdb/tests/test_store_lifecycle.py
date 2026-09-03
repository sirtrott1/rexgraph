"""Closing a store releases its connections, and the default store closes when reset.

SQLStore defined no close, so it inherited RCStore's no-op and every caller that
dutifully called close kept the engine's pool open. The connections then surfaced at
garbage collection, which reports whichever frame happened to be running rather than
where they were opened, so one lifecycle bug here read as scattered warnings from
sqlalchemy, scipy and the standard library.

These tests promote ResourceWarning to an error, because the defect's whole character was
that it never raised.
"""
from __future__ import annotations

import gc
import warnings

import pytest

from rcdb.core import RCStore, SQLStore, default_store, reset_default_store


def _sqlite_uri(tmp_path, name: str = "store") -> str:
    """A URI under pytest's tmp_path, so the file is cleaned up with the test.

    Not tempfile.mktemp: it leaves the database behind in /tmp, and a test about
    releasing resources should not leak one of its own.
    """
    return "sqlite:///" + str(tmp_path / f"{name}.sqlite")


def test_sqlstore_defines_its_own_close():
    """The inherited no-op is the bug; pin that it is no longer what resolves."""
    assert SQLStore.close is not RCStore.close, "close would be a no-op again"


def test_closing_a_store_releases_its_connections(tmp_path):
    store = SQLStore(_sqlite_uri(tmp_path))
    store.put_bytes("k", b"v") if hasattr(store, "put_bytes") else None

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        store.close()
        del store
        gc.collect()            # an undisposed pool reports here, and used to


def test_close_is_idempotent(tmp_path):
    """It is called from several paths and from reset_default_store."""
    store = SQLStore(_sqlite_uri(tmp_path))
    store.close()
    store.close()               # dispose on a disposed engine is a no-op in SQLAlchemy


def test_resetting_the_default_store_closes_it(monkeypatch, tmp_path):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", _sqlite_uri(tmp_path, "default"))
    reset_default_store()
    store = default_store()
    assert isinstance(store, SQLStore), "this test needs the SQL backend to be the default"

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        reset_default_store()   # must close, not merely forget
        del store
        gc.collect()

    reset_default_store()


def test_reset_survives_a_store_that_fails_to_close(monkeypatch):
    """A store that cannot close must not pin the default forever."""
    import rcdb.core as core

    class Stubborn:
        def close(self):
            raise RuntimeError("no")

    monkeypatch.setattr(core, "_DEFAULT_STORE", Stubborn(), raising=False)
    core.reset_default_store()

    assert core._DEFAULT_STORE is None, "the reset must clear even when close raises"


@pytest.mark.parametrize("uri_scheme", ["memory://"])
def test_other_backends_still_close_without_error(uri_scheme):
    from rcdb.core import open_store

    store = open_store(uri_scheme)
    store.close()
