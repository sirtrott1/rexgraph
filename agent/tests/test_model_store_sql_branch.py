"""Loading a bundle from a database table returns its connection to the pool.

load_bundle's SQL branch handed `engine.connect()` straight to the batch reader and never
closed it. The engine itself is cached for the life of the process, which is what makes
asking for one per load correct; a checked-out connection is the opposite, and every load
kept one. No test reached this branch, which is why it went unnoticed.
"""
from __future__ import annotations

import gc
import warnings

import pytest

sa = pytest.importorskip("sqlalchemy")

from agent.models.store import load_bundle  # noqa: E402


@pytest.fixture
def table_uri(tmp_path):
    """A real sqlite table with the columns the loader expects."""
    uri = f"sqlite:///{tmp_path / 'feat.db'}"
    engine = sa.create_engine(uri)
    try:
        with engine.begin() as conn:
            conn.execute(sa.text(
                "CREATE TABLE feats (a REAL, b REAL, y INTEGER)"))
            conn.execute(sa.text(
                "INSERT INTO feats VALUES (1.0, 2.0, 0), (3.0, 4.0, 1), (5.0, 6.0, 1)"))
    finally:
        engine.dispose()
    return uri


def test_a_bundle_loads_from_a_sql_table(table_uri):
    bundle = load_bundle(table_uri, table="feats", y_col="y")

    assert bundle.X.shape == (3, 2), bundle.X.shape
    assert bundle.y.tolist() == [0, 1, 1]


def test_the_load_returns_its_connection(table_uri):
    """ResourceWarning as an error: a held connection reports at collection."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        load_bundle(table_uri, table="feats", y_col="y")
        gc.collect()


def test_repeated_loads_do_not_accumulate_connections(table_uri):
    """The engine is shared on purpose; the connections must not be."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        for _ in range(5):
            load_bundle(table_uri, table="feats", y_col="y")
        gc.collect()

    from rexgraph.io.sql_bridge import dispose_engines
    dispose_engines()
