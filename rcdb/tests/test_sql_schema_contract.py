"""SQL identifiers are literal names; migrations must preserve occupied tables."""
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor

import pytest

from rcdb import SQLStore
from rexgraph import RexGraph


def columns(sa, engine):
    # Reflection creates fresh TypeEngine objects; compare their SQL declarations.
    return [{**row, "type": str(row["type"])} for row in sa.inspect(engine).get_columns("rc_complexes")]


@pytest.mark.parametrize("name", ['relations with spaces', 'relations"quoted', 'relations; DROP TABLE unrelated'])
def test_sql_literal_table_names_keep_their_indexes_and_record_contract(tmp_path, name):
    sa = pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'literal.sqlite'}"
    with closing(SQLStore(uri, table=name)) as store:
        indexes = {index["name"] for index in sa.inspect(store.engine).get_indexes(name)}
        assert {f"ix_{name}_{suffix}" for suffix in ("betti1", "kappa_mean", "source", "id_txto", "id_validfrom")} <= indexes
        assert f"ix_{name}_labels_label" in {
            index["name"] for index in sa.inspect(store.engine).get_indexes(name+"_labels")}
        value = RexGraph.from_graph([0], [1])
        store.commit_mutation("r", value, meta={"vertex_labels": ["term"]}, analytics=False)
        assert store.get("r").nE == 1
        assert [r.id for r in store.query(labels_any=["term"])] == ["r"]
        assert store.verify_commits("r")
    with closing(SQLStore(uri, table=name)) as reopened:
        assert reopened.get_record("r").version == 1
        assert reopened.verify_commits("r")


@pytest.mark.parametrize("name", ["rc_complexes", 'legacy "relations"'])
def test_sql_legacy_primary_key_migration_preserves_an_occupied_temporary_name(tmp_path, name):
    sa = pytest.importorskip("sqlalchemy")
    from rcdb.core import serialize_complex
    uri = f"sqlite:///{tmp_path / 'legacy.sqlite'}"
    engine = sa.create_engine(uri)
    metadata = sa.MetaData()
    legacy = sa.Table(name, metadata, sa.Column("id", sa.Text, primary_key=True),
                      sa.Column("signature", sa.Text), sa.Column("meta", sa.Text),
                      sa.Column("created", sa.Float), sa.Column("blob", sa.LargeBinary))
    occupied = sa.Table(name+"__pkmig", metadata, sa.Column("owned", sa.Text))
    try:
        metadata.create_all(engine)
        value = RexGraph.from_graph([0], [1])
        with engine.begin() as conn:
            conn.execute(legacy.insert().values(id="r", signature="{}", meta="{}", created=10.0,
                                               blob=serialize_complex(value)))
            conn.execute(occupied.insert().values(owned="preserve this existing table"))
        with closing(SQLStore(uri, table=name)) as store:
            assert store.get("r").nE == 1
            assert store.put("r", value, analytics=False).version == 2
            assert [r.version for r in store.history("r")] == [1, 2]
            with store.engine.connect() as conn:
                assert conn.execute(occupied.select()).scalar_one() == "preserve this existing table"
            assert set(sa.inspect(store.engine).get_pk_constraint(name)["constrained_columns"]) == {"id", "version"}
    finally:
        engine.dispose()


def test_failed_sqlite_schema_migration_rolls_back_ddl_and_backfills(tmp_path, monkeypatch):
    sa = pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'rollback.sqlite'}"
    engine = sa.create_engine(uri)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE rc_complexes (id TEXT PRIMARY KEY, signature TEXT, "
                                 "meta TEXT, created FLOAT, blob BLOB)")
            conn.exec_driver_sql("INSERT INTO rc_complexes VALUES ('r', '{}', '{}', 10, X'78')")
        before = columns(sa, engine)
        def failed(*args):
            raise OSError("index creation failed")
        monkeypatch.setattr(SQLStore, "_create_label_index", failed)
        with pytest.raises(OSError, match="index creation failed"):
            SQLStore(uri)
        assert columns(sa, engine) == before
        assert sa.inspect(engine).get_table_names() == ["rc_complexes"]
        assert sa.inspect(engine).get_pk_constraint("rc_complexes")["constrained_columns"] == ["id"]
        with engine.connect() as conn:
            assert tuple(conn.exec_driver_sql("SELECT * FROM rc_complexes").one()) == ("r", "{}", "{}", 10, b"x")
    finally:
        engine.dispose()


def test_concurrent_sqlite_creation_publishes_one_schema_and_owner(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'new.sqlite'}"
    handles = []
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            handles = list(pool.map(lambda _: SQLStore(uri), range(6)))
            owners = list(pool.map(lambda store: store.store_id, handles))
        assert len(set(owners)) == 1
        assert all(store.header == handles[0].header for store in handles)
        assert handles[0].header is not None
        assert len(sa.inspect(handles[0].engine).get_indexes("rc_complexes")) == 5
        handles[0].put("r", RexGraph.from_graph([0], [1]), analytics=False)
        assert handles[-1].get("r").nE == 1
    finally:
        for store in handles:
            store.close()


def test_incompatible_sql_primary_key_requires_explicit_migration(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'incompatible.sqlite'}"
    engine = sa.create_engine(uri)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE rc_complexes (id TEXT, signature TEXT, meta TEXT, "
                                 "created FLOAT, blob BLOB)")
        before = columns(sa, engine)
        with pytest.raises(ValueError, match="explicit migration required"):
            SQLStore(uri)
        assert columns(sa, engine) == before
        assert sa.inspect(engine).get_table_names() == ["rc_complexes"]
    finally:
        engine.dispose()
