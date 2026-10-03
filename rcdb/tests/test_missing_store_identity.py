"""A missing durable identity cannot silently create a second logical store."""

import pytest

from rexgraph import RexGraph
from rcdb import FileStore, RexStore, SQLStore, ObjectStore


@pytest.mark.parametrize("kind", ["file", "rex", "sql", "object", "memory-object"])
@pytest.mark.parametrize("compacted", [False, True])
def test_bound_durable_records_refuse_identity_loss_before_a_new_record_write(kind, compacted, tmp_path):
    import uuid
    uri = "memory://missing-identity-"+uuid.uuid4().hex
    def opened():
        if kind in {"file", "rex"}:
            return (FileStore if kind == "file" else RexStore)(str(tmp_path), read_only=False)
        if kind == "sql":
            pytest.importorskip("sqlalchemy")
            return SQLStore(f"sqlite:///{tmp_path / 'store.sqlite'}")
        pytest.importorskip("fsspec")
        return ObjectStore(uri if kind == "memory-object" else f"file://{tmp_path}", read_only=False)
    store = opened()
    store.put("r", RexGraph.from_graph([0], [1]), analytics=False)
    _ = store.store_id
    if compacted and kind != "sql":
        store.compact()
    if kind == "sql":
        from sqlalchemy import text
        with store.engine.begin() as connection:
            connection.execute(text("DELETE FROM rc_complexes_store_identity"))
    elif kind == "memory-object":
        store.fs.rm(store._p(".rcdb-identity"))
    else:
        (tmp_path / ".rcdb-identity").unlink()
    store.close()
    reopened = None
    with pytest.raises(ValueError, match="identity is missing|ownership anchor is missing"):
        reopened = opened()
        try:
            reopened.put("s", RexGraph.from_graph([0], [1]), analytics=False)
        finally:
            reopened.close()
    if kind == "sql":
        import sqlalchemy as sa
        engine = sa.create_engine(f"sqlite:///{tmp_path / 'store.sqlite'}")
        try:
            with engine.connect() as connection:
                assert connection.execute(sa.text("SELECT count(*) FROM rc_complexes_store_identity")).scalar_one() == 0
                assert connection.execute(sa.text("SELECT id FROM rc_complexes")).scalars().all() == ["r"]
        finally:
            engine.dispose()
    elif kind == "memory-object":
        assert not store.fs.exists(store._p(".rcdb-identity"))
    else:
        assert not (tmp_path / ".rcdb-identity").exists()


def test_existing_identity_must_match_the_checked_journal(tmp_path):
    from rcdb.store_identity import StoreIdentity
    store = FileStore(str(tmp_path), read_only=False)
    store.put("r", RexGraph.from_graph([0], [1]), analytics=False)
    store.close()
    path = tmp_path / ".rcdb-identity"
    other = StoreIdentity.create("file").to_bytes()
    path.write_bytes(other)
    with pytest.raises(ValueError, match="durable ownership"):
        FileStore(str(tmp_path), read_only=False)
    assert path.read_bytes() == other
