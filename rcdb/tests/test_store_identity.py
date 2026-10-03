"""Store identity survives reopen and cannot be confused with another backend."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from rcdb.store_identity import StoreIdentity, local_identity, sql_identity


def test_local_identity_survives_reopening_and_concurrent_initialization(tmp_path):
    path = tmp_path / "identity"
    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(lambda _: local_identity(path, backend="file"), range(12)))
    assert len({value.id for value in values}) == 1
    assert local_identity(path, backend="file") == values[0]
    with pytest.raises(ValueError, match="backend"):
        local_identity(path, backend="rex")
    assert local_identity(tmp_path / "another", backend="file").id != values[0].id


def test_identity_corruption_and_symbolic_link_are_refused(tmp_path):
    path = tmp_path / "identity"
    value = local_identity(path, backend="rex")
    data = path.read_bytes()
    path.write_bytes(data[:-1]+bytes([data[-1]^1]))
    with pytest.raises(ValueError, match="digest"):
        local_identity(path, backend="rex")
    path.write_bytes(value.to_bytes())
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symbolic"):
        local_identity(link, backend="rex")


def test_sql_identity_is_stable_per_logical_store_and_different_between_stores(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'store.sqlite'}")
    try:
        value = sql_identity(engine, "records")
        assert sql_identity(engine, "records") == value
        assert sql_identity(engine, "different_records").id != value.id
    finally:
        engine.dispose()


def test_sqlite_identity_arbitrates_independent_initializing_handles(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'concurrent.sqlite'}"
    engines = [sa.create_engine(uri) for _ in range(6)]
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            values = list(pool.map(lambda engine: sql_identity(engine, "records"), engines))
        assert len({value.id for value in values}) == 1
    finally:
        for engine in engines:
            engine.dispose()


def test_store_identity_is_closed_and_does_not_persist_a_storage_path():
    value = StoreIdentity.create("plugin")
    assert StoreIdentity.from_bytes(value.to_bytes(), backend="plugin") == value
    with pytest.raises(ValueError):
        StoreIdentity.from_bytes(value.to_bytes()+b"extra", backend="plugin")


def test_memory_object_identity_is_shared_across_handles():
    pytest.importorskip("fsspec")
    from rcdb import ObjectStore
    import uuid
    uri = "memory://rcdb-identity-"+uuid.uuid4().hex
    stores = [ObjectStore(uri, read_only=False) for _ in range(6)]
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            identities = list(pool.map(lambda store: store.store_id, stores))
        assert len(set(identities)) == 1
    finally:
        for store in stores:
            store.close()


@pytest.mark.parametrize("kind", ["memory", "file", "rex", "sql", "object"])
def test_public_store_identity_is_distinct_and_durable_where_supported(kind, tmp_path):
    from rcdb import FileStore, MemoryStore, RexStore, SQLStore, ObjectStore
    def opened(root):
        root.mkdir(exist_ok=True)
        if kind == "memory":
            return MemoryStore()
        if kind == "sql":
            pytest.importorskip("sqlalchemy")
            return SQLStore(f"sqlite:///{root / 'store.sqlite'}")
        if kind == "object":
            pytest.importorskip("fsspec")
            return ObjectStore(f"file://{root}", read_only=False)
        return (FileStore if kind == "file" else RexStore)(str(root), read_only=False)
    store = opened(tmp_path / "first")
    other = None
    try:
        identity = store.store_id
        assert store.store_id == identity and len(identity) == 32
        other = opened(tmp_path / "other")
        assert other.store_id != identity
        if kind != "memory":
            store.close()
            store = opened(tmp_path / "first")
            assert store.store_id == identity
    finally:
        store.close()
        if other is not None:
            other.close()
