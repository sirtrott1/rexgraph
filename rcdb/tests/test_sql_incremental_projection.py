"""Native SQLite changes get exact bounded audits; other changes remain untrusted."""
from contextlib import closing
from fractions import Fraction

import pytest

from rcdb import SQLStore, MemoryStore, copy_record
from rexgraph.value_codec import pack_value

sa = pytest.importorskip("sqlalchemy")


@pytest.fixture(params=["file", "memory"])
def store(request, tmp_path):
    uri = f"sqlite:///{tmp_path}/store.sqlite" if request.param == "file" else "sqlite://"
    with closing(SQLStore(uri)) as st:
        for index in range(12):
            st.put_record(f"prior-{index}", Fraction(index, 7), tx_time=10.,
                          meta={"vertex_labels": ["shared", f"term-{index}"]})
        st.list(limit=1)
        yield st


def audit_work(store, monkeypatch):
    native = store._native
    full, delta, pages = [], [], []
    old_full, old_delta, old_rows = native._audit_projection, native._audit_projection_delta, native._audit_projection_rows
    def audit_full(connection):
        full.append(native.state.cursor)
        return old_full(connection)
    def audit_delta(connection, addresses):
        delta.append(frozenset(addresses))
        return old_delta(connection, addresses)
    def audit_rows(connection, expected, **kwargs):
        pages.append(frozenset(expected))
        return old_rows(connection, expected, **kwargs)
    monkeypatch.setattr(native, "_audit_projection", audit_full)
    monkeypatch.setattr(native, "_audit_projection_delta", audit_delta)
    monkeypatch.setattr(native, "_audit_projection_rows", audit_rows)
    return full, delta, pages


@pytest.mark.parametrize("operation", ["insert", "replace", "delete", "revive", "absent-delete"])
def test_audit_exact_changed_addresses_and_keep_selection_semantics(store, monkeypatch, operation):
    if operation == "revive":
        store.delete("prior-0", tx_time=11.)
        store.list(limit=1)
    full, delta, pages = audit_work(store, monkeypatch)
    if operation == "insert":
        store.put_record("new", Fraction(2, 7), tx_time=20., meta={"vertex_labels": ["NEW"]})
        expected = {("new", 1)}
    elif operation in ("replace", "revive"):
        store.put_record("prior-0", Fraction(2, 7), tx_time=20., meta={"vertex_labels": ["NEW"]})
        expected = {("prior-0", 2)} | ({("prior-0", 1)} if operation == "replace" else set())
    elif operation == "delete":
        assert store.delete("prior-0", tx_time=20.)
        expected = {("prior-0", 1)}
    else:
        assert not store.delete("absent", tx_time=20.)
        expected = set()
    assert full == delta == pages == []  # Publication is not certification.
    assert bool(store._native._projection_delta) == bool(expected)
    assert len(store.list(limit=100, include_history=True)) == (13 if operation in ("insert", "replace", "revive") else 12)
    assert full == [] and delta == ([frozenset(expected)] if expected else [])
    assert pages == delta
    assert store._native._projection_delta is None and store._native._projection_certificate is not None
    if operation in ("replace", "revive"):
        assert store.read_record("prior-0").value == Fraction(2, 7)
        assert store.read_record("prior-0", version=1).value == Fraction(0, 7)
        assert store.get_record("prior-0", as_of=10.5).version == 1
        assert store.query(labels_any=["new"])[0].version == 2
        between = store.get_record("prior-0", as_of=15.)
        assert (None if between is None else between.version) == (None if operation == "revive" else 1)
    elif operation == "delete":
        assert store.get_record("prior-0") is None
        assert store.read_record("prior-0", version=1).value == Fraction(0, 7)
    store.list(limit=1)
    assert len(delta) == int(bool(expected)) and full == []


def test_multiple_provisional_writes_accumulate_only_distinct_addresses(store, monkeypatch):
    full, delta, pages = audit_work(store, monkeypatch)
    with store.write_scope():
        store.put_record("prior-0", None, tx_time=20.)
        store.put_record("prior-0", Fraction(1, 7), tx_time=20.)
        store.delete("prior-0", tx_time=21.)
        store.put_record("prior-0", Fraction(2, 7), tx_time=22.)
        store.put_record("new", None, tx_time=22.)
        assert full == delta == []
        assert store.get_record("prior-0").version == 4
    store.list(limit=1)
    expected = frozenset({("prior-0", version) for version in range(1, 5)} | {("new", 1)})
    assert full == [] and delta == pages == [expected]
    assert store.read_record("prior-0").value == Fraction(2, 7)
    assert store.get_record("prior-0", as_of=21.5) is None


def test_many_changed_addresses_use_bounded_sql_parameter_pages(store, monkeypatch):
    import sqlite3
    full, delta, pages = audit_work(store, monkeypatch)
    with store.write_scope():
        for index in range(600):
            store.put_record(f"new-{index}", None, tx_time=20.)
    with store.engine.connect() as connection:
        raw = connection.connection.driver_connection
        setlimit = getattr(raw, "setlimit", None)
        previous = None if setlimit is None else setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 600)
    try:
        assert len(store.list(limit=1)) == 1
    finally:
        if setlimit is not None:
            setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)
    assert full == [] and len(delta) == 1 and len(delta[0]) == 600
    assert [len(page) for page in pages] == [256, 256, 88]


@pytest.mark.parametrize("operation", ["list", "query"])
def test_indexed_reads_inside_write_scope_audit_pending_changes(store, monkeypatch, operation):
    full, delta, pages = audit_work(store, monkeypatch)
    with store.write_scope():
        store.put_record("new", None, meta={"vertex_labels": ["wanted"]})
        assert delta == []
        result = store.list(limit=1) if operation == "list" else store.query(limit=1, labels_any=["wanted"])
        assert [record.id for record in result] == ["new"]
        assert delta == [frozenset({("new", 1)})] and full == []
    store.list(limit=1)
    assert len(delta) == 1 and full == []


@pytest.mark.parametrize("operation", ["list", "query"])
def test_corrupt_projection_inside_write_scope_is_refused_before_indexed_reads(store, monkeypatch, operation):
    full, delta, pages = audit_work(store, monkeypatch)
    with pytest.raises(ValueError, match="projection"):
        with store.write_scope():
            store.put_record("new", None)
            store._sql_connection.execute(store.table.update().where(store.table.c.id == "prior-0").values(meta="{}"))
            if operation == "list": store.list(limit=1)
            else: store.query(limit=1, labels_any=["shared"])
    assert len(full) == 1 and delta == []
    assert store._native.state is store._native._projection_delta is store._native._projection_certificate is None
    assert store.get_record("new") is None
    assert store.get_record("prior-0").meta["vertex_labels"] == ["shared", "term-0"]


def test_rollback_after_a_provisional_incremental_audit_discards_its_certificate(store, monkeypatch):
    full, delta, pages = audit_work(store, monkeypatch)
    with pytest.raises(OSError, match="abort"):
        with store.write_scope():
            store.put_record("new", None)
            assert store.list(limit=1)[0].id == "new"
            assert store._native._projection_certificate is not None
            raise OSError("abort")
    assert full == [] and delta == [frozenset({("new", 1)})]
    assert store._native.state is store._native._projection_delta is store._native._projection_certificate is None
    assert store.get_record("new") is None and len(full) == 1


@pytest.mark.parametrize("location", ["before-write", "after-write", "rolled-back-side-write"])
def test_unaccounted_same_connection_changes_force_full_audit(store, monkeypatch, location):
    full, delta, pages = audit_work(store, monkeypatch)
    with store.write_scope():
        connection = store._sql_connection
        if location == "before-write":
            connection.execute(store.table.update().where(store.table.c.id == "prior-1").values(meta="{}"))
        elif location == "rolled-back-side-write":
            nested = connection.begin_nested()
            connection.execute(store.table.update().where(store.table.c.id == "prior-1").values(meta="{}"))
            nested.rollback()
        store.put_record("new", None)
        if location == "after-write":
            connection.execute(store.table.update().where(store.table.c.id == "prior-1").values(meta="{}"))
    if location == "rolled-back-side-write":
        assert len(store.list(limit=1)) == 1
    else:
        with pytest.raises(ValueError, match="projection"): store.list(limit=1)
    assert len(full) == 1 and delta == [] and store._native._projection_delta is None


@pytest.mark.parametrize("timing", ["during-publication", "after-nomination"])
@pytest.mark.parametrize("damage", ["old-metadata", "new-metadata", "old-label", "new-label", "missing-new", "extra-row"])
def test_side_effects_cannot_certify_unchecked_rows(store, monkeypatch, timing, damage):
    full, delta, pages = audit_work(store, monkeypatch)
    def damage_rows(connection):
        if damage.endswith("metadata"):
            address = "prior-0" if damage.startswith("old") else "new"
            connection.execute(store.table.update().where(store.table.c.id == address).values(meta="{}"))
        elif damage.endswith("label"):
            address = "prior-0" if damage.startswith("old") else "new"
            connection.execute(store.labels_table.delete().where(store.labels_table.c.id == address))
        elif damage == "missing-new":
            connection.execute(store.table.delete().where(store.table.c.id == "new"))
        else:
            row = dict(connection.execute(sa.select(store.table).where(store.table.c.id == "new")).one()._mapping)
            row["id"] = "unpublished"
            connection.execute(store.table.insert().values(**row))
    if timing == "during-publication":
        original = store._native.journal.publish
        def publish(connection, *args, **kwargs):
            result = original(connection, *args, **kwargs)
            damage_rows(connection)
            return result
        monkeypatch.setattr(store._native.journal, "publish", publish)
    store.put_record("new", None, meta={"vertex_labels": ["new"]})
    if timing == "after-nomination":
        with store.engine.begin() as connection:
            damage_rows(connection)
    with pytest.raises(ValueError, match="projection|label index"): store.list(limit=1)
    assert len(full) == 1 and delta == []
    assert store._native._projection_certificate is store._native._projection_delta is None


@pytest.mark.parametrize("damage", ["meta", "signature", "record_envelope", "source", "tx_to", "label"])
def test_incremental_audit_checks_written_values_without_trusting_the_write(store, monkeypatch, damage):
    full, delta, pages = audit_work(store, monkeypatch)
    from sqlalchemy.sql.dml import Insert, Update
    target = store.labels_table if damage == "label" else store.table
    operation = Update if damage == "tx_to" else Insert
    def alter(connection, cursor, statement, parameters, context, executemany):
        compiled = context.compiled
        if compiled is None or not isinstance(compiled.statement, operation) or compiled.statement.table is not target:
            return statement, parameters
        position = compiled.positiontup.index(damage)
        replacement = 9. if damage == "tx_to" else "wrong" if damage in ("source", "label") else "{}"
        def changed(row):
            result = list(row); result[position] = replacement
            return tuple(result)
        return statement, [changed(row) for row in parameters] if executemany else changed(parameters)
    sa.event.listen(store.engine, "before_cursor_execute", alter, retval=True)
    try:
        store.put_record("prior-0", None, tx_time=20., meta={"vertex_labels": ["new"]})
    finally:
        sa.event.remove(store.engine, "before_cursor_execute", alter)
    assert store._native._projection_delta is not None  # Exact DML budget still agrees.
    with pytest.raises(ValueError, match="projection|label index"): store.list(limit=1)
    assert full == [] and delta == [frozenset({("prior-0", 1), ("prior-0", 2)})]
    assert store._native._projection_certificate is store._native._projection_delta is None
    with pytest.raises(ValueError, match="projection|label index"): store.list(limit=1)
    assert len(full) == 1


@pytest.mark.parametrize("temporary", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
def test_main_and_temp_triggers_always_keep_full_audits(store, monkeypatch, temporary, corrupt):
    with store.engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE trigger_sink(value INTEGER)")
        statement = ("UPDATE rc_complexes SET meta = '{}' WHERE id = 'prior-0'" if corrupt
                     else "INSERT INTO trigger_sink VALUES (1)")
        connection.exec_driver_sql(f"CREATE {'TEMP ' if temporary else ''}TRIGGER side_effect "
            f"AFTER INSERT ON rc_complexes BEGIN {statement}; END")
    store.list(limit=1)
    full, delta, pages = audit_work(store, monkeypatch)
    store.put_record("new", None)
    assert store._native._projection_delta is None
    if corrupt:
        with pytest.raises(ValueError, match="projection"): store.list(limit=1)
    else:
        assert store.get_record("new").version == 1
    assert len(full) == 1 and delta == []


def test_temp_schema_change_between_nomination_and_read_invalidates_it(store, monkeypatch):
    full, delta, pages = audit_work(store, monkeypatch)
    store.put_record("new", None)
    assert store._native._projection_delta is not None
    with store.engine.begin() as connection:
        connection.exec_driver_sql("CREATE TEMP TABLE temp_probe(value INTEGER)")
    store.get_record("new")
    assert len(full) == 1 and delta == []


def test_outer_and_caught_nested_failures_discard_every_provisional_change(store, monkeypatch):
    full, delta, pages = audit_work(store, monkeypatch)
    with pytest.raises(RuntimeError, match="aborted"):
        with store.write_scope():
            store.put_record("new", None)
            try:
                store.put_record("prior-0", None, tx_time=5.)
            except ValueError:
                pass
    assert store._native._projection_delta is store._native._projection_certificate is None
    assert store.get_record("new") is None
    assert len(full) == 1 and delta == []


def test_connection_replacement_never_transfers_a_provisional_nomination(tmp_path, monkeypatch):
    with closing(SQLStore(f"sqlite:///{tmp_path}/db.sqlite")) as store:
        store.put_record("prior", None); store.list(limit=1)
        full, delta, pages = audit_work(store, monkeypatch)
        store.put_record("new", None)
        with store.engine.connect() as held:
            assert held.connection.driver_connection is store._native._projection_delta[0]
            assert store.get_record("new").version == 1
        assert len(full) == 1 and delta == []


def test_other_handle_native_change_cannot_extend_a_local_nomination(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/db.sqlite"
    with closing(SQLStore(uri)) as store, closing(SQLStore(uri)) as other:
        store.list(limit=1)
        full, delta, pages = audit_work(store, monkeypatch)
        store.put_record("new", None)
        other.put_record("peer", Fraction(2, 7))
        assert store.read_record("peer").value == Fraction(2, 7)
        assert len(full) == 1 and delta == []


def test_exact_transfer_rcql_and_system_share_incrementally_checked_sql(store, monkeypatch):
    rcql = pytest.importorskip("rcql")
    pytest.importorskip("system")
    from rexgraph import RexGraph
    from system.inspection import source_row
    full, delta, pages = audit_work(store, monkeypatch)
    with closing(MemoryStore()) as source:
        source.put("graph", RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)]), analytics=False)
        source.put_record("exact", {"q": Fraction(2, 7)})
        source_meta = pack_value(source.get_record("exact").meta)
        copy_record(source, store, source.get_record("graph"))
        copy_record(source, store, source.get_record("exact"))
    assert store.get("graph").edge_metric_exact == [Fraction(1, 7)]
    assert store.read_record("exact").value == {"q": Fraction(2, 7)}
    query = rcql.Executor(sources={"db": store}).execute(rcql.parse(
        'FROM RCDB_VERSION(RCDB("db"), "graph", 1) RETURN 1 / 7, RANK(1)'))
    assert query.values == (Fraction(1, 7), 1)
    assert source_row("db", store)["accessible"]
    assert full == [] and delta
    assert pack_value(store.get_record("exact").meta) == source_meta
