"""Native records and provider locks preserve the existing workspace boundary."""
from contextlib import closing
from fractions import Fraction

import pytest

from rcdb import LocalStore, MemoryStore, NativeObjectStore, SQLStore
from agent.server.scope import ScopedStore


def opened(kind, root):
    if kind == "memory": return MemoryStore()
    if kind == "local": return LocalStore(root)
    if kind.startswith("object-"):
        pytest.importorskip("fsspec")
        scheme = "memory" if kind == "object-memory" else "file"
        return NativeObjectStore(f"{scheme}://{root}")
    pytest.importorskip("sqlalchemy")
    return SQLStore(f"sqlite:///{root}.sqlite")


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_replay_maintenance_requires_an_unscoped_operator(kind, tmp_path):
    with closing(opened(kind, tmp_path / "store")) as inner:
        with pytest.raises(PermissionError, match="unscoped"):
            ScopedStore(inner, "alpha").checkpoint()
        with pytest.raises(PermissionError, match="unscoped"):
            ScopedStore(inner, "alpha").plan_retention()
        with pytest.raises(PermissionError, match="unscoped"):
            ScopedStore(inner, "alpha").apply_retention(None)
        assert inner.change_cursor.sequence == 0


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_typed_workspace_writes_stamp_ownership_audit_and_refuse_other_owner(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path / "store")) as inner:
        actions = []
        from agent.server import audit
        monkeypatch.setattr(audit, "record", lambda *args, **kwargs: actions.append((args, kwargs)))
        alpha, beta = ScopedStore(inner, "alpha", "alice"), ScopedStore(inner, "beta", "bob")
        record = alpha.put_record("value", Fraction(1, 7), meta={"workspace": "beta", "stored_by": "bob"})
        assert record.meta == {"workspace": "alpha", "stored_by": "alice"}
        assert alpha.read_record("value").value == Fraction(1, 7)
        assert beta.get("value") is None and beta.read_record("value") is None
        assert beta.get_version("value", 1) is None
        with pytest.raises(PermissionError, match="another workspace"):
            beta.put_record("value", None)
        assert inner.change_cursor.sequence == 1
        beta.put_record("null", None)
        assert beta.read_record("null") is not None and alpha.read_record("null") is None
        assert [r["id"] for r in beta.state_manifest()["records"]] == ["null"]
        with pytest.raises(PermissionError, match="unscoped"):
            beta.changes()
        assert actions[0][1]["workspace"] == "alpha"
        assert actions[0][1]["detail"]["record_type"] == "NativeValue"


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_metadata_projection_cannot_silently_discard_workspace_ownership(kind, tmp_path, monkeypatch):
    with closing(opened(kind, tmp_path / "store")) as inner:
        from agent.server import audit
        monkeypatch.setattr(audit, "record", lambda *args, **kwargs: None)
        inner.configure_security(metadata_fields=[])
        scoped = ScopedStore(inner, "alpha", "alice")
        with pytest.raises(PermissionError, match="discard workspace"):
            scoped.put_record("private", Fraction(1, 7))
        assert inner.change_cursor.sequence == 0
        inner.configure_security(metadata_fields=["workspace"])
        assert scoped.put_record("private", Fraction(1, 7)).meta == {"workspace": "alpha"}


@pytest.mark.parametrize("kind", ["local", "sql", "object-file", "object-memory"])
def test_an_independent_native_handle_cannot_reassign_an_owned_record(kind, tmp_path, monkeypatch):
    root = tmp_path / "shared"
    with closing(opened(kind, root)) as first, closing(opened(kind, root)) as second:
        from agent.server import audit
        monkeypatch.setattr(audit, "record", lambda *args, **kwargs: None)
        ScopedStore(first, "alpha", "alice").put_record("owned", Fraction(1, 7))
        with pytest.raises(PermissionError, match="another workspace"):
            ScopedStore(second, "beta", "bob").put_record("owned", Fraction(2, 7))
        assert first.read_record("owned").value == Fraction(1, 7)


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_exact_version_scope_filters_before_decode_without_cloning_history(kind, tmp_path, monkeypatch):
    from rcdb import copy_record, record_packet
    from rcql import Executor, parse
    with closing(opened(kind, tmp_path/"store")) as inner, closing(MemoryStore()) as target:
        hidden = inner.put_record("r@1", None, tx_time=1., meta={"workspace": "alpha"})
        inner.put_record("r@1", Fraction(2, 7), tx_time=2., meta={"workspace": "beta"})
        inner.put_record("public", None, tx_time=1.)
        inner.put_record("other", Fraction(3, 7), tx_time=1., meta={"workspace": "alpha"})
        inner.delete("other", tx_time=2.)
        view = ScopedStore(inner, "beta", "bob")
        nested = ScopedStore(view, "alpha", "alice")
        decoded, original = [], inner.get_version
        def decode(rid, version):
            decoded.append((rid, version))
            assert (rid, version) not in {(hidden.id, 1), ("other", 1)}
            return original(rid, version)
        def forbidden(*args, **kwargs): pytest.fail("scoped native exact read cloned history")
        monkeypatch.setattr(inner, "get_version", decode)
        monkeypatch.setattr(inner, "history", forbidden)
        assert view.read_record("r@1", version=1) is None
        assert view.get_version("r@1", 1) is None
        assert view.read_record("other", version=1) is None
        assert view.read_record("r@1", version=3) is None
        assert record_packet(view, "r@1", version=1) is None
        assert copy_record(view, target, hidden) is None
        assert not decoded and target.stats()["n_versions"] == 0
        assert view.read_record("r@1", version=2).value == Fraction(2, 7)
        assert view.get_version("r@1", 2) == Fraction(2, 7)
        assert view.read_record("public", version=1).value is None
        assert nested.read_record("r@1", version=2) is None
        assert nested.read_record("public", version=1).value is None
        with pytest.raises(KeyError, match="not present"):
            Executor(sources={"db": view}).execute(parse(
                'FROM RCDB_VERSION($db,"r@1",1) RETURN RCDB_STATE_HASH()'))
