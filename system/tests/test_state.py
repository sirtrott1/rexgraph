from system.state import SourceStore


def test_source_store_registers_and_removes_values():
    store = SourceStore()
    value = object()
    store.register("main", value)
    assert store.get("main") is value
    assert store.snapshot()["main"] is value
    store.remove("main")
    assert "main" not in store.snapshot()
