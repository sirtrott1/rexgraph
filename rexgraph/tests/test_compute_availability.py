import pytest

from rexgraph import compute as C


@pytest.fixture
def isolated_ops(monkeypatch):
    monkeypatch.setattr(C, "_OPS", {})
    monkeypatch.setattr(C, "_DEFAULT_BACKEND", None)
    monkeypatch.setattr(C, "_auto_backend", lambda: None)


def test_explicit_unavailable_backend_falls_back_without_calling_it(monkeypatch, isolated_ops):
    C.register_op("product", "cpu", lambda: "cpu")
    C.register_op("product", "cuda", lambda: pytest.fail("unavailable GPU invoked"))
    monkeypatch.setattr(C, "_ok", lambda b: b["name"] == "cpu")
    assert C.dispatch("product", prefer="cuda") == "cpu"


def test_no_available_implementation_is_an_error(monkeypatch, isolated_ops):
    C.register_op("product", "cuda", lambda: pytest.fail("unavailable GPU invoked"))
    monkeypatch.setattr(C, "_ok", lambda b: b["name"] == "cpu")
    with pytest.raises(RuntimeError, match="no available backend"):
        C.dispatch("product", prefer="cuda")


def test_rocm_preference_maps_to_the_registered_torch_backend(monkeypatch, isolated_ops):
    C.register_op("product", "cpu", lambda: "cpu")
    C.register_op("product", "cuda", lambda: "rocm")
    monkeypatch.setattr(C, "_ok", lambda b: b["name"] in {"cpu", "cuda"})
    assert C.dispatch("product", prefer="rocm") == "rocm"
