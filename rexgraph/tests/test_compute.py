"""The compute backend layer (rexgraph.compute): backend registry + availability, thread control,
and op dispatch to the best available backend with fallback."""
import rexgraph.compute as C


def test_builtin_backends_present():
    names = {b["name"] for b in C.backends()}
    assert {"cpu", "openmp", "cuda", "mps"} <= names
    assert "cpu" in C.available_backends()                   # cpu is always available
    kinds = {b["name"]: b["kind"] for b in C.backends()}
    assert kinds["cpu"] == "cpu" and kinds["cuda"] == "gpu"


def test_best_backend_prefers_available_then_falls_back_to_cpu():
    assert C.best_backend(prefer="cpu") == "cpu"
    assert C.best_backend(prefer="does-not-exist") in C.available_backends()  # unknown -> best real
    # with no GPU present best_backend is cpu; with a GPU it is that GPU - either way it is available
    assert C.best_backend() in C.available_backends()


def test_thread_control_roundtrip():
    old = C.get_threads()
    C.set_threads(3)
    assert C.get_threads() == 3
    C.set_threads(None)
    assert C.get_threads() is None
    if old is not None:
        C.set_threads(old)


def test_op_dispatch_routes_and_falls_back():
    C.register_op("_t_double", "cpu", lambda x: x * 2)
    C.register_op("_t_double", "cuda", lambda x: x * 20)     # a "gpu" impl
    # prefer cpu -> cpu impl
    assert C.dispatch("_t_double", 5, prefer="cpu") == 10
    # prefer a backend with no impl for this op -> falls through to an available one
    assert C.dispatch("_t_double", 5, prefer="mps") in (10, 100)
    # unknown op raises
    try:
        C.dispatch("_t_nope", 1); assert False
    except KeyError:
        pass


def test_inventory_shape():
    inv = C.inventory()
    assert {"backends", "threads", "ops"} <= set(inv)
    assert any(b["name"] == "cpu" for b in inv["backends"])


def test_parallel_map_order_edges_and_serial():
    # order preserved, GIL-releasing or not
    assert C.parallel_map(lambda x: x * x, [1, 2, 3, 4], threads=2) == [1, 4, 9, 16]
    assert C.parallel_map(lambda x: x + 1, [], threads=4) == []            # empty
    assert C.parallel_map(lambda x: x + 1, [5], threads=4) == [6]          # single -> serial path
    assert C.parallel_map(lambda x: x * 2, range(4), threads=1) == [0, 2, 4, 6]  # forced serial


def test_default_backend_steers_dispatch_and_apply_config():
    C.register_op("_t_pick", "cpu", lambda: "cpu")
    C.register_op("_t_pick", "openmp", lambda: "openmp")
    C.set_default_backend(None)
    C.register_op("_t_pick", "cuda", lambda: "cuda")
    # a set default backend is preferred by dispatch when the op implements it
    C.set_default_backend("openmp")
    assert C.dispatch("_t_pick") == "openmp"
    C.set_default_backend(None)                              # cleared -> best available / cpu path
    assert C.dispatch("_t_pick", prefer="cpu") == "cpu"
    # apply_config sets both knobs and reports the effective config
    eff = C.apply_config({"threads": 2, "backend": "openmp"})
    assert eff["threads"] == 2 and eff["backend"] == "openmp"
    assert C.dispatch("_t_pick") == "openmp"
    C.apply_config({"threads": None, "backend": "auto"})     # reset
    assert C.get_default_backend() is None


def test_parallel_map_order_and_bit_identity():
    """parallel_map is order-preserving and bit-identical to a serial map, regardless of
    worker/inner-thread config (thread count never changes results)."""
    import numpy as np
    fn = lambda i: float(np.sin(np.linspace(0, 1, 500) + i).sum())
    items = list(range(37))
    serial = [fn(i) for i in items]
    for th in (1, 3, 8, None):
        got = C.parallel_map(fn, items, threads=th)
        assert got == serial                                  # exact, order-preserving


def test_parallel_map_inner_thread_budget_arithmetic():
    """The inner-threadpool cap is BUDGET ARITHMETIC (max(1, budget // workers)), not a
    fixed threshold: with a budget of B and W workers, the limiter is asked for ~B//W. We
    assert the resolved (workers, inner) split tracks the budget rather than oversubscribing."""
    captured = {}
    import contextlib

    @contextlib.contextmanager
    def _spy(inner):
        captured["inner"] = inner
        yield

    orig = C._inner_thread_limiter
    C._inner_thread_limiter = _spy
    try:
        C.set_threads(16)                                     # budget = 16
        C.parallel_map(lambda x: x, range(4))                 # 4 workers -> inner ~ 16//4 = 4
        assert captured["inner"] == 4
        C.parallel_map(lambda x: x, range(64))                # workers capped at budget 16 -> inner 1
        assert captured["inner"] == 1
        captured.clear()
        C.parallel_map(lambda x: x, range(8), inner_threads=0)  # explicit uncapped override
        assert captured["inner"] == 0
    finally:
        C._inner_thread_limiter = orig
        C.set_threads(None)


def test_inner_thread_limiter_graceful_without_threadpoolctl(monkeypatch):
    """_inner_thread_limiter degrades to a no-op context when threadpoolctl is unavailable,
    so the fan-out still runs (just uncapped) - the cap is an optimization, never a hard dep."""
    import builtins
    real_import = builtins.__import__

    def _no_tpc(name, *a, **k):
        if name == "threadpoolctl":
            raise ImportError("simulated missing threadpoolctl")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_tpc)
    with C._inner_thread_limiter(1):                          # must not raise
        pass
    assert C.parallel_map(lambda x: x * 2, range(5), threads=4) == [0, 2, 4, 6, 8]
