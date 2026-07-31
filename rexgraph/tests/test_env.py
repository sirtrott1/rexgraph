"""rexgraph._env: host/environment detection: Python env manager, compute backends, and the
per-host backend recommendation. These tests must pass on ANY machine (including CI with no GPU
and no toolchain), so they assert on STRUCTURE and invariants, never on specific hardware."""
import os

import rexgraph._env as E


def test_detect_python_env_wellformed():
    env = E.detect_python_env()
    assert isinstance(env, dict)
    # required, always-present keys
    for k in ("manager", "python", "prefix", "in_venv", "compiler", "has_toolchain", "warnings"):
        assert k in env, f"missing key {k!r}"
    assert env["manager"] in {
        "conda", "mamba", "micromamba", "venv", "virtualenv", "uv", "poetry", "pdm", "system",
    }
    assert isinstance(env["in_venv"], bool)
    assert isinstance(env["warnings"], list)
    assert isinstance(env["compiler"], dict)
    # compiler sub-report is well-formed even when no compiler exists
    for k in ("env_cc", "env_version", "system_cc", "system_version", "consistent", "warning"):
        assert k in env["compiler"]
    assert isinstance(env["compiler"]["consistent"], bool)


def test_detect_compute_backends_wellformed_and_cpu_present():
    backends = E.detect_compute_backends()
    assert isinstance(backends, list) and backends, "must always return at least CPU"
    names = [b["name"] for b in backends]
    assert "cpu" in names, "CPU must always be available"
    for b in backends:
        assert isinstance(b, dict)
        for k in ("name", "kind", "available", "integrated", "vendor", "via", "detail"):
            assert k in b, f"backend {b.get('name')!r} missing key {k!r}"
        assert b["available"] is True                      # only available backends are returned
        assert b["kind"] in {"cpu", "gpu"}
        assert isinstance(b["integrated"], bool)
    # cpu entry is the CPU kind and reports a core count
    cpu = next(b for b in backends if b["name"] == "cpu")
    assert cpu["kind"] == "cpu"
    assert cpu.get("cores", 0) >= 1


def test_recommend_backend_is_a_member_of_available():
    backends = E.detect_compute_backends()
    names = {b["name"] for b in backends}
    # do not let a stray env override skew this invariant check
    old = os.environ.pop(E.REXGRAPH_BACKEND_ENV, None)
    try:
        rec = E.recommend_backend(backends)
        assert rec in names, f"recommended {rec!r} not in available {names}"
        # calling with no argument (auto-detect) also yields something sane
        assert isinstance(E.recommend_backend(), str)
        # accepts a plain list of names too
        assert E.recommend_backend(list(names)) in names
    finally:
        if old is not None:
            os.environ[E.REXGRAPH_BACKEND_ENV] = old


def test_recommend_backend_env_override_wins():
    old = os.environ.get(E.REXGRAPH_BACKEND_ENV)
    try:
        os.environ[E.REXGRAPH_BACKEND_ENV] = "cpu"
        assert E.recommend_backend() == "cpu"
        os.environ[E.REXGRAPH_BACKEND_ENV] = "CUDA"        # normalized, wins verbatim
        assert E.recommend_backend(["cpu"]) == "cuda"
    finally:
        if old is None:
            os.environ.pop(E.REXGRAPH_BACKEND_ENV, None)
        else:
            os.environ[E.REXGRAPH_BACKEND_ENV] = old


def test_recommend_backend_falls_back_to_cpu():
    old = os.environ.pop(E.REXGRAPH_BACKEND_ENV, None)
    try:
        assert E.recommend_backend([]) == "cpu"            # empty availability -> cpu
        assert E.recommend_backend(["cpu"]) == "cpu"
    finally:
        if old is not None:
            os.environ[E.REXGRAPH_BACKEND_ENV] = old


def test_summary_runs_and_mentions_backends():
    old = os.environ.pop(E.REXGRAPH_BACKEND_ENV, None)
    try:
        s = E.summary()
        assert isinstance(s, str) and s
        assert "Compute backends" in s
        assert "recommended" in s
    finally:
        if old is not None:
            os.environ[E.REXGRAPH_BACKEND_ENV] = old


def test_nothing_raises_repeatedly():
    # detection must be safe to call many times (caching / probes must not accumulate state)
    for _ in range(3):
        E.detect_python_env()
        E.detect_compute_backends()
        E.recommend_backend()
