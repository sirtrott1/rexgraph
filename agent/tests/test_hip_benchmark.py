import json

import pytest

from agent.benchmarks import bench_hip_ternary as B


def test_cpu_report_cannot_be_mistaken_for_gpu_qualification():
    report = B.run(device="cpu", rows=7, cols=65, iterations=2)
    assert report["ok"] and not report["gpu_qualified"]
    assert report["parameters_trained"] == 0
    assert set(report["cpu"]) == {"pm1", "float64"}
    assert report["operator_bytes"] == 7 * 2 * 8 * 2


def test_hip_request_cannot_silently_time_cpu(monkeypatch, tmp_path):
    from rexgraph import hip_ternary as H
    monkeypatch.setattr(H, "available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        B.run(device="hip", rows=7, cols=65, iterations=2)
    path = tmp_path / "unavailable.json"
    assert B.main(["--output", str(path)]) == 1
    report = json.loads(path.read_text())
    assert report["ok"] is report["gpu_qualified"] is False
    with pytest.raises(SystemExit):
        B.main(["--output", str(path)])


def test_qualifier_checks_witnesses_empty_columns_and_declared_math():
    """Exercise the qualifier using the independent compiled CPU tower at its seam."""
    class CPUOracle:
        @staticmethod
        def channel_tower(bp, bi, nV, weights, *, block, coefficients):
            return B._channel_tower.channel_diagonals_any_arity(bp, bi, nV, weights, coefficients=coefficients)
    assert len(B._tower_cases(CPUOracle)) == 2


def test_actual_hip_channel_qualification():
    from rexgraph import hip_ternary as H
    if not H.available():
        pytest.skip("a HIP device and compatible kernels are required")
    assert all(case["agrees"] for case in B._tower_cases(H))
