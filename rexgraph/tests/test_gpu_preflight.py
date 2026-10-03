import numpy as np
import pytest

from rexgraph import gpu_preflight as G


def test_numerical_false_is_a_failed_qualification_and_preserves_diagnostic():
    result = G._check("test", lambda: {"max_abs_err": 42., "matches_cpu": False})
    assert result["ok"] is False
    assert result["detail"]["max_abs_err"] == 42.
    assert "matches_cpu" in result["error"]


@pytest.mark.parametrize("n", [2, 8, 64, 256])
def test_solver_fixture_is_symmetric_positive_definite(n):
    A, B = G._sparse_spd(n)
    np.testing.assert_array_equal(A.toarray(), A.T.toarray())
    assert np.linalg.eigvalsh(A.toarray()).min() >= 1.0-1e-12
    assert B.shape == (n, 4)


def test_preflight_exit_truth_cannot_pass_a_failed_numeric_test(monkeypatch):
    monkeypatch.setattr(G, "_probe_torch", lambda: {"torch": "test", "device_count": 1})
    for name in ("_check_float64_matmul", "_check_sparse_mm", "_check_block_cg", "_check_end_to_end", "_check_multi_gpu", "_check_autograd", "_check_packed_training"):
        monkeypatch.setattr(G, name, lambda n: {"matches_cpu": False})
    report = G.run(8)
    assert not report["ok"] and all(not c["ok"] for c in report["checks"])


def test_no_device_report_does_not_claim_execution(monkeypatch):
    monkeypatch.setattr(G, "_probe_torch", lambda: {"torch": "test", "device_count": 0})
    monkeypatch.setattr(G, "probe_vulkan", lambda: {"available": False, "devices": []})
    report = G.run(8)
    assert not report["ok"] and not report["checks"]
    assert "did not run" in report["verdict"]
    assert "host GPU absence" in report["next_step"]


def test_vulkan_gpu_does_not_pass_torch_training_qualification(monkeypatch):
    monkeypatch.setattr(G, "_probe_torch", lambda: {"torch": "test", "device_count": 0})
    monkeypatch.setattr(G, "probe_vulkan", lambda: {"available": True, "devices": [{"name": "AMD"}]})
    report = G.run(8)
    assert report["vulkan"]["available"] and not report["ok"]
    assert not report["checks"] and "Vulkan hardware" in report["next_step"]


def test_missing_torch_does_not_assert_that_every_gpu_backend_is_unreachable(monkeypatch):
    monkeypatch.setattr(G, "_probe_torch", lambda: {"torch": None, "device_count": 0})
    monkeypatch.setattr(G, "probe_vulkan", lambda: {"available": True, "devices": [{"name": "AMD"}]})
    report = G.run(8)
    assert report["vulkan"]["available"] and not report["ok"]
    assert "Torch GPU qualification did not run" in report["verdict"]


def test_diagnostic_enumeration_never_runs_or_claims_numerical_qualification(monkeypatch, capsys):
    report = {"hip": {"available": False}, "vulkan": {"available": True},
              "numerical_qualification": False}
    monkeypatch.setattr(G, "diagnose", lambda: report)
    monkeypatch.setattr(G, "run", lambda size: pytest.fail("enumeration ran training qualification"))
    assert G.main(["--diagnose", "--json"]) == 0
    import json
    assert json.loads(capsys.readouterr().out)["numerical_qualification"] is False
