"""How the agent turns the machine's hardware into a model budget.

The probe itself lives in rexgraph.hardware and is tested beside it. What is here is
the agent's own reasoning on top: which device will actually run the work, and how
much of memory a model may claim. A unified carve out is not a budget: on a host whose
GPU memory IS system memory, treating the carve out as the budget hands back a number
far smaller than the machine can actually hold.
"""
import pytest
from agent.local_runtime import _compute_gpu, detect_hardware


@pytest.fixture
def no_hip(monkeypatch):
    from agent import local_runtime as R
    monkeypatch.setattr(R, "probe_hip", lambda: {"available": False, "devices": []})


@pytest.fixture
def linux_host(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")


def test_the_compute_gpu_is_the_card_not_the_igpu():
    """A 9950X3D has integrated graphics AND whatever is in the slot. The work goes to
    the card, so the card's memory is what the work contends for."""
    igpu = {"vendor": "amd", "vram_gb": 4.0, "unified": True}
    card = {"vendor": "amd", "vram_gb": 24.0, "unified": False}
    assert _compute_gpu([igpu, card]) is card
    assert _compute_gpu([card, igpu]) is card
    assert _compute_gpu([igpu]) is igpu
    assert _compute_gpu([]) is None


def _assert_unified_budget(hw):
    """What a unified host has to satisfy, in either shape. Shared deliberately: the
    Apple case is driven from a fixture below, and if these rules only lived in the
    host reading test then a rule that breaks on the no carveout shape would again be
    invisible until CI reached a mac."""
    gpu = hw.get("gpu") or {}
    assert gpu.get("unified"), hw
    assert hw["model_budget_gb"] == pytest.approx(round(hw["ram_gb"] * 0.75, 1))
    if gpu.get("vram_gb"):
        assert hw["model_budget_gb"] > gpu["vram_gb"] * 2, hw


def test_a_unified_carveout_does_not_become_the_model_budget():
    """The bug filling vram_gb would have introduced: `if vram: budget = vram` would have
    called this 121 GiB machine a 4 GiB one, a 30x understatement, and every model fit
    decision downstream would have refused everything.

    Unified does NOT imply a carveout, which is the trap: an APU reports one, Apple
    silicon reports no vram at all because there is no separate pool to carve from.
    So the RAM rule is read first, on every unified host, and the carveout comparison
    only where a carveout exists. Asserting the second unconditionally multiplied
    None on the macOS runners.
    """
    hw = detect_hardware()
    if not (hw.get("gpu") or {}).get("unified"):
        pytest.skip("not a unified-memory host")
    _assert_unified_budget(hw)


def test_a_unified_host_reporting_no_carveout_still_budgets_from_ram(monkeypatch, no_hip):
    """The Apple shape, driven from a fixture so it is checked on every platform.

    This is the one that took the macOS runners down, and it was invisible here
    because it needs hardware none of us has to hand. Standing the host up as Apple
    silicon reproduces it in one call, so the next reader who treats `unified` as
    implying a vram figure fails on Linux rather than in CI.
    """
    import platform
    import shutil

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)

    hw = detect_hardware()
    _assert_unified_budget(hw)          # the same rules, on the no carveout shape
    gpu = hw["gpu"]
    assert gpu["unified"] is True
    assert gpu["vram_gb"] is None, "no separate pool means there is no figure"
    assert "metal" in hw["backends"] and hw["os"] == "Darwin"
    # the budget still comes from RAM, and RAM itself resolves on Darwin through
    # sysconf rather than /proc/meminfo
    assert hw["ram_gb"] > 0
    assert hw["model_budget_gb"] == pytest.approx(round(hw["ram_gb"] * 0.75, 1))
    # and the probe agrees: the pool it can address is all of system memory
    apple = [g for g in hw["gpus"] if g["vendor"] == "apple"]
    assert len(apple) == 1 and apple[0]["gtt_gb"] == pytest.approx(hw["ram_gb"], rel=0.01)


def test_detect_hardware_still_reports_its_old_keys():
    hw = detect_hardware()
    for k in ("os", "backends", "gpu", "ram_gb", "model_budget_gb", "recommended_backend"):
        assert k in hw, k
    assert "gpus" in hw and isinstance(hw["gpus"], list)


def test_vulkan_device_is_detected_without_vulkaninfo_and_preserves_unified_budget(monkeypatch, no_hip):
    from agent import local_runtime as R
    igpu = {"vendor": "amd", "vram_gb": 4., "unified": True}
    monkeypatch.setattr(R.shutil, "which", lambda *args: None)
    monkeypatch.setattr(R, "detect_gpus", lambda: [igpu])
    monkeypatch.setattr(R, "probe_vulkan", lambda: {"available": True, "devices": [{
        "hardware_gpu": True, "compute": True, "integrated": True,
        "vendor_id": 0x1002, "name": "Radeon 8060S"}]})
    monkeypatch.setattr(R, "_ram_gb", lambda: 120.)
    hw = R.detect_hardware()
    assert hw["recommended_backend"] == "vulkan" and hw["gpu"] is igpu
    assert hw["model_budget_gb"] == 90.


def test_vulkaninfo_executable_alone_does_not_select_gpu_backend(monkeypatch, no_hip, linux_host):
    from agent import local_runtime as R
    monkeypatch.setattr(R.shutil, "which", lambda name: '/usr/bin/vulkaninfo' if name == 'vulkaninfo' else None)
    monkeypatch.setattr(R, "detect_gpus", lambda: [])
    monkeypatch.setattr(R, "probe_vulkan", lambda: {"available": False, "runtime_loaded": True, "devices": []})
    assert R.detect_hardware()["recommended_backend"] == "cpu"


def test_vulkan_enumeration_can_supply_gpu_identity_without_sysfs(monkeypatch, no_hip):
    from agent import local_runtime as R
    monkeypatch.setattr(R.shutil, "which", lambda name: None)
    monkeypatch.setattr(R, "detect_gpus", lambda: [])
    monkeypatch.setattr(R, "probe_vulkan", lambda: {"available": True, "devices": [{
        "hardware_gpu": True, "compute": True, "integrated": True,
        "vendor_id": 0x1002, "name": "Radeon 8060S"}]})
    hw = R.detect_hardware()
    assert hw["gpu"]["vendor"] == "amd" and hw["gpu"]["name"] == "Radeon 8060S"
    assert hw["gpu"]["unified"] is None


def test_vulkan_memory_heap_enables_auto_offload_without_sysfs(monkeypatch, no_hip):
    from agent import local_runtime as R
    monkeypatch.setattr(R.shutil, "which", lambda name: None)
    monkeypatch.setattr(R, "detect_gpus", lambda: [])
    monkeypatch.setattr(R, "probe_vulkan", lambda: {"available": True, "devices": [{
        "hardware_gpu": True, "compute": True, "integrated": True,
        "vendor_id": 0x1002, "name": "Radeon", "device_local_bytes": 32 * 2**30}]})
    monkeypatch.setattr(R.os.path, "getsize", lambda path: 2**30)
    assert R.detect_hardware()["model_budget_gb"] == 32.
    assert R._auto_ngl('/model.gguf') == 999


def test_hip_is_detected_without_rocminfo_and_uses_existing_memory_probe(monkeypatch):
    from agent import local_runtime as R
    igpu = {"vendor": "amd", "vram_gb": 4., "unified": True}
    monkeypatch.setattr(R.shutil, "which", lambda name: None)
    monkeypatch.setattr(R, "detect_gpus", lambda: [igpu])
    monkeypatch.setattr(R, "probe_hip", lambda: {"available": True, "devices": [{"name": "AMD"}]})
    monkeypatch.setattr(R, "probe_vulkan", lambda: {"available": False, "devices": []})
    monkeypatch.setattr(R, "_ram_gb", lambda: 120.)
    hw = R.detect_hardware()
    assert hw["recommended_backend"] == "rocm" and hw["gpu"] is igpu
    assert hw["model_budget_gb"] == 90.


def test_rocminfo_executable_alone_does_not_select_rocm(monkeypatch, no_hip, linux_host):
    from agent import local_runtime as R
    monkeypatch.setattr(R.shutil, "which", lambda name: '/usr/bin/rocminfo' if name == 'rocminfo' else None)
    monkeypatch.setattr(R, "detect_gpus", lambda: [])
    monkeypatch.setattr(R, "probe_vulkan", lambda: {"available": False, "devices": []})
    assert R.detect_hardware()["recommended_backend"] == "cpu"
