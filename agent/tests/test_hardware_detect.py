"""Hardware detection has to report what sysfs knows, not None.

`detect_hardware` returned {"vendor":"amd","vram_gb":None,"unified":None} on a machine
where sysfs reports both, so the coordinator could not auto-configure its bus topology
and every caller had to declare it by hand.

The functional test for unified memory is whether the GPU can address system memory: an
APU's GTT covers essentially all of RAM because its "VRAM" is a carveout OF that RAM.
Unified comes in two shapes, and the difference is load bearing: an APU reports a
carveout alongside the full GTT, Apple silicon reports no vram at all because there is
no second pool to carve. Reading `unified` as "there is a vram figure" is wrong on the
second.
Measured here on a Strix Halo 8060S: gtt exactly equals MemTotal with vram a 4 GiB
carveout. The discrete branch is REASONED, not measured, because there is no dGPU on
this machine, which is why an inconclusive read returns None instead of a guess.
"""
import pytest
from agent.local_runtime import (
    _compute_gpu,
    _unified_from_drm,
    detect_gpus,
    detect_hardware,
    drm_devices,
)

RAM = 130452873216      # this machine, for the fixture cases


def test_drm_devices_reads_or_returns_nothing():
    """No exception on a machine with no GPU, no partial dicts."""
    for d in drm_devices():
        assert set(d) == {"path", "driver", "pci_id", "vram_bytes", "gtt_bytes"}
        assert d["vram_bytes"] is not None or d["gtt_bytes"] is not None


@pytest.mark.parametrize("vram_gb", [4, 8, 16])
def test_a_carveout_with_full_gtt_reads_as_unified(vram_gb):
    """The APU shape at several carveout settings: a small pool relative to the RAM it is
    taken from, and a GTT covering all of system memory."""
    dev = {"vram_bytes": vram_gb * 1024 ** 3, "gtt_bytes": RAM}
    unified, why = _unified_from_drm(dev, RAM)
    assert unified is True, why
    assert "system memory" in why


@pytest.mark.parametrize("vram_gb,ram_gb", [(24, 64), (20, 64), (8, 32)])
def test_a_dedicated_pool_reads_as_split(vram_gb, ram_gb):
    """The discrete shape at the AMD cards in this fleet: 7900XTX 24/64 and 7900XT 20/64.
    The 8/32 case is the 3070's ratio, though an NVIDIA card takes the nvidia-smi path
    rather than this one. REASONED from the sysfs contract, not measured: there is no dGPU
    on this host, so this is the branch to check first if a desktop reads wrong."""
    ram = ram_gb * 1024 ** 3
    dev = {"vram_bytes": vram_gb * 1024 ** 3, "gtt_bytes": int(ram * 0.5)}
    unified, why = _unified_from_drm(dev, ram)
    assert unified is False, why
    assert "independently" in why


def test_an_inconclusive_read_is_none_and_says_why():
    """A guess here silently mis-prices every bandwidth decision; a None only asks the
    caller to declare."""
    for dev, ram in (({"vram_bytes": None, "gtt_bytes": None}, RAM),
                     # too big to be a carveout, yet still addressing all of RAM: the
                     # signals contradict, so it declines rather than picking a side
                     ({"vram_bytes": 30 * 1024 ** 3, "gtt_bytes": 64 * 1024 ** 3},
                      64 * 1024 ** 3),
                     ({"vram_bytes": 4 * 1024 ** 3, "gtt_bytes": RAM}, 0)):
        unified, why = _unified_from_drm(dev, ram)
        assert unified is None, (dev, why)
        assert why


def test_this_machine_is_detected_as_unified():
    """The reported bug, as a test. Skips where there is no amdgpu to read."""
    gpus = [g for g in detect_gpus() if g.get("driver") == "amdgpu"]
    if not gpus:
        pytest.skip("no amdgpu device on this host")
    g = gpus[0]
    assert g["unified"] is True, g["unified_evidence"]
    assert g["vram_gb"] and g["gtt_gb"]
    assert g["gtt_gb"] > g["vram_gb"], g


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
    host-reading test then a rule that breaks on the no-carveout shape would again be
    invisible until CI reached a mac."""
    gpu = hw.get("gpu") or {}
    assert gpu.get("unified"), hw
    assert hw["model_budget_gb"] == pytest.approx(round(hw["ram_gb"] * 0.75, 1))
    if gpu.get("vram_gb"):
        assert hw["model_budget_gb"] > gpu["vram_gb"] * 2, hw


def test_a_unified_carveout_does_not_become_the_model_budget():
    """The bug filling vram_gb would have introduced: `if vram: budget = vram` would have
    called this 121 GiB machine a 4 GiB one, a 30x understatement, and every model-fit
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


def test_a_unified_host_reporting_no_carveout_still_budgets_from_ram(monkeypatch):
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
    _assert_unified_budget(hw)          # the same rules, on the no-carveout shape
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
