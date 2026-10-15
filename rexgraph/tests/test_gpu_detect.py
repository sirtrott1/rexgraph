"""DRM memory classification with generated fixtures and live driver contract checks."""
import pytest

from rexgraph.hardware import _unified_from_drm, detect_gpus, drm_devices

RAM = 128 * 1024 ** 3


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
    """Synthetic dedicated memory pools select the discrete memory classification."""
    ram = ram_gb * 1024 ** 3
    dev = {"vram_bytes": vram_gb * 1024 ** 3, "gtt_bytes": int(ram * 0.5)}
    unified, why = _unified_from_drm(dev, ram)
    assert unified is False, why
    assert "independently" in why


def test_an_inconclusive_read_is_none_and_says_why():
    """A guess here silently mis prices every bandwidth decision; a None only asks the
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


def test_amdgpu_classification_matches_its_memory_fields():
    """Live detection follows the same classification as the reported memory fields."""
    gpus = [g for g in detect_gpus() if g.get("driver") == "amdgpu"]
    if not gpus:
        pytest.skip("AMD GPU device is not visible to this process")
    from rexgraph.hardware import _ram_bytes
    for g in gpus:
        expected, _ = _unified_from_drm(g, _ram_bytes())
        assert g["unified"] is expected, g["unified_evidence"]
