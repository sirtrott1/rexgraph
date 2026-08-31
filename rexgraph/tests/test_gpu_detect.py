"""What the DRM nodes say about this machine, and when they decline to say.

`detect_hardware` returned {"vendor":"amd","vram_gb":None,"unified":None} on a machine
whose sysfs answers all three, so these read the nodes rather than trusting a summary.
A carve-out is not a dedicated pool and the difference decides bus topology, which is
why an inconclusive read returns None instead of a guess.

These moved here with the probe: it is core code now, because the coordinator needs
the unified answer and was reaching up into the application to get it.
"""
import pytest

from rexgraph.hardware import _unified_from_drm, detect_gpus, drm_devices

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
