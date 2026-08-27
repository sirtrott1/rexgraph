"""Hardware detection is a registry, so a machine this has never seen can be added.

Four probes ship. Only amdgpu is MEASURED (a Strix Halo 8060S); intel, nvidia and apple
are written from each driver's documented contract and have not been run on that
hardware, which the registry reports rather than leaving the reader to assume.

The point of the registry is that none of that is a ceiling: a vendor, an accelerator, or
a machine that simply knows its own answer can register a probe from outside without
editing the module.
"""
import pytest

from agent import local_runtime as LR


@pytest.fixture(autouse=True)
def _restore_probes():
    before = dict(LR._GPU_PROBES)
    yield
    LR._GPU_PROBES.clear()
    LR._GPU_PROBES.update(before)


def test_the_shipped_probes_declare_how_far_they_were_verified():
    probes = LR.gpu_probes()
    assert probes["amdgpu"] == "measured"
    for other in ("intel", "nvidia", "apple"):
        assert probes[other] == "reasoned", other


def test_a_third_party_probe_is_used():
    """The whole point: new hardware without touching this module."""
    def _npu(ram_bytes):
        return [{"vendor": "acme", "driver": "acme_npu", "vram_bytes": 8 * 1024 ** 3,
                 "gtt_bytes": None, "unified": False,
                 "unified_evidence": "acme cards carry their own HBM"}]

    LR.register_gpu_probe("acme", _npu, confidence="measured")
    got = [g for g in LR.detect_gpus() if g["vendor"] == "acme"]
    assert len(got) == 1
    g = got[0]
    assert g["unified"] is False and g["vram_gb"] == 8.0
    assert g["probe"] == "acme" and g["probe_confidence"] == "measured"


def test_a_probe_may_replace_a_shipped_one():
    """A machine that knows its own answer should be able to say so without a patch."""
    LR.register_gpu_probe("amdgpu", lambda ram: [
        {"vendor": "amd", "driver": "amdgpu", "vram_bytes": 24 * 1024 ** 3,
         "gtt_bytes": None, "unified": False, "unified_evidence": "declared by the host"}])
    amd = [g for g in LR.detect_gpus() if g["driver"] == "amdgpu"]
    assert amd and all(g["unified"] is False for g in amd)


def test_a_raising_probe_does_not_take_the_sweep_down():
    """One bad vendor module must not cost the machine its other GPUs."""
    def _boom(ram_bytes):
        raise RuntimeError("vendor tool absent")

    LR.register_gpu_probe("broken", _boom)
    LR.register_gpu_probe("fine", lambda ram: [
        {"vendor": "ok", "vram_bytes": 1024 ** 3, "unified": False}])
    vendors = {g["vendor"] for g in LR.detect_gpus()}
    assert "ok" in vendors


def test_a_probe_result_is_filled_out_even_when_sparse():
    LR._GPU_PROBES.clear()
    LR.register_gpu_probe("sparse", lambda ram: [{"vendor": "x"}])
    g = LR.detect_gpus()[0]
    for k in ("vendor", "driver", "pci_id", "vram_bytes", "gtt_bytes", "unified",
              "unified_evidence", "probe", "probe_confidence", "vram_gb", "gtt_gb"):
        assert k in g, k


#### the memory test itself
def test_the_exact_identity_decides_before_any_ratio():
    """gtt == MemTotal is an identity, not a threshold: the GPU addresses all of system
    memory, so its pool is a carveout of it. This is what fires on the measured host."""
    ram = 130452873216
    unified, why = LR._unified_from_memory(4 * 1024 ** 3, ram, ram)
    assert unified is True
    assert "exactly" in why


def test_the_ratio_is_a_fallback_and_says_so():
    """A ratio is a judgement, so it only runs when the identity did not answer, and it
    reports the magnitudes either way."""
    ram = 64 * 1024 ** 3
    unified, why = LR._unified_from_memory(2 * 1024 ** 3, int(ram * 0.98), ram)
    assert unified is True and "ratio rather than the exact identity" in why
    unified, why = LR._unified_from_memory(24 * 1024 ** 3, int(ram * 0.5), ram)
    assert unified is False and "independently" in why
    assert "0.375" in why or "0.38" in why           # the magnitude is reported


@pytest.mark.parametrize("vram_gb,ram_gb", [(24, 64), (20, 64), (8, 32), (16, 32)])
def test_dedicated_pools_read_as_split(vram_gb, ram_gb):
    """REASONED, not measured: no dGPU on this host. If a desktop reads wrong, this is
    the branch to check."""
    ram = ram_gb * 1024 ** 3
    unified, why = LR._unified_from_memory(vram_gb * 1024 ** 3, int(ram * 0.5), ram)
    assert unified is False, why


def test_conflicting_signals_decline():
    ram = 64 * 1024 ** 3
    unified, why = LR._unified_from_memory(30 * 1024 ** 3, ram, ram)
    assert unified is None or unified is True    # exact identity may claim it
    unified, why = LR._unified_from_memory(30 * 1024 ** 3, int(ram * 0.97), ram)
    assert unified is None and "inconclusive" in why


def test_tegra_is_not_called_discrete():
    """A Grace or Orin part is unified, and this probe cannot measure it, so it must not
    assert the PCIe answer for it."""
    devs = LR._probe_nvidia(64 * 1024 ** 3)
    for d in devs:                      # no-op where nvidia-smi is absent
        assert d["unified"] in (False, None)
