"""GPU installation selection and failures with synthetic hardware reports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup():
    return load_module("gpu_setup_test", ROOT / "scripts/setup_gpu.py")


@pytest.fixture
def builder():
    return load_module("hip_build_test", ROOT / "rexgraph/_hip_build.py")


@pytest.mark.parametrize("vendors,devices,expected", [
    (set(), [], "cpu"), ({0x1002}, [], "rocm"), ({0x10DE}, [], "cuda"),
    ({0x1002, 0x10DE}, [], "cuda"), (set(), [{"driver": 580}], "cuda"),
])
def test_selection_uses_gpu_vendors_and_respects_explicit_backend(setup, vendors, devices, expected):
    assert setup.select_backend("auto", vendors, devices) == expected
    for explicit in ("cpu", "cuda", "rocm", "vulkan"):
        assert setup.select_backend(explicit, vendors, devices) == explicit


@pytest.mark.parametrize("driver,capability,index", [
    (580, 9.0, "cu130"), (580, 7.0, "cu126"), (570, 8.6, "cu128"),
    (560, 8.6, "cu126"), (550, 8.0, "cu124"), (535, 8.0, "cu121"),
    (470, 7.0, "cu118"),
])
def test_cuda_selection_respects_driver_and_device_architecture(setup, driver, capability, index):
    assert setup.cuda_index([{"driver": driver, "capability": capability}]).endswith("/" + index)


def test_multiple_nvidia_devices_use_a_compatible_common_wheel(setup):
    assert setup.cuda_index([{"driver": 580, "capability": 9.0}, {"driver": 580, "capability": 6.1}]).endswith("/cu126")
    with pytest.raises(RuntimeError, match="driver"):
        setup.cuda_index([])
    with pytest.raises(RuntimeError, match="too old"):
        setup.cuda_index([{"driver": 390, "capability": 3.0}])


def test_cuda_driver_api_is_used_without_nvidia_smi(setup, monkeypatch):
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(setup, "command_output", lambda *a: '[{"cuda_version":12080,"capability":8.6}]')
    devices = setup.nvidia_devices()
    assert setup.cuda_index(devices).endswith("/cu128")
    assert setup.cuda_index([{"cuda_version": 11040, "capability": 7.0}]).endswith("/cu118")


def test_cpu_pci_devices_are_not_selected_as_graphics_hardware(setup, tmp_path):
    for name, cls, vendor in (("graphics", "0x030000", "0x1002"), ("audio", "0x040300", "0x10de")):
        device = tmp_path / name
        device.mkdir()
        (device / "class").write_text(cls)
        (device / "vendor").write_text(vendor)
    (tmp_path / "incomplete").mkdir()
    assert setup.pci_vendors(tmp_path) == {0x1002}


def stub_setup(monkeypatch, setup, vendor=0x1002, flavour=""):
    monkeypatch.setattr(setup, "pci_vendors", lambda: {vendor} if vendor else set())
    monkeypatch.setattr(setup, "nvidia_devices", lambda: [{"driver": 580, "capability": 9.0}] if vendor == 0x10DE else [])
    monkeypatch.setattr(setup, "torch_flavour", lambda: flavour)
    monkeypatch.setattr(setup, "amd_targets", lambda: ["gfx1151", "gfx1201"])
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.delenv("TORCH_INDEX_URL", raising=False)
    monkeypatch.delenv("ROCM_INDEX_URL", raising=False)
    calls = []
    monkeypatch.setattr(setup, "pip_install", lambda *args, **kw: calls.append((args, kw)))
    return calls


def test_amd_installs_matching_device_packages_and_requires_native_build(setup, monkeypatch):
    calls = stub_setup(monkeypatch, setup)
    state = setup.prepare()
    assert state["backend"] == "rocm" and state["native_hip"] == "enabled"
    args, kw = calls[0]
    assert "torch[device-gfx1151,device-gfx1201]" in args
    assert "rocm[libraries,devel,device-gfx1151,device-gfx1201]" in args
    assert kw["index"] == setup.ROCM_INDEX
    assert state["numerical_qualification"] is False


def test_nvidia_setup_does_not_install_amd_sdk(setup, monkeypatch):
    calls = stub_setup(monkeypatch, setup, vendor=0x10DE)
    state = setup.prepare()
    assert state["backend"] == "cuda" and state["native_hip"] == "disabled"
    assert calls[0] == (("torch",), {"index": "https://download.pytorch.org/whl/cu130"})
    assert calls[1][0] == ("vulkan>=1.3.275.1",)


def test_nvidia_vulkan_driver_is_selected_without_a_working_cuda_driver(setup, monkeypatch):
    calls = stub_setup(monkeypatch, setup, vendor=0x10DE)
    monkeypatch.setattr(setup, "nvidia_devices", lambda: [])
    monkeypatch.setattr(setup.runpy, "run_path", lambda path: {"probe_vulkan": lambda: {"available": True}})
    state = setup.prepare(require_gpu=True)
    assert state["backend"] == "vulkan" and not state["numerical_qualification"]
    assert calls[0] == (("torch",), {"index": setup.CPU_INDEX})


def test_gpu_required_does_not_silently_install_cpu(setup, monkeypatch):
    calls = stub_setup(monkeypatch, setup, vendor=None)
    with pytest.raises(RuntimeError, match="No supported GPU"):
        setup.prepare(require_gpu=True)
    assert not calls


def test_incompatible_existing_torch_is_replaced(setup, monkeypatch):
    calls = stub_setup(monkeypatch, setup, vendor=0x10DE, flavour="cpu")
    setup.prepare()
    assert calls[0][0][:2] == ("--upgrade", "--force-reinstall")


def test_explicit_index_is_preserved(setup, monkeypatch):
    calls = stub_setup(monkeypatch, setup, vendor=0x10DE)
    monkeypatch.setenv("TORCH_INDEX_URL", "https://example.invalid/torch")
    setup.prepare()
    assert calls[0][1]["index"] == "https://example.invalid/torch"


def test_rocm_framework_and_sdk_overrides_use_separate_package_sources(setup, monkeypatch):
    calls = stub_setup(monkeypatch, setup)
    monkeypatch.setenv("TORCH_INDEX_URL", "https://example.invalid/torch")
    monkeypatch.setenv("ROCM_INDEX_URL", "https://example.invalid/sdk")
    setup.prepare()
    assert calls[0] == (("--upgrade", "--force-reinstall", "torch"), {"index": "https://example.invalid/torch"})
    assert calls[1] == (("rocm[libraries,devel,device-gfx1151,device-gfx1201]",), {"index": "https://example.invalid/sdk"})


def test_rocm_framework_override_can_reuse_a_selected_compiler(setup, monkeypatch):
    calls = stub_setup(monkeypatch, setup)
    monkeypatch.setenv("TORCH_INDEX_URL", "https://example.invalid/torch")
    monkeypatch.setattr(setup.runpy, "run_path", lambda path: {"compiler_info": lambda: {"compiler": "/sdk/bin/hipcc"}})
    setup.prepare()
    assert not any("rocm[" in arg for args, kw in calls for arg in args)


def test_failed_package_installation_propagates(setup, monkeypatch):
    stub_setup(monkeypatch, setup)
    def fail(*args, **kw):
        raise subprocess.CalledProcessError(1, ["pip"])
    monkeypatch.setattr(setup, "pip_install", fail)
    with pytest.raises(subprocess.CalledProcessError):
        setup.prepare()


def test_required_hip_qualification_cannot_accept_a_skip(setup, monkeypatch):
    from rexgraph import gpu_preflight
    state = {"backend": "rocm", "native_hip": "enabled"}
    monkeypatch.setattr(gpu_preflight, "run", lambda size: {
        "ok": True, "environment": {"flavour": "rocm"},
        "checks": [{"check": "native_packed_training", "ok": True, "detail": {"skipped": "not built"}}],
    })
    assert not setup.verify(state, 8)
    assert state["status"] == "failed" and not state["numerical_qualification"]


def test_failed_vulkan_calculations_do_not_qualify_setup(setup, monkeypatch):
    from rexgraph import vulkan_compute
    monkeypatch.setattr(vulkan_compute, "qualify", lambda size: {"ok": False, "sparse_max_abs_error": 1.0})
    state = {"backend": "vulkan", "native_hip": "disabled"}
    assert not setup.verify(state, 8)
    assert state["status"] == "failed" and not state["numerical_qualification"]


def test_vulkan_verification_preserves_the_installed_rocm_selection(setup, monkeypatch, tmp_path):
    state_file = tmp_path / "gpu-install.json"
    state_file.write_text(json.dumps({"backend": "rocm", "native_hip": "enabled", "status": "passed"}))
    monkeypatch.setattr(setup, "state_path", lambda: state_file)
    def qualify(state, size):
        assert state["backend"] == "vulkan" and state["native_hip"] == "disabled"
        state.update(status="passed", numerical_qualification=True, qualification={"ok": True})
        return True
    monkeypatch.setattr(setup, "verify", qualify)
    assert setup.main(["--verify", "--backend", "vulkan"]) == 0
    saved = json.loads(state_file.read_text())
    assert saved["backend"] == "rocm" and saved["native_hip"] == "enabled"
    assert saved["qualifications"]["vulkan"]["numerical_qualification"]


def test_failed_dependency_setup_invalidates_an_earlier_passed_state(setup, monkeypatch, tmp_path):
    state_file = tmp_path / "gpu-install.json"
    state_file.write_text(json.dumps({"status": "passed", "numerical_qualification": True}))
    monkeypatch.setattr(setup, "state_path", lambda: state_file)
    def fail(*args, **kwargs):
        raise RuntimeError("SDK unavailable")
    monkeypatch.setattr(setup, "prepare", fail)
    assert setup.main(["--backend", "rocm"]) == 1
    saved = json.loads(state_file.read_text())
    assert saved["status"] == "failed" and not saved["numerical_qualification"]


def test_gpu_verification_cannot_pass_a_cpu_only_installation(setup, monkeypatch, tmp_path):
    state_file = tmp_path / "gpu-install.json"
    state_file.write_text(json.dumps({"backend": "cpu", "native_hip": "disabled", "status": "cpu_selected"}))
    monkeypatch.setattr(setup, "state_path", lambda: state_file)
    monkeypatch.setattr(setup, "verify", lambda *a: pytest.fail("CPU state entered GPU verification"))
    assert setup.main(["--verify", "--require-gpu"]) == 1
    assert not json.loads(state_file.read_text())["numerical_qualification"]


@pytest.mark.parametrize("record", [{"status": "failed"}, [], {"backend": "auto", "native_hip": "auto"}])
def test_incomplete_setup_records_produce_a_retry_instruction(setup, monkeypatch, tmp_path, capsys, record):
    state_file = tmp_path / "gpu-install.json"
    state_file.write_text(json.dumps(record))
    monkeypatch.setattr(setup, "state_path", lambda: state_file)
    assert setup.main(["--verify"]) == 1
    assert "make install-gpu" in capsys.readouterr().err


def test_hip_architecture_override_is_validated_before_compilation(builder, monkeypatch):
    monkeypatch.setenv("HIP_ARCH", "gfx1151,gfx1201,gfx1151")
    assert builder.architectures() == ["gfx1151", "gfx1201"]
    for invalid in ("gfx000", "gfx11", "gfx9", "gfx11-generic", "native", "gfx1151;touch anything", "gfx1151,", "--help"):
        with pytest.raises(ValueError):
            builder.validate_architectures(invalid)


@pytest.mark.parametrize("tool,report", [
    ("rocminfo", """Agent 1
  Name: gfx000
Agent 2
  Name: gfx1151
  ISA Info:
    Name: amdgcn-amd-amdhsa--gfx1151
    Name: amdgcn-amd-amdhsa--gfx11-generic
Agent 3
  Name: gfx90a
Agent 4
  Name: gfx1151
"""),
    ("rocm_agent_enumerator", "gfx000\ngfx1151\ngfx90a\ngfx1151\ngfx11\n"),
], ids=("rocminfo", "enumerator"))
def test_fresh_hip_detection_uses_device_targets_without_generic_isa_names(builder, monkeypatch, tool, report):
    monkeypatch.delenv("HIP_ARCH", raising=False)
    monkeypatch.setitem(sys.modules, "rocm_bootstrap", None)
    monkeypatch.setattr(builder, "sdk_roots", lambda: [])
    monkeypatch.setattr(builder.shutil, "which", lambda name: "/usr/bin/" + name if name == tool else None)
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return report
    monkeypatch.setattr(builder.subprocess, "check_output", query)
    assert builder.architectures() == ["gfx1151", "gfx90a"]
    assert calls == [["/usr/bin/" + tool, *(["-name"] if tool == "rocm_agent_enumerator" else [])]]


def test_system_hip_compilation_preserves_compiler_failure(builder, monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "compiler_info", lambda: {"compiler": "/usr/bin/hipcc", "python_sdk": False})
    calls = []
    def compile(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.CalledProcessError(3, command)
    monkeypatch.setattr(builder.subprocess, "run", compile)
    with pytest.raises(subprocess.CalledProcessError):
        builder.compile_kernel(tmp_path / "source.hip", tmp_path / "kernel.so", ["gfx1151", "gfx1201"])
    command, options = calls[0]
    assert "--offload-arch=gfx1151" in command and "--offload-arch=gfx1201" in command
    assert options["check"]


def test_python_sdk_compilation_uses_owned_overlay_without_changing_sdk(builder, monkeypatch, tmp_path):
    sdk = tmp_path / "sdk/_rocm_sdk_core"
    for path in ("bin/hipcc", "include/hip/hip_runtime.h", "lib/libamdhip64.so.7", "lib/llvm/bin/clang++"):
        target = sdk / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    before = sorted(str(p.relative_to(sdk)) for p in sdk.rglob("*"))
    monkeypatch.setattr(builder, "compiler_info", lambda: {"compiler": str(sdk / "bin/hipcc"), "root": str(sdk), "python_sdk": True})
    monkeypatch.setattr(builder, "sdk_roots", lambda: [sdk])
    calls = []
    monkeypatch.setattr(builder.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    output = tmp_path / "build/kernel.so"
    builder.compile_kernel(tmp_path / "source.hip", output, ["gfx1151"])
    command, options = calls[0]
    assert options["env"]["HIP_PLATFORM"] == "amd"
    assert (output.parent / "hip-toolchain/lib/libamdhip64.so").resolve() == sdk / "lib/libamdhip64.so.7"
    assert sorted(str(p.relative_to(sdk)) for p in sdk.rglob("*")) == before
    assert "-fno-rtlib-add-rpath" in command
    assert "-Wl,-rpath,$ORIGIN/../../_rocm_sdk_core/lib" in command
    assert not any(flag.startswith("-Wl,-rpath,") and str(tmp_path) in flag for flag in command)


def test_python_sdk_compiler_precedes_its_environment_launcher(builder, monkeypatch, tmp_path):
    site = tmp_path / "site-packages"
    sdk = site / "_rocm_sdk_core"
    compiler = sdk / "bin/hipcc"
    launcher = tmp_path / "environment/bin/hipcc"
    for path in (compiler, launcher):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    for key in ("HIP_PATH", "ROCM_PATH", "HIPCC"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PATH", str(launcher.parent))
    monkeypatch.setattr(builder.sysconfig, "get_path", lambda key: str(site))
    info = builder.compiler_info()
    assert info["compiler"] == str(compiler) and info["python_sdk"]
