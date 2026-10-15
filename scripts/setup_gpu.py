#!/usr/bin/env python3
"""Select and install GPU dependencies in the active Python environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROCM_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CUDA_PROBE = r'''
import ctypes as C, ctypes.util, json
devices = []
try:
    lib = C.CDLL(C.util.find_library('cuda') or 'libcuda.so.1')
    lib.cuInit.argtypes = [C.c_uint]
    lib.cuDriverGetVersion.argtypes = [C.POINTER(C.c_int)]
    lib.cuDeviceGetCount.argtypes = [C.POINTER(C.c_int)]
    lib.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
    lib.cuDeviceComputeCapability.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int), C.c_int]
    version, count = C.c_int(), C.c_int()
    if lib.cuInit(0) == 0 and lib.cuDriverGetVersion(C.byref(version)) == 0 and lib.cuDeviceGetCount(C.byref(count)) == 0:
        for i in range(min(count.value, 512)):
            device, major, minor = C.c_int(), C.c_int(), C.c_int()
            if lib.cuDeviceGet(C.byref(device), i) == 0 and lib.cuDeviceComputeCapability(C.byref(major), C.byref(minor), device) == 0:
                devices.append({'cuda_version': version.value, 'capability': major.value + minor.value / 10})
except (OSError, AttributeError):
    pass
print(json.dumps(devices))
'''


def command_output(command, timeout=20):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def pci_vendors(base=Path("/sys/bus/pci/devices")):
    vendors = set()
    for device in base.glob("*"):
        try:
            if int((device / "class").read_text().strip(), 16) >> 16 == 3:
                vendors.add(int((device / "vendor").read_text().strip(), 16))
        except (OSError, ValueError):
            pass
    return vendors


def nvidia_devices():
    tool = shutil.which("nvidia-smi")
    output = command_output([tool, "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"]) if tool else ""
    devices = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and re.fullmatch(r"[0-9.]+", parts[0]) and re.fullmatch(r"[0-9.]+", parts[1]):
            devices.append({"driver": int(parts[0].split(".")[0]), "capability": float(parts[1])})
    if devices:
        return devices
    try:
        return json.loads(command_output([sys.executable, "-B", "-I", "-c", CUDA_PROBE]))
    except ValueError:
        return []


def cuda_index(devices):
    if not devices:
        raise RuntimeError("NVIDIA GPU detected but this process could not query CUDA. Check the NVIDIA driver, GPU device access and visibility settings, then retry.")
    capability = min(d["capability"] for d in devices)
    if "cuda_version" in devices[0]:
        version = min(d["cuda_version"] for d in devices)
        choices = [(13000, 7.5, "cu130"), (12080, 7.5, "cu128"), (12060, 5.0, "cu126"),
                   (12040, 5.0, "cu124"), (12010, 5.0, "cu121"), (11000, 5.0, "cu118")]
        driver = version
    else:
        driver = min(d["driver"] for d in devices)
        choices = [(580, 7.5, "cu130"), (570, 7.5, "cu128"), (560, 5.0, "cu126"),
                   (550, 5.0, "cu124"), (525, 5.0, "cu121"), (450, 5.0, "cu118")]
    # Keep older architectures on CUDA 12 wheels. Runtime qualification checks
    # the actual wheel's kernel support after installation.
    for minimum, arch, index in choices:
        if driver >= minimum and capability >= arch:
            return "https://download.pytorch.org/whl/" + index
    raise RuntimeError("The NVIDIA driver or GPU is too old for the configured PyTorch wheels. Set TORCH_INDEX_URL for a supported build.")


def select_backend(requested, vendors, devices):
    if requested != "auto":
        return requested
    if devices or 0x10DE in vendors:
        return "cuda"
    if 0x1002 in vendors:
        return "rocm"
    return "cpu"


def pip_install(*packages, index=None):
    command = [sys.executable, "-B", "-I", "-m", "pip", "install", *packages]
    if index:
        command.extend(["--index-url", index])
    subprocess.run(command, check=True)


def amd_targets():
    explicit = os.environ.get("HIP_ARCH", "")
    builder = runpy.run_path(str(ROOT / "rexgraph/_hip_build.py"))
    if explicit:
        return builder["validate_architectures"](explicit)
    targets = builder["architectures"]()
    if targets:
        return targets
    pip_install("rocm-bootstrap>=0.3")
    # A fresh interpreter sees newly installed SDK/bootstrap packages.
    output = command_output([sys.executable, "-B", "-I", "-c",
                             "from rocm_bootstrap import detect_gfx_targets; print(','.join(t.name for t in detect_gfx_targets()))"])
    if not output:
        raise RuntimeError("AMD GPU target could not be read. Check GPU device access or set HIP_ARCH, for example HIP_ARCH=gfx1151.")
    return builder["validate_architectures"](output)


def torch_flavour():
    return command_output([sys.executable, "-B", "-I", "-c",
                           "import torch; print('rocm' if torch.version.hip else 'cuda' if torch.version.cuda else 'cpu')"])


def prepare(backend="auto", native_hip="auto", require_gpu=False):
    devices = nvidia_devices()
    vendors = pci_vendors()
    chosen = select_backend(backend, vendors, devices)
    if backend == "auto" and vendors and (chosen == "cpu" or chosen == "cuda" and not devices):
        probe = runpy.run_path(str(ROOT / "rexgraph/gpu_access.py"))["probe_vulkan"]()
        if probe["available"]:
            chosen = "vulkan"
    if require_gpu and chosen == "cpu":
        raise RuntimeError("No supported GPU detected. Check device access, or select --gpu cuda, rocm or vulkan explicitly.")
    if chosen == "rocm" and platform.system() != "Linux":
        raise RuntimeError("The RexGraph ROCm installer requires Linux or WSL.")
    state = {"requested": backend, "backend": chosen, "native_hip": "disabled",
             "targets": [], "status": "dependencies_pending", "numerical_qualification": False}
    override = os.environ.get("TORCH_INDEX_URL")
    if chosen == "rocm":
        targets = amd_targets()
        state["targets"] = targets
        state["native_hip"] = "enabled" if native_hip != "disabled" else "disabled"
        index = override or os.environ.get("ROCM_INDEX_URL", ROCM_INDEX)
        extras = ",".join("device-" + t for t in targets)
        packages = ["--upgrade", f"torch[{extras}]"]
        if state["native_hip"] == "enabled":
            packages.append(f"rocm[libraries,devel,{extras}]")
    elif chosen == "cuda":
        index = override or cuda_index(devices)
        packages = ["torch"]
    elif chosen == "vulkan":
        index = CPU_INDEX
        packages = ["torch"]
    else:
        index = override or CPU_INDEX
        packages = ["torch"]
    if native_hip == "enabled" and chosen != "rocm":
        raise RuntimeError("NATIVE_HIP=enabled requires the ROCm backend.")
    old_flavour = torch_flavour()
    expected = "cpu" if chosen == "vulkan" else chosen
    # Reuse an existing compatible Torch build unless the caller selected an
    # index explicitly. Replace an incompatible build rather than letting pip
    # accept its already installed version.
    if old_flavour and (old_flavour != expected or override):
        packages = ["--upgrade", "--force-reinstall", *packages]
    if chosen == "rocm" and override:
        # A custom framework index need not carry AMD SDK packages or device
        # extras. Keep its resolver separate from the selected compiler SDK.
        pip_install("--upgrade", "--force-reinstall", "torch", index=override)
        if state["native_hip"] == "enabled":
            builder = runpy.run_path(str(ROOT / "rexgraph/_hip_build.py"))
            if os.environ.get("ROCM_INDEX_URL"):
                pip_install(f"rocm[libraries,devel,{extras}]", index=os.environ["ROCM_INDEX_URL"])
            elif not builder["compiler_info"]():
                raise RuntimeError("The selected ROCm PyTorch build needs a compatible HIP compiler. Set HIPCC for an existing compiler or ROCM_INDEX_URL for its matching SDK; NATIVE_HIP=disabled omits native kernels.")
    elif not old_flavour or old_flavour != expected or override or state["native_hip"] == "enabled":
        pip_install(*packages, index=index)
    # Vulkan remains available alongside CUDA/ROCm for explicit selection.
    if platform.system() == "Linux":
        pip_install("vulkan>=1.3.275.1")
    state.update(index_url=index, status="dependencies_installed")
    return state


def verify(state, size):
    if state["backend"] == "cpu":
        state.update(status="cpu_selected", numerical_qualification=False)
        return True
    if state["backend"] == "vulkan":
        from rexgraph.vulkan_compute import qualify
        state["qualification"] = qualify(size)
        ok = state["qualification"]["ok"]
        state.update(status="passed" if ok else "failed", numerical_qualification=bool(ok))
        return bool(ok)
    from rexgraph.gpu_preflight import run
    checks = run(size)
    state["qualification"] = checks
    ok = checks["ok"] and checks["environment"]["flavour"] == state["backend"]
    if state["native_hip"] == "enabled":
        native = next((c for c in checks["checks"] if c["check"] == "native_packed_training"), None)
        ok = bool(ok and native and native["ok"] and native["detail"] and not native["detail"].get("skipped"))
    from rexgraph.gpu_access import probe_vulkan
    vulkan = probe_vulkan()
    if vulkan["available"]:
        from rexgraph.vulkan_compute import qualify
        state["vulkan_qualification"] = qualify(min(size, 64))
        ok = bool(ok and state["vulkan_qualification"]["ok"])
    else:
        state["vulkan_qualification"] = {"status": "unavailable", "diagnostic": vulkan}
    state.update(status="passed" if ok else "failed", numerical_qualification=bool(ok))
    return bool(ok)


def state_path():
    return Path(sys.prefix) / "share/rexgraph/gpu-install.json"


def save_state(state):
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "rocm", "vulkan"), default="auto")
    parser.add_argument("--native-hip", choices=("auto", "enabled", "disabled"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--field", choices=("native_hip", "targets", "backend"))
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args(argv)
    state = None
    checked = None
    try:
        if args.field:
            state = json.loads(state_path().read_text())
            value = state[args.field]
            print(",".join(value) if isinstance(value, list) else value)
            return 0
        if args.verify:
            loaded = json.loads(state_path().read_text())
            if not isinstance(loaded, dict):
                raise RuntimeError("GPU setup record is invalid. Run make install-gpu before verification.")
            state = loaded
            if state.get("backend") not in {"cpu", "cuda", "rocm", "vulkan"} or state.get("native_hip") not in {"enabled", "disabled"}:
                raise RuntimeError("GPU dependency setup has not completed. Run make install-gpu before verification.")
            checked = dict(state)
            if args.backend != "auto":
                checked["backend"] = args.backend
                if args.backend != "rocm":
                    checked["native_hip"] = "disabled"
            if args.require_gpu and checked["backend"] == "cpu":
                raise RuntimeError("CPU execution is selected; GPU numerical verification did not run.")
            ok = verify(checked, args.size)
            state.setdefault("qualifications", {})[checked["backend"]] = {
                key: checked[key] for key in ("status", "numerical_qualification", "qualification", "vulkan_qualification") if key in checked}
            if checked["backend"] == state["backend"]:
                state.update({key: checked[key] for key in ("status", "numerical_qualification", "qualification", "vulkan_qualification") if key in checked})
        else:
            if sys.prefix == sys.base_prefix and not (Path(sys.prefix) / "conda-meta").is_dir():
                raise RuntimeError("Run GPU setup in the RexGraph virtual or conda environment.")
            state = {"requested": args.backend, "status": "preparing", "numerical_qualification": False}
            save_state(state)
            state = prepare(args.backend, args.native_hip, args.require_gpu)
            ok = True
        save_state(state)
        print(json.dumps(checked or state, indent=2))
        return 0 if ok else 1
    except Exception as exc:
        if state is not None:
            failure = {"status": "failed", "error": str(exc), "numerical_qualification": False}
            if checked is not None:
                state.setdefault("qualifications", {})[checked["backend"]] = failure
                if checked["backend"] == state["backend"]:
                    state.update(failure)
            else:
                state.update(failure)
            save_state(state)
        print(f"GPU setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
