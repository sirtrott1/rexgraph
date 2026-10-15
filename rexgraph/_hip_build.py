"""Discover a HIP compiler and build the optional packed kernels.

This module uses only the standard library so Meson can run it before Core is
installed. System ROCm and the ROCm Python SDK are supported.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig


def sdk_roots():
    candidates = [os.environ.get(key) for key in ("HIP_PATH", "ROCM_PATH")]
    compiler = shutil.which("hipcc")
    sites = {sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}
    for site in filter(None, sites):
        candidates.extend(str(Path(site) / name) for name in ("_rocm_sdk_core", "_rocm_sdk_devel"))
    if compiler:
        candidates.append(str(Path(compiler).resolve().parent.parent))
    candidates.append("/opt/rocm")
    return list(dict.fromkeys(Path(p).resolve() for p in candidates if p and Path(p).is_dir()))


def compiler_info():
    roots = sdk_roots()
    explicit = os.environ.get("HIPCC")
    candidates = ([explicit] if explicit else []) + [str(p / "bin/hipcc") for p in roots]
    for name in candidates:
        compiler = shutil.which(name)
        if compiler:
            root = Path(compiler).resolve().parent.parent
            return {"compiler": compiler, "root": str(root),
                    "python_sdk": root.name in {"_rocm_sdk_core", "_rocm_sdk_devel"}}
    return None


def architectures():
    explicit = os.environ.get("HIP_ARCH", "")
    if explicit:
        return validate_architectures(explicit)
    # The bootstrap package reads the kernel topology without loading HIP.
    try:
        from rocm_bootstrap import detect_gfx_targets
        targets = [target.name for target in detect_gfx_targets()]
        if targets:
            return validate_architectures(",".join(targets))
    except ImportError:
        pass
    for tool, args in (("rocm_agent_enumerator", ["-name"]), ("rocminfo", [])):
        path = shutil.which(tool)
        if not path:
            path = next((str(p / "bin" / tool) for p in sdk_roots() if (p / "bin" / tool).is_file()), None)
        if path:
            try:
                text = subprocess.check_output([path, *args], text=True, stderr=subprocess.DEVNULL, timeout=15)
                # rocminfo also lists generic ISA names. Only agent Name
                # fields identify devices; the enumerator prints one per line.
                pattern = (r"(?m)^\s*Name:\s*(gfx[0-9a-f]{3,})\s*$" if tool == "rocminfo"
                           else r"(?m)^\s*(gfx[0-9a-f]{3,})\s*$")
                targets = sorted(set(re.findall(pattern, text)) - {"gfx000"})
                if targets:
                    return targets
            except (OSError, subprocess.SubprocessError):
                pass
    # Query Torch in a child so an incompatible driver cannot terminate Meson.
    code = "import torch,json; print(json.dumps([torch.cuda.get_device_properties(i).gcnArchName.split(':')[0] for i in range(torch.cuda.device_count())] if torch.version.hip else []))"
    try:
        text = subprocess.check_output([sys.executable, "-B", "-I", "-c", code],
                                       text=True, stderr=subprocess.DEVNULL, timeout=20)
        targets = json.loads(text)
        return validate_architectures(",".join(targets)) if targets else []
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def validate_architectures(value):
    targets = value.split(",")
    if not targets or any(not re.fullmatch(r"gfx[0-9a-f]{3,}", t) or t == "gfx000" for t in targets):
        raise ValueError("HIP_ARCH must contain GPU targets such as gfx1151, separated by commas")
    return sorted(set(targets))


def compile_kernel(source, output, targets):
    info = compiler_info()
    if not info:
        raise RuntimeError("HIP compiler not found; run make install-gpu or set HIPCC")
    command = [info["compiler"]]
    env = dict(os.environ)
    if info["python_sdk"]:
        roots = sdk_roots()
        root = Path(info["root"])
        headers = next((p / "include" for p in roots if (p / "include/hip/hip_runtime.h").is_file()), None)
        runtime = next((p for p in (root / "lib").glob("libamdhip64.so*") if p.is_file()), None)
        llvm = root / "lib/llvm"
        if not headers or not runtime or not (llvm / "bin/clang++").is_file():
            raise RuntimeError("Incomplete ROCm Python SDK; install rocm[libraries,devel] from the selected AMD index")
        # Legacy Python SDKs omit the unversioned HIP linker name. Keep the
        # compatibility links in the build directory, beside the output.
        overlay = Path(output).resolve().parent / "hip-toolchain"
        (overlay / "lib").mkdir(parents=True, exist_ok=True)
        for name, target in ((overlay / "include", headers), (overlay / "lib/libamdhip64.so", runtime)):
            if not name.is_symlink() and not name.exists():
                name.symlink_to(target)
        env.update(HIP_PLATFORM="amd", HIP_PATH=str(overlay), ROCM_PATH=str(overlay),
                   HIP_CLANG_PATH=str(llvm / "bin"))
        # Resolve the SDK beside the installed package after a wheel moves
        # between environments. Build directory paths must not enter RUNPATH.
        runtime_path = "$ORIGIN/../../" + root.name + "/lib"
        command.extend(["--rocm-path=" + str(overlay), "--rocm-device-lib-path=" + str(llvm / "amdgcn/bitcode"),
                        "-L" + str(overlay / "lib"), "-fno-rtlib-add-rpath", "-Wl,-rpath," + runtime_path])
    command.extend("--offload-arch=" + target for target in targets)
    command.extend(["-O3", "-shared", "-fPIC", "-o", str(output), str(source)])
    subprocess.run(command, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--arch", default="")
    parser.add_argument("--compile", nargs=2, metavar=("SOURCE", "OUTPUT"))
    args = parser.parse_args(argv)
    try:
        if args.probe:
            info = compiler_info()
            if info:
                targets = validate_architectures(args.arch) if args.arch else architectures()
                if targets:
                    print(",".join(targets))
            return 0
        if not args.compile:
            parser.error("use --probe or --compile SOURCE OUTPUT")
        targets = validate_architectures(args.arch) if args.arch else architectures()
        if not targets:
            raise RuntimeError("No AMD GPU target detected; set HIP_ARCH for a build without device access")
        compile_kernel(*args.compile, targets)
        return 0
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"HIP build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
