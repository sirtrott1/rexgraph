"""
agent.cli.config: paths, platform detection, config persistence.

Config: ~/.config/rexgraph/config.json (XDG on Linux, APPDATA on Windows).
Cache:  ~/.cache/rexgraph/models/ (model weights, PID file).
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Paths

def _xdg_config_home() -> Path:
    """XDG config directory (Linux/macOS) or APPDATA (Windows)."""
    if platform.system() == "Windows":
        base = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
        return Path(base) / "rexgraph"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "rexgraph"


def _xdg_cache_home() -> Path:
    """XDG cache directory (Linux/macOS) or LOCALAPPDATA (Windows)."""
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        return Path(base) / "rexgraph" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rexgraph"


CONFIG_DIR = _xdg_config_home()
CACHE_DIR = _xdg_cache_home()
CONFIG_FILE = CONFIG_DIR / "config.json"
PID_FILE = CACHE_DIR / "server.pid"
MODELS_DIR = CACHE_DIR / "models"


# Platform detection

@dataclass
class PlatformInfo:
    """Detected platform capabilities."""
    os: str = ""                    # linux, darwin, windows
    arch: str = ""                  # x86_64, aarch64, arm64
    package_manager: str = ""       # apt, dnf, brew, conda, choco, none
    gpu: str = ""                   # nvidia, amd, apple, none
    gpu_name: str = ""              # human-readable GPU name
    gpu_vram_mb: int = 0            # VRAM in megabytes
    cuda_version: str = ""          # CUDA toolkit version if installed
    python: str = ""                # Python version
    conda_env: str = ""             # active conda env name or ""
    scheduler: str = ""             # slurm, pbs, sge, none
    is_hpc: bool = False            # detected HPC environment
    scratch_dir: str = ""           # $SCRATCH or $TMPDIR if on HPC
    has_sudo: bool = False          # can run sudo non-interactively


def detect_platform() -> PlatformInfo:
    """Detect OS, GPU, package manager, and HPC scheduler."""
    import shutil
    import subprocess

    info = PlatformInfo()

    # OS and arch
    info.os = platform.system().lower()
    if info.os == "darwin":
        info.os = "macos"
    info.arch = platform.machine().lower()
    info.python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Conda environment
    info.conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    # Package manager
    if info.conda_env:
        info.package_manager = "conda"
    elif info.os == "linux":
        if shutil.which("apt-get"):
            info.package_manager = "apt"
        elif shutil.which("dnf") or shutil.which("yum"):
            info.package_manager = "dnf"
        elif shutil.which("pacman"):
            info.package_manager = "pacman"
    elif info.os == "macos":
        if shutil.which("brew"):
            info.package_manager = "brew"
    elif info.os == "windows":
        if shutil.which("choco"):
            info.package_manager = "choco"
        elif shutil.which("winget"):
            info.package_manager = "winget"

    if not info.package_manager:
        info.package_manager = "none"

    # GPU detection
    info.gpu, info.gpu_name, info.gpu_vram_mb, info.cuda_version = _detect_gpu()

    # Sudo availability (non-interactive check)
    if info.os != "windows":
        try:
            r = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output=True, timeout=2,
            )
            info.has_sudo = r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            info.has_sudo = False

    # HPC scheduler detection
    if os.environ.get("SLURM_JOB_ID") or shutil.which("sbatch"):
        info.scheduler = "slurm"
        info.is_hpc = True
    elif os.environ.get("PBS_JOBID") or shutil.which("qsub"):
        info.scheduler = "pbs"
        info.is_hpc = True
    elif shutil.which("qsub") and os.environ.get("SGE_ROOT"):
        info.scheduler = "sge"
        info.is_hpc = True
    else:
        info.scheduler = "none"

    # Scratch directory for HPC builds
    for env_var in ("SCRATCH", "TMPDIR", "WORK"):
        d = os.environ.get(env_var, "")
        if d and os.path.isdir(d):
            info.scratch_dir = d
            break

    return info


def _detect_gpu():
    """Detect GPU type, name, VRAM, and CUDA version."""
    import shutil
    import subprocess

    gpu_type = "none"
    gpu_name = ""
    vram_mb = 0
    cuda_ver = ""

    # NVIDIA
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                line = r.stdout.strip().split("\n")[0]
                parts = line.split(", ")
                gpu_name = parts[0].strip()
                vram_mb = int(parts[1].strip()) if len(parts) > 1 else 0
                gpu_type = "nvidia"
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            gpu_type = "nvidia"
            gpu_name = "NVIDIA GPU (details unavailable)"

        # CUDA version
        try:
            r = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                import re
                m = re.search(r"release (\d+\.\d+)", r.stdout)
                if m:
                    cuda_ver = m.group(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # AMD ROCm
    elif shutil.which("rocm-smi") or shutil.which("rocminfo"):
        gpu_type = "amd"
        gpu_name = "AMD GPU (ROCm)"

        # Method 1: rocm-smi --showmeminfo vram
        try:
            r = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.split("\n"):
                    low = line.lower()
                    if "total" in low:
                        # Parse any number from the line
                        import re
                        nums = re.findall(r'[\d]+', line)
                        for n in nums:
                            val = int(n)
                            if val > 1_000_000_000:  # bytes
                                vram_mb = val // (1024 * 1024)
                            elif val > 1_000_000:  # KB
                                vram_mb = val // 1024
                            elif val > 1000:  # MB
                                vram_mb = val
                        if vram_mb > 0:
                            break
        except Exception:
            pass

        # Method 2: rocm-smi --showmeminfo vram --csv
        if vram_mb == 0:
            try:
                r = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    for line in r.stdout.strip().split("\n"):
                        if "total" in line.lower():
                            import re
                            nums = [int(x) for x in re.findall(r'\d+', line)]
                            for n in nums:
                                if n > 1_000_000_000:
                                    vram_mb = n // (1024 * 1024)
                                elif n > 1000:
                                    vram_mb = n
            except Exception:
                pass

        # Method 3: rocminfo for GPU name and memory
        if vram_mb == 0:
            try:
                r = subprocess.run(
                    ["rocminfo"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    for line in r.stdout.split("\n"):
                        if "Marketing Name" in line:
                            gpu_name = line.split(":")[-1].strip() or gpu_name
                        if "Size" in line and "pool" not in line.lower():
                            import re
                            nums = re.findall(r'(\d+)\s*\(', line)
                            for n in nums:
                                val = int(n)
                                if val > 1_000_000_000:
                                    vram_mb = val // (1024 * 1024)
            except Exception:
                pass

        # Method 4: PyTorch ROCm
        if vram_mb == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    vram_mb = props.total_mem // (1024 * 1024)
                    gpu_name = props.name or gpu_name
            except Exception:
                pass

        # Method 5: 7900 XT is 24GB - if we detect it by name
        if vram_mb == 0 and "7900" in gpu_name:
            vram_mb = 24576  # 24GB

    # Apple Silicon
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        gpu_type = "apple"
        gpu_name = "Apple Silicon (MPS)"
        # Unified memory - estimate 75% of total RAM available for GPU
        try:
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                total_bytes = int(r.stdout.strip())
                vram_mb = int(total_bytes * 0.75 / (1024 * 1024))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

    return gpu_type, gpu_name, vram_mb, cuda_ver


# Config read/write

@dataclass
class RexGraphConfig:
    """Persisted state in config.json."""

    # Installed backends
    tesseract: bool = False
    paddleocr: bool = False
    paddleocr_model_path: str = ""
    gpu_model: str = ""
    gpu_model_path: str = ""

    # Server
    gpu_server_port: int = 10000
    agent_server_port: int = 8000
    gpu_server_backend: str = "vllm"   # vllm or sglang
    auto_start_gpu: bool = False

    # Cache
    cache_dir: str = ""
    models_dir: str = ""

    # TrustGraph
    trustgraph_url: str = ""

    # Custom model paths (model_id -> local path)
    model_paths: dict[str, str] = field(default_factory=dict)

    # Pipeline model assignments (purpose -> model_id)
    pipeline_models: dict[str, str] = field(default_factory=dict)

    # Metadata
    last_setup: str = ""
    platform_os: str = ""
    platform_gpu: str = ""


def load_config() -> RexGraphConfig:
    """Load the config file, or return defaults."""
    cfg = RexGraphConfig(
        cache_dir=str(CACHE_DIR),
        models_dir=str(MODELS_DIR),
    )

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            for key, val in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
        except (json.JSONDecodeError, OSError):
            pass

    return cfg


def save_config(cfg: RexGraphConfig) -> None:
    """Write the config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


# PID file management

def save_pid(pid: int, port: int, backend: str = "") -> None:
    """Write server PID and port to the PID file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"pid": pid, "port": port, "backend": backend}
    with open(PID_FILE, "w") as f:
        json.dump(data, f)


def read_pid() -> dict[str, Any] | None:
    """Read the PID file. Returns None if missing or unparseable."""
    if not PID_FILE.exists():
        return None
    try:
        with open(PID_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def clear_pid() -> None:
    """Remove the PID file."""
    if PID_FILE.exists():
        PID_FILE.unlink(missing_ok=True)


def process_alive(pid: int) -> bool:
    """Check whether a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def main(argv=None) -> int:
    """CLI entry: rexgraph-config show|path."""
    import argparse
    import dataclasses
    import json as _json

    p = argparse.ArgumentParser(
        prog="rexgraph-config",
        description="Show the resolved RexGraph configuration.",
    )
    sub = p.add_subparsers(dest="command")
    sub.add_parser("show", help="Print the current configuration")
    sub.add_parser("platform", help="Print detected platform info")
    args = p.parse_args(argv)

    if args.command == "platform":
        pi = detect_platform()
        data = dataclasses.asdict(pi) if dataclasses.is_dataclass(pi) else vars(pi)
        print(_json.dumps(data, indent=2, default=str))
        return 0
    # default / "show"
    cfg = load_config()
    data = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)
    print(_json.dumps(data, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
