"""Platform detection, persistent configuration and OCR server process records.

Directory defaults and environment overrides are resolved by ``agent.data_paths``.
Model downloads and server process records use the same cache root as analysis.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from agent.data_paths import cache_root, config_root, models_root

# Paths

def _xdg_config_home() -> Path:
    """Return the configured application configuration directory."""
    return config_root()


def _xdg_cache_home() -> Path:
    """Return the configured application cache directory."""
    return cache_root()


CONFIG_DIR = _xdg_config_home()
CACHE_DIR = _xdg_cache_home()
CONFIG_FILE = CONFIG_DIR / "config.json"
PID_FILE = CACHE_DIR / "server.pid"
MODELS_DIR = models_root()


# Platform detection

@dataclass
class PlatformInfo:
    """Detected platform capabilities."""
    os: str = ""                    # linux, darwin, windows
    arch: str = ""                  # x86_64, aarch64, arm64
    package_manager: str = ""       # apt, dnf, brew, conda, choco, none
    gpu: str = ""                   # nvidia, amd, apple, none
    gpu_name: str = ""              # human readable GPU name
    gpu_vram_mb: int = 0            # VRAM in megabytes
    cuda_version: str = ""          # CUDA toolkit version if installed
    python: str = ""                # Python version
    conda_env: str = ""             # active conda env name or ""
    scheduler: str = ""             # slurm, pbs, sge, none
    is_hpc: bool = False            # detected HPC environment
    scratch_dir: str = ""           # $SCRATCH or $TMPDIR if on HPC
    has_sudo: bool = False          # can run sudo non interactively


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

    # Sudo availability (non interactive check)
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

        # Method 1: rocm smi --showmeminfo vram
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

        # Method 2: rocm smi --showmeminfo vram --csv
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

    # Apple Silicon
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        gpu_type = "apple"
        gpu_name = "Apple Silicon (MPS)"
        # Unified memory: estimate 75% of total RAM available for GPU
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
    """Load valid configuration fields, retaining defaults for invalid values."""
    cfg = RexGraphConfig(
        cache_dir=str(CACHE_DIR),
        models_dir=str(MODELS_DIR),
    )

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for item in fields(cfg):
                    key = item.name
                    val = data.get(key, getattr(cfg, key))
                    default = getattr(cfg, key)
                    if type(val) is not type(default):
                        continue
                    if key in {"cache_dir", "models_dir"} and not val:
                        continue
                    if isinstance(val, dict) and not all(
                            isinstance(k, str) and isinstance(v, str) for k, v in val.items()):
                        continue
                    if key.endswith("_port") and not 1 <= val <= 65535:
                        continue
                    setattr(cfg, key, val)
        except (json.JSONDecodeError, OSError, UnicodeError):
            pass

    # Explicit directory settings take precedence over paths saved by an earlier setup.
    if os.environ.get("REXGRAPH_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME"):
        cfg.cache_dir = str(cache_root())
    if (os.environ.get("REXGRAPH_MODELS_DIR") or os.environ.get("REXGRAPH_CACHE_DIR")
            or os.environ.get("XDG_CACHE_HOME")):
        cfg.models_dir = str(models_root())
    return cfg


def _write_json(path: Path, data: dict) -> None:
    """Replace a JSON file only after the complete contents have been written."""
    payload = json.dumps(data, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_config(cfg: RexGraphConfig) -> None:
    """Atomically replace the configuration file."""
    _write_json(CONFIG_FILE, asdict(cfg))


# PID file management

def save_pid(pid: int, port: int, backend: str = "") -> None:
    """Write server PID and port to the PID file."""
    data = {"pid": pid, "port": port, "backend": backend}
    if not _valid_pid_data(data):
        raise ValueError("server records require a positive PID, a valid port and a string backend")
    _write_json(PID_FILE, data)


def _valid_pid_data(data) -> bool:
    return (isinstance(data, dict) and type(data.get("pid")) is int and data["pid"] > 0
            and type(data.get("port")) is int and 1 <= data["port"] <= 65535
            and isinstance(data.get("backend", ""), str))


def read_pid() -> dict[str, Any] | None:
    """Read the PID file. Returns None if missing or unparseable."""
    try:
        with open(PID_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if _valid_pid_data(data) else None
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None


def clear_pid() -> None:
    """Remove the PID file."""
    PID_FILE.unlink(missing_ok=True)


def process_alive(pid: int) -> bool:
    """Check whether a process is still running."""
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, OverflowError):
        return False


def main(argv=None) -> int:
    """CLI entry: rexgraph config show|path."""
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
