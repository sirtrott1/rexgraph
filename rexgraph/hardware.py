"""
rexgraph.hardware: what this machine actually gives us.

The tree could generate SLURM submission scripts but never read the allocation
back, so thread counts fell through to os.cpu_count() -- on a cluster, the NODE's
core count rather than the job's. An eight-core allocation on a 128-core node
would start 128 workers, which on a shared cluster gets the job killed rather
than merely running slowly. Memory was not detected at all.

Every figure here is the MINIMUM across the sources that could constrain it,
because a limit is a limit: it is the smallest one that evicts you. Each also
reports WHERE it came from, since on a cluster "who said so" is the question you
actually need answered when a job is throttled.

    from rexgraph import hardware
    hardware.cpu_count()        # honours affinity, cgroups, SLURM, an override
    hardware.memory_bytes()     # honours cgroups, SLURM, MemAvailable
    hardware.detect()           # the full picture, for a setup profile or a log
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

#: explicit overrides, which always win: an operator who states a number has a
#: reason, and guessing past them is how a tuned job gets detuned.
ENV_CPUS = "REXGRAPH_CPUS"
ENV_MEMORY = "REXGRAPH_MEMORY_BYTES"

_CGROUP_V2_CPU = "/sys/fs/cgroup/cpu.max"
_CGROUP_V2_MEM = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
_CGROUP_V1_MEM = "/sys/fs/cgroup/memory/memory.limit_in_bytes"


def _int_env(name: str) -> Optional[int]:
    """A positive int from the environment, or None. A malformed value is ignored
    rather than fatal: a scheduler quirk should not stop the run."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        v = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as fh:
            text = fh.read().strip()
    except OSError:
        return None
    if text in ("max", "-1", ""):
        return None
    try:
        v = int(text.split()[0])
    except (ValueError, IndexError):
        return None
    return v if v > 0 else None


def _affinity() -> Optional[int]:
    """Cores this process may actually run on. Covers taskset and cgroup cpusets,
    which is how both schedulers and containers narrow a process."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _cgroup_cpu_quota() -> Optional[int]:
    """A CPU quota expressed as a whole number of cores, rounded up: half a core of
    quota still needs one worker to use it."""
    try:
        with open(_CGROUP_V2_CPU) as fh:
            parts = fh.read().split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if quota > 0 and period > 0:
                return max(1, -(-quota // period))
    except (OSError, ValueError):
        pass
    quota = _read_int(_CGROUP_V1_QUOTA)
    period = _read_int(_CGROUP_V1_PERIOD)
    if quota and period:
        return max(1, -(-quota // period))
    return None


def cpu_count(*, with_source: bool = False):
    """Usable cores: the smallest of an override, SLURM, affinity, a cgroup quota
    and the machine's own count. Never below 1."""
    candidates: List[Tuple[int, str]] = []
    override = _int_env(ENV_CPUS)
    if override:
        return (override, ENV_CPUS) if with_source else override

    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
                 "PBS_NCPUS", "NSLOTS", "OMP_NUM_THREADS"):
        v = _int_env(name)
        if v:
            candidates.append((v, name))
    aff = _affinity()
    if aff:
        candidates.append((aff, "sched_getaffinity"))
    quota = _cgroup_cpu_quota()
    if quota:
        candidates.append((quota, "cgroup"))
    total = os.cpu_count()
    if total:
        candidates.append((total, "os.cpu_count"))

    if not candidates:
        return (1, "fallback") if with_source else 1
    value, source = min(candidates, key=lambda c: c[0])
    value = max(1, int(value))
    return (value, source) if with_source else value


def _meminfo_available() -> Optional[int]:
    """MemAvailable, not MemTotal: what can actually be allocated without swapping
    is the number that decides whether a chunk size is safe."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def memory_bytes(*, with_source: bool = False):
    """Usable memory in bytes: the smallest of an override, SLURM, a cgroup limit
    and MemAvailable."""
    override = _int_env(ENV_MEMORY)
    if override:
        return (override, ENV_MEMORY) if with_source else override

    candidates: List[Tuple[int, str]] = []
    per_node = _int_env("SLURM_MEM_PER_NODE")           # SLURM reports MiB
    if per_node:
        candidates.append((per_node * 1024 * 1024, "SLURM_MEM_PER_NODE"))
    per_cpu = _int_env("SLURM_MEM_PER_CPU")
    if per_cpu:
        candidates.append((per_cpu * 1024 * 1024 * cpu_count(), "SLURM_MEM_PER_CPU"))
    for path, label in ((_CGROUP_V2_MEM, "cgroup"), (_CGROUP_V1_MEM, "cgroup")):
        v = _read_int(path)
        # cgroup v1 writes a sentinel near 2^63 to mean "unlimited"
        if v and v < (1 << 62):
            candidates.append((v, label))
    avail = _meminfo_available()
    if avail:
        candidates.append((avail, "MemAvailable"))

    if not candidates:
        return (1 << 31, "fallback") if with_source else (1 << 31)
    value, source = min(candidates, key=lambda c: c[0])
    return (int(value), source) if with_source else int(value)


def gpus() -> List[Dict[str, Any]]:
    """Visible GPUs and their memory. Honours CUDA_VISIBLE_DEVICES, because torch
    does -- so this reports the job's GPUs, not the node's."""
    out: List[Dict[str, Any]] = []
    try:
        import torch
    except ImportError:
        return out
    try:
        if not torch.cuda.is_available():
            return out
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append({
                "index": i,
                "name": p.name,
                "total_memory_bytes": int(p.total_memory),
                "capability": f"{p.major}.{p.minor}",
                "multi_processor_count": int(getattr(p, "multi_processor_count", 0)),
            })
    except Exception:
        return out
    return out


def _torch_info() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    return {
        "available": True,
        "version": torch.__version__,
        # torch presents ROCm's HIP under the cuda namespace, so the version
        # string is the only honest way to say which runtime is underneath.
        "flavour": "rocm" if getattr(torch.version, "hip", None) else "cuda",
    }


def detect() -> Dict[str, Any]:
    """Everything a setup profile or a run log should record about the host."""
    cpus, cpu_source = cpu_count(with_source=True)
    mem, mem_source = memory_bytes(with_source=True)
    devices = gpus()
    try:
        from rexgraph import compute
        backend = compute.best_backend()
    except Exception:
        backend = "cpu"
    return {
        "cpus": cpus,
        "cpu_source": cpu_source,
        "memory_bytes": mem,
        "memory_source": mem_source,
        "gpus": devices,
        "gpu_count": len(devices),
        "backend": backend,
        "torch": _torch_info(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "node": platform.node(),
        },
        "scheduler": ("slurm" if os.environ.get("SLURM_JOB_ID")
                      else "pbs" if os.environ.get("PBS_JOBID")
                      else None),
    }


def summary() -> str:
    """One line for a log header."""
    d = detect()
    gb = d["memory_bytes"] / 2 ** 30
    parts = [f"{d['cpus']} cpu ({d['cpu_source']})",
             f"{gb:.1f} GiB ({d['memory_source']})"]
    if d["gpus"]:
        names = ", ".join(f"{g['name']} {g['total_memory_bytes'] / 2 ** 30:.0f}GiB"
                          for g in d["gpus"])
        parts.append(f"{d['gpu_count']} gpu [{names}]")
    else:
        parts.append("no gpu")
    if d["scheduler"]:
        parts.append(d["scheduler"])
    return " | ".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Report the usable hardware.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.json:
        print(json.dumps(detect(), indent=2, default=str))
    else:
        print(summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
