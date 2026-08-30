"""
rexgraph.hardware: what this machine actually gives us.

The tree could generate SLURM submission scripts but never read the allocation
back, so thread counts fell through to os.cpu_count(): on a cluster, the NODE's
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
from typing import Any

#: explicit overrides, which always win: an operator who states a number has a
#: reason, and guessing past them is how a tuned job gets detuned.
ENV_CPUS = "REXGRAPH_CPUS"
ENV_MEMORY = "REXGRAPH_MEMORY_BYTES"

_CGROUP_V2_CPU = "/sys/fs/cgroup/cpu.max"
_CGROUP_V2_MEM = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
_CGROUP_V1_MEM = "/sys/fs/cgroup/memory/memory.limit_in_bytes"


def _int_env(name: str) -> int | None:
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


def _read_int(path: str) -> int | None:
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


def _affinity() -> int | None:
    """Cores this process may actually run on. Covers taskset and cgroup cpusets,
    which is how both schedulers and containers narrow a process."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _cgroup_cpu_quota() -> int | None:
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
    candidates: list[tuple[int, str]] = []
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


def physical_cores(*, with_source: bool = False):
    """Cores that do not share a load/store path, capped by `cpu_count`.

    SMT siblings share L1 and the load/store units, so on a MEMORY-BOUND kernel they add
    contention without adding memory parallelism. The channel tower is exactly that shape
    and measures it: on a 16c/32t machine the best width is 14 and 32 costs 10 to 15%
    against it, while the curve is flat from 10 to 24. Physical cores lands inside that
    flat region without anyone choosing a number.

    Counted as distinct `(package, core)` pairs over the CPUs this process may actually
    run on, so an affinity mask or a cgroup narrows it the same way `cpu_count` does.
    Falls back to `cpu_count` wherever the topology is not readable, which is the honest
    answer rather than a guess: without the topology there is no way to tell a sibling
    from a core.
    """
    def _topology_id(path):
        # NOT `_read_int`: that treats 0 as absent, which is right for a quota and wrong
        # here, since cpu0's core_id IS 0 and every id is 0-based.
        try:
            with open(path) as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            return None

    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        allowed = None
    total = cpu_count()
    pairs = set()
    try:
        base = "/sys/devices/system/cpu"
        for name in os.listdir(base):
            if not name.startswith("cpu") or not name[3:].isdigit():
                continue
            n = int(name[3:])
            if allowed is not None and n not in allowed:
                continue
            core = _topology_id(f"{base}/{name}/topology/core_id")
            pkg = _topology_id(f"{base}/{name}/topology/physical_package_id")
            if core is None:
                pairs = set()
                break
            pairs.add((pkg if pkg is not None else 0, core))
    except OSError:
        pairs = set()
    if not pairs:
        return (total, "cpu_count") if with_source else total
    value = max(1, min(len(pairs), total))
    return (value, "sysfs topology") if with_source else value


def _meminfo_available() -> int | None:
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

    candidates: list[tuple[int, str]] = []
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


def gpus() -> list[dict[str, Any]]:
    """Visible GPUs and their memory. Honours CUDA_VISIBLE_DEVICES, because torch
    does, so this reports the job's GPUs, not the node's."""
    out: list[dict[str, Any]] = []
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


def _torch_info() -> dict[str, Any]:
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



#### cloud
#
# Detection reads LOCAL signals only. The instance metadata service lives on a
# link-local address that HANGS rather than refuses when you are not on that cloud,
# so a provider probe that dials it turns "which cloud am I on" into a multi-second
# stall on every machine that is on none of them. DMI answers the same question
# from a file read.

_DMI_DIR = "/sys/class/dmi/id"

#: Azure's own documented marker. Hyper-V on a desk is also "Microsoft Corporation",
#: so the vendor string alone must not claim an Azure VM.
_AZURE_ASSET_TAG = "7783-7084-3265-9085-8269-3286-77"


def _dmi(field: str) -> str | None:
    """One DMI field, or None. Unreadable is the normal case in a container."""
    try:
        with open(os.path.join(_DMI_DIR, field)) as fh:
            return fh.read().strip()
    except OSError:
        return None


def cloud() -> dict[str, Any]:
    """Which cloud this is running on, from local signals only.

    Returns provider (aws/azure/gcp/oci/None), whatever instance identifier is
    available without a network call, and whether this is inside Kubernetes.
    """
    vendor = (_dmi("sys_vendor") or "")
    product = (_dmi("product_name") or "")
    asset = (_dmi("chassis_asset_tag") or "")
    uuid = (_dmi("product_uuid") or "")

    provider: str | None = None
    if "amazon" in vendor.lower() or uuid.lower().startswith("ec2"):
        provider = "aws"
    elif "google" in vendor.lower() or "google" in product.lower():
        provider = "gcp"
    elif asset.strip() == _AZURE_ASSET_TAG:
        provider = "azure"
    elif "oraclecloud" in vendor.lower().replace(" ", ""):
        provider = "oci"

    # env-provided identifiers, which containers get even when DMI is masked
    for env_name, name in (("AWS_EXECUTION_ENV", "aws"),
                           ("ECS_CONTAINER_METADATA_URI_V4", "aws"),
                           ("AZURE_CLIENT_ID", None), ("GCE_METADATA_HOST", "gcp")):
        if name and os.environ.get(env_name):
            provider = provider or name

    return {
        "provider": provider,
        "instance_type": os.environ.get("REXGRAPH_INSTANCE_TYPE") or None,
        "vendor": vendor or None,
        "product": product or None,
        "kubernetes": bool(os.environ.get("KUBERNETES_SERVICE_HOST")),
        "container": _in_container(),
    }


def _in_container() -> bool:
    """Whether this is inside a container, which is how cloud GPU instances are
    almost always run, and therefore whether the cgroup limits are the real ones."""
    if os.path.exists("/.dockerenv"):
        return True
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    try:
        with open("/proc/1/cgroup") as fh:
            text = fh.read()
        return any(m in text for m in ("docker", "kubepods", "containerd", "lxc"))
    except OSError:
        return False


def detect() -> dict[str, Any]:
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
        "cloud": cloud(),
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
    c = d["cloud"]
    if c["provider"]:
        parts.append(c["provider"] + (" k8s" if c["kubernetes"] else ""))
    elif c["container"]:
        parts.append("container")
    if d["scheduler"]:
        parts.append(d["scheduler"])
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
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
