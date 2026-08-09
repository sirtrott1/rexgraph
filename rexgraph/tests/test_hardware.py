"""What the machine actually gives us, not what the node has.

The tree generated SLURM submission scripts but never read the allocation back:
thread counts fell through to os.cpu_count(), which on a cluster is the NODE's core
count, not the job's. An eight-core allocation on a 128-core node would start 128
workers: oversubscription that on a shared cluster gets the job killed rather than
merely running slowly. Memory was not detected at all, so nothing could size itself
to the box.

Everything here reports the MINIMUM of what each source allows, because a limit is a
limit: exceeding the smallest one is what gets you evicted.
"""

import os

import pytest

from rexgraph import hardware


def test_cpu_count_is_positive_and_sane():
    n = hardware.cpu_count()
    assert isinstance(n, int) and n >= 1
    assert n <= (os.cpu_count() or 1)


def test_cpu_count_respects_a_slurm_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert hardware.cpu_count() == 3


def test_cpu_count_respects_scheduler_affinity(monkeypatch):
    """taskset and cgroup cpusets both show up here, so it covers containers too."""
    monkeypatch.setattr(hardware, "_affinity", lambda: 2)
    assert hardware.cpu_count() <= 2


def test_the_smallest_limit_wins(monkeypatch):
    # the machine's own count is one of the candidates, so it has to be pinned too or
    # the test measures the host instead of the rule. A three-core runner made the
    # answer 3, correctly, against an expectation of 4.
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    monkeypatch.setattr(hardware, "_affinity", lambda: 4)
    assert hardware.cpu_count() == 4


def test_a_nonsense_slurm_value_is_ignored(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    assert hardware.cpu_count() >= 1


def test_an_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("REXGRAPH_CPUS", "5")
    assert hardware.cpu_count() == 5


def test_memory_is_reported_in_bytes():
    m = hardware.memory_bytes()
    assert isinstance(m, int) and m > 0


def test_memory_respects_a_slurm_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "2048")      # SLURM reports MiB
    assert hardware.memory_bytes() == 2048 * 1024 * 1024


def test_memory_per_cpu_scales_with_the_allocation(monkeypatch):
    # per-CPU memory multiplies by cpu_count(), which takes the smallest candidate
    # including the machine's own. Pin it, or a small host silently changes the product.
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    monkeypatch.setattr(hardware, "_affinity", lambda: 64)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1024")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert hardware.memory_bytes() == 4 * 1024 * 1024 * 1024


def test_gpus_report_their_memory():
    gpus = hardware.gpus()
    assert isinstance(gpus, list)
    for g in gpus:
        assert g["index"] >= 0
        assert g["total_memory_bytes"] > 0
        assert g["name"]


def test_detect_returns_everything_a_setup_needs():
    d = hardware.detect()
    for key in ("cpus", "memory_bytes", "gpus", "gpu_count", "backend",
                "torch", "platform"):
        assert key in d, f"{key} missing"
    assert d["gpu_count"] == len(d["gpus"])


def test_detect_names_where_each_limit_came_from():
    """On a cluster the useful question is not just 'how many cores' but 'who said
    so', because that is what you check when a job is throttled."""
    d = hardware.detect()
    assert "cpu_source" in d and d["cpu_source"]
    assert "memory_source" in d and d["memory_source"]


def test_summary_is_one_readable_line():
    s = hardware.summary()
    assert isinstance(s, str) and s
    assert "cpu" in s.lower()


def test_compute_threads_default_to_the_allocation(monkeypatch):
    """The whole point: parallel_map used os.cpu_count(), so an allocation was
    ignored and the node's full width was used."""
    from rexgraph import compute

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert compute.effective_threads() == 3


def test_an_explicit_set_threads_still_wins(monkeypatch):
    from rexgraph import compute

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    compute.set_threads(2)
    try:
        assert compute.effective_threads() == 2
    finally:
        compute.set_threads(None)


#### device selection and op dispatch
def test_the_character_gpu_path_accepts_a_device():
    """sparse_character hardcoded torch.device('cuda'), so on a multi-GPU node it
    always landed on device 0 and could not be pointed anywhere else."""
    import inspect

    from rexgraph import sparse_character as sc

    src = inspect.getsource(sc._compute_sparse_phi_gpu)
    assert 'torch.device("cuda")' not in src, "device is still hardcoded"
    assert "device" in inspect.signature(sc._compute_sparse_phi_gpu).parameters


def test_ops_are_registered_not_an_empty_registry():
    """compute.register_op/dispatch existed with zero registrations and zero
    callers: an extension point that had never been extended, so dispatch could
    only ever raise."""
    from rexgraph import compute

    names = [o["name"] for o in compute.ops()]
    assert names, "the op registry is empty"
    assert "block_cg" in names


def test_dispatch_routes_and_agrees_with_the_cpu_implementation():
    import numpy as np
    import scipy.sparse as sp

    from rexgraph import compute

    rng = np.random.default_rng(0)
    n = 60
    A = sp.random(n, n, density=0.1, format="csr", random_state=0, dtype=np.float64)
    A = (A + A.T) * 0.5 + sp.eye(n, format="csr") * (n * 0.05 + 1.0)
    B = rng.standard_normal((n, 3))

    X = compute.dispatch("block_cg", A.tocsr(), B, prefer="cpu")
    assert np.abs(A @ X - B).max() < 1e-6


def test_dispatch_on_every_available_backend_gives_the_same_answer():
    import numpy as np
    import scipy.sparse as sp

    from rexgraph import compute

    rng = np.random.default_rng(1)
    n = 80
    A = sp.random(n, n, density=0.1, format="csr", random_state=1, dtype=np.float64)
    A = (A + A.T) * 0.5 + sp.eye(n, format="csr") * (n * 0.05 + 1.0)
    B = rng.standard_normal((n, 2))

    ref = compute.dispatch("block_cg", A.tocsr(), B, prefer="cpu")
    for backend in compute.available_backends():
        got = compute.dispatch("block_cg", A.tocsr(), B, prefer=backend)
        assert np.abs(got - ref).max() < 1e-6, f"{backend} disagreed with cpu"


def test_an_unknown_op_names_what_is_registered():
    from rexgraph import compute

    with pytest.raises(KeyError) as ei:
        compute.dispatch("no_such_op", 1)
    assert "block_cg" in str(ei.value)


#### cloud, not just the scheduler
def test_cloud_detection_never_touches_the_network_by_default(monkeypatch):
    """The metadata service is a link-local address that HANGS rather than refuses
    when you are not on that cloud, so detection has to answer from local signals."""
    import socket

    def _no_network(*a, **kw):
        raise AssertionError("cloud detection opened a socket")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    info = hardware.cloud()
    assert isinstance(info, dict) and "provider" in info


def test_aws_is_recognised_from_the_local_dmi_signal(monkeypatch):
    monkeypatch.setattr(hardware, "_dmi", lambda f: {"sys_vendor": "Amazon EC2"}.get(f))
    assert hardware.cloud()["provider"] == "aws"


def test_gcp_is_recognised(monkeypatch):
    monkeypatch.setattr(hardware, "_dmi",
                        lambda f: {"product_name": "Google Compute Engine"}.get(f))
    assert hardware.cloud()["provider"] == "gcp"


def test_azure_is_recognised(monkeypatch):
    monkeypatch.setattr(hardware, "_dmi", lambda f: {
        "sys_vendor": "Microsoft Corporation",
        "chassis_asset_tag": "7783-7084-3265-9085-8269-3286-77"}.get(f))
    assert hardware.cloud()["provider"] == "azure"


def test_a_plain_microsoft_vm_is_not_azure(monkeypatch):
    """Hyper-V on someone's desk is also 'Microsoft Corporation'. The asset tag is
    what distinguishes an Azure VM, so vendor alone must not claim it."""
    monkeypatch.setattr(hardware, "_dmi",
                        lambda f: {"sys_vendor": "Microsoft Corporation"}.get(f))
    assert hardware.cloud()["provider"] != "azure"


def test_kubernetes_is_reported_alongside_the_provider(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert hardware.cloud()["kubernetes"] is True


def test_no_cloud_signal_is_reported_honestly(monkeypatch):
    monkeypatch.setattr(hardware, "_dmi", lambda f: None)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    info = hardware.cloud()
    assert info["provider"] is None
    assert info["kubernetes"] is False


def test_detect_carries_the_cloud_block():
    d = hardware.detect()
    assert "cloud" in d and "provider" in d["cloud"]


def test_container_memory_limits_are_honoured(monkeypatch, tmp_path):
    """Cloud GPU instances almost always run containerized, so the cgroup limit is
    the real ceiling even when /proc/meminfo reports the whole host."""
    limit = tmp_path / "memory.max"
    limit.write_text(str(3 * 1024 ** 3))
    monkeypatch.setattr(hardware, "_CGROUP_V2_MEM", str(limit))
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.delenv("REXGRAPH_MEMORY_BYTES", raising=False)
    assert hardware.memory_bytes() <= 3 * 1024 ** 3


def test_container_cpu_quota_is_honoured(monkeypatch, tmp_path):
    quota = tmp_path / "cpu.max"
    quota.write_text("200000 100000")            # 2 cores
    monkeypatch.setattr(hardware, "_CGROUP_V2_CPU", str(quota))
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("REXGRAPH_CPUS", raising=False)
    assert hardware.cpu_count() <= 2
