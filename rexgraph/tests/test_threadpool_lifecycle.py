"""Runtime limits preserve pool state across concurrent work and older BLAS builds."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from rexgraph import compute


def _blas_widths():
    import threadpoolctl
    return {pool['filepath']: pool['num_threads'] for pool in threadpoolctl.threadpool_info()
            if pool['user_api'] == 'blas'}


def test_overlapping_limits_restore_the_initial_pool_state():
    import threadpoolctl
    compute.set_threads(None)
    with threadpoolctl.threadpool_limits(1):
        before = _blas_widths()
        assert before
        first_entered, second_entered, first_closed = Event(), Event(), Event()
        def first():
            try:
                with compute._inner_thread_limiter(2):
                    first_entered.set()
                    assert second_entered.wait(5)
            finally:
                first_closed.set()
        def second():
            assert first_entered.wait(5)
            with compute._inner_thread_limiter(3):
                at_overlap = _blas_widths()
                second_entered.set()
                assert first_closed.wait(5)
                after_first = _blas_widths()
            return at_overlap, after_first
        with ThreadPoolExecutor(max_workers=2) as executor:
            a, b = executor.submit(first), executor.submit(second)
            a.result(timeout=10)
            overlap, remaining = b.result(timeout=10)
        assert _blas_widths() == before
        for path in before:
            assert overlap[path] <= 2
            assert remaining[path] <= 3


@pytest.mark.parametrize('mode', ['configured', 'parallel', 'worker'])
@pytest.mark.parametrize('initial_width', [1, 3])
def test_exponential_survives_runtime_thread_changes(mode, initial_width, tmp_path):
    pytest.importorskip('scipy', reason='SciPy compatibility is checked when installed')
    script = r'''
import json, sys
import numpy as np
from scipy.linalg import expm
from rexgraph import compute
import threadpoolctl
B = np.zeros((6, 7))
for e, (s, t) in enumerate([(0,1),(1,2),(2,3),(0,3),(1,4),(4,5),(2,5)]):
    B[s,e] = -1
    B[t,e] = 1
A = -.15 * (B.T @ B)
w, V = np.linalg.eigh(A)
expected = (V * np.exp(w)) @ V.T
if sys.argv[1] == 'configured':
    compute.set_threads(3)
    np.testing.assert_allclose(expm(A), expected, atol=1e-12)
    compute.set_threads(2)
    np.testing.assert_allclose(expm(A), expected, atol=1e-12)
    compute.set_threads(None)
elif sys.argv[1] == 'parallel':
    for value in compute.parallel_map(lambda _: expm(A), range(2), threads=2, inner_threads=3):
        np.testing.assert_allclose(value, expected, atol=1e-12)
else:
    from rexgraph.coordinator import _proc_worker_init
    _proc_worker_init(3, False)
    np.testing.assert_allclose(expm(A), expected, atol=1e-12)
    _proc_worker_init(2, False)
    np.testing.assert_allclose(expm(A), expected, atol=1e-12)
np.testing.assert_allclose(expm(A), expected, atol=1e-12)
print(json.dumps({'status':'passed','pools':threadpoolctl.threadpool_info()}))
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, mode], cwd=tmp_path,
                            env=dict(os.environ, OPENBLAS_NUM_THREADS=str(initial_width), OMP_NUM_THREADS='1',
                                     PYTHONDONTWRITEBYTECODE='1'),
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['status'] == 'passed'


@pytest.mark.parametrize('warm', [False, True])
def test_process_tasks_use_spawn_when_forkserver_is_unavailable(monkeypatch, warm):
    from functools import partial
    import multiprocessing
    from rexgraph.coordinator import LanePools, execute
    monkeypatch.setattr(multiprocessing, 'get_all_start_methods', lambda: ['spawn'])
    original = multiprocessing.get_context
    def get_context(method):
        assert method == 'spawn'
        return original(method)
    monkeypatch.setattr(multiprocessing, 'get_context', get_context)
    units = [{'id': 'power', 'type': 'cpu', 'fn': partial(pow, 2, 5)}]
    if warm:
        pools = LanePools(cap={'proc': 1}, cores_budget=1)
        try:
            assert pools.run(units, {'power': 'proc'}) == {'power': 32}
        finally:
            pools.shutdown()
    else:
        assert execute(units, {'power': 'proc'}) == {'power': 32}
