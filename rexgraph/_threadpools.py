"""Coordinate native pool limits and process startup for RexGraph calls."""
from __future__ import annotations

from threading import RLock
import os

_guard = RLock()
_requests = {}
_original = {}


def configuration_lock():
    return _guard


def process_start_method():
    """Use forkserver when its generated Unix socket path fits, otherwise spawn."""
    import multiprocessing
    import tempfile
    # Reserve space for Python's generated subdirectory and listener name.
    # A sixty byte parent keeps the complete path below the Unix socket limit.
    if ("forkserver" in multiprocessing.get_all_start_methods()
            and len(os.fsencode(tempfile.gettempdir())) <= 60):
        return "forkserver"
    return "spawn"


def _serial_blas(info):
    # OpenBLAS 0.3.28 can crash in parallel triangular solves after resizing.
    # Restrict that runtime; NumPy and other BLAS builds keep their own widths.
    return (info.get("internal_api") == "openblas"
            and info.get("version") == "0.3.28"
            and info.get("threading_layer") == "pthreads")


def _apply_limits():
    width = min(_requests.values()) if _requests else None
    for controller, original_width, serial in _original.values():
        selected = original_width if width is None else 1 if serial else width
        controller.limit(limits=selected)
    if width is None:
        _original.clear()


class ThreadLimit:
    """Apply a shared BLAS cap and restore caller OpenMP settings on close.

    BLAS libraries use the smallest active request until the last scope closes.
    Original widths are retained independently for each loaded library. OpenMP
    settings are restored by the scope's own controller in the calling thread.
    """

    def __init__(self, width):
        from threadpoolctl import ThreadpoolController
        self._token = object()
        self._pid = os.getpid()
        self._openmp = None
        controller = ThreadpoolController()
        with _guard:
            for info in controller.select(user_api="blas").info():
                path = info["filepath"]
                if path not in _original:
                    _original[path] = (controller.select(filepath=path),
                                       info["num_threads"], _serial_blas(info))
            _requests[self._token] = width
            try:
                self._openmp = controller.select(user_api="openmp").limit(limits=width)
                # BLAS using OpenMP shares this runtime. Apply its active cap last.
                _apply_limits()
            except BaseException:
                _requests.pop(self._token, None)
                self._token = None
                try:
                    if self._openmp is not None:
                        self._openmp.restore_original_limits()
                finally:
                    _apply_limits()
                raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore_original_limits()

    def restore_original_limits(self):
        if self._token is None or self._pid != os.getpid():
            return
        with _guard:
            _requests.pop(self._token, None)
            self._token = None
            try:
                if self._openmp is not None:
                    self._openmp.restore_original_limits()
            finally:
                _apply_limits()


def _after_fork():
    global _guard, _requests, _original
    _guard, _requests, _original = RLock(), {}, {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
