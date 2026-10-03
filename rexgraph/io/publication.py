"""Same filesystem staged publication for local files and directory containers.

Encoding and writes finish before replacement. Directory replacement uses locked
rename and rollback, rather than promising a crash atomic directory exchange.
Updates hold the lock while copying and editing to avoid lost group updates.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
from inspect import signature
from pathlib import Path
import os
import secrets
import shutil
import stat
import tempfile
import threading

try:
    import fcntl
except ImportError:
    fcntl = None

_LOCK_PID = os.getpid()
_REGISTRY_GUARD = threading.Lock()
_GATES = {}
_OPEN_FDS = set()
_LOCAL = threading.local()


class _DirectoryGate:
    def __init__(self):
        self.lock = threading.RLock()
        self.users = 0


def _before_fork():
    # Descriptor registration/close and fork must observe one consistent set.
    _REGISTRY_GUARD.acquire()


def _after_fork_parent():
    _REGISTRY_GUARD.release()


def _after_fork_child():
    global _LOCK_PID, _REGISTRY_GUARD, _GATES, _OPEN_FDS, _LOCAL
    # An inherited open file description would keep the parent's flock alive
    # after its owner closed it, making the child block on its own inheritance.
    for fd in _OPEN_FDS:
        try:
            os.close(fd)
        except OSError:
            pass
    _LOCK_PID = os.getpid()
    _REGISTRY_GUARD = threading.Lock()
    _GATES, _OPEN_FDS, _LOCAL = {}, set(), threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent,
                        after_in_child=_after_fork_child)


def _close_descriptor(fd, owner_pid):
    if fd is not None and owner_pid == os.getpid():
        with _REGISTRY_GUARD:
            try:
                os.close(fd)
            finally:
                _OPEN_FDS.discard(fd)


@contextmanager
def publication_lock(parent):
    """Arbitrate one directory, with reentrant scopes and POSIX process exclusion.

    Directory aliases share their inode gate. Independent directories can publish
    concurrently. Nested distinct gates must follow canonical path order (ancestors
    before descendants); reverse acquisition refuses before waiting, avoiding cycles.
    Children close inherited gate descriptors and acquire their own locks after fork.
    """
    if os.getpid() != _LOCK_PID:
        _after_fork_child()
    owner_pid = os.getpid()
    canonical = os.path.normcase(os.path.realpath(parent))
    fd, gate, key = None, None, None
    held = getattr(_LOCAL, "held", None)
    if held is None:
        held = _LOCAL.held = {}
    try:
        with _REGISTRY_GUARD:
            if fcntl is None:
                if not stat.S_ISDIR(os.stat(canonical).st_mode):
                    raise ValueError("publication scope requires a directory")
                key = ("path", canonical)
            else:
                fd = os.open(canonical, os.O_RDONLY)
                _OPEN_FDS.add(fd)
                directory = os.fstat(fd)
                if not stat.S_ISDIR(directory.st_mode):
                    raise ValueError("publication scope requires a directory")
                key = ("inode", directory.st_dev, directory.st_ino)
            if key not in held and any(path > canonical for path in held.values()):
                raise RuntimeError("distinct publication directory locks require canonical path order")
            gate = _GATES.get(key)
            if gate is None:
                gate = _GATES[key] = _DirectoryGate()
            gate.users += 1
        with gate.lock:
            if key in held:
                # flock is attached to an open file description. Acquiring a
                # second descriptor for the same directory would self deadlock.
                _close_descriptor(fd, owner_pid)
                fd = None
                yield
            else:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                held[key] = canonical
                try:
                    yield
                finally:
                    held.pop(key, None)
                    _close_descriptor(fd, owner_pid)
                    fd = None
    finally:
        _close_descriptor(fd, owner_pid)
        if gate is not None and owner_pid == os.getpid():
            with _REGISTRY_GUARD:
                gate.users -= 1
                if not gate.users:
                    del _GATES[key]


def _remove(path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _check_target(target, directory):
    if target.is_symlink():
        raise ValueError("publication target must not be a symbolic link")
    if target.exists() and target.is_dir() != directory:
        raise ValueError("publication target has an incompatible container type")
    if target.exists() and not directory and not target.is_file():
        raise ValueError("publication target must be a regular file")


def _staging(target, directory):
    for _ in range(100):
        path = target.with_name(f".{target.name}.tmp-{secrets.token_hex(8)}{target.suffix}")
        try:
            if directory:
                path.mkdir()
            else:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                os.close(fd)
            return path
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a staging path beside the target")


def _sync(path):
    if path.is_symlink():
        raise ValueError("published containers must not contain symbolic links")
    if path.is_dir():
        for child in path.iterdir():
            _sync(child)
        if os.name == "nt":
            return
    fd = os.open(path, os.O_RDONLY)
    try:
        # Windows has no portable directory fsync; file contents still flush.
        if not path.is_dir() or os.name != "nt":
            os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def staged_publication(path, *, directory=False, update=False):
    """Yield an owned staging path and publish only after successful writing."""
    target = Path(path)
    parent = target.parent
    _check_target(target, directory)
    with publication_lock(parent) if update else nullcontext():
        _check_target(target, directory)
        staged = _staging(target, directory)
        recovery = None
        published = False
        try:
            if update:
                _remove(staged)
                if target.exists():
                    if directory:
                        shutil.copytree(target, staged, symlinks=True)
                        for base, directories, files in os.walk(staged):
                            if any((Path(base) / name).is_symlink() for name in directories + files):
                                raise ValueError("published containers must not contain symbolic links")
                    else:
                        shutil.copy2(target, staged)
            yield staged
            _check_target(staged, directory)
            if not staged.exists():
                raise ValueError("writer did not produce its staged container")
            if target.exists():
                os.chmod(staged, stat.S_IMODE(target.stat().st_mode))
            _sync(staged)
            with nullcontext() if update else publication_lock(parent):
                _check_target(target, directory)
                if directory and target.exists():
                    recovery = Path(tempfile.mkdtemp(prefix=f".{target.name}.recovery-", dir=parent))
                    backup = recovery / "previous"
                    os.replace(target, backup)
                    try:
                        os.replace(staged, target)
                    except BaseException:
                        os.replace(backup, target)
                        raise
                else:
                    os.replace(staged, target)
                if os.name != "nt":
                    fd = os.open(parent, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                published = True
        finally:
            _remove(staged)
            # If rollback itself failed, retain the only surviving old version.
            if recovery is not None and not (recovery / "previous").exists():
                _remove(recovery)
            elif recovery is not None and published:
                _remove(recovery)


def atomic_writer(*, normalize, directory=False, update=False, prepare=None, finalize=None):
    """Wrap a writer whose destination is its second positional argument."""
    def decorate(writer):
        writer_signature = signature(writer)
        destination = tuple(writer_signature.parameters)[1]
        @wraps(writer)
        def publish(*args, **kwargs):
            bound = writer_signature.bind(*args, **kwargs)
            path = bound.arguments[destination]
            with staged_publication(normalize(os.fspath(path)), directory=directory, update=update) as staged:
                if prepare is not None and staged.exists():
                    prepare(str(staged))
                bound.arguments[destination] = str(staged)
                result = writer(*bound.args, **bound.kwargs)
                if finalize is not None:
                    finalize(str(staged))
                return result
        return publish
    return decorate
