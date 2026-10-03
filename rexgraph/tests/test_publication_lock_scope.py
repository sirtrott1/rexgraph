"""Transaction publication can nest safely and independent directories can progress."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import subprocess
import sys
from threading import Event

import pytest

from rexgraph.io import publication


@pytest.mark.parametrize("alias", [False, True])
def test_nested_scope_can_publish_without_locking_itself(alias, tmp_path):
    target = tmp_path / "target"
    other = tmp_path
    if alias:
        other = tmp_path / "alias"
        other.symlink_to(tmp_path, target_is_directory=True)
    script = """
from pathlib import Path
import sys
from rexgraph.io.publication import publication_lock, staged_publication
root, alias, target = map(Path, sys.argv[1:])
with publication_lock(root):
    with publication_lock(alias):
        with staged_publication(target, update=True) as staged:
            staged.write_text('published')
assert target.read_text() == 'published'
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path), str(other), str(target)],
                   check=True, timeout=5, capture_output=True)


def test_distinct_directories_do_not_share_a_process_wide_publication_gate(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    entered, release, independent = Event(), Event(), Event()
    def hold():
        with publication.publication_lock(first):
            entered.set()
            assert release.wait(5)
    def publish():
        with publication.publication_lock(second):
            independent.set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(hold)
        try:
            assert entered.wait(5)
            b = pool.submit(publish)
            assert independent.wait(1), "an unrelated directory was blocked"
        finally:
            release.set()
        a.result(timeout=5)
        b.result(timeout=5)


def test_same_directory_aliases_still_exclude_other_threads(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    attempted, admitted = Event(), Event()
    def contender():
        attempted.set()
        with publication.publication_lock(alias):
            admitted.set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with publication.publication_lock(tmp_path):
            task = pool.submit(contender)
            assert attempted.wait(5)
            assert not admitted.wait(.1)
        task.result(timeout=5)
    assert admitted.is_set()


def _process_contender(path, attempted, admitted):
    attempted.set()
    with publication.publication_lock(path):
        admitted.set()


def test_forked_contender_does_not_keep_the_parents_lock_alive(tmp_path):
    if os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("the qualified process lock profile uses POSIX fork")
    context = multiprocessing.get_context("fork")
    attempted, admitted = context.Event(), context.Event()
    process = context.Process(target=_process_contender, args=(str(tmp_path), attempted, admitted))
    try:
        with publication.publication_lock(tmp_path):
            process.start()
            assert attempted.wait(5)
            assert not admitted.wait(.1)
        assert admitted.wait(5), "the child inherited a descriptor retaining the parent lock"
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_body_exception_releases_directory_arbitration(tmp_path):
    with pytest.raises(ValueError, match="body failed"):
        with publication.publication_lock(tmp_path):
            raise ValueError("body failed")
    with publication.publication_lock(tmp_path):
        pass


def test_nested_distinct_directories_require_a_consistent_order(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    with publication.publication_lock(first):
        with publication.publication_lock(second):
            pass
    with publication.publication_lock(second):
        with pytest.raises(RuntimeError, match="path order"):
            with publication.publication_lock(first):
                pytest.fail("reverse acquisition was admitted")


def test_completed_and_failed_scopes_retain_no_gate_or_descriptor(tmp_path):
    with publication.publication_lock(tmp_path):
        with publication.publication_lock(tmp_path):
            pass
    with pytest.raises(ValueError):
        with publication.publication_lock(tmp_path):
            raise ValueError("failed")
    assert not publication._GATES and not publication._OPEN_FDS
    assert not publication._LOCAL.held


def test_thread_only_fallback_remains_reentrant(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, "fcntl", None)
    with publication.publication_lock(tmp_path):
        with publication.publication_lock(tmp_path):
            assert not publication._OPEN_FDS
    assert not publication._GATES


@pytest.mark.parametrize("fallback", [False, True])
def test_nondirectory_scope_refuses_without_retaining_a_descriptor(fallback, tmp_path, monkeypatch):
    if fallback:
        monkeypatch.setattr(publication, "fcntl", None)
    path = tmp_path / "file"
    path.touch()
    with pytest.raises(ValueError, match="directory"):
        with publication.publication_lock(path):
            pytest.fail("regular file was accepted as a publication directory")
    assert not publication._GATES and not publication._OPEN_FDS
