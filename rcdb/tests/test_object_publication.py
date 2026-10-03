"""The process capability supplies conditional publication, not ambient fsspec."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import uuid

import pytest

from rcdb import LocalObjectPublication, MemoryObjectPublication, PublishedObject, VersionConflictError
from rcdb.object_publication import publication_for


@pytest.fixture(params=["file", "memory"])
def capability(request, tmp_path):
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem(request.param)
    root = str(tmp_path / "prefix") if request.param == "file" else "/publication-"+uuid.uuid4().hex
    return publication_for(fs, root)


def test_read_limits_conditional_creation_replacement_and_tokens(capability):
    assert not capability.has_objects() and capability.read("head", limit=0) is None
    first = capability.compare_and_swap("head", None, b"first")
    assert capability.read("head", limit=5) == first and capability.has_objects()
    with pytest.raises(ValueError): capability.read("head", limit=4)
    with pytest.raises(VersionConflictError): capability.compare_and_swap("head", None, b"stale")
    second = capability.compare_and_swap("head", first.token, b"second")
    assert second.token != first.token and capability.read("head", limit=6) == second
    with pytest.raises(VersionConflictError): capability.compare_and_swap("head", first.token, b"stale")
    assert capability.read("head", limit=6) == second


def test_independent_capabilities_have_one_cas_winner(capability):
    initial = capability.compare_and_swap("head", None, b"initial")
    peers = [publication_for(capability.fs, capability.root) for _ in range(6)]
    barrier = Barrier(len(peers))
    def write(pair):
        i, peer = pair
        barrier.wait(timeout=15)
        try:
            return peer.compare_and_swap("head", initial.token, str(i).encode())
        except VersionConflictError:
            return None
    with ThreadPoolExecutor(max_workers=len(peers)) as pool:
        results = list(pool.map(write, enumerate(peers)))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1 and capability.read("head", limit=7) == winners[0]


@pytest.mark.parametrize("key", ["", ".", "..", "../x", "/x", "a/../b", "a//b", "a\\b", "x\x00y"])
def test_literal_keys_cannot_escape_or_alias_the_owned_prefix(capability, key):
    with pytest.raises(ValueError): capability.path(key)
    assert not capability.has_objects()


@pytest.mark.parametrize("limit", [True, -1, 1.5])
def test_read_contract_refuses_invalid_limits(capability, limit):
    with pytest.raises(ValueError): capability.read("head", limit=limit)


@pytest.mark.parametrize("data,token", [(bytearray(b"x"), b"t"), (b"x", "t"), (b"x", b""), (b"x", b"t"*4097)])
def test_publication_result_is_owned_bytes_and_bounded_opaque_token(data, token):
    with pytest.raises(ValueError): PublishedObject(data, token)


def test_publication_capabilities_cannot_be_rebound_or_inferred_from_remote_writes(capability):
    with pytest.raises(ValueError, match="bound"):
        publication_for(capability.fs, capability.root, lambda fs, root: publication_for(fs, root+"/other"))
    with pytest.raises(ValueError, match="bound"):
        publication_for(capability.fs, capability.root, lambda fs, root: object())
    class Remote:
        def open(self, *args): pytest.fail("generic remote write was invoked")
    with pytest.raises(NotImplementedError, match="explicit"):
        publication_for(Remote(), "bucket/prefix")


def test_local_capability_refuses_symbolic_links_on_owned_paths(tmp_path):
    fsspec = pytest.importorskip("fsspec")
    outside = tmp_path / "outside"; outside.mkdir()
    alias = tmp_path / "alias"; alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError): LocalObjectPublication(fsspec.filesystem("file"), str(alias))
    root = tmp_path / "root"; root.mkdir()
    (root / "nested").symlink_to(outside, target_is_directory=True)
    cap = LocalObjectPublication(fsspec.filesystem("file"), str(root))
    with pytest.raises(ValueError): cap.compare_and_swap("nested/head", None, b"owned")
    assert not list(outside.iterdir())


def test_provider_constructors_require_the_declared_filesystem():
    fsspec = pytest.importorskip("fsspec")
    with pytest.raises(TypeError): LocalObjectPublication(fsspec.filesystem("memory"), "/prefix")
    with pytest.raises(TypeError): MemoryObjectPublication(fsspec.filesystem("file"), "/prefix")
