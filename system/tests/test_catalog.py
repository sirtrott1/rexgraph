
import pytest
from rexgraph.io.catalog import FileCatalog


def test_catalog_hashes_safetensors_without_exposing_root(tmp_path):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"safe tensor bytes")
    catalog = FileCatalog([tmp_path])
    entry = catalog.list()[0]
    assert entry.name == "root0/weights.safetensors"
    assert str(tmp_path) not in repr(entry)
    assert entry.sha256 is None
    digest = catalog.hash(entry.name)
    assert catalog.info(entry.name).sha256 == digest


def test_catalog_rejects_unknown_and_escape_names(tmp_path):
    path = tmp_path / "graph.safetensors"
    path.write_bytes(b"x")
    catalog = FileCatalog([tmp_path])
    with pytest.raises(KeyError):
        catalog.info("root0/../outside.safetensors")


def test_catalog_ignores_symlinks(tmp_path):
    outside = tmp_path.parent / "outside.safetensors"
    outside.write_bytes(b"outside")
    link = tmp_path / "linked.safetensors"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    catalog = FileCatalog([tmp_path])
    assert catalog.list() == []
