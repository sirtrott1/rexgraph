import pytest

from rcql import Executor, parse

FileCatalog = pytest.importorskip("rexgraph.io.catalog").FileCatalog


def test_catalog_queries_use_registered_relative_names(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"a")
    catalog = FileCatalog([tmp_path])
    result = Executor(sources={"files": catalog}).execute(
        parse('FROM CATALOG("files") RETURN FILES(), SEARCH("a.safetensors"), '
              'FILE_HASH("root0/a.safetensors")'))
    assert len(result.values[0]) == 1
    assert len(result.values[1]) == 1
    assert result.values[2] == catalog.info("root0/a.safetensors").sha256
