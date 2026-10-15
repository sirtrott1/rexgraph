"""Failed, interrupted and oversized uploads remove all owned staging files."""
import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException
from agent.server.upload_files import temporary_upload, upload_limit_bytes


class Source:
    filename = "synthetic.txt"
    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail
    async def read(self, size):
        self.calls += 1
        if self.calls == 1:
            return b"synthetic"
        if self.fail:
            raise asyncio.CancelledError
        return b""


@pytest.fixture
def temporary(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    return tmp_path


def test_successful_upload_lives_only_inside_processing_context(temporary):
    async def process():
        async with temporary_upload(Source()) as path:
            assert Path(path).read_bytes() == b"synthetic"
            return path
    path = asyncio.run(process())
    assert not Path(path).exists()


def test_interrupted_read_removes_partially_written_upload(temporary):
    async def process():
        async with temporary_upload(Source(fail=True)):
            pytest.fail("interrupted upload reached processing")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(process())
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("limit", ["nan", "inf", "-inf", "0", "-1", "invalid"])
def test_invalid_upload_limits_use_default(monkeypatch, limit):
    monkeypatch.setenv("REXGRAPH_MAX_UPLOAD_MB", limit)
    assert upload_limit_bytes() == 100 * 1024 * 1024


def test_oversized_upload_is_rejected_and_removed(temporary, monkeypatch):
    monkeypatch.setenv("REXGRAPH_MAX_UPLOAD_MB", "0.000001")
    async def process():
        async with temporary_upload(Source()):
            pytest.fail("oversized upload reached processing")
    with pytest.raises(HTTPException) as error:
        asyncio.run(process())
    assert error.value.status_code == 413
    assert not list(temporary.iterdir())


def test_failed_batch_staging_removes_prior_and_current_files(temporary):
    from agent.server.routes.pipeline import _save_uploads
    with pytest.raises(TypeError):
        _save_uploads([{"name": "one.txt", "content": b"one"},
                       {"name": "two.txt", "content": object()}])
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("options", ["null", "[]", "1", "broken"])
def test_invalid_upload_options_are_rejected_before_reading(temporary, options):
    from agent.server.routes.upload import upload_data
    source = Source()
    with pytest.raises(HTTPException) as error:
        asyncio.run(upload_data(file=source, options=options))
    assert error.value.status_code == 400
    assert source.calls == 0 and not list(temporary.iterdir())


def test_failed_ocr_creation_removes_uploaded_file(temporary, monkeypatch):
    from agent.server.routes.ocr import ocr_file
    source = Source()
    source.filename = "synthetic.png"
    def fail(**kwargs):
        raise RuntimeError("synthetic backend failure")
    monkeypatch.setattr("agent.integrations.unlimited_ocr.create_ocr_client", fail)
    with pytest.raises(HTTPException):
        asyncio.run(ocr_file(file=source, path=None, backend=None, dpi=72))
    assert not list(temporary.iterdir())


class BoundedSource:
    filename = "synthetic.txt"

    def __init__(self, content):
        self.content = content
        self.requests = []

    async def read(self, size):
        self.requests.append(size)
        chunk, self.content = self.content[:size], self.content[size:]
        return chunk


def test_pipeline_batch_shares_one_upload_budget(temporary, monkeypatch):
    from agent.server.routes.pipeline import stream_pipeline
    monkeypatch.setenv("REXGRAPH_MAX_UPLOAD_MB", str(10 / 1024 / 1024))
    sources = [BoundedSource(b"oneone"), BoundedSource(b"twotwo")]
    with pytest.raises(HTTPException) as error:
        asyncio.run(stream_pipeline(files=sources, workspace="default",
                                   reader_options="{}", face_selection="none"))
    assert error.value.status_code == 413
    assert sources[1].requests == [5]
    assert not list(temporary.iterdir())


def test_pipeline_accepts_a_batch_exactly_at_the_upload_limit(monkeypatch):
    from agent.server.upload_files import read_uploads
    monkeypatch.setenv("REXGRAPH_MAX_UPLOAD_MB", str(10 / 1024 / 1024))
    sources = [BoundedSource(b"first"), BoundedSource(b"other"), BoundedSource(b"")]
    assert asyncio.run(read_uploads(sources)) == [
        {"name": "synthetic.txt", "content": b"first"},
        {"name": "synthetic.txt", "content": b"other"},
        {"name": "synthetic.txt", "content": b""},
    ]
    assert sources[-1].requests == [1]
