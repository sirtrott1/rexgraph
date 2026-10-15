"""Bound upload reads and release owned temporary files on every exit path."""
from __future__ import annotations

from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
import tempfile

from fastapi import HTTPException


def upload_limit_bytes() -> int:
    """Return the configured positive limit, defaulting to 100 MiB."""
    try:
        megabytes = float(os.environ.get("REXGRAPH_MAX_UPLOAD_MB", "100"))
        if not math.isfinite(megabytes) or megabytes <= 0:
            raise ValueError
        return max(1, int(megabytes * 1024 * 1024))
    except (ValueError, OverflowError):
        return 100 * 1024 * 1024


async def _upload_chunks(file, limit):
    size = 0
    while True:
        chunk = await file.read(min(1024 * 1024, limit - size + 1))
        if not chunk:
            return
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, "Upload exceeds REXGRAPH_MAX_UPLOAD_MB")
        yield chunk


async def read_uploads(files):
    """Read a batch with one size budget before starting pipeline processing."""
    remaining = upload_limit_bytes()
    uploads = []
    for file in files:
        content = bytearray()
        async for chunk in _upload_chunks(file, remaining):
            content.extend(chunk)
        remaining -= len(content)
        uploads.append({"name": file.filename or "file", "content": bytes(content)})
    return uploads


@asynccontextmanager
async def temporary_upload(file):
    """Stream an uploaded file to temporary storage and remove it after use."""
    path = None
    try:
        suffix = Path(file.filename or "input.bin").suffix
        limit = upload_limit_bytes()
        with tempfile.NamedTemporaryFile(prefix="rexgraph_upload_", suffix=suffix,
                                         delete=False) as output:
            path = Path(output.name)
            async for chunk in _upload_chunks(file, limit):
                output.write(chunk)
        yield str(path)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
