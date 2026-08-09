"""
agent.server.routes.releases: an ontology across its releases.

Upload the releases in order and get back what changed, which terms were merged as
opposed to deleted, and which release was a surprise rather than ordinary growth.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/v1/releases")


@router.post("/analyze")
async def analyze_series(
    files: list[UploadFile] = File(...),
    labels: str = Form(""),
    download: str = Form(""),
):
    """Read the uploaded releases as one series, in the order given.

    Order is the caller's: a filename is not a date, so guessing one would silently
    reorder the series. `labels` is an optional comma-separated list matching the
    upload order.

    `download=temporal` returns the series itself as a temporal complex rather than a
    report of it.
    """
    from agent.ontology_releases import load_releases, navigate, summary

    if len(files) < 2:
        raise HTTPException(
            400, "a series needs at least two releases; one file is not a series")

    tmpdir = tempfile.mkdtemp(prefix="rexgraph_releases_")
    try:
        paths, names = [], [x.strip() for x in labels.split(",") if x.strip()]
        for f in files:
            name = os.path.basename(f.filename or "release")
            path = os.path.join(tmpdir, name)
            with open(path, "wb") as fh:
                fh.write(await f.read())
            paths.append(path)
        if not names:
            names = [os.path.basename(p) for p in paths]

        try:
            releases = load_releases(paths, labels=names)
        except ValueError as e:
            raise HTTPException(400, str(e)[:300]) from e

        if download.strip().lower() == "temporal":
            from agent.ontology_releases import temporal_complex
            from agent.server.artifacts import complex_file
            temporal, _vocab = temporal_complex(releases)
            return complex_file(temporal, "releases", "safetensors")

        out = summary(releases)
        out["navigation"] = navigate(releases)
        return out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
