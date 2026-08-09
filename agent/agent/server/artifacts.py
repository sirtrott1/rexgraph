"""
agent.server.artifacts: what a route hands back when the answer is not a number.

The library has a complete binary I/O stack and there is no reason for a route to
summarize a complex into JSON and drop the object. A complex goes out as `.rex`,
safetensors, HDF5 or Zarr; a feature matrix goes out as the labeled vector container;
a per-cell table goes out through the canonical table writers, which every other
consumer of this data already reads.

JSON stays for what JSON is for: a summary a browser renders. It is not the transport
for a relational complex.

Two constraints the writers here exist to satisfy.

FastAPI's `jsonable_encoder` runs inside `serialize_response`, ahead of any custom
`Response`, so a response class cannot protect a route that returns raw kernel output.
The kernels answer in `ndarray` and `np.generic` and neither survives that encoder;
`plain()` converts, and a route returning kernel output calls it.

`FileResponse` streams after the handler returns, so a scratch file it owns cannot be
deleted in a `finally`. Every writer here returns bytes and removes its own scratch.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from fastapi import HTTPException
from fastapi.responses import Response

#: complex containers, by the extension each is written under
COMPLEX_FORMATS = {
    "rex": ".rex",
    "safetensors": ".safetensors",
    "hdf5": ".h5",
    "h5": ".h5",
    "zarr": ".zarr",
}

#: per-cell table containers
TABLE_FORMATS = {"parquet": ".parquet"}


def plain(obj):
    """numpy out, builtins in.

    Call on anything that came from a kernel before returning it. See the module
    docstring for why the response class cannot do this.
    """
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _download(data: bytes, filename: str) -> Response:
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Content-Disposition":
                             f'attachment; filename="{filename}"'})


def _write_and_read(writer, suffix: str) -> bytes:
    """Run a writer against a fresh scratch path and return the bytes.

    The path is handed over NOT EXISTING, because the containers disagree about what
    they are: `.rex` and `.zarr` are directories the writer creates, and safetensors,
    HDF5 and parquet are single files. Pre-creating with `mkstemp` broke the bundle
    writers, which found a file where they wanted to make a directory.

    A directory container comes back zipped. Reading the bytes here rather than
    streaming the path is what lets the scratch be deleted: `FileResponse` streams
    after the handler returns, so a `finally` that removes it races the response.
    """
    import shutil

    scratch = tempfile.mkdtemp(prefix="rexgraph_artifact_")
    target = os.path.join(scratch, f"artifact{suffix}")
    try:
        writer(target)
        if os.path.isdir(target):
            import io
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for root, _dirs, files in os.walk(target):
                    for f in files:
                        full = os.path.join(root, f)
                        z.write(full, os.path.relpath(full, target))
            return buf.getvalue()
        with open(target, "rb") as fh:
            return fh.read()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _is_directory_container(fmt: str) -> bool:
    """Whether a container is written as a directory, so downloads zipped."""
    return fmt in ("rex", "zarr")


def complex_file(rex, name: str, fmt: str = "rex") -> Response:
    """A complex as a downloadable artifact in any container the library writes.

    Accepts a `RexGraph` or a `TemporalRex`. Every writer used here dispatches on the
    object's type: `save_safetensors` is the dispatching entry point and
    `rex_to_safetensors` is not, so a temporal series handed to the latter fails on a
    `RexGraph` attribute it does not have.

    `.rex` and `.zarr` are directories, so they come back zipped; the rest are single
    files.
    """
    fmt = str(fmt).strip().lower()
    suffix = COMPLEX_FORMATS.get(fmt)
    if suffix is None:
        raise HTTPException(
            400, f"unknown format {fmt!r}. Available: "
                 f"{', '.join(sorted(COMPLEX_FORMATS))}")

    def write(path):
        if fmt == "rex":
            from rexgraph.io import save_rex
            save_rex(path, rex)
        elif fmt == "safetensors":
            from rexgraph.io.safetensors_bridge import save_safetensors
            save_safetensors(path, rex)
        elif fmt in ("hdf5", "h5"):
            from rexgraph.io import save_hdf5
            save_hdf5(path, rex)
        else:
            from rexgraph.io import save_zarr
            save_zarr(path, rex)

    try:
        data = _write_and_read(write, suffix)
    except ImportError as e:
        raise HTTPException(
            400, f"'{fmt}' needs an optional dependency: {e}") from e
    out_suffix = f"{suffix}.zip" if _is_directory_container(fmt) else suffix
    return _download(data, f"{name}{out_suffix}")


def vectors_file(matrix, labels, name: str, *, feature_names=None,
                 metadata=None) -> Response:
    """A feature matrix as the labeled vector container.

    This is what `save_vectors` exists for, so a feature matrix produced by a route
    lands in the same container every other corpus of vectors here uses.
    """
    from rexgraph.io import save_vectors

    def write(path):
        save_vectors(np.asarray(matrix), labels, path,
                     feature_names=list(feature_names) if feature_names else None,
                     metadata=metadata or {})

    return _download(_write_and_read(write, ".safetensors"),
                     f"{name}.safetensors")


def metrics_file(columns: dict, name: str, *, index_name: str = "cell_idx"
                 ) -> Response:
    """Per-cell metrics through the canonical table writer.

    `columns` maps a column name to an equal-length array. Anything that is already
    per-cell belongs here rather than in a JSON list, because the parquet table is
    what the SQL bridge and the warehouse already read.
    """
    from rexgraph.io import write_metrics_table

    arrays = {k: np.asarray(v) for k, v in columns.items()}
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) > 1:
        raise HTTPException(
            400, f"metric columns have different lengths ({sorted(lengths)}), so "
                 "they are not all per-cell")

    def write(path):
        write_metrics_table(arrays, path, index_name=index_name)

    try:
        data = _write_and_read(write, ".parquet")
    except ImportError as e:
        raise HTTPException(400, f"parquet needs pyarrow: {e}") from e
    return _download(data, f"{name}.parquet")


def character_file(rex, name: str) -> Response:
    """The per-edge structural character through its own table writer."""
    from rexgraph.io import write_character_table

    def write(path):
        write_character_table(rex, path)

    try:
        data = _write_and_read(write, ".parquet")
    except ImportError as e:
        raise HTTPException(400, f"parquet needs pyarrow: {e}") from e
    return _download(data, f"{name}.character.parquet")


def persistence_file(result, name: str) -> Response:
    """A persistence result through its own table writer."""
    from rexgraph.io import write_persistence_table

    def write(path):
        write_persistence_table(result, path)

    try:
        data = _write_and_read(write, ".parquet")
    except ImportError as e:
        raise HTTPException(400, f"parquet needs pyarrow: {e}") from e
    return _download(data, f"{name}.persistence.parquet")
