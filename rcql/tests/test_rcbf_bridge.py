"""An RCBF stream reaches both RCDB and RCQL without a pairwise expansion."""
from __future__ import annotations

import struct

import numpy as np
from rcdb import MemoryStore
from rexgraph.io import FileCatalog
from rexgraph.io.rcbf import MAGIC

from rcql import Executor, parse


def _cstring(value: str) -> bytes:
    return value.encode("utf-8") + b"\0"


def _write_branching_rcbf(path):
    """Write an RCBF v5 fixture directly, independent of the reader under test."""
    supports = ((0, 1, 2), (2, 3))
    n_vertices, n_edges = 4, 2
    parts = [
        struct.pack("<8s7IQ", MAGIC, 5, 0, n_vertices, n_edges, 0, 0, 0, 0),
        b"".join(_cstring(f"gene_{vertex}") for vertex in range(n_vertices)),
        np.asarray([0, 2], dtype="<i4").tobytes(),
        np.asarray([1, 3], dtype="<i4").tobytes(),
        np.asarray([1.0, 2.0], dtype="<f8").tobytes(),
        np.ones(n_edges, dtype="<f8").tobytes(),
        np.ones(n_edges, dtype="u1").tobytes(),
        np.zeros(n_edges, dtype="u1").tobytes(),
        b"".join(_cstring(f"relation_{edge}") for edge in range(n_edges)),
        b"".join(_cstring("drug_gene") for _ in range(n_edges)),
        np.zeros((n_edges, 4), dtype="<f8").tobytes(),
        np.asarray([3, 2], dtype="<u2").tobytes(),
        np.asarray([vertex for support in supports for vertex in support], dtype="<i4").tobytes(),
        struct.pack("<I", 0),  # no stored temporal states
    ]
    path.write_bytes(b"".join(parts))


def test_rcbf_can_be_loaded_from_a_catalog_stored_in_rcdb_and_queried(tmp_path):
    path = tmp_path / "drug_gene.rcbf"
    _write_branching_rcbf(path)
    catalog = FileCatalog([tmp_path])

    # The FILE source form proves that RCQL sees the primary 3-ary C1 cell directly.
    catalog_result = Executor(sources={"files": catalog}).execute(parse(
        'FROM FILE("files", "root0/drug_gene.rcbf") '
        'RETURN BETTI(0), BETTI(1), ARITY(CELL(1, 0))'
    ))
    # The exact C1 rank is two: support connectivity is one component, while
    # the relational complex β0 invariant correctly remains 4 - rank(B1) = 2.
    assert catalog_result.values == (2, 0, 3)

    # RCDB persists the loaded relational complex; RCDB_GET then makes that exact
    # stored basis the source of a structural RCQL phrase.
    store = MemoryStore()
    store.put("drug-gene", catalog.load("root0/drug_gene.rcbf"), analytics=False)
    stored_result = Executor(sources={"db": store}).execute(parse(
        'FROM RCDB_GET($db, "drug-gene") '
        'RETURN BETTI(0), BETTI(1), ARITY(CELL(1, 0))'
    ))
    assert stored_result.values == (2, 0, 3)
