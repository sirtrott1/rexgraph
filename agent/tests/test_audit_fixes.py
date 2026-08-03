"""
Tests for the audit-driven agent fixes.

These exercise the parts that run without the compiled Cython core, so
they pass in a source checkout.  Paths that require compiled kernels
(full Hodge/spectral/persistence) are covered structurally - we assert
the wiring is correct and degrades gracefully - and run fully once the
extensions are built.
"""

import gzip
import os
import tempfile

import numpy as np
import pytest


# 1.3 L-R interaction adapter
def test_lr_interaction_adapter():
    import pandas as pd
    from agent.adapters.lr_interaction import LRInteractionAdapter

    df = pd.DataFrame(
        [[5.0, 0.0, 1.0, 0.0],
         [0.0, 4.0, 0.0, 3.0],
         [2.0, 0.0, 0.0, 2.0]],
        index=["Fibroblast", "T_cell", "Endothelial"],
        columns=["TGFB1", "CD3D", "TGFBR1", "KDR"],
    )
    ec = LRInteractionAdapter().build(
        df, lr_pairs=[("TGFB1", "TGFBR1"), ("TGFB1", "KDR")]
    )
    assert ec.nE > 0
    assert ec.vertex_labels == ["Fibroblast", "T_cell", "Endothelial"]
    assert ec.input_type == "lr_interaction"
    # weights normalised to [0, 1]
    assert ec.weights.max() <= 1.0 + 1e-9


def test_lr_adapter_no_matching_pairs():
    from agent.adapters.lr_interaction import LRInteractionAdapter
    ec = LRInteractionAdapter().build(
        np.eye(3), lr_pairs=[("NOPE", "ALSO_NOPE")],
        gene_names=["a", "b", "c"], cell_types=["x", "y", "z"],
    )
    assert ec.nE == 0  # graceful empty, no crash


# 1.2 single-cell / 10X adapter
def _make_synthetic_10x(dirpath):
    from scipy import sparse
    from scipy.io import mmwrite

    genes = ["CD3D", "CD3E", "LYZ", "CD68", "PECAM1", "VWF", "TGFB1", "TGFBR1"]
    rng = np.random.default_rng(1)
    cells = 30
    G = np.zeros((len(genes), cells))
    G[0, :10] = rng.poisson(8, 10); G[1, :10] = rng.poisson(6, 10)
    G[2, 10:20] = rng.poisson(9, 10); G[3, 10:20] = rng.poisson(5, 10)
    G[4, 20:] = rng.poisson(7, 10); G[5, 20:] = rng.poisson(6, 10)
    G[6, 20:] = rng.poisson(5, 10); G[7, :10] = rng.poisson(4, 10)

    tmp = os.path.join(dirpath, "matrix.mtx")
    mmwrite(tmp, sparse.csr_matrix(G))
    with open(tmp, "rb") as f:
        data = f.read()
    with gzip.open(os.path.join(dirpath, "matrix.mtx.gz"), "wb") as f:
        f.write(data)
    os.remove(tmp)
    with gzip.open(os.path.join(dirpath, "features.tsv.gz"), "wt") as f:
        for g in genes:
            f.write(f"ENSG_{g}\t{g}\tGene Expression\n")
    with gzip.open(os.path.join(dirpath, "barcodes.tsv.gz"), "wt") as f:
        for i in range(cells):
            f.write("BC%d-1\n" % i)
    return genes


def test_single_cell_adapter():
    from agent.adapters.single_cell import (
        SingleCellAdapter,
        is_10x_dir,
        load_10x,
    )
    with tempfile.TemporaryDirectory() as d:
        _make_synthetic_10x(d)
        assert is_10x_dir(d)
        cxg, bc, gn = load_10x(d)
        assert cxg.shape == (30, 8)

        markers = {
            "T_cell": ["CD3D", "CD3E"],
            "Myeloid": ["LYZ", "CD68"],
            "Endothelial": ["PECAM1", "VWF"],
        }
        ec = SingleCellAdapter().build(
            d, markers=markers, lr_pairs=[("TGFB1", "TGFBR1")]
        )
        counts = ec.cell_type_expression["label_counts"]
        assert counts == {"Endothelial": 10, "Myeloid": 10, "T_cell": 10}
        assert ec.nE >= 1


def test_single_cell_detected_by_auto():
    from agent.auto import detect_input_type
    with tempfile.TemporaryDirectory() as d:
        _make_synthetic_10x(d)
        assert detect_input_type(d) == "single_cell"


# 2.2 OCR table detection
def test_detect_tables_pipe():
    from agent.adapters.table_detect import detect_tables
    text = (
        "prose line one that should be ignored entirely\n\n"
        "gene | tumor_A | tumor_B | tumor_C\n"
        "EGFR | 5.2 | 3.1 | 8.0\n"
        "TP53 | 1.0 | 2.2 | 0.5\n"
        "MYC | 4.4 | 4.1 | 3.9\n"
    )
    frames = detect_tables(text)
    assert len(frames) == 1
    assert list(frames[0].columns) == ["gene", "tumor_A", "tumor_B", "tumor_C"]
    numeric = frames[0].select_dtypes(include=["number"])
    assert numeric.shape[1] == 3


def test_detect_tables_whitespace():
    from agent.adapters.table_detect import detect_tables
    text = ("sample    score    depth\n"
            "A01       0.91     30\n"
            "A02       0.44     22\n"
            "A03       0.72     41\n")
    frames = detect_tables(text)
    assert len(frames) == 1
    assert list(frames[0].columns) == ["sample", "score", "depth"]


def test_no_table_in_prose():
    from agent.adapters.table_detect import text_has_table
    assert not text_has_table(
        "This is ordinary prose with no tabular structure whatsoever. "
        "It has several sentences. None of them form columns."
    )


# EdgeConstruction fast path + shared rex construction
def test_build_rex_from_edges_fastpath():
    from agent.adapters import EdgeConstruction
    from agent.auto import auto_rex, build_rex_from_edges
    ec = EdgeConstruction(
        sources=np.array([0, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 0], dtype=np.int32),
        weights=np.ones(3), signs=np.ones(3),
        type_labels=np.zeros(3, dtype=np.int32),
        vertex_labels=["a", "b", "c"], n_types=1, type_names=["t"],
    )
    r1 = auto_rex(ec, face_selection="none")
    r2 = build_rex_from_edges(ec, face_selection="none")
    assert r1.nV == r2.nV == 3
    assert r1.nE == r2.nE == 3


# 2.1 OCRAdapter.build_from_text without re-OCR
def test_ocr_build_from_text_layout():
    from agent.adapters.ocr import OCRAdapter
    text = (
        "INTRODUCTION\n"
        "This document describes the study design and its many facets.\n\n"
        "METHODS\n"
        "We collected samples across several cohorts and analysed them.\n\n"
        "RESULTS\n"
        "The findings were consistent across all measured conditions here.\n"
    )
    ec = OCRAdapter().build_from_text(text, strategy="layout", detect_tables=False)
    assert ec.nE > 0


# 3.3 sparse eigensolver
def test_sparse_eigensolver_path_graph():
    from agent.pipeline import _smallest_eigenvalues_L0
    from scipy import sparse

    class Fake:
        nV = 6
        L0 = None

    f = Fake()
    L = np.array([
        [1, -1, 0, 0, 0, 0], [-1, 2, -1, 0, 0, 0], [0, -1, 2, -1, 0, 0],
        [0, 0, -1, 2, -1, 0], [0, 0, 0, -1, 2, -1], [0, 0, 0, 0, -1, 1],
    ], float)
    f.L0 = sparse.csr_matrix(L)
    ev = _smallest_eigenvalues_L0(f, k=4)
    assert ev is not None
    assert abs(ev[0]) < 1e-6           # connected graph -> smallest eval ~ 0
    assert ev[1] > 0                    # Fiedler value positive


# 3.4 cache module
def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("REXGRAPH_NO_CACHE", raising=False)
    from agent import cache
    key = cache.content_key("hello world", depth="standard")
    assert cache.get(key) is None
    assert cache.set(key, {"analysis": {"x": 1}})
    got = cache.get(key)
    assert got["analysis"]["x"] == 1


def test_cache_disabled(monkeypatch):
    monkeypatch.setenv("REXGRAPH_NO_CACHE", "1")
    from agent import cache
    key = cache.content_key("abc", depth="quick")
    assert cache.set(key, {"a": 1}) is False
    assert cache.get(key) is None


# 4.2 stage callbacks flow through CorpusBuilder.build
def test_corpus_stage_callbacks():
    from agent.adapters import EdgeConstruction
    from agent.corpus import CorpusBuilder
    ec = EdgeConstruction(
        sources=np.array([0, 1, 2, 3, 0], dtype=np.int32),
        targets=np.array([1, 2, 3, 0, 2], dtype=np.int32),
        weights=np.ones(5), signs=np.ones(5),
        type_labels=np.zeros(5, dtype=np.int32),
        vertex_labels=["a", "b", "c", "d"], n_types=1, type_names=["t"],
    )
    cb = CorpusBuilder()
    cb.add_document(source="<pre>", doc_id="doc1", edge_construction=ec)
    seen = []
    cb.build(depth="quick",
             stage_callback=lambda did, stage, data: seen.append((did, stage)))
    stages = [s for _, s in seen]
    assert "construction" in stages and "topology" in stages
    assert cb.documents[0].analysis["construction"]["nV"] == 4


# diagnostics (3.2) runs and reports structure
def test_diagnostics_summary():
    from agent.diagnostics import format_report, summary
    s = summary()
    assert "modules" in s and "method_dispatch" in s
    assert isinstance(format_report(), str)


# AnalysisPipeline optional stages degrade gracefully
def test_optional_stages_graceful_without_faces():
    from agent.adapters import EdgeConstruction
    from agent.auto import build_rex_from_edges
    from agent.pipeline import AnalysisPipeline
    ec = EdgeConstruction(
        sources=np.array([0, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 0], dtype=np.int32),
        weights=np.ones(3), signs=np.ones(3),
        type_labels=np.zeros(3, dtype=np.int32),
        vertex_labels=["a", "b", "c"], n_types=1, type_names=["t"],
    )
    rex = build_rex_from_edges(ec, face_selection="none")
    pipe = AnalysisPipeline(rex)
    # directly exercise the optional stages
    assert pipe._stage_sigma_sweep().get("available") is False
    assert pipe._stage_ricci_flow().get("available") is False
    # continuum limit needs a small spectrum; on tiny graphs it may skip
    cl = pipe._stage_continuum_limit()
    assert "available" in cl


# auto_rex must treat long text as text, not as a filesystem path
def test_auto_rex_accepts_text_longer_than_the_filename_limit():
    """auto.py probed Path(data).is_file() on the raw input. On Python 3.13 that
    propagates OSError ENAMETOOLONG for any text carrying a path-like segment past
    the filesystem name limit, so 12% of a real corpus was dropped with a warning."""
    from agent.auto import auto_rex

    # a path-like segment well past the 255-byte component limit
    text = ("alpha beta gamma delta " * 40) + "\n" + ("x" * 400) + "\nalpha beta gamma delta\n"
    assert len(text) > 255
    rex = auto_rex(text)
    assert rex is not None
    assert int(rex.nE) > 0


def test_auto_rex_still_reads_a_real_file_path():
    """The path branch must keep working for short strings that are real files."""
    from agent.auto import auto_rex

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "doc.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("alpha beta gamma delta alpha beta gamma alpha beta\n" * 6)
        rex = auto_rex(p)
        assert rex is not None and int(rex.nE) > 0


# open_secret_store must not silently turn an unknown scheme into a filename
def test_open_secret_store_rejects_an_unrecognized_scheme():
    """The fallback returned FileSecretStore(uri), so 'vault://team/prod' became a
    file literally named 'vault://team/prod' and every secret went to the wrong place
    while reporting success."""
    from agent.secrets import open_secret_store

    with pytest.raises(ValueError) as ei:
        open_secret_store("vault://team/prod")
    assert "vault" in str(ei.value)


def test_open_secret_store_accepts_a_bare_path_and_known_schemes(tmp_path):
    """A bare path stays a file store, and the two supported schemes keep working."""
    from agent.secrets import EnvSecretStore, FileSecretStore, open_secret_store

    assert isinstance(open_secret_store(str(tmp_path / "conn.json")), FileSecretStore)
    assert isinstance(open_secret_store("file://" + str(tmp_path / "c.json")), FileSecretStore)
    assert isinstance(open_secret_store("env://"), EnvSecretStore)


# partition_communities must not raise NameError past its early return
def test_partition_communities_runs_past_the_early_return():
    """graph.py called _standard.build_adj_weights but never bound _standard, so any
    graph with nE > max_size raised NameError. The early return hid it for small ones."""
    import numpy as np

    from rexgraph.graph import RexGraph

    rex = RexGraph.from_graph(np.arange(10, dtype=np.int32),
                              np.arange(1, 11, dtype=np.int32))
    parts = rex.partition_communities(max_size=3)      # nE = 10 > 3, so it must do work
    assert parts is not None


# Packaging and export gaps
def test_curvature_kernel_is_exported_from_the_core_namespace():
    """_curvature is compiled (core/meson.build) but was absent from __init__'s _MODULES,
    so five public kernel functions were reachable only by full dotted path."""
    import rexgraph.core as core

    assert hasattr(core, "_curvature")
    for name in ("lagrangian_curvature", "star_curvature", "weighted_degree",
                 "curvature_operator", "lagrangian_L_T_integer"):
        assert hasattr(core, name), f"core.{name} missing from the flattened namespace"


def test_harmonic_wrapper_is_declared_for_installation():
    """rexgraph/harmonic.py is a public wrapper over core._harmonic but was not listed in
    rexgraph/meson.build, so it was absent from an installed package."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    build = (root / "rexgraph" / "meson.build").read_text(encoding="utf-8")
    assert "'harmonic.py'" in build


def test_inverse_centrality_ratio_does_not_warn_on_an_isolated_vertex():
    """np.where(deg > 0, med / deg, 0.0) evaluated med/deg for every entry first, so any
    complex with a zero-degree vertex emitted a divide-by-zero RuntimeWarning."""
    import warnings

    import numpy as np

    from rexgraph.graph import RexGraph

    # the edge list skips vertex 2, so nV covers it and its degree is 0
    rex = RexGraph(sources=np.array([0, 1, 3], dtype=np.int32),
                   targets=np.array([1, 3, 4], dtype=np.int32))
    assert int(np.asarray(rex.degree)[2]) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mu = rex.inverse_centrality_ratio
    assert np.all(np.isfinite(mu))
