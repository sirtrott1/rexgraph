"""
Regression tests for the second audit-fix pass (AUDIT_REPORT_2).

Pure-Python-safe: these exercise routing / wiring that does not require
the compiled Cython core. Kernel-dependent behaviour is verified
separately on a compiled build.
"""

import os
import queue

import numpy as np
import pytest


# P1: pipeline ingestion is OCR-lazy
def test_ingest_direct_only_needs_no_ocr_client(tmp_path):
    """A CSV-only batch must ingest with ocr_client=None (no OCR init)."""
    from agent.server.routes.pipeline import _ingest_files

    csv = tmp_path / "data.csv"
    csv.write_text("source,target\na,b\nb,c\n")
    q = queue.Queue()
    ocr_texts, ocr_ids, direct = _ingest_files(
        [str(csv)], None, q, use_fusion=False
    )
    assert ocr_texts == [] and ocr_ids == []
    assert direct == [str(csv)]


def test_ingest_ocr_file_without_client_is_skipped(tmp_path):
    """Defensive: an OCR file with no client is skipped, not a crash."""
    from agent.server.routes.pipeline import _ingest_files

    png = tmp_path / "scan.png"
    png.write_bytes(b"\x89PNG\r\n")  # not a real image; just needs the ext
    q = queue.Queue()
    ocr_texts, ocr_ids, direct = _ingest_files(
        [str(png)], None, q, use_fusion=False
    )
    assert ocr_texts == [] and direct == []  # skipped, no exception


# P2: mixed batch keeps both OCR text and direct files
def test_mixed_batch_keeps_both(tmp_path):
    from agent.pipeline_runner import PipelineRunner

    csv = tmp_path / "edges.csv"
    csv.write_text("source,target,weight\na,b,1\nb,c,2\nc,a,1\n")
    r = PipelineRunner()
    r._stage_callback = None
    corpus = r._build_corpus(
        ["some ocr text about genes and cells and pathways"],
        ["ocrdoc"],
        [str(csv)],
        depth="quick",
    )
    ids = {d.doc_id for d in corpus.documents}
    assert "ocrdoc" in ids
    assert "edges" in ids  # the CSV is no longer dropped


# P3/max_vocab leak: non-text adapters tolerate extra kwargs
def test_edge_and_feature_adapters_ignore_extra_kwargs(tmp_path):
    from agent.adapters.edge_list import EdgeListAdapter
    from agent.adapters.feature_matrix import FeatureMatrixAdapter

    csv = tmp_path / "e.csv"
    csv.write_text("source,target\na,b\nb,c\nc,a\n")
    # max_vocab is a text-only concept; must be ignored, not raise.
    ec = EdgeListAdapter().build(str(csv), max_vocab=200, window=3)
    assert ec.nE >= 1

    # The regression is specifically that a stray max_vocab kwarg no longer
    # raises TypeError. `typing="none"` avoids the spectral-clustering
    # kernel so this passes without the compiled core too.
    X = np.random.default_rng(0).standard_normal((20, 5))
    try:
        FeatureMatrixAdapter().build(X, max_vocab=200, typing="none")
    except TypeError as e:
        pytest.fail("max_vocab leaked as a TypeError: %s" % e)
    except ModuleNotFoundError:
        pass  # compiled kernel absent in a source checkout - not our concern


# B1: builder collects documents / chunks / query_results
def test_builder_collects_results(tmp_path):
    from agent.builder import AgentBuilder

    doc = tmp_path / "d.txt"
    doc.write_text(
        "Cells signal through receptors. Genes express proteins. "
        "Pathways regulate the cell cycle across tissues and organs."
    )
    b = AgentBuilder({
        "name": "t",
        "steps": [
            {"type": "corpus", "params": {"depth": "quick"}},
            {"type": "chunk"},
        ],
    })
    r = b.run(files=[str(doc)])
    # documents is now populated from state["corpus"] (was always empty).
    assert isinstance(r.documents, list)
    assert len(r.documents) == 1
    assert isinstance(r.chunks, list)  # collected, even if empty w/o kernels


# P5: vllm_router normalises channel_map keys
def test_router_channel_map_int_keys():
    from agent.integrations.vllm_router import RexRouter
    r = RexRouter(
        models={"reasoning": "m1", "creative": "m2"},
        channel_map={"0": "reasoning", "1": "creative"},
    )
    assert set(r.channel_map.keys()) == {0, 1}
    assert all(isinstance(k, int) for k in r.channel_map)


# C1: rexgraph-ocr entry point is a real CLI now
def test_ocr_cli_entrypoint_exists():
    from agent.cli.ocr import ocr_main
    assert callable(ocr_main)
    # no-args prints help and returns 0
    assert ocr_main([]) == 0


def test_new_cli_mains_exist():
    from agent.cli.serve import main as serve_main
    from agent.cli.setup import main as setup_main
    from agent.cli.config import main as config_main
    assert callable(serve_main) and callable(setup_main) and callable(config_main)
    # config show/platform run without side effects
    assert config_main(["platform"]) == 0


# B4 regression: prose .txt (with commas) must not fail as CSV
def test_prose_txt_falls_back_to_text(tmp_path):
    from agent.auto import detect_input_type, _fallback_text_or_raise
    from agent.adapters.text import TextAdapter

    doc = tmp_path / "prose.txt"
    doc.write_text(
        "Cells signal through receptors, ligands, and channels. "
        "Genes express proteins, which regulate transcription, "
        "translation, and the cell cycle across tissues, organs, "
        "and developing embryos over long stretches of time."
    )
    # It may be classified as a csv type by the content peek; the fallback
    # helper must still yield a usable text construction rather than raise.
    ec = _fallback_text_or_raise(
        str(doc), "edge_csv", RuntimeError("Empty CSV"),
    )
    assert ec.nE > 0

    # And a genuine edge CSV must still classify as edge_csv (no over-fallback).
    csv = tmp_path / "e.csv"
    csv.write_text("source,target\na,b\nb,c\nc,a\n")
    assert detect_input_type(str(csv)) == "edge_csv"


# Void tractability gate: dense multigraph uses the spectral path
def test_void_gate_dense_vs_sparse():
    """A dense typed multigraph must route to the spectral/quotient void
    path (fast, homologically meaningful); a small complex stays exact."""
    import numpy as np
    from agent.pipeline import AnalysisPipeline

    class FakeRex:
        def __init__(self, deg):
            self.degree = np.asarray(deg)

    p = AnalysisPipeline.__new__(AnalysisPipeline)  # no __init__ needed
    # sparse: 20 vertices, degree ~6 -> well under the cap
    assert p._void_bruteforce_intractable(FakeRex([6] * 20)) is False
    # dense multigraph: 6 vertices, degree ~300 -> over the cap
    assert p._void_bruteforce_intractable(FakeRex([300] * 6)) is True
