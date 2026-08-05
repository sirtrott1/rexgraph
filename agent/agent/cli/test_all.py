#!/usr/bin/env python3
"""
rexgraph-test: exercise every code path in the agent repo.

Usage:
    python -m agent.cli.test_all                   # run all tests
    python -m agent.cli.test_all --only ocr        # just OCR
    python -m agent.cli.test_all --only pipeline   # just pipeline
    python -m agent.cli.test_all --file paper.pdf  # test with a specific file
    python -m agent.cli.test_all --verbose          # show details
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback


class TestRunner:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.passed = []
        self.failed = []
        self.skipped = []

    def run(self, name, fn, skip_reason=None):
        if skip_reason:
            self.skipped.append((name, skip_reason))
            print(f"  SKIP  {name} - {skip_reason}")
            return
        try:
            t0 = time.time()
            result = fn()
            elapsed = time.time() - t0
            self.passed.append(name)
            detail = f" ({result})" if result else ""
            print(f"  OK    {name} ({elapsed:.1f}s){detail}")
        except Exception as e:
            self.failed.append((name, str(e)))
            print(f"  FAIL  {name} - {e}")
            if self.verbose:
                traceback.print_exc()

    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print("\n" + "=" * 50)
        print("  %d passed, %d failed, %d skipped (of %d)" % (
            len(self.passed), len(self.failed), len(self.skipped), total))
        if self.failed:
            print("\n  Failed:")
            for name, err in self.failed:
                print(f"    {name}: {err}")
        print("=" * 50)
        return len(self.failed) == 0


# Test groups

def test_imports(t):
    """Test that all agent modules import cleanly."""
    print("\n── Imports ──")
    modules = [
        ("agent.auto", "auto_rex"),
        ("agent.pipeline", "AnalysisPipeline"),
        ("agent.pipeline_runner", "PipelineRunner"),
        ("agent.model_manager", "get_manager"),
        ("agent.corpus", "CorpusBuilder"),
        ("agent.adapters.text", "TextAdapter"),
        ("agent.adapters.edge_list", None),
        ("agent.adapters.feature_matrix", None),
        ("agent.adapters.correlation", None),
        ("agent.integrations.unlimited_ocr", "create_ocr_client"),
        ("agent.integrations.trustgraph_adapter", None),
        ("agent.server.auth", "AuthManager"),
        ("agent.server.persistence", None),
        ("agent.server.state", "SessionStore"),
        ("agent.cli.config", "detect_platform"),
        ("agent.cli.serve", None),
        ("agent.cli.hpc", None),
        ("agent.cli.setup", None),
    ]
    for mod, attr in modules:
        def _test(m=mod, a=attr):
            import importlib
            module = importlib.import_module(m)
            if a:
                assert hasattr(module, a), f"missing {a}"
            return a or "ok"
        t.run(f"import {mod}", _test)


def test_platform(t):
    """Test platform detection."""
    print("\n── Platform ──")
    def _detect():
        from agent.cli.config import detect_platform
        p = detect_platform()
        return "gpu=%s, ram=%dGB" % (p.gpu, p.system_ram_gb)
    t.run("detect_platform", _detect)

    def _config():
        from agent.cli.config import load_config
        cfg = load_config()
        return f"cache={cfg.cache_dir}"
    t.run("load_config", _config)


def test_rexgraph_core(t):
    """Test the Cython core."""
    print("\n── RexGraph Core ──")
    def _basic():
        from rexgraph.graph import RexGraph
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2])
        assert rex.nV == 3
        assert rex.nE == 3
        return "%dV %dE betti=%s" % (rex.nV, rex.nE, rex.betti)
    t.run("RexGraph.from_graph", _basic)

    def _text_adapter():
        from agent.adapters.text import TextAdapter
        ta = TextAdapter()
        text = "The quick brown fox jumps over the lazy dog. " * 20
        edges = ta.build(text, max_vocab=50)
        return "%d edges, %d types" % (len(edges.sources), edges.n_types)
    t.run("TextAdapter.build", _text_adapter)

    def _auto_rex():
        from agent.auto import auto_rex
        text = "The quick brown fox jumps over the lazy dog. " * 20
        rex = auto_rex(text, max_vocab=50, min_count=2)
        if rex is None:
            return "None (text too short)"
        return "%dV %dE %dF" % (rex.nV, rex.nE, rex.nF)
    t.run("auto_rex", _auto_rex)

    def _analysis():
        from agent.auto import auto_rex
        from agent.pipeline import AnalysisPipeline
        text = "The quick brown fox jumps over the lazy dog. " * 30
        text += "Physics is the study of nature and natural phenomena. " * 30
        rex = auto_rex(text, max_vocab=50, min_count=2)
        if rex is None:
            return "skipped (no rex)"
        pipe = AnalysisPipeline(rex)
        results = pipe.run(depth="quick")
        return "%d stages" % len(results)
    t.run("AnalysisPipeline.run", _analysis)


def test_ocr(t, test_file=None):
    """Test OCR backends."""
    print("\n── OCR ──")
    import shutil

    def _client():
        from agent.integrations.unlimited_ocr import create_ocr_client
        client = create_ocr_client()
        return type(client).__name__
    t.run("create_ocr_client", _client)

    has_tesseract = shutil.which("tesseract") is not None
    def _tesseract():
        from agent.integrations.unlimited_ocr import OfflineOCRClient
        client = OfflineOCRClient()
        assert client.is_available()
        return "available"
    t.run("tesseract", _tesseract,
          skip_reason=None if has_tesseract else "tesseract not installed")

    def _got_status():
        import importlib
        from pathlib import Path
        libs = (importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("torch") is not None)
        model_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--stepfun-ai--GOT-OCR-2.0-hf"
        snapshots = model_dir / "snapshots"
        model = snapshots.exists() and any(snapshots.iterdir())
        return f"libs={libs}, model={model}"
    t.run("GOT-OCR status", _got_status)

    if test_file and os.path.isfile(test_file):
        def _ocr_file():
            from agent.integrations.unlimited_ocr import create_ocr_client, is_pdf_file
            client = create_ocr_client()
            if is_pdf_file(test_file):
                result = client.ocr_pdf(test_file)
                return "%d chars, %d pages" % (len(result.full_text), len(result.pages))
            else:
                result = client.ocr_image(test_file)
                return "%d chars" % len(result.text)
        t.run(f"OCR file: {os.path.basename(test_file)}", _ocr_file)


def test_pipeline(t, test_file=None):
    """Test the full pipeline."""
    print("\n── Pipeline ──")

    def _runner_text():
        from agent.pipeline_runner import PipelineRunner
        runner = PipelineRunner()
        phases = []
        runner.on_phase(lambda p, d: phases.append(p))
        texts = ["The quick brown fox jumps over the lazy dog. " * 30 +
                 "Physics studies nature and natural phenomena. " * 30]
        doc_ids = ["test_doc"]
        result = runner.run(texts=texts, doc_ids=doc_ids, depth="quick")
        return "%d docs, %d phases, %.1fs" % (
            len(result.documents), len(phases), result.elapsed)
    t.run("PipelineRunner.run(texts=...)", _runner_text)

    if test_file and os.path.isfile(test_file):
        def _runner_file():
            from agent.pipeline_runner import PipelineRunner
            runner = PipelineRunner()
            phases = []
            runner.on_phase(lambda p, d: phases.append(p))
            result = runner.run(files=[test_file], depth="quick")
            return "%d docs, %d chunks, %.1fs" % (
                len(result.documents),
                sum(len(c.get("chunks", [])) for c in result.chunks) if result.chunks else 0,
                result.elapsed)
        t.run(f"PipelineRunner.run(file={os.path.basename(test_file)})", _runner_file)


def test_corpus(t):
    """Test corpus builder."""
    print("\n── Corpus ──")

    def _build():
        from agent.corpus import CorpusBuilder
        corpus = CorpusBuilder()
        corpus.add_text("The fox jumps over the dog. " * 30, doc_id="doc1")
        corpus.add_text("Physics studies natural phenomena. " * 30, doc_id="doc2")
        corpus.build(depth="quick")
        summary = corpus.summary()
        return "%d docs" % summary.get("n_documents", 0)
    t.run("CorpusBuilder.build", _build)


def test_model_manager(t):
    """Test model manager."""
    print("\n── Model Manager ──")

    def _scan():
        from agent.model_manager import get_manager
        mgr = get_manager()
        models = mgr.scan()
        downloaded = sum(1 for m in models if m.downloaded)
        return "%d known, %d downloaded" % (len(models), downloaded)
    t.run("ModelManager.scan", _scan)

    def _status():
        from agent.model_manager import get_manager
        mgr = get_manager()
        s = mgr.status()
        return "%d loaded, %d available" % (s["n_loaded"], s["n_available"])
    t.run("ModelManager.status", _status)

    def _pipeline_config():
        from agent.model_manager import get_manager
        mgr = get_manager()
        pc = mgr.pipeline_config()
        return str(pc) if pc else "(empty)"
    t.run("pipeline_config", _pipeline_config)


def test_sessions(t):
    """Test session store."""
    print("\n── Sessions ──")

    def _store():
        from agent.server.state import SessionStore
        store = SessionStore()
        sessions = store.list_all()
        return "%d sessions" % len(sessions)
    t.run("SessionStore.list_all", _store)


def test_auth(t):
    """Test auth system."""
    print("\n── Auth ──")

    def _auth():
        from agent.server.auth import get_auth_manager
        mgr = get_auth_manager()
        return "enabled=%s, tokens=%d" % (mgr.auth_enabled, len(mgr.list_tokens()))
    t.run("AuthManager", _auth)


def test_server_routes(t):
    """Test that all route modules import and have routers."""
    print("\n── Server Routes ──")
    routes = [
        "agent.server.routes.admin",
        "agent.server.routes.analysis",
        "agent.server.routes.chat",
        "agent.server.routes.corpus",
        "agent.server.routes.explore",
        "agent.server.routes.export",
        "agent.server.routes.integrations",
        "agent.server.routes.model",
        "agent.server.routes.models",
        "agent.server.routes.ocr",
        "agent.server.routes.pipeline",
        "agent.server.routes.session",
        "agent.server.routes.upload",
    ]
    for route in routes:
        def _test(r=route):
            import importlib
            mod = importlib.import_module(r)
            assert hasattr(mod, "router"), "no router"
            n = len(mod.router.routes)
            return "%d endpoints" % n
        t.run(route.split(".")[-1], _test)


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Test all rexgraph agent code paths")
    parser.add_argument("--only", choices=[
        "imports", "platform", "core", "ocr", "pipeline",
        "corpus", "models", "sessions", "auth", "routes", "all"],
        default="all")
    parser.add_argument("--file", help="PDF/image to test OCR and pipeline with")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("rexgraph agent - test runner\n")

    t = TestRunner(verbose=args.verbose)
    groups = {
        "imports": lambda: test_imports(t),
        "platform": lambda: test_platform(t),
        "core": lambda: test_rexgraph_core(t),
        "ocr": lambda: test_ocr(t, args.file),
        "pipeline": lambda: test_pipeline(t, args.file),
        "corpus": lambda: test_corpus(t),
        "models": lambda: test_model_manager(t),
        "sessions": lambda: test_sessions(t),
        "auth": lambda: test_auth(t),
        "routes": lambda: test_server_routes(t),
    }

    if args.only == "all":
        for fn in groups.values():
            fn()
    else:
        groups[args.only]()

    ok = t.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
