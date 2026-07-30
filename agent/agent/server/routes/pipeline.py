"""
agent.server.routes.pipeline: SSE pipeline endpoint.

Architecture:
  - Files are split by type: OCR (PDF/image) vs direct (CSV/TSV/JSON/text)
  - OCR runs in the main process (model cached in VRAM)
  - Analysis runs in a subprocess (crash-isolated from server)
  - Non-OCR files are passed as paths to auto_rex which routes them
    through the correct loader (csv_loader, json_loader, etc.)
  - Results feed into both Sessions and Workspace persistence
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import math
import multiprocessing
import os
import queue as queue_module
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/pipeline")

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Isolated + warm analysis workers. A `forkserver` imports the heavy stack once,
# then each analysis forks from it cheaply; a crash/OOM in a child never touches
# the server. Created lazily and cached (the first request pays the one-time
# forkserver+preload startup; the rest are fast forks).
_ANALYSIS_CTX = None


def _analysis_ctx():
    global _ANALYSIS_CTX
    if _ANALYSIS_CTX is None:
        ctx = multiprocessing.get_context("forkserver")
        try:
            ctx.set_forkserver_preload(
                ["numpy", "scipy", "agent.pipeline_runner"])
        except Exception:
            pass  # preload is an optimization; forks still work without it
        _ANALYSIS_CTX = ctx
    return _ANALYSIS_CTX

OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


# OCR client cache

def _get_ocr_client(backend=None):
    from agent.model_manager import get_manager
    mgr = get_manager()
    if not backend or backend == "auto":
        assigned = mgr.get_pipeline_model("ocr")
        if assigned:
            lm = mgr.get_or_load(assigned, purpose="ocr")
            if lm.model_obj is not None and hasattr(lm.model_obj, 'generate'):
                from agent.integrations.unlimited_ocr import GOTOCRClient
                client = GOTOCRClient(
                    model_name=assigned,
                    device=lm.device or "auto",
                    skip_load=True,
                )
                client._model = lm.model_obj
                client._processor = lm.processor_obj
                return client
    from agent.integrations.unlimited_ocr import create_ocr_client
    if backend and backend != "auto":
        return create_ocr_client(prefer=backend)
    return create_ocr_client()


# Helpers

def _sanitize(obj):
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return _sanitize(obj.tolist())
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _save_uploads(files):
    paths = []
    for f in files:
        suffix = os.path.splitext(f["name"])[1] or ".bin"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="rexgraph_")
        os.close(fd)
        with open(path, "wb") as fh:
            fh.write(f["content"])
        paths.append(path)
    return paths


def _cleanup_temps(paths):
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


# File ingestion (main process)

def _ingest_files(file_paths, ocr_client, event_queue, use_fusion=False):
    """Split files by type. OCR images/PDFs, pass everything else through.

    Returns (ocr_texts, ocr_doc_ids, direct_paths).

    When ``use_fusion`` is set and more than one OCR backend is available,
    each document is run through every backend and the highest-confidence
    result is kept (audit 2.3).
    """
    import re
    import sys
    from agent.integrations.unlimited_ocr import is_pdf_file, is_image_file

    print("[ingest] %d files, OCR client: %s" % (
        len(file_paths), type(ocr_client).__name__),
        file=sys.stderr, flush=True)

    event_queue.put(("phase", "ingest",
        {"status": "running", "n_files": len(file_paths)}))

    ocr_texts, ocr_doc_ids, direct_paths = [], [], []

    # Decide whether fusion is actually possible (needs 2+ backends).
    fusion = None
    if use_fusion:
        try:
            from agent.integrations.ocr_fusion import OCRFusion
            candidate = OCRFusion()
            if len(candidate._resolve_clients()) >= 2:
                fusion = candidate
                print("[ingest] OCR fusion enabled (%d backends)"
                      % len(fusion._resolve_clients()),
                      file=sys.stderr, flush=True)
        except Exception as e:
            print("[ingest] fusion unavailable: %s" % e,
                  file=sys.stderr, flush=True)

    for path in file_paths:
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix in OCR_EXTENSIONS:
            try:
                text = ""
                if fusion is not None:
                    try:
                        report = fusion.compare(path)
                        text = report.best_text()
                    except Exception as e:
                        print("[ingest]   %s -> fusion failed (%s), "
                              "falling back" % (p.name, e),
                              file=sys.stderr, flush=True)
                        text = ""
                if not text:
                    if ocr_client is None:
                        # No OCR backend was initialised (direct-only
                        # batch). Skip rather than dereference None.
                        print("[ingest]   %s -> skipped (no OCR backend)"
                              % p.name, file=sys.stderr, flush=True)
                        continue
                    if is_pdf_file(path):
                        text = ocr_client.ocr_pdf(path).full_text
                    elif is_image_file(path):
                        text = ocr_client.ocr_image(path).text
                    else:
                        text = ""

                if text:
                    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
                    text = re.sub(r'[ \t]+', ' ', text)
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    lines = text.split('\n')
                    lines = [ln for ln in lines if len(ln.strip()) > 15 or not ln.strip()]
                    text = '\n'.join(lines)
                    if len(text) > 80000:
                        text = text[:80000]

                if text and len(text.strip()) > 10:
                    ocr_texts.append(text)
                    ocr_doc_ids.append(p.stem)
                    print("[ingest]   %s -> OCR -> %d chars" % (p.name, len(text)),
                          file=sys.stderr, flush=True)
                else:
                    print("[ingest]   %s -> OCR -> empty" % p.name,
                          file=sys.stderr, flush=True)
            except Exception as e:
                print("[ingest]   %s -> OCR FAILED: %s" % (p.name, e),
                      file=sys.stderr, flush=True)
        else:
            direct_paths.append(path)
            print("[ingest]   %s -> direct (%s)" % (p.name, suffix or "text"),
                  file=sys.stderr, flush=True)

    event_queue.put(("phase", "ingest",
        {"status": "done", "n_ocr": len(ocr_texts),
         "n_direct": len(direct_paths)}))

    return ocr_texts, ocr_doc_ids, direct_paths


# Analysis subprocess

def _analysis_subprocess(ocr_texts, ocr_doc_ids, direct_paths,
                          query, max_rechunk, depth, workspace, event_queue,
                          ontology=False):
    """Run analysis in a subprocess using PipelineRunner.

    Receives OCR'd texts AND raw file paths. PipelineRunner routes
    each through the correct adapter (csv_loader, text, etc.).
    """
    import sys
    import traceback

    try:
        n_total = len(ocr_texts) + len(direct_paths)
        print("[analysis] subprocess: %d OCR texts + %d direct files" % (
            len(ocr_texts), len(direct_paths)),
              file=sys.stderr, flush=True)

        from agent.pipeline_runner import PipelineRunner

        def on_phase(phase, data):
            try:
                event_queue.put(("phase", phase, _sanitize(data)), timeout=5)
            except Exception:
                pass

        runner = PipelineRunner()
        runner.on_phase(on_phase)

        # run() now builds one corpus from whatever is present, so a
        # single call handles texts-only, files-only, and mixed batches
        # without dropping either (audit P2).
        result = runner.run(
            files=direct_paths or None,
            texts=ocr_texts or None,
            doc_ids=ocr_doc_ids or None,
            query=query or None,
            max_rechunk=max_rechunk,
            depth=depth,
            ontology=ontology,
        )

        print("[analysis] complete: %d docs, %.1fs" % (
            len(result.documents), result.elapsed),
              file=sys.stderr, flush=True)

        if workspace:
            _save_to_workspace(runner, workspace)

        result_dict = _sanitize({
            "documents": result.documents,
            "corpus_summary": result.corpus_summary,
            "temporal": result.temporal,
            "chunks": result.chunks,
            "query_result": result.query_result,
            "model_response": result.model_response,
            "hallucination_report": result.hallucination_report,
            "ontology": result.ontology,
            "elapsed": result.elapsed,
        })
        event_queue.put(("done", result_dict, None), timeout=10)

    except Exception as e:
        print("[analysis] ERROR: %s" % e, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        try:
            event_queue.put(("error", str(e), None), timeout=5)
        except Exception:
            pass


def _save_to_workspace(runner, workspace):
    try:
        from agent.server.persistence import save_document_rex, _docs_dir
        corpus = getattr(runner, '_last_corpus', None)
        if corpus is None:
            return
        for doc in corpus.documents:
            if doc.rex is None:
                continue
            try:
                save_document_rex(workspace, doc.doc_id, doc.rex)
                if doc.text:
                    text_path = _docs_dir(workspace) / ("%s.txt" % doc.doc_id)
                    text_path.write_text(doc.text, encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass


def _register_documents(result_data, workspace):
    # Merge chunks into their parent documents so frontend can read doc.chunks
    for doc in result_data.get("documents", []):
        for chunk_group in result_data.get("chunks", []):
            if chunk_group.get("doc_id") == doc.get("doc_id"):
                doc["chunks"] = chunk_group.get("chunks", [])
                break
    try:
        from agent.server.app import get_store
        from agent.server.persistence import load_document_rex
        store = get_store()
        for doc in result_data.get("documents", []):
            doc_id = doc.get("doc_id", "")
            if not doc_id:
                continue
            try:
                rex = load_document_rex(workspace, doc_id)
                if rex is None:
                    continue
                session = store.create(name=doc_id)
                analysis = doc.get("analysis", {})
                if not analysis:
                    analysis = {}
                    if doc.get("betti"):
                        analysis["topology"] = {"betti": doc["betti"]}
                    if doc.get("hodge"):
                        analysis["hodge"] = doc["hodge"]
                    if doc.get("kappa_mean") is not None:
                        analysis["relational"] = {
                            "kappa_mean": doc["kappa_mean"],
                            "chi_mean": doc.get("chi_mean"),
                        }
                session.add_snapshot(
                    rex=rex, action="pipeline", params={},
                    results=analysis,
                    summary="Pipeline: %s (%dV %dE %dF)" % (
                        doc_id, doc.get("nV", 0),
                        doc.get("nE", 0), doc.get("nF", 0)),
                )
                doc["session_id"] = session.session_id
                doc["workspace"] = workspace
                # Make the storage linkage bidirectional (audit 5.4):
                # tag the session with its workspace and record the
                # doc -> session mapping in the workspace index.
                try:
                    session._metadata["workspace"] = workspace
                    session._metadata["doc_id"] = doc_id
                    session._save_index()
                except Exception:
                    pass
                try:
                    from agent.server.persistence import link_doc_session
                    link_doc_session(workspace, doc_id, session.session_id)
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass
    return result_data


# SSE endpoint

@router.post("/stream")
async def stream_pipeline(
    files: List[UploadFile] = File(...),
    query: str = Form(None),
    depth: str = Form("quick"),
    max_rechunk: int = Form(2),
    backend: str = Form(None),
    workspace: str = Form("default"),
    ontology: bool = Form(False),
    fusion: bool = Form(False),
):
    upload_data = []
    for f in files:
        content = await f.read()
        upload_data.append({"name": f.filename or "file", "content": content})

    async def generate():
        temp_paths = _save_uploads(upload_data)
        loop = asyncio.get_event_loop()
        proc = None
        try:
            # Phase 1: Ingest - OCR images/PDFs, identify direct files
            import queue as thread_queue
            ingest_queue = thread_queue.Queue()

            # Only initialise an OCR backend if at least one uploaded file
            # actually needs OCR. A pure CSV/JSON/TSV/text batch must never
            # load an OCR model or fail on OCR-init (audit P1/A1).
            ocr_needed = any(
                os.path.splitext(p)[1].lower() in OCR_EXTENSIONS
                for p in temp_paths
            )
            ocr_client = None
            if ocr_needed:
                try:
                    ocr_client = _get_ocr_client(backend)
                except Exception as e:
                    logger.error("OCR client failed: %s", e)
                    yield 'event: error\ndata: {"error": "OCR init failed: %s"}\n\n' % str(e).replace('"', '\\"')
                    return

            try:
                ocr_texts, ocr_doc_ids, direct_paths = await loop.run_in_executor(
                    _executor,
                    lambda: _ingest_files(temp_paths, ocr_client, ingest_queue,
                                          use_fusion=fusion),
                )
            except Exception as e:
                logger.error("Ingestion failed: %s", e)
                yield 'event: error\ndata: {"error": "Ingestion failed: %s"}\n\n' % str(e).replace('"', '\\"')
                return

            # Yield ingest events
            while not ingest_queue.empty():
                try:
                    msg = ingest_queue.get_nowait()
                    payload = json.dumps(
                        {"phase": msg[1], **msg[2]}, cls=_SafeEncoder)
                    yield "event: phase\ndata: %s\n\n" % payload
                except Exception:
                    break

            if not ocr_texts and not direct_paths:
                yield 'event: done\ndata: {"documents":[]}\n\n'
                return

            # Phase 2: Analysis - ALWAYS in an isolated subprocess.
            #
            # The compiled core runs with bounds-checks off, so a pathological or
            # oversized input can segfault or OOM. Running it in-process would take
            # the whole server down with it (a crash we actually hit on a 100k-node
            # graph). Isolating it means such a failure kills only the child; the
            # server survives and returns an error for that one request.
            #
            # To avoid paying a fresh-interpreter re-import tax (~0.3-0.5s) on every
            # request, we use a `forkserver`: a clean, single-threaded helper that
            # imports numpy/scipy/the analysis stack ONCE, then each request forks
            # from it - warm AND isolated. Forking from that helper (not the
            # multithreaded server, and before any OCR/torch is loaded) also avoids
            # fork-with-threads hazards and never inherits a CUDA context (audit 4.1).
            ctx = _analysis_ctx()
            analysis_queue = ctx.Queue()
            proc = ctx.Process(
                target=_analysis_subprocess,
                args=(ocr_texts, ocr_doc_ids, direct_paths,
                      query, max_rechunk, depth, workspace,
                      analysis_queue, ontology),
            )
            proc.start()

            finished = False
            while not finished:
                try:
                    msg = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: analysis_queue.get(timeout=2.0)),
                        timeout=5.0,
                    )
                except (asyncio.TimeoutError, queue_module.Empty, Exception):
                    if not proc.is_alive() and analysis_queue.empty():
                        break
                    continue

                kind = msg[0]
                if kind == "phase":
                    payload = json.dumps(
                        {"phase": msg[1], **msg[2]}, cls=_SafeEncoder)
                    yield "event: phase\ndata: %s\n\n" % payload
                elif kind == "done":
                    result_data = _register_documents(msg[1], workspace)
                    yield "event: done\ndata: %s\n\n" % json.dumps(
                        result_data, cls=_SafeEncoder)
                    finished = True
                elif kind == "error":
                    yield 'event: error\ndata: {"error": "%s"}\n\n' % str(
                        msg[1]).replace('"', '\\"')
                    finished = True

            while not analysis_queue.empty():
                try:
                    msg = analysis_queue.get_nowait()
                    if msg[0] == "phase":
                        payload = json.dumps(
                            {"phase": msg[1], **msg[2]}, cls=_SafeEncoder)
                        yield "event: phase\ndata: %s\n\n" % payload
                except Exception:
                    break

            if proc is not None:
                proc.join(timeout=10)
                if not finished and proc.exitcode is not None and proc.exitcode != 0:
                    if proc.exitcode < 0:
                        import signal
                        try:
                            sig_name = signal.Signals(-proc.exitcode).name
                        except (ValueError, AttributeError):
                            sig_name = "signal %d" % (-proc.exitcode)
                    else:
                        sig_name = "exit code %d" % proc.exitcode
                    yield ('event: error\ndata: {"error": '
                           '"Analysis worker exited (%s) - most likely the graph '
                           'was too large/dense for available memory. The server '
                           'is unaffected; try a smaller or sparser input, or lower '
                           'REXGRAPH_MAX_ANALYSIS_EDGES."}\n\n' % sig_name)
        finally:
            # Always let analysis finish before deleting temp files, even if the
            # client disconnected mid-run, so the worker never reads a file that
            # cleanup already removed (audit 4.3) - for both the subprocess and the
            # in-process paths.
            if proc is not None:
                try:
                    if proc.is_alive():
                        proc.terminate()
                    proc.join(timeout=5)
                except Exception:
                    pass
            _cleanup_temps(temp_paths)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )