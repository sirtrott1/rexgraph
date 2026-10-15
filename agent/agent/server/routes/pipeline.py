"""
agent.server.routes.pipeline: SSE pipeline endpoint.

Architecture:
  - Files are split by type: OCR (PDF/image) vs direct (CSV/TSV/JSON/text)
  - OCR runs in the main process (model cached in VRAM)
  - Analysis runs in a subprocess (crash isolated from server)
  - Non OCR files are passed as paths to auto_rex which routes them
    through the correct loader (csv_loader, json_loader, etc.)
  - Results feed into both Sessions and Workspace persistence
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import math
import multiprocessing
import os
import queue as queue_module
import tempfile
from pathlib import Path
from threading import Event

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.responses import StreamingResponse

from agent.server.scope import effective_workspace
from agent.server.upload_files import read_uploads

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/pipeline")

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Analysis runs in a separate process. Use forkserver with preloaded modules
# where supported, or spawn a fresh interpreter on other platforms.
_ANALYSIS_CTX = None


def _analysis_ctx():
    global _ANALYSIS_CTX
    if _ANALYSIS_CTX is None:
        method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        ctx = multiprocessing.get_context(method)
        if method == "forkserver":
            try:
                ctx.set_forkserver_preload(
                    ["numpy", "scipy", "agent.pipeline_runner"])
            except Exception:
                pass  # Module preloading is optional.
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


def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, cls=_SafeEncoder)}\n\n"


def _save_uploads(files):
    paths = []
    try:
        for f in files:
            suffix = os.path.splitext(f["name"])[1] or ".bin"
            fd, path = tempfile.mkstemp(suffix=suffix, prefix="rexgraph_")
            paths.append(path)
            with os.fdopen(fd, "wb") as fh:
                fh.write(f["content"])
    except BaseException:
        _cleanup_temps(paths)
        raise
    return paths


def _cleanup_temps(paths):
    for p in paths:
        with contextlib.suppress(OSError):
            os.unlink(p)


# File ingestion (main process)

def _ingest_files(file_paths, ocr_client, event_queue, use_fusion=False):
    """Split files by type. OCR images/PDFs, pass everything else through.

    Returns (ocr_texts, ocr_doc_ids, direct_paths).

    When ``use_fusion`` is set and more than one OCR backend is available,
    each document is run through every backend and the highest confidence
    result is kept.
    """
    import re
    import sys

    from agent.integrations.unlimited_ocr import is_image_file, is_pdf_file

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
            print(f"[ingest] fusion unavailable: {e}",
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
                        print(f"[ingest]   {p.name} -> fusion failed ({e}), "
                              "falling back",
                              file=sys.stderr, flush=True)
                        text = ""
                if not text:
                    if ocr_client is None:
                        # No OCR backend was initialised (direct only
                        # batch). Skip rather than dereference None.
                        print(f"[ingest]   {p.name} -> skipped (no OCR backend)", file=sys.stderr, flush=True)
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
                    print(f"[ingest]   {p.name} -> OCR -> empty",
                          file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[ingest]   {p.name} -> OCR FAILED: {e}",
                      file=sys.stderr, flush=True)
        else:
            direct_paths.append(path)
            print("[ingest]   {} -> direct ({})".format(p.name, suffix or "text"),
                  file=sys.stderr, flush=True)

    event_queue.put(("phase", "ingest",
        {"status": "done", "n_ocr": len(ocr_texts),
         "n_direct": len(direct_paths)}))

    return ocr_texts, ocr_doc_ids, direct_paths


# Analysis subprocess

def _analysis_subprocess(ocr_texts, ocr_doc_ids, direct_paths,
                          query, max_rechunk, depth, workspace, event_queue,
                          ontology=False, face_selection=None, reader_options=None,
                          workspace_config_dir=None):
    """Run analysis in a subprocess using PipelineRunner.

    Receives OCR'd texts AND raw file paths. PipelineRunner routes
    each through the correct adapter (csv_loader, text, etc.).
    """
    import sys
    import traceback

    try:
        if workspace_config_dir is not None:
            os.environ["REXGRAPH_CONFIG_DIR"] = workspace_config_dir
        len(ocr_texts) + len(direct_paths)
        print("[analysis] subprocess: %d OCR texts + %d direct files" % (
            len(ocr_texts), len(direct_paths)),
              file=sys.stderr, flush=True)

        from agent.pipeline_runner import PipelineRunner

        def on_phase(phase, data):
            with contextlib.suppress(Exception):
                event_queue.put(("phase", phase, _sanitize(data)), timeout=5)

        runner = PipelineRunner()
        runner.on_phase(on_phase)

        # run() now builds one corpus from whatever is present, so a
        # single call handles texts only, files only, and mixed batches
        # without dropping either.
        result = runner.run(
            files=direct_paths or None,
            texts=ocr_texts or None,
            doc_ids=ocr_doc_ids or None,
            query=query or None,
            max_rechunk=max_rechunk,
            depth=depth,
            ontology=ontology,
            face_selection=face_selection,
            reader_options=reader_options,
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
        print(f"[analysis] ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        with contextlib.suppress(Exception):
            event_queue.put(("error", str(e), None), timeout=5)


def _save_to_workspace(runner, workspace):
    from agent.server.persistence import doc_path, save_document_rex
    from rexgraph.io.publication import staged_publication
    corpus = getattr(runner, '_last_corpus', None)
    if corpus is None:
        return
    for doc in corpus.documents:
        if doc.rex is None:
            continue
        save_document_rex(workspace, doc.doc_id, doc.rex)
        if doc.text:
            text_path = doc_path(workspace, doc.doc_id, suffix=".txt")
            with staged_publication(text_path) as staged:
                staged.write_text(doc.text, encoding="utf-8")


def _register_documents(result_data, workspace):
    # Merge chunks into their parent documents so frontend can read doc.chunks
    for doc in result_data.get("documents", []):
        for chunk_group in result_data.get("chunks", []):
            if chunk_group.get("doc_id") == doc.get("doc_id"):
                doc["chunks"] = chunk_group.get("chunks", [])
                break
    from agent.server.app import get_store
    from agent.server.persistence import load_document_rex
    store = get_store()
    for doc in result_data.get("documents", []):
        doc_id = doc.get("doc_id", "")
        if not doc_id:
            continue
        rex = load_document_rex(workspace, doc_id)
        if rex is None:
            raise FileNotFoundError(f"Saved document is unavailable: {doc_id}")
        session = store.create(name=doc_id)
        session._metadata["workspace"] = workspace
        session._metadata["doc_id"] = doc_id
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
        # the picture of what was built, beside the numbers about it. Same
        # render_payload the Graph screen and the rexgraph_render tool take, so a
        # pipeline run and a later look at the same session cannot disagree.
        # Bounded and REPORTED rather than gated: a truncated picture says so.
        try:
            from agent.graph_view import render_payload
            from agent.render_svg import render_svg

            labels = (getattr(rex, "_agent_meta", {}) or {}).get("vertex_labels")
            payload = render_payload(rex, labels=labels, limit=400)
            drawn = {r["index"] for r in payload.get("relations", [])}
            doc["drawing"] = {
                "drawn": True,
                "svg": render_svg(payload),
                "cells_drawn": len(drawn), "cells_total": int(rex.nE),
                "truncated": len(drawn) < int(rex.nE),
                "faces_drawn": sum(1 for f in payload.get("faces", [])
                                   if set(f["relations"]) <= drawn),
                "faces_total": int(rex.nF_hodge),
                "reading": ("positions are exact rationals from each cell's own "
                            "star; length is the quadrance, so it carries arity; "
                            "colour is the character through K7's spectrum"),
            }
        except Exception as exc:
            # a run that could not be drawn has still been analysed
            doc["drawing"] = {"drawn": False,
                              "reason": f"{type(exc).__name__}: {exc}"}
        session.add_snapshot(
            rex=rex, action="pipeline", params={},
            results=analysis,
            summary="Pipeline: %s (%dV %dE %dF)" % (
                doc_id, doc.get("nV", 0),
                doc.get("nE", 0), doc.get("nF", 0)),
        )
        doc["session_id"] = session.session_id
        doc["workspace"] = workspace
        # Make the storage linkage bidirectional:
        # tag the session with its workspace and record the
        # doc -> session mapping in the workspace index.
        from agent.server.persistence import link_doc_session
        link_doc_session(workspace, doc_id, session.session_id)
    return result_data


# SSE endpoint

@router.post("/stream")
async def stream_pipeline(
    files: list[UploadFile] = File(...),
    query: str = Form(None),
    depth: str = Form("quick"),
    max_rechunk: int = Form(2),
    backend: str = Form(None),
    workspace: str = Form("default"),
    ontology: bool = Form(False),
    fusion: bool = Form(False),
    face_selection: str = Form("none"),
    reader_options: str = Form("{}"),
):
    """`reader_options` is JSON forwarded to whichever reader the file selects, filtered
    to what that reader actually declares. `{"aromatic": "pairwise"}` makes an SDF read
    its rings as separate bonds instead of one delocalised relation; `{"backbone": false}`
    makes a PDB take only its CONECT records. Asserting a face rule is a claim about the
    data, so `face_selection` stays the caller's too and defaults to none.
    """
    # The workspace names where this run writes. Inside a request it is the bound
    # one, not the form field: a form field naming another tenant wrote documents,
    # text and a stamped session into their namespace.
    workspace = effective_workspace(workspace)

    try:
        options = json.loads(reader_options)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(400, "reader_options must be a JSON object") from exc
    if not isinstance(options, dict):
        raise HTTPException(400, "reader_options must be a JSON object")
    from agent.auto import FACE_WORDS
    if face_selection not in FACE_WORDS:
        raise HTTPException(400, "face_selection must be one of " + ", ".join(FACE_WORDS))

    upload_data = await read_uploads(files)

    async def generate():
        temp_paths = []
        loop = asyncio.get_event_loop()
        proc = None
        analysis_queue = None
        ingestion_done, cleanup_requested = Event(), Event()
        ingestion_started = False
        try:
            try:
                temp_paths = _save_uploads(upload_data)
            except Exception as exc:
                yield _sse("error", {"error": f"Upload staging failed: {exc}"})
                return
            # Phase 1, ingest: OCR images and PDFs, identify direct files
            import queue as thread_queue
            ingest_queue = thread_queue.Queue()

            # Only initialise an OCR backend if at least one uploaded file
            # actually needs OCR. A pure CSV/JSON/TSV/text batch must never
            # load an OCR model or fail on OCR init.
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
                    yield _sse("error", {"error": f"OCR init failed: {e}"})
                    return

            def ingest():
                try:
                    return _ingest_files(temp_paths, ocr_client, ingest_queue,
                                         use_fusion=fusion)
                finally:
                    ingestion_done.set()
                    if cleanup_requested.is_set():
                        _cleanup_temps(temp_paths)

            try:
                future = loop.run_in_executor(_executor, ingest)
                ingestion_started = True
                ocr_texts, ocr_doc_ids, direct_paths = await asyncio.shield(future)
            except Exception as e:
                logger.error("Ingestion failed: %s", e)
                yield _sse("error", {"error": f"Ingestion failed: {e}"})
                return

            # Yield ingest events
            while not ingest_queue.empty():
                try:
                    msg = ingest_queue.get_nowait()
                    payload = json.dumps(
                        {"phase": msg[1], **msg[2]}, cls=_SafeEncoder)
                    yield f"event: phase\ndata: {payload}\n\n"
                except Exception:
                    break

            if not ocr_texts and not direct_paths:
                yield 'event: done\ndata: {"documents":[]}\n\n'
                return

            # Phase 2: run compiled analysis in an isolated subprocess. The selected
            # context uses forkserver with preloaded analysis modules where available.
            ctx = _analysis_ctx()
            from agent.server.handles import config_dir
            analysis_queue = ctx.Queue()
            proc = ctx.Process(
                target=_analysis_subprocess,
                args=(ocr_texts, ocr_doc_ids, direct_paths,
                      query, max_rechunk, depth, workspace,
                      analysis_queue, ontology, face_selection, options,
                      str(config_dir())),
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
                except (asyncio.TimeoutError, queue_module.Empty):
                    if not proc.is_alive() and analysis_queue.empty():
                        break
                    continue
                except Exception as exc:
                    yield _sse("error", {"error": f"Analysis queue failed: {exc}"})
                    finished = True
                    break

                kind = msg[0]
                if kind == "phase":
                    payload = json.dumps(
                        {"phase": msg[1], **msg[2]}, cls=_SafeEncoder)
                    yield f"event: phase\ndata: {payload}\n\n"
                elif kind == "done":
                    try:
                        result_data = _register_documents(msg[1], workspace)
                    except Exception as exc:
                        yield _sse("error", {"error": f"Unable to save analysis session: {exc}"})
                    else:
                        yield _sse("done", result_data)
                    finished = True
                elif kind == "error":
                    yield _sse("error", {"error": str(msg[1])})
                    finished = True

            if proc is not None:
                proc.join(timeout=10)
                if not finished:
                    if proc.exitcode is not None and proc.exitcode < 0:
                        import signal
                        try:
                            sig_name = signal.Signals(-proc.exitcode).name
                        except (ValueError, AttributeError):
                            sig_name = "signal %d" % (-proc.exitcode)
                    else:
                        sig_name = "exit code %d" % proc.exitcode
                    yield _sse("error", {"error": f"Analysis worker exited without a result ({sig_name})."})
        finally:
            if proc is not None:
                try:
                    if proc.is_alive():
                        proc.terminate()
                    proc.join(timeout=5)
                except Exception:
                    pass
            if analysis_queue is not None:
                with contextlib.suppress(Exception):
                    analysis_queue.close()
                    analysis_queue.join_thread()
            cleanup_requested.set()
            if not ingestion_started or ingestion_done.is_set():
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
