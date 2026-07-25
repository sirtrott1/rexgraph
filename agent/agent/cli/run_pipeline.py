#!/usr/bin/env python3
"""
rexgraph-run - headless pipeline for CLI and HPC batch processing.

Usage:
    python -m agent.cli.run_pipeline paper.pdf
    python -m agent.cli.run_pipeline *.pdf --depth standard --backend tesseract
    python -m agent.cli.run_pipeline doc1.pdf doc2.pdf --query "What is the main result?"
    python -m agent.cli.run_pipeline --input-dir /data/pdfs --output results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path


def _run_via_server(args, files):
    """Send files to a running server's pipeline API."""
    import urllib.request
    import urllib.error
    import ssl
    import io

    url = args.server.rstrip("/")

    # Get stored auth token
    token = ""
    try:
        from agent.cli.auth import get_stored_auth
        stored_url, stored_token = get_stored_auth()
        token = stored_token
    except ImportError:
        pass

    print("Sending %d file(s) to %s" % (len(files), url), file=sys.stderr)

    # Build multipart form data
    boundary = "----RexGraphBoundary%d" % int(time.time() * 1000)
    body = io.BytesIO()

    for filepath in files:
        filename = os.path.basename(filepath)
        body.write(("--%s\r\n" % boundary).encode())
        body.write(('Content-Disposition: form-data; name="files"; '
                    'filename="%s"\r\n' % filename).encode())
        body.write(b"Content-Type: application/octet-stream\r\n\r\n")
        with open(filepath, "rb") as f:
            body.write(f.read())
        body.write(b"\r\n")

    for key, val in [("depth", args.depth), ("workspace", args.workspace)]:
        if val:
            body.write(("--%s\r\n" % boundary).encode())
            body.write(('Content-Disposition: form-data; name="%s"\r\n\r\n' % key).encode())
            body.write(val.encode())
            body.write(b"\r\n")

    if args.query:
        body.write(("--%s\r\n" % boundary).encode())
        body.write(b'Content-Disposition: form-data; name="query"\r\n\r\n')
        body.write(args.query.encode())
        body.write(b"\r\n")

    body.write(("--%s--\r\n" % boundary).encode())
    body_bytes = body.getvalue()

    headers = {
        "Content-Type": "multipart/form-data; boundary=%s" % boundary,
    }
    if token:
        headers["Authorization"] = "Bearer %s" % token

    req = urllib.request.Request(
        "%s/api/v1/pipeline/stream" % url,
        data=body_bytes,
        headers=headers,
        method="POST",
    )

    ctx = None
    if url.startswith("https://localhost") or url.startswith("https://127."):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=600) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "phase" in data:
                        print("  [%s] %s" % (
                            data.get("phase", "?"),
                            data.get("status", "")),
                            file=sys.stderr)
                    elif "documents" in data:
                        if args.json:
                            print(json.dumps(data, indent=2))
                        else:
                            for doc in data.get("documents", []):
                                print("  %s: %dV %dE κ=%s" % (
                                    doc.get("doc_id", "?"),
                                    doc.get("nV", 0),
                                    doc.get("nE", 0),
                                    "%.3f" % doc["kappa_mean"]
                                    if doc.get("kappa_mean") is not None
                                    else "-"))
                    elif "error" in data:
                        print("Error: %s" % data["error"], file=sys.stderr)
                        sys.exit(1)
    except urllib.error.HTTPError as e:
        print("Server error (%d): %s" % (e.code, e.read().decode()),
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("Connection failed: %s" % e, file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run rexgraph pipeline from the command line")
    parser.add_argument("files", nargs="*", help="PDF/image/text files to process")
    parser.add_argument("--input-dir", "-d", help="Directory of files to process")
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument("--depth", default="standard",
                        choices=["quick", "standard", "full", "deep"])
    parser.add_argument("--ontology", action="store_true",
                        help="Run TrustGraph ontology enrichment stage")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Print compiled-kernel diagnostics and exit")
    parser.add_argument("--backend", default=None,
                        help="OCR backend: auto, tesseract, server, got-ocr")
    parser.add_argument("--query", "-q", default=None,
                        help="Query to run against the corpus")
    parser.add_argument("--max-rechunk", type=int, default=2)
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save to workspace")
    parser.add_argument("--server", "-s", default=None,
                        help="Use a running server instead of direct imports "
                             "(e.g. https://localhost:8000). Uses stored auth.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON (default: human-readable)")
    args = parser.parse_args()

    if args.diagnostics:
        from agent.diagnostics import format_report
        print(format_report())
        return

    # Collect files
    files = list(args.files or [])
    if args.input_dir:
        for ext in ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.txt", "*.csv"):
            files.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not files:
        parser.error("No files specified. Use positional args or --input-dir")

    files = [os.path.abspath(f) for f in files if os.path.isfile(f)]
    if not files:
        parser.error("No valid files found")

    # Server mode: send to running server via API
    if args.server:
        _run_via_server(args, files)
        return

    if not args.json:
        print("rexgraph pipeline - %d file(s)\n" % len(files), file=sys.stderr)

    # Create OCR client
    from agent.integrations.unlimited_ocr import create_ocr_client
    if args.backend and args.backend != "auto":
        ocr_client = create_ocr_client(prefer=args.backend)
    else:
        ocr_client = create_ocr_client()

    if not args.json:
        print("  OCR: %s" % type(ocr_client).__name__, file=sys.stderr)

    # Run pipeline
    from agent.pipeline_runner import PipelineRunner

    def on_phase(phase, data):
        if not args.json:
            status = data.get("status", "")
            doc_id = data.get("doc_id", "")
            extra = " - %s" % doc_id if doc_id else ""
            print("  [%s] %s%s" % (phase, status, extra), file=sys.stderr)

    runner = PipelineRunner(ocr_client=ocr_client)
    runner.on_phase(on_phase)

    t0 = time.time()
    result = runner.run(
        files=files,
        query=args.query,
        max_rechunk=args.max_rechunk,
        depth=args.depth,
        ontology=args.ontology,
    )
    elapsed = time.time() - t0

    # Save to workspace
    if not args.no_save:
        try:
            from agent.server.persistence import save_document_rex, _docs_dir
            corpus = getattr(runner, '_last_corpus', None)
            if corpus:
                for doc in corpus.documents:
                    if doc.rex:
                        save_document_rex(args.workspace, doc.doc_id, doc.rex)
                        if doc.text:
                            text_path = _docs_dir(args.workspace) / ("%s.txt" % doc.doc_id)
                            text_path.write_text(doc.text, encoding="utf-8")
                if not args.json:
                    print("\n  Saved to workspace: %s" % args.workspace,
                          file=sys.stderr)
        except Exception as e:
            if args.verbose:
                print("  Warning: workspace save failed: %s" % e,
                      file=sys.stderr)

    # Format output
    output = {
        "documents": result.documents,
        "corpus_summary": result.corpus_summary,
        "chunks": result.chunks,
        "query_result": result.query_result,
        "model_response": result.model_response,
        "elapsed": elapsed,
        "n_files": len(files),
    }

    # Sanitize numpy types
    def sanitize(obj):
        import numpy as np
        import math
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return sanitize(obj.tolist())
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    output = sanitize(output)

    if args.json or args.output:
        json_str = json.dumps(output, indent=2, default=str)
        if args.output:
            Path(args.output).write_text(json_str)
            if not args.json:
                print("  Output: %s" % args.output, file=sys.stderr)
        else:
            print(json_str)
    else:
        # Human-readable summary
        print("\n── Results ──\n")
        for doc in output.get("documents", []):
            print("  %s: %dV %dE %dF  betti=%s  κ=%s" % (
                doc.get("doc_id", "?"),
                doc.get("nV", 0), doc.get("nE", 0), doc.get("nF", 0),
                doc.get("betti", "?"),
                "%.3f" % doc["kappa_mean"] if doc.get("kappa_mean") is not None else "-",
            ))
            hodge = doc.get("hodge", {})
            if hodge:
                print("    Hodge: G=%.0f%% C=%.0f%% H=%.0f%%" % (
                    (hodge.get("gradient") or 0) * 100,
                    (hodge.get("curl") or 0) * 100,
                    (hodge.get("harmonic") or 0) * 100,
                ))

        if output.get("model_response"):
            print("\n── Model Response ──\n")
            print("  " + output["model_response"][:500])

        print("\n  %.1fs elapsed" % elapsed)


if __name__ == "__main__":
    main()
