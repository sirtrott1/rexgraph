"""
agent.cli.ocr - command-line entry point for OCR + the GPU server.

    rexgraph-ocr status              show platform, backends, server state
    rexgraph-ocr serve [opts]        start the GPU inference/OCR server
    rexgraph-ocr stop                stop the GPU server
    rexgraph-ocr run FILE...         OCR files and print/save the text
    rexgraph-ocr setup [--yes]       install OCR/GPU dependencies

This is the console-script target declared in pyproject as
``rexgraph-ocr = "agent.cli.ocr:ocr_main"``. It is a *CLI*, not a server
route - the OCR HTTP endpoint lives in ``agent.server.routes.ocr``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _cmd_status(args) -> int:
    import shutil

    gpu = "rocm" if shutil.which("rocminfo") else (
        "cuda" if shutil.which("nvidia-smi") else "cpu")
    print("Platform")
    print("  accelerator : %s" % gpu)
    try:
        from agent.cli.config import detect_platform
        pi = detect_platform()
        print("  os / arch   : %s / %s" % (
            getattr(pi, "os", "?"), getattr(pi, "arch", "?")))
    except Exception:
        pass

    print("\nOCR backends")
    # Probe each backend's availability without loading heavy models.
    try:
        from agent.integrations.unlimited_ocr import (
            OfflineOCRClient, PaddleOCRClient, GOTOCRClient,
        )
        probes = [
            ("tesseract", lambda: OfflineOCRClient().is_available()),
            ("paddleocr", lambda: (gpu != "rocm") and PaddleOCRClient().is_available()),
            ("got-ocr", lambda: GOTOCRClient().is_available()),
        ]
        for name, probe in probes:
            try:
                ok = bool(probe())
            except Exception:
                ok = False
            print("  %-10s : %s" % (name, "available" if ok else "not available"))
        if gpu == "rocm":
            print("  (paddleocr skipped: requires CUDA)")
    except Exception as e:
        print("  backend probe failed: %s" % e)

    print("\nGPU server")
    try:
        from agent.cli.serve import server_status
        st = server_status()
        print("  status : %s" % st.get("status"))
        if st.get("port"):
            print("  port   : %s" % st.get("port"))
    except Exception as e:
        print("  status : unknown (%s)" % e)
    return 0


def _cmd_serve(args) -> int:
    from agent.cli.serve import serve
    ok = serve(
        port=args.port,
        model=args.model,
        backend=args.backend,
        foreground=args.foreground,
    )
    return 0 if ok else 1


def _cmd_stop(args) -> int:
    from agent.cli.serve import stop
    return 0 if stop() else 1


def _cmd_run(args) -> int:
    from agent.integrations.unlimited_ocr import (
        create_ocr_client, is_pdf_file, is_image_file,
    )
    client = create_ocr_client(prefer=args.backend) if args.backend \
        else create_ocr_client()
    print("OCR backend: %s" % type(client).__name__, file=sys.stderr)

    outputs = []
    rc = 0
    for path in args.files:
        if not os.path.isfile(path):
            print("  skip (not a file): %s" % path, file=sys.stderr)
            rc = 1
            continue
        try:
            if is_pdf_file(path):
                text = client.ocr_pdf(path).full_text
            elif is_image_file(path):
                text = client.ocr_image(path).text
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            outputs.append({"file": path, "chars": len(text), "text": text})
            print("  %s -> %d chars" % (path, len(text)), file=sys.stderr)
        except Exception as e:
            print("  %s -> FAILED: %s" % (path, e), file=sys.stderr)
            rc = 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            if args.json:
                json.dump(outputs, fh, indent=2)
            else:
                fh.write("\n\n".join(o["text"] for o in outputs))
        print("Wrote %s" % args.output, file=sys.stderr)
    else:
        if args.json:
            print(json.dumps(outputs, indent=2))
        else:
            for o in outputs:
                print(o["text"])
    return rc


def _cmd_setup(args) -> int:
    from agent.cli.setup import auto_setup
    auto_setup(interactive=not args.yes)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rexgraph-ocr",
        description="OCR and GPU-server management for the RexGraph agent.",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("status", help="Show platform, OCR backends, server state")

    ps = sub.add_parser("serve", help="Start the GPU inference/OCR server")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--model", default=None)
    ps.add_argument("--backend", default="vllm")
    ps.add_argument("--foreground", action="store_true")

    sub.add_parser("stop", help="Stop the GPU server")

    pr = sub.add_parser("run", help="OCR one or more files")
    pr.add_argument("files", nargs="+")
    pr.add_argument("--backend", default=None,
                    help="auto|tesseract|paddleocr|got-ocr|server|mistral")
    pr.add_argument("--output", "-o", default=None)
    pr.add_argument("--json", action="store_true")

    pst = sub.add_parser("setup", help="Install OCR/GPU dependencies")
    pst.add_argument("--yes", "-y", action="store_true",
                     help="Non-interactive install")
    return p


_DISPATCH = {
    "status": _cmd_status,
    "serve": _cmd_serve,
    "stop": _cmd_stop,
    "run": _cmd_run,
    "setup": _cmd_setup,
}


def ocr_main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


# Backwards/forwards-compatible alias.
main = ocr_main


if __name__ == "__main__":
    sys.exit(ocr_main())
