"""
agent.server.routes.ocr: OCR endpoint for the agent server.

    POST /api/v1/ocr          multipart upload or JSON path
    GET  /api/v1/ocr/status   available backends
"""

from __future__ import annotations

import os
import tempfile
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/v1")


@router.post("/ocr")
async def ocr_file(
    file: UploadFile = File(None),
    path: str = Form(None),
    backend: str = Form(None),
    dpi: int = Form(300),
):
    """OCR an uploaded file or server-local path."""
    from agent.integrations.unlimited_ocr import (
        create_ocr_client,
        is_image_file,
        is_pdf_file,
    )

    # Resolve the file to OCR
    cleanup_path = None
    if file and file.filename:
        # Save upload to temp file
        suffix = os.path.splitext(file.filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, dir=tempfile.gettempdir()
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            ocr_path = tmp.name
            cleanup_path = tmp.name
    elif path:
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"File not found: {path}")
        ocr_path = path
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file upload or a path parameter",
        )

    # Create client
    kwargs = {}
    if backend:
        kwargs["prefer"] = backend

    try:
        client = create_ocr_client(**kwargs)
    except Exception as e:
        if cleanup_path:
            os.unlink(cleanup_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create OCR client: {e}",
        )

    # Run OCR
    start = time.time()
    try:
        if os.path.isdir(ocr_path):
            result = client.ocr_directory(ocr_path)
            text = result.full_text
            pages = [
                {"page": p.page, "text": p.text, "tokens": p.tokens}
                for p in result.pages
            ]
        elif is_pdf_file(ocr_path):
            result = client.ocr_pdf(ocr_path, dpi=dpi)
            text = result.full_text
            pages = [
                {"page": p.page, "text": p.text, "tokens": p.tokens}
                for p in result.pages
            ]
        elif is_image_file(ocr_path):
            result = client.ocr_image(ocr_path)
            text = result.text
            pages = [{"page": 0, "text": result.text, "tokens": result.tokens}]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ocr_path}",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            os.unlink(cleanup_path)

    elapsed = time.time() - start

    return {
        "text": text,
        "pages": pages,
        "backend": getattr(client, "backend_name", type(client).__name__),
        "elapsed": round(elapsed, 2),
        "source": file.filename if file else path,
    }


@router.get("/ocr/status")
async def ocr_status():
    """OCR backend detection with install status and commands."""
    import importlib
    import shutil
    from pathlib import Path

    backends = {}
    gpu = "rocm" if shutil.which("rocminfo") else (
        "cuda" if shutil.which("nvidia-smi") else "cpu")

    # Get model status from ModelManager
    try:
        from agent.model_manager import get_manager
        mgr = get_manager()
        mgr_status = mgr.status()
    except Exception:
        mgr_status = {"loaded": [], "available": []}

    # vLLM / Unlimited-OCR
    vllm_installed = importlib.util.find_spec("vllm") is not None
    model_dir = Path.home() / ".cache" / "rexgraph" / "models" / "deepseek-ai--DeepSeek-OCR-2"
    model_downloaded = model_dir.exists() and any(model_dir.iterdir()) if model_dir.exists() else False

    try:
        from agent.cli.serve import server_status
        srv = server_status()
        server_running = srv["status"] in ("healthy", "running")
    except Exception:
        srv = {"status": "stopped"}
        server_running = False

    # Check if loaded in ModelManager
    ocr_loaded = any(m.get("purpose") == "ocr" and m.get("loaded")
                     for m in mgr_status.get("loaded", []))

    backends["unlimited_ocr"] = {
        "installed": vllm_installed and model_downloaded,
        "ready": server_running,
        "vllm": vllm_installed,
        "model_downloaded": model_downloaded,
        "server_status": srv.get("status", "stopped"),
        "type": "local-gpu",
        "install": "make install-ocr-server" if not (vllm_installed and model_downloaded) else None,
        "start": "make ocr-serve" if not server_running else None,
    }

    # Tesseract
    has_tesseract = shutil.which("tesseract") is not None
    backends["tesseract"] = {
        "installed": has_tesseract,
        "type": "local-cpu",
        "install": "sudo apt install tesseract-ocr" if not has_tesseract else None,
    }

    # GOT-OCR2.0
    got_libs = (importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("torch") is not None)
    got_model = False
    if got_libs:
        got_model_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--stepfun-ai--GOT-OCR-2.0-hf"
        got_snapshots = got_model_dir / "snapshots"
        got_model = got_snapshots.exists() and any(got_snapshots.iterdir())
    backends["got_ocr"] = {
        "installed": got_libs and got_model,
        "libraries": got_libs,
        "model_downloaded": got_model,
        "loaded": ocr_loaded,
        "type": "local-gpu",
        "install": "make install-got-ocr" if not (got_libs and got_model) else None,
    }

    # PaddleOCR
    paddle_installed = importlib.util.find_spec("paddleocr") is not None
    backends["paddleocr"] = {
        "installed": paddle_installed,
        "type": "local-gpu",
        "note": "Requires CUDA (not compatible with ROCm)" if gpu == "rocm" else None,
    }

    # Mistral
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    backends["mistral"] = {
        "installed": bool(api_key),
        "type": "cloud",
        "has_key": bool(api_key),
    }

    # PyTorch (spec check only, no import)
    torch_info = {"installed": importlib.util.find_spec("torch") is not None}

    return {
        "backends": backends,
        "gpu": gpu,
        "torch": torch_info,
        "models": mgr_status,
        "recommended": "unlimited_ocr" if gpu != "cpu" else "tesseract",
    }
