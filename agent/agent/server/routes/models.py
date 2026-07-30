"""
agent.server.routes.models: model lifecycle management via ModelManager.

    GET  /api/v1/models/list          all models with status
    GET  /api/v1/models/status        loaded/available/VRAM summary
    POST /api/v1/models/pull          download a model
    POST /api/v1/models/load          load into VRAM
    POST /api/v1/models/unload        free VRAM
    POST /api/v1/models/deploy        start vLLM server
    POST /api/v1/models/stop          stop server
    POST /api/v1/models/set-path      register custom model path
    POST /api/v1/models/set-pipeline  assign model to pipeline stage
    GET  /api/v1/models/pipeline      get pipeline model config
    GET  /api/v1/models/cache         cache directory contents
    DELETE /api/v1/models/cache/{id}  delete a cached model
"""

from __future__ import annotations

import shutil

from fastapi import APIRouter, HTTPException, Body

router = APIRouter(prefix="/v1/models")


def _mgr():
    from agent.model_manager import get_manager
    return get_manager()


@router.get("/list")
async def list_models():
    """List all models with download/loaded status."""
    return {"models": _mgr().list_models()}


@router.get("/status")
async def model_status():
    """Full status: loaded models, available models, VRAM usage."""
    return _mgr().status()


@router.post("/pull")
async def pull_model(model_id: str = Body(..., embed=True)):
    """Download a model from HuggingFace."""
    try:
        path = _mgr().download(model_id)
        return {"status": "downloaded", "model_id": model_id, "path": path}
    except Exception as e:
        raise HTTPException(500, "Download failed: %s" % e)


@router.post("/load")
async def load_model(
    model_id: str = Body(..., embed=True),
    device: str = Body("auto", embed=True),
    purpose: str = Body("", embed=True),
):
    """Load a model into VRAM. Cached for server lifetime."""
    try:
        lm = _mgr().load(model_id, device=device, purpose=purpose)
        return {
            "status": "loaded",
            "model_id": model_id,
            "device": lm.device,
            "vram_mb": lm.vram_mb,
            "type": lm.model_type.value,
        }
    except Exception as e:
        raise HTTPException(500, "Load failed: %s" % e)


@router.post("/unload")
async def unload_model(model_id: str = Body(..., embed=True)):
    """Unload a model and free VRAM."""
    ok = _mgr().unload(model_id)
    if not ok:
        raise HTTPException(404, "Model not loaded: %s" % model_id)
    return {"status": "unloaded", "model_id": model_id}


@router.post("/deploy")
async def deploy_model(
    model_id: str = Body(..., embed=True),
    port: int = Body(10000, embed=True),
    backend: str = Body("vllm", embed=True),
):
    """Start a vLLM/SGLang server for a model."""
    try:
        lm = _mgr().load(model_id, purpose="server")
        return {
            "status": "started",
            "model_id": model_id,
            "port": lm.server_port,
            "url": lm.server_url,
        }
    except Exception as e:
        raise HTTPException(500, "Deploy failed: %s" % e)


@router.post("/stop")
async def stop_server(model_id: str = Body(None, embed=True)):
    """Stop a model server."""
    if model_id:
        ok = _mgr().unload(model_id)
    else:
        # Stop the first running server
        for mid, lm in list(_mgr()._loaded.items()):
            if lm.server_port:
                ok = _mgr().unload(mid)
                break
        else:
            ok = False
    return {"status": "stopped" if ok else "no server running"}


@router.post("/set-path")
async def set_model_path(
    model_id: str = Body(..., embed=True),
    path: str = Body(..., embed=True),
    model_type: str = Body("transformers", embed=True),
    purpose: str = Body("", embed=True),
):
    """Register a custom local path for a model."""
    try:
        entry = _mgr().set_model_path(model_id, path, model_type, purpose)
        return {
            "status": "registered",
            "model_id": model_id,
            "path": entry.local_path,
            "type": entry.model_type.value,
        }
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))


@router.delete("/path/{model_id}")
async def remove_model_path(model_id: str):
    """Remove a custom model path registration."""
    ok = _mgr().remove_model_path(model_id)
    if not ok:
        raise HTTPException(404, "No custom path for: %s" % model_id)
    return {"status": "removed", "model_id": model_id}


@router.post("/set-pipeline")
async def set_pipeline_model(
    purpose: str = Body(..., embed=True),
    model_id: str = Body(..., embed=True),
):
    """Assign a model to a pipeline stage (ocr, chat, embedding, etc.)."""
    _mgr().set_pipeline_model(purpose, model_id)
    return {"status": "set", "purpose": purpose, "model_id": model_id}


@router.get("/pipeline")
async def get_pipeline_config():
    """Get the current pipeline model assignments."""
    return {"pipeline": _mgr().pipeline_config()}


@router.get("/cache")
async def list_cache():
    """List cached model directories with sizes."""
    from agent.cli.config import MODELS_DIR
    entries = []
    total = 0
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir():
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                total += size
                entries.append({
                    "name": d.name,
                    "path": str(d),
                    "size_mb": round(size / (1024 * 1024)),
                })
    return {"entries": entries, "total_mb": round(total / (1024 * 1024))}


@router.delete("/cache/{model_name}")
async def delete_cached_model(model_name: str):
    """Delete a cached model."""
    from agent.cli.config import MODELS_DIR
    target = MODELS_DIR / model_name
    if not target.exists():
        raise HTTPException(404, "Not found: %s" % model_name)
    # Unload first if loaded
    for mid in list(_mgr()._loaded.keys()):
        if model_name in mid.replace("/", "--"):
            _mgr().unload(mid)
    shutil.rmtree(target)
    return {"status": "deleted", "name": model_name}
