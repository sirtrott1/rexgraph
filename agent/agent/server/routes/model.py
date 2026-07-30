"""
agent.server.routes.model: context-aware model inference and system status.

    POST /api/v1/model/generate   structured prompt with session context
    GET  /api/v1/status           system health
"""

from __future__ import annotations

import json
import os
import shutil

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/v1")


def _build_context(session, results: dict) -> str:
    """Build a system prompt from session analysis results.

    Includes topology, Hodge decomposition, structural character,
    coherence, void structure, and axiom verification. The model
    gets the mathematical structure of the data, not just the text.
    """
    rex = session.current() if session else None
    meta = getattr(rex, "_agent_meta", {}) if rex else {}
    lines = []

    lines.append("You are a research assistant analyzing a document that has been "
                 "processed through the RexGraph relational complex framework. "
                 "You have access to both the raw text and the mathematical "
                 "structure of the document.")

    # Construction
    if rex:
        lines.append(f"\nRelational complex: {rex.nV} vertices, {rex.nE} edges, {rex.nF} faces")
        if meta.get("type_names"):
            lines.append(f"Edge types: {', '.join(meta['type_names'])}")
        input_type = meta.get("input_type", "")
        if input_type:
            lines.append(f"Input type: {input_type}")

    # Topology
    topo = results.get("topology", {})
    if topo:
        betti = topo.get("betti", [])
        if betti:
            lines.append(f"\nTopology: betti = {betti}")
            if len(betti) > 1 and betti[1] > 0:
                lines.append(f"  beta_1 = {betti[1]} independent cycles (unresolved loops in the structure)")
        euler = topo.get("euler_characteristic")
        if euler is not None:
            lines.append(f"  Euler characteristic: {euler}")
        chain = results.get("construction", {}).get("chain_valid")
        if chain is not None:
            lines.append(f"  Chain condition (d^2=0): {'holds' if chain else 'violated'}")

    # Hodge decomposition
    hodge = results.get("hodge", {})
    if hodge:
        g = hodge.get("pct_gradient", 0)
        c = hodge.get("pct_curl", 0)
        h = hodge.get("pct_harmonic", 0)
        if g + c + h > 0:
            lines.append(f"\nHodge decomposition: {g:.1%} gradient, {c:.1%} curl, {h:.1%} harmonic")
            lines.append("  Gradient = hierarchical structure (information flows top-down)")
            lines.append("  Curl = circular references between sections")
            lines.append("  Harmonic = thematic threads that span the document without closing")

    # Structural character
    rel = results.get("relational", {})
    if rel:
        chi = rel.get("chi_mean")
        if chi and len(chi) >= 4:
            names = ["T(topology)", "G(geometry)", "F(frustration)", "C(coparticipation)"]
            parts = [f"{names[i]}={chi[i]:.3f}" for i in range(4)]
            lines.append(f"\nStructural character: {', '.join(parts)}")
            dominant = max(range(4), key=lambda i: chi[i])
            lines.append(f"  Dominant channel: {names[dominant]}")

        kappa = rel.get("kappa_mean")
        if kappa is not None:
            lines.append(f"\nCoherence (kappa): {kappa:.4f}")
            if kappa > 0.8:
                lines.append("  High coherence: sections relate consistently")
            elif kappa < 0.3:
                lines.append("  Low coherence: fragmented or loosely connected structure")

    # Void complex
    void = results.get("void", {})
    if void:
        nv = void.get("n_voids", 0)
        np_ = void.get("n_potential", 0)
        if np_ > 0:
            pct = nv / np_
            lines.append(f"\nVoid complex: {nv}/{np_} potential structures unrealized ({pct:.0%})")
            if pct > 0.5:
                lines.append("  Many expected relationships are missing from the document")

    # Epsilon
    eps = results.get("epsilon", {})
    if eps:
        e1 = eps.get("eps1_chain")
        if e1 is not None:
            lines.append(f"\nAxiom verification: chain condition residual = {e1:.2e}")

    lines.append("\nAnswer the user's questions using both the document text "
                 "and the structural analysis. When relevant, reference "
                 "specific structural features (Hodge fractions, coherence, "
                 "void count) to support your answers.")

    return "\n".join(lines)


@router.post("/model/generate")
async def model_generate(
    prompt: str = Body(..., embed=True),
    session_id: str = Body(None, embed=True),
    context: str = Body(None, embed=True),
    max_tokens: int = Body(1024, embed=True),
    temperature: float = Body(0.7, embed=True),
    stream: bool = Body(False, embed=True),
    include_structural: bool = Body(False, embed=True),
):
    """Send a prompt to the GPU model with session context.

    If session_id is provided, the analysis results are injected
    as a system prompt so the model understands the document's
    relational structure. If context is provided (e.g. OCR text),
    it's appended to the system prompt.

    Model resolved via ModelManager pipeline config, then fallbacks:
        1. ModelManager chat model (if assigned)
        2. CHAT_MODEL_URL env var
        3. Running GPU server
        4. UNLIMITED_OCR_URL env var
    """
    MAX_PROMPT_LENGTH = 50000
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(400, "Prompt exceeds maximum length (%d chars)" % MAX_PROMPT_LENGTH)

    # Resolve server URL via ModelManager first
    server_url = ""
    model_name = ""
    try:
        from agent.model_manager import get_manager
        mgr = get_manager()
        chat_model_id = mgr.get_pipeline_model("chat")
        if chat_model_id:
            lm = mgr.get(chat_model_id)
            if lm and lm.server_url:
                server_url = lm.server_url
                model_name = chat_model_id
    except Exception:
        pass

    # Fallbacks
    if not server_url:
        server_url = os.environ.get("CHAT_MODEL_URL", "")
    if not server_url:
        try:
            from agent.cli.serve import find_running_server
            server_url = find_running_server() or ""
        except ImportError:
            pass
    if not server_url:
        server_url = os.environ.get("UNLIMITED_OCR_URL", "")
    if not server_url:
        raise HTTPException(status_code=503,
            detail="No chat model available. Load one via Models tab "
                   "or start a GPU server: make ocr-serve")

    # Build context from session
    system_parts = []

    if session_id:
        try:
            from agent.server.app import get_store
            store = get_store()
            session = store.get(session_id)
            if session:
                # Get the latest analysis results
                snap = session.snapshots[session.current_step] if session.snapshots else None
                results = snap.results if snap else {}
                if not results:
                    from agent.pipeline import AnalysisPipeline
                    rex = session.current()
                    if rex:
                        results = AnalysisPipeline(rex).run(depth="standard")
                system_parts.append(_build_context(session, results))
        except Exception as e:
            system_parts.append(f"(Session context unavailable: {e})")

    if context:
        # Truncate to avoid blowing the context window
        max_ctx = 12000
        if len(context) > max_ctx:
            context = context[:max_ctx] + f"\n\n[... truncated, {len(context)} total chars]"
        system_parts.append(f"\nDocument text:\n{context}")

    messages = []
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.append({"role": "user", "content": prompt})

    import httpx

    # Let the server use its loaded model
    if not model_name:
        model_name = os.environ.get("CHAT_MODEL_NAME", "")

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    payload["logprobs"] = True   # for token perplexity/varentropy (both paths, ~free)
    if model_name:
        payload["model"] = model_name

    if stream:
        async def _stream():
            import json as _json
            logprobs = []
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", f"{server_url}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    async for line in resp.aiter_lines():
                        body = line[5:].strip() if line.startswith("data:") else ""
                        if body == "[DONE]":
                            continue  # hold the terminator; emit ours after metrics
                        yield line + "\n"
                        if body:       # accumulate token logprobs off the stream (free)
                            try:
                                obj = _json.loads(body)
                                for ch in obj.get("choices", []):
                                    lp = ch.get("logprobs") or {}
                                    for tok in (lp.get("content") or []):
                                        v = tok.get("logprob")
                                        if v is not None:
                                            logprobs.append(float(v))
                            except Exception:
                                pass
            # final metrics frame - token tier only (free); no per-reply complex build
            if logprobs:
                try:
                    from agent.metrics import token_metrics
                    yield ("data: " + _json.dumps({"metrics": {"token": token_metrics(logprobs)}}) + "\n\n")
                except Exception:
                    pass
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream(), media_type="text/event-stream")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            text = ""
            logprobs = []
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                text += msg.get("content", "")
                lp = choice.get("logprobs") or {}
                for tok in (lp.get("content") or []):
                    v = tok.get("logprob")
                    if v is not None:
                        logprobs.append(float(v))
            # Sanitize model output
            from agent.server.security import sanitize_model_response
            text = sanitize_model_response(text)

            # Metrics on the reply. Token metrics (perplexity/varentropy) are ~free -
            # extracted from the logprobs the model already returned - so always on.
            # The structural tier (build the reply's own complex, ~250 ms) is opt-in
            # via include_structural, so we never pay it on every reply.
            from agent.metrics import reply_metrics
            metrics = reply_metrics(text, logprobs=logprobs,
                                    structural=include_structural)

            return {
                "text": text,
                "model": data.get("model", ""),
                "usage": data.get("usage", {}),
                "context_included": bool(system_parts),
                "metrics": metrics,
            }
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="GPU server not reachable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def system_status():
    """Full system health."""
    status = {}

    try:
        import rexgraph
        from rexgraph.graph import RexGraph
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2])
        status["rexgraph"] = {
            "ok": True,
            "version": getattr(rexgraph, "__version__", "?"),
            "betti": list(rex.betti),
        }
    except Exception as e:
        status["rexgraph"] = {"ok": False, "error": str(e)}

    try:
        from agent.auto import auto_rex
        status["agent"] = {"ok": True}
    except Exception as e:
        status["agent"] = {"ok": False, "error": str(e)}

    backends = {}
    backends["tesseract"] = shutil.which("tesseract") is not None
    try:
        import paddleocr
        backends["paddleocr"] = True
    except ImportError:
        backends["paddleocr"] = False
    backends["mistral"] = bool(os.environ.get("MISTRAL_API_KEY", ""))
    try:
        import transformers, torch
        backends["got_ocr"] = True
    except ImportError:
        backends["got_ocr"] = False
    status["ocr_backends"] = backends

    try:
        from agent.cli.serve import server_status as _srv
        status["gpu_server"] = _srv()
    except Exception:
        status["gpu_server"] = {"status": "unknown"}

    # Chat model (may differ from OCR server)
    chat_url = os.environ.get("CHAT_MODEL_URL", "")
    if chat_url:
        status["chat_model"] = {"url": chat_url, "dedicated": True}
    else:
        status["chat_model"] = {
            "url": status.get("gpu_server", {}).get("url", ""),
            "dedicated": False,
            "note": "sharing OCR server",
        }

    try:
        from agent.cli.config import load_config
        cfg = load_config()
        if cfg.trustgraph_url:
            import urllib.request
            try:
                with urllib.request.urlopen(f"{cfg.trustgraph_url}/health", timeout=3) as resp:
                    status["trustgraph"] = {"ok": resp.status == 200, "url": cfg.trustgraph_url}
            except Exception:
                status["trustgraph"] = {"ok": False, "url": cfg.trustgraph_url}
        else:
            status["trustgraph"] = {"ok": False, "url": ""}
    except Exception:
        status["trustgraph"] = {"ok": False}

    try:
        from agent.cli.config import detect_platform
        p = detect_platform()
        status["platform"] = {
            "os": p.os, "arch": p.arch, "gpu": p.gpu_name or p.gpu,
            "gpu_vram_mb": p.gpu_vram_mb, "python": p.python, "scheduler": p.scheduler,
        }
    except Exception:
        status["platform"] = {}

    return status


@router.get("/model/chat-config")
async def chat_model_config():
    """Current chat-model setup status (which model chat/synthesis will use)."""
    from agent.chat_model import status
    return status()


@router.post("/model/chat-config")
async def set_chat_model_config(body: dict = Body(...)):
    """Configure the chat model.

    Body: {url?, model?, api_key?}. Pass url="" to clear the override and
    fall back to auto-resolution (ModelManager / running GPU server / env).
    """
    from agent.chat_model import configure, status
    configure(
        url=body.get("url"),
        model=body.get("model"),
        api_key=body.get("api_key"),
    )
    return {"ok": True, "status": status()}


# Managed local runtime (llama.cpp-family server as a first-class local backend)
@router.get("/model/local/status")
async def local_runtime_status():
    """Whether a managed local model server is running, its config, whether a
    llama.cpp binary is installed, and the recommended model catalog for this box."""
    from agent import local_runtime
    return local_runtime.status()


@router.get("/model/local/discover")
async def local_runtime_discover():
    """Auto-detect models already on disk - GGUF files (llama.cpp-loadable) and HF
    transformers snapshots (vLLM/transformers) across the common toolchain locations
    (HF cache, ollama, LM Studio, ~/models, our pull dir, + REXGRAPH_MODEL_DIRS)."""
    from agent import local_runtime
    return {"models": local_runtime.discover_local_models(),
            "searched": local_runtime._default_scan_dirs()}


@router.get("/model/local/endpoints")
async def local_runtime_endpoints():
    """Probe LIVE inference servers running on this host (Ollama / vLLM / llama.cpp /
    LM Studio / TGI on well-known ports + REXGRAPH_PROBE_URLS). Returns the reachable
    ones and the model ids each is serving - real backends the swarm can wire to."""
    from agent import local_runtime
    return {"endpoints": local_runtime.probe_endpoints(),
            "probed": [t["url"] for t in local_runtime._default_probe_targets()]}


@router.post("/model/local/start")
async def local_runtime_start(body: dict = Body(...)):
    """Launch a local llama.cpp server for a GGUF and make it the chat backend, so
    chat + metrics + agentic all run on the local model.
    body: {model_path (required), ctx_size?, n_gpu_layers?, flash_attn?, port?}."""
    from agent import local_runtime
    mp = body.get("model_path")
    if not mp:
        raise HTTPException(400, "provide 'model_path' (path to a .gguf)")
    try:
        return local_runtime.start(
            mp, ctx_size=body.get("ctx_size"), n_gpu_layers=body.get("n_gpu_layers"),
            flash_attn=body.get("flash_attn"), port=body.get("port"))
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@router.post("/model/local/stop")
async def local_runtime_stop():
    """Stop the managed local server and clear the chat-backend override."""
    from agent import local_runtime
    local_runtime.stop()
    return {"ok": True, "status": local_runtime.status()}


@router.post("/model/embedder/start")
async def embedder_start(body: dict = Body(...)):
    """Launch the dedicated embedding worker (the beehive's nomic-embed bee) ALONGSIDE the chat
    model, so the swarm's alignment/hallucination signal is always live. body: {model_path}."""
    from agent import local_runtime
    mp = body.get("model_path")
    if not mp:
        raise HTTPException(400, "provide 'model_path' (a .gguf embedding model, e.g. nomic-embed-text)")
    try:
        return local_runtime.start_embedder(mp, port=body.get("port"))
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@router.post("/model/embedder/stop")
async def embedder_stop():
    from agent import local_runtime
    local_runtime.stop_embedder()
    return {"ok": True, "embedder": local_runtime.embed_status()}


@router.get("/model/embedder/status")
async def embedder_status():
    from agent import local_runtime
    return local_runtime.embed_status()


@router.post("/model/introspect")
async def model_introspect_embeddings(body: dict = Body(...)):
    """Run the RCF math on the MODEL'S OWN embedding geometry (Tier-1 bridge): embed the
    items on the running local server, build a similarity complex, and return structural
    perplexity, coherence, Betti, and the load-bearing (bridge) concept pairs.
    body: {texts: [str, ...], top_p?}."""
    from agent import model_introspect
    texts = body.get("texts") or []
    if not isinstance(texts, list) or len(texts) < 3:
        raise HTTPException(400, "provide 'texts': a list of at least 3 strings")
    try:
        return model_introspect.embedding_complex(texts, top_p=float(body.get("top_p", 0.9)))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"introspection failed: {e}")


@router.get("/model/introspect/attention/available")
async def attention_capture_available():
    """Whether the Tier-2 attention-capture host is built (llama.cpp cb_eval bridge)."""
    from agent import attn_introspect
    return {"available": attn_introspect.available()}


@router.post("/model/introspect/attention")
async def model_introspect_attention(body: dict = Body(...)):
    """Tier-2: capture the running model's OWN per-layer attention (llama.cpp cb_eval, no ggml
    patch) and run the RCF analysis on each layer - Hodge grad/curl/harmonic, the four channels,
    Betti, coherence. The model reading its own attention through the relational-complex math.
    body: {prompt: str, layers?: [int], model_path?}."""
    from agent import attn_introspect
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(400, "provide 'prompt'")
    try:
        return attn_introspect.attention_complex(
            prompt, model_path=body.get("model_path"), layers=body.get("layers"))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"attention introspection failed: {e}")
