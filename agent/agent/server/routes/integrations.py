"""
agent.server.routes.integrations: TrustGraph, HuggingFace, LangChain, LangGraph.

All integration code already exists in agent/integrations/. These routes
are thin wrappers that expose it over HTTP.
"""

from __future__ import annotations

import logging
import math
import json

import numpy as np
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


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


def _rex_from_body(body: dict):
    """Resolve a RexGraph from a request body.

    Accepts either ``session_id`` (uses the uploaded document's complex)
    or ``text`` (builds a complex on the fly). Returns ``(rex, source)``
    or ``(None, reason)``.
    """
    sid = body.get("session_id")
    if sid:
        try:
            from agent.server.app import get_store
            session = get_store().get(sid)
            if session is not None and session.current() is not None:
                return session.current(), f"session:{sid}"
        except Exception:
            pass
    text = body.get("text", "")
    if text and text.strip():
        try:
            from agent.auto import auto_rex
            return auto_rex(text), "text"
        except Exception as e:
            return None, f"build failed: {e}"
    return None, "no session_id or text provided"


# TrustGraph

def _get_tg_adapter(url=None):
    try:
        from agent.integrations.trustgraph_adapter import TrustGraphAdapter
    except ImportError:
        raise HTTPException(
            500, "TrustGraph adapter not available. "
            "Install with: pip install trustgraph-base"
        )
    import os
    tg_url = url or os.environ.get("TRUSTGRAPH_URL", "")
    return TrustGraphAdapter(url=tg_url or None)


@router.post("/trustgraph/health")
async def trustgraph_health(body: dict = Body(...)):
    """Health snapshot for a TrustGraph flow."""
    adapter = _get_tg_adapter(body.get("url"))
    flow = body.get("flow", "default")

    try:
        if adapter.url:
            result = adapter.health_snapshot(flow=flow)
        else:
            # Standalone mode - need triples
            triples = body.get("triples")
            if not triples:
                return JSONResponse(_sanitize({
                    "status": "no_connection",
                    "nV": 0, "nE": 0, "nF": 0,
                    "dim_H": 0, "health_ratio": None,
                    "cost_multiplier": 1.0,
                    "message": "No TrustGraph URL configured and no triples provided. "
                    "Set TRUSTGRAPH_URL or pass triples in the request body.",
                }))
            from agent.integrations.trustgraph_adapter import SimpleTriple
            tlist = [SimpleTriple(t[0], t[1], t[2]) for t in triples]
            rex, meta = adapter.from_triples(tlist)
            result = adapter.health_snapshot(rex=rex, meta=meta)
    except Exception as e:
        raise HTTPException(500, f"TrustGraph health check failed: {e}")

    # Remove non-serializable rex object if present
    result.pop("rex", None)
    return JSONResponse(_sanitize(result))


@router.post("/trustgraph/compare")
async def trustgraph_compare(body: dict = Body(...)):
    """Compare structural health of multiple TrustGraph flows."""
    adapter = _get_tg_adapter(body.get("url"))
    flows = body.get("flows", [])
    if not flows:
        raise HTTPException(400, "Provide a list of flow names")

    try:
        result = adapter.compare_flows(flows, depth=body.get("depth", "standard"))
    except Exception as e:
        raise HTTPException(500, f"Flow comparison failed: {e}")

    # Strip rex objects from per_flow results
    for k, v in result.get("per_flow", {}).items():
        if isinstance(v, dict):
            v.pop("rex", None)

    return JSONResponse(_sanitize(result))


@router.post("/trustgraph/evolution")
async def trustgraph_evolution(body: dict = Body(...)):
    """Track how a knowledge graph evolves across core versions."""
    adapter = _get_tg_adapter(body.get("url"))
    flow = body.get("flow", "default")
    snapshots = body.get("snapshots")

    try:
        result = adapter.track_evolution(flow=flow, snapshots=snapshots)
    except Exception as e:
        raise HTTPException(500, f"Evolution tracking failed: {e}")

    return JSONResponse(_sanitize(result))


@router.post("/trustgraph/assess")
async def trustgraph_assess(body: dict = Body(...)):
    """Assess a query's structural complexity."""
    adapter = _get_tg_adapter(body.get("url"))
    entities = body.get("entities", [])
    flow = body.get("flow", "default")

    if not entities:
        raise HTTPException(400, "Provide a list of entity names")

    try:
        if adapter.url:
            result = adapter.assess_query(entities, flow=flow)
        else:
            triples = body.get("triples")
            if not triples:
                raise HTTPException(
                    400, "No TrustGraph URL configured. Provide triples or set TRUSTGRAPH_URL."
                )
            from agent.integrations.trustgraph_adapter import SimpleTriple
            tlist = [SimpleTriple(t[0], t[1], t[2]) for t in triples]
            rex, meta = adapter.from_triples(tlist)
            result = adapter.assess_query(entities, rex=rex, meta=meta)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Query assessment failed: {e}")

    return JSONResponse(_sanitize(result))


# HuggingFace

@router.post("/huggingface/analyze")
async def huggingface_analyze(body: dict = Body(...)):
    """Measure RCF axiom compliance.

    With ``model``: hooks a HuggingFace transformer and measures how far
    its attention structure departs from a valid relational complex
    (∂²=0 / chain condition, equiweight). Without ``model``: analyzes the
    text's co-occurrence complex and reports its axiom compliance - clearly
    labeled as ``mode: text_cooccurrence`` so it's not mistaken for a
    transformer probe. Accepts ``session_id`` in place of ``text``.
    """
    text = body.get("text", "")
    model_name = body.get("model", "")
    if not text and not body.get("session_id"):
        raise HTTPException(400, "Provide 'text', a 'session_id', or a 'model'")

    if model_name:
        try:
            from agent.integrations.huggingface_analyzer import analyze_transformer
        except ImportError:
            raise HTTPException(
                500, "Transformer analysis needs torch + transformers. "
                "Install with: pip install torch transformers "
                "(or omit 'model' for text-level axiom analysis).")
        try:
            result = analyze_transformer(model_name=model_name, text=text,
                                         device=body.get("device", "cuda"))
            result["mode"] = "transformer"
            return JSONResponse(_sanitize(result))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Transformer analysis failed: {e}")

    # Standalone: axiom compliance of the text's co-occurrence complex.
    rex, source = _rex_from_body(body)
    if rex is None:
        raise HTTPException(400, f"No input: {source}")
    try:
        result = {
            "mode": "text_cooccurrence",
            "source": source,
            "nV": rex.nV, "nE": rex.nE, "nF": rex.nF,
            "betti": [int(b) for b in rex.betti],
            "chain_condition": {
                "satisfied": bool(rex.chain_valid),
                "meaning": "∂²=0 holds - the co-occurrence structure is a "
                           "valid relational complex" if rex.chain_valid else
                           "∂²≠0 - boundary-of-boundary is nonzero",
            },
        }
        try:
            kappa = rex.coherence
            result["kappa_mean"] = round(float(kappa.mean()), 4)
        except Exception:
            pass
        result["note"] = ("Pass 'model' (with torch+transformers) to measure a "
                          "transformer's attention against these axioms.")
        return JSONResponse(_sanitize(result))
    except Exception as e:
        raise HTTPException(500, f"Text axiom analysis failed: {e}")


# LangChain

@router.post("/langchain/tools")
async def langchain_tools(body: dict = Body({})):
    """List available LangChain tools with descriptions."""
    tools = [
        {
            "name": "RexConfidenceTool",
            "description": "Check structural confidence before answering. "
            "Returns void affinity, dipole ratio, chain condition, and coherence. "
            "If void_affinity > 0.5 or kappa < 0.3, qualify the response.",
        },
        {
            "name": "RexAnalyzeTool",
            "description": "Full structural analysis on demand. Returns Hodge "
            "decomposition percentages, Betti numbers, and channel character.",
        },
        {
            "name": "RexHodgeTool",
            "description": "Decompose a specific signal into gradient, curl, "
            "and harmonic components. Reports energy fraction in each.",
        },
        {
            "name": "RexExplainTool",
            "description": "Explain a specific topological feature (a Betti "
            "generator, a void, a persistent cycle) in terms of edges and vertices.",
        },
    ]

    # Check if langchain is actually installed
    installed = False
    try:
        from langchain_core.tools import BaseTool  # noqa: F401
        installed = True
    except ImportError:
        pass

    return {
        "tools": tools,
        "installed": installed,
        "install_cmd": "pip install langchain-core" if not installed else None,
        "usage": 'from agent.integrations.langchain_tools import get_rex_tools; tools = get_rex_tools(rex)',
    }


# LangGraph

@router.post("/langgraph/state")
async def langgraph_state(body: dict = Body({})):
    """Analyze an agent state machine as a relational complex.

    Works standalone (no langgraph needed) - nodes are states, edges are
    transitions. Reports whether the agent is making progress (gradient),
    circulating (curl), or structurally stuck (harmonic), plus cycles and
    channel character.
    """
    try:
        from agent.integrations.langgraph_rex import RexStateGraph
    except Exception as e:
        raise HTTPException(500, f"State-graph analysis unavailable: {e}")

    try:
        rsg = RexStateGraph()
        states = body.get("states", ["retrieve", "reason", "answer"])
        transitions = body.get("transitions", [
            {"from": "retrieve", "to": "reason", "weight": 1.0},
            {"from": "reason", "to": "answer", "weight": 1.0},
            {"from": "reason", "to": "retrieve", "weight": 0.5},
        ])
        for s in states:
            if isinstance(s, str):
                rsg.add_state(s)
            elif isinstance(s, dict):
                rsg.add_state(s["name"], metadata=s.get("metadata"))
        for t in transitions:
            rsg.add_transition(t["from"], t["to"], weight=t.get("weight", 1.0))

        a = rsg.analyze()
        topo = a.get("topology", {}) or {}
        hodge = a.get("hodge", {}) or {}
        decision = rsg.should_continue(
            harmonic_threshold=body.get("harmonic_threshold", 0.4))

        result = {
            "nV": len(states),
            "nE": len(transitions),
            "betti": [topo.get("b0"), topo.get("b1_filled"), topo.get("b2")],
            "euler": topo.get("euler"),
            "chain_valid": topo.get("chainOk"),
            "hodge": {
                "gradient": round(float(hodge.get("gradPct", 0)) / 100.0, 4),
                "curl": round(float(hodge.get("curlPct", 0)) / 100.0, 4),
                "harmonic": round(float(hodge.get("harmPct", 0)) / 100.0, 4),
            },
            "recommendation": decision.get("recommendation", "continue"),
            "reason": decision.get("reason", ""),
            "cycles": rsg.detect_cycles(),
            "channel_profile": rsg.channel_profile(),
        }
        return JSONResponse(_sanitize(result))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"LangGraph analysis failed: {e}")


# Training

@router.post("/model/training")
async def generate_training_data(body: dict = Body(...)):
    """Generate structural training data from the active corpus."""
    target = body.get("target", "summary")
    fmt = body.get("format", "safetensors")

    from agent.server.auth import require_workspace, get_auth_manager
    mgr = get_auth_manager()
    ws = mgr.get_workspace("default")

    corpus = ws.get_corpus()
    if not corpus or not corpus._built:
        raise HTTPException(400, "Build a corpus first before generating training data")

    try:
        from agent.training import TrainingExporter
        exporter = TrainingExporter(corpus=corpus)

        import tempfile
        import os

        result = {
            "n_samples": len(exporter.examples),
            "n_features": len(exporter.feature_names),
            "features": exporter.feature_names,
            "target": target,
            "format": fmt,
        }

        # Export to temp file
        tmp_dir = tempfile.mkdtemp(prefix="rexgraph_training_")

        if fmt == "safetensors":
            path = os.path.join(tmp_dir, "features.safetensors")
            exporter.export_features(path)
            result["path"] = path
        elif fmt == "pairs":
            path = os.path.join(tmp_dir, "training_pairs.json")
            exporter.export_training_pairs(path, target=target)
            result["path"] = path
        elif fmt == "hf_dataset":
            try:
                ds = exporter.to_hf_dataset()
                result["dataset_info"] = str(ds) if ds else "Generated"
            except Exception as e:
                result["error"] = f"HF dataset export failed: {e}"
        elif fmt == "rex":
            exporter.export_rex_bundles(tmp_dir)
            result["path"] = tmp_dir
        else:
            raise HTTPException(400, f"Unknown format: {fmt}")

        return JSONResponse(_sanitize(result))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Training data generation failed: {e}")


# vLLM structural router

@router.post("/vllm/route")
async def vllm_route(body: dict = Body(...)):
    """Route a prompt to a model *class* using structural character.

    Builds a relational complex from the prompt's token relationships and
    routes on the dominant RCF channel - no second LLM needed:
      T (topology) -> reasoning · G (geometry) -> creative ·
      F (frustration) -> analytical · C (copath) -> multi-hop.
    Pass optional ``models`` {capability: model_id} to get a concrete id.
    """
    text = body.get("text", "") or body.get("prompt", "")
    if not text.strip():
        raise HTTPException(400, "Provide 'text' (the prompt) to route")
    try:
        from agent.integrations.vllm_router import RexRouter
    except Exception as e:
        raise HTTPException(500, f"Router unavailable: {e}")

    caps = ["reasoning", "creative", "analytical", "multi-hop"]
    models = body.get("models") or {c: c for c in caps}
    try:
        router_ = RexRouter(models, default=body.get("default", "reasoning"),
                            void_threshold=body.get("void_threshold", 0.5))
        choice, diag = router_.route(text, window=body.get("window", 3))
        return JSONResponse(_sanitize({
            "routed_to": choice,
            "dominant_channel": diag.get("dominant_channel"),
            "dominant_fraction": diag.get("dominant_fraction"),
            "confidence": diag.get("confidence"),
            "reason": diag.get("reason"),
            "character": {"T": diag.get("chi_T"), "G": diag.get("chi_G"),
                          "F": diag.get("chi_F"), "C": diag.get("chi_C")},
            "hodge": {"gradient": diag.get("hodge_gradient"),
                      "curl": diag.get("hodge_curl"),
                      "harmonic": diag.get("hodge_harmonic")},
            "kappa_mean": diag.get("kappa_mean"),
            "nE": diag.get("nE"), "nF": diag.get("nF"),
        }))
    except Exception as e:
        raise HTTPException(500, f"Routing failed: {e}")


# LangChain confidence / analyze (runnable, no langchain dep)

def _confidence_report(rex) -> dict:
    """The RexConfidenceTool computation, callable without langchain."""
    import numpy as _np
    out = {"nV": rex.nV, "nE": rex.nE, "nF": rex.nF}
    f_E = _np.ones(rex.nE, dtype=_np.float64)
    try:
        dipole = rex.face_void_dipole(f_E)
        out["void_affinity"] = round(float(dipole.get("void_affinity", 0)), 4)
        out["face_affinity"] = round(float(dipole.get("face_affinity", 0)), 4)
        out["dipole_ratio"] = round(float(dipole.get("dipole_ratio", 0)), 4)
    except Exception:
        out["void_affinity"] = None
    try:
        kappa = rex.coherence
        out["kappa_mean"] = round(float(kappa.mean()), 4)
        out["kappa_min"] = round(float(kappa.min()), 4)
    except Exception:
        pass
    try:
        out["chain_valid"] = bool(rex.chain_valid)
    except Exception:
        pass
    try:
        vc = rex.void_complex
        out["n_voids"] = vc.get("n_voids", 0)
        out["n_potential"] = vc.get("n_potential", 0)
    except Exception:
        pass
    # verdict an agent can act on
    va = out.get("void_affinity") or 0
    km = out.get("kappa_mean")
    if km is not None and (va > 0.5 or km < 0.3):
        out["verdict"] = "low_confidence"
        out["guidance"] = ("High void affinity or low coherence - the structure "
                           "supporting an answer is weak. Qualify or refuse.")
    else:
        out["verdict"] = "supported"
        out["guidance"] = "Structural support is adequate for a direct answer."
    return out


@router.post("/langchain/confidence")
async def langchain_confidence(body: dict = Body(...)):
    """Run RexConfidenceTool against a session document or text.

    Gives an agent an exact structural confidence signal (void affinity,
    dipole ratio, coherence, chain condition) - a theorem, not a guess.
    No langchain install required to call it.
    """
    rex, source = _rex_from_body(body)
    if rex is None:
        raise HTTPException(400, f"No document: {source}")
    try:
        report = _confidence_report(rex)
        report["source"] = source
        return JSONResponse(_sanitize(report))
    except Exception as e:
        raise HTTPException(500, f"Confidence check failed: {e}")


@router.post("/langchain/analyze")
async def langchain_analyze(body: dict = Body(...)):
    """Run RexAnalyzeTool (full structural analysis) against a session or text."""
    rex, source = _rex_from_body(body)
    if rex is None:
        raise HTTPException(400, f"No document: {source}")
    try:
        import numpy as _np
        a = {"nV": rex.nV, "nE": rex.nE, "nF": rex.nF,
             "betti": [int(b) for b in rex.betti],
             "chain_valid": bool(rex.chain_valid), "source": source}
        try:
            h = rex.hodge_full(_np.ones(rex.nE, dtype=_np.float64))
            a["hodge"] = {"gradient": round(float(h.get("pct_grad", 0)), 4),
                          "curl": round(float(h.get("pct_curl", 0)), 4),
                          "harmonic": round(float(h.get("pct_harm", 0)), 4)}
        except Exception:
            pass
        try:
            a["kappa_mean"] = round(float(rex.coherence.mean()), 4)
        except Exception:
            pass
        return JSONResponse(_sanitize(a))
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")


@router.post("/trustgraph/analyze")
async def trustgraph_analyze(body: dict = Body(...)):
    """Full structural analysis of a knowledge graph given as triples.

    Standalone - no TrustGraph server needed. Turns (subject, predicate,
    object) triples into a typed relational complex and returns the
    topology, Hodge decomposition, void complex, and channel character
    that RexGraph computes but a plain knowledge graph can't see.
    """
    triples = body.get("triples")
    if not triples:
        raise HTTPException(400, "Provide 'triples' as [[subject, predicate, object], …]")
    try:
        from agent.integrations.trustgraph_adapter import (
            TrustGraphAdapter, SimpleTriple)
        adapter = TrustGraphAdapter(url=None)
        tlist = [SimpleTriple(t[0], t[1], t[2]) for t in triples]
        rex, meta = adapter.from_triples(tlist)
    except Exception as e:
        raise HTTPException(500, f"Failed to build complex from triples: {e}")

    try:
        import numpy as _np
        out = {
            "nV": rex.nV, "nE": rex.nE, "nF": rex.nF,
            "n_entities": rex.nV,
            "n_relations": len(triples),
            "predicate_types": meta.get("type_names") if isinstance(meta, dict) else None,
            "betti": [int(b) for b in rex.betti],
            "chain_valid": bool(rex.chain_valid),
        }
        try:
            h = rex.hodge_full(_np.ones(rex.nE, dtype=_np.float64))
            out["hodge"] = {"gradient": round(float(h.get("pct_grad", 0)), 4),
                            "curl": round(float(h.get("pct_curl", 0)), 4),
                            "harmonic": round(float(h.get("pct_harm", 0)), 4)}
        except Exception:
            pass
        try:
            vc = rex.void_complex
            out["void_complex"] = {"n_voids": vc.get("n_voids", 0),
                                   "n_potential": vc.get("n_potential", 0)}
        except Exception:
            pass
        try:
            out["kappa_mean"] = round(float(rex.coherence.mean()), 4)
        except Exception:
            pass
        out["interpretation"] = (
            "Harmonic mass flags knowledge that loops without grounding; "
            "voids are relationships the graph's structure implies but that "
            "are absent - candidate missing edges.")
        return JSONResponse(_sanitize(out))
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")


@router.get("/model/training/download")
async def download_training_data(fmt: str = "safetensors", target: str = "summary"):
    """Stream generated structural training data as a downloadable file.

    Formats: ``safetensors`` (feature matrix) or ``pairs`` (input->target
    training pairs). Both are safetensors - directly loadable in
    PyTorch/JAX/HuggingFace for training or fine-tuning on RexGraph's
    structural features.
    """
    from agent.server.auth import get_auth_manager
    ws = get_auth_manager().get_workspace("default")
    corpus = ws.get_corpus()
    if not corpus or not getattr(corpus, "_built", False):
        raise HTTPException(400, "Build a corpus first (Corpus tab) before exporting")
    try:
        from agent.training import TrainingExporter
        exporter = TrainingExporter(corpus=corpus)
        import tempfile, os
        fd, tmp = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        if fmt == "pairs":
            exporter.export_training_pairs(tmp, target=target)
            fname = "rexgraph_training_pairs.safetensors"
        elif fmt == "safetensors":
            exporter.export_features(tmp)
            fname = "rexgraph_features.safetensors"
        else:
            raise HTTPException(400, f"Downloadable formats: safetensors, pairs "
                                     f"('{fmt}' is metadata-only)")
        return FileResponse(tmp, filename=fname,
                            media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Training export failed: {e}")
