"""
Chat route: natural language interaction with the analysis.

Classifies user intent and dispatches:
  - data question -> look up in existing results
  - drill-down -> call specific rex method
  - general -> format results as structured summary (or LLM narration)
"""

from __future__ import annotations

import json
import logging

import numpy as np
from fastapi import APIRouter, Body

logger = logging.getLogger(__name__)

router = APIRouter()

# Safe RexGraph properties that can be accessed via chat
_SAFE_PROPERTIES = frozenset({
    "nV", "nE", "nF", "betti", "euler_characteristic",
    "coherence", "structural_character", "vertex_character",
    "degree", "nhats", "void_complex",
})


def _get_tracker(session_id: str, ws=None):
    """Get conversation tracker, workspace-scoped if available, session-scoped fallback."""
    if ws is not None:
        return ws.get_tracker(session_id)
    from agent.conversation import ConversationTracker
    # Fallback for when workspace isn't available
    if not hasattr(_get_tracker, '_fallback'):
        _get_tracker._fallback = {}
    if session_id not in _get_tracker._fallback:
        _get_tracker._fallback[session_id] = ConversationTracker()
    return _get_tracker._fallback[session_id]


#: the one encoder (rexgraph.io._compat). Non-finite floats go out as null:
#: a bare NaN token is not JSON and every browser JSON.parse rejects it.
from agent.server.scope import effective_workspace
from rexgraph.io._compat import json_sanitize


def _sanitize(obj):
    return json_sanitize(obj, nan="null")


# Simple intent keywords -> property mappings
_INTENT_MAP = {
    # Topology
    "betti": "betti",
    "euler": "euler_characteristic",
    "chain": "chain_valid",
    "topology": "betti",
    "connected": "betti",
    # Spectral
    "dirac": "dirac_eigenvalues",
    "eigenvalue": "eigenvalues_L0",
    "spectrum": "eigenvalues_L0",
    "fiedler": "fiedler_vector_L0",
    "spectral": "eigenvalues_L0",
    # Relational
    "character": "structural_character",
    "coherence": "coherence",
    "kappa": "coherence",
    "channel": "structural_character",
    "mixing": "per_channel_mixing_times",
    # Hodge
    "hodge": "_hodge",
    "gradient": "_hodge",
    "curl": "_hodge",
    "harmonic": "_hodge",
    # Void
    "void": "void_complex",
    "gap": "void_complex",
    "missing": "void_complex",
    # Energy
    "energy": "energy_kin_pot",
    # Persistence
    "persistence": "_persistence",
    "barcode": "_persistence",
}


def _classify_intent(message: str) -> tuple:
    """Simple keyword-based intent classification.

    Returns (intent_type, target) where:
        intent_type: 'property', 'hodge', 'explain', 'summary', 'unknown'
        target: property name or cell spec
    """
    msg = message.lower().strip()

    # Check for cell explanation requests (dim=0 vertex, dim=1 edge only)
    for dim_word, dim in [("vertex", 0), ("edge", 1)]:
        if dim_word in msg and any(c.isdigit() for c in msg):
            # Extract the index
            tokens = msg.split()
            for t in tokens:
                if t.isdigit():
                    return ("explain", (dim, int(t)))

    # Keyword matching (check before summary to catch "tell me about the eigenvalues")
    for keyword, prop in _INTENT_MAP.items():
        if keyword in msg:
            if prop == "_hodge":
                return ("hodge", None)
            if prop == "_persistence":
                return ("property", "betti")  # simplified for now
            return ("property", prop)

    # Check for summary request
    if any(w in msg for w in ["summary", "overview", "tell me about", "what is", "describe"]):
        return ("summary", None)

    return ("summary", None)


def _build_summary(rex, results: dict) -> str:
    """Build a structured text summary from available results."""
    lines = []
    meta = getattr(rex, "_agent_meta", {})

    lines.append(f"Relational complex: {rex.nV} vertices, {rex.nE} edges, {rex.nF} faces")

    if meta.get("type_names"):
        types = meta["type_names"]
        lines.append(f"Edge types: {', '.join(types)}")

    # Topology
    topo = results.get("topology", {})
    if topo:
        betti = topo.get("betti", [])
        if betti:
            lines.append(f"Betti numbers: β₀={betti[0]}, β₁={betti[1]}" +
                         (f", β₂={betti[2]}" if len(betti) > 2 else ""))
        euler = topo.get("euler_characteristic")
        if euler is not None:
            lines.append(f"Euler characteristic: {euler}")

    # Relational
    rel = results.get("relational", {})
    if rel:
        kappa = rel.get("kappa_mean")
        if kappa is not None:
            lines.append(f"Mean coherence κ: {kappa:.4f} (range {rel.get('kappa_min', 0):.3f}-{rel.get('kappa_max', 0):.3f})")
        chi = rel.get("chi_mean")
        if chi:
            channels = ["T(Hodge)", "G(Overlap)", "F(Frustration)", "C(Copath)"]
            parts = [f"{channels[i]}={chi[i]:.3f}" for i in range(min(len(chi), 4))]
            lines.append(f"Structural character: {', '.join(parts)}")

    # Hodge
    hodge = results.get("hodge", {})
    if hodge:
        g = hodge.get("pct_gradient", 0)
        c = hodge.get("pct_curl", 0)
        h = hodge.get("pct_harmonic", 0)
        if g + c + h > 0:
            lines.append(f"Hodge decomposition: {g:.1%} gradient, {c:.1%} curl, {h:.1%} harmonic")

    # Void
    void = results.get("void", {})
    if void:
        nv = void.get("n_voids", 0)
        np_ = void.get("n_potential", 0)
        if np_ > 0:
            lines.append(f"Void complex: {nv} voids out of {np_} potential ({nv/np_:.0%} unrealized)")

    # Epsilon
    eps = results.get("epsilon", {})
    if eps:
        e1 = eps.get("eps1_chain", 0)
        e3 = eps.get("eps3_equiweight")
        if e3 is not None:
            lines.append(f"Axiom verification: chain={e1:.2e}, equiweight={e3:.2e}")

    return "\n".join(lines)


@router.get("/chat/{session_id}/metrics")
async def chat_session_metrics(session_id: str, structural: bool = False):
    """Per-session information metrics: the trend of coherence/perplexity over turns,
    plus per-message metrics. Token metrics are free (always present). The structural
    tier (~250 ms/message) is computed lazily and cached ONLY when `?structural=1` -
    so the interface pays for it exactly when the user drills into structure."""
    _ws = None
    try:
        from agent.server.auth import get_auth_manager
        _ws = get_auth_manager().get_workspace(effective_workspace("default"))
    except Exception:
        pass
    tracker = _get_tracker(session_id, _ws)
    return {
        "session": tracker.session_metrics(),                 # trends (cheap)
        "per_message": tracker.exchange_metrics(structural=structural),
    }


@router.post("/chat/{session_id}")
async def chat(session_id: str, body: dict = Body(...)):
    """Process a chat message."""
    from agent.server.app import get_store
    # Get workspace if auth module is available
    _ws = None
    try:
        from agent.server.auth import get_auth_manager
        mgr = get_auth_manager()
        _ws = mgr.get_workspace(effective_workspace("default"))
    except Exception:
        pass
    store = get_store()
    session = store.get(session_id)

    # Auto-create session if it doesn't exist (chat-initiated sessions)
    if session is None:
        session = store.create(name="chat-" + session_id)

    rex = session.current()
    message = body.get("message", "")

    # If no data is loaded, respond helpfully instead of erroring
    if rex is None:
        return {
            "response": "No document loaded in this session. Upload a file "
            "through the Pipeline tab first, then come back to chat. I'll "
            "be able to answer questions using the structural analysis.",
            "session_id": session_id,
        }
    intent_type, target = _classify_intent(message)

    # Get cached results from the current snapshot
    current = session.snapshots[session.current_step] if session.snapshots else None
    cached_results = current.results if current else {}

    if intent_type == "property" and target:
        if target not in _SAFE_PROPERTIES:
            response = {"text": f"Property '{target}' is not accessible.", "property": None, "viz_update": None}
        else:
            try:
                value = getattr(rex, target)
                result = _sanitize(value)
                response = {
                    "text": f"{target}: {result}",
                    "property": target,
                    "viz_update": {"highlight_property": target},
                }
            except Exception as e:
                response = {
                    "text": f"Couldn't compute {target}: {e}",
                    "property": None,
                    "viz_update": None,
                }

    elif intent_type == "explain":
        dim, idx = target
        try:
            result = rex.explain(dim, idx)
            dim_name = ["vertex", "edge", "face"][dim]
            response = {
                "text": f"Explanation for {dim_name} {idx}:\n{json.dumps(_sanitize(result), indent=2)}",
                "property": f"explain_{dim}_{idx}",
                "viz_update": {"highlight_cell": {"dim": dim, "idx": idx}},
            }
        except Exception as e:
            response = {"text": str(e), "property": None, "viz_update": None}

    elif intent_type == "hodge":
        try:
            flow = np.ones(rex.nE, dtype=np.float64)
            result = rex.hodge_full(flow)
            hodge_text = (
                f"Hodge decomposition (uniform flow):\n"
                f"  Gradient: {result.get('pct_grad', 0):.1%}\n"
                f"  Curl: {result.get('pct_curl', 0):.1%}\n"
                f"  Harmonic: {result.get('pct_harm', 0):.1%}"
            )
            response = {
                "text": hodge_text,
                "property": "hodge",
                "viz_update": {"show_hodge": True},
            }
        except Exception as e:
            response = {"text": str(e), "property": None, "viz_update": None}

    else:
        # General question -> per-query relational complex, structural
        # retrieval from the document/corpus, and grounded synthesis
        # (language model if configured, structural answer otherwise).
        from agent import query_engine
        doc_summary = _build_summary(rex, cached_results or {})
        # The chat is scoped to this session's document, so that document is
        # the primary retrieval source. Only fall back to the workspace corpus
        # when the session has no usable document of its own (e.g. a fresh
        # session), so building a corpus elsewhere doesn't hijack single-doc chat.
        corpus = None
        has_doc = False
        try:
            meta = getattr(rex, "_agent_meta", {}) if rex is not None else {}
            has_doc = bool(meta.get("source_text"))
        except Exception:
            has_doc = False
        if not has_doc:
            try:
                if _ws is not None and hasattr(_ws, "get_corpus"):
                    cp = _ws.get_corpus()
                    if cp is not None and len(getattr(cp, "documents", []) or []) > 0:
                        corpus = cp
            except Exception:
                corpus = None
        # ...and then the PERSISTED corpus, which is the last resort and the largest one.
        # `default_store()` resolves REXGRAPH_RCDB_URI and is workspace-scoped when auth
        # is on, so this reaches what was ingested rather than a throwaway. It comes last
        # for the same reason the workspace corpus does: a session that has its own
        # document is asking about THAT document, and a store holding 61,353 others must
        # not answer over it.
        rc_store = None
        if not has_doc and corpus is None:
            try:
                from agent.rcdb import default_store
                rc_store = default_store()
            except Exception:
                # The corpus is optional here: with no store configured the chat
                # answers over the complex it was given and does not reach past it.
                rc_store = None
        try:
            qa = query_engine.answer_query(
                rex, message, cached_results,
                corpus=corpus, doc_summary=doc_summary, store=rc_store)
            response = {
                "text": qa["answer"] or doc_summary or "Upload data to begin analysis.",
                "property": None,
                "viz_update": None,
                "query_complex": qa.get("query_complex"),
                "sections": qa.get("sections"),
                "relation": qa.get("relation"),
                "model_used": qa.get("model_used", False),
                "cached": qa.get("cached", False),
                "_token_metrics": qa.get("token_metrics") or {},
            }
        except Exception as e:
            response = {
                "text": doc_summary or "Upload data to begin analysis.",
                "property": None,
                "viz_update": None,
                "engine_error": str(e),
            }

    # Record exchange and attach drift info
    tracker = _get_tracker(session_id, _ws)
    try:
        response_text = response.get("text", "")
        ex_result = tracker.record_exchange(message, response_text)
        response["exchange"] = {
            "n_shared": ex_result.n_shared,
            "exchange_edges": ex_result.n_exchange_edges,
            "kappa": ex_result.kappa_mean,
        }
        # Per-message metrics: token (from the reply's logprobs, if the model path
        # ran) + structural (the reply's own complex) + fluent-but-hollow advisory -
        # attached to THIS message and stored on the exchange, so returning to any
        # point in the conversation shows its metrics, and the session trend fills in.
        try:
            from agent.metrics import reply_metrics
            tok = response.pop("_token_metrics", None)
            # Token tier only (free) on every reply; the reply text is stored so the
            # structural tier is computed lazily via GET /chat/{sid}/metrics?structural=1.
            msg_metrics = reply_metrics(response_text, token=tok, structural=False)
            response["metrics"] = msg_metrics
            tracker.note_exchange_metrics(msg_metrics, text=response_text)
        except Exception:
            response.pop("_token_metrics", None)
        if tracker.n_exchanges >= 2:
            drift = tracker.get_drift_report()
            response["drift"] = {
                "kappa_drift": drift.get("kappa_drift", 0),
                "context_status": drift.get("context_status", ""),
                "n_exchanges": drift.get("n_exchanges", 0),
            }
            memory = tracker.get_memory_edges()
            if memory:
                response["persistent_entities"] = memory[:20]
    except Exception:
        pass

    # Conversational memory: one step of this session's temporal rex per turn, when
    # the workspace has asked for it. A recording failure is not a chat failure.
    try:
        from agent import work_recorder as wr
        turns = []
        for i, ex in enumerate(getattr(tracker, "exchanges", []) or []):
            turns.append(f"{i}:{(getattr(ex, 'query', '') or '')[:60]}")
        if len(turns) < 2:
            turns = [f"{i}:turn" for i in range(max(2, len(turns)))]
        wr.record("conversation", turns, lineage_id=f"chat:{session_id}",
                  workspace=getattr(_ws, "name", "default") if _ws else "default")
    except Exception:
        logger.debug("conversation not recorded for %s", session_id, exc_info=True)

    return response
