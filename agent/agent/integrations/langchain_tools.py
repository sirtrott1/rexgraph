"""
LangChain tools for RexGraph - mathematically grounded agent operations.

The core value: your LLM agent hallucinates and doesn't know it.
These tools give it exact structural confidence from the void complex
and epsilon tower. Not a probability estimate - a theorem.

Usage:

    from rexgraph.integrations.langchain_tools import (
        RexConfidenceTool,
        RexAnalyzeTool,
        RexHodgeTool,
        RexExplainTool,
    )
    from langchain.agents import AgentExecutor, create_tool_calling_agent

    tools = [RexConfidenceTool(rex), RexAnalyzeTool(rex)]
    agent = create_tool_calling_agent(llm, tools, prompt)

Requirements: pip install rexgraph[langchain]
"""

from __future__ import annotations

from typing import Any, Type

import numpy as np

try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    # Stub so the module can be imported for inspection
    class BaseTool:
        pass
    class BaseModel:
        pass
    def Field(*a, **kw):
        return None


def _require_langchain():
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "LangChain integration requires langchain-core.\n"
            "Install with: pip install rexgraph[langchain]"
        )


def _resolve_query_seeds(query: str, rex) -> list:
    """Resolve a natural-language query to the vertex indices of the entities it
    names, by matching query tokens against the complex's vertex_labels. Returns the
    seed indices for a demand-driven, topic-scoped reading (empty if none match)."""
    meta = getattr(rex, "_agent_meta", {}) or {}
    labels = list(meta.get("vertex_labels", []) or [])
    if not labels or not query:
        return []
    q = query.lower()
    toks = {w for w in "".join(c if c.isalnum() else " " for c in q).split() if len(w) > 2}
    seeds = []
    for i, lab in enumerate(labels):
        ll = str(lab).lower()
        if ll in q or ll in toks or any(tok in ll for tok in toks if len(tok) > 3):
            seeds.append(i)
    return seeds


class ConfidenceInput(BaseModel):
    """Input for confidence check."""
    query: str = Field(description="The question or topic to check confidence on")
    signal: str = Field(
        default="uniform",
        description="Edge signal to evaluate: 'uniform' or comma-separated floats"
    )


class RexConfidenceTool(BaseTool):
    """Check structural confidence before answering.

    Returns a mathematical confidence assessment based on:
    - void_affinity: how much of the signal falls in structural gaps
    - dipole_ratio: face vs void balance (-1 = all void, +1 = all face)
    - eps1: chain condition violation (should be ~0)
    - kappa_mean: cross-dimensional coherence

    If void_affinity > 0.5 or kappa_mean < 0.3, the structure is
    unreliable in this region. The agent should say so.

    This is not a probability estimate. It's a count of structural gaps
    and measured axiom violations. The math either has structure here
    or it doesn't.
    """

    name: str = "rex_confidence"
    description: str = (
        "Check mathematical confidence for a topic. Returns structural "
        "reliability metrics. Use BEFORE answering uncertain questions. "
        "If void_affinity > 0.5, say you don't have reliable structure."
    )
    args_schema: Type[BaseModel] = ConfidenceInput

    rex: Any = None  # RexGraph instance

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, rex, **kwargs):
        _require_langchain()
        super().__init__(rex=rex, **kwargs)

    def _run(self, query: str = "", signal: str = "uniform") -> str:
        rex = self.rex
        nE = rex.nE

        # TOPIC-SCOPED path: if the query names entities in the complex, give a
        # demand-driven reading of the sub-complex the topic actually activates -
        # not a whole-graph mean (which the void/κ globals below would dilute).
        seeds = _resolve_query_seeds(query, rex)
        if seeds:
            try:
                ar = rex.agentic_reading(vertices=seeds, t=1.0)
                kr = np.asarray(rex.coherence_response(seeds), dtype=float)
                meta = getattr(rex, "_agent_meta", {}) or {}
                labels = list(meta.get("vertex_labels", []) or [])
                km = round(float(kr.mean()), 4)
                n_bridge = sum(1 for lb in ar["load_bearing"]
                               if lb["effective_resistance"] > 0.9)
                scoped = {
                    "scope": "topic",
                    "query_entities": [labels[i] if i < len(labels) else str(i)
                                       for i in seeds],
                    "kappa_mean": km,
                    "kappa_min": round(float(kr.min()), 4),
                    "context_size": ar["context_size"],
                    "load_bearing_relations": n_bridge,
                    "frustrated_entities": len(ar["frustrated"]),
                }
                if km < 0.3:
                    scoped["confidence"] = ("LOW - the entities in this topic are "
                                            "incoherent (edge and vertex structure disagree)")
                elif km > 0.7 and n_bridge == 0:
                    scoped["confidence"] = ("HIGH - coherent topic with no fragile "
                                            "bridge relations")
                elif n_bridge > 0:
                    scoped["confidence"] = (f"MODERATE - {n_bridge} load-bearing relation(s); "
                                            "answer depends on links that have no backup")
                else:
                    scoped["confidence"] = "MODERATE - some structural support"
                return "\n".join(f"{k}: {v}" for k, v in scoped.items())
            except Exception:
                pass  # fall through to the global reading

        # GLOBAL path (no resolvable topic): whole-complex signal reading.
        if signal == "uniform":
            f_E = np.ones(nE, dtype=np.float64)
        else:
            try:
                f_E = np.array([float(x) for x in signal.split(",")], dtype=np.float64)
            except Exception:
                f_E = np.ones(nE, dtype=np.float64)

        # Void check
        result = {}
        try:
            dipole = rex.face_void_dipole(f_E)
            result["void_affinity"] = round(float(dipole.get("void_affinity", 0)), 4)
            result["face_affinity"] = round(float(dipole.get("face_affinity", 0)), 4)
            result["dipole_ratio"] = round(float(dipole.get("dipole_ratio", 0)), 4)
        except Exception:
            result["void_affinity"] = None

        # Coherence
        try:
            kappa = rex.coherence
            result["kappa_mean"] = round(float(kappa.mean()), 4)
            result["kappa_min"] = round(float(kappa.min()), 4)
        except Exception:
            pass

        # Chain condition
        result["chain_valid"] = rex.chain_valid

        # Void count
        try:
            vc = rex.void_complex
            result["n_voids"] = vc.get("n_voids", 0)
            result["n_potential"] = vc.get("n_potential", 0)
        except Exception:
            pass

        # Confidence flag
        va = result.get("void_affinity")
        km = result.get("kappa_mean")
        if va is not None and va > 0.5:
            result["confidence"] = "LOW - high void affinity, structural gaps present"
        elif km is not None and km < 0.3:
            result["confidence"] = "LOW - low coherence, edge and vertex structure disagree"
        elif va is not None and va < 0.2 and km is not None and km > 0.7:
            result["confidence"] = "HIGH - strong structural support"
        else:
            result["confidence"] = "MODERATE - some structural support"

        lines = [f"{k}: {v}" for k, v in result.items()]
        return "\n".join(lines)




class AnalyzeInput(BaseModel):
    """Input for structural analysis."""
    aspect: str = Field(
        default="summary",
        description="What to analyze: 'summary', 'topology', 'channels', 'hodge', 'voids'"
    )


class RexAnalyzeTool(BaseTool):
    """Get structural analysis of the relational complex.

    Returns mathematical properties: Betti numbers, structural character
    (which channels dominate), Hodge decomposition (gradient/curl/harmonic),
    void complex (structural gaps), and coherence.
    """

    name: str = "rex_analyze"
    description: str = (
        "Analyze the mathematical structure. Returns topology, channel "
        "decomposition, Hodge fractions, and void complex. Use to understand "
        "the structural properties of the data."
    )
    args_schema: Type[BaseModel] = AnalyzeInput

    rex: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, rex, **kwargs):
        _require_langchain()
        super().__init__(rex=rex, **kwargs)

    def _run(self, aspect: str = "summary") -> str:
        rex = self.rex
        lines = []

        if aspect in ("summary", "topology"):
            lines.append(f"Complex: {rex.nV}V {rex.nE}E {rex.nF}F")
            lines.append(f"Betti: β₀={rex.betti[0]} β₁={rex.betti[1]} β₂={rex.betti[2]}")
            lines.append(f"Euler: {rex.euler_characteristic}")
            lines.append(f"Chain valid: {rex.chain_valid}")

        if aspect in ("summary", "channels"):
            try:
                chi = rex.structural_character
                means = chi.mean(axis=0)
                names = ["T(Hodge)", "G(Overlap)", "F(Frustration)", "C(Copath)"]
                for i in range(min(len(means), 4)):
                    lines.append(f"{names[i]}: {means[i]:.3f}")
            except Exception:
                pass
            try:
                kappa = rex.coherence
                lines.append(f"Coherence κ: {kappa.mean():.4f} (range {kappa.min():.3f}-{kappa.max():.3f})")
            except Exception:
                pass

        if aspect in ("summary", "hodge"):
            try:
                flow = np.ones(rex.nE, dtype=np.float64)
                h = rex.hodge_full(flow)
                lines.append(f"Hodge: {h['pct_grad']:.1%} gradient, {h['pct_curl']:.1%} curl, {h['pct_harm']:.1%} harmonic")
            except Exception:
                pass

        if aspect in ("summary", "voids"):
            try:
                vc = rex.void_complex
                nv = vc.get("n_voids", 0)
                np_ = vc.get("n_potential", 0)
                lines.append(f"Voids: {nv}/{np_} ({nv/np_:.0%} unrealized)" if np_ > 0 else "Voids: none")
            except Exception:
                pass

        return "\n".join(lines) if lines else "No data available."




class HodgeInput(BaseModel):
    """Input for Hodge decomposition."""
    signal: str = Field(
        default="uniform",
        description="Edge signal: 'uniform' or comma-separated floats"
    )


class RexHodgeTool(BaseTool):
    """Decompose a signal into gradient, curl, and harmonic components.

    Gradient = explainable from individual nodes.
    Curl = circulating through triangles.
    Harmonic = topological, persists through cycles.

    For an LLM agent: gradient-dominated answers come from local context,
    curl-dominated answers require relational reasoning, harmonic-dominated
    answers depend on global structure.
    """

    name: str = "rex_hodge"
    description: str = (
        "Decompose a signal into gradient (local), curl (relational), and "
        "harmonic (global) components. Tells you WHERE the information lives."
    )
    args_schema: Type[BaseModel] = HodgeInput

    rex: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, rex, **kwargs):
        _require_langchain()
        super().__init__(rex=rex, **kwargs)

    def _run(self, signal: str = "uniform") -> str:
        rex = self.rex
        if signal == "uniform":
            f_E = np.ones(rex.nE, dtype=np.float64)
        else:
            try:
                f_E = np.array([float(x) for x in signal.split(",")], dtype=np.float64)
            except Exception:
                f_E = np.ones(rex.nE, dtype=np.float64)

        try:
            h = rex.hodge_full(f_E)
            return (
                f"Gradient: {h['pct_grad']:.1%} (local/node-explainable)\n"
                f"Curl: {h['pct_curl']:.1%} (relational/face-circulating)\n"
                f"Harmonic: {h['pct_harm']:.1%} (global/topological)\n"
                f"Orthogonal: {h['orthogonality'].get('orthogonal', False)}"
            )
        except Exception as e:
            return f"Hodge decomposition failed: {e}"




class ExplainInput(BaseModel):
    """Input for cell explanation."""
    dim: int = Field(description="Dimension: 0 for vertex, 1 for edge")
    idx: int = Field(description="Cell index")


class RexExplainTool(BaseTool):
    """Get a full structural diagnostic for a single vertex or edge."""

    name: str = "rex_explain"
    description: str = (
        "Explain a specific vertex (dim=0) or edge (dim=1) by index. "
        "Returns structural character, energy, neighborhood, and role."
    )
    args_schema: Type[BaseModel] = ExplainInput

    rex: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, rex, **kwargs):
        _require_langchain()
        super().__init__(rex=rex, **kwargs)

    def _run(self, dim: int = 0, idx: int = 0) -> str:
        try:
            result = self.rex.explain(dim, idx)
            lines = [f"{k}: {v}" for k, v in result.items() if not isinstance(v, np.ndarray)]
            return "\n".join(lines) if lines else str(result)
        except Exception as e:
            return f"Explain failed: {e}"




def get_rex_tools(rex) -> list:
    """Return all RexGraph LangChain tools for a given complex.

    Usage:

        tools = get_rex_tools(rex)
        agent = create_tool_calling_agent(llm, tools, prompt)
    """
    _require_langchain()
    return [
        RexConfidenceTool(rex),
        RexAnalyzeTool(rex),
        RexHodgeTool(rex),
        RexExplainTool(rex),
    ]
