"""
LangGraph + RexGraph - agent state machines as relational complexes.

The insight: a LangGraph graph IS a relational complex. Nodes are states,
edges are transitions, and the math tells you everything about the agent's
behavior that the agent itself can't see.

- Hodge decomposition of the execution path: is the agent making progress
  (gradient), going in circles (curl), or stuck in a structural loop (harmonic)?
- Void complex: which state transitions COULD exist but don't? These are
  the agent's blind spots.
- Structural character: is each transition topological (following the graph
  structure), geometric (taking shortcuts), or frustrated (fighting conflicts)?
- Coherence: do the agent's local decisions agree with the global structure?

Usage:

    from agent.integrations.langgraph_rex import RexStateGraph

    rsg = RexStateGraph()

    # Register states and transitions as the agent runs
    rsg.add_state("retrieve", metadata={"tool": "search"})
    rsg.add_state("reason", metadata={"tool": "llm"})
    rsg.add_transition("retrieve", "reason", weight=0.9, sign=+1)

    # Build the rex and analyze
    rex = rsg.build()
    analysis = rsg.analyze()

    # Real-time checks during execution
    confidence = rsg.transition_confidence("reason", "answer")
    should_continue = rsg.should_continue()  # False if harmonic-dominated (stuck in loops)
    cycle_report = rsg.detect_cycles()

    # Post-execution analysis
    path_hodge = rsg.decompose_path(["start", "retrieve", "reason", "retrieve", "reason", "answer"])
    # -> gradient: 60% (making progress), curl: 35% (retrieve<->reason loop), harmonic: 5%

Requirements: pip install rexgraph-agent[langgraph]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.metrics import coherence_kappa

try:
    from rexgraph.graph import RexGraph
    _HAS_REXGRAPH = True
except ImportError:
    _HAS_REXGRAPH = False


class RexStateGraph:
    """Model an agent's state machine as a relational complex.

    Build incrementally as the agent runs. Query the math at any point
    for structural diagnostics about the agent's behavior.
    """

    def __init__(self):
        self._states: dict[str, dict] = {}       # name -> metadata
        self._transitions: list[dict] = []        # list of {src, tgt, weight, sign, type}
        self._state_order: list[str] = []         # insertion order
        self._execution_log: list[str] = []       # sequence of state visits
        self._rex: Any | None = None           # cached RexGraph
        self._dirty = True                        # needs rebuild

    # Building

    def add_state(self, name: str, metadata: dict | None = None):
        """Register a state (node) in the graph."""
        if name not in self._states:
            self._states[name] = metadata or {}
            self._state_order.append(name)
            self._dirty = True

    def add_transition(
        self,
        src: str,
        tgt: str,
        weight: float = 1.0,
        sign: float = 1.0,
        transition_type: str = "default",
    ):
        """Register a transition (edge) between states."""
        self.add_state(src)
        self.add_state(tgt)
        self._transitions.append({
            "src": src, "tgt": tgt,
            "weight": abs(weight), "sign": np.sign(sign) or 1.0,
            "type": transition_type,
        })
        self._dirty = True

    def log_visit(self, state: str):
        """Record a state visit during execution."""
        self.add_state(state)
        self._execution_log.append(state)

    # Rex Construction

    def build(self) -> Any:
        """Build (or rebuild) the RexGraph from current states and transitions.

        Returns the RexGraph. Also accessible via .rex property.
        """
        if not _HAS_REXGRAPH:
            raise ImportError("rexgraph is required. Install with: pip install rexgraph")

        if not self._transitions:
            raise ValueError("No transitions registered. Add transitions first.")

        name_to_idx = {name: i for i, name in enumerate(self._state_order)}
        len(self._state_order)

        sources = np.array([name_to_idx[t["src"]] for t in self._transitions], dtype=np.int32)
        targets = np.array([name_to_idx[t["tgt"]] for t in self._transitions], dtype=np.int32)
        weights = np.array([t["weight"] for t in self._transitions], dtype=np.float64)
        signs = np.array([t["sign"] for t in self._transitions], dtype=np.float64)

        w_E = weights if not np.allclose(weights, 1.0) else None
        signs_arg = signs if np.any(signs < 0) else None

        rex = RexGraph(sources=sources, targets=targets, w_E=w_E, signs=signs_arg)

        # Type-based face selection if we have typed transitions
        types = [t["type"] for t in self._transitions]
        unique_types = sorted(set(types))
        if len(unique_types) > 1:
            type_map = {t: i for i, t in enumerate(unique_types)}
            type_labels = np.array([type_map[t] for t in types], dtype=np.int32)
            from agent.auto import attach_faces
            rex = attach_faces(rex, type_labels=type_labels)
        else:
            rex = rex.promote()

        self._rex = rex
        self._dirty = False
        return rex

    @property
    def rex(self):
        if self._dirty or self._rex is None:
            return self.build()
        return self._rex

    # Analysis

    def analyze(self) -> dict:
        """Full structural analysis of the state graph."""
        from rexgraph.analysis import analyze
        return analyze(self.rex, vertex_labels=self._state_order)

    def transition_confidence(self, src: str, tgt: str) -> dict:
        """Confidence assessment for a specific transition.

        Uses the void complex and interfacing vector to determine
        whether this transition has structural support.

        Returns dict with: confidence flag, void_affinity, structural reasons.
        """
        rex = self.rex
        name_to_idx = {name: i for i, name in enumerate(self._state_order)}

        if src not in name_to_idx or tgt not in name_to_idx:
            return {"confidence": "UNKNOWN", "reason": "State not in graph"}

        src_idx = name_to_idx[src]
        tgt_idx = name_to_idx[tgt]

        result = {"src": src, "tgt": tgt}

        # Check if this transition exists as an edge
        edge_exists = False
        for t in self._transitions:
            if t["src"] == src and t["tgt"] == tgt:
                edge_exists = True
                break
        result["edge_exists"] = edge_exists

        # Void check: is this transition in a structural gap?
        try:
            signal = np.zeros(rex.nE, dtype=np.float64)
            # Edges incident to the source vertex. `star_of_vertex` answers this
            # directly; reading a row of a materialized dense B1 built an nV x nE
            # array to look at one row of it.
            _, e_mask, _ = rex.star_of_vertex(int(src_idx))
            signal[np.asarray(e_mask, dtype=bool)] = 1.0

            dipole = rex.face_void_dipole(signal)
            result["void_affinity"] = round(float(dipole.get("void_affinity", 0)), 4)
            result["face_affinity"] = round(float(dipole.get("face_affinity", 0)), 4)
        except Exception:
            pass

        # Coherence at source vertex
        try:
            kappa = coherence_kappa(rex)
            result["src_coherence"] = round(float(kappa[src_idx]), 4)
            result["tgt_coherence"] = round(float(kappa[tgt_idx]), 4)
        except Exception:
            pass

        # Confidence flag
        va = result.get("void_affinity", 0)
        sk = result.get("src_coherence", 1)
        if not edge_exists:
            result["confidence"] = "NONE - transition doesn't exist in the graph"
        elif va > 0.5:
            result["confidence"] = "LOW - high void affinity around source state"
        elif sk < 0.3:
            result["confidence"] = "LOW - source state has low structural coherence"
        elif va < 0.2 and sk > 0.7:
            result["confidence"] = "HIGH - strong structural support"
        else:
            result["confidence"] = "MODERATE"

        return result

    def should_continue(self, harmonic_threshold: float = 0.4) -> dict:
        """Should the agent continue executing?

        If the execution path is harmonic-dominated (stuck in topological
        loops that can't be broken by local decisions), the agent should stop.

        Returns dict with recommendation and Hodge fractions.
        """
        rex = self.rex
        result = {"recommendation": "continue"}

        try:
            flow = np.ones(rex.nE, dtype=np.float64)
            h = rex.hodge_full(flow)
            result["pct_gradient"] = round(h["pct_grad"], 3)
            result["pct_curl"] = round(h["pct_curl"], 3)
            result["pct_harmonic"] = round(h["pct_harm"], 3)

            if h["pct_harm"] > harmonic_threshold:
                result["recommendation"] = "stop"
                result["reason"] = (
                    f"Harmonic component is {h['pct_harm']:.0%}. The agent is stuck in "
                    f"topological loops that can't be resolved by local transitions."
                )
            elif h["pct_curl"] > 0.6:
                result["recommendation"] = "caution"
                result["reason"] = (
                    f"Curl component is {h['pct_curl']:.0%}. The agent is circulating "
                    f"through face cycles. Consider breaking the loop."
                )
            else:
                result["reason"] = (
                    f"Gradient-dominated ({h['pct_grad']:.0%}). Making progress."
                )
        except Exception as e:
            result["reason"] = f"Analysis failed: {e}"

        return result

    def detect_cycles(self) -> dict:
        """Detect structural cycles in the state graph.

        Uses Betti numbers: β₁ > 0 means independent cycles exist.
        Returns the number of independent cycles and their edges.
        """
        rex = self.rex
        betti = rex.betti

        result = {
            "n_independent_cycles": betti[1],
            "has_cycles": betti[1] > 0,
        }

        if betti[1] > 0:
            # Which edges carry the independent cycles is structural, so it is read from
            # the harmonic basis rather than from the harmonic part of some chosen flow.
            #
            # This previously projected the all-ones flow and kept edges where
            # |harm| > 1e-6. That answers a different question, the harmonic content of
            # that one flow, and it fails outright whenever the chosen flow happens to be
            # orthogonal to the harmonic space. A plain 4-cycle with two edges reversed is
            # enough: beta_1 is 1, the all-ones harmonic part is 2.22e-16, and the
            # threshold returns no edges at all while the method still reports
            # has_cycles True. The magnitude is also frame dependent, so no threshold on
            # it is the structural answer.
            #
            # harmonic_basis spans ker(B1) cap ker(B2^T), which is exactly what beta_1
            # counts, so its support is the edge set this method claims to return. The
            # cycle basis would be wrong here: it spans ker(B1) alone, so a filled cycle
            # would still appear even though it is no longer a hole.
            try:
                from rexgraph.harmonic_sparse import harmonic_basis

                H = harmonic_basis(rex).tocoo()
                cycle_edges = np.unique(np.asarray(H.row, dtype=np.int64))
                result["cycle_edge_indices"] = cycle_edges.tolist()
                result["cycle_edge_labels"] = [
                    f"{self._state_order[self._transitions[e]['src' if e < len(self._transitions) else 0]]} -> ..."
                    for e in cycle_edges[:10]
                ]
            except Exception:
                pass

        return result

    def decompose_path(self, path: list[str]) -> dict:
        """Hodge-decompose a specific execution path.

        Given a sequence of state visits, construct the path signal
        on the edge space and decompose it.

        Returns:
            pct_gradient: fraction that is goal-directed progress
            pct_curl: fraction that is circular (revisiting)
            pct_harmonic: fraction that is structurally stuck
        """
        rex = self.rex
        name_to_idx = {name: i for i, name in enumerate(self._state_order)}

        # Build edge signal from path: activate each edge traversed
        signal = np.zeros(rex.nE, dtype=np.float64)

        # An edge joining two states is in both their stars, so the transition is
        # the intersection. The scan it replaces was O(len(path) * nE) over a dense
        # nV x nE matrix to find one edge at a time.
        stars: dict[int, np.ndarray] = {}

        def _star(v: int) -> np.ndarray:
            if v not in stars:
                stars[v] = np.asarray(rex.star_of_vertex(int(v))[1], dtype=bool)
            return stars[v]

        for i in range(len(path) - 1):
            src = name_to_idx.get(path[i])
            tgt = name_to_idx.get(path[i + 1])
            if src is None or tgt is None:
                continue
            both = np.flatnonzero(_star(src) & _star(tgt))
            if both.size:
                signal[both[0]] += 1.0     # one traversal, as the scan did

        if np.allclose(signal, 0):
            return {"error": "No edges found for path", "path": path}

        try:
            h = rex.hodge_full(signal)
            return {
                "path": path,
                "path_length": len(path),
                "edges_traversed": int(np.sum(signal > 0)),
                "pct_gradient": round(h["pct_grad"], 3),
                "pct_curl": round(h["pct_curl"], 3),
                "pct_harmonic": round(h["pct_harm"], 3),
                "interpretation": _interpret_path_hodge(h),
            }
        except Exception as e:
            return {"error": str(e)}

    def channel_profile(self) -> dict:
        """Four-channel decomposition of the state graph.

        T (Hodge): structural transitions, the agent follows the graph
        G (Overlap): geometric shortcuts, transitions sharing context
        F (Frustration): conflicted transitions, sign disagreements
        C (Copath): higher-order structure, meta-transitions
        """
        rex = self.rex
        try:
            chi = rex.structural_character
            means = chi.mean(axis=0)
            names = ["T(structure)", "G(context)", "F(conflict)", "C(meta)"]
            result = {}
            for i in range(min(len(means), 4)):
                result[names[i]] = round(float(means[i]), 4)
            return result
        except Exception as e:
            return {"error": str(e)}

    # LangGraph Integration

    def as_langgraph_checker(self):
        """Return a callable that can be used as a LangGraph conditional edge.

        Usage in LangGraph:

            from langgraph.graph import StateGraph
            rsg = RexStateGraph()
            # ... add states and transitions ...

            def should_continue(state):
                check = rsg.should_continue()
                return "continue" if check["recommendation"] == "continue" else "end"

            graph.add_conditional_edges("agent", should_continue)
        """
        def checker(state=None):
            result = self.should_continue()
            return result["recommendation"]
        return checker

    def as_langgraph_confidence_gate(self, src_state: str, tgt_state: str):
        """Return a callable confidence gate for a specific transition.

        Usage:

            gate = rsg.as_langgraph_confidence_gate("reason", "answer")
            # Returns "proceed" or "reconsider" based on structural confidence
        """
        def gate(state=None):
            result = self.transition_confidence(src_state, tgt_state)
            conf = result.get("confidence", "")
            if conf.startswith("HIGH"):
                return "proceed"
            elif conf.startswith("LOW") or conf.startswith("NONE"):
                return "reconsider"
            else:
                return "proceed"
        return gate


def _interpret_path_hodge(h: dict) -> str:
    """Human-readable interpretation of a path's Hodge decomposition."""
    g, c, hm = h.get("pct_grad", 0), h.get("pct_curl", 0), h.get("pct_harm", 0)
    parts = []
    if g > 0.6:
        parts.append("Goal-directed: mostly making forward progress")
    elif g > 0.3:
        parts.append("Partially goal-directed")

    if c > 0.4:
        parts.append("Circular: significant revisiting of states")
    if hm > 0.3:
        parts.append("Stuck: trapped in structural loops that local decisions can't break")

    if not parts:
        parts.append("Mixed: no dominant pattern")
    return ". ".join(parts) + "."
