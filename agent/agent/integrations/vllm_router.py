"""
vLLM prompt router: route prompts using structural character.

Instead of using another LLM to decide which model handles a prompt,
use math. Build a rex from the prompt's token relationships, compute
its structural character, and route based on which channel dominates.

T-dominant prompt -> logical reasoning model
G-dominant prompt -> associative/creative model
F-dominant prompt -> contradiction-handling model
C-dominant prompt -> multi-hop reasoning model

High void affinity -> refuse or flag uncertainty

Usage:

    from rexgraph.integrations.vllm_router import RexRouter

    router = RexRouter(models={
        "reasoning": "mistralai/Mistral-7B-v0.1",
        "creative": "meta-llama/Llama-3-8B",
        "analytical": "Qwen/Qwen2-7B",
    })

    model, confidence = router.route("Explain why P implies Q in formal logic")
    # -> ("reasoning", {"confidence": "HIGH", "dominant": "T", ...})

Requirements: pip install rexgraph
Note: Does NOT require vllm itself: the router just picks a model name.
      The caller handles inference with whatever serving framework they use.
"""

from __future__ import annotations

import numpy as np

from agent.metrics import coherence_kappa


def _tokenize_simple(text: str) -> list[str]:
    """Simple whitespace tokenizer. Production would use a real tokenizer."""
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def _build_prompt_rex(tokens: list[str], window: int = 3):
    """Build a small relational complex from prompt tokens.

    Edges = co-occurrence within a sliding window.
    Types = adjacent (bigram) vs skip (within window, not adjacent).
    """
    try:
        from rexgraph.graph import RexGraph
    except ImportError as exc:
        raise ImportError("rexgraph is required. Install with: pip install rexgraph") from exc

    vocab = sorted(set(tokens))
    if len(vocab) < 3:
        return None, vocab

    w2i = {w: i for i, w in enumerate(vocab)}

    edge_set = {}  # (i,j) -> {"weight": float, "type": int}
    for pos in range(len(tokens)):
        wi = w2i[tokens[pos]]
        for offset in range(1, window + 1):
            if pos + offset >= len(tokens):
                break
            wj = w2i[tokens[pos + offset]]
            if wi == wj:
                continue
            key = (min(wi, wj), max(wi, wj))
            etype = 0 if offset == 1 else 1  # 0=adjacent, 1=skip
            if key not in edge_set:
                edge_set[key] = {"weight": 0.0, "type": etype}
            edge_set[key]["weight"] += 1.0

    if len(edge_set) < 3:
        return None, vocab

    edges = sorted(edge_set.keys())
    sources = np.array([e[0] for e in edges], dtype=np.int32)
    targets = np.array([e[1] for e in edges], dtype=np.int32)
    weights = np.array([edge_set[e]["weight"] for e in edges], dtype=np.float64)
    types = np.array([edge_set[e]["type"] for e in edges], dtype=np.int32)

    rex = RexGraph(sources=sources, targets=targets, w_E=weights)

    # One rule, the canonical one. This branched on the number of edge types, so a
    # single-type prompt got its whole cycle basis filled and a multi-type one got a
    # type filter: two different complexes for the same shape of input.
    from agent.auto import attach_faces
    rex = attach_faces(rex, type_labels=types)

    return rex, vocab


class RexRouter:
    """Route prompts to models using structural character.

    Maps the four RCF channels to model capabilities:
        T (Hodge/topology) -> logical structure -> reasoning model
        G (Overlap/geometry) -> associative context -> creative model
        F (Frustration) -> contradiction, tension -> analytical model
        C (Copath) -> higher-order structure -> multi-hop model

    Falls back to a default model when confidence is low.
    """

    # Default channel -> capability mapping
    CHANNEL_MAP = {
        0: "reasoning",    # T: topological, logical
        1: "creative",     # G: geometric, associative
        2: "analytical",   # F: frustration, contradiction
        3: "reasoning",    # C: copath, higher-order (defaults to reasoning)
    }

    def __init__(
        self,
        models: dict[str, str],
        default: str = "reasoning",
        void_threshold: float = 0.5,
        channel_map: dict[int, str] | None = None,
    ):
        """
        Parameters
        ----------
        models : dict mapping capability name -> model identifier
            E.g., {"reasoning": "mistral-7b", "creative": "llama-3-8b"}
        default : str
            Capability to use when confidence is low or no clear dominant channel.
        void_threshold : float
            If void_affinity exceeds this, flag as low confidence.
        channel_map : dict, optional
            Override the default channel -> capability mapping.
        """
        self.models = models
        self.default = default
        self.void_threshold = void_threshold
        # Normalise channel_map keys to int: a config loaded from JSON/YAML
        # turns integer keys into strings, which would silently miss the
        # `channel_map.get(dominant_idx, ...)` lookup and route everything
        # to the default.
        raw_map = channel_map or self.CHANNEL_MAP
        self.channel_map = {}
        for k, v in raw_map.items():
            try:
                self.channel_map[int(k)] = v
            except (TypeError, ValueError):
                self.channel_map[k] = v

    def route(self, text: str, window: int = 3) -> tuple[str, dict]:
        """Route a prompt to the best model.

        Returns (model_identifier, diagnostics_dict).
        """
        tokens = _tokenize_simple(text)

        if len(tokens) < 3:
            return self.models.get(self.default, self.default), {
                "confidence": "LOW",
                "reason": "Prompt too short for structural analysis",
                "routed_to": self.default,
            }

        rex, vocab = _build_prompt_rex(tokens, window)

        if rex is None:
            return self.models.get(self.default, self.default), {
                "confidence": "LOW",
                "reason": "Not enough structure to build relational complex",
                "routed_to": self.default,
            }

        diagnostics = {
            "nV": rex.nV,
            "nE": rex.nE,
            "nF": rex.nF,
            "betti": list(rex.betti),
        }

        # Structural character -> dominant channel
        try:
            chi = rex.structural_character
            mean_chi = chi.mean(axis=0)
            n_channels = min(len(mean_chi), 4)
            channel_names = ["T", "G", "F", "C"]

            for i in range(n_channels):
                diagnostics[f"chi_{channel_names[i]}"] = round(float(mean_chi[i]), 4)

            dominant_idx = int(np.argmax(mean_chi[:n_channels]))
            dominant_name = channel_names[dominant_idx]
            diagnostics["dominant_channel"] = dominant_name
            diagnostics["dominant_fraction"] = round(float(mean_chi[dominant_idx]), 4)
        except Exception:
            dominant_idx = 0
            dominant_name = "T"
            diagnostics["dominant_channel"] = "T"
            diagnostics["note"] = "Character computation failed, defaulting to T"

        # Void check -> confidence
        try:
            flow = np.ones(rex.nE, dtype=np.float64)
            dipole = rex.face_void_dipole(flow)
            va = float(dipole.get("void_affinity", 0))
            diagnostics["void_affinity"] = round(va, 4)
            diagnostics["face_affinity"] = round(float(dipole.get("face_affinity", 0)), 4)
        except Exception:
            va = 0.0

        # Coherence
        try:
            kappa = coherence_kappa(rex)
            diagnostics["kappa_mean"] = round(float(kappa.mean()), 4)
        except Exception:
            pass

        # Hodge decomposition of the prompt signal
        try:
            flow = np.ones(rex.nE, dtype=np.float64)
            h = rex.hodge_full(flow)
            diagnostics["hodge_gradient"] = round(h["pct_grad"], 3)
            diagnostics["hodge_curl"] = round(h["pct_curl"], 3)
            diagnostics["hodge_harmonic"] = round(h["pct_harm"], 3)
        except Exception:
            pass

        # Route decision
        if va > self.void_threshold:
            capability = self.default
            diagnostics["confidence"] = "LOW"
            diagnostics["reason"] = f"High void affinity ({va:.2f}), structural gaps present"
        else:
            capability = self.channel_map.get(dominant_idx, self.default)
            if diagnostics.get("dominant_fraction", 0) > 0.4:
                diagnostics["confidence"] = "HIGH"
                diagnostics["reason"] = f"{dominant_name}-dominant ({diagnostics.get('dominant_fraction', 0):.0%})"
            else:
                diagnostics["confidence"] = "MODERATE"
                diagnostics["reason"] = f"No strong dominant channel, slight {dominant_name} preference"

        model = self.models.get(capability, self.models.get(self.default, self.default))
        diagnostics["routed_to"] = capability
        diagnostics["model"] = model

        return model, diagnostics

    def route_batch(self, texts: list[str]) -> list[tuple[str, dict]]:
        """Route multiple prompts."""
        return [self.route(text) for text in texts]
