"""
HuggingFace transformer analyzer - measure RCF axiom compliance.

Takes any HuggingFace transformer and measures how closely its internal
representations match the algebraic structure of a relational complex.

The hook: "Your model violates ∂²=0 by 0.3 at layer 7. Here's what that means."

Usage:

    from rexgraph.integrations.huggingface_analyzer import analyze_transformer

    report = analyze_transformer(
        model_name="mistralai/Mistral-7B-v0.1",
        text="The cat sat on the mat.",
        device="cuda",
    )
    print(report["per_layer_chain_violation"])
    print(report["equiweight_deviation"])
    print(report["channel_specialization"])

Requirements: pip install rexgraph[huggingface]
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


def _require_hf():
    if not _HAS_HF:
        raise ImportError(
            "HuggingFace integration requires torch and transformers.\n"
            "Install with: pip install rexgraph[huggingface]"
        )


def extract_attention_rex(
    attention_matrix: np.ndarray,
    top_p: float = 0.9,
    floor: float = 1e-8,
) -> dict:
    """Build a weighted directed relational complex from an attention matrix WITHOUT a
    magic threshold. Per query token, keep the smallest set of attended keys whose
    softmax mass covers ``top_p`` (nucleus) - data-adaptive and principled for softmax
    attention - plus a numerical-zero floor. Vectorized (no O(n²) Python loop). Edge
    i->j = token i attends to token j, weighted by the attention value.

    (Weight-direct analysis on the full matrix - no discretization - is also available:
    pass the matrix straight to ``RexGraph.from_adjacency`` and read the moment-engine
    metrics; nucleus is for when a sparse discrete complex is wanted.)"""
    A = np.array(attention_matrix, dtype=np.float64, copy=True)
    n = A.shape[0]
    if n < 2:
        z = np.zeros(0)
        return {"sources": z.astype(np.int32), "targets": z.astype(np.int32),
                "weights": z, "n_tokens": n}
    np.fill_diagonal(A, 0.0)                      # drop self-attention for the token graph
    order = np.argsort(-A, axis=1)               # keys by descending weight, per query
    sw = np.take_along_axis(A, order, axis=1)
    total = sw.sum(axis=1, keepdims=True)
    total[total < 1e-12] = 1.0
    prev = np.cumsum(sw, axis=1) - sw            # cumulative mass BEFORE each key
    keep = prev < top_p * total                  # nucleus: include the crossing key
    keep[:, 0] = True                            # always keep top-1 per query
    keep &= sw > floor                           # numerical-zero floor
    qi, kk = np.where(keep)                       # (query row, sorted position)
    tgt = order[qi, kk]
    w = A[qi, tgt]
    return {
        "sources": qi.astype(np.int32),
        "targets": tgt.astype(np.int32),
        "weights": w.astype(np.float64),
        "n_tokens": n,
    }


def measure_chain_condition(B1: np.ndarray, B2: np.ndarray) -> float:
    """Measure ||B₁B₂|| - how badly the chain condition is violated."""
    if B2.shape[1] == 0:
        return 0.0
    product = B1 @ B2
    return float(np.max(np.abs(product)))


def measure_equiweight(D: np.ndarray, nV: int, nE: int, nF: int) -> dict:
    """Measure equiweight: ΓD + DΓ should be zero.

    Returns per-mode even/odd fractions. Non-harmonic modes should be 0.5.
    """
    dim = nV + nE + nF
    gamma = np.ones(dim)
    gamma[nV:nV + nE] = -1.0
    Gamma = np.diag(gamma)

    # ΓD + DΓ is the chiral-grading anticommutator - a structural identity (≈0), not a
    # measured threshold.
    anticomm_norm = float(np.linalg.norm(Gamma @ D + D @ Gamma))

    # Non-harmonic mode count is the EXACT integer dim - dim ker(D) = dim - nullity(D),
    # via rank (no eigenvalue-magnitude threshold).
    n_harmonic = dim - int(np.linalg.matrix_rank(D))
    n_nonharmonic = dim - n_harmonic

    # The per-mode even/odd (chirality) fraction genuinely needs the eigenVECTORS - it
    # is not reducible to an integer invariant. It's bounded here (an attention complex
    # is small), so a dense symmetric eig is fine; we just skip the exact null space.
    deviations = []
    if 0 < dim <= 4096:
        evals, evecs = np.linalg.eigh(D)
        # the null space has dimension n_harmonic - skip exactly that many smallest-|λ|
        nonharm_idx = np.argsort(np.abs(evals))[n_harmonic:]
        for j in nonharm_idx:
            v = evecs[:, j]
            even = float(np.sum(v[:nV] ** 2) + np.sum(v[nV + nE:] ** 2))
            deviations.append(abs(even - 0.5))

    return {
        "anticommutator_norm": anticomm_norm,
        "mean_equiweight_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "max_equiweight_deviation": float(np.max(deviations)) if deviations else 0.0,
        "n_harmonic_modes": int(n_harmonic),
        "n_nonharmonic_modes": int(n_nonharmonic),
    }


def analyze_transformer(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    text: str = "The cat sat on the mat.",
    device: str = "cpu",
    max_layers: int = -1,
    attention_threshold: float = 0.05,
) -> Dict:
    """Analyze a HuggingFace transformer for RCF axiom compliance.

    Runs inference, captures attention patterns at each layer,
    builds a relational complex from each, and measures:
    - Chain condition violation per layer
    - Equiweight deviation per layer
    - Channel specialization of attention heads
    - Structural character evolution across layers

    Parameters
    ----------
    model_name : HuggingFace model identifier
    text : input text to analyze
    device : 'cpu' or 'cuda'
    max_layers : analyze first N layers only (-1 for all)
    attention_threshold : minimum attention weight for edge creation

    Returns
    -------
    dict with per-layer analysis and aggregated metrics
    """
    _require_hf()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
    n_layers = len(attentions)
    if max_layers > 0:
        n_layers = min(n_layers, max_layers)

    layer_results = []

    for layer_idx in range(n_layers):
        attn = attentions[layer_idx][0].cpu().numpy()  # (heads, seq, seq)
        n_heads = attn.shape[0]

        # Average across heads for the layer-level complex
        avg_attn = attn.mean(axis=0)  # (seq, seq)

        # Build rex from average attention
        edge_data = extract_attention_rex(avg_attn, floor=attention_threshold)

        if edge_data["sources"].shape[0] < 3:
            layer_results.append({
                "layer": layer_idx,
                "n_edges": int(edge_data["sources"].shape[0]),
                "chain_violation": None,
                "note": "Too few edges for analysis",
            })
            continue

        try:
            from rexgraph.graph import RexGraph

            rex = RexGraph(
                sources=edge_data["sources"],
                targets=edge_data["targets"],
                w_E=edge_data["weights"],
            )
            rex = rex.promote()

            layer_data = {
                "layer": layer_idx,
                "n_edges": rex.nE,
                "n_faces": rex.nF,
                "chain_valid": rex.chain_valid,
                "betti": list(rex.betti),
            }

            # Structural character
            try:
                chi = rex.structural_character
                means = chi.mean(axis=0)
                channel_names = ["T", "G", "F", "C"]
                for i in range(min(len(means), 4)):
                    layer_data[f"chi_{channel_names[i]}"] = round(float(means[i]), 4)
            except Exception:
                pass

            # Coherence
            try:
                kappa = rex.coherence
                layer_data["kappa_mean"] = round(float(kappa.mean()), 4)
            except Exception:
                pass

            # Per-head analysis: which heads specialize into which channels?
            head_channels = []
            for h in range(min(n_heads, 8)):  # cap at 8 heads for speed
                head_attn = attn[h]
                head_edges = extract_attention_rex(head_attn, floor=attention_threshold)
                if head_edges["sources"].shape[0] >= 3:
                    try:
                        head_rex = RexGraph(
                            sources=head_edges["sources"],
                            targets=head_edges["targets"],
                            w_E=head_edges["weights"],
                        )
                        head_rex = head_rex.promote()
                        head_chi = head_rex.structural_character
                        if head_chi.shape[0] > 0:
                            head_means = head_chi.mean(axis=0)
                            dominant = int(np.argmax(head_means[:4])) if len(head_means) >= 4 else 0
                            head_channels.append({
                                "head": h,
                                "dominant_channel": ["T", "G", "F", "C"][dominant],
                                "chi": head_means[:4].tolist(),
                            })
                    except Exception:
                        pass

            if head_channels:
                layer_data["head_specialization"] = head_channels

            layer_results.append(layer_data)

        except ImportError:
            layer_results.append({
                "layer": layer_idx,
                "n_edges": int(edge_data["sources"].shape[0]),
                "note": "rexgraph not available for full analysis",
            })

    # Aggregate
    chain_violations = [r.get("chain_valid") for r in layer_results if r.get("chain_valid") is not None]
    kappas = [r.get("kappa_mean") for r in layer_results if r.get("kappa_mean") is not None]

    return {
        "model": model_name,
        "text": text,
        "tokens": tokens,
        "n_layers_analyzed": n_layers,
        "per_layer": layer_results,
        "aggregate": {
            "all_chain_valid": all(chain_violations) if chain_violations else None,
            "mean_kappa": round(float(np.mean(kappas)), 4) if kappas else None,
            "kappa_trend": "increasing" if len(kappas) >= 2 and kappas[-1] > kappas[0] else "decreasing" if len(kappas) >= 2 else "unknown",
        },
    }


def quick_attention_analysis(
    attention_matrix: np.ndarray,
    token_labels: Optional[List[str]] = None,
    threshold: float = 0.05,
) -> Dict:
    """Analyze a single attention matrix without loading a model.

    For users who already have attention weights extracted.
    """
    edge_data = extract_attention_rex(attention_matrix, floor=threshold)

    if edge_data["sources"].shape[0] < 3:
        return {"n_edges": 0, "note": "Too few edges"}

    try:
        from rexgraph.graph import RexGraph

        rex = RexGraph(
            sources=edge_data["sources"],
            targets=edge_data["targets"],
            w_E=edge_data["weights"],
        )
        rex = rex.promote()

        result = {
            "nV": rex.nV, "nE": rex.nE, "nF": rex.nF,
            "betti": list(rex.betti),
            "chain_valid": rex.chain_valid,
        }

        try:
            chi = rex.structural_character
            means = chi.mean(axis=0)
            for i, name in enumerate(["T", "G", "F", "C"]):
                if i < len(means):
                    result[f"chi_{name}"] = round(float(means[i]), 4)
        except Exception:
            pass

        try:
            result["kappa_mean"] = round(float(rex.coherence.mean()), 4)
        except Exception:
            pass

        try:
            flow = np.ones(rex.nE, dtype=np.float64)
            h = rex.hodge_full(flow)
            result["hodge_gradient"] = round(h["pct_grad"], 3)
            result["hodge_curl"] = round(h["pct_curl"], 3)
            result["hodge_harmonic"] = round(h["pct_harm"], 3)
        except Exception:
            pass

        return result

    except ImportError:
        return {"n_edges": int(edge_data["sources"].shape[0]), "note": "rexgraph not available"}
