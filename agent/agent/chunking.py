"""
agent.chunking: Hodge-based text chunking.

Splits text at gradient peaks of the Hodge decomposition.
Gradient energy concentrates at topic boundaries. Each
resulting chunk gets a local kappa (coherence) and Hodge
profile computed from the subcomplex induced by its edges.

Uses compiled kernels:
    _hodge.compute_energy_percentages
    _character.compute_chi (via RexGraph.structural_character)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agent.adapters import EdgeSpan, SentenceSpan


@dataclass
class Chunk:
    """A structurally coherent text segment."""
    idx: int
    text: str
    char_start: int
    char_end: int
    sentence_start: int
    sentence_end: int
    edge_indices: list[int] = field(default_factory=list)
    n_edges: int = 0
    kappa: float = 0.0
    hodge_gradient: float = 0.0
    hodge_curl: float = 0.0
    hodge_harmonic: float = 0.0
    dominant_channel: str = ""
    chi_mean: list[float] | None = None


def hodge_chunk(
    rex,
    edge_spans: list[EdgeSpan],
    sentence_spans: list[SentenceSpan],
    source_text: str,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 2000,
) -> list[Chunk]:
    """Split text into chunks at Hodge gradient peaks.

    The gradient component of the Hodge decomposition measures
    hierarchical flow. Its energy concentrates at topic boundaries
    where information transitions from one subject to another.
    Local maxima of per-sentence gradient energy are natural
    split points.
    """
    if not edge_spans or not sentence_spans:
        return [Chunk(
            idx=0, text=source_text,
            char_start=0, char_end=len(source_text),
            sentence_start=0, sentence_end=0,
        )]

    n_sents = len(sentence_spans)

    # Get gradient energy per edge from the Hodge decomposition
    grad_energy = _gradient_energy_per_sentence(rex, edge_spans, n_sents)

    # Find split points (local maxima of gradient energy)
    splits = _find_gradient_peaks(grad_energy, n_sents)

    # Build chunks from splits
    boundaries = [0] + splits + [n_sents]
    chunks = []
    for i in range(len(boundaries) - 1):
        s_start = boundaries[i]
        s_end = boundaries[i + 1]
        if s_start >= len(sentence_spans) or s_end <= s_start:
            continue

        c_start = sentence_spans[s_start].char_start
        c_end = sentence_spans[min(s_end - 1, len(sentence_spans) - 1)].char_end
        chunk_text = source_text[c_start:c_end]

        # Collect edges belonging to this chunk
        chunk_edges = [
            sp.edge_idx for sp in edge_spans
            if s_start <= sp.sentence_idx < s_end
        ]

        chunks.append(Chunk(
            idx=i,
            text=chunk_text,
            char_start=c_start,
            char_end=c_end,
            sentence_start=s_start,
            sentence_end=s_end,
            edge_indices=chunk_edges,
            n_edges=len(chunk_edges),
        ))

    # Merge small chunks, split large ones
    chunks = _enforce_size_limits(chunks, source_text, min_chunk_chars, max_chunk_chars)

    # Compute per-chunk structural properties
    _compute_chunk_properties(chunks, rex)

    return chunks


def _gradient_energy_per_sentence(rex, edge_spans, n_sents):
    """Aggregate gradient energy per sentence.

    Combines two boundary signals:
    1. Hodge gradient energy (topic transitions)
    2. Diffusion dissipation (structural barriers)
    """
    grad_per_sent = np.zeros(n_sents, dtype=np.float64)

    try:
        flow = np.ones(rex.nE, dtype=np.float64)
        hc = rex.hodge_full(flow)
        grad = hc.get("grad")
        if grad is None:
            return grad_per_sent
    except Exception:
        return grad_per_sent

    # Per-edge gradient energy
    grad_sq = np.asarray(grad) ** 2

    # Add diffusion dissipation as a second boundary signal. EIGEN-FREE / GPU-capable:
    # e^{-t·RL} flow via matrix-free Chebyshev on the SPARSE relational Laplacian (no
    # dense eigendecomposition of RL through spectral_bundle) - a per-chunk hot loop.
    try:
        from rexgraph import scale_propagator as _spg
        RL = rex.relational_laplacian
        if RL is None:
            RL = rex.L1_sparse
        times = np.array([0.1, 1.0, 5.0], dtype=np.float64)
        diffused = _spg.heat_trajectory(RL, flow, times)   # (T, nE), no eigensolve
        # Dissipation = how much signal is lost at each edge by t=5
        if isinstance(diffused, np.ndarray) and diffused.ndim == 2:
            dissipation = np.abs(diffused[0] - diffused[-1])
            grad_sq = grad_sq + dissipation  # combine both signals
    except Exception:
        pass  # fall back to gradient-only

    # Map to sentences
    count_per_sent = np.zeros(n_sents, dtype=np.float64)
    for sp in edge_spans:
        si = sp.sentence_idx
        if 0 <= si < n_sents and sp.edge_idx < len(grad_sq):
            grad_per_sent[si] += grad_sq[sp.edge_idx]
            count_per_sent[si] += 1

    mask = count_per_sent > 0
    grad_per_sent[mask] /= count_per_sent[mask]
    return grad_per_sent


def _find_gradient_peaks(energy, n_sents, min_gap=3):
    """Find local maxima of gradient energy.

    Only keeps peaks separated by at least min_gap sentences
    to avoid over-splitting.
    """
    if n_sents < min_gap * 2:
        return []

    # Smooth with a small window to avoid noise
    if n_sents >= 5:
        kernel = np.ones(3) / 3
        energy = np.convolve(energy, kernel, mode="same")

    # Find local maxima
    peaks = []
    for i in range(1, n_sents - 1):
        if energy[i] > energy[i - 1] and energy[i] > energy[i + 1]:
            if energy[i] > np.mean(energy) * 0.5:  # above half-mean threshold
                peaks.append((i, energy[i]))

    # Filter by minimum gap
    if not peaks:
        return []

    peaks.sort(key=lambda x: -x[1])
    selected = []
    for idx, _ in peaks:
        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(idx)

    selected.sort()
    return selected


def _enforce_size_limits(chunks, text, min_chars, max_chars):
    """Merge small chunks, split large ones."""
    result = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        # Merge small chunks with next
        while len(c.text) < min_chars and i + 1 < len(chunks):
            i += 1
            nxt = chunks[i]
            c = Chunk(
                idx=c.idx,
                text=text[c.char_start:nxt.char_end],
                char_start=c.char_start,
                char_end=nxt.char_end,
                sentence_start=c.sentence_start,
                sentence_end=nxt.sentence_end,
                edge_indices=c.edge_indices + nxt.edge_indices,
                n_edges=c.n_edges + nxt.n_edges,
            )

        # Split large chunks at midpoint
        if len(c.text) > max_chars:
            mid = c.char_start + len(c.text) // 2
            # Find nearest sentence boundary
            mid_sent = (c.sentence_start + c.sentence_end) // 2
            result.append(Chunk(
                idx=len(result), text=text[c.char_start:mid],
                char_start=c.char_start, char_end=mid,
                sentence_start=c.sentence_start, sentence_end=mid_sent,
                edge_indices=[e for e in c.edge_indices],
                n_edges=c.n_edges,
            ))
            result.append(Chunk(
                idx=len(result), text=text[mid:c.char_end],
                char_start=mid, char_end=c.char_end,
                sentence_start=mid_sent, sentence_end=c.sentence_end,
                edge_indices=[], n_edges=0,
            ))
        else:
            c.idx = len(result)
            result.append(c)
        i += 1

    return result


def _compute_chunk_properties(chunks, rex):
    """Compute kappa, Hodge profile, and dominant channel per chunk."""
    chan_names = ["T", "G", "F", "C"]

    for chunk in chunks:
        if not chunk.edge_indices:
            continue

        # Local kappa from the edges in this chunk
        try:
            kappa = rex.coherence
            if kappa is not None:
                local_kappa = np.mean([
                    kappa[e] for e in chunk.edge_indices
                    if e < len(kappa)
                ])
                chunk.kappa = float(local_kappa)
        except Exception:
            pass

        # Local Hodge profile
        try:
            flow = np.ones(rex.nE, dtype=np.float64)
            hc = rex.hodge_full(flow)
            grad = hc.get("grad")
            curl = hc.get("curl")
            harm = hc.get("harm")

            if grad is not None:
                edges = [e for e in chunk.edge_indices if e < len(grad)]
                if edges:
                    eg = float(np.sum(np.asarray(grad)[edges] ** 2))
                    ec = float(np.sum(np.asarray(curl)[edges] ** 2)) if curl is not None else 0
                    eh = float(np.sum(np.asarray(harm)[edges] ** 2)) if harm is not None else 0
                    total = eg + ec + eh
                    if total > 0:
                        chunk.hodge_gradient = eg / total
                        chunk.hodge_curl = ec / total
                        chunk.hodge_harmonic = eh / total
        except Exception:
            pass

        # Dominant channel from structural character
        try:
            chi = rex.structural_character
            if chi is not None:
                edges = [e for e in chunk.edge_indices if e < chi.shape[0]]
                if edges:
                    chi_mean = np.mean(chi[edges], axis=0)
                    chunk.chi_mean = chi_mean.tolist()
                    n_chan = min(len(chan_names), len(chi_mean))
                    chunk.dominant_channel = chan_names[int(np.argmax(chi_mean[:n_chan]))]
        except Exception:
            pass
