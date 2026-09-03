"""
agent.training: export structural analysis as training data.

Converts rexgraph structural analysis into training features
for model fine-tuning. Uses rexgraph/io/safetensors_bridge
directly for HuggingFace/PyTorch/JAX compatibility.

Usage:

    from agent.training import TrainingExporter

    # From a built corpus
    te = TrainingExporter(corpus)
    te.export_features("features.safetensors")
    te.export_training_pairs("pairs.safetensors", target="summary")

    # From files directly
    te = TrainingExporter.from_files(["doc1.pdf", "doc2.pdf"])
    te.export_features("features.safetensors")

    # For HuggingFace datasets
    dataset = te.to_hf_dataset()
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example with structural features."""
    text: str
    doc_id: str
    chunk_idx: int = 0
    kappa: float = 0.0
    chi: list = field(default_factory=list)      # [T, G, F, C]
    hodge: list = field(default_factory=list)     # [grad, curl, harm]
    channel: str = ""
    betti: list = field(default_factory=list)
    n_voids: int = 0
    persistence_entropy: float = 0.0
    target: str = ""  # for supervised pairs


class TrainingExporter:
    """Export structural analysis as training data.

    Each chunk in the corpus becomes a training example with
    structural features computed from the relational complex.
    """

    def __init__(self, corpus=None):
        self.corpus = corpus
        self.examples: list[TrainingExample] = []
        if corpus and corpus._built:
            self._extract_from_corpus()

    @classmethod
    def from_files(cls, paths: list[str], **kwargs) -> TrainingExporter:
        """Build from files directly."""
        from agent.corpus import CorpusBuilder
        corpus = CorpusBuilder()
        for p in paths:
            corpus.add_document(source=p, **kwargs)
        corpus.build(depth="standard")
        return cls(corpus)

    @classmethod
    def from_texts(cls, texts: list[str], doc_ids: list[str] = None) -> TrainingExporter:
        """Build from text strings."""
        from agent.corpus import CorpusBuilder
        corpus = CorpusBuilder()
        for i, text in enumerate(texts):
            did = doc_ids[i] if doc_ids else "doc_%d" % i
            corpus.add_text(text, doc_id=did)
        corpus.build(depth="standard")
        return cls(corpus)

    def _extract_from_corpus(self):
        """Extract training examples from every document chunk."""
        from agent.adapters.text import TextAdapter
        from agent.chunking import hodge_chunk
        from rexgraph.graph import RexGraph

        ta = TextAdapter()

        for doc in self.corpus.documents:
            if doc.rex is None:
                continue

            rex = doc.rex
            source_text = doc.text or ''
            if not source_text:
                continue

            # Build edge/sentence spans for chunking
            ec = ta.build(source_text, min_count=1, max_vocab=400)
            if ec.nE == 0:
                self.examples.append(TrainingExample(
                    text=source_text, doc_id=doc.doc_id,
                ))
                continue

            chunk_rex = RexGraph(sources=ec.sources, targets=ec.targets)
            if ec.n_types > 1:
                from agent.auto import attach_faces
                chunk_rex = attach_faces(chunk_rex, type_labels=ec.type_labels)

            chunks = hodge_chunk(
                chunk_rex, ec.edge_spans, ec.sentence_spans,
                source_text, min_chunk_chars=100,
            )

            # Per-chunk features
            for chunk in chunks:
                ex = TrainingExample(
                    text=chunk.text,
                    doc_id=doc.doc_id,
                    chunk_idx=chunk.idx,
                    kappa=chunk.kappa,
                    chi=list(chunk.chi) if hasattr(chunk, 'chi') and chunk.chi is not None else [],
                    hodge=[chunk.hodge_gradient, chunk.hodge_curl, chunk.hodge_harmonic],
                    channel=chunk.dominant_channel,
                )

                # Document-level features
                with contextlib.suppress(Exception):
                    ex.betti = list(rex.betti)

                try:
                    vc = rex.void_complex
                    if vc:
                        ex.n_voids = vc.get('n_voids', 0)
                except Exception:
                    pass

                try:
                    from rexgraph.core._persistence import persistence_diagram, persistence_entropy
                    fv = np.zeros(chunk_rex.nV, dtype=np.float64)
                    fe = np.ones(chunk_rex.nE, dtype=np.float64)
                    ff = np.zeros(chunk_rex.nF, dtype=np.float64) if chunk_rex.nF > 0 else np.array([], dtype=np.float64)
                    dgm = persistence_diagram(fv, fe, ff,
                        chunk_rex.boundary_ptr, chunk_rex.boundary_idx,
                        chunk_rex._B2_col_ptr, chunk_rex._B2_row_idx)
                    ex.persistence_entropy = float(persistence_entropy(dgm['pairs']))
                except Exception:
                    pass

                self.examples.append(ex)

    def feature_matrix(self) -> np.ndarray:
        """Build a feature matrix (n_examples, n_features).

        Features per example:
            [kappa, chi_T, chi_G, chi_F, chi_C,
             hodge_grad, hodge_curl, hodge_harm,
             n_voids, persistence_entropy]
        """
        rows = []
        for ex in self.examples:
            chi = ex.chi if len(ex.chi) == 4 else [0, 0, 0, 0]
            hodge = ex.hodge if len(ex.hodge) == 3 else [0, 0, 0]
            rows.append([
                ex.kappa,
                chi[0], chi[1], chi[2], chi[3],
                hodge[0], hodge[1], hodge[2],
                float(ex.n_voids),
                ex.persistence_entropy,
            ])
        return np.array(rows, dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        return [
            "kappa", "chi_T", "chi_G", "chi_F", "chi_C",
            "hodge_gradient", "hodge_curl", "hodge_harmonic",
            "n_voids", "persistence_entropy",
        ]

    def export_features(self, path: str):
        """Export structural features as safetensors.

        Uses rexgraph.io.safetensors_bridge.fingerprints_to_safetensors.
        Output is directly loadable by PyTorch/JAX/HuggingFace.
        """
        from rexgraph.io.safetensors_bridge import fingerprints_to_safetensors

        matrix = self.feature_matrix()
        labels = np.array([ex.doc_id for ex in self.examples])
        # n_features / n_spans are set by fingerprints_to_safetensors itself
        # (they are reserved metadata keys), so we must not pass them here.
        metadata = {
            "n_examples": str(len(self.examples)),
            "source": "rexgraph-agent",
        }

        fingerprints_to_safetensors(
            matrix, labels, path,
            feature_names=self.feature_names,
            metadata=metadata,
        )
        logger.info("Exported %d examples to %s", len(self.examples), path)
        return path

    def export_training_pairs(self, path: str, target: str = "summary"):
        """Export (input + structural context) -> target pairs.

        Each example becomes:
            input:  chunk text + structural features as prefix
            target: depends on mode
                'summary'   -> chunk text (for self-supervised pretraining)
                'channel'   -> dominant channel label (T/G/F/C classification)
                'kappa'     -> kappa value (regression)
                'custom'    -> uses ex.target field (set by user)

        Saved as safetensors with 'inputs', 'targets', 'features' tensors.
        """
        save_file, _, _ = _st_import()

        texts = []
        targets = []
        for ex in self.examples:
            # Structural prefix
            prefix = "[kappa=%.3f hodge=%.2f/%.2f/%.2f channel=%s voids=%d] " % (
                ex.kappa,
                ex.hodge[0] if len(ex.hodge) > 0 else 0,
                ex.hodge[1] if len(ex.hodge) > 1 else 0,
                ex.hodge[2] if len(ex.hodge) > 2 else 0,
                ex.channel, ex.n_voids,
            )
            texts.append(prefix + ex.text)

            if target == "summary":
                targets.append(ex.text)
            elif target == "channel":
                targets.append(ex.channel)
            elif target == "kappa":
                targets.append(f"{ex.kappa:.4f}")
            elif target == "custom":
                targets.append(ex.target)
            else:
                targets.append(ex.text)

        # Encode as byte arrays for safetensors
        max_input_len = max(len(t.encode()) for t in texts) if texts else 1
        max_target_len = max(len(t.encode()) for t in targets) if targets else 1

        input_arr = np.zeros((len(texts), max_input_len), dtype=np.uint8)
        target_arr = np.zeros((len(targets), max_target_len), dtype=np.uint8)
        input_lens = np.zeros(len(texts), dtype=np.int32)
        target_lens = np.zeros(len(targets), dtype=np.int32)

        for i, t in enumerate(texts):
            b = t.encode()[:max_input_len]
            input_arr[i, :len(b)] = list(b)
            input_lens[i] = len(b)

        for i, t in enumerate(targets):
            b = t.encode()[:max_target_len]
            target_arr[i, :len(b)] = list(b)
            target_lens[i] = len(b)

        tensors = {
            "inputs": input_arr,
            "targets": target_arr,
            "input_lengths": input_lens,
            "target_lengths": target_lens,
            "features": self.feature_matrix(),
        }

        metadata = {
            "rex_meta": json.dumps({
                "n_examples": len(texts),
                "target_type": target,
                "feature_names": self.feature_names,
                "source": "rexgraph-agent",
            }),
        }

        save_file(tensors, path, metadata=metadata)
        logger.info("Exported %d training pairs to %s", len(texts), path)
        return path

    def to_hf_dataset(self):
        """Convert to a HuggingFace Dataset (if datasets is installed)."""
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError("pip install datasets") from exc

        records = []
        matrix = self.feature_matrix()
        for i, ex in enumerate(self.examples):
            row = {
                "text": ex.text,
                "doc_id": ex.doc_id,
                "chunk_idx": ex.chunk_idx,
                "channel": ex.channel,
            }
            for j, name in enumerate(self.feature_names):
                row[name] = float(matrix[i, j]) if i < len(matrix) else 0.0
            records.append(row)
        return Dataset.from_list(records)

    def export_rex_bundles(self, output_dir: str):
        """Export each document as a .rex bundle using rexgraph.io.save_rex."""
        from rexgraph.io import save_rex
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        paths = []
        for doc in self.corpus.documents:
            if doc.rex is not None:
                p = str(Path(output_dir) / (f"{doc.doc_id}.rex"))
                save_rex(p, doc.rex, cache="all")
                paths.append(p)
        return paths


def _st_import():
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    return save_file, load_file, safe_open
