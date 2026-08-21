"""
Text adapter: raw text to typed relational complex.

Converts text into a relational complex where:
    - Vertices are distinct words.
    - Edges are word co-occurrences within the same sentence.
    - Edge weight is co-occurrence count.
    - Edge type is derived from relative position (adjacent, near, distant).
    - Faces are word triples that all co-occur in at least one sentence.

No external NLP libraries required. Tokenization is whitespace + punctuation
stripping. Sentence splitting is on period, question mark, exclamation mark,
and newline boundaries.

The resulting relational complex captures the text's relational structure:
    - Gradient content: hierarchical word frequency structure.
    - Curl content: closed rhetorical loops (A mentions B, B mentions C,
      C mentions A within sentences).
    - Harmonic content: unresolved thematic connections (words that
      co-occur in open chains but never close into triangles).
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from . import DomainAdapter, EdgeConstruction


def _tokenize(text: str) -> list[tuple[list[str], int, int]]:
    """Split text into sentences with character offsets.

    Returns list of (words, char_start, char_end) tuples.
    """
    raw_sents = re.split(r'([.?!]\s+|\n\n+|\n)', text)
    results = []
    offset = 0
    for i, part in enumerate(raw_sents):
        if i % 2 == 0:  # content part (not delimiter)
            words = re.findall(r'[a-zA-Z0-9]+(?:\'[a-zA-Z]+)?', part.lower())
            if len(words) >= 1:  # allow single-word sentences
                results.append((words, offset, offset + len(part)))
        offset += len(part)
    return results


def _build_cooccurrence(
    sentences: list[tuple[list[str], int, int]],
    window: int = 0,
    min_count: int = 1,
    max_vocab: int = 500,
    stopwords: set | None = None,
) -> tuple[dict[str, int], dict[tuple[int, int], float],
           dict[tuple[int, int], int], dict[tuple[int, int], list[int]]]:
    """Build word co-occurrence graph from tokenized sentences.

    Returns
    -------
    vocab : dict mapping word to vertex index
    edges : dict mapping (src, tgt) to co-occurrence weight
    types : dict mapping (src, tgt) to type index
    edge_sents : dict mapping (src, tgt) to list of sentence indices
    """
    if stopwords is None:
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'not',
            'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
            'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
            'because', 'if', 'when', 'where', 'how', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'me',
            'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'it', 'its', 'they', 'them', 'their', 'about', 'up',
        }

    # Count word frequencies
    freq = defaultdict(int)
    for sent_words, _, _ in sentences:
        for w in sent_words:
            if w not in stopwords and len(w) > 1:
                freq[w] += 1

    # Top N by frequency
    top = sorted(freq.items(), key=lambda x: -x[1])[:max_vocab]
    vocab = {w: i for i, (w, _) in enumerate(top)}

    # Count co-occurrences and track sentence membership
    cooc = defaultdict(int)
    dist_sum = defaultdict(int)
    dist_count = defaultdict(int)
    edge_sents = defaultdict(list)
    forward_count = defaultdict(int)  # how often source precedes target

    for sent_idx, (sent_words, _, _) in enumerate(sentences):
        filtered = [(pos, w) for pos, w in enumerate(sent_words) if w in vocab]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                pos_i, w_i = filtered[i]
                pos_j, w_j = filtered[j]
                d = abs(pos_j - pos_i)
                if window > 0 and d > window:
                    continue
                a, b = min(vocab[w_i], vocab[w_j]), max(vocab[w_i], vocab[w_j])
                cooc[(a, b)] += 1
                dist_sum[(a, b)] += d
                dist_count[(a, b)] += 1
                edge_sents[(a, b)].append(sent_idx)
                # Track reading order: does the lower-index vertex come first in text?
                if vocab[w_i] <= vocab[w_j]:
                    forward_count[(a, b)] += 1  # natural order

    # Filter by min_count and assign types
    edges = {}
    types = {}
    for (a, b), count in cooc.items():
        if count >= min_count:
            edges[(a, b)] = float(count)
            avg_dist = dist_sum[(a, b)] / dist_count[(a, b)]
            if avg_dist <= 1.5:
                types[(a, b)] = 0  # adjacent
            elif avg_dist <= 3.5:
                types[(a, b)] = 1  # near
            else:
                types[(a, b)] = 2  # distant

    return vocab, edges, types, dict(edge_sents), dict(forward_count)


class TextAdapter(DomainAdapter):
    """Convert raw text to a typed relational complex.

    Two constructions, and they are different objects rather than two settings of one:

        relation_mode="pairwise"    windowed co-occurrence. Every sentence becomes
                                    C(k,2) or windowed word PAIRS, which is what
                                    `sources`/`targets` can hold. This is what the
                                    adapter has always produced and it stays the
                                    default, because `exchange`, `training` and the
                                    corpus builder read those arrays directly.
        relation_mode="branching"   a sentence is ONE k-ary relation over its words,
                                    carried in `branching` and built as a single
                                    boundary column. No pairs are enumerated, because
                                    they are not in the text: measured on one book,
                                    1,469 sentences give 1,469 relations against 12,890
                                    under spanning pairs, and the extra 11,421 columns
                                    manufacture 1,129 dimensions of rank and 10,292 of
                                    the 10,324 cycles.

    "branching" is what `rexgraph.document.build_document` produces, so it is the mode a
    QUERY must use: `interfacing_score` compares a query complex against a document one,
    and scoring a pairwise object against a branching object compares two different
    things however good the score function is.

    Both modes share one tokenizer (`rexgraph.construct.from_text`) so the vocabulary
    a query and a document align on is produced by the same code.
    """

    name = "text"

    def build(
        self,
        text: str,
        window: int = 0,
        min_count: int = 1,
        max_vocab: int = 500,
        face_selection: str = "auto",
        relation_mode: str = "pairwise",
        min_terms: int = 1,
        **kwargs,
    ) -> EdgeConstruction:
        """Build a relational complex from raw text with span tracking."""
        from . import EdgeSpan, SentenceSpan

        if relation_mode not in ("pairwise", "branching"):
            raise ValueError(
                f"relation_mode must be 'pairwise' or 'branching', "
                f"not {relation_mode!r}")
        if relation_mode == "branching":
            return self._build_branching(text, min_terms=min_terms,
                                         max_vocab=max_vocab)

        sentences = _tokenize(text)
        vocab, cooc_edges, cooc_types, edge_sents, fwd_count = _build_cooccurrence(
            sentences,
            window=window,
            min_count=min_count,
            max_vocab=max_vocab,
        )

        # Build vertex labels from vocabulary even if no edges
        rev_vocab = {v: k for k, v in vocab.items()}
        vlabels = [rev_vocab.get(i, "v%d" % i) for i in range(len(vocab))]

        if not cooc_edges:
            return EdgeConstruction(
                sources=np.array([], dtype=np.int32),
                targets=np.array([], dtype=np.int32),
                weights=np.array([], dtype=np.float64),
                signs=np.array([], dtype=np.float64),
                type_labels=np.array([], dtype=np.int32),
                vertex_labels=vlabels,
                n_types=0,
                type_names=[],
                source_text=text,
                sentence_spans=[SentenceSpan(
                    idx=i, char_start=cs, char_end=ce,
                    text=text[cs:ce],
                ) for i, (_, cs, ce) in enumerate(sentences)],
            )

        edge_list = sorted(cooc_edges.keys())
        sources = np.array([e[0] for e in edge_list], dtype=np.int32)
        targets = np.array([e[1] for e in edge_list], dtype=np.int32)
        weights = np.array([cooc_edges[e] for e in edge_list], dtype=np.float64)
        # Sign from reading order: +1 if source typically precedes target, -1 if reversed
        signs = np.array([
            +1.0 if fwd_count.get(e, 0) >= cooc_edges[e] / 2 else -1.0
            for e in edge_list
        ], dtype=np.float64)
        type_labels = np.array([cooc_types[e] for e in edge_list], dtype=np.int32)

        # Build edge-to-key mapping
        edge_key_to_idx = {e: i for i, e in enumerate(edge_list)}

        # Build sentence spans
        sent_spans = []
        for idx, (_words, cs, ce) in enumerate(sentences):
            sent_spans.append(SentenceSpan(
                idx=idx, char_start=cs, char_end=ce,
                text=text[cs:ce],
            ))

        # Build edge spans
        rev_vocab = {v: k for k, v in vocab.items()}
        vlabels = [rev_vocab.get(i, f"v{i}") for i in range(len(vocab))]

        edge_spans = []
        for edge_key, sent_indices in edge_sents.items():
            eidx = edge_key_to_idx.get(edge_key)
            if eidx is None:
                continue
            # Use the first sentence this edge appeared in
            first_sent = sent_indices[0]
            if first_sent < len(sent_spans):
                sp = sent_spans[first_sent]
                edge_spans.append(EdgeSpan(
                    edge_idx=eidx,
                    source_label=vlabels[edge_key[0]],
                    target_label=vlabels[edge_key[1]],
                    char_start=sp.char_start,
                    char_end=sp.char_end,
                    sentence_idx=first_sent,
                ))

        ec = EdgeConstruction(
            sources=sources,
            targets=targets,
            weights=weights,
            signs=signs,
            type_labels=type_labels,
            vertex_labels=vlabels,
            n_types=3,
            type_names=["adjacent", "near", "distant"],
            edge_spans=edge_spans,
            sentence_spans=sent_spans,
            source_text=text,
        )

        return ec

    def _build_branching(self, text, *, min_terms=1, max_vocab=500):
        """Sentences as k-ary relations, through the canonical constructor.

        The tokenisation, the vocabulary and the sentence segmentation all come from
        `rexgraph`, not from a second implementation here: `segment_sentences` decides the
        boundaries by channel agreement and `from_text(pair_mode="none")` builds the
        field. That is the whole point: a query tokenised differently from the documents
        aligns on a vocabulary neither of them has.

        `sources`/`targets` come back EMPTY, which is honest rather than lossy: there are
        no 2-ary relations in this construction, and `build_rex_from_edges` reads the
        supports out of `branching`.
        """
        from rexgraph.construct import from_text
        from rexgraph.segment import segment_sentences

        from . import EdgeSpan, SentenceSpan

        spans, _method = segment_sentences(text)
        if not spans:
            spans = [(0, len(text))] if text.strip() else []
        sents = [text[a:a + n] for a, n in spans]
        sent_spans = [SentenceSpan(idx=i, char_start=a, char_end=a + n,
                                   text=text[a:a + n])
                      for i, (a, n) in enumerate(spans)]

        empty = dict(sources=np.array([], dtype=np.int32),
                     targets=np.array([], dtype=np.int32),
                     weights=np.array([], dtype=np.float64),
                     signs=np.array([], dtype=np.float64),
                     type_labels=np.array([], dtype=np.int32),
                     n_types=0, type_names=[], source_text=text,
                     sentence_spans=sent_spans)
        if not sents:
            return EdgeConstruction(vertex_labels=[], **empty)
        try:
            _rex, info = from_text(None, sentences=sents, pair_mode="none",
                                   min_terms=int(min_terms), max_terms=None,
                                   document_vertex=False, verify=False)
        except ValueError:
            # no sentence cleared the filter. A vocabulary still exists and callers
            # align on labels, so return it rather than raising: a one-word query is a
            # true reading of the input, not an error.
            from rexgraph.construct import _TOKEN
            toks = [w for w in re.findall(_TOKEN, text.lower()) if w]
            return EdgeConstruction(vertex_labels=list(dict.fromkeys(toks)), **empty)

        vertex_of = info["vertex_of"]
        labels = list(info.get("vocab") or info.get("members") or [])
        kept = list(info["kept"])
        branching, edge_spans = [], []
        for j, seq in enumerate(info["sequences"]):
            support = sorted({vertex_of[w] for w in dict.fromkeys(seq)})
            branching.append(support)
            si = kept[j]
            if si < len(sent_spans) and len(support) >= 2:
                sp = sent_spans[si]
                # source/target name the relation's first two participants. A k-ary
                # relation has no two distinguished endpoints, so these are a
                # compatibility shim for readers written against pairs; `branching`
                # carries the whole support.
                edge_spans.append(EdgeSpan(
                    edge_idx=j, source_label=labels[support[0]],
                    target_label=labels[support[1]],
                    char_start=sp.char_start, char_end=sp.char_end,
                    sentence_idx=si))
        return EdgeConstruction(vertex_labels=labels, branching=branching,
                                edge_spans=edge_spans, **empty)
