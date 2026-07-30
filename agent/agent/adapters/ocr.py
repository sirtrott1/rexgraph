"""
OCR adapter: image / PDF -> typed relational complex.

Bridges Baidu's Unlimited-OCR with the rexgraph agent pipeline.
Takes an image path, PDF path, or directory of images, runs OCR
to extract text, then builds a relational complex encoding the
text's relational structure.

Two construction strategies:

    **text** (default): feeds the extracted text through the
    TextAdapter, producing a word co-occurrence relational complex.
    Best for analyzing the *content* structure of a document.

    **layout**: parses structural elements from the OCR output
    (headings, paragraphs, tables, lists) and builds a relational
    complex where vertices are document sections and edges encode
    structural adjacency, containment, and cross-reference.
    Best for analyzing *document structure*.

Usage:

    from agent.adapters.ocr import OCRAdapter

    adapter = OCRAdapter()

    # From a scanned image
    edges = adapter.build("receipt.jpg")

    # From a PDF with layout analysis
    edges = adapter.build("contract.pdf", strategy="layout")

    # With a pre-configured OCR client
    from agent.integrations.unlimited_ocr import UnlimitedOCRClient
    client = UnlimitedOCRClient(server_url="http://gpu-server:10000")
    adapter = OCRAdapter(client=client)
    edges = adapter.build("scan.png")
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from . import DomainAdapter, EdgeConstruction

logger = logging.getLogger(__name__)

# Section parser
def _parse_sections(text: str) -> List[Dict]:
    """Parse OCR output (markdown) into structural sections.

    Returns a list of dicts, each with:
        label : str   - section heading or type
        level : int   - heading depth (0 = body, 1 = h1, 2 = h2, ...)
        body  : str   - section body text
        kind  : str   - 'heading', 'paragraph', 'table', 'list', 'code'
    """
    sections: List[Dict] = []
    current_heading = "document"
    current_level = 0
    current_body: List[str] = []

    def flush():
        body = "\n".join(current_body).strip()
        if body:
            kind = _classify_block(body)
            sections.append({
                "label": current_heading,
                "level": current_level,
                "body": body,
                "kind": kind,
            })

    for line in text.split("\n"):
        heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
        if heading_match:
            flush()
            current_body = []
            hashes = heading_match.group(1)
            current_level = len(hashes)
            current_heading = heading_match.group(2).strip()
            continue

        current_body.append(line)

    flush()
    return sections


def _classify_block(text: str) -> str:
    """Classify a text block by its structural type."""
    lines = text.strip().split("\n")
    if not lines:
        return "paragraph"

    # Table: has pipe-separated rows
    pipe_lines = sum(1 for l in lines if "|" in l and l.count("|") >= 2)
    if pipe_lines >= 2:
        return "table"

    # List: most lines start with bullets or numbers
    list_lines = sum(
        1 for l in lines
        if re.match(r'^\s*[-*•]\s', l) or re.match(r'^\s*\d+[.)]\s', l)
    )
    if list_lines > len(lines) * 0.5:
        return "list"

    # Code: indented or fenced
    if text.strip().startswith("```"):
        return "code"
    indented = sum(1 for l in lines if l.startswith("    ") or l.startswith("\t"))
    if indented > len(lines) * 0.7:
        return "code"

    return "paragraph"


# Layout graph construction
def _build_layout_graph(
    sections: List[Dict],
    max_vertices: int = 500,
) -> Tuple[
    List[str],       # vertex labels
    List[Tuple[int, int, float, int]],  # (src, tgt, weight, type)
    List[str],       # type names
]:
    """Build a graph from document layout structure.

    Edge types:
        0 = sequential  - sections that appear consecutively
        1 = hierarchical - parent heading -> child section
        2 = thematic     - sections sharing significant word overlap

    Returns vertex labels, edge tuples, and type names.
    """
    if not sections:
        return [], [], []

    # Limit to max_vertices
    sections = sections[:max_vertices]
    n = len(sections)

    labels = []
    for i, sec in enumerate(sections):
        label = sec["label"][:60]
        if not label or label == "document":
            label = f"{sec['kind']}_{i}"
        labels.append(label)

    edges: List[Tuple[int, int, float, int]] = []

    # Type 0: sequential adjacency
    for i in range(n - 1):
        edges.append((i, i + 1, 1.0, 0))

    # Type 1: hierarchical containment
    # Each section at level > 0 is a child of the nearest preceding
    # section at a lower level
    heading_stack: List[int] = []
    for i, sec in enumerate(sections):
        level = sec["level"]
        # Pop deeper or equal levels
        while heading_stack and sections[heading_stack[-1]]["level"] >= level and level > 0:
            heading_stack.pop()
        if heading_stack and level > 0:
            parent = heading_stack[-1]
            depth_diff = level - sections[parent]["level"]
            weight = 1.0 / max(depth_diff, 1)
            edges.append((parent, i, weight, 1))
        if level > 0:
            heading_stack.append(i)

    # Type 2: thematic overlap
    # Build word sets for each section and connect sections
    # that share significant vocabulary
    word_sets: List[set] = []
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'to', 'of', 'in', 'for', 'on', 'with',
        'at', 'by', 'from', 'and', 'but', 'or', 'not', 'this', 'that',
    }
    for sec in sections:
        words = set(re.findall(r'[a-zA-Z]{3,}', sec["body"].lower()))
        words -= stopwords
        word_sets.append(words)

    for i in range(n):
        for j in range(i + 2, min(i + 20, n)):  # skip immediate neighbor
            if not word_sets[i] or not word_sets[j]:
                continue
            overlap = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            if union == 0:
                continue
            jaccard = overlap / union
            if jaccard > 0.15:
                edges.append((i, j, jaccard, 2))

    type_names = ["sequential", "hierarchical", "thematic"]
    return labels, edges, type_names


# Adapter
class OCRAdapter(DomainAdapter):
    """Convert images/PDFs to typed relational complexes via OCR.

    Parameters
    ----------
    client : UnlimitedOCRClient or OfflineOCRClient, optional
        Pre-configured OCR client.  If not provided, one is
        created automatically using ``create_ocr_client()``.
    """

    name = "ocr"

    def __init__(self, client=None):
        self.client = client
        self._text_adapter = None

    def _get_client(self):
        """Lazily initialize the OCR client."""
        if self.client is None:
            from agent.integrations.unlimited_ocr import create_ocr_client
            self.client = create_ocr_client()
        return self.client

    def _get_text_adapter(self):
        """Lazily initialize the text adapter."""
        if self._text_adapter is None:
            from agent.adapters.text import TextAdapter
            self._text_adapter = TextAdapter()
        return self._text_adapter

    def build_from_text(
        self,
        text: str,
        strategy: str = "layout",
        window: int = 0,
        min_count: int = 1,
        max_vocab: int = 500,
        face_selection: str = "typed",
        detect_tables: bool = True,
    ) -> EdgeConstruction:
        """Build a relational complex from *already-extracted* OCR text.

        This is the entry point the pipeline should use when OCR has
        already run (e.g. in the main process, with the model cached in
        VRAM).  It skips the OCR-extraction step in :meth:`build` and
        preserves document structure via the ``layout`` strategy, which
        keeps headings / tables / columns as typed edges instead of
        flattening everything into word co-occurrence.

        Parameters
        ----------
        text : str
            Text previously produced by an OCR backend.
        strategy : str
            ``'layout'`` (default): document structure from OCR sections,
            with automatic fallback to ``'text'`` when too few sections
            are detected.  ``'text'``: word co-occurrence only.
        window, min_count, max_vocab, face_selection
            Forwarded to the text strategy (used directly or as the
            layout fallback).

        Returns
        -------
        EdgeConstruction
        """
        if not text or len(text.strip()) < 10:
            raise RuntimeError(
                "build_from_text received no usable text. "
                "The OCR backend likely returned empty output."
            )

        # If the OCR output is really a table, recover its structure so
        # column headers become vertex labels (as a native CSV would),
        # instead of dissolving into word co-occurrence (audit 2.2).
        if detect_tables:
            try:
                from agent.adapters.table_detect import detect_tables as _dt
                frames = _dt(text)
                if frames:
                    ec = self._build_from_table(frames, face_selection)
                    if ec is not None and ec.nE > 0:
                        return ec
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("OCR table detection skipped: %s", e)

        if strategy == "layout":
            return self._build_layout(text, face_selection=face_selection)
        return self._build_text(
            text,
            window=window,
            min_count=min_count,
            max_vocab=max_vocab,
            face_selection=face_selection,
        )

    def build(
        self,
        data: Union[str, List[str]],
        strategy: str = "text",
        window: int = 0,
        min_count: int = 1,
        max_vocab: int = 500,
        face_selection: str = "typed",
        ocr_prompt: Optional[str] = None,
        dpi: int = 300,
        **kwargs,
    ) -> EdgeConstruction:
        """Build a relational complex from image/PDF via OCR.

        Parameters
        ----------
        data : str or list of str
            Image file path, PDF file path, directory of images,
            or list of image file paths.
        strategy : str
            ``'text'``: word co-occurrence from OCR output (default).
            ``'layout'``: document structure from OCR sections.
        window : int
            Co-occurrence window for text strategy. 0 = full sentence.
        min_count : int
            Minimum co-occurrence count (text strategy).
        max_vocab : int
            Maximum vocabulary size (text strategy).
        face_selection : str
            Face selection mode: 'typed', 'all', 'none'.
        ocr_prompt : str, optional
            Custom prompt for the OCR model.
        dpi : int
            PDF rasterization DPI.

        Returns
        -------
        EdgeConstruction
        """
        # Step 1: Extract text via OCR
        text = self._extract_text(data, ocr_prompt=ocr_prompt, dpi=dpi)

        if not text or len(text.strip()) < 10:
            logger.error("OCR produced no usable text from %s", data)
            raise RuntimeError(
                f"OCR produced no text from {data}. "
                "Check that the file is readable and an OCR backend is installed. "
                "Run: rexgraph-ocr status"
            )

        # Step 2: Build relational complex
        if strategy == "layout":
            return self._build_layout(text, face_selection=face_selection)
        else:
            return self._build_text(
                text,
                window=window,
                min_count=min_count,
                max_vocab=max_vocab,
                face_selection=face_selection,
            )

    def _extract_text(
        self,
        data: Union[str, List[str]],
        ocr_prompt: Optional[str] = None,
        dpi: int = 300,
    ) -> str:
        """Run OCR on the input and return the combined text."""
        from agent.integrations.unlimited_ocr import (
            is_image_file, is_pdf_file,
        )

        client = self._get_client()

        # List of image paths
        if isinstance(data, list):
            result = client.ocr_images(data, prompt=ocr_prompt)
            return result.full_text

        path = str(data)

        # Directory
        if os.path.isdir(path):
            result = client.ocr_directory(path, prompt=ocr_prompt)
            return result.full_text

        # PDF
        if is_pdf_file(path):
            result = client.ocr_pdf(path, dpi=dpi, prompt=ocr_prompt)
            return result.full_text

        # Single image
        if is_image_file(path):
            result = client.ocr_image(path, prompt=ocr_prompt)
            return result.text

        # Assume it's already text (passthrough)
        if isinstance(data, str) and not os.path.exists(path):
            return data

        raise ValueError(
            f"Cannot determine input type for OCR: {path}"
        )

    def _build_text(
        self,
        text: str,
        window: int = 0,
        min_count: int = 1,
        max_vocab: int = 500,
        face_selection: str = "typed",
    ) -> EdgeConstruction:
        """Delegate to TextAdapter for word co-occurrence construction."""
        adapter = self._get_text_adapter()
        return adapter.build(
            text,
            window=window,
            min_count=min_count,
            max_vocab=max_vocab,
            face_selection=face_selection,
        )

    def _build_from_table(self, frames, face_selection: str = "typed"):
        """Turn a recovered OCR table into an EdgeConstruction.

        Uses the largest detected frame and routes it through the same
        DataFrame classification auto_rex uses, so an OCR'd CSV yields
        the column-labelled complex a native CSV would.
        """
        frame = max(frames, key=lambda f: f.shape[0] * f.shape[1])
        numeric = frame.select_dtypes(include=["number"])
        if numeric.shape[1] < 2 or numeric.shape[0] < 2:
            return None
        from agent.adapters.feature_matrix import FeatureMatrixAdapter
        X = numeric.to_numpy(dtype=float)
        adapter = FeatureMatrixAdapter()
        return adapter.build(
            X, feature_names=list(numeric.columns.astype(str))
        )

    def _build_layout(
        self,
        text: str,
        face_selection: str = "typed",
    ) -> EdgeConstruction:
        """Build a relational complex from document layout structure."""
        sections = _parse_sections(text)

        if len(sections) < 2:
            # Fall back to text strategy if layout parsing finds too few sections
            return self._build_text(text)

        labels, edges, type_names = _build_layout_graph(sections)

        if not edges:
            return self._build_text(text)

        n_types = len(type_names)
        sources = np.array([e[0] for e in edges], dtype=np.int32)
        targets = np.array([e[1] for e in edges], dtype=np.int32)
        weights = np.array([e[2] for e in edges], dtype=np.float64)
        signs = np.ones(len(edges), dtype=np.float64)
        type_labels = np.array([e[3] for e in edges], dtype=np.int32)

        return EdgeConstruction(
            sources=sources,
            targets=targets,
            weights=weights,
            signs=signs,
            type_labels=type_labels,
            vertex_labels=labels,
            n_types=n_types,
            type_names=type_names,
        )

    @staticmethod
    def _empty_construction() -> EdgeConstruction:
        """Return an empty EdgeConstruction when OCR fails."""
        return EdgeConstruction(
            sources=np.array([], dtype=np.int32),
            targets=np.array([], dtype=np.int32),
            weights=np.array([], dtype=np.float64),
            signs=np.array([], dtype=np.float64),
            type_labels=np.array([], dtype=np.int32),
            vertex_labels=[],
            n_types=0,
            type_names=[],
        )

    def interpret(self, results: dict) -> dict:
        """Add OCR-specific interpretation to analysis results."""
        interp = dict(results)
        interp["domain"] = "document_ocr"

        # Hodge interpretation for document structure
        hodge = results.get("hodge", {})
        g = hodge.get("pct_gradient", 0)
        c = hodge.get("pct_curl", 0)
        h = hodge.get("pct_harmonic", 0)

        if g > 0.6:
            interp["structure_assessment"] = (
                "Hierarchical document: gradient-dominant structure "
                "indicates clear top-down organization."
            )
        elif c > 0.3:
            interp["structure_assessment"] = (
                "Cross-referenced document: significant curl content "
                "indicates circular references between sections."
            )
        elif h > 0.2:
            interp["structure_assessment"] = (
                "Document has unresolved thematic threads: harmonic "
                "content indicates topics that span sections without "
                "closing into complete references."
            )

        return interp
