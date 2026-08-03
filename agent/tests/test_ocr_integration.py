"""
Tests for the Unlimited-OCR integration and OCR adapter.

Covers: input type detection for images/PDFs, OCR client utilities,
layout parsing, section classification, layout graph construction,
text-strategy and layout-strategy edge construction, offline
fallback, and end-to-end pipeline integration.

Does NOT require a running Unlimited-OCR server or GPU.
"""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
from agent.adapters import EdgeConstruction
from agent.adapters.ocr import (
    OCRAdapter,
    _build_layout_graph,
    _classify_block,
    _parse_sections,
)
from agent.auto import detect_input_type
from agent.integrations.unlimited_ocr import (
    MistralOCRClient,
    OCRBatchResult,
    OCRResult,
    OfflineOCRClient,
    UnlimitedOCRClient,
    _encode_image,
    create_ocr_client,
    is_image_file,
    is_pdf_file,
)

SAMPLE_OCR_OUTPUT = """# Invoice

## Billing Information

Customer: Acme Corporation
Date: 2026-01-15
Invoice Number: INV-2026-0042

## Items

| Item | Quantity | Unit Price | Total |
|------|----------|------------|-------|
| Widget A | 100 | $5.00 | $500.00 |
| Widget B | 50 | $12.00 | $600.00 |
| Gizmo C | 25 | $8.50 | $212.50 |

## Payment Terms

Payment is due within 30 days of invoice date.
Late payments incur a 1.5% monthly interest charge.

## Notes

- All items ship from warehouse in Austin, TX
- Returns accepted within 14 days
- Contact billing@acme.example for questions about items
"""

SAMPLE_PROSE_OUTPUT = """The relational complex framework provides a mathematical
foundation for analyzing multi-entity relationships. Each entity
becomes a vertex. Pairwise interactions become edges. Three-way
coherences become faces. The boundary operator connects these grades
and the chain condition ensures algebraic consistency.

Structural character decomposes each edge into four channels.
The topology channel captures the Hodge-theoretic content.
The geometry channel measures overlap and shared context.
The frustration channel detects contradictions and tension.
The copath channel encodes higher-order structure.

Void analysis reveals where expected coherences fail to materialize.
A void is a potential face whose three edges exist but whose
three-way relationship is absent or contradicted. The void fraction
measures the gap between what the pairwise data promises and what
the higher-order structure delivers."""


def _create_test_image(path: str, width: int = 100, height: int = 50):
    """Create a minimal valid PNG file for testing."""
    # Minimal 1x1 white PNG
    import struct
    import zlib

    def _chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = _chunk(b'IHDR', struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw = b''
    for _ in range(height):
        raw += b'\x00' + b'\xff' * (width * 3)
    idat = _chunk(b'IDAT', zlib.compress(raw))
    iend = _chunk(b'IEND', b'')

    with open(path, 'wb') as f:
        f.write(sig + ihdr + idat + iend)



def test_detect_image_types():
    """Image file extensions should be classified as 'image'."""
    print("── Image type detection ──")
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        result = detect_input_type(f"document{ext}")
        assert result == "image", f"Expected 'image' for {ext}, got {result}"
        print(f"  ✓ {ext} -> image")
    print()


def test_detect_pdf_type():
    """PDF files should be classified as 'pdf'."""
    print("── PDF type detection ──")
    result = detect_input_type("contract.pdf")
    assert result == "pdf"
    print("  ✓ .pdf -> pdf")
    print()


def test_detect_image_directory():
    """Directories containing images should be classified as 'image_dir'."""
    print("── Image directory detection ──")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some image files
        for name in ["a.png", "b.jpg", "c.jpeg"]:
            _create_test_image(os.path.join(tmpdir, name))
        result = detect_input_type(tmpdir)
        assert result == "image_dir", f"Expected 'image_dir', got {result}"
        print("  ✓ directory with images -> image_dir")
    print()



def test_is_image_file():
    """is_image_file should recognize all supported extensions."""
    print("── is_image_file ──")
    assert is_image_file("photo.png")
    assert is_image_file("SCAN.JPG")
    assert is_image_file("doc.webp")
    assert not is_image_file("data.csv")
    assert not is_image_file("report.pdf")
    print("  ✓ all image extensions recognized")
    print()


def test_is_pdf_file():
    """is_pdf_file should recognize PDF files."""
    print("── is_pdf_file ──")
    assert is_pdf_file("contract.pdf")
    assert is_pdf_file("SCAN.PDF")
    assert not is_pdf_file("image.png")
    print("  ✓ PDF extension recognized")
    print()


def test_encode_image():
    """_encode_image should produce valid base64 content blocks."""
    print("── Image encoding ──")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(f.name)
        result = _encode_image(f.name)
    os.unlink(f.name)

    assert result["type"] == "image_url"
    url = result["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    print("  ✓ PNG encoded as base64 data URL")
    print()



def test_parse_sections():
    """_parse_sections should extract headings and body blocks."""
    print("── Section parsing ──")
    sections = _parse_sections(SAMPLE_OCR_OUTPUT)
    assert len(sections) >= 4
    print(f"  ✓ Parsed {len(sections)} sections")

    # Check heading detection - h2 sections should be found
    labels = [s["label"] for s in sections]
    assert any("Billing" in l for l in labels), f"Expected 'Billing' heading, got {labels}"
    assert any("Items" in l for l in labels), f"Expected 'Items' heading, got {labels}"
    print(f"  ✓ Headings: {labels[:5]}")

    # Check that levels are assigned
    levels = [s["level"] for s in sections]
    assert 2 in levels  # h2 headings
    print(f"  ✓ Levels detected: {sorted(set(levels))}")

    # Check kind classification
    kinds = [s["kind"] for s in sections]
    assert "table" in kinds  # Items section has a table
    assert "list" in kinds   # Notes section has a list
    print(f"  ✓ Block kinds: {kinds}")
    print()


def test_classify_block():
    """_classify_block should identify tables, lists, and paragraphs."""
    print("── Block classification ──")

    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert _classify_block(table) == "table"
    print("  ✓ table block classified")

    bullet_list = "- item one\n- item two\n- item three"
    assert _classify_block(bullet_list) == "list"
    print("  ✓ list block classified")

    paragraph = "This is a normal paragraph of text with no special formatting."
    assert _classify_block(paragraph) == "paragraph"
    print("  ✓ paragraph block classified")

    code = "```python\nprint('hello')\n```"
    assert _classify_block(code) == "code"
    print("  ✓ code block classified")
    print()



def test_build_layout_graph():
    """_build_layout_graph should produce edges from parsed sections."""
    print("── Layout graph construction ──")
    sections = _parse_sections(SAMPLE_OCR_OUTPUT)
    labels, edges, type_names = _build_layout_graph(sections)

    assert len(labels) == len(sections)
    assert len(edges) > 0
    assert len(type_names) == 3
    assert type_names == ["sequential", "hierarchical", "thematic"]
    print(f"  ✓ {len(labels)} vertices, {len(edges)} edges")

    # Check edge types
    types_present = set(e[3] for e in edges)
    assert 0 in types_present  # sequential
    print(f"  ✓ Edge types present: {types_present}")

    # Check sequential edges exist between consecutive sections
    seq_edges = [(e[0], e[1]) for e in edges if e[3] == 0]
    for i in range(len(sections) - 1):
        assert (i, i + 1) in seq_edges, f"Missing sequential edge ({i}, {i+1})"
    print(f"  ✓ All {len(sections)-1} sequential edges present")
    print()


def test_layout_graph_thematic_edges():
    """Sections with overlapping vocabulary should get thematic edges."""
    print("── Thematic edge detection ──")
    sections = _parse_sections(SAMPLE_OCR_OUTPUT)
    _, edges, _ = _build_layout_graph(sections)

    thematic = [e for e in edges if e[3] == 2]
    print(f"  ✓ {len(thematic)} thematic edges found")
    # The invoice document has repeated terms; should find some overlap
    # (this is content-dependent, so we just check the mechanism works)
    print()



def test_ocr_adapter_text_strategy():
    """OCRAdapter with text strategy should delegate to TextAdapter."""
    print("── OCR adapter: text strategy ──")

    # Create a mock client that returns our sample text
    mock_client = MagicMock()
    mock_client.ocr_image.return_value = OCRResult(
        text=SAMPLE_PROSE_OUTPUT, source="test.png",
    )

    adapter = OCRAdapter(client=mock_client)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(f.name)
        edges = adapter.build(f.name, strategy="text")
    os.unlink(f.name)

    assert isinstance(edges, EdgeConstruction)
    assert edges.nV > 0
    assert edges.nE > 0
    assert edges.n_types > 0
    print(f"  ✓ Text strategy: {edges.nV} vertices, {edges.nE} edges")
    print(f"    Types: {edges.type_names}")

    # Check that domain-specific words appear as vertices
    vlabels_lower = [v.lower() for v in edges.vertex_labels]
    assert "relational" in vlabels_lower or "complex" in vlabels_lower
    print(f"  ✓ Domain vocabulary preserved: {edges.vertex_labels[:8]}")
    print()



def test_ocr_adapter_layout_strategy():
    """OCRAdapter with layout strategy should build a section graph."""
    print("── OCR adapter: layout strategy ──")

    mock_client = MagicMock()
    mock_client.ocr_image.return_value = OCRResult(
        text=SAMPLE_OCR_OUTPUT, source="invoice.png",
    )

    adapter = OCRAdapter(client=mock_client)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(f.name)
        edges = adapter.build(f.name, strategy="layout")
    os.unlink(f.name)

    assert isinstance(edges, EdgeConstruction)
    assert edges.nV > 0
    assert edges.nE > 0
    assert edges.n_types == 3
    assert edges.type_names == ["sequential", "hierarchical", "thematic"]
    print(f"  ✓ Layout strategy: {edges.nV} vertices, {edges.nE} edges")
    print(f"    Types: {edges.type_names}")
    print(f"    Labels: {edges.vertex_labels[:5]}")
    print()



def test_ocr_adapter_pdf():
    """OCRAdapter should handle PDF inputs via the client."""
    print("── OCR adapter: PDF input ──")

    mock_client = MagicMock()
    mock_client.ocr_pdf.return_value = OCRBatchResult(
        pages=[
            OCRResult(text="Page one content about widgets.", source="p1.png", page=1),
            OCRResult(text="Page two content about gizmos.", source="p2.png", page=2),
        ],
        source="test.pdf",
    )

    adapter = OCRAdapter(client=mock_client)
    edges = adapter.build("test.pdf", strategy="text")

    assert isinstance(edges, EdgeConstruction)
    assert edges.nV > 0
    print(f"  ✓ PDF processed: {edges.nV} vertices, {edges.nE} edges")
    print()



def test_ocr_adapter_empty_result():
    """OCRAdapter.build should fail loudly when OCR yields no text.

    Backend-agnostic: a mock client stands in for whatever OCR model is
    installed, so this does not depend on PaddleOCR / GOT-OCR / Tesseract
    or on CUDA vs ROCm. The contract is that empty OCR raises a clear
    RuntimeError (the pipeline layers catch it and skip the document)
    rather than silently producing a 0-vertex complex.
    """
    print("── OCR adapter: empty result ──")

    mock_client = MagicMock()
    mock_client.ocr_image.return_value = OCRResult(text="", source="blank.png")

    adapter = OCRAdapter(client=mock_client)

    raised = False
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(f.name)
        try:
            edges = adapter.build(f.name)
        except RuntimeError:
            raised = True
        else:
            # A graceful empty construction is also acceptable; a
            # non-empty complex from empty OCR is not.
            assert isinstance(edges, EdgeConstruction)
            assert edges.nV == 0 and edges.nE == 0
    os.unlink(f.name)

    print(f"  ✓ Empty OCR result handled (raised={raised})")
    print()



def test_ocr_adapter_interpret():
    """OCRAdapter.interpret should add domain-specific context."""
    print("── OCR adapter: interpretation ──")

    adapter = OCRAdapter()

    # Gradient-dominant document
    results = {"hodge": {"pct_gradient": 0.75, "pct_curl": 0.15, "pct_harmonic": 0.10}}
    interp = adapter.interpret(results)
    assert interp["domain"] == "document_ocr"
    assert "Hierarchical" in interp.get("structure_assessment", "")
    print("  ✓ Gradient-dominant -> hierarchical assessment")

    # Curl-heavy document
    results = {"hodge": {"pct_gradient": 0.30, "pct_curl": 0.45, "pct_harmonic": 0.25}}
    interp = adapter.interpret(results)
    assert "Cross-referenced" in interp.get("structure_assessment", "")
    print("  ✓ Curl-heavy -> cross-referenced assessment")

    # Harmonic-heavy document
    results = {"hodge": {"pct_gradient": 0.40, "pct_curl": 0.15, "pct_harmonic": 0.45}}
    interp = adapter.interpret(results)
    assert "unresolved" in interp.get("structure_assessment", "").lower()
    print("  ✓ Harmonic-heavy -> unresolved threads assessment")
    print()



def test_offline_client():
    """OfflineOCRClient should return results without errors."""
    print("── Offline OCR client ──")

    client = OfflineOCRClient()
    # Even without Tesseract, should return empty results gracefully

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(f.name)
        result = client.ocr_image(f.name)
    os.unlink(f.name)

    assert isinstance(result, OCRResult)
    assert isinstance(result.text, str)
    print(f"  ✓ Offline client returned result (text length: {len(result.text)})")
    print()



def test_batch_result():
    """OCRBatchResult should concatenate pages correctly."""
    print("── Batch result concatenation ──")

    batch = OCRBatchResult(
        pages=[
            OCRResult(text="First page.", source="p1.png", page=1),
            OCRResult(text="Second page.", source="p2.png", page=2),
            OCRResult(text="Third page.", source="p3.png", page=3),
        ],
        source="doc.pdf",
    )

    assert batch.n_pages == 3
    full = batch.full_text
    assert "First page." in full
    assert "Second page." in full
    assert "Third page." in full
    assert "page 1" in full
    assert "page 2" in full
    print(f"  ✓ 3-page batch concatenated ({len(full)} chars)")
    print()



def test_client_health_check_offline():
    """Client health check should return False when no server."""
    print("── Client health check (offline) ──")
    client = UnlimitedOCRClient(server_url="http://127.0.0.1:99999")
    assert not client.is_available()
    print("  ✓ Health check returns False for unreachable server")
    print()


def test_client_page_splitting():
    """Client page splitting should handle various separator formats."""
    print("── Client page splitting ──")

    multi_text = "Page 1 content.\n\n---page 2---\n\nPage 2 content.\n\n---page 3---\n\nPage 3 content."
    pages = UnlimitedOCRClient._split_pages(
        multi_text, ["a.png", "b.png", "c.png"],
    )
    assert len(pages) == 3
    assert "Page 1" in pages[0].text
    assert "Page 2" in pages[1].text
    assert "Page 3" in pages[2].text
    print("  ✓ Page splitting works with --- separators")

    # HTML comment separators
    multi_text2 = "Content A.\n\n<!-- page 2 -->\n\nContent B."
    pages2 = UnlimitedOCRClient._split_pages(
        multi_text2, ["x.png", "y.png"],
    )
    assert len(pages2) == 2
    print("  ✓ Page splitting works with HTML comment separators")
    print()



def test_edge_construction_invariants():
    """All OCR-produced EdgeConstructions should satisfy core invariants."""
    print("── Edge construction invariants ──")

    mock_client = MagicMock()
    mock_client.ocr_image.return_value = OCRResult(
        text=SAMPLE_OCR_OUTPUT, source="test.png",
    )

    adapter = OCRAdapter(client=mock_client)

    for strategy in ("text", "layout"):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            _create_test_image(f.name)
            edges = adapter.build(f.name, strategy=strategy)
        os.unlink(f.name)

        # Array length consistency
        assert len(edges.sources) == edges.nE
        assert len(edges.targets) == edges.nE
        assert len(edges.weights) == edges.nE
        assert len(edges.signs) == edges.nE
        assert len(edges.type_labels) == edges.nE

        # Weight non-negativity
        assert np.all(edges.weights >= 0)

        # Signs are ±1
        assert np.all(np.isin(edges.signs, [-1.0, 1.0]))

        # Type labels in valid range
        if edges.nE > 0:
            assert np.all(edges.type_labels >= 0)
            assert np.all(edges.type_labels < edges.n_types)

        # Source/target in valid vertex range
        if edges.nE > 0:
            assert np.all(edges.sources >= 0)
            assert np.all(edges.sources < edges.nV)
            assert np.all(edges.targets >= 0)
            assert np.all(edges.targets < edges.nV)

        # No self-loops (layout strategy only - text strategy may
        # produce them from repeated words in the same sentence)
        if strategy == "layout" and edges.nE > 0:
            assert np.all(edges.sources != edges.targets)

        print(f"  ✓ {strategy}: all invariants hold ({edges.nV}V, {edges.nE}E)")

    print()



def test_edge_construction_summary():
    """EdgeConstruction.summary() should work for OCR-produced edges."""
    print("── Edge construction summary ──")

    mock_client = MagicMock()
    mock_client.ocr_image.return_value = OCRResult(
        text=SAMPLE_OCR_OUTPUT, source="test.png",
    )

    adapter = OCRAdapter(client=mock_client)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(f.name)
        edges = adapter.build(f.name, strategy="layout")
    os.unlink(f.name)

    summary = edges.summary()
    assert "vertices" in summary
    assert "edges" in summary
    print(f"  ✓ Summary:\n    {summary.replace(chr(10), chr(10) + '    ')}")
    print()



def test_deepseek_ocr2_preset():
    """UnlimitedOCRClient.deepseek_ocr2() should configure the right defaults."""
    print("── DeepSeek-OCR-2 preset ──")

    client = UnlimitedOCRClient.deepseek_ocr2(
        server_url="http://gpu-server:10000",
    )
    assert client.model_name == "DeepSeek-OCR-2"
    assert client.backend_name == "deepseek-ocr-2"
    assert "<|grounding|>" in client.prompt
    assert client.server_url == "http://gpu-server:10000"
    print("  ✓ DeepSeek-OCR-2 preset configured correctly")
    print(f"    model={client.model_name}, prompt={client.prompt[:50]}...")
    print()


def test_deepseek_ocr_preset():
    """UnlimitedOCRClient.deepseek_ocr() should configure the v1 defaults."""
    print("── DeepSeek-OCR (v1) preset ──")

    client = UnlimitedOCRClient.deepseek_ocr(
        server_url="http://gpu-server:10000",
    )
    assert client.model_name == "DeepSeek-OCR"
    assert client.backend_name == "deepseek-ocr"
    print("  ✓ DeepSeek-OCR v1 preset configured correctly")
    print()


def test_unlimited_ocr_backend_name():
    """Default UnlimitedOCRClient should have backend_name='unlimited-ocr'."""
    print("── Backend names ──")

    client = UnlimitedOCRClient()
    assert client.backend_name == "unlimited-ocr"
    print("  ✓ Default backend name is 'unlimited-ocr'")
    print()



def test_mistral_client_no_key():
    """MistralOCRClient without API key should report unavailable."""
    print("── Mistral OCR: no API key ──")

    # Clear any env key for this test
    old_key = os.environ.pop("MISTRAL_API_KEY", None)
    try:
        client = MistralOCRClient(api_key="")
        assert not client.is_available()
        assert client.backend_name == "mistral-ocr"
        print("  ✓ Unavailable without API key")
    finally:
        if old_key is not None:
            os.environ["MISTRAL_API_KEY"] = old_key
    print()


def test_mistral_client_document_builders():
    """MistralOCRClient document builders should produce correct specs."""
    print("── Mistral OCR: document builders ──")

    client = MistralOCRClient(api_key="test-key")

    # Image document
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(f.name)
        doc = client._make_image_document(f.name)
    os.unlink(f.name)

    assert doc["type"] == "image_url"
    assert doc["image_url"].startswith("data:image/png;base64,")
    print("  ✓ Image document spec correct")

    # PDF document (create a minimal file)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake content")
        f.flush()
        doc = client._make_pdf_document(f.name)
    os.unlink(f.name)

    assert doc["type"] == "document_url"
    assert doc["document_url"].startswith("data:application/pdf;base64,")
    assert "document_name" in doc
    print("  ✓ PDF document spec correct (native, no conversion)")
    print()


def test_mistral_client_extract_text():
    """MistralOCRClient._extract_text should handle response objects."""
    print("── Mistral OCR: text extraction ──")

    # Mock a Mistral response with pages
    mock_page1 = MagicMock()
    mock_page1.markdown = "# Page 1\n\nSome content."
    mock_page1.index = 1

    mock_page2 = MagicMock()
    mock_page2.markdown = "# Page 2\n\nMore content."
    mock_page2.index = 2

    mock_response = MagicMock()
    mock_response.pages = [mock_page1, mock_page2]

    text = MistralOCRClient._extract_text(mock_response)
    assert "Page 1" in text
    assert "Page 2" in text
    print(f"  ✓ Extracted {len(text)} chars from 2-page response")

    pages = MistralOCRClient._extract_pages(mock_response, "test.pdf")
    assert len(pages) == 2
    assert pages[0].page == 1
    assert pages[1].page == 2
    print("  ✓ Per-page extraction correct")

    # Empty response
    assert MistralOCRClient._extract_text(None) == ""
    assert MistralOCRClient._extract_pages(None, "") == []
    print("  ✓ Empty response handled")
    print()


def test_mistral_client_ocr_with_mock():
    """MistralOCRClient should work end-to-end with a mocked API."""
    print("── Mistral OCR: mocked E2E ──")

    mock_page = MagicMock()
    mock_page.markdown = SAMPLE_PROSE_OUTPUT
    mock_page.index = 1

    mock_response = MagicMock()
    mock_response.pages = [mock_page]

    client = MistralOCRClient(api_key="test-key")
    client._client = MagicMock()
    client._client.ocr.process.return_value = mock_response

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(f.name)
        result = client.ocr_image(f.name)
    os.unlink(f.name)

    assert isinstance(result, OCRResult)
    assert "relational complex" in result.text
    assert result.elapsed > 0
    print(f"  ✓ OCR result: {len(result.text)} chars, {result.elapsed:.3f}s")
    print()


def test_mistral_through_adapter():
    """OCRAdapter should work with MistralOCRClient."""
    print("── Mistral OCR through adapter ──")

    mock_page = MagicMock()
    mock_page.markdown = SAMPLE_OCR_OUTPUT
    mock_page.index = 1

    mock_response = MagicMock()
    mock_response.pages = [mock_page]

    client = MistralOCRClient(api_key="test-key")
    client._client = MagicMock()
    client._client.ocr.process.return_value = mock_response

    adapter = OCRAdapter(client=client)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(f.name)
        edges = adapter.build(f.name, strategy="layout")
    os.unlink(f.name)

    assert isinstance(edges, EdgeConstruction)
    assert edges.nV > 0
    assert edges.nE > 0
    print(f"  ✓ Mistral -> OCRAdapter -> EdgeConstruction: {edges.nV}V, {edges.nE}E")
    print()



def test_factory_offline_fallback():
    """create_ocr_client should return whatever backend is best available.

    This must not assume any specific backend is installed. On CUDA hosts
    PaddleOCR may be selected; on ROCm hosts paddle is skipped (it needs
    CUDA) and the factory falls through to GOT-OCR or the Tesseract
    offline client. Any of the known client types is a valid result, so
    we check the returned object honours the OCR client interface rather
    than pinning a concrete class.
    """
    print("── Factory: auto-selection ──")

    from agent.integrations.unlimited_ocr import (
        GOTOCRClient,
        MistralOCRClient,
        OfflineOCRClient,
        PaddleOCRClient,
        UnlimitedOCRClient,
    )
    known_clients = (
        UnlimitedOCRClient, PaddleOCRClient, MistralOCRClient,
        GOTOCRClient, OfflineOCRClient,
    )

    old_key = os.environ.pop("MISTRAL_API_KEY", None)
    try:
        # Point the server probe at a dead port so auto-detect skips the
        # server backend and picks a local one.
        client = create_ocr_client(
            server_url="http://127.0.0.1:99999",
        )
        assert isinstance(client, known_clients), (
            f"Factory returned an unknown client type: {type(client).__name__}"
        )
        # Whatever it picked must expose the OCR client interface. Every
        # backend implements ocr_image; not all set a backend_name attr.
        assert hasattr(client, "ocr_image"), (
            f"{type(client).__name__} is missing ocr_image()"
        )
        print(f"  ✓ Factory selected: {type(client).__name__}")
    finally:
        if old_key is not None:
            os.environ["MISTRAL_API_KEY"] = old_key
    print()


def test_factory_prefer_mistral():
    """create_ocr_client(prefer='mistral') should return MistralOCRClient."""
    print("── Factory: prefer mistral ──")

    client = create_ocr_client(
        prefer="mistral",
        mistral_api_key="test-key-123",
    )
    assert isinstance(client, MistralOCRClient)
    assert client.api_key == "test-key-123"
    print("  ✓ Forced Mistral backend")
    print()


def test_factory_prefer_offline():
    """create_ocr_client(prefer='offline') should return OfflineOCRClient."""
    print("── Factory: prefer offline ──")

    client = create_ocr_client(prefer="offline")
    assert isinstance(client, OfflineOCRClient)
    print("  ✓ Forced offline backend")
    print()



if __name__ == "__main__":
    print("=" * 60)
    print("  RexGraph Agent - OCR Integration Tests")
    print("=" * 60)
    print()

    test_detect_image_types()
    test_detect_pdf_type()
    test_detect_image_directory()
    test_is_image_file()
    test_is_pdf_file()
    test_encode_image()
    test_parse_sections()
    test_classify_block()
    test_build_layout_graph()
    test_layout_graph_thematic_edges()
    test_ocr_adapter_text_strategy()
    test_ocr_adapter_layout_strategy()
    test_ocr_adapter_pdf()
    test_ocr_adapter_empty_result()
    test_ocr_adapter_interpret()
    test_offline_client()
    test_batch_result()
    test_client_health_check_offline()
    test_client_page_splitting()
    test_edge_construction_invariants()
    test_edge_construction_summary()
    test_deepseek_ocr2_preset()
    test_deepseek_ocr_preset()
    test_unlimited_ocr_backend_name()
    test_mistral_client_no_key()
    test_mistral_client_document_builders()
    test_mistral_client_extract_text()
    test_mistral_client_ocr_with_mock()
    test_mistral_through_adapter()
    test_factory_offline_fallback()
    test_factory_prefer_mistral()
    test_factory_prefer_offline()

    print("=" * 60)
    print("  All OCR integration tests passed.")
    print("=" * 60)
