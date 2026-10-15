"""Synthetic vendor responses preserve recognized text and report inference failures."""
from types import SimpleNamespace
import sys

import pytest
from agent.integrations.unlimited_ocr import PaddleOCRClient


def test_modern_dictionary_results_preserve_recognized_text():
    client = PaddleOCRClient(use_structure=False)
    client._engine = SimpleNamespace(predict=lambda path: [{"rec_texts": ["first", "second"]}])
    assert client.ocr_image("synthetic.png").text == "first\nsecond"


def test_structured_markdown_dictionary_returns_text():
    client = PaddleOCRClient()
    client._engine = SimpleNamespace(predict=lambda path: [SimpleNamespace(
        markdown={"markdown_texts": "# Synthetic document", "markdown_images": {}})])
    assert client.ocr_image("synthetic.png").text == "# Synthetic document"


def test_legacy_plain_ocr_preserves_lines():
    client = PaddleOCRClient()
    client._engine = SimpleNamespace(ocr=lambda path, cls: [[
        [[], ("first", 0.9)], [[], ("second", 0.8)]]])
    assert client.ocr_image("synthetic.png").text == "first\nsecond"


def test_inference_failure_is_not_a_successful_empty_document():
    def fail(path):
        raise OSError("synthetic inference failure")
    client = PaddleOCRClient()
    client._engine = SimpleNamespace(predict=fail)
    with pytest.raises(RuntimeError, match="inference failure"):
        client.ocr_image("synthetic.png")


def test_missing_structured_api_uses_legacy_plain_engine(monkeypatch):
    class Legacy:
        def __init__(self, **kwargs):
            assert kwargs["use_gpu"] is False
        def ocr(self, path, cls):
            return [[ [[], ("legacy", 0.9)] ]]
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=Legacy))
    assert PaddleOCRClient().ocr_image("synthetic.png").text == "legacy"


def test_structured_initialization_failure_uses_modern_plain_engine(monkeypatch):
    class Plain:
        def __init__(self, **kwargs):
            assert kwargs["device"] == "cpu"
        def predict(self, path):
            return [{"rec_texts": ["fallback"]}]
    def fail(**kwargs):
        raise RuntimeError("synthetic structured failure")
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=Plain, PPStructureV3=fail))
    assert PaddleOCRClient().ocr_image("synthetic.png").text == "fallback"
