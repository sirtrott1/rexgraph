"""PDF conversion fails on partial output and removes temporary pages on failure."""
import multiprocessing
from pathlib import Path
import subprocess
import sys

import pytest

from agent.integrations import unlimited_ocr as ocr


@pytest.fixture
def render_directory(tmp_path, monkeypatch):
    pytest.importorskip("fitz")
    root = tmp_path / "rendered"
    root.mkdir()
    monkeypatch.setattr(ocr.tempfile, "mkdtemp", lambda **kwargs: str(root))
    return root


def test_worker_failure_does_not_return_the_pages_written_before_failure(render_directory, monkeypatch):
    def fail(pdf, out_dir, dpi):
        (Path(out_dir) / "page_0001.png").write_bytes(b"synthetic")
        raise RuntimeError("second page failed")

    monkeypatch.setattr(ocr, "_run_pdf_conversion", fail)
    with pytest.raises(RuntimeError, match="second page failed"):
        ocr._pdf_to_images("synthetic.pdf")
    assert not render_directory.exists()


def test_interrupted_conversion_removes_temporary_pages(render_directory, monkeypatch):
    def interrupt(pdf, out_dir, dpi):
        (Path(out_dir) / "page_0001.png").write_bytes(b"synthetic")
        raise KeyboardInterrupt

    monkeypatch.setattr(ocr, "_run_pdf_conversion", interrupt)
    with pytest.raises(KeyboardInterrupt):
        ocr._pdf_to_images("synthetic.pdf")
    assert not render_directory.exists()


@pytest.mark.parametrize("count", [0, 2])
def test_empty_or_incomplete_output_is_rejected(render_directory, monkeypatch, count):
    def render(pdf, out_dir, dpi):
        if count:
            (Path(out_dir) / "page_0001.png").write_bytes(b"synthetic")
        return count

    monkeypatch.setattr(ocr, "_run_pdf_conversion", render)
    with pytest.raises(RuntimeError):
        ocr._pdf_to_images("synthetic.pdf")
    assert not render_directory.exists()


def test_timeout_terminates_the_worker_before_removing_output(render_directory, monkeypatch):
    class Worker:
        alive = True
        closed = False
        waits = []

        def start(self):
            (render_directory / "page_0001.png").write_bytes(b"synthetic")

        def join(self, timeout=None):
            self.waits.append(timeout)

        def is_alive(self):
            return self.alive

        def terminate(self):
            assert render_directory.exists()
            self.alive = False

        def close(self):
            self.closed = True

    worker = Worker()
    pipe = multiprocessing.Pipe

    class Context:
        Pipe = staticmethod(pipe)

        def Process(self, **kwargs):
            return worker

    monkeypatch.setattr(multiprocessing, "get_context", lambda *args: Context())
    monkeypatch.setattr(ocr, "PDF_CONVERSION_TIMEOUT", 0.01)
    with pytest.raises(RuntimeError, match="exceeded"):
        ocr._pdf_to_images("synthetic.pdf")
    assert worker.waits == [0.01, 5]
    assert worker.closed and not worker.alive
    assert not render_directory.exists()


def test_real_pdf_renders_all_pages_in_a_spawned_process(tmp_path):
    pytest.importorskip("fitz")
    script = tmp_path / "convert.py"
    script.write_text('''from pathlib import Path
import sys
from agent.integrations.unlimited_ocr import _pdf_to_images, _cleanup_pdf_temp

if __name__ == "__main__":
    import fitz
    source = Path(sys.argv[1])
    with fitz.open() as doc:
        for text in ("first page", "second page"):
            page = doc.new_page()
            page.insert_text((40, 40), text)
        doc.save(source)
    images = _pdf_to_images(str(source), dpi=72)
    try:
        assert [Path(image).name for image in images] == ["page_0001.png", "page_0002.png"]
        assert all(Path(image).read_bytes().startswith(b"\\x89PNG") for image in images)
    finally:
        _cleanup_pdf_temp(str(Path(images[0]).parent))
    assert not Path(images[0]).parent.exists()
    try:
        _pdf_to_images(str(source.with_name("missing.pdf")))
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing document was accepted")
''')
    result = subprocess.run([sys.executable, "-I", str(script), str(tmp_path / "synthetic.pdf")],
                            text=True, capture_output=True, timeout=60, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
