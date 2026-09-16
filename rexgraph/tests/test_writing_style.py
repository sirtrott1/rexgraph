"""The prose checker preserves code, links and mathematical notation."""
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_writing.py"
SPEC = importlib.util.spec_from_file_location("writing_check", SCRIPT)
WRITING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRITING)


def test_punctuation_is_reported():
    text = "A matrix" + "-free method.\n" + "\u2014" + "\n" + "-" * 3 + "\n"
    assert {kind for _, kind, _ in WRITING.findings(Path("README.md"), text)} == {
        "prose hyphen", "em dash", "divider"}


def test_partial_and_native_comment_dividers_are_reported():
    for line in ("#### -", "# " + "=" * 4, "// " + "-" * 3, "/* " + "=" * 4 + " */"):
        assert WRITING.DECORATION.match(line)
    assert list(WRITING.findings(Path("README.md"), "- `some_method()`\n")) == []
    for line in ("#### a heading " + "#" * 8, "# " + "-" * 8 + " a heading " + "-" * 8):
        assert list(WRITING.findings(Path("module.py"), line))[0][1] == "divider"
        assert list(WRITING.findings(Path("install.sh"), "    " + line))[0][1] == "divider"


def test_numbered_prose_preserves_encoding_and_matrix_identifiers():
    issues = list(WRITING.findings(Path("README.md"), "A grade" + "-1 field over nV-1 cells in UTF-8.\n"))
    assert issues == [(1, "prose hyphen", "grade" + "-1")]


def test_code_and_multiline_spans_remain_literal():
    text = ('`No matching\ndistribution found for rexgraph-rcdb`\n'
            '```sh\npython --no-build-isolation\n```\n'
            '[the source](https://example.org/some-project)\n'
            '$B_{k}-B_{k+1}$ and conda-forge.\n')
    assert list(WRITING.findings(Path("README.md"), text)) == []


def test_comments_and_docs_do_not_include_runtime_strings():
    text = ('def f():\n    """A matrix' + '-free method."""\n'
            '    return "pair-wise"  # source' + '-bound value\n')
    found = list(WRITING.findings(Path("module.py"), text))
    assert [item[0] for item in found] == [2, 3]
    assert all("pair-wise" != item[2] for item in found)


def test_the_check_does_not_rewrite_files(tmp_path):
    path = tmp_path / "README.md"
    original = "An exact" + "-value example.\n"
    path.write_text(original)
    assert list(WRITING.findings(path, path.read_text()))
    assert path.read_text() == original


def test_native_and_ui_comments_preserve_strings():
    text = 'const char *url = "https://host/matrix-free"; // matrix' + '-free action\n'
    for suffix in ("cpp", "js", "jsx", "cu"):
        assert list(WRITING.findings(Path("example." + suffix), text)) == [
            (1, "prose hyphen", "matrix" + "-free")]
