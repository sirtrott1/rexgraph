#!/usr/bin/env python3
"""Report prose punctuation in tracked documentation and source comments.

This reader never edits files. Code, identifiers, links, literal examples and
mathematical notation retain their syntax. A clean report is a punctuation
check, not a review of mathematical claims or prose quality.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
from pathlib import Path
import re
import subprocess
import textwrap
import tokenize


ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".md", ".py", ".pyx", ".pxd", ".sh", ".toml", ".yml", ".yaml",
            ".c", ".h", ".cpp", ".cu", ".cuh", ".js", ".jsx", ".css", ".html"}
DECORATION = re.compile(r"^\s*(?:(?:#+|//+|/\*+|\*)\s*)?(?:[-=*_─━][\s\-=*_─━]*)(?:\*/)?\s*$")
LABELED_DECORATION = re.compile(r"^\s*(?:#+|//+|/\*+|\*)[^\n]*\w[^\n]*(?:#{4,}|={4,}|-{4,})\s*(?:\*/)?\s*$")
COMPOUND = re.compile(r"\b(?:[A-Za-z]{2,}(?:-(?:[A-Za-z]{2,}|\d+))+|\d+(?:-[A-Za-z]{2,})+)\b")
PROTECTED = re.compile(r"(`+)(.*?)(?<!`)\1|\$[^$\n]+\$|https?://\S+|\]\([^)]*\)")
TECHNICAL = re.compile(r"\b[\w.-]+[/\\][\w./\\-]+|\b[\w-]+\.[A-Za-z0-9_]+\b|\b\w*_\w*\b|--[A-Za-z][\w-]*|\b(?:rexgraph-(?:rcql|rcdb|system|agent)|meson-python|conda-forge|pkg-config|(?:SHA|sha|UTF|utf|AES|aes|OCR)-\d+|(?:nV|nE|nF|nG|nhats|dim)-\d+)\b")


def regions(path, text):
    """Return prose with original line numbers, excluding executable strings."""
    lines = text.splitlines()
    if path.suffix == ".md":
        yield from enumerate(lines, 1)
        return
    if path.suffix in {".py", ".pyx", ".pxd"}:
        docs = set()
        if path.suffix == ".py":
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                        docs.add((first.value.lineno, first.value.col_offset))
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT:
                    yield token.start[0], token.string
                elif token.type == tokenize.STRING and (
                        token.start in docs or (path.suffix != ".py" and token.string.startswith(('"""', "'''")))):
                    parts = token.string.splitlines()
                    parts = parts[:1] + textwrap.dedent("\n".join(parts[1:])).splitlines()
                    for offset, line in enumerate(parts):
                        yield token.start[0] + offset, line.strip('"\'')
        except (tokenize.TokenError, IndentationError) as exc:
            raise ValueError(f"cannot scan {path}: {exc}") from exc
        return
    if path.suffix in {".sh", ".toml", ".yml", ".yaml"}:
        yield from ((i, line.lstrip()) for i, line in enumerate(lines, 1) if line.lstrip().startswith("#"))
    if path.suffix in {".c", ".h", ".cpp", ".cu", ".cuh", ".js", ".jsx", ".css", ".html"}:
        if ".min." in path.name:
            return
        tokens = re.compile(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|//[^\n]*|/\*[\s\S]*?\*/|<!--[\s\S]*?-->''')
        for token in tokens.finditer(text):
            if not token.group().startswith(("//", "/*", "<!--")):
                continue
            number = text.count("\n", 0, token.start()) + 1
            for offset, line in enumerate(token.group().splitlines()):
                yield number + offset, line


def prose_regions(path, text):
    """Mask code spans across line breaks while retaining character positions."""
    fence = None
    inline = None
    previous = 0
    for number, line in regions(path, text):
        if number > previous + 1:
            inline = None
        previous = number
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            tag = marker.group(1)[0]
            fence = None if fence == tag else tag
            continue
        if fence or line.startswith("    ") or stripped.startswith((">>>", "...")):
            continue
        pieces, last = [], 0
        for token in re.finditer(r"`+", line):
            part = line[last:token.start()]
            pieces.append(part if inline is None else " " * len(part))
            ticks = len(token.group())
            inline = None if inline == ticks else ticks if inline is None else inline
            pieces.append(" " * ticks)
            last = token.end()
        part = line[last:]
        pieces.append(part if inline is None else " " * len(part))
        prose = "".join(pieces)
        prose = PROTECTED.sub(lambda m: " " * len(m.group()), prose)
        prose = TECHNICAL.sub(lambda m: " " * len(m.group()), prose)
        yield number, line, prose


def findings(path, text):
    for number, line, prose in prose_regions(path, text):
        if DECORATION.match(line) or LABELED_DECORATION.match(line):
            yield number, "divider", line.strip()
        if "\u2014" in prose:
            yield number, "em dash", line.strip()
        for match in COMPOUND.finditer(prose):
            yield number, "prose hyphen", match.group()


def tracked_paths():
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / name for name in output.decode().split("\0") if name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = []
    for path in args.paths or tracked_paths():
        if path.suffix not in SUFFIXES:
            continue
        for number, kind, context in findings(path, path.read_text()):
            result.append({"path": str(path), "line": number, "kind": kind, "context": context})
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for item in result:
            print(f"{item['path']}:{item['line']}: {item['kind']}: {item['context']}")
        print(f"{len(result)} prose punctuation findings")
    return bool(result)


if __name__ == "__main__":
    raise SystemExit(main())
