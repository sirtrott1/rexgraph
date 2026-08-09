"""
Table detection for OCR output.

OCR backends flatten tables into text.  When that text later enters the
pipeline through the plain :class:`TextAdapter`, column headers become
ordinary word tokens, so the *same data* yields a different complex
depending on whether it arrived as a file or as a scan (audit item 2.2).

This module recovers tabular structure from OCR text and parses it back
into a :class:`pandas.DataFrame`, so it can be routed through the same
feature/edge adapters a native CSV would use - giving column headers
back their role as vertex labels.

The detector is deliberately conservative: it only reports a table when
several consecutive lines share a consistent column count under a single
delimiter model (explicit delimiter or aligned whitespace).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # pandas is imported where it is used, not at load
    import pandas as pd

logger = logging.getLogger(__name__)

_DELIMS = ["\t", "|", ",", ";"]
_MIN_ROWS = 3          # header + at least 2 data rows
_MIN_COLS = 2


def _split_delim(line: str, delim: str) -> list[str]:
    parts = [c.strip() for c in line.split(delim)]
    # Drop empty leading/trailing cells produced by border pipes.
    if delim == "|":
        parts = [c for c in parts if c != ""]
    return parts


def _split_whitespace(line: str) -> list[str]:
    # Two-or-more spaces are treated as a column boundary; single spaces
    # inside a cell are preserved.
    return [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]


def _looks_like_rule(line: str) -> bool:
    """Markdown / ASCII horizontal rules such as ``---|---`` or ``====``."""
    s = line.strip()
    return bool(s) and bool(re.fullmatch(r"[-=+_|:\s]+", s)) and (
        "-" in s or "=" in s or "_" in s
    )


def _consistent_block(lines: list[str], splitter) -> list[list[str]] | None:
    """Return the parsed rows if ``lines`` form a consistent column block."""
    rows = []
    counts = []
    for ln in lines:
        if _looks_like_rule(ln):
            continue
        cells = splitter(ln)
        if len(cells) < _MIN_COLS:
            return None
        rows.append(cells)
        counts.append(len(cells))
    if len(rows) < _MIN_ROWS:
        return None
    # Require the modal column count to dominate.
    mode = max(set(counts), key=counts.count)
    if counts.count(mode) < 0.8 * len(counts):
        return None
    # Normalise every row to the modal width.
    norm = []
    for r in rows:
        if len(r) > mode:
            r = r[: mode - 1] + [" ".join(r[mode - 1:])]
        elif len(r) < mode:
            r = r + [""] * (mode - len(r))
        norm.append(r)
    return norm


def _rows_to_frame(rows: list[list[str]]):
    import pandas as pd

    header = rows[0]
    # De-duplicate / fill blank headers.
    seen = {}
    cols = []
    for i, hraw in enumerate(header):
        hh = hraw.strip() or f"col_{i}"
        if hh in seen:
            seen[hh] += 1
            hh = f"{hh}_{seen[hh]}"
        else:
            seen[hh] = 0
        cols.append(hh)
    data = rows[1:]
    df = pd.DataFrame(data, columns=cols)
    # Coerce numeric columns where possible.
    for c in df.columns:
        coerced = pd.to_numeric(
            df[c].str.replace(",", "", regex=False), errors="coerce"
        )
        if coerced.notna().mean() >= 0.8:
            df[c] = coerced
    return df


def detect_tables(text: str, min_rows: int = _MIN_ROWS) -> list[pd.DataFrame]:
    """Extract tables from OCR text as DataFrames (best-effort).

    Scans for maximal runs of non-empty lines and tries each delimiter
    model (explicit delimiters first, then aligned whitespace), keeping
    the parse that yields the most columns.
    """
    if not text or not text.strip():
        return []

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        logger.warning("pandas unavailable; table detection disabled.")
        return []

    frames = []
    # Split into blocks separated by blank lines.
    for block in re.split(r"\n\s*\n", text):
        block_lines = [ln for ln in block.split("\n") if ln.strip()]
        if len(block_lines) < min_rows:
            continue

        best = None
        best_cols = 0
        for delim in _DELIMS:
            if sum(delim in ln for ln in block_lines) < min_rows - 1:
                continue
            parsed = _consistent_block(
                block_lines, lambda l, d=delim: _split_delim(l, d)
            )
            if parsed and len(parsed[0]) > best_cols:
                best, best_cols = parsed, len(parsed[0])

        if best is None:
            parsed = _consistent_block(block_lines, _split_whitespace)
            if parsed:
                best, best_cols = parsed, len(parsed[0])

        if best is not None:
            try:
                frames.append(_rows_to_frame(best))
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Table parse failed: %s", e)

    return frames


def text_has_table(text: str) -> bool:
    """Cheap boolean check used to decide whether to attempt parsing."""
    return len(detect_tables(text)) > 0
