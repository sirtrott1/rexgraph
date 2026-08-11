"""
Auto-rex: automatic relational complex construction from any supported input.

This is the core of the agent. Data goes in, a typed RexGraph with faces
and voids comes out. The math configures itself.

    rex = auto_rex("patients.csv")
    results = auto_analyze("patients.csv")

That's the entire API for going from raw data to complete structural analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .adapters import EdgeConstruction
from .adapters.correlation import AdjacencyAdapter, CorrelationAdapter
from .adapters.edge_list import EdgeListAdapter
from .adapters.feature_matrix import FeatureMatrixAdapter

# CSV cell tokens treated as missing (mirrors pandas.read_csv default na_values, lowercased)
_NA_TOKENS = frozenset({"", "na", "n/a", "null", "none", "#n/a", "nan"})


def _can_be_path(data: Any) -> bool:
    """Whether `data` may be probed against the filesystem.

    A `Path` always may. A `str` may only if it is short enough to be a filename and
    carries no newline: `Path(text).is_file()` on a long string raises OSError
    ENAMETOOLONG rather than returning False, so probing raw document text with it
    turns a text input into a crash. `detect_input_type` has always applied this
    guard; the other probe sites did not.
    """
    if isinstance(data, Path):
        return True
    return isinstance(data, str) and len(data) < 256 and "\n" not in data


def _is_missing_cell(v) -> bool:
    return str(v).strip().lower() in _NA_TOKENS


def check_analysis_size(nV: int, nE: int) -> None:
    """Fast-fail before the core's face-finding / boundary kernels run on a graph
    too large for reliable construction/analysis (segfault / OOM risk). Shared by
    the edge-construction chokepoint and the schema/ontology/lineage builders so
    every RexGraph construction path is capped. Tune/disable with
    REXGRAPH_MAX_ANALYSIS_NODES / REXGRAPH_MAX_ANALYSIS_EDGES (0 disables).
    Raises ValueError when exceeded."""
    import os as _os
    try:
        _max_n = int(_os.environ.get("REXGRAPH_MAX_ANALYSIS_NODES", "200000"))
        _max_e = int(_os.environ.get("REXGRAPH_MAX_ANALYSIS_EDGES", "1000000"))
    except ValueError:
        _max_n, _max_e = 200000, 1000000
    if (_max_n > 0 and nV > _max_n) or (_max_e > 0 and nE > _max_e):
        raise ValueError(
            f"Graph too large for reliable construction/analysis: {nV} nodes / "
            f"{nE} edges exceeds the {_max_n}-node / {_max_e}-edge limit. Reduce or "
            f"sparsify the graph, or raise REXGRAPH_MAX_ANALYSIS_NODES / "
            f"REXGRAPH_MAX_ANALYSIS_EDGES (larger graphs may exhaust memory).")


def _is_dataframe(obj) -> bool:
    """True if obj is a pandas DataFrame, without importing pandas."""
    t = type(obj)
    return t.__name__ == "DataFrame" and t.__module__.split(".")[0] == "pandas"


# Input type detection
#: extensions whose absence means a missing file rather than prose. Only the
#: ambiguous ones: a .pdf or .png that is not there fails when its reader opens it,
#: and classifying by name is what lets a caller ask what a name would be read as.
_FILE_SUFFIXES = frozenset({".txt"})


def _refuse_if_missing_file(data, p: Path, suffix: str) -> None:
    """Raise when a string names a file that is not there.

    `source` is documented as a path *or* raw text, and the ambiguity was resolved
    in favour of text: a mistyped path was tokenised into a co-occurrence complex, so
    `/data/nope.txt` produced a plausible nV, nE and kappa over the words of the path.
    A wrong answer that looks right is worse than no answer.

    Only the ambiguous suffixes are guarded, and a phrase that happens to end in one
    (`notes on chapter.txt`) is left alone: a path is either rooted in a separator or
    carries no whitespace at all.
    """
    if suffix not in _FILE_SUFFIXES:
        return
    s = str(data)
    if "\n" in s or len(s) >= 256:
        return
    looks_like_a_path = ("/" in s or "\\" in s) or not any(c.isspace() for c in s)
    if not looks_like_a_path:
        return
    try:
        if p.exists():
            return
    except OSError:
        return
    raise FileNotFoundError(
        f"{s!r} names a {suffix} file that does not exist. Pass an existing path, "
        f"or pass the text itself if this was meant as content.")


def detect_input_type(data: Any) -> str:
    """Inspect data and classify it for adapter dispatch.

    Returns one of:
        'rex_file'         - loadable rexgraph format (.rex, .zarr, .h5, .arrow)
        'edge_csv'         - CSV/TSV with source/target columns
        'feature_csv'      - CSV with many numeric columns (samples × features)
        'json'             - JSON (auto-detected by rexgraph.io)
        'feature_matrix'   - numpy array, rectangular (n_samples × n_features)
        'correlation'      - numpy array, square symmetric
        'adjacency'        - numpy array, square (possibly asymmetric)
        'dataframe'        - pandas DataFrame (re-dispatched by shape)
        'text'             - string of prose, or list of strings (paragraphs)
        'image'            - image file (.png, .jpg, .jpeg, .webp, .bmp, .tiff)
        'pdf'              - PDF document (.pdf)
        'image_dir'        - directory containing image files
    """
    # File paths and text strings
    if isinstance(data, (str, Path)):
        # First check if it looks like a path that exists or has a file suffix
        try:
            p = Path(data)
            suffix = p.suffix.lower()
        except (TypeError, ValueError, OSError):
            # Pure text that can't be parsed as a path
            return "text"

        # If it's an existing file or has a recognized extension, dispatch as file
        # registered scientific containers (SDF/PDB/FASTA/VCF/GFF/BED/h5ad/loom).
        # Checked before the generic suffixes so .h5ad is not read as .h5.
        try:
            from agent.adapters.formats import reader_for
            if reader_for(p) is not None:
                return "science_file"
        except Exception:
            pass
        if suffix in (".rex", ".zarr", ".h5", ".hdf5", ".arrow", ".parquet", ".safetensors"):
            return "rex_file"
        if suffix == ".json":
            return "json"
        if suffix in (".csv", ".tsv"):
            return _classify_csv(p)
        if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"):
            return "image"
        if suffix == ".pdf":
            return "pdf"
        if suffix == ".txt":
            # .txt is ambiguous: could be a tabular file or prose. Classify by
            # content, but a .txt has to LOOK tabular to be read as a table;
            # otherwise it is prose, which is what the extension usually means.
            if p.exists():
                try:
                    return _classify_csv(p, default="text")
                except Exception:
                    return "text"
            # A path that is not there is a missing file, not the prose of its own
            # name. Falling through to "text" tokenised the path into a complex, so
            # a mistyped path returned a plausible reading of text nobody supplied.
            _refuse_if_missing_file(data, p, suffix)
            return "text"

        # No recognized extension, but it could still be a file path if it exists
        if isinstance(data, Path) or (isinstance(data, str) and len(data) < 256
                                       and "\n" not in data and p.exists()):
            # Check if it's a 10X single-cell directory first (matrix.mtx +
            # features/genes.tsv). This must precede the image-dir check.
            if p.is_dir():
                try:
                    from agent.adapters.single_cell import is_10x_dir
                    if is_10x_dir(p):
                        return "single_cell"
                except Exception:
                    pass
            # Check if it's a directory of images
            if p.is_dir():
                from agent.integrations.unlimited_ocr import IMAGE_EXTENSIONS
                image_files = [
                    f for f in p.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                ]
                if image_files:
                    return "image_dir"
            raise ValueError(f"Unsupported file format: {suffix}")

        # String input that isn't a file path -> text
        # Heuristic: contains whitespace (multi-word) or newlines, treat as prose
        if isinstance(data, str) and (" " in data or "\n" in data or len(data) > 100):
            return "text"

        # Single-word string with no path: assume the user meant a file
        raise ValueError(
            f"String input not recognized as file or text: {data[:60]!r}"
        )

    # List of strings -> text (multiple paragraphs)
    if isinstance(data, list) and len(data) > 0 and all(isinstance(x, str) for x in data):
        return "text"

    # Numpy arrays
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        n, m = data.shape
        if n == m:
            # Square matrix, so check symmetry
            if np.allclose(data, data.T, atol=1e-10):
                # Could be correlation or adjacency.
                diag = np.diag(data)
                off_diag = data[np.triu_indices(n, k=1)]

                # Adjacency: values are mostly integers (0, 1, 2, ...)
                off_nonzero = off_diag[np.abs(off_diag) > 1e-10]
                if len(off_nonzero) > 0:
                    frac_integer = np.mean(np.abs(off_nonzero - np.round(off_nonzero)) < 1e-10)
                    if frac_integer > 0.9:
                        return "adjacency"

                # Correlation: diagonal ~ 1, values in [-1, 1], continuous
                if np.allclose(np.abs(diag), 1.0, atol=0.1):
                    if np.all(np.abs(off_diag) <= 1.0 + 1e-6):
                        return "correlation"

                # Diagonal ~ 0, values in [-1, 1] -> correlation with zeroed diagonal
                if np.allclose(diag, 0.0, atol=0.1):
                    if np.all(np.abs(off_diag) <= 1.0 + 1e-6):
                        # Check if continuous (not integer-like)
                        if len(off_nonzero) > 0 and frac_integer < 0.5:
                            return "correlation"

                return "adjacency"
            else:
                return "adjacency"
        else:
            return "feature_matrix"

    # Pandas DataFrame
    if _is_dataframe(data):
        return _classify_dataframe(data)

    raise TypeError(f"Unsupported input type: {type(data).__name__}")


def _classify_csv(path: Path, default: str = "edge_csv") -> str:
    """Peek at a delimited file to decide if it's an edge list, feature matrix, or
    text. `default` is what to return when nothing in the content settles it: an
    edge list for a .csv/.tsv, but prose for a .txt, where the extension is the
    stronger signal and a table is the exception."""
    import csv
    sep = '\t' if str(path).endswith('.tsv') else ','
    try:
        with open(path, newline="") as fh:
            reader = csv.reader(fh, delimiter=sep)
            header = next(reader)
            rows = []
            for i, r in enumerate(reader):
                if i >= 10:
                    break
                rows.append(r)
    except Exception:
        return "text"
    n_cols = len(header)
    if n_cols == 0:
        return "text"

    # per-column sampled string values (aligned to header width)
    def _col(j):
        return [r[j] for r in rows if j < len(r)]

    def _is_numeric_col(vals):
        vals = [v for v in vals if not _is_missing_cell(v)]
        if not vals:
            return False
        for v in vals:
            try:
                float(v)
            except (ValueError, TypeError):
                return False
        return True

    # Text-heavy columns (annotation tables, ontologies) -> treat as text
    for j in range(n_cols):
        col = _col(j)
        strvals = [str(v) for v in col if not _is_missing_cell(v)]
        if strvals and not _is_numeric_col(col):
            if sum(len(s) for s in strvals) / len(strvals) > 80:
                return "text"

    # A table has the same number of fields on every line. Ragged rows mean the
    # separator is punctuation inside prose, not a delimiter. Only applied where
    # prose is the fallback (a .txt), since a real CSV may legitimately be ragged.
    if default == "text" and rows and any(len(r) != n_cols for r in rows):
        return "text"

    # Edge list heuristics: 2-6 columns, first two look like node IDs
    if n_cols <= 6 and n_cols >= 2:
        col0, col1 = header[0].lower(), header[1].lower()
        edge_keywords = {"source", "src", "from", "head", "target", "tgt", "to", "tail", "dest"}
        if col0 in edge_keywords or col1 in edge_keywords:
            return "edge_csv"
        # first two columns non-numeric (node names) -> edge list. Requires actual
        # rows: with none, `_is_numeric_col([])` is False for every column and a
        # single line of prose reads as a two-node edge list.
        if rows and not _is_numeric_col(_col(0)) and not _is_numeric_col(_col(1)):
            return "edge_csv"

    # Feature matrix: many numeric columns
    numeric_cols = sum(1 for j in range(n_cols) if _is_numeric_col(_col(j)))
    if numeric_cols >= 5 and numeric_cols / n_cols > 0.5:
        return "feature_csv"

    return default


def _classify_dataframe(df):
    """Classify a DataFrame as edge list or feature matrix."""
    n_cols = len(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]

    # Small number of columns with string identifiers -> edge list
    if n_cols <= 6:
        non_numeric = n_cols - numeric_cols
        if non_numeric >= 2:
            return "edge_csv"

    # Many numeric columns -> feature matrix
    if numeric_cols >= 5:
        return "feature_matrix"

    return "edge_csv"


# Auto construction
def auto_rex(
    data: Any,
    *,
    threshold: str | float = "auto",
    typing: str = "auto",
    sign: str = "auto",
    face_selection: str = "none",
    feature_names: list[str] | None = None,
    vertex_labels: list[str] | None = None,
    **kwargs,
):
    """Build a typed RexGraph from any supported input.

    This is the main entry point. Hand it a file path, array, or
    DataFrame and get back a RexGraph with typed faces and voids.

    Parameters
    ----------
    data : str, Path, ndarray, or DataFrame
        The input data. See detect_input_type() for supported formats.
    threshold : 'auto' or float
        Correlation/similarity threshold for edge creation.
    typing : 'auto', 'spectral', 'column_family', or 'none'
        Edge typing strategy.
    sign : 'auto', 'correlation', or 'positive'
        Edge sign strategy. 'auto' chooses based on input type.
    face_selection : 'none' (default), 'auto', 'promote', 'hyper', 'typed',
        an int, or an iterable of ints
        A face is a filled cycle, solved from B1 c = 0 and of any gon.
        'none': no faces. Asserting a face asserts something is enclosed, so it
            is asked for rather than assumed.
        'auto': every gon the cycle basis actually contains.
        3, 5, [3, 6]: those gons only.
        'promote': fill every independent cycle, so beta_1 becomes 0.
        'hyper': close each branching relation against the relations bounding it.
        'typed': same-type triangles only. A filter over faces, not their
            definition, kept for a caller who wants exactly that.
    feature_names : list of str, optional
        Column names for feature matrices.
    vertex_labels : list of str, optional
        Vertex names for correlation/adjacency matrices.
    **kwargs
        Additional adapter-specific options.

    Returns
    -------
    RexGraph
        A relational complex with typed faces, voids, and ∂²=0 holds by construction.
    """

    # Fast path: caller already built the edges (e.g. an adapter that
    # runs outside auto_rex, such as OCR-layout, single-cell, or L-R
    # scoring).  Construct the rex directly so every adapter shares the
    # same face-selection and metadata handling.
    if isinstance(data, EdgeConstruction):
        return build_rex_from_edges(
            data,
            face_selection=face_selection,
            input_type=getattr(data, "input_type", "edge_construction"),
            threshold=threshold,
            typing=typing,
        )

    input_type = detect_input_type(data)

    # Direct load (already a rex format)
    if input_type == "rex_file":
        from rexgraph.io import load as _io_load
        return _io_load(str(data))

    if input_type == "json":
        from rexgraph.io.json_loader import load_json
        return load_json(str(data))

    # Resolve 'auto' sign based on input type
    if sign == "auto":
        if input_type in ("feature_matrix", "feature_csv", "correlation"):
            sign = "correlation"
        else:
            sign = "positive"

    # Build edges via the appropriate adapter
    if input_type in ("feature_matrix", "feature_csv"):
        try:
            edges = _build_feature_edges(data, input_type, threshold, typing,
                                          sign, feature_names, **kwargs)
        except Exception as _csv_err:
            edges = _fallback_text_or_raise(data, input_type, _csv_err, **kwargs)
    elif input_type == "edge_csv":
        try:
            edges = _build_edge_list_edges(data, **kwargs)
        except Exception as _csv_err:
            edges = _fallback_text_or_raise(data, input_type, _csv_err, **kwargs)
    elif input_type == "correlation":
        adapter = CorrelationAdapter()
        edges = adapter.build(data, labels=vertex_labels,
                              threshold=threshold, sign=sign)
    elif input_type == "adjacency":
        adapter = AdjacencyAdapter()
        edges = adapter.build(data, labels=vertex_labels)
    elif input_type == "text":
        from agent.adapters.text import TextAdapter
        if _can_be_path(data):
            p = Path(data)
            if p.is_file():
                data = p.read_text(encoding="utf-8", errors="replace")
        if isinstance(data, list):
            data = "\n\n".join(data)
        text_kwargs = {k: v for k, v in kwargs.items()
                        if k in ("window", "min_count", "max_vocab",
                                 "face_selection")}
        adapter = TextAdapter()
        edges = adapter.build(data, **text_kwargs)
    elif input_type == "science_file":
        from agent.adapters.formats import read, reader_for
        name = reader_for(data)
        # Pass through what the chosen reader actually declares. Filtering to ("k",)
        # made every other documented switch unreachable from the public entry
        # point, silently: load_pdb(backbone=False) had no effect through auto_rex.
        import inspect

        from agent.adapters.formats import reader_fn
        try:
            accepts = set(inspect.signature(reader_fn(name)).parameters)
        except Exception:
            accepts = {"k"}
        reader_kwargs = {k: v for k, v in kwargs.items() if k in accepts}
        out = read(data, **reader_kwargs)
        if name in ("h5ad", "loom"):
            # a matrix and its axis labels: exactly what the feature path already
            # takes, so it goes there rather than growing a second one.
            matrix, _obs, var = out
            from agent.adapters.feature_matrix import FeatureMatrixAdapter
            feat_kwargs = {k: v for k, v in kwargs.items()
                           if k in ("threshold", "typing", "sign", "n_clusters")}
            adapter = FeatureMatrixAdapter()
            # features are the vertices, so the VAR axis carries the names
            edges = adapter.build(matrix, feature_names=var or None, **feat_kwargs)
        else:
            edges = out
    elif input_type == "single_cell":
        from agent.adapters.single_cell import SingleCellAdapter
        sc_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("markers", "lr_pairs", "n_clusters",
                              "min_score")}
        adapter = SingleCellAdapter()
        edges = adapter.build(data, **sc_kwargs)
    elif input_type in ("image", "pdf", "image_dir"):
        from agent.adapters.ocr import OCRAdapter
        ocr_kwargs = {k: v for k, v in kwargs.items()
                       if k in ("strategy", "window", "min_count",
                                "max_vocab", "face_selection",
                                "ocr_prompt", "dpi")}
        adapter = OCRAdapter()
        edges = adapter.build(data, **ocr_kwargs)
    else:
        raise ValueError(f"Unhandled input type: {input_type}")

    # Construct the RexGraph via the shared helper so every code path
    # (auto_rex dispatch and out-of-band adapters) build faces and
    # metadata identically.
    return build_rex_from_edges(
        edges,
        face_selection=face_selection,
        input_type=input_type,
        threshold=threshold,
        typing=typing,
    )

#: The face rule every path in the agent layer builds under.
#:
#: A face is a filled cycle: whatever satisfies B1 c_f = 0 on the relations it
#: spans, of any gon. `rexgraph.faces` solves that exactly over the rationals and
#: reads the gon off the cycle basis, so nothing here hardcodes a triangle and
#: nothing conditions a face on edge type. Type and weight are attributes of a
#: relation; they weight the complex and say nothing about whether a cycle closes.
#:
#: One name, because a query complex is compared against document complexes and two
#: complexes built under different rules are not comparable.
FACE_RULE = "auto"

FACE_WORDS = ("none", "auto", "all", "promote", "hyper", "typed")


def attach_faces(rex, rule=FACE_RULE, *, type_labels=None):
    """Attach faces to `rex` under `rule`, returning the complex.

    `rule` is 'none', 'auto'/'all' (every gon the cycle basis contains), 'promote'
    (fill the basis, beta_1 -> 0), 'hyper' (close branching relations), 'typed' (a
    filter that keeps only same-type triangles, which is not what a face is and is
    kept for a caller who wants exactly that), or a gon: 3, 5, [3, 6].

    Every path that used to call `typed_face_selection` directly goes through here,
    so a document, a query and a chunk are built the same way rather than each
    deciding for itself.
    """
    if isinstance(rule, str) and rule not in FACE_WORDS:
        raise ValueError(
            f"face_selection={rule!r} is not a face rule. Use one of "
            f"{', '.join(FACE_WORDS)}, or a gon: 3, 5, [3, 6].")
    if rex is None or int(getattr(rex, "nE", 0)) == 0 or rule in ("none", None):
        return rex
    from rexgraph.faces import auto_hyperface, autoface
    if rule == "promote":
        return rex.promote()
    if rule == "hyper":
        auto_hyperface(rex)
        return rex
    if rule == "typed":
        if type_labels is None:
            raise ValueError("face_selection='typed' needs type_labels")
        return rex.typed_face_selection(type_labels)
    k = rule
    if k in ("auto", "all"):
        # the supports, not the coefficient vectors: "which gons are present" is a
        # question about |supp(c)|, and autoface re-solves the coefficients anyway.
        # Reading the basis here and again inside autoface cost one cycle-space solve
        # per distinct gon, so the basis is computed once and handed down.
        # which gons are present is a question about tree distances, not cycle vectors:
        # cycle_gons reads it off the spanning forest with an LCA per non-tree relation
        # and solves nothing. autoface then solves only the gons actually present.
        from rexgraph.faces import cycle_gons
        k = sorted(set(cycle_gons(rex)))
        autoface(rex, k=k)
        return rex
    autoface(rex, k=k)
    return rex



def build_rex_from_edges(
    edges: EdgeConstruction,
    *,
    face_selection: str = "none",
    input_type: str = "edge_construction",
    threshold: str | float = "auto",
    typing: str = "auto",
):
    """Construct a RexGraph from an EdgeConstruction.

    This is the single place where an EdgeConstruction becomes a
    RexGraph, so adapters that run outside :func:`auto_rex` (OCR-layout,
    single-cell, L-R scoring) get the same face selection, weight/sign
    handling and ``_agent_meta`` attachment.

    Mirrors ``rexgraph.io.csv_loader.GraphData.to_rex()``: ``w_E`` is the
    magnitude only, signs are passed separately.
    """
    from rexgraph.graph import RexGraph

    # Construction guard. The core's face-finding / boundary kernels can segfault
    # or exhaust memory on large, dense graphs (a real core limitation). Fail fast
    # and clearly HERE, before the C code runs.
    _nV = int(max(int(edges.sources.max()), int(edges.targets.max())) + 1) if edges.nE else 0
    for _support in getattr(edges, "branching", []) or []:
        _nV = max(_nV, int(max(_support)) + 1)
    check_analysis_size(_nV, edges.nE)

    if edges.nE == 0:
        import logging
        logging.getLogger(__name__).warning(
            "Edge construction produced 0 edges from %s input. "
            "For text/OCR: the document may be too short, contain only "
            "stopwords, or the OCR backend returned unusable output. "
            "The resulting RexGraph has no structure to analyze.",
            input_type,
        )

    w_mag = edges.weights
    w_E_arg = w_mag if len(w_mag) > 0 and not np.allclose(w_mag, 1.0) else None
    signs_arg = (
        edges.signs
        if len(edges.signs) > 0 and np.any(edges.signs < 0)
        else None
    )

    branching = [list(map(int, r)) for r in getattr(edges, "branching", []) or []]
    if branching:
        # a wider relation cannot be said in (sources, targets), so the whole complex is
        # built from a boundary CSR instead: the 2-ary relations first, in their original
        # order so every aligned array still lines up, then the k-ary ones.
        supports = [[int(a), int(b)]
                    for a, b in zip(edges.sources, edges.targets, strict=True)]
        supports.extend(branching)
        ptr = np.zeros(len(supports) + 1, dtype=np.int32)
        for i, support in enumerate(supports):
            ptr[i + 1] = ptr[i] + len(support)
        idx = np.fromiter((v for support in supports for v in support),
                          dtype=np.int32, count=int(ptr[-1]))
        if w_E_arg is not None:
            w_E_arg = np.concatenate([np.asarray(w_E_arg, dtype=float),
                                      np.ones(len(branching))])
        if signs_arg is not None:
            signs_arg = np.concatenate([np.asarray(signs_arg),
                                        np.ones(len(branching), dtype=signs_arg.dtype)])
        rex = RexGraph(boundary_ptr=ptr, boundary_idx=idx,
                       w_E=w_E_arg, signs=signs_arg)
    else:
        rex = RexGraph(
            sources=edges.sources,
            targets=edges.targets,
            w_E=w_E_arg,
            signs=signs_arg,
        )

    # Faces, on request.
    #
    # A face is a filled cycle: whatever satisfies B1 c_f = 0 on the relations it
    # spans. It is not triangles, and it is not conditioned on edge type, which is
    # an attribute that weights the complex. `rexgraph.faces` already solves this
    # exactly over the rationals and arity-general, reading the gon off the cycle
    # basis; this path used to ignore it for a triangle-only, type-gated rule, so no
    # ring with a double bond could close and no 4-gon could close at all.
    #
    # Nothing is filled unless asked. Asserting a face is asserting that something
    # is enclosed, and that is the caller's claim about their data, not a default.
    rex = attach_faces(rex, face_selection, type_labels=edges.type_labels)

    # the embedding travels with the complex when the source carried one, because the
    # lengths and angles taken against it are a different reading from the intrinsic ones
    # and both are wanted: a ring's intrinsic quadrance is a function of arity whatever
    # the conformation, and the embedded one moves when the ring puckers.
    embedding = list(getattr(edges, "embedding", []) or [])
    if embedding:
        rex._embedding = embedding

    # attributes the reader parsed, onto the cells they belong to. Same shape as the
    # store, so this is a hand-off rather than a translation.
    for grade, cells in (getattr(edges, "attributes", None) or {}).items():
        for index, values in cells.items():
            for key, value in values.items():
                rex.attach_metadata(int(grade), int(index), str(key), value)

    # Honour the declared vertex count, AFTER faces: attaching them can rebuild the
    # complex from its boundary arrays, which re-derives nV from the edge supports
    # and drops any vertex with no incident edge. Sizing from the supports alone
    # loses only the TRAILING isolated ones, so the same records in a different
    # order gave a different complex and a different beta_0. An interval that
    # overlaps nothing, an atom that bonds to nothing and a gene nothing correlates
    # with are all real 0-cells.
    n_declared = len(edges.vertex_labels or ())
    if n_declared > rex.nV:
        rex._nV = n_declared

    # Attach metadata for downstream use
    rex._agent_meta = {
        "input_type": input_type,
        "adapter": edges.__class__.__name__
        if hasattr(edges, "__class__")
        else "unknown",
        "vertex_labels": edges.vertex_labels,
        "type_names": edges.type_names,
        "n_types": edges.n_types,
        "threshold": threshold,
        "typing": typing,
        "face_selection": face_selection,
    }
    # Preserve text-position mapping when present (OCR/text adapters)
    if getattr(edges, "source_text", ""):
        rex._agent_meta["source_text"] = edges.source_text

    return rex


def _fallback_text_or_raise(data, input_type, err, **kwargs):
    """Fall back to text construction when a CSV-classified file won't load.

    ``detect_input_type`` classifies any existing ``.txt`` (and some
    ambiguous ``.csv``) by peeking at content, which can misfire on prose
    that merely contains commas. Rather than failing the whole document
    with an "Empty CSV" style error, re-read the file as text and build a
    word co-occurrence complex.
    """
    from pathlib import Path as _Path

    if _can_be_path(data):
        p = _Path(data)
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                # `err` is the ORIGINAL csv failure this function was called to rescue,
                # and the exception being handled here is the rescue itself failing. So
                # `from exc` would assert the wrong direction: the csv error did not
                # arise from the read error, it preceded it. Raising bare keeps Python's
                # implicit __context__, which already says "while handling that, this",
                # and that framing is the accurate one.
                raise err  # noqa: B904
            if text and len(text.strip()) >= 10:
                from agent.adapters.text import TextAdapter
                text_kwargs = {k: v for k, v in kwargs.items()
                               if k in ("window", "min_count", "max_vocab",
                                        "face_selection")}
                return TextAdapter().build(text, **text_kwargs)
    # Not a file we can salvage, so surface the original error.
    raise err


def _read_numeric_csv(path):
    """Read a CSV and return (X float64[n, k], names) for its numeric columns only. Pandas-free."""
    import csv
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    n_cols = len(header)
    cols = [[r[j] if j < len(r) else "" for r in rows] for j in range(n_cols)]

    def _numeric(vals):
        out = []
        seen_real = False
        for v in vals:
            if _is_missing_cell(v):
                out.append(float("nan"))
                continue
            try:
                out.append(float(str(v).strip()))
                seen_real = True
            except ValueError:
                return None
        return out if seen_real else None

    keep_names, keep_cols = [], []
    for j in range(n_cols):
        parsed = _numeric(cols[j])
        if parsed is not None:
            keep_names.append(header[j])
            keep_cols.append(parsed)
    if not keep_cols:
        raise ValueError(f"feature CSV {path!r} has no numeric columns")
    X = np.asarray(keep_cols, dtype=np.float64).T   # (n_rows, n_numeric_cols)
    return X, keep_names


def _build_feature_edges(data, input_type, threshold, typing, sign,
                          feature_names, **kwargs) -> EdgeConstruction:
    """Handle feature matrix inputs (array, DataFrame, or CSV path)."""
    adapter = FeatureMatrixAdapter()

    if input_type == "feature_csv":
        X, names = _read_numeric_csv(str(data))
        if feature_names is not None:
            names = feature_names
    elif _is_dataframe(data):
        numeric_df = data.select_dtypes(include=[np.number])   # DataFrame's own method, no pandas import
        X = numeric_df.values
        names = list(numeric_df.columns) if feature_names is None else feature_names
    else:
        X = np.asarray(data, dtype=np.float64)
        names = feature_names

    return adapter.build(
        X, feature_names=names, threshold=threshold,
        typing=typing, sign=sign, **kwargs,
    )


def _build_edge_list_edges(data, **kwargs) -> EdgeConstruction:
    """Handle edge list inputs (CSV path or DataFrame)."""
    adapter = EdgeListAdapter()

    if isinstance(data, (str, Path)):
        return adapter.build(str(data), **kwargs)
    elif _is_dataframe(data):
        # Save to temp CSV and load (reuse the classifier)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            data.to_csv(f, index=False)
            return adapter.build(f.name, **kwargs)
    else:
        raise TypeError(f"EdgeListAdapter needs a file path or DataFrame, got {type(data)}")


# One-call analysis
def auto_analyze(
    data: Any,
    *,
    depth: str = "standard",
    **kwargs,
) -> dict:
    """Build a rex and run the full analysis pipeline in one call.

    Parameters
    ----------
    data : any supported input (file path, array, DataFrame)
    depth : 'quick', 'standard', or 'full'
        'quick': topology + spectral (< 1 second)
        'standard': full analyze() output
        'full': analyze_all() with signal + quotient
    **kwargs
        Forwarded to auto_rex().

    Returns
    -------
    dict
        Complete analysis results. Keys match rexgraph.analysis.analyze().
    """
    from rexgraph.analysis import analyze, analyze_all

    rex = auto_rex(data, **kwargs)

    # Build vertex labels for the analysis
    meta = getattr(rex, "_agent_meta", {})
    vertex_labels = meta.get("vertex_labels")

    if depth == "quick":
        # Just trigger the cheap properties
        result = {
            "meta": {
                "nV": rex.nV,
                "nE": rex.nE,
                "nF": rex.nF,
            },
            "topology": {
                "betti": rex.betti,
                "euler": rex.euler_characteristic,
                "chain_valid": rex.chain_valid,
            },
        }
        return result
    elif depth == "full":
        return analyze_all(rex, vertex_labels=vertex_labels)
    else:
        return analyze(rex, vertex_labels=vertex_labels)
