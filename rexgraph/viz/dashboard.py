# rexgraph/viz/dashboard.py
"""
rexgraph.viz.dashboard - Generate self-contained HTML dashboards.

Wires rexgraph.analysis.analyze() output into the React JSX
template and produces a single `.html` file that opens in any browser
with zero dependencies beyond the library itself.

Usage::

    from rexgraph.viz import generate_dashboard

    generate_dashboard(
        rex,
        vertex_labels=names,
        edge_attrs={"type": types, "weight": weights},
        negative_types=["inhibition"],
        output_path="network.html",
        open_browser=True,
    )

Optionally, a Flask-based live server is available:

    from rexgraph.viz import run_dashboard
    run_dashboard(rex, port=5000)
"""

from __future__ import annotations

import json
import os
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from ..graph import RexGraph

from ..analysis import analyze

__all__ = [
    "generate_dashboard",
    "to_json",
    "run_dashboard",
    "create_app",
]


# Paths and constants


_VIZ_DIR = Path(__file__).parent
_TEMPLATE_DIR = _VIZ_DIR / "templates"
_DEFAULT_TEMPLATE = _TEMPLATE_DIR / "rex_dashboard_template.jsx"

# Sentinel marker in the JSX template replaced with live data.
_SENTINEL = "/*__REX_DATA__*/null"

# CDN versions pinned for reproducibility.
_REACT_CDN = "https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"
_REACT_DOM_CDN = "https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"
_BABEL_CDN = "https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.9/babel.min.js"

_HTML_SHELL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script crossorigin src="{react_cdn}"></script>
<script crossorigin src="{react_dom_cdn}"></script>
<script crossorigin src="{babel_cdn}"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#060810;overflow-x:hidden;overflow-y:auto;font-family:sans-serif}}
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
{jsx_source}
ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
</script>
</body>
</html>
"""


# Template loading and data injection


def _load_template(path: Optional[Union[str, Path]] = None) -> str:
    """Read the JSX template file.

    Parameters
    ----------
    path : str or Path, optional
        Custom template path.  Falls back to the bundled template.

    Raises
    ------
    FileNotFoundError
        If neither the custom nor bundled template exists.
    """
    p = Path(path) if path else _DEFAULT_TEMPLATE
    if not p.exists():
        raise FileNotFoundError(
            f"JSX template not found at {p}.  "
            f"Expected at {_DEFAULT_TEMPLATE} or provide custom_template=."
        )
    return p.read_text(encoding="utf-8")


class _SafeEncoder(json.JSONEncoder):
    """Handle numpy types and reject NaN/Inf."""

    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if v != v or v == float("inf") or v == float("-inf"):
                return 0.0
            return v
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _inject_data(jsx: str, data: dict) -> str:
    """Replace the sentinel in the JSX template with serialized data."""
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False,
                         cls=_SafeEncoder, allow_nan=False)
    if _SENTINEL not in jsx:
        raise ValueError(
            f"Template missing sentinel '{_SENTINEL}'.  "
            "Use the bundled rex_dashboard_template.jsx or ensure your "
            "custom template contains the sentinel."
        )
    return jsx.replace(_SENTINEL, payload, 1)


# Static dashboard generation (core API - no server dependency)


def generate_dashboard(
    rex: "RexGraph",
    *,
    output_path: Union[str, Path] = "rex_dashboard.html",
    vertex_labels: Optional[Sequence[str]] = None,
    edge_attrs: Optional[Dict[str, Any]] = None,
    negative_types: Optional[List[str]] = None,
    svg_size: Tuple[int, int] = (700, 500),
    title: str = "Rex Dashboard",
    open_browser: bool = False,
    data_only: bool = False,
    custom_template: Optional[Union[str, Path]] = None,
) -> Union[str, dict]:
    """Generate a self-contained HTML dashboard from a `RexGraph`.

    Calls analyze() to compute the full data contract, injects the
    result into the React JSX template, and writes a single `.html` file
    that opens in any modern browser with no server required.

    Parameters
    ----------
    rex : RexGraph
        The relational complex to visualize.
    output_path : str or Path
        Where to write the HTML file.
    vertex_labels : sequence of str, optional
        Human-readable vertex names.
    edge_attrs : dict, optional
        Named edge attributes (`{"type": arr, "weight": arr}`).
    negative_types : list of str, optional
        Edge types representing inhibition/negative flow.
    svg_size : (int, int)
        Canvas dimensions `(width, height)` for the SVG layout.
    title : str
        HTML page title.
    open_browser : bool
        Open the result in the default browser.
    data_only : bool
        If `True`, skip HTML generation and return the raw data dict.
        Useful for notebooks, JSON export, or custom frontends.
    custom_template : str or Path, optional
        Path to a custom JSX template.  Must contain the sentinel
        `/*__REX_DATA__*/null`.

    Returns
    -------
    str or dict
        If `data_only`, the analysis dict.  Otherwise the absolute
        path to the written HTML file.

    Examples
    --------
    >>> generate_dashboard(rex, output_path="my_network.html", open_browser=True)
    >>> data = generate_dashboard(rex, data_only=True)
    """
    # 1. Compute full analysis via analysis.py -> Cython pipeline
    data = analyze(
        rex,
        vertex_labels=vertex_labels,
        edge_attrs=edge_attrs,
        negative_types=negative_types,
        svg_w=svg_size[0],
        svg_h=svg_size[1],
    )

    if data_only:
        return data

    # 2. Load JSX template and inject data
    jsx = _load_template(custom_template)
    jsx = _inject_data(jsx, data)

    # 3. Wrap in self-contained HTML shell
    html = _HTML_SHELL.format(
        title=title,
        react_cdn=_REACT_CDN,
        react_dom_cdn=_REACT_DOM_CDN,
        babel_cdn=_BABEL_CDN,
        jsx_source=jsx,
    )

    # 4. Write
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    # 5. Open browser
    if open_browser:
        webbrowser.open(f"file://{out}")

    return str(out)


def to_json(
    rex: "RexGraph",
    path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> dict:
    """Export the analysis as a JSON file or dict.

    Parameters
    ----------
    rex : RexGraph
        The relational complex to analyze.
    path : str or Path, optional
        If provided, write JSON to this file.
    **kwargs
        Forwarded to analyze() (`vertex_labels`, `edge_attrs`,
        `negative_types`, `svg_w`, `svg_h`).

    Returns
    -------
    dict
        The analysis data dictionary.

    Examples
    --------
    >>> data = to_json(rex)
    >>> to_json(rex, "data.json", vertex_labels=names)
    """
    data = analyze(rex, **kwargs)
    if path is not None:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False,
                      cls=_SafeEncoder, allow_nan=False)
    return data


# Flask live server (optional - only imported if Flask is installed)


def create_app(
    rex: Optional["RexGraph"] = None,
    data: Optional[Dict[str, Any]] = None,
    **analyze_kwargs,
) -> Any:
    """Create a Flask app serving the dashboard with a live JSON API.

    Requires `pip install flask`.  The dashboard is the same React
    component used by generate_dashboard(), served via Flask
    with an additional `/api/data` endpoint for programmatic access.

    Parameters
    ----------
    rex : RexGraph, optional
        Graph to visualize.  Mutually exclusive with *data*.
    data : dict, optional
        Pre-computed analysis dict (from analyze()).
    **analyze_kwargs
        Forwarded to analyze() if *rex* is provided.

    Returns
    -------
    Flask
        Configured Flask application.

    Raises
    ------
    ImportError
        If Flask is not installed.
    """
    try:
        from flask import Flask, jsonify, request
    except ImportError:
        raise ImportError(
            "Flask is required for the live dashboard server.\n"
            "Install it with: pip install flask"
        )

    if rex is not None:
        dashboard_data = analyze(rex, **analyze_kwargs)
    elif data is not None:
        dashboard_data = data
    else:
        dashboard_data = {}

    app = Flask(__name__)
    app.config["dashboard_data"] = dashboard_data

    @app.route("/")
    def index():
        d = app.config["dashboard_data"]
        jsx = _load_template()
        jsx = _inject_data(jsx, d)
        html = _HTML_SHELL.format(
            title="Rex Dashboard",
            react_cdn=_REACT_CDN,
            react_dom_cdn=_REACT_DOM_CDN,
            babel_cdn=_BABEL_CDN,
            jsx_source=jsx,
        )
        return html

    @app.route("/api/data")
    def api_data():
        return jsonify(app.config["dashboard_data"])

    @app.route("/api/update", methods=["POST"])
    def api_update():
        app.config["dashboard_data"] = request.json
        return jsonify({"status": "ok"})

    return app


def run_dashboard(
    rex: Optional["RexGraph"] = None,
    data: Optional[Dict[str, Any]] = None,
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    open_browser: bool = True,
    **analyze_kwargs,
) -> None:
    """Run the interactive dashboard as a local Flask server.

    Requires `pip install flask`.

    Parameters
    ----------
    rex : RexGraph, optional
        Graph to visualize.
    data : dict, optional
        Pre-computed analysis dict.
    host : str
        Host to bind to.
    port : int
        Port to run on.
    debug : bool
        Enable Flask debug mode.
    open_browser : bool
        Open browser automatically.
    **analyze_kwargs
        Forwarded to analyze() if *rex* is provided.

    Examples
    --------
    >>> run_dashboard(rex, port=5000)
    >>> run_dashboard(data=analyze(rex), port=8080)
    """
    app = create_app(rex=rex, data=data, **analyze_kwargs)

    if open_browser:
        import threading
        url = f"http://{host}:{port}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"RexGraph Dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


# CLI entry point


def main():
    """CLI entry point for `rex-dashboard`."""
    import argparse

    parser = argparse.ArgumentParser(description="RexGraph Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser")
    parser.add_argument("--data", type=str,
                        help="Path to JSON data file")

    args = parser.parse_args()

    data = None
    if args.data:
        with open(args.data) as f:
            data = json.load(f)

    run_dashboard(
        data=data,
        host=args.host,
        port=args.port,
        debug=args.debug,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
