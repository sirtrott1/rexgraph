"""Launch RexGraph System."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_UI_ASSETS = ("index.html", "app.jsx", "react.production.min.js",
              "react-dom.production.min.js", "styles/theme.css", "styles/system.css")


def _ensure_ui_assets() -> None:
    """Validate packaged UI assets without network access or site package writes."""
    frontend = Path(__file__).parent.parent / "frontend"
    missing = [name for name in _UI_ASSETS
               if not (frontend / name).exists() or (frontend / name).stat().st_size == 0]
    if missing:
        raise RuntimeError("System installation is missing packaged UI assets: "
                           + ", ".join(missing) + "; reinstall rexgraph-system")


def _load_sources(specs) -> None:
    """Load named Rex sources before the server starts."""
    if not specs:
        return
    from rexgraph.io import load

    from system import register_source

    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = Path(path).stem
        name = name.strip()
        if not name:
            raise ValueError("source name cannot be empty")
        register_source(name, load(path))



def _load_catalogs(specs) -> None:
    """Register named safe file catalogs before the server starts."""
    if not specs:
        return
    from system import register_catalog
    for spec in specs:
        if "=" not in spec:
            raise ValueError("catalog expects NAME=PATH")
        name, raw = spec.split("=", 1)
        name = name.strip()
        roots = [part for part in raw.split(os.pathsep) if part]
        if not name or not roots:
            raise ValueError("catalog expects NAME=PATH")
        register_catalog(name, roots)

def main() -> None:
    """Run the System server with uvicorn."""
    import uvicorn

    parser = argparse.ArgumentParser(prog="rexgraph-system")
    parser.add_argument("--host", default=os.environ.get("REXGRAPH_SYSTEM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("REXGRAPH_SYSTEM_PORT", "8010")))
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--catalog", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()

    _ensure_ui_assets()
    env_source = os.environ.get("REXGRAPH_SYSTEM_SOURCE")
    specs = list(args.source)
    if env_source:
        specs.append(env_source)
    _load_sources(specs)
    _load_catalogs(args.catalog)
    uvicorn.run("system.server.app:app", host=args.host, port=args.port)
