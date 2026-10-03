"""One asset root for the source checkout and the installed Agent wheel."""
from pathlib import Path

REQUIRED = ("index.html", "app.jsx", "styles/theme.css", "assets/favicon64.png",
            "react.production.min.js", "react-dom.production.min.js", "REACT-LICENSE.txt")


def frontend_dir():
    package = Path(__file__).parent
    installed = package / "frontend"
    if installed.is_dir(): return installed
    # Only a source layout with its own build declaration gets this fallback;
    # installed packages never search a generic site-packages/frontend directory.
    source = package.parent
    if (source / "pyproject.toml").is_file(): return source / "frontend"
    return installed


def validate_ui_assets():
    root = frontend_dir()
    missing = [name for name in REQUIRED if not (root/name).is_file() or (root/name).stat().st_size == 0]
    if missing:
        raise RuntimeError("Agent frontend is incomplete ("+", ".join(missing)+"); reinstall rexgraph-agent")
    return root
