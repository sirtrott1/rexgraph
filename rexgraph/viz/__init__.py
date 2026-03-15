# rexgraph/viz/__init__.py
"""
rexgraph.viz - Visualization for relational complexes.

    from rexgraph.viz import generate_dashboard, to_json
    generate_dashboard(rex, output_path="dashboard.html", open_browser=True)
"""

from .dashboard import generate_dashboard, to_json, run_dashboard, create_app

__all__ = [
    "generate_dashboard",
    "to_json",
    "run_dashboard",
    "create_app",
]
