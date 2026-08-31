"""RexGraph System observatory."""
from .state import sources

#: Kept here rather than read back from installed metadata, so a source checkout
#: reports what it is. pyproject.toml has to match; a test enforces it.
__version__ = "1.1.3"


def register_source(name, value, *, policy=None):
    """Register a live value with an optional RCQL source policy."""
    sources.register(name, value, policy=policy)


def register_catalog(name, roots, *, max_entries=100000, policy=None):
    """Register a safe local RexGraph file catalog with an optional RCQL policy."""
    from rexgraph.io import FileCatalog
    catalog = FileCatalog(roots, max_entries=max_entries)
    sources.register(name, catalog, policy=policy)
    return catalog


def remove_source(name):
    """Remove a live value from the System observatory."""
    sources.remove(name)


__all__ = ["register_source", "register_catalog", "remove_source", "sources"]
