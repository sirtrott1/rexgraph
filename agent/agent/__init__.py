"""
RexGraph Mathematical Agent: auto-construction and analysis pipeline.

Data in -> math computes -> results out.
No black box. Every claim is a matrix operation. Every uncertainty is a void count.
"""

#: The agent version, kept here rather than read back from installed metadata so a
#: source checkout reports what it is. The carrier contract changed this release, so
#: every inter-distribution floor requires a sibling at >=1.1.4: rcdb.index.repair only
#: means anything against a core that actually refuses duplicate incidence, and a
#: pre-1.1.4 core still accepts it. That rejects an older sibling rather than pinning
#: the five to each other. pyproject.toml has to match, and one test pins every
#: declaration across all five.
__version__ = "1.1.4"

from .auto import auto_analyze, auto_rex, detect_input_type
from .engine import DecisionEngine
from .pipeline import AnalysisPipeline
from .rcdb_integration import configure as _configure_rcdb
from .session import Session

# The sibling rcdb package does not import the agent, so this is where the agent's
# activity feed, request scoping, metadata privacy and similarity scoring are handed to
# it. Done at import, so a caller reaching for agent.rcdb gets the agent's behaviour
# without having to know this wiring exists.
_configure_rcdb()

__all__ = [
    "auto_rex",
    "auto_analyze",
    "detect_input_type",
    "AnalysisPipeline",
    "Session",
    "DecisionEngine",
    "rcql_runtime",
]


def __getattr__(name):
    """Bind the RCQL runtime on first use.

    rexgraph-rcql is a separate distribution and an optional extra, so importing the
    agent must not require it. It is reached here rather than at module scope so an
    install without the extra still works and only a caller that actually asks for the
    runtime sees the ImportError.
    """
    if name == "rcql_runtime":
        from .rcql_runtime import rcql_runtime
        return rcql_runtime
    raise AttributeError(name)
