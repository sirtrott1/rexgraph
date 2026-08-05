"""
RexGraph Mathematical Agent: auto-construction and analysis pipeline.

Data in -> math computes -> results out.
No black box. Every claim is a matrix operation. Every uncertainty is a void count.
"""

#: The agent version, kept here rather than read back from installed metadata so a
#: source checkout reports what it is. `rexgraph.__version__` is the core's, and the
#: two ship on their own cadences. pyproject.toml has to match; a test enforces it.
__version__ = "1.0.6"

from .auto import auto_analyze, auto_rex, detect_input_type
from .engine import DecisionEngine
from .pipeline import AnalysisPipeline
from .session import Session

__all__ = [
    "auto_rex",
    "auto_analyze",
    "detect_input_type",
    "AnalysisPipeline",
    "Session",
    "DecisionEngine",
]
