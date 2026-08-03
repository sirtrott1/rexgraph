"""
RexGraph Mathematical Agent - auto-construction and analysis pipeline.

Data in -> math computes -> results out.
No black box. Every claim is a matrix operation. Every uncertainty is a void count.
"""

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
