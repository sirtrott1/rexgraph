"""
RexGraph Mathematical Agent - auto-construction and analysis pipeline.

Data in -> math computes -> results out.
No black box. Every claim is a matrix operation. Every uncertainty is a void count.
"""

from .auto import auto_rex, auto_analyze, detect_input_type
from .pipeline import AnalysisPipeline
from .session import Session
from .engine import DecisionEngine

__all__ = [
    "auto_rex",
    "auto_analyze",
    "detect_input_type",
    "AnalysisPipeline",
    "Session",
    "DecisionEngine",
]
