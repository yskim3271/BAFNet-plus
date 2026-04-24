"""
BAFNet-plus source package

This package contains the complete BAFNet-plus implementation:

Core Library:
- data: Dataset and data augmentation
- solver: Training loop (Solver class)
- stft: STFT/iSTFT utilities
- runtime_common: Shared CLI/runtime helpers
- utils: Utility functions

Executable Scripts:
- train: Training entry point
- enhance: Inference/enhancement
- evaluate: Evaluation with metrics
- compute_metrics: Metric computation

Analysis:
- analysis: Offline model diagnostics and analysis utilities
"""

__version__ = "0.1.0"

# Core library imports
from .data import *
from .stft import *
from .utils import *


def __getattr__(name):
    """Lazily expose heavier objects without import-time side effects."""
    if name == "Solver":
        from .solver import Solver

        return Solver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'Solver',
]
