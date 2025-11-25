"""
BAFNet-plus source package

This package contains the complete BAFNet-plus implementation:

Core Library:
- data: Dataset and data augmentation
- solver: Training loop (Solver class)
- stft: STFT/iSTFT utilities
- utils: Utility functions

Executable Scripts:
- train: Training entry point
- enhance: Inference/enhancement
- evaluate: Evaluation with metrics
- compute_metrics: Metric computation
"""

__version__ = "0.1.0"

# Core library imports
# Note: solver is not imported here to avoid TensorBoard/TensorFlow conflicts
# Import Solver directly: from src.solver import Solver
from .data import *
from .stft import *
from .utils import *

__all__ = []
