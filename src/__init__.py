"""BAFNet-plus source package.

Core library:
    - ``data``           : Dataset and augmentation
    - ``stft``           : STFT/iSTFT and complex utilities
    - ``losses``         : Phase/PESQ training losses
    - ``checkpoint``     : Model loading, checkpoint I/O, ``ConfigDict``
    - ``utils``          : Progress logger, ANSI colors, config/file parsing
    - ``compute_metrics``: Intrusive metrics (PESQ/CSIG/CBAK/COVL/SSNR/STOI)
    - ``receptive_field``: RF calculator and auto-segment helper
    - ``runtime_common`` : Shared evaluate/enhance runtime setup + CLI helpers
    - ``solver``         : Training loop (``Solver`` class, lazily exposed)

Entry points:
    - ``src.train``      : Hydra-driven training CLI
    - ``src.evaluate``   : Multi-SNR evaluation CLI
    - ``src.enhance``    : Inference / wav-saving CLI

Analysis:
    - ``src.analysis.complexity``     : Params/MACs/RTF measurement
    - ``src.analysis.forward_traces`` : Calibration gain / fusion alpha extraction
"""

__version__ = "0.1.0"


def __getattr__(name):
    """Lazily expose heavier objects without import-time side effects."""
    if name == "Solver":
        from .solver import Solver
        return Solver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Solver",
]
