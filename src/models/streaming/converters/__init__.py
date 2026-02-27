"""
Model conversion utilities for streaming inference.

This package provides functions to transform standard models into
streaming-compatible versions:

- conv_converter: Conv -> StatefulConv

TSBlock conversion is now handled directly by StreamingTSBlock.from_backbone_tsblock()
and StreamingTSBlock.convert_sequence_block() in layers/tsblock.py.

Example:
    >>> from src.models.streaming.converters import (
    ...     convert_to_stateful,
    ... )
"""

from .conv_converter import (
    convert_to_stateful,
    get_stateful_layer_count,
    reset_streaming_state,
    set_streaming_mode,
)

__all__ = [
    # Convolution conversion
    "convert_to_stateful",
    "set_streaming_mode",
    "reset_streaming_state",
    "get_stateful_layer_count",
]
