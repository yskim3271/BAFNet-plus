"""
Model conversion utilities for streaming inference.

This package provides functions to transform standard models into
streaming-compatible versions:

- conv_converter: Conv -> StatefulConv
- reshape_free_converter: TSBlock -> ReshapeFreeTSBlock (batch_size=1 optimized)

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
