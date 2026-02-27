"""
Streaming-compatible layer implementations.

This package provides drop-in replacements for standard layers that enable
streaming inference:

- StatefulCausalConv1d/2d: State-buffering convolutions
- TSBlock layers: Conv2d-based layers eliminating reshape operations

Example:
    >>> from src.models.streaming.layers import (
    ...     StatefulAsymmetricConv2d,
    ...     StreamingTSBlock,
    ... )
"""

from .stateful_conv import (
    StatefulAsymmetricConv2d,
    StatefulCausalConv1d,
    StatefulCausalConv2d,
)
from .tsblock import (
    CausalConv2dTime,
    ChannelLayerNorm2d,
    FreqCAB,
    FreqGPKFFN,
    SimpleGate2d,
    StreamingCAB,
    StreamingConv2d,
    StreamingGPKFFN,
    StreamingTSBlock,
)

__all__ = [
    # Convolutions (original stateful)
    "StatefulCausalConv1d",
    "StatefulAsymmetricConv2d",
    "StatefulCausalConv2d",
    # TSBlock building blocks
    "ChannelLayerNorm2d",
    "SimpleGate2d",
    "CausalConv2dTime",
    "FreqCAB",
    "FreqGPKFFN",
    # Streaming TSBlock layers
    "StreamingConv2d",
    "StreamingCAB",
    "StreamingGPKFFN",
    "StreamingTSBlock",
]
