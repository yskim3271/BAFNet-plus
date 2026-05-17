"""Thread-local state-frame context and stateful-layer helpers for streaming.

The streaming convolutions process *extended* inputs (current chunk + lookahead
frames) so a chunk's output can use real future context. Only the current-chunk
frames may update the recurrent conv state; the lookahead frames must not leak
into it. ``StateFramesContext`` injects that bound (the chunk size) as a
thread-local value the stateful convs read on every forward, so callers do not
have to thread ``state_frames`` through every layer call.

Ported from LaCoSENet ``src/models/streaming/utils.py`` (the ``StateFramesContext``
/ ``get_state_frames_context`` part plus the small stateful-layer helpers).

Example:
    >>> with StateFramesContext(8):  # chunk_size = 8
    ...     out = streaming_backbone(extended_input)  # 8 + L_enc + L_dec frames
"""

from __future__ import annotations

import threading
import warnings
from typing import Literal, Optional

import torch

# ---------------------------------------------------------------------------
# Thread-local state-frame context.
#
# CANONICAL STORAGE — do not duplicate this elsewhere. All streaming layers
# import ``get_state_frames_context`` from this module.
# ---------------------------------------------------------------------------
_state_frames_context = threading.local()


def get_state_frames_context() -> Optional[int]:
    """Return the current ``state_frames`` context value.

    Returns:
        Number of leading frames allowed to update conv state, or ``None`` if no
        context manager is active (all frames update state).
    """
    return getattr(_state_frames_context, "value", None)


def set_state_frames_context(value: Optional[int]) -> None:
    """Set the ``state_frames`` context value.

    Args:
        value: Number of leading frames allowed to update conv state, or
            ``None`` to clear the bound.
    """
    _state_frames_context.value = value


class StateFramesContext:
    """Context manager bounding which input frames may update conv state.

    Lets the model process an extended input (current chunk + lookahead) while
    only the first ``state_frames`` frames feed the recurrent conv state for the
    next chunk. Nesting is supported: each manager saves and restores the prior
    value on exit.

    Attributes:
        state_frames: Number of leading frames allowed to update conv state.
        prev_value: Previous context value, restored on ``__exit__``.

    Example:
        >>> # Process 14 frames (8 current + 3 enc + 3 dec lookahead) but only
        >>> # let the first 8 update conv state.
        >>> with StateFramesContext(8):
        ...     out = model(extended_input)
    """

    def __init__(self, state_frames: Optional[int]):
        """Initialize the context manager.

        Args:
            state_frames: Number of leading frames allowed to update conv state.
                ``None`` applies no bound (all frames update state).
        """
        self.state_frames = state_frames
        self.prev_value: Optional[int] = None

    def __enter__(self) -> "StateFramesContext":
        """Enter the context, installing ``state_frames``."""
        self.prev_value = get_state_frames_context()
        set_state_frames_context(self.state_frames)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        """Exit the context, restoring the previous value."""
        set_state_frames_context(self.prev_value)
        return False  # do not suppress exceptions

    def __repr__(self) -> str:
        return f"StateFramesContext(state_frames={self.state_frames})"


# ---------------------------------------------------------------------------
# Small helpers shared by the stateful conv layers.
# ---------------------------------------------------------------------------
def check_streaming_allowed(training: bool, layer_name: str = "StatefulLayer") -> None:
    """Raise if streaming mode is being enabled during training.

    Streaming requires eval mode: state buffers must not be part of gradient
    computation, and BatchNorm/Dropout behaviour differs in training mode.

    Args:
        training: ``module.training`` of the host module.
        layer_name: Name used in the error message.

    Raises:
        RuntimeError: If ``training`` is ``True``.
    """
    if training:
        raise RuntimeError(
            f"{layer_name}: streaming mode is not supported during training. "
            "Call model.eval() before enabling streaming."
        )


def check_batch_size_change(
    state: Optional[torch.Tensor],
    current_batch: int,
    layer_name: str = "StatefulLayer",
) -> bool:
    """Detect a batch-size change between chunks and warn if found.

    A batch-size change during streaming means utterances are being mixed in a
    batch, which corrupts per-utterance state buffers.

    Args:
        state: Current state buffer (``None`` on the first chunk).
        current_batch: Batch size of the current input.
        layer_name: Name used in the warning message.

    Returns:
        ``True`` if the state should be reset, ``False`` otherwise.
    """
    if state is None:
        return False
    if state.shape[0] != current_batch:
        warnings.warn(
            f"[{layer_name}] batch size changed ({state.shape[0]} -> {current_batch}); "
            "resetting state. This usually indicates incorrect streaming usage.",
            RuntimeWarning,
        )
        return True
    return False


def sync_state_device_dtype(
    state: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Move ``state`` to match the input device/dtype if needed.

    Args:
        state: State buffer to sync (may be ``None``).
        device: Target device.
        dtype: Target dtype.

    Returns:
        The synced state tensor, or ``None`` if ``state`` was ``None``.
    """
    if state is None:
        return None
    if state.device != device or state.dtype != dtype:
        return state.to(device=device, dtype=dtype)
    return state


__all__ = [
    "StateFramesContext",
    "get_state_frames_context",
    "set_state_frames_context",
    "check_streaming_allowed",
    "check_batch_size_change",
    "sync_state_device_dtype",
]
