"""Convert a ``Backbone`` to its stateful streaming form and manage state.

Recursively swaps the convolution leaf modules for their stateful equivalents
(weights copied, only the forward logic changes), then provides streaming-mode
toggling, state reset, and layer counting.

Key functions:
    - ``convert_to_stateful``: ``CausalConv1d`` -> ``StatefulCausalConv1d``,
      ``AsymmetricConv2d`` -> ``StatefulAsymmetricConv2d``,
      ``CausalConv2d`` -> ``StatefulCausalConv2d`` (recursively).
    - ``set_streaming_mode`` / ``reset_streaming_state``: toggle streaming and
      clear per-utterance state across all stateful layers.
    - ``prepare_streaming_model``: convert -> ``.to(device).eval()`` -> enable
      streaming, on an already-instantiated model (no checkpoint I/O — that
      belongs to the streaming wrapper / export driver).

Ported from LaCoSENet ``src/models/streaming/converters/conv_converter.py`` plus
the model-preparation core of ``src/models/streaming/utils.py``
(``apply_stateful_conv`` / ``prepare_streaming_model``), adjusted for BAFNet+
module paths.
"""

from __future__ import annotations

import copy
import logging
import warnings
from typing import Any, Callable, Dict, Tuple, Type

import torch.nn as nn

from src.models.backbone import AsymmetricConv2d, CausalConv1d, CausalConv2d
from src.models.streaming.layers.stateful_conv import (
    StatefulAsymmetricConv2d,
    StatefulCausalConv1d,
    StatefulCausalConv2d,
)

logger = logging.getLogger(__name__)


def _replace_modules_recursive(
    parent: nn.Module,
    target_type: Type[nn.Module],
    factory_fn: Callable[[Any], nn.Module],
) -> int:
    """Recursively replace modules of ``target_type`` under ``parent``.

    Uses ``named_children()``-based recursion (with a materialised list) to avoid
    iterator-invalidation issues from mutating modules during traversal.

    Args:
        parent: Module subtree to search.
        target_type: Module type to replace.
        factory_fn: Builds the replacement from the original module.

    Returns:
        Number of modules replaced.
    """
    replaced = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, target_type):
            setattr(parent, name, factory_fn(child))
            replaced += 1
        else:
            replaced += _replace_modules_recursive(child, target_type, factory_fn)
    return replaced


def convert_to_stateful(model: nn.Module, verbose: bool = True, inplace: bool = False) -> nn.Module:
    """Convert a ``Backbone``'s convolution leaves to their stateful versions.

    Args:
        model: Original model.
        verbose: Log a conversion summary.
        inplace: Mutate ``model`` in place; otherwise operate on a deep copy.

    Returns:
        The (possibly copied) model with stateful convs. Streaming mode is
        **disabled** by default — call :func:`set_streaming_mode` to enable it.
    """
    if not inplace:
        model = copy.deepcopy(model)

    count_1d = _replace_modules_recursive(model, CausalConv1d, StatefulCausalConv1d.from_causal_conv)
    count_asym = _replace_modules_recursive(model, AsymmetricConv2d, StatefulAsymmetricConv2d.from_asymmetric_conv)
    count_2d = _replace_modules_recursive(model, CausalConv2d, StatefulCausalConv2d.from_causal_conv2d)
    total = count_1d + count_asym + count_2d

    if verbose:
        logger.info(
            "Converted to stateful: %d CausalConv1d, %d AsymmetricConv2d, %d CausalConv2d (total %d)",
            count_1d,
            count_asym,
            count_2d,
            total,
        )
    if total == 0:
        warnings.warn(
            "No convertible layers found; expected CausalConv1d / AsymmetricConv2d / CausalConv2d.",
            UserWarning,
        )
    return model


def set_streaming_mode(model: nn.Module, streaming: bool) -> int:
    """Enable/disable streaming on every stateful layer in ``model``.

    Args:
        model: Stateful model.
        streaming: ``True`` to enable streaming.

    Returns:
        Number of layers whose streaming mode was set.

    Raises:
        RuntimeError: If ``streaming`` is ``True`` while the model is training.
    """
    if streaming and model.training:
        raise RuntimeError("Cannot enable streaming while the model is training; call model.eval() first.")

    layers = 0
    for module in model.modules():
        set_streaming = getattr(module, "set_streaming", None)
        if callable(set_streaming):
            set_streaming(streaming)
            layers += 1
    if layers == 0:
        warnings.warn("No stateful layers found; did you forget convert_to_stateful()?", UserWarning)
    return layers


def reset_streaming_state(model: nn.Module) -> int:
    """Clear per-utterance state on every stateful layer (call before a new utterance).

    Args:
        model: Stateful model.

    Returns:
        Number of layers reset.
    """
    reset = 0
    for module in model.modules():
        reset_state = getattr(module, "reset_state", None)
        if callable(reset_state):
            reset_state()
            reset += 1
    return reset


def get_stateful_layer_count(model: nn.Module) -> Dict[str, int]:
    """Count stateful layers by type.

    Args:
        model: Model to inspect.

    Returns:
        Dict with ``StatefulCausalConv1d`` / ``StatefulAsymmetricConv2d`` /
        ``StatefulCausalConv2d`` / ``total`` counts.
    """
    counts = {
        "StatefulCausalConv1d": 0,
        "StatefulAsymmetricConv2d": 0,
        "StatefulCausalConv2d": 0,
        "total": 0,
    }
    for module in model.modules():
        if isinstance(module, StatefulCausalConv1d):
            counts["StatefulCausalConv1d"] += 1
            counts["total"] += 1
        elif isinstance(module, StatefulAsymmetricConv2d):
            counts["StatefulAsymmetricConv2d"] += 1
            counts["total"] += 1
        elif isinstance(module, StatefulCausalConv2d):
            counts["StatefulCausalConv2d"] += 1
            counts["total"] += 1
    return counts


def prepare_streaming_model(
    model: nn.Module,
    device: str = "cpu",
    inplace: bool = False,
    verbose: bool = True,
) -> Tuple[nn.Module, int]:
    """Prepare an instantiated model for stateful streaming.

    Convert convs to stateful, move to ``device``, set ``eval()``, then enable
    streaming. This is the model-level core of LaCoSENet's
    ``prepare_streaming_model`` without the checkpoint/Hydra loading or the
    reshape-free / BN-fold transforms (those are handled by the streaming
    wrapper / export driver in later sessions).

    Args:
        model: Instantiated ``Backbone`` (or any module containing the conv
            leaves listed in :func:`convert_to_stateful`).
        device: Target device for the prepared model.
        inplace: Mutate ``model`` in place; otherwise operate on a deep copy.
        verbose: Log a conversion summary.

    Returns:
        ``(prepared_model, stateful_layer_count)``.
    """
    model = convert_to_stateful(model, verbose=verbose, inplace=inplace)
    model.to(device)
    model.eval()
    set_streaming_mode(model, True)
    count = get_stateful_layer_count(model)["total"]
    if verbose:
        logger.info("Prepared streaming model on %s (%d stateful layers)", device, count)
    return model, count


__all__ = [
    "convert_to_stateful",
    "set_streaming_mode",
    "reset_streaming_state",
    "get_stateful_layer_count",
    "prepare_streaming_model",
]
