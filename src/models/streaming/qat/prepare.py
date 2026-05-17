"""Wrap a BAFNet+ model into a PT2E QAT-prepared :class:`GraphModule`.

This is Step 2 of cycle 17 — the *prep* half. The *warm-start* half lives in
:mod:`.init_scales`. The two are kept separate because warm-start is brittle
(PT-to-ONNX node-name heuristic) while prep is mostly mechanical.

Usage::

    from src.models.streaming.qat import prepare_bafnetplus_for_qat

    student = build_bafnetplus_from_checkpoint(...)
    bcs_example = torch.zeros(1, 201, 400, 2)
    acs_example = torch.zeros(1, 201, 400, 2)
    prepared = prepare_bafnetplus_for_qat(student, ((bcs_example, acs_example),))

The returned object is a :class:`torch.fx.GraphModule` with PT2E fake-quant
modules inserted; it is trainable (``prepared.train()``) and emits the same
output shape as the FP32 base in eval mode (fake-quant collapses to identity
when ``training=False`` and no observer has accumulated stats).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
from torch.export import Dim

from .quantizer import QNNQuantizer

logger = logging.getLogger(__name__)


def _default_example_inputs(*, batch: int = 1, freq: int = 201, time_frames: int = 400) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    """Canonical BAFNet+ example input: ``((bcs_com, acs_com),)`` shape [B, F, T, 2]."""
    bcs = torch.zeros(batch, freq, time_frames, 2)
    acs = torch.zeros(batch, freq, time_frames, 2)
    return ((bcs, acs),)


def _default_dynamic_shapes() -> tuple:
    """Mark the time axis (dim 2) as dynamic so the exported FX graph accepts
    arbitrary STFT segment lengths at training time. Batch + freq + last dim
    stay static — the model specializes ``B`` in internal reshapes
    (``view(B*F, …)``) so ``Dim.AUTO`` is used to let the tracer decide.

    Cycle 18a fix: cycle 17 omitted ``dynamic_shapes`` so the FX graph baked
    ``T=400`` into reshape ops, which crashes at the first batch when the
    training dataloader emits segments with a different STFT frame count
    (e.g. 161 frames at ``segment=32000``).
    """
    T = Dim("time", min=16, max=4096)
    # Per-tensor spec mirrors the [B, F, T, 2] layout. Dim.AUTO on batch lets
    # the tracer collapse to static when the model internally specializes B.
    tensor_spec = {0: Dim.AUTO, 1: None, 2: T, 3: None}
    # ``example_inputs`` structure is ``((bcs, acs),)`` — a 1-tuple containing
    # a 2-tuple of tensors. dynamic_shapes mirrors this pytree.
    return ((tensor_spec, tensor_spec),)


def prepare_bafnetplus_for_qat(
    model: nn.Module,
    example_inputs: Optional[Tuple[Any, ...]] = None,
    *,
    quantizer: Optional[QNNQuantizer] = None,
    strict_export: bool = False,
    dynamic_shapes: Optional[Any] = None,
) -> nn.Module:
    """Prepare a BAFNet+ model for PT2E QAT.

    Args:
        model: FP32 BAFNet+ instance to wrap. May be on any device; the returned
            GraphModule lives on the same device as the input model.
        example_inputs: Positional args tuple matching ``model.forward(*args)``.
            For BAFNet+ the canonical form is ``((bcs_com, acs_com),)`` — one
            positional arg that is itself a 2-tuple. Pass ``None`` to use the
            default 1×201×400 zeros shape.
        quantizer: Optional :class:`QNNQuantizer`. Default constructs one with
            ``skip_phase_conv=True`` and ``skip_head=True`` (the canonical D1
            trunk scope).
        strict_export: Forwarded to :func:`torch.export.export`. Default
            ``False`` because BAFNet+ has conditional branches
            (``ablation_mode``, ``use_calibration``, ``mask_only_alpha``) that
            the strict tracer rejects.

    Returns:
        :class:`torch.fx.GraphModule` ready for QAT training. Call ``.train()``
        before iterating; in eval mode the fake-quant nodes are inert.

    Raises:
        RuntimeError: If :func:`torch.export.export` fails. The most common
            cause on BAFNet+ is an in-place mutation or a non-traceable Python
            branch — escalate to AIMET (cycle 18 fallback) in that case.
    """
    if example_inputs is None:
        example_inputs = _default_example_inputs()
    if quantizer is None:
        quantizer = QNNQuantizer()
    if dynamic_shapes is None:
        dynamic_shapes = _default_dynamic_shapes()

    was_training = model.training
    model.train()
    try:
        exported_program = torch.export.export(
            model, example_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=strict_export,
        )
    except Exception as e:  # pragma: no cover — bubble up with context
        raise RuntimeError(
            f"torch.export.export failed on BAFNet+ (strict={strict_export}). "
            "If the cause is a conditional branch, try freezing ablation_mode "
            "or escalate to AIMET (cycle 18 fallback)."
        ) from e
    finally:
        if not was_training:
            model.eval()

    prepared = prepare_qat_pt2e(exported_program.module(), quantizer)
    # PT2E exported GraphModules reject the standard ``train()``/``eval()`` API.
    # Monkey-patch them to redirect to the PT2E helpers (this is the documented
    # mitigation for "Calling train() or eval() is not supported for exported
    # models") so the rest of the training code (Solver, tests) can stay generic.
    try:
        from torch.ao.quantization import allow_exported_model_train_eval

        allow_exported_model_train_eval(prepared)
    except Exception:  # pragma: no cover — best-effort patch
        logger.debug("[QAT prepare] allow_exported_model_train_eval unavailable", exc_info=True)
    logger.info(
        "[QAT prepare] annotated=%d skipped=%d (skip_phase_conv=%s skip_head=%s)",
        len(quantizer.annotated_node_names),
        len(quantizer.skipped_node_names),
        quantizer.skip_phase_conv,
        quantizer.skip_head,
    )
    return prepared
