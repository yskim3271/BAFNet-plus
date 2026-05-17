"""ONNX QDQ export from a PT2E QAT model.

**Cycle 17 status: SKELETON ONLY.** Full implementation lands in cycle 18 of
the Phase C plan — the V79 export probe is the cycle-18 decision point that
will choose between two paths:

- **Option α (direct)**: ``convert_pt2e(prepared)`` followed by
  ``torch.onnx.export(...)``. Output graph has QuantizeLinear /
  DequantizeLinear nodes natively. May produce a graph that QNN HTP rejects
  (different op layout vs the calibrated PTQ pipeline).
- **Option β (override)**: extract per-fake-quant ``(scale, zero_point)``
  tensors from the converted PT model, then run the existing FP32 streaming
  export + INT8 quantization pipeline (``quantize_bafnetplus_qdq``) with the
  QAT-learned values forced via ``init_overrides``. Guarantees graph identity
  with the D1 PTQ pipeline; only the initializer values change.

The current file pins the function signatures so the cycle 18 work can flesh
out the body without touching call-sites, and so unit tests in cycle 17 can
import the names without ``ImportError``. All exported helpers raise
``NotImplementedError`` for now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def extract_qat_scales(qat_model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Extract ``(scale, zero_point)`` from every FakeQuantize in a QAT model.

    Walk the GraphModule, find each :class:`torch.ao.quantization.fake_quantize.FakeQuantize`
    (or its derived classes), and return a mapping
    ``{module_path: {'scale': Tensor, 'zero_point': Tensor, 'dtype': torch.dtype,
       'qscheme': torch.qscheme, 'ch_axis': int_or_None}}``.

    Cycle 18 will use this to feed ``init_overrides`` into
    :func:`src.models.streaming.onnx.export.quantize_bafnetplus_qdq` (Option β).
    """
    from torch.ao.quantization.fake_quantize import FakeQuantize  # local import

    out: Dict[str, Dict[str, Any]] = {}
    for name, mod in qat_model.named_modules():
        if isinstance(mod, FakeQuantize):
            entry: Dict[str, Any] = {
                "scale": mod.scale.detach().clone(),
                "zero_point": mod.zero_point.detach().clone(),
                "dtype": getattr(mod, "dtype", None),
                "qscheme": getattr(mod, "qscheme", None),
                "ch_axis": getattr(mod, "ch_axis", None),
            }
            out[name] = entry
    return out


def export_qat_to_onnx_qdq_direct(
    qat_model: nn.Module,
    example_inputs: Tuple[Any, ...],
    output_path: Union[str, Path],
    *,
    opset_version: int = 17,
) -> Path:
    """Option α — convert_pt2e + torch.onnx.export.

    Args:
        qat_model: Output of :func:`prepare_bafnetplus_for_qat` after training
            (i.e. with FakeQuantize stats converged).
        example_inputs: Same shape contract as cycle-17 prepare.
        output_path: Where to write the QDQ ``.onnx``.
        opset_version: Defaults to 17 (locked S4 export contract).

    Returns:
        Path to the written ONNX file.

    Raises:
        NotImplementedError: Cycle-17 skeleton — implement in cycle 18 V79 probe.
    """
    raise NotImplementedError(
        "Option α (direct PT2E export) is cycle-18 work. "
        "See BAFNetPlus/results/profiling/phase_c_qat_plan.md § Cycle 18."
    )


def export_qat_to_onnx_qdq_override(
    qat_model: nn.Module,
    fp32_streaming_onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    calibration_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Option β — re-run PTQ pipeline with QAT-learned init_overrides.

    Strategy:
        1. ``extract_qat_scales(qat_model)`` → dict of learned (scale, zp).
        2. Build a name-mapped ``init_overrides`` matching ORT's QDQ
           initializer convention (cycle 18 will write the PT→ONNX mapping).
        3. Call :func:`src.models.streaming.onnx.export.quantize_bafnetplus_qdq`
           with the override dict; calibration data is unnecessary because the
           scales are pre-learned.

    Raises:
        NotImplementedError: Cycle-17 skeleton — implement in cycle 18 V79 probe.
    """
    raise NotImplementedError(
        "Option β (init_overrides path) is cycle-18 work. "
        "See BAFNetPlus/results/profiling/phase_c_qat_plan.md § Cycle 18."
    )


__all__ = [
    "extract_qat_scales",
    "export_qat_to_onnx_qdq_direct",
    "export_qat_to_onnx_qdq_override",
]
