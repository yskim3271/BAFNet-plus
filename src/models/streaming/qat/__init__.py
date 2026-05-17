"""Phase C QAT scaffolding for BAFNet+ (cycle 17).

PT2E-based quantization-aware training with PESQ-GAN dropped + Self-KD against
the FP32 D1 teacher (`bafnetplus_50ms/best.th`). This module is the **PT-side**
training framework; ONNX QDQ export is handled by `qat/export.py` (skeleton for
cycle 17, full impl is cycle 18 V79 probe).

See ``BAFNetPlus/results/profiling/phase_c_qat_plan.md`` for the master plan.

Public API
----------
- :class:`QNNQuantizer` — custom PT2E :class:`Quantizer` that mirrors the D1
  PTQ scheme (per-channel QInt8 weight, per-tensor asymmetric QUInt16 activation
  on Conv1d/Conv2d/Linear).
- :func:`prepare_bafnetplus_for_qat` — wraps a BAFNet+ model via
  :func:`torch.export.export` + :func:`prepare_qat_pt2e`.
- :func:`warm_start_from_ptq` — extracts (scale, zp) from a D1 INT8 ONNX
  and injects them into the prepared fake-quant modules to shorten convergence.
- :class:`KDLoss` — output-level + TSBlock feature-alignment KD against the
  frozen FP32 teacher.
"""

from __future__ import annotations

from .kd import KDLoss
from .init_scales import warm_start_from_ptq
from .prepare import prepare_bafnetplus_for_qat
from .quantizer import QNNQuantizer

__all__ = [
    "QNNQuantizer",
    "prepare_bafnetplus_for_qat",
    "warm_start_from_ptq",
    "KDLoss",
]
