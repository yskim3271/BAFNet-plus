"""ONNX QDQ export from a PT2E QAT model.

**Cycle 18b status: Option β implemented.** Cycle 17 skeleton fleshed out
with the V79-bound QDQ export driver.

- **Option β (override)** — :func:`export_qat_to_onnx_qdq_override`. Extracts
  per-fake-quant ``(scale, zero_point)`` tensors from the trained PT2E QAT
  GraphModule, maps them to ORT QDQ tensor names by reusing the warm-start
  hint→onnx-key matcher in reverse, and feeds the result to
  :func:`src.models.streaming.onnx.export.quantize_bafnetplus_qdq` via the
  new ``init_overrides`` kwarg (which routes them through ORT's
  ``extra_options["TensorQuantOverrides"]``). Calibration is still required
  for the non-QAT'd activations (e.g. phase-decoder, head, alpha) but the
  overrides take precedence wherever they apply. Guarantees graph identity
  with the D1 PTQ pipeline; only the initializer values change for the
  QAT-covered tensors.
- **Option α (direct)** — :func:`export_qat_to_onnx_qdq_direct`. Direct
  ``convert_pt2e + torch.onnx.export``. Skeleton only; left as fallback for
  the cycle 18b decision point if Option β fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ----------------------------------------------------------- scale extraction

def extract_qat_scales(qat_model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Extract ``(scale, zero_point)`` from every FakeQuantize in a QAT model.

    Walk the GraphModule, find each :class:`torch.ao.quantization.fake_quantize.FakeQuantize`
    (or its derived classes), and return a mapping
    ``{module_path: {'scale': Tensor, 'zero_point': Tensor, 'dtype': torch.dtype,
       'qscheme': torch.qscheme, 'ch_axis': int_or_None}}``.

    Cycle 18b uses this to feed ``init_overrides`` into
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


# -------------------------------------------- PT→ONNX tensor-name resolution

_DTYPE_TO_QUANT_TYPE: Dict[torch.dtype, str] = {
    torch.int8: "QInt8",
    torch.uint8: "QUInt8",
    torch.int16: "QInt16",
    torch.uint16: "QUInt16",
}


def _build_override_for_tensor(
    info: Dict[str, Any],
    *,
    fallback_quant_type: str = "QUInt16",
) -> Optional[Dict[str, Any]]:
    """Convert one ``extract_qat_scales`` entry into an ORT TensorQuantOverrides
    sub-dict (the inner dict; caller wraps it in a 1-element list).

    Returns ``None`` if the entry can't be serialized (unsupported dtype).
    """
    dtype = info.get("dtype")
    quant_type = _DTYPE_TO_QUANT_TYPE.get(dtype, fallback_quant_type)
    scale = info["scale"]
    zp = info["zero_point"]

    # ORT TensorQuantOverrides rejects `symmetric` together with `scale` /
    # `zero_point` — they're mutually exclusive override modes (cycle 18b-1
    # bug fix). The supplied (scale, zp) already encodes any symmetry
    # decision: symmetric quant produces zp=0 (signed) or zp=2**(b-1)
    # (unsigned), asymmetric produces a fitted zp. We drop `symmetric` and
    # let ORT consume the explicit values.

    if scale.ndim == 0 or scale.numel() == 1:
        # Per-tensor.
        return {
            "quant_type": quant_type,
            "scale": float(scale.item() if scale.ndim == 0 else scale.flatten()[0].item()),
            "zero_point": int(zp.item() if zp.ndim == 0 else zp.flatten()[0].item()),
        }

    # Per-channel — keep as plain lists for JSON-friendly transport into ORT.
    ch_axis = info.get("ch_axis")
    return {
        "quant_type": quant_type,
        "scale": scale.detach().cpu().float().numpy().tolist(),
        "zero_point": zp.detach().cpu().to(torch.int64).numpy().tolist(),
        "axis": int(ch_axis) if ch_axis is not None else 0,
    }


def build_init_overrides_from_qat(
    qat_model: nn.Module,
    reference_int8_onnx_path: Union[str, Path],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Build the ORT ``TensorQuantOverrides`` dict from a trained QAT model.

    Strategy:
        1. ``extract_qat_scales(qat_model)`` → ``{pt_path: {scale, zp, dtype, ...}}``.
        2. Load the D1 INT8 QDQ ONNX as the tensor-name reference (its QDQ
           initializer names map 1-to-1 to the FP32 graph's tensor names that
           the QDQ pass produces).
        3. For each PT fake-quant, derive an FX-node hint (via
           :func:`src.models.streaming.qat.init_scales._fq_hint`) and match it
           to an ONNX key (via :func:`._match_keys`). This is the *inverse* of
           the warm-start mapping (which goes ONNX→PT to inject scales into
           PT2E observers).
        4. For each matched ``(pt_path, onnx_key)``, convert the PT scale/zp
           into an ORT-friendly sub-dict and emit
           ``overrides[onnx_key] = [sub_dict]``.

    Returns:
        Tuple of ``(overrides, stats)`` where ``stats`` mirrors the
        warm-start report shape: ``num_matched``, ``num_pt_total``,
        ``num_onnx_total``, ``mapping_rate``, ``unmatched_pt``, ``unused_onnx``.
    """
    # Local imports — reuse the warm-start machinery without circular deps.
    from .init_scales import _load_onnx_qdq_scales, _fq_hint, _match_keys

    pt_scales = extract_qat_scales(qat_model)
    onnx_pairs = _load_onnx_qdq_scales(reference_int8_onnx_path)
    onnx_keys = list(onnx_pairs.keys())

    overrides: Dict[str, List[Dict[str, Any]]] = {}
    used_onnx: set = set()
    matched: Dict[str, str] = {}
    unmatched_pt: List[str] = []

    for pt_path in pt_scales.keys():
        info = pt_scales[pt_path]
        hint = _fq_hint(qat_model, pt_path)
        candidate = _match_keys(hint, [k for k in onnx_keys if k not in used_onnx])
        if candidate is None:
            unmatched_pt.append(pt_path)
            continue
        sub = _build_override_for_tensor(info)
        if sub is None:
            unmatched_pt.append(pt_path)
            continue
        # If multiple PT fake-quants map to the same ONNX key (graph branches
        # sharing an activation), keep the first (deterministic).
        overrides.setdefault(candidate, [sub])
        matched[pt_path] = candidate
        used_onnx.add(candidate)

    unused_onnx = [k for k in onnx_keys if k not in used_onnx]
    stats = {
        "num_matched": len(matched),
        "num_pt_total": len(pt_scales),
        "num_onnx_total": len(onnx_keys),
        "mapping_rate": (len(matched) / max(1, len(pt_scales))),
        "unmatched_pt": unmatched_pt,
        "unused_onnx": unused_onnx,
        "matched": matched,
    }
    return overrides, stats


# --------------------------------------------------------- public drivers

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
        NotImplementedError: Option α is the fallback path for cycle 18b
            and is deferred to the next session if Option β fails.
    """
    raise NotImplementedError(
        "Option α (direct PT2E export) is the cycle-18b fallback path; "
        "the canonical Option β was implemented first. Implement here if Option β fails."
    )


def export_qat_to_onnx_qdq_override(
    qat_model: nn.Module,
    fp32_streaming_onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    reference_int8_onnx_path: Union[str, Path],
    calibration_dir: Union[str, Path],
    activation_type: str = "QUInt16",
    weight_type: str = "QUInt8",
    auto_exclude_sensitive: bool = True,
    verbose: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """Option β — re-run the D1 PTQ pipeline with QAT-learned init_overrides.

    Algorithm:
        1. :func:`build_init_overrides_from_qat` produces ORT TensorQuantOverrides
           keyed by ONNX tensor name (matched to PT fake-quants via the
           warm-start heuristic in reverse direction).
        2. Calls :func:`src.models.streaming.onnx.export.quantize_bafnetplus_qdq`
           on the FP32 streaming ONNX with ``init_overrides=overrides``. Tensors
           covered by overrides skip the calibration scale; the rest fall back
           to the standard MinMax pass over ``calibration_dir``.

    Args:
        qat_model: Trained PT2E GraphModule (after warm-start + N epochs).
        fp32_streaming_onnx_path: FP32 streaming ONNX whose QDQ form we want
            (typically the S21 trunk graph).
        output_path: Output INT8 ``.onnx`` path.
        reference_int8_onnx_path: A D1 INT8 QDQ ONNX with the same graph
            geometry as ``fp32_streaming_onnx_path``. Used only as a
            tensor-name reference for the override mapping (its scales are
            ignored).
        calibration_dir: Directory of ``calib_*.npz`` files. Still required
            because not all activations get QAT'd (phase-decoder, head, alpha
            fall back to FP32; non-QAT'd activations still need ORT
            calibration).
        activation_type / weight_type: forwarded to
            :func:`quantize_bafnetplus_qdq`. Defaults match the D1 PTQ recipe.

    Returns:
        Tuple of ``(output_onnx_path, stats)`` where ``stats`` includes
        ``mapping_rate``, ``num_matched``, ``num_pt_total``, ``num_onnx_total``,
        ``onnx_md5``, ``onnx_size_bytes``.
    """
    from src.models.streaming.onnx.export import quantize_bafnetplus_qdq
    import hashlib

    overrides, mapping_stats = build_init_overrides_from_qat(
        qat_model, reference_int8_onnx_path
    )

    if verbose:
        print(
            f"[Option β] PT2E QAT -> ONNX QDQ override\n"
            f"  PT fake-quants  : {mapping_stats['num_pt_total']}\n"
            f"  ONNX keys       : {mapping_stats['num_onnx_total']}\n"
            f"  matched         : {mapping_stats['num_matched']} "
            f"({mapping_stats['mapping_rate']:.1%})\n"
            f"  unmatched PT    : {len(mapping_stats['unmatched_pt'])}\n"
            f"  unused ONNX     : {len(mapping_stats['unused_onnx'])}\n"
        )

    result = quantize_bafnetplus_qdq(
        fp32_streaming_onnx_path,
        output_path,
        calibration_dir=calibration_dir,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=False,
        auto_exclude_sensitive=auto_exclude_sensitive,
        verbose=verbose,
        init_overrides=overrides,
    )

    out_path = Path(result.onnx_path)
    md5 = hashlib.md5(out_path.read_bytes()).hexdigest()
    final_stats = dict(mapping_stats)
    final_stats["onnx_path"] = str(out_path)
    final_stats["onnx_md5"] = md5
    final_stats["onnx_size_bytes"] = out_path.stat().st_size
    final_stats["sidecar_path"] = result.metadata_path
    return out_path, final_stats


__all__ = [
    "extract_qat_scales",
    "build_init_overrides_from_qat",
    "export_qat_to_onnx_qdq_direct",
    "export_qat_to_onnx_qdq_override",
]
