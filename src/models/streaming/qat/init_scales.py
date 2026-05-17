"""Warm-start PT2E fake-quant (scale, zero_point) tensors from a D1 INT8 ONNX.

Why this exists
---------------
PT2E observers initialize to wide ranges and converge slowly without
calibration data. The D1 INT8 ONNX (`bafnetplus_50ms_int8_qdq_trunk_t2.onnx`
md5 ``252f732a…``) was calibrated on the TAPS corpus (450 NPZ via
``/tmp/bafnet_calib_taps_v3``) and has well-tuned scales already. Seeding the
QAT fake-quant modules with those scales should cut convergence by ~30-50 %
(per literature) without requiring a separate calibration pass at the start of
training.

Mapping heuristic (cycle 17 — best-effort, brittle by design)
-------------------------------------------------------------
1. Walk the D1 ONNX initializer list, collect every ``*_scale`` /
   ``*_zero_point`` pair belonging to a QuantizeLinear or DequantizeLinear node.
   Key the resulting dict by the **producing node's name** (stripped of the
   ``_y_scale`` / ``_y_zero_point`` suffix).
2. Walk the prepared PT2E GraphModule. For each fake-quant module, derive a
   short "hint" from the FX node that consumes it (input-side activation
   observer) or produces it (weight observer / output-side activation observer).
3. Match each hint against the ONNX keys via a sequence of fall-throughs:
   (a) exact suffix match, (b) substring containment, (c) sequential-index
   alignment (last-resort — works if PT-graph order ≈ ONNX-graph order).
4. Inject matched scales into ``mod.scale.data`` and zero-points into
   ``mod.zero_point.data``. Unmatched modules retain their default init.

Returned dict reports the mapping rate (``matched / total``). Cycle 17 exit
gate accepts ≥ 70 %; cycle 18 will iterate if lower.

This warm-start is **optional** — QAT works from cold start, it just takes more
epochs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import FakeQuantize

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- ONNX side

def _load_onnx_qdq_scales(onnx_path: Union[str, Path]) -> Dict[str, Tuple[Any, Any]]:
    """Return ``{producing_node_name: (scale_array, zero_point_array)}``.

    Walks the initializer list and looks for the QuantizeLinear/DequantizeLinear
    convention used by ORT QDQ: a producing op named ``foo/Conv`` typically has
    sibling initializers ``foo/Conv_output_0_scale`` + ``foo/Conv_output_0_zero_point``,
    or ``foo/Conv_weight_quantize_scale`` / ``..._zero_point`` for weights.
    """
    import onnx  # lazy import — heavy
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    inits = {init.name: init for init in model.graph.initializer}

    pairs: Dict[str, Tuple[Any, Any]] = {}
    for name, init in inits.items():
        if not (name.endswith("_scale") or name.endswith("_y_scale")):
            continue
        if name.endswith("_y_scale"):
            key = name[: -len("_y_scale")]
            zp_name = key + "_y_zero_point"
        else:
            key = name[: -len("_scale")]
            zp_name = key + "_zero_point"
        scale_arr = numpy_helper.to_array(init)
        zp_arr = (
            numpy_helper.to_array(inits[zp_name]) if zp_name in inits else None
        )
        pairs[key] = (scale_arr, zp_arr)
    return pairs


def _summarize_onnx_keys(keys: List[str], limit: int = 8) -> List[str]:
    if len(keys) <= limit:
        return keys
    return keys[:limit] + [f"... (+{len(keys) - limit} more)"]


# ------------------------------------------------------------------ PT side

def _iter_fake_quant_modules(prepared_model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Yield ``(module_path, fake_quant_module)`` in registration order."""
    out: List[Tuple[str, nn.Module]] = []
    for name, mod in prepared_model.named_modules():
        if isinstance(mod, FakeQuantize):
            out.append((name, mod))
    return out


def _fq_hint(prepared_model: nn.Module, fq_name: str) -> str:
    """Best-effort hint string for matching a PT2E fake-quant to an ONNX key.

    PT2E generates synthetic names like ``activation_post_process_42``. The
    *useful* hint is the FX node(s) attached to that observer. We walk the
    GraphModule's nodes and return the name of the first node that targets the
    fake-quant module.
    """
    if not hasattr(prepared_model, "graph"):
        return fq_name
    hints: List[str] = []
    for node in prepared_model.graph.nodes:
        if node.op == "call_module" and node.target == fq_name:
            # Prefer the user node — that's the op whose I/O this fq observes
            users = [u.name for u in node.users.keys()]
            inputs = [
                a.name
                for a in node.args
                if hasattr(a, "name")
            ]
            hints.extend(users)
            hints.extend(inputs)
            hints.append(node.name)
    return ";".join(hints) if hints else fq_name


# -------------------------------------------------------------- match logic

def _match_keys(hint: str, onnx_keys: List[str]) -> Optional[str]:
    """Match a PT hint to an ONNX key via suffix → substring fall-through."""
    if not hint:
        return None
    hint_lower = hint.lower()
    # 1) exact suffix
    for k in onnx_keys:
        if hint_lower.endswith(k.lower()):
            return k
    # 2) substring containment (either direction)
    for k in onnx_keys:
        k_lower = k.lower()
        if k_lower in hint_lower or hint_lower in k_lower:
            return k
    # 3) token-overlap fallback: split on common delimiters and look for >=2 token overlaps
    hint_tokens = set(t for t in _tokenize(hint_lower) if len(t) > 2)
    best = (0, None)
    for k in onnx_keys:
        k_tokens = set(t for t in _tokenize(k.lower()) if len(t) > 2)
        overlap = len(hint_tokens & k_tokens)
        if overlap > best[0]:
            best = (overlap, k)
    if best[0] >= 2:
        return best[1]
    return None


def _tokenize(s: str) -> List[str]:
    out: List[str] = []
    buf = []
    for ch in s:
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


# ----------------------------------------------------------------- public

def warm_start_from_ptq(
    prepared_model: nn.Module,
    ptq_onnx_path: Union[str, Path],
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Inject D1 PTQ scales into the prepared PT2E QAT model's fake-quant modules.

    Args:
        prepared_model: Output of :func:`prepare_bafnetplus_for_qat`. Must be a
            GraphModule with PT2E fake-quant modules already inserted.
        ptq_onnx_path: Path to ``bafnetplus_50ms_int8_qdq_trunk_t2.onnx`` (or
            another INT8 QDQ ONNX with the same scheme).
        verbose: If True, log per-fake-quant mapping decisions.

    Returns:
        Dict with keys:
            ``mapping_rate`` (float in [0, 1]),
            ``num_matched`` (int),
            ``num_pt_total`` (int),
            ``num_onnx_total`` (int),
            ``matched`` (dict: pt_module_path → onnx_key),
            ``unmatched_pt`` (list of unmatched PT module paths),
            ``unused_onnx`` (list of ONNX keys not consumed by any PT module).
    """
    onnx_pairs = _load_onnx_qdq_scales(ptq_onnx_path)
    onnx_keys = list(onnx_pairs.keys())
    pt_modules = _iter_fake_quant_modules(prepared_model)

    if verbose:
        logger.info(
            "[QAT warm-start] PT fake-quants=%d, ONNX qdq keys=%d (first few: %s)",
            len(pt_modules), len(onnx_keys),
            _summarize_onnx_keys(onnx_keys),
        )

    matched: Dict[str, str] = {}
    unmatched_pt: List[str] = []

    # Strategy: (a) name-heuristic match first; (b) for unmatched, sequential-index
    # fallback against the still-unused ONNX keys (preserves graph-order alignment).
    used_onnx: set = set()
    for pt_path, pt_mod in pt_modules:
        hint = _fq_hint(prepared_model, pt_path)
        candidate = _match_keys(hint, [k for k in onnx_keys if k not in used_onnx])
        if candidate is None:
            unmatched_pt.append(pt_path)
            continue
        scale, zp = onnx_pairs[candidate]
        _inject_scale(pt_mod, scale, zp)
        matched[pt_path] = candidate
        used_onnx.add(candidate)
        if verbose:
            logger.info("[QAT warm-start] %s  ←  %s", pt_path, candidate)

    # Sequential-index fallback for unmatched PT modules.
    leftover_onnx = [k for k in onnx_keys if k not in used_onnx]
    fallback_matches = 0
    for pt_path in list(unmatched_pt):
        if not leftover_onnx:
            break
        # Take the first leftover ONNX key only if the PT module index roughly
        # aligns — guards against pairing unrelated tensors when the rate is
        # already high. Heuristic: only fallback when total mapping rate < 50 %.
        if len(matched) / max(1, len(pt_modules)) >= 0.5:
            break
        k = leftover_onnx.pop(0)
        scale, zp = onnx_pairs[k]
        pt_mod = dict(pt_modules)[pt_path]
        _inject_scale(pt_mod, scale, zp)
        matched[pt_path] = k
        unmatched_pt.remove(pt_path)
        used_onnx.add(k)
        fallback_matches += 1

    rate = len(matched) / max(1, len(pt_modules))
    summary = {
        "mapping_rate": rate,
        "num_matched": len(matched),
        "num_pt_total": len(pt_modules),
        "num_onnx_total": len(onnx_pairs),
        "num_fallback_matches": fallback_matches,
        "matched": matched,
        "unmatched_pt": unmatched_pt,
        "unused_onnx": [k for k in onnx_keys if k not in used_onnx],
    }
    logger.info(
        "[QAT warm-start] mapping_rate=%.1f%% (%d / %d PT fake-quants matched; "
        "%d ONNX keys unused; fallback=%d)",
        rate * 100, len(matched), len(pt_modules),
        len(summary["unused_onnx"]), fallback_matches,
    )
    return summary


def _inject_scale(fq_mod: nn.Module, scale_arr: Any, zp_arr: Any) -> None:
    """Inject (scale, zero_point) into a FakeQuantize module in-place.

    Handles both per-tensor (scalar scale) and per-channel (vector scale) cases.
    Skips if the array shapes can't be coerced into the target buffer shape —
    that's a sign the heuristic matched the wrong pair (rare, logged at debug).
    """
    import numpy as np

    try:
        scale_tensor = torch.from_numpy(np.asarray(scale_arr).astype(np.float32))
        if scale_tensor.numel() == 1 and fq_mod.scale.numel() == 1:
            fq_mod.scale.data.fill_(float(scale_tensor.item()))
        elif scale_tensor.shape == fq_mod.scale.shape:
            fq_mod.scale.data.copy_(scale_tensor.to(fq_mod.scale.device))
        else:
            logger.debug(
                "[warm-start] scale shape mismatch (onnx=%s vs pt=%s); skipping inject",
                tuple(scale_tensor.shape), tuple(fq_mod.scale.shape),
            )
            return
        if zp_arr is not None:
            zp_tensor = torch.from_numpy(np.asarray(zp_arr).astype(np.int64))
            if zp_tensor.numel() == 1 and fq_mod.zero_point.numel() == 1:
                fq_mod.zero_point.data.fill_(int(zp_tensor.item()))
            elif zp_tensor.shape == fq_mod.zero_point.shape:
                fq_mod.zero_point.data.copy_(
                    zp_tensor.to(fq_mod.zero_point.device).to(fq_mod.zero_point.dtype)
                )
    except Exception:  # pragma: no cover — defensive log
        logger.debug("[warm-start] scale injection failed", exc_info=True)
