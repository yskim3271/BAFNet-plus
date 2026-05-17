"""Export the S25 cycle-6 H1 variant: T2 trunk INT8 + head INT8 (autofp).

H1 hypothesis (2026-05-16): the head FP32 ONNX (9 Conv + 6 PReLU + 19
FP-sensitive ops) can be quantized to INT8 W8A16 with the existing autofp
recipe (``auto_exclude_sensitive=True`` → 19 nodes FP-fallback identified
by ``_auto_precision_sensitive_nodes``). On V79, head INT8 might run
slightly faster than head FP16 (Step 0 baseline: 1.00 ms p50) — though
the ceiling is sub-millisecond given the FP16 starting point.

This is the head twin of cycle 5 T2 (trunk PReLU decomposition). The
trunk side is reused unchanged: T2 PReLU-decomposed INT8 trunk
``bafnetplus_50ms_int8_qdq_trunk_t2.onnx``. Only the head is re-quantized.

The head autofp pattern dispatch in ``_auto_precision_sensitive_nodes``
matches 19 nodes on head FP32: 1 Atan + 1 Softmax + 12 Pow + 5 Sqrt
(`complex_to_mag_pha` energy features + LayerNorm-free Pow/Sqrt
cluster — top-level pattern). The trunk-conditional phase_conv pattern
does NOT activate on head (schema gate).

Outputs:
  - bafnetplus_50ms_int8_qdq_head_h1.onnx (+ sidecar) — head INT8 with
    19-op FP-fallback
  - bafnetplus_50ms_split_int8_t2_h1.json — combined sidecar (T2 trunk +
    H1 head)

Usage::

    python -m scripts.export_bafnetplus_split_t2_int8head \\
        --fp32-head results/onnx/bafnetplus_50ms_fp32_head.onnx \\
        --int8-trunk results/onnx/bafnetplus_50ms_int8_qdq_trunk_t2.onnx \\
        --calibration-dir /tmp/bafnet_calib_head_v1 \\
        --output-dir results/onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.models.streaming.onnx.export import (
    ExportResult,
    _auto_precision_sensitive_nodes,
    quantize_bafnetplus_qdq,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fp32-head",
        type=Path,
        default=Path("results/onnx/bafnetplus_50ms_fp32_head.onnx"),
    )
    parser.add_argument(
        "--int8-trunk",
        type=Path,
        default=Path("results/onnx/bafnetplus_50ms_int8_qdq_trunk_t2.onnx"),
        help="Active T2 trunk INT8 to reference in the combined sidecar (no re-export)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("/tmp/bafnet_calib_head_v1"),
        help="Head calibration NPZ corpus from scripts.dump_head_calibration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/onnx"),
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help=(
            "Use per-channel weight quantization for head Convs. The head has "
            "no LayerNorm (unlike the trunk), so the trunk's S19/S22 B2-PC "
            "regression doesn't apply — try this if the per-tensor recipe "
            "tanks accuracy."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    if not args.fp32_head.exists():
        raise FileNotFoundError(args.fp32_head)
    if not args.int8_trunk.exists():
        raise FileNotFoundError(args.int8_trunk)
    if not args.calibration_dir.is_dir():
        raise FileNotFoundError(args.calibration_dir)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    int8_head_path = output_dir / "bafnetplus_50ms_int8_qdq_head_h1.onnx"
    combined_path = output_dir / "bafnetplus_50ms_split_int8_t2_h1.json"

    # Pre-flight: dump sensitive node list for the JSON record.
    sensitive_nodes = _auto_precision_sensitive_nodes(args.fp32_head)
    logger.info("head autofp: %d sensitive nodes to FP-fallback", len(sensitive_nodes))
    logger.info("  sensitive: %s", sensitive_nodes)

    # Quantize: head FP32 -> head INT8 W8A16 with autofp.
    result: ExportResult = quantize_bafnetplus_qdq(
        args.fp32_head,
        int8_head_path,
        calibration_dir=args.calibration_dir,
        activation_type="QUInt16",
        weight_type="QUInt8",
        per_channel=args.per_channel,
        auto_exclude_sensitive=True,
        verbose=True,
    )
    logger.info(
        "head INT8 written: %s (size=%.2f MB)",
        int8_head_path,
        int8_head_path.stat().st_size / (1024 * 1024),
    )

    # Build combined sidecar (T2 trunk reused unchanged + new H1 head).
    t2_combined = json.loads(
        (output_dir / "bafnetplus_50ms_split_int8_t2.json").read_text()
    )
    int8_trunk_sidecar = json.loads(
        args.int8_trunk.with_suffix(".onnx.json").read_text()
    )
    int8_head_sidecar = json.loads(
        int8_head_path.with_suffix(".onnx.json").read_text()
    )

    combined: Dict[str, Any] = dict(t2_combined)
    combined["produced_by"] = "scripts.export_bafnetplus_split_t2_int8head"
    combined["trunk"] = {
        "onnx_file": args.int8_trunk.name,
        "sidecar_file": args.int8_trunk.with_suffix(".onnx.json").name,
        "onnx_size_bytes": args.int8_trunk.stat().st_size,
        "schema_version": int8_trunk_sidecar.get("schema_version"),
        "num_states": int8_trunk_sidecar.get("core", {}).get("num_states"),
        "quantization": int8_trunk_sidecar.get("model_info", {}).get("quantization"),
    }
    combined["head"] = {
        "onnx_file": int8_head_path.name,
        "sidecar_file": int8_head_path.with_suffix(".onnx.json").name,
        "onnx_size_bytes": int8_head_path.stat().st_size,
        "schema_version": int8_head_sidecar.get("schema_version"),
        "num_states": int8_head_sidecar.get("core", {}).get("num_states"),
        "quantization": int8_head_sidecar.get("model_info", {}).get("quantization"),
    }
    combined["h1_head_int8_autofp"] = {
        "applied": True,
        "sensitive_node_count": len(sensitive_nodes),
        "sensitive_nodes": sensitive_nodes,
        "calibration_dir": str(args.calibration_dir),
        "rationale": (
            "S25 cycle-6 H1: head FP32 graph quantized to INT8 W8A16 with autofp "
            "FP-fallback over 19 sensitive ops (1 Atan + 1 Softmax + 12 Pow + 5 Sqrt). "
            "Head calibration corpus is the 7-tensor boundary distribution from the "
            "T2 trunk INT8 ONNX on the same 450 trunk-calibration audio chunks "
            "(cold-start states, matches trunk calibration convention)."
        ),
    }
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=False))
    logger.info("combined sidecar: %s", combined_path)

    # Echo MD5 for the record.
    md5 = hashlib.md5(int8_head_path.read_bytes()).hexdigest()
    logger.info("head INT8 md5: %s", md5)


if __name__ == "__main__":
    main()
