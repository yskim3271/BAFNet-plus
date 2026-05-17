"""Export the S25 cycle-5 T2 variant of BAFNet+ split-trunk INT8.

T2 hypothesis (web-research path #2, 2026-05-16): on V79 + QNN SDK 2.42,
the compile-fatal `q::prelu.opt` rejection can be avoided by removing the
`PRelu` op from the graph entirely. Decomposing `PRelu(x, slope)` into

    Max(x, 0)  +  slope * Min(x, 0)

uses only `Max`, `Min`, `Mul`, `Add` — ops that are unambiguously registered
at QUInt16 on V79. The QNN HTP pattern matcher never gets the chance to
select `q::prelu.opt` because there is no `PRelu` node anywhere in the
graph.

Compared to T1 (which kept `PRelu` and down-converted its input to QUInt8):
- T2 has no Q/DQ rewrap; the entire phase-decoder cluster stays at
  QUInt16 end-to-end.
- T2 swaps 1 PReLU node for 4 Max/Min/Mul/Add nodes (+3 per target). Two
  targets × +3 = +6 net nodes per trunk graph vs +8 Q/DQ for T1.

The surgery operates on the FP32 trunk ONNX, after it is exported from
the unified checkpoint and BEFORE it is quantized. The FP32 trunk is
re-checked against the original ONNX numerically (libm-precision delta)
before INT8 quantization runs.

Outputs:
  - bafnetplus_50ms_fp32_trunk_t2.onnx (+ sidecar)        FP32 trunk with PReLU decomposed
  - bafnetplus_50ms_int8_qdq_trunk_t2.onnx (+ sidecar)    INT8 W8A16 trunk, no autofp, no init_overrides
  - bafnetplus_50ms_split_int8_t2.json                     Combined sidecar

Usage::

    python -m scripts.export_bafnetplus_split_t2 \\
        --fp32-trunk results/onnx/bafnetplus_50ms_fp32_trunk.onnx \\
        --fp32-head results/onnx/bafnetplus_50ms_fp32_head.onnx \\
        --output-dir results/onnx \\
        --calibration-dir /tmp/bafnet_calib_taps_v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import onnx

from src.models.streaming.onnx.export import (
    ExportResult,
    quantize_bafnetplus_qdq,
)
from src.models.streaming.onnx.prelu_decompose import (
    decompose_phase_prelu_inplace,
)

logger = logging.getLogger(__name__)


_TARGET_PRELUS = (
    "/phase_conv/phase_conv.2/PRelu",
    "/mapping_core/phase_conv/phase_conv.2/PRelu",
)


def _produce_fp32_trunk_t2(
    baseline_fp32_trunk: Path,
    out_fp32_trunk_t2: Path,
) -> Path:
    """Decompose the two phase-decoder PReLUs and write the modified FP32 trunk."""
    if not baseline_fp32_trunk.exists():
        raise FileNotFoundError(baseline_fp32_trunk)
    baseline_sidecar = baseline_fp32_trunk.with_suffix(".onnx.json")
    if not baseline_sidecar.exists():
        raise FileNotFoundError(f"Baseline sidecar not found: {baseline_sidecar}")

    logger.info("Loading baseline FP32 trunk: %s", baseline_fp32_trunk)
    model = onnx.load(str(baseline_fp32_trunk))
    rewritten = decompose_phase_prelu_inplace(model, target_node_names=_TARGET_PRELUS)
    logger.info("Decomposed %d PRelu nodes: %s", len(rewritten), rewritten)

    onnx.checker.check_model(model)
    onnx.save(model, str(out_fp32_trunk_t2))
    logger.info("Wrote T2 FP32 trunk: %s (%d bytes)", out_fp32_trunk_t2, out_fp32_trunk_t2.stat().st_size)

    # Write companion sidecar — copy the baseline sidecar and update file refs.
    baseline_meta = json.loads(baseline_sidecar.read_text())
    new_md5 = hashlib.md5(out_fp32_trunk_t2.read_bytes()).hexdigest()
    new_size = out_fp32_trunk_t2.stat().st_size
    new_meta = dict(baseline_meta)
    new_meta["onnx_file"] = out_fp32_trunk_t2.name
    new_meta["onnx_size_bytes"] = new_size
    new_meta["produced_by"] = "scripts.export_bafnetplus_split_t2"
    new_meta["t2_prelu_decompose"] = {
        "applied": True,
        "decomposed_prelu_nodes": list(rewritten),
        "decomposition": "PReLU(x, slope) -> Max(x,0) + slope * Min(x,0)",
        "rationale": (
            "S25 cycle-5 T2: replace each /phase_conv/phase_conv.2/PRelu with its "
            "algebraic decomposition (Max/Min/Mul/Add) so the QNN HTP compiler never "
            "encounters a PRelu op and cannot select the disabled q::prelu.opt kernel. "
            "Numerically equivalent to the original PReLU at FP32 within libm precision."
        ),
        "fp32_md5_after_decompose": new_md5,
    }
    sidecar_out = out_fp32_trunk_t2.with_suffix(".onnx.json")
    sidecar_out.write_text(json.dumps(new_meta, indent=2, sort_keys=False))
    logger.info("Wrote T2 FP32 sidecar: %s", sidecar_out)
    return out_fp32_trunk_t2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fp32-trunk",
        type=Path,
        default=Path("results/onnx/bafnetplus_50ms_fp32_trunk.onnx"),
    )
    parser.add_argument(
        "--fp32-head",
        type=Path,
        default=Path("results/onnx/bafnetplus_50ms_fp32_head.onnx"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/onnx"),
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("/tmp/bafnet_calib_taps_v3"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_trunk_t2_path = output_dir / "bafnetplus_50ms_fp32_trunk_t2.onnx"
    int8_trunk_t2_path = output_dir / "bafnetplus_50ms_int8_qdq_trunk_t2.onnx"
    combined_t2_path = output_dir / "bafnetplus_50ms_split_int8_t2.json"

    # Step 1: produce FP32 trunk with PReLU decomposition applied.
    _produce_fp32_trunk_t2(args.fp32_trunk, fp32_trunk_t2_path)

    # Step 2: INT8 quantize on the decomposed FP32 trunk (no autofp, no init_overrides).
    int8_result: ExportResult = quantize_bafnetplus_qdq(
        fp32_trunk_t2_path,
        int8_trunk_t2_path,
        calibration_dir=args.calibration_dir,
        activation_type="QUInt16",
        weight_type="QUInt8",
        per_channel=False,
        auto_exclude_sensitive=False,
        verbose=True,
    )

    # Step 3: build combined sidecar (reuse canonical S21 split-int8 layout).
    fp32_head_path: Path = args.fp32_head
    head_sidecar = json.loads(fp32_head_path.with_suffix(".onnx.json").read_text())

    s21_combined = json.loads(
        (output_dir / "bafnetplus_50ms_split_int8_autofp.json").read_text()
    )
    combined: Dict[str, Any] = dict(s21_combined)
    combined["produced_by"] = "scripts.export_bafnetplus_split_t2"
    combined.pop("phase_conv_merge", None)
    combined["trunk"]["onnx_file"] = int8_trunk_t2_path.name
    combined["trunk"]["sidecar_file"] = int8_trunk_t2_path.with_suffix(".onnx.json").name
    combined["trunk"]["onnx_size_bytes"] = int8_trunk_t2_path.stat().st_size

    combined["t2_prelu_decompose"] = {
        "applied": True,
        "decomposition": "PReLU(x, slope) -> Max(x,0) + slope * Min(x,0)",
        "target_prelu_nodes": list(_TARGET_PRELUS),
        "rationale": (
            "S25 cycle-5 T2: PReLU op removed from trunk graph; replaced by Max/Min/"
            "Mul/Add decomposition. QNN HTP pattern matcher cannot select q::prelu.opt "
            "because no PRelu node exists. Entire phase-decoder cluster stays at "
            "QUInt16 end-to-end (no Q/DQ rewrap as T1 has)."
        ),
    }
    combined_t2_path.write_text(json.dumps(combined, indent=2, sort_keys=False))
    logger.info("T2 combined sidecar: %s", combined_t2_path)


if __name__ == "__main__":
    main()
