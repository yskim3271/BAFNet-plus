"""Cycle 18b — Option β driver: trained PT2E QAT ckpt -> INT8 QDQ ONNX.

Loads a pilot/cycle-18b QAT checkpoint, rebuilds the PT2E prepared GraphModule,
restores trained fake-quant scales via ``load_state_dict``, then runs Option β
(:func:`src.models.streaming.qat.export.export_qat_to_onnx_qdq_override`) to
write a V79-bound INT8 QDQ ONNX with the QAT-learned scales injected via
ORT's ``TensorQuantOverrides``.

Usage::

    python -m scripts.qat_export_option_b \
        --pilot-ckpt results/experiments/cycle18b_pilot/best.th \
        --fp32-trunk results/onnx/bafnetplus_50ms_fp32_trunk_t2.onnx \
        --reference-int8 results/onnx/bafnetplus_50ms_int8_qdq_trunk_t2.onnx \
        --output results/onnx/bafnetplus_50ms_int8_qat_pilot_trunk_t2.onnx \
        --calib-dir /tmp/bafnet_calib_taps_v3 \
        --bm-map results/experiments/bm_map_50ms/best.th \
        --bm-mask results/experiments/bm_mask_50ms/best.th \
        --batch-size 2 --time-frames 400
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Make ``src.*`` importable when run as a script from the BAFNetPlus root.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.qat import prepare_bafnetplus_for_qat
from src.models.streaming.qat.export import export_qat_to_onnx_qdq_override


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("qat_export_option_b")


def _build_example_inputs(*, batch: int, time_frames: int, n_fft: int = 400) -> tuple:
    freq = n_fft // 2 + 1
    bcs = torch.zeros(batch, freq, time_frames, 2)
    acs = torch.zeros(batch, freq, time_frames, 2)
    return ((bcs, acs),)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-ckpt", required=True, type=Path,
                    help="QAT pilot checkpoint (best.th or checkpoint.th).")
    ap.add_argument("--fp32-trunk", required=True, type=Path,
                    help="FP32 trunk ONNX to quantize (D1 fp32 trunk_t2).")
    ap.add_argument("--reference-int8", required=True, type=Path,
                    help="D1 INT8 QDQ trunk ONNX (used as tensor-name reference only).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output INT8 QAT-overridden QDQ ONNX path.")
    ap.add_argument("--calib-dir", required=True, type=Path,
                    help="Calibration NPZ corpus (still required for non-QAT'd activations).")
    ap.add_argument("--bm-map", required=True, type=Path,
                    help="bm_map_50ms/best.th sub-checkpoint.")
    ap.add_argument("--bm-mask", required=True, type=Path,
                    help="bm_mask_50ms/best.th sub-checkpoint.")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Must match the pilot's batch_size (PT2E specializes B).")
    ap.add_argument("--time-frames", type=int, default=400,
                    help="Match the pilot's qat.example_time_frames.")
    ap.add_argument("--stats-json", type=Path, default=None,
                    help="Optional path to dump the mapping stats as JSON.")
    args = ap.parse_args()

    for path, name in [
        (args.pilot_ckpt, "pilot-ckpt"),
        (args.fp32_trunk, "fp32-trunk"),
        (args.reference_int8, "reference-int8"),
        (args.calib_dir, "calib-dir"),
        (args.bm_map, "bm-map"),
        (args.bm_mask, "bm-mask"),
    ]:
        if not path.exists():
            raise SystemExit(f"[FATAL] {name} not found at {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build a fresh BAFNetPlus matching the QAT training config.
    logger.info("Building BAFNetPlus (ablation=full, depth=4, channels=16, k=7)")
    model = BAFNetPlus(
        ablation_mode="full",
        conv_depth=4,
        conv_channels=16,
        conv_kernel_size=7,
        calibration_hidden_channels=16,
        calibration_depth=2,
        calibration_kernel_size=5,
        calibration_max_common_log_gain=0.5,
        calibration_max_relative_log_gain=1.0,
        checkpoint_mapping=str(args.bm_map),
        checkpoint_masking=str(args.bm_mask),
    )
    model.eval()

    # 2. PT2E prepare with the same batch + time-frame contract used at training.
    example_inputs = _build_example_inputs(batch=args.batch_size, time_frames=args.time_frames)
    logger.info(
        f"PT2E prepare with example batch={args.batch_size}, T={args.time_frames} "
        f"(must match pilot's batch_size + qat.example_time_frames)"
    )
    prepared = prepare_bafnetplus_for_qat(model, example_inputs=example_inputs)

    # 3. Load the pilot's state_dict.
    logger.info(f"Loading pilot ckpt: {args.pilot_ckpt}")
    pkg = torch.load(args.pilot_ckpt, map_location="cpu", weights_only=False)
    model_state = pkg.get("model") if isinstance(pkg, dict) else None
    if model_state is None:
        raise SystemExit(
            f"[FATAL] pilot ckpt is not a Solver package (expected 'model' key); got keys={list(pkg.keys()) if isinstance(pkg, dict) else type(pkg)}"
        )
    missing, unexpected = prepared.load_state_dict(model_state, strict=False)
    logger.info(
        f"load_state_dict: matched={len(model_state) - len(unexpected)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        logger.warning("Missing keys (first 5): %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (first 5): %s", unexpected[:5])

    # 4. Option β export.
    logger.info(
        f"Running Option β: fp32_trunk={args.fp32_trunk}, "
        f"reference_int8={args.reference_int8}, calib_dir={args.calib_dir}"
    )
    out_path, stats = export_qat_to_onnx_qdq_override(
        prepared,
        fp32_streaming_onnx_path=args.fp32_trunk,
        output_path=args.output,
        reference_int8_onnx_path=args.reference_int8,
        calibration_dir=args.calib_dir,
        activation_type="QUInt16",
        weight_type="QUInt8",
        auto_exclude_sensitive=True,
        verbose=True,
    )

    logger.info(
        "G7 Option β success: %s (md5=%s, %.2f MB)",
        out_path,
        stats["onnx_md5"],
        stats["onnx_size_bytes"] / (1024 * 1024),
    )
    logger.info(
        "Mapping stats: %d/%d matched (%.1f%%), %d unmatched_pt, %d unused_onnx",
        stats["num_matched"], stats["num_pt_total"], stats["mapping_rate"] * 100,
        len(stats["unmatched_pt"]), len(stats["unused_onnx"]),
    )

    if args.stats_json is not None:
        # Trim before json dump — drop the large unmatched/unused lists and matched dict.
        slim = {
            k: v for k, v in stats.items()
            if k not in ("matched", "unmatched_pt", "unused_onnx")
        }
        slim["unmatched_pt_count"] = len(stats["unmatched_pt"])
        slim["unused_onnx_count"] = len(stats["unused_onnx"])
        slim["matched_count"] = len(stats["matched"])
        args.stats_json.write_text(json.dumps(slim, indent=2, sort_keys=False))
        logger.info(f"Wrote stats JSON: {args.stats_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
