"""Export the S25 v1-split-E-4 variant of BAFNet+ from a unified checkpoint.

E-4 surgery (``src.models.streaming.onnx.phase_conv_merge.merge_phase_conv_ri_inplace``)
merges the two parallel 1×1 ``phase_conv_r`` + ``phase_conv_i`` Conv2d
projections inside each branch's PhaseDecoder into a single 2-output Conv2d,
dropping the upstream ``phase_conv.2/PReLU`` fanout from 2 to 1. The hypothesis
(cycle-4 mini-fix A2, 2026-05-16) is that the parallel fanout is the trigger
that makes QNN HTP's pattern matcher select ``q::prelu.opt`` (the 3-input
TCM-staged optimized PReLU kernel variant, disabled at QUInt16 on V79 + QNN
SDK 2.42), so removing it should let the unmodified reference ``q::prelu`` be
bound and the W8A16 trunk compile cleanly on V79 — WITHOUT the
``_auto_precision_sensitive_nodes`` FP-fallback that S23 mini-fix A1 introduced.

This script produces the asset set required for that experiment:

  - bafnetplus_50ms_fp32_trunk_e4.onnx        (+ sidecar)         FP32 trunk
  - bafnetplus_50ms_fp32_head_e4.onnx         (+ sidecar)         FP32 head (byte-identical to canonical S21 head)
  - bafnetplus_50ms_split_e4.json                                  FP32 combined sidecar
  - bafnetplus_50ms_int8_qdq_trunk_e4.onnx    (+ sidecar)         INT8 W8A16 trunk, **no autofp** (hypothesis variant)
  - bafnetplus_50ms_int8_qdq_trunk_e4_autofp.onnx (+ sidecar)     INT8 W8A16 trunk, **+ autofp** (fallback variant)
  - bafnetplus_50ms_split_int8_e4.json                             INT8 combined sidecar (no autofp)
  - bafnetplus_50ms_split_int8_e4_autofp.json                      INT8 combined sidecar (autofp)

Usage::

    python -m scripts.export_bafnetplus_split_e4 \\
        --chkpt-dir results/experiments/bafnetplus_50ms \\
        --output-dir results/onnx \\
        --calibration-dir /tmp/bafnet_calib_taps_v3 \\
        --chunk-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.models.streaming.onnx.bafnetplus_core import (
    BAFNetPlusHeadCore,
    BAFNetPlusTrunkCore,
)
from src.models.streaming.onnx.export import (
    _BAFNETPLUS_SCHEMA_SPLIT_COMBINED,
    _SPLIT_BOUNDARY_TENSOR_NAMES,
    ExportResult,
    export_bafnetplus_head_to_onnx,
    export_bafnetplus_trunk_to_onnx,
    load_bafnetplus_from_checkpoint,
    quantize_bafnetplus_qdq,
)
from src.models.streaming.onnx.phase_conv_merge import merge_phase_conv_ri_inplace

logger = logging.getLogger(__name__)


def _build_combined_sidecar(
    *,
    schema_version: str,
    info: Dict[str, Any],
    chunk_size: int,
    time_frames: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    compress_factor: float,
    trunk: BAFNetPlusTrunkCore,
    head: BAFNetPlusHeadCore,
    trunk_result: ExportResult,
    head_result: ExportResult,
    extra_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Replicate the canonical split-combined sidecar with an extra E-4 marker."""
    trunk_path_p = Path(trunk_result.onnx_path)
    head_path_p = Path(head_result.onnx_path)
    md: Dict[str, Any] = {
        "schema_version": schema_version,
        "produced_by": "scripts.export_bafnetplus_split_e4",
        "checkpoint": info,
        "geometry": {
            "chunk_size": int(chunk_size),
            "encoder_lookahead": int(trunk.mapping_core.encoder_lookahead),
            "decoder_lookahead": int(trunk.mapping_core.decoder_lookahead),
            "total_lookahead": int(trunk.total_lookahead),
            "alpha_time_lookahead": int(head.total_lookahead),
            "T_export": int(time_frames),
            "freq_size": int(trunk_result.freq_size),
        },
        "stft": {
            "n_fft": int(trunk.n_fft),
            "hop_size": int(hop_size),
            "win_size": int(win_size),
            "compress_factor": float(compress_factor),
            "center": True,
            "sample_rate": int(sample_rate),
        },
        "boundary": {
            "tensor_names": list(_SPLIT_BOUNDARY_TENSOR_NAMES),
            "shape": [1, int(trunk_result.freq_size), int(time_frames)],
            "dtype": "float32",
            "ordering_note": (
                "Per-branch trunk outputs in mapping-then-masking order; "
                "acs_mask is the masking branch's post-LearnableSigmoid raw mask."
            ),
        },
        "state_partition": {
            "trunk": {
                "num_states": int(trunk.num_states),
                "names_prefix": ["mapping/", "masking/"],
            },
            "head": {
                "num_states": int(head.num_states),
                "names_prefix": ["calibration/", "alpha/"],
            },
            "total": int(trunk.num_states + head.num_states),
        },
        "trunk": {
            "onnx_file": trunk_path_p.name,
            "sidecar_file": Path(trunk_result.metadata_path).name,
            "onnx_size_bytes": trunk_result.metadata.get("onnx_size_bytes"),
            "schema_version": trunk_result.metadata["schema_version"],
            "num_states": int(trunk.num_states),
            "input_names": trunk_result.metadata["io"]["input_names"],
            "output_names": trunk_result.metadata["io"]["output_names"],
        },
        "head": {
            "onnx_file": head_path_p.name,
            "sidecar_file": Path(head_result.metadata_path).name,
            "onnx_size_bytes": head_result.metadata.get("onnx_size_bytes"),
            "schema_version": head_result.metadata["schema_version"],
            "num_states": int(head.num_states),
            "input_names": head_result.metadata["io"]["input_names"],
            "output_names": head_result.metadata["io"]["output_names"],
        },
    }
    md.update(extra_metadata)
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--chkpt-dir",
        type=Path,
        default=Path("results/experiments/bafnetplus_50ms"),
        help="Unified BAFNet+ checkpoint directory (default: results/experiments/bafnetplus_50ms).",
    )
    parser.add_argument(
        "--chkpt-file",
        type=str,
        default="best.th",
        help="Checkpoint filename (default: best.th).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/onnx"),
        help="Where to write the E-4 ONNX assets + sidecars (default: results/onnx).",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("/tmp/bafnet_calib_taps_v3"),
        help="Calibration corpus for INT8 quantization (default: /tmp/bafnet_calib_taps_v3).",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--skip-int8",
        action="store_true",
        help="Only export FP32 (skip INT8 quantization). Useful for fast iteration.",
    )
    parser.add_argument(
        "--int8-only",
        action="store_true",
        help="Skip FP32 export (assume FP32 trunk_e4 already exists); only run INT8 quantization.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_trunk_path = output_dir / "bafnetplus_50ms_fp32_trunk_e4.onnx"
    fp32_head_path = output_dir / "bafnetplus_50ms_fp32_head_e4.onnx"
    combined_fp32_path = output_dir / "bafnetplus_50ms_split_e4.json"

    int8_no_autofp_path = output_dir / "bafnetplus_50ms_int8_qdq_trunk_e4.onnx"
    int8_autofp_path = output_dir / "bafnetplus_50ms_int8_qdq_trunk_e4_autofp.onnx"
    combined_int8_no_autofp_path = output_dir / "bafnetplus_50ms_split_int8_e4.json"
    combined_int8_autofp_path = output_dir / "bafnetplus_50ms_split_int8_e4_autofp.json"

    # ---------------------------------------------------------------- load + surgery
    bafnet, info = load_bafnetplus_from_checkpoint(
        args.chkpt_dir, args.chkpt_file, device="cpu", verbose=True
    )
    mapping_branch = bafnet.mapping
    hop_size = int(getattr(mapping_branch, "hop_size", 100))
    win_size = int(getattr(mapping_branch, "win_size", 400))
    compress_factor = float(getattr(mapping_branch, "compress_factor", 0.3))

    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet)
    head = BAFNetPlusHeadCore.from_bafnetplus(bafnet)

    # E-4 surgery: merge phase_conv_r / phase_conv_i on both branches.
    merge_phase_conv_ri_inplace(trunk.mapping_core)
    merge_phase_conv_ri_inplace(trunk.masking_core)
    logger.info("E-4 surgery applied to mapping_core + masking_core")

    chunk_size = int(args.chunk_size)
    time_frames = chunk_size + trunk.total_lookahead

    # ---------------------------------------------------------------- FP32 export
    if args.int8_only:
        logger.info("--int8-only: skipping FP32 export (must already exist at %s)", fp32_trunk_path)
        if not fp32_trunk_path.exists():
            raise FileNotFoundError(f"--int8-only requires {fp32_trunk_path} to exist")
        trunk_result = ExportResult(
            onnx_path=str(fp32_trunk_path),
            metadata_path=str(fp32_trunk_path.with_suffix(".onnx.json")),
            metadata=json.loads(fp32_trunk_path.with_suffix(".onnx.json").read_text()),
            freq_size=trunk.n_fft // 2 + 1,
        )
        head_result = ExportResult(
            onnx_path=str(fp32_head_path),
            metadata_path=str(fp32_head_path.with_suffix(".onnx.json")),
            metadata=json.loads(fp32_head_path.with_suffix(".onnx.json").read_text()),
            freq_size=trunk.n_fft // 2 + 1,
        )
    else:
        trunk_result = export_bafnetplus_trunk_to_onnx(
            trunk,
            fp32_trunk_path,
            chunk_size=chunk_size,
            time_frames=time_frames,
            sample_rate=args.sample_rate,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            opset_version=args.opset,
            checkpoint_info=info,
            verbose=True,
        )
        head_result = export_bafnetplus_head_to_onnx(
            head,
            fp32_head_path,
            chunk_size=chunk_size,
            time_frames=time_frames,
            sample_rate=args.sample_rate,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            opset_version=args.opset,
            checkpoint_info=info,
            verbose=True,
        )

        combined_fp32 = _build_combined_sidecar(
            schema_version=_BAFNETPLUS_SCHEMA_SPLIT_COMBINED,
            info=info,
            chunk_size=chunk_size,
            time_frames=time_frames,
            sample_rate=args.sample_rate,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            trunk=trunk,
            head=head,
            trunk_result=trunk_result,
            head_result=head_result,
            extra_metadata={
                "phase_conv_merge": {
                    "applied": True,
                    "rationale": (
                        "S25 Path E-4: collapses phase_conv_r/phase_conv_i parallel "
                        "1x1 Conv2d into phase_conv_ri (out_channels=2) to drop the "
                        "upstream phase_conv.2/PReLU fanout from 2 to 1, breaking "
                        "QNN HTP's q::prelu.opt pattern match on V79 W8A16."
                    ),
                    "surgery_module": "src.models.streaming.onnx.phase_conv_merge.merge_phase_conv_ri_inplace",
                },
            },
        )
        combined_fp32_path.write_text(json.dumps(combined_fp32, indent=2, sort_keys=False))
        logger.info("FP32 combined sidecar: %s", combined_fp32_path)

    # ---------------------------------------------------------------- INT8 quantize
    if args.skip_int8:
        logger.info("--skip-int8: skipping INT8 quantization")
        return

    # Variant A: E-4 only (no autofp) — the hypothesis-test variant.
    int8_no_autofp_result = quantize_bafnetplus_qdq(
        fp32_trunk_path,
        int8_no_autofp_path,
        calibration_dir=args.calibration_dir,
        activation_type="QUInt16",
        weight_type="QUInt8",
        per_channel=False,
        auto_exclude_sensitive=False,
        verbose=True,
    )

    # Variant B: E-4 + autofp — the fallback variant.
    int8_autofp_result = quantize_bafnetplus_qdq(
        fp32_trunk_path,
        int8_autofp_path,
        calibration_dir=args.calibration_dir,
        activation_type="QUInt16",
        weight_type="QUInt8",
        per_channel=False,
        auto_exclude_sensitive=True,
        verbose=True,
    )

    for label, int8_result, combined_path in [
        ("e4_no_autofp", int8_no_autofp_result, combined_int8_no_autofp_path),
        ("e4_autofp", int8_autofp_result, combined_int8_autofp_path),
    ]:
        combined_int8 = _build_combined_sidecar(
            schema_version=_BAFNETPLUS_SCHEMA_SPLIT_COMBINED,
            info=info,
            chunk_size=chunk_size,
            time_frames=time_frames,
            sample_rate=args.sample_rate,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            trunk=trunk,
            head=head,
            trunk_result=int8_result,
            head_result=head_result,
            extra_metadata={
                "phase_conv_merge": {
                    "applied": True,
                    "variant": label,
                    "auto_exclude_sensitive": label == "e4_autofp",
                    "nodes_excluded": int8_result.metadata.get("quantization", {}).get(
                        "auto_excluded_node_count", 0
                    ),
                    "rationale": (
                        "S25 Path E-4 INT8 variant. e4_no_autofp tests the "
                        "hypothesis that the q::prelu.opt rejection on V79 is "
                        "caused solely by the PReLU output fanout (which E-4 "
                        "eliminates). e4_autofp is the fallback variant if "
                        "additional cascade ops still require FP-fallback."
                    ),
                },
            },
        )
        combined_path.write_text(json.dumps(combined_int8, indent=2, sort_keys=False))
        logger.info("INT8 combined sidecar (%s): %s", label, combined_path)


if __name__ == "__main__":
    main()
