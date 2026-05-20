"""Export the cycle 18b-11 Tier 1A debug-tsblock variant of BAFNet+ FP32 complex ONNX.

Adds 8 mapping-branch TSBlock-internal intermediate outputs (last block's
freq_stage SimpleGate + SCA multiply + 4× GPKFFN attn*conv + GPKFFN
proj_last + final state_84 input) to the standard
``s20-bafnetplus-functional-fp32-complex`` graph so on-device measurement
can bisect which sub-op in the freq_stage cascade (Codex's narrowing of
the cycle 18b-10 "TSBlock cascade at step 4" hypothesis) first diverges
between host CPU FP32 and device QNN HTP-FP16.

Production ``bafnetplus_50ms_fp32_complex.onnx`` (md5 ``980aa363…``) and
its sidecar (md5 ``804f7b1d…``) are NOT modified — this script writes a
separate file pair under a distinct schema
(``…-debug-tsblock-v1``). Mutually exclusive with cycle 18b-8 debug-alpha,
cycle 18b-10 debug-mapping, cycle 18b-11 Tier 1B fp16-safe-epsilon, and
cycle 18b-9 alpha workaround variants.

Usage::

    python -m scripts.export_bafnetplus_debug_tsblock \\
        --chkpt-dir results/experiments/bafnetplus_50ms \\
        --output results/onnx/bafnetplus_50ms_fp32_complex_debug_tsblock.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.models.streaming.onnx.export import export_bafnetplus_to_onnx_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chkpt-dir",
        type=Path,
        default=Path("results/experiments/bafnetplus_50ms"),
        help="Unified BAFNet+ experiment directory.",
    )
    parser.add_argument(
        "--chkpt-file",
        default="best.th",
        help="Checkpoint filename inside --chkpt-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_debug_tsblock.onnx"),
        help="Output ONNX path (sidecar JSON is written alongside).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Streaming chunk size (frames per call).",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = export_bafnetplus_to_onnx_from_checkpoint(
        chkpt_dir=args.chkpt_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chkpt_file=args.chkpt_file,
        phase_output_mode="complex",
        emit_debug_mapping_tsblock_internals=True,
        verbose=args.verbose,
    )
    print(f"Wrote {result.onnx_path}")
    print(f"Wrote {result.metadata_path}")


if __name__ == "__main__":
    main()
