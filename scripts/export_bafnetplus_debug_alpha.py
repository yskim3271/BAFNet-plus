"""Export the cycle 18b-8 debug variant of BAFNet+ FP32 complex ONNX.

Adds 4 alpha-stack intermediate outputs (debug_alpha_features +
first-block conv/BN/PReLU) to the standard ``s20-bafnetplus-functional-fp32-complex``
graph so on-device measurement can bisect which sub-op in the alpha layer
first diverges between host CPU FP32 and device QNN HTP.

Production ``bafnetplus_50ms_fp32_complex.onnx`` (md5 ``980aa363…``) and its
sidecar (md5 ``804f7b1d…``) are NOT modified — this script writes a separate
file pair under a distinct schema (``…-debug-alpha-v1``).

Usage::

    python -m scripts.export_bafnetplus_debug_alpha \\
        --chkpt-dir results/experiments/bafnetplus_50ms \\
        --output results/onnx/bafnetplus_50ms_fp32_complex_debug_alpha.onnx
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
        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_debug_alpha.onnx"),
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
        emit_debug_alpha=True,
        verbose=args.verbose,
    )
    print(f"Wrote {result.onnx_path}")
    print(f"Wrote {result.metadata_path}")


if __name__ == "__main__":
    main()
