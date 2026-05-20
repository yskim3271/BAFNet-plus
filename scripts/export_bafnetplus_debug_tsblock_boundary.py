"""Export the cycle 18b-12 Track (v) boundary-only TSBlock debug variant.

Adds one mapping-branch debug output per TSBlock (the block's output
tensor, captured post time_stage + freq_stage + residual) to the
standard ``s20-bafnetplus-functional-fp32-complex`` graph WITHOUT
touching any intra-block code. For the 50ms BAFNet+ recipe
(num_tsblock=4) the result is 4 additional trailing outputs:
``debug_mapping_tsblock_{0,1,2,3}_output``.

Hypothesis being tested: cycle 18b-11's 8-tap intra-block probe on the
LAST TSBlock appeared to suppress the cycle 18b-10 step-4 catastrophic
event (Heisenberg-style observation effect — the same logical tensor
``state_84_input`` was max 6.48e+04 + 16 inf at step 4 in cycle 18b-10's
``seq_block_out`` but max 227.8 with 0 inf in cycle 18b-11's
``debug_tsblock_3_state_84_input``). The boundary-only variant reduces
the per-block capture count from 8 to 1 AND keeps intra-block code
unchanged, so it tests whether the catastrophic event survives a
minimum-perturbation probe and (if it survives) which TSBlock is the
root.

Production ``bafnetplus_50ms_fp32_complex.onnx`` (md5 ``980aa363…``)
and its sidecar (md5 ``804f7b1d…``) are NOT modified. Mutually
exclusive with cycle 18b-8 debug-alpha, cycle 18b-9 alpha workaround,
cycle 18b-10 debug-mapping, cycle 18b-11 Tier 1A debug-tsblock, and
cycle 18b-11 Tier 1B fp16-safe-epsilon variants.

Usage::

    python -m scripts.export_bafnetplus_debug_tsblock_boundary \\
        --chkpt-dir results/experiments/bafnetplus_50ms \\
        --output results/onnx/bafnetplus_50ms_fp32_complex_debug_tsblock_boundary.onnx
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
        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_debug_tsblock_boundary.onnx"),
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
        emit_debug_mapping_tsblock_boundaries=True,
        verbose=args.verbose,
    )
    print(f"Wrote {result.onnx_path}")
    print(f"Wrote {result.metadata_path}")


if __name__ == "__main__":
    main()
