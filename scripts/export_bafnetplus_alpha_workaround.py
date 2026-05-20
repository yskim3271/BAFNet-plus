"""Export cycle 18b-9 alpha-feature numerical-workaround BAFNet+ FP32 complex ONNX.

Cycle 18b-8 forensic identified ``_build_alpha_features``'s ``sqrt(x²+y²+eps)``
as the root sub-op of the catastrophic mapping-decoder cascade on Mali-G715
NNAPI FP16: 2/8442 spot positions at device step 0 overflow FP16 max (65504)
after squaring, producing ``inf`` magnitudes that fan out through the 7×7
causal conv and saturate the rest of the alpha stack.

This script writes a numerical-workaround variant of the production
``bafnetplus_50ms_fp32_complex.onnx`` graph. Two surgical export-time knobs
(both optional; both can be combined):

- ``--clamp-threshold THRESH``: clamp ``bcs/acs_com_cal`` real and imag
  components to ``[-THRESH, +THRESH]`` BEFORE squaring. Suppresses the rare
  pre-square overflow spots. Cycle 18b-9 default candidate ``THRESH=200``
  gives ``200²·2+1e-8 ≈ 8e4`` → still slightly above FP16 max but most of
  the FP16 overflow spots are gone.
- ``--safe-sqrt``: clamp ``x²+y²+eps`` to ``≤ 6.5e4`` (FP16-safe) BEFORE
  ``sqrt``. Saturates the residual overflow risk after pre-clamp.

Production ``bafnetplus_50ms_fp32_complex.onnx`` (md5 ``980aa363…``) and its
sidecar (md5 ``804f7b1d…``) are NOT modified — this script writes a separate
file pair under a distinct schema
(``s20-bafnetplus-functional-fp32-complex-clamp-v1`` /
 ``...-safe-sqrt-v1`` / ``...-clamp-safe-sqrt-v1``).

Usage::

    python -m scripts.export_bafnetplus_alpha_workaround \\
        --clamp-threshold 200.0 \\
        --output results/onnx/bafnetplus_50ms_fp32_complex_clamp.onnx

    python -m scripts.export_bafnetplus_alpha_workaround \\
        --safe-sqrt \\
        --output results/onnx/bafnetplus_50ms_fp32_complex_safe_sqrt.onnx

    python -m scripts.export_bafnetplus_alpha_workaround \\
        --clamp-threshold 200.0 --safe-sqrt \\
        --output results/onnx/bafnetplus_50ms_fp32_complex_clamp_safe_sqrt.onnx
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
        required=True,
        help="Output ONNX path (sidecar JSON is written alongside).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Streaming chunk size (frames per call).",
    )
    parser.add_argument(
        "--clamp-threshold",
        type=float,
        default=None,
        help="Pre-clamp bcs/acs_com_cal re/im components to [-T,+T] before squaring.",
    )
    parser.add_argument(
        "--safe-sqrt",
        action="store_true",
        help="Saturate x²+y²+eps to FP16-safe max (6.5e4) before sqrt.",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.clamp_threshold is None and not args.safe_sqrt:
        parser.error("At least one of --clamp-threshold / --safe-sqrt must be set.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = export_bafnetplus_to_onnx_from_checkpoint(
        chkpt_dir=args.chkpt_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chkpt_file=args.chkpt_file,
        phase_output_mode="complex",
        emit_debug_alpha=False,
        alpha_clamp_threshold=args.clamp_threshold,
        alpha_safe_sqrt=args.safe_sqrt,
        verbose=args.verbose,
    )
    print(f"Wrote {result.onnx_path}")
    print(f"Wrote {result.metadata_path}")


if __name__ == "__main__":
    main()
