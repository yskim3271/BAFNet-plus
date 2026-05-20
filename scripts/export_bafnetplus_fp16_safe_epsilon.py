"""Export the cycle 18b-11 Tier 1B FP16-safe epsilon variant of BAFNet+ FP32 complex ONNX.

Raises the two epsilons inside ``_phase_components_to_complex`` from the
cycle 18b-3 production values (``1e-8`` / ``1e-16``) to FP16-safe values
(default ``1e-3`` / ``1e-6``) so the QNN HTP-FP16 delegate does not
flush-to-zero on the squared epsilon and reach the ``0/0 → NaN/inf``
corner Codex identified as the real cause of the cycle 18b-8 alpha
2/8442 nonfinite spots (FP16 ``min_normal ≈ 6.10e-5``, ``min_subnormal ≈
5.96e-8`` — ``1e-16`` always flushes, ``1e-8`` flushes in most modes).

Production ``bafnetplus_50ms_fp32_complex.onnx`` (md5 ``980aa363…``) and
its sidecar (md5 ``804f7b1d…``) are NOT modified — this script writes a
separate file pair under a distinct schema
(``…-fp16-safe-epsilon-v1``). Mutually exclusive with cycle 18b-8
debug-alpha, cycle 18b-10 debug-mapping, cycle 18b-11 Tier 1A
debug-tsblock, and cycle 18b-9 alpha workaround variants.

Usage::

    python -m scripts.export_bafnetplus_fp16_safe_epsilon \\
        --chkpt-dir results/experiments/bafnetplus_50ms \\
        --phase-norm-epsilon 1e-3 \\
        --phase-norm-squared-epsilon 1e-6 \\
        --output results/onnx/bafnetplus_50ms_fp32_complex_fp16_safe_epsilon.onnx
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
        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_fp16_safe_epsilon.onnx"),
        help="Output ONNX path (sidecar JSON is written alongside).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Streaming chunk size (frames per call).",
    )
    parser.add_argument(
        "--phase-norm-epsilon",
        type=float,
        default=1e-3,
        help="Linear bias added to phase_real/phase_imag before squaring (default 1e-3, "
        "FP16-safe; production default is 1e-8).",
    )
    parser.add_argument(
        "--phase-norm-squared-epsilon",
        type=float,
        default=1e-6,
        help="Floor added inside sqrt(re**2 + im**2 + eps_sq) (default 1e-6, FP16-safe; "
        "production default is 1e-16).",
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
        phase_normalization_epsilon=args.phase_norm_epsilon,
        phase_normalization_squared_epsilon=args.phase_norm_squared_epsilon,
        verbose=args.verbose,
    )
    print(f"Wrote {result.onnx_path}")
    print(f"Wrote {result.metadata_path}")
    print(
        f"  phase_normalization_epsilon={args.phase_norm_epsilon}, "
        f"phase_normalization_squared_epsilon={args.phase_norm_squared_epsilon}"
    )


if __name__ == "__main__":
    main()
