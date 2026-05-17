"""Dump head calibration NPZs from the active trunk INT8 ONNX.

S25 cycle-6 H1 Step 2: the head FP32 → INT8 quantization needs a calibration
corpus whose audio inputs reflect what the head will actually see in
deployment — i.e. the 7 boundary tensors emitted by the trunk INT8 ONNX,
not the 4 BCS/ACS spectra that the trunk consumes.

This script reads the existing 450-NPZ trunk calibration corpus
(``/tmp/bafnet_calib_taps_v3/``), runs each through the active T2 trunk
INT8 ONNX on CPU with zero state, and writes a new 450-NPZ head
calibration corpus (``/tmp/bafnet_calib_head_v1/``) whose keys are the 7
boundary tensor names. The H1 export driver picks up the 7-key NPZs via
``_BAFNetPlusCalibrationDataReader``'s auto-detect path.

Cold-start assumption: each trunk run uses zero state. This matches the
trunk calibration convention (the trunk corpus also feeds zero state per
chunk). The implication is that the head calibration distribution covers
the cold-start boundary distribution but not the in-utterance steady-state
distribution. For H1's PTQ purposes this is consistent with the existing
calibration assumptions; a future warm-state corpus would be a separate
mini-cycle.

Usage::

    python -m scripts.dump_head_calibration \\
        --trunk-onnx results/onnx/bafnetplus_50ms_int8_qdq_trunk_t2.onnx \\
        --input-dir  /tmp/bafnet_calib_taps_v3 \\
        --output-dir /tmp/bafnet_calib_head_v1
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


_AUDIO_INPUT_NAMES = ("bcs_mag", "bcs_pha", "acs_mag", "acs_pha")
_BOUNDARY_OUTPUT_NAMES = (
    "bcs_est_mag",
    "bcs_phase_real",
    "bcs_phase_imag",
    "acs_est_mag",
    "acs_phase_real",
    "acs_phase_imag",
    "acs_mask",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trunk-onnx",
        type=Path,
        default=Path("results/onnx/bafnetplus_50ms_int8_qdq_trunk_t2.onnx"),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/tmp/bafnet_calib_taps_v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/bafnet_calib_head_v1"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N NPZ files (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    if not args.trunk_onnx.exists():
        raise FileNotFoundError(args.trunk_onnx)
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading trunk INT8 ONNX: %s", args.trunk_onnx)
    sess = ort.InferenceSession(str(args.trunk_onnx), providers=["CPUExecutionProvider"])
    sess_inputs = {m.name: m for m in sess.get_inputs()}
    sess_output_names = {m.name for m in sess.get_outputs()}

    missing_audio = [n for n in _AUDIO_INPUT_NAMES if n not in sess_inputs]
    if missing_audio:
        raise KeyError(f"trunk ONNX missing audio inputs: {missing_audio}")
    missing_boundary = [n for n in _BOUNDARY_OUTPUT_NAMES if n not in sess_output_names]
    if missing_boundary:
        raise KeyError(f"trunk ONNX missing boundary outputs: {missing_boundary}")

    # State inputs (everything not in _AUDIO_INPUT_NAMES). Shapes resolved once.
    state_inputs = []
    for meta in sess.get_inputs():
        if meta.name in _AUDIO_INPUT_NAMES:
            continue
        shape = tuple(int(d) if isinstance(d, int) else 1 for d in meta.shape)
        state_inputs.append((meta.name, shape))
    logger.info("trunk state inputs: %d (zero-filled per chunk)", len(state_inputs))

    files = sorted(args.input_dir.glob("calib_*.npz"))
    if args.limit is not None:
        files = files[: args.limit]
    logger.info("dumping %d head calibration NPZ -> %s", len(files), args.output_dir)

    t_start = time.monotonic()
    for i, fp in enumerate(files):
        arr = np.load(fp)
        feed = {k: arr[k].astype(np.float32, copy=False) for k in _AUDIO_INPUT_NAMES}
        for name, shape in state_inputs:
            feed[name] = np.zeros(shape, dtype=np.float32)

        outputs = sess.run(list(_BOUNDARY_OUTPUT_NAMES), feed)
        head_npz = {bn: outputs[j] for j, bn in enumerate(_BOUNDARY_OUTPUT_NAMES)}

        out_path = args.output_dir / fp.name
        np.savez(out_path, **head_npz)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            elapsed = time.monotonic() - t_start
            logger.info("  %d/%d done (%.1fs elapsed, %.2fs/file)", i + 1, len(files), elapsed, elapsed / (i + 1))

    logger.info("dumped %d head calibration NPZ to %s", len(files), args.output_dir)


if __name__ == "__main__":
    main()
