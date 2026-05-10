"""D5b runner — per-config FP32 + INT8 ONNX evaluation on TAPS multi-SNR test.

For each ``T_alg ∈ {12, 25, 50, 75, 100}`` ms config:
1. Build the TAPS test loader via :func:`prepare_evaluation_runtime` (one
   loader instance reused across the FP32 PyTorch and ONNX paths).
2. Run :func:`src.evaluate.evaluate` on the FP32 PyTorch model
   ``bafnetplus_<T>ms/best.th`` → trained-fusion reference per SNR.
3. Run :func:`src.eval_onnx_streaming.evaluate_onnx` on the FP32 ONNX graph and
   on the INT8 QDQ ONNX graph at the same chunk geometry → ONNX-side metrics.
4. Emit a per-config JSON with ``fp32_pytorch / fp32_onnx / int8_onnx`` blocks
   at ``BAFNetPlus/results/eval/d5b_<T>ms.json`` for aggregation by
   ``Paper_TASLP/tables/scripts/aggregate_d5b.py`` (which derives three
   deltas from the three blocks — see that module's docstring).

**Important — fusion weight provenance**: The unified ``bafnetplus_<T>ms``
checkpoint contains TRAINED fusion weights (calibration_encoder +
alpha_convblocks). The ONNX export pipeline used by ``d5b_export_all_configs.sh``
rebuilds BAFNet+ from the dual backbone checkpoints (``bm_map_<T>ms`` +
``bm_mask_<T>ms``) and **re-initialises the fusion weights via Kaiming-random
under seed=42** (see
:class:`src.models.streaming.bafnetplus_streaming.BAFNetPlusStreaming`
docstring §"Stage 1 scope"). Consequently:

- ``fp32_pytorch`` block uses TRAINED fusion (== Tab I numbers).
- ``fp32_onnx`` and ``int8_onnx`` blocks use UNTRAINED seed=42 fusion (== the
  weights actually deployed in the published 50 ms QDQ ONNX).

This three-block layout lets the aggregator separate the deployment delta
(INT8 vs PyTorch — what users experience) from the pure quantization delta
(INT8 vs FP32 ONNX — what INT8 alone introduces) and the untrained-fusion
gap (FP32 ONNX vs PyTorch).

Usage (from BAFNetPlus repo root):
    python scripts/d5b_run_eval.py \\
        --noise_dir /path/to/noise --rir_dir /path/to/rir \\
        --noise_test dataset/taps/noise_test.txt \\
        --rir_test   dataset/taps/rir_test.txt \\
        --snr_step -5 0 5 10 15
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_onnx_streaming import evaluate_onnx  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.runtime_common import prepare_evaluation_runtime  # noqa: E402

T_ALG_MS = (12, 25, 50, 75, 100)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--noise_dir", type=str, required=True)
    p.add_argument("--noise_test", type=str, required=True)
    p.add_argument("--rir_dir", type=str, required=True)
    p.add_argument("--rir_test", type=str, required=True)
    p.add_argument("--snr_step", nargs="+", type=int, default=[-5, 0, 5, 10, 15])
    p.add_argument("--test_augment_numb", type=int, default=2)
    p.add_argument("--reverb_proportion", type=float, default=0.0)
    p.add_argument("--target_dB_FS", type=float, default=-25.0)
    p.add_argument("--target_dB_FS_floating_value", type=float, default=0.0)
    p.add_argument("--silence_length", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--configs",
        nargs="+",
        type=int,
        default=list(T_ALG_MS),
        help="Subset of T_alg ms values to evaluate (default: all 5).",
    )
    p.add_argument("--bcs_gain_db", type=float, default=0.0,
                   help="BCS-channel gain perturbation in dB (default 0.0). "
                        "Applied to both PyTorch and ONNX paths for fair comparison.")
    p.add_argument("--acs_gain_db", type=float, default=0.0,
                   help="ACS-channel gain perturbation in dB (default 0.0). "
                        "Applied to both PyTorch and ONNX paths for fair comparison.")
    return p.parse_args()


def _build_eval_args(cli: argparse.Namespace, t_ms: int) -> SimpleNamespace:
    """Build a runtime args namespace for one config.

    The model_config / chkpt_dir point at the unified ``bafnetplus_<T>ms``
    experiment so the FP32 PyTorch reference uses the same trained fusion. The
    ONNX graphs at ``results/onnx/bafnetplus_<T>ms/`` are produced by the
    sibling export shell script.
    """
    chkpt_dir = str(PROJECT_ROOT / "results" / "experiments" / f"bafnetplus_{t_ms}ms")
    model_config = str(Path(chkpt_dir) / ".hydra" / "config.yaml")
    return SimpleNamespace(
        model_config=model_config,
        chkpt_dir=chkpt_dir,
        chkpt_file="best.th",
        dataset="taps",
        device=cli.device,
        num_workers=cli.num_workers,
        noise_dir=cli.noise_dir,
        noise_test=cli.noise_test,
        rir_dir=cli.rir_dir,
        rir_test=cli.rir_test,
        snr_step=cli.snr_step,
        test_augment_numb=cli.test_augment_numb,
        reverb_proportion=cli.reverb_proportion,
        target_dB_FS=cli.target_dB_FS,
        target_dB_FS_floating_value=cli.target_dB_FS_floating_value,
        silence_length=cli.silence_length,
        eval_stt=False,
        save_per_utterance_json=False,
        bcs_gain_db=cli.bcs_gain_db,
        acs_gain_db=cli.acs_gain_db,
        test_bypass_mixing=False,
        test_hf_dataset=None,
        test_subset=None,
        stt_language=None,
    )


def _run_one_config(
    cli: argparse.Namespace, t_ms: int, logger: logging.Logger,
) -> dict:
    logger.info("=" * 64)
    logger.info(f"D5b: evaluating T_alg = {t_ms} ms")
    logger.info("=" * 64)

    eval_args = _build_eval_args(cli, t_ms)
    runtime = prepare_evaluation_runtime(eval_args, logger=logger)

    onnx_dir = PROJECT_ROOT / "results" / "onnx" / f"bafnetplus_{t_ms}ms"
    fp32_onnx = onnx_dir / "bafnetplus.onnx"
    qdq_onnx = onnx_dir / "bafnetplus_qdq.onnx"
    cfg_json = onnx_dir / "bafnetplus_streaming_config.json"
    for p in (fp32_onnx, qdq_onnx, cfg_json):
        if not p.exists():
            raise SystemExit(
                f"Missing ONNX artifact: {p}\n"
                f"Run scripts/d5b_export_all_configs.sh first.",
            )

    fp32_pytorch_metrics = evaluate(
        args=runtime.conf,
        model=runtime.model,
        data_loader_list=runtime.data_loader_list,
        logger=logger,
        epoch=None,
        stft_args=runtime.stft_args,
    )

    fp32_onnx_metrics = evaluate_onnx(
        onnx_path=str(fp32_onnx),
        config_path=str(cfg_json),
        data_loader_list=runtime.data_loader_list,
        logger=logger,
        bcs_gain_db=cli.bcs_gain_db,
        acs_gain_db=cli.acs_gain_db,
    )

    int8_onnx_metrics = evaluate_onnx(
        onnx_path=str(qdq_onnx),
        config_path=str(cfg_json),
        data_loader_list=runtime.data_loader_list,
        logger=logger,
        bcs_gain_db=cli.bcs_gain_db,
        acs_gain_db=cli.acs_gain_db,
    )

    return {
        "t_alg_ms": t_ms,
        "snr_step": list(cli.snr_step),
        "fp32_pytorch": fp32_pytorch_metrics,
        "fp32_onnx": fp32_onnx_metrics,
        "int8_onnx": int8_onnx_metrics,
    }


def main() -> int:
    cli = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.warning(
        "D5b runner: fusion-weight provenance differs across blocks. "
        "fp32_pytorch uses TRAINED fusion (bafnetplus_<T>ms/best.th); "
        "fp32_onnx and int8_onnx use UNTRAINED seed=42 fusion (export pipeline). "
        "See aggregate_d5b.py docstring for the three-delta breakdown.",
    )

    out_dir = PROJECT_ROOT / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    for t_ms in cli.configs:
        if t_ms not in T_ALG_MS:
            logger.warning(f"skipping unknown T_alg={t_ms} ms (valid: {T_ALG_MS})")
            continue
        result = _run_one_config(cli, t_ms, logger)
        out_path = out_dir / f"d5b_{t_ms}ms.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
