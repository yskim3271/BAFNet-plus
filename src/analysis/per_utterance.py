"""Per-utterance evaluation for statistical analysis.

Evaluates a trained model on the test set and stores per-utterance PESQ,
STOI, CSIG, CBAK, COVL, and segSNR. MOS and STT are skipped for speed.

Output JSON structure:
    {
        "-20dB": {
            "<speaker_id>_<sentence_id>": {
                "pesq": float, "stoi": float, ...
            },
            ...
        },
        "-10dB": { ... }
    }

Usage:
    python -m src.analysis.per_utterance \\
        --model_config <hydra_config_yaml> \\
        --chkpt_dir <experiment_dir> \\
        --noise_dir <noise_dir> \\
        --noise_test <noise_test_list> \\
        --rir_dir <rir_dir> \\
        --rir_test <rir_test_list> \\
        --snr_step -20 -10 0 10 15 \\
        --output_json <path>.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.runtime_common import prepare_evaluation_runtime, write_json
from src.evaluate import evaluate as run_evaluation

logger = logging.getLogger(__name__)


def evaluate_per_utterance(
    conf,
    model,
    data_loader_list,
    stft_args,
):
    """Return only the per-utterance block from the shared evaluator."""
    conf.eval_stt = False
    conf.save_per_utterance = True

    metrics = run_evaluation(
        args=conf,
        model=model,
        data_loader_list=data_loader_list,
        logger=logger,
        epoch=None,
        stft_args=stft_args,
    )
    return {
        snr_key: snr_metrics.get("per_utterance", {})
        for snr_key, snr_metrics in metrics.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--chkpt_dir", type=str, required=True)
    parser.add_argument("--chkpt_file", type=str, default="best.th")
    parser.add_argument("--dataset", type=str, default="taps", choices=["taps", "vibravox"])
    parser.add_argument("--noise_dir", type=str, required=True)
    parser.add_argument("--noise_test", type=str, required=True)
    parser.add_argument("--rir_dir", type=str, required=True)
    parser.add_argument("--rir_test", type=str, required=True)
    parser.add_argument("--snr_step", nargs="+", type=int, required=True)
    parser.add_argument("--test_augment_numb", type=int, default=2)
    parser.add_argument("--target_dB_FS", type=float, default=-25)
    parser.add_argument("--target_dB_FS_floating_value", type=float, default=0)
    parser.add_argument("--silence_length", type=float, default=0.2)
    parser.add_argument("--reverb_proportion", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_file", type=str, default="per_utterance.log")
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(),
        ],
    )

    runtime = prepare_evaluation_runtime(
        args,
        logger=logger,
        eval_stt=False,
        save_per_utterance=True,
    )

    logger.info(
        f"Model: {runtime.model_args.model_class}, input_type={runtime.model_args.input_type}"
    )
    logger.info(f"Checkpoint: {args.chkpt_dir}")
    logger.info(f"SNR steps: {args.snr_step}")

    metrics = evaluate_per_utterance(
        runtime.conf,
        runtime.model,
        runtime.data_loader_list,
        runtime.stft_args,
    )

    write_json(metrics, args.output_json, logger=logger)


if __name__ == "__main__":
    main()
