"""Per-utterance evaluation for bootstrap CI (CR4).

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
    python -m src.evaluate_per_utterance \\
        --model_config <hydra_config_yaml> \\
        --chkpt_dir <experiment_dir> \\
        --snr_step -20 -10 0 10 15 \\
        --output_json <path>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

from src.compute_metrics import compute_metrics
from src.data import Noise_Augmented_Dataset
from src.stft import mag_pha_stft, mag_pha_istft
from src.utils import (
    bold,
    LogProgress,
    load_model,
    load_checkpoint,
    parse_file_list,
    get_stft_args_from_config,
)

logger = logging.getLogger(__name__)


def evaluate_per_utterance(
    conf,
    model,
    data_loader_list,
    stft_args,
):
    model.eval()
    all_metrics = {}

    for snr, data_loader in data_loader_list.items():
        iterator = LogProgress(logger, data_loader, name=f"Evaluate on {snr}dB")
        per_utt = {}
        pesq_sum = stoi_sum = 0.0
        n = 0

        with torch.no_grad():
            for data in iterator:
                bcs, noisy_acs, clean_acs, utt_id, _text = data

                if conf.model.input_type == "acs":
                    input_ = mag_pha_stft(noisy_acs, **stft_args)[2].to(conf.device)
                elif conf.model.input_type == "bcs":
                    input_ = mag_pha_stft(bcs, **stft_args)[2].to(conf.device)
                elif conf.model.input_type == "acs+bcs":
                    input_ = (
                        mag_pha_stft(bcs, **stft_args)[2].to(conf.device),
                        mag_pha_stft(noisy_acs, **stft_args)[2].to(conf.device),
                    )
                else:
                    raise ValueError(f"Unknown input_type: {conf.model.input_type}")

                mag_hat, pha_hat, _ = model(input_)
                enhanced = mag_pha_istft(mag_hat, pha_hat, **stft_args)

                clean_np = clean_acs.squeeze().detach().cpu().numpy()
                enhanced_np = enhanced.squeeze().detach().cpu().numpy()
                if len(clean_np) != len(enhanced_np):
                    L = min(len(clean_np), len(enhanced_np))
                    clean_np = clean_np[:L]
                    enhanced_np = enhanced_np[:L]

                pesq_v, csig, cbak, covl, ssnr, stoi_v = compute_metrics(
                    clean_np, enhanced_np
                )

                # utt_id is a list with one element due to batch_size=1
                key = utt_id[0] if isinstance(utt_id, (list, tuple)) else utt_id
                per_utt[key] = {
                    "pesq": float(pesq_v),
                    "stoi": float(stoi_v),
                    "csig": float(csig),
                    "cbak": float(cbak),
                    "covl": float(covl),
                    "segSNR": float(ssnr),
                }
                pesq_sum += float(pesq_v)
                stoi_sum += float(stoi_v)
                n += 1

        pesq_avg = pesq_sum / max(n, 1)
        stoi_avg = stoi_sum / max(n, 1)
        logger.info(bold(
            f"{snr}dB average over {n} utts: PESQ={pesq_avg:.4f} STOI={stoi_avg:.4f}"
        ))
        all_metrics[f"{snr}dB"] = per_utt

    return all_metrics


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

    conf = OmegaConf.load(args.model_config)
    conf.device = args.device

    model_args = conf.model
    model = load_model(model_args.model_lib, model_args.model_class, model_args.param, args.device)
    model = load_checkpoint(model, args.chkpt_dir, args.chkpt_file, args.device)
    bcs_only = "acs" not in model_args.input_type

    if args.dataset.lower() == "taps":
        testset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")
    elif args.dataset.lower() == "vibravox":
        testset = load_dataset("yskim3271/vibravox_16k", split="test")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    testset = concatenate_datasets([testset] * args.test_augment_numb)

    stft_args = get_stft_args_from_config(model_args)
    if hasattr(conf, "compress_factor"):
        stft_args["compress_factor"] = conf.compress_factor

    if bcs_only:
        noise_test_list, rir_test_list = [], []
    else:
        noise_test_list = parse_file_list(args.noise_dir, args.noise_test)
        rir_test_list = parse_file_list(args.rir_dir, args.rir_test)

    ev_loader_list = {}
    for fixed_snr in args.snr_step:
        ev_dataset = Noise_Augmented_Dataset(
            datapair_list=testset,
            noise_list=noise_test_list,
            rir_list=rir_test_list,
            snr_range=[fixed_snr, fixed_snr],
            reverb_proportion=args.reverb_proportion,
            target_dB_FS=args.target_dB_FS,
            target_dB_FS_floating_value=args.target_dB_FS_floating_value,
            silence_length=args.silence_length,
            deterministic=True,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
            bcs_only=bcs_only,
        )
        ev_loader_list[f"{fixed_snr}"] = DataLoader(
            ev_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True
        )

    logger.info(f"Model: {model_args.model_class}, input_type={model_args.input_type}")
    logger.info(f"Checkpoint: {args.chkpt_dir}")
    logger.info(f"SNR steps: {args.snr_step}")

    metrics = evaluate_per_utterance(conf, model, ev_loader_list, stft_args)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Per-utterance metrics saved to {out_path}")


if __name__ == "__main__":
    main()
