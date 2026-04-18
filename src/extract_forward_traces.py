"""Extract per-frame calibration gains and fusion alpha from BAFNetPlus (CR3/MJ2/MJ3).

Loads a BAFNetPlus checkpoint, runs inference on the TAPS test set, and saves
per-utterance traces of:
    - common_log_gain (c_t)
    - relative_log_gain (delta_t)
    - alpha_bcs, alpha_acs (TF-wise fusion weights)

Traces are aggregated per SNR into distribution-friendly numpy arrays.

Usage:
    python -m src.extract_forward_traces \\
        --model_config results/experiments/bafnetplus_50ms/.hydra/config.yaml \\
        --chkpt_dir results/experiments/bafnetplus_50ms \\
        --snr_step -10 0 10 \\
        --max_utt 100 \\
        --output_npz paper_work/figures/output/bafnetplus_50ms_traces.npz
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

from src.data import Noise_Augmented_Dataset
from src.stft import mag_pha_stft
from src.utils import (
    load_model,
    load_checkpoint,
    parse_file_list,
    get_stft_args_from_config,
)


logger = logging.getLogger(__name__)


def register_hooks(model):
    """Register forward hooks to capture calibration gains and alpha pre-softmax.

    Returns:
        buf: dict updated in-place on each forward pass with keys
             'common_log_gain' [B,1,T], 'relative_log_gain' [B,1,T] (optional),
             'alpha_pre_softmax' [B,2,T,F]
        handles: list of hook handles to remove later
    """
    buf = {}
    handles = []

    def _hook_common(_m, _i, o):
        buf["common_log_gain"] = o.detach().cpu()

    def _hook_relative(_m, _i, o):
        buf["relative_log_gain"] = o.detach().cpu()

    def _hook_alpha(_m, _i, o):
        buf["alpha_pre_softmax"] = o.detach().cpu()

    if hasattr(model, "common_gain_head") and model.use_calibration:
        handles.append(model.common_gain_head.register_forward_hook(_hook_common))
    if hasattr(model, "relative_gain_head") and getattr(model, "use_relative_gain", False):
        handles.append(model.relative_gain_head.register_forward_hook(_hook_relative))
    if hasattr(model, "alpha_out"):
        handles.append(model.alpha_out.register_forward_hook(_hook_alpha))

    return buf, handles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--chkpt_dir", type=str, required=True)
    parser.add_argument("--chkpt_file", type=str, default="best.th")
    parser.add_argument("--dataset", type=str, default="taps")
    parser.add_argument("--noise_dir", type=str, default="/home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k")
    parser.add_argument("--noise_test", type=str, default="dataset/taps/noise_test.txt")
    parser.add_argument("--rir_dir", type=str, default="/home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k")
    parser.add_argument("--rir_test", type=str, default="dataset/taps/rir_test.txt")
    parser.add_argument("--snr_step", nargs="+", type=int, required=True)
    parser.add_argument("--max_utt", type=int, default=100, help="Cap utterances per SNR")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--test_augment_numb", type=int, default=2)
    parser.add_argument("--target_dB_FS", type=float, default=-25)
    parser.add_argument("--target_dB_FS_floating_value", type=float, default=0)
    parser.add_argument("--silence_length", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # Load config (rewrite pod paths)
    cfg_text = Path(args.model_config).read_text().replace("/workspace", "/home/yskim/workspace")
    conf = OmegaConf.create(cfg_text)

    model_args = conf.model
    model = load_model(model_args.model_lib, model_args.model_class, model_args.param, args.device)
    model = load_checkpoint(model, args.chkpt_dir, args.chkpt_file, args.device)
    model.eval()

    buf, handles = register_hooks(model)

    stft_args = get_stft_args_from_config(model_args)
    if hasattr(conf, "compress_factor"):
        stft_args["compress_factor"] = conf.compress_factor

    testset = load_dataset(
        "yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test"
    )
    testset = concatenate_datasets([testset] * args.test_augment_numb)

    noise_test_list = parse_file_list(args.noise_dir, args.noise_test)
    rir_test_list = parse_file_list(args.rir_dir, args.rir_test)

    results_per_snr = {}
    for snr in args.snr_step:
        ds = Noise_Augmented_Dataset(
            datapair_list=testset,
            noise_list=noise_test_list,
            rir_list=rir_test_list,
            snr_range=[snr, snr],
            reverb_proportion=0.0,
            target_dB_FS=args.target_dB_FS,
            target_dB_FS_floating_value=args.target_dB_FS_floating_value,
            silence_length=args.silence_length,
            deterministic=True,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
            bcs_only=False,
        )
        loader = DataLoader(ds, batch_size=1, num_workers=args.num_workers)

        cg_all, rg_all = [], []
        alpha_bcs_means, alpha_acs_means = [], []
        alpha_bcs_frame_curves = []

        for i, data in enumerate(loader):
            if i >= args.max_utt:
                break
            bcs, noisy_acs, _clean, _uid, _text = data

            bcs_com = mag_pha_stft(bcs, **stft_args)[2].to(args.device)
            acs_com = mag_pha_stft(noisy_acs, **stft_args)[2].to(args.device)

            with torch.no_grad():
                _ = model((bcs_com, acs_com))

            if "common_log_gain" in buf:
                cg = buf["common_log_gain"].squeeze().numpy()  # [T]
                cg_all.append(cg)
            if "relative_log_gain" in buf:
                rg = buf["relative_log_gain"].squeeze().numpy()  # [T]
                rg_all.append(rg)

            if "alpha_pre_softmax" in buf:
                # [B,2,T,F] logits → softmax on dim=1 → alpha_bcs, alpha_acs
                a_pre = buf["alpha_pre_softmax"]
                a_soft = torch.softmax(a_pre, dim=1).squeeze(0).numpy()  # [2,T,F]
                alpha_bcs_mean = float(a_soft[0].mean())
                alpha_acs_mean = float(a_soft[1].mean())
                alpha_bcs_means.append(alpha_bcs_mean)
                alpha_acs_means.append(alpha_acs_mean)
                # Frame-wise curve (averaged over freq)
                alpha_bcs_frame_curves.append(a_soft[0].mean(axis=1))  # [T]

        results_per_snr[snr] = {
            "common_log_gain": np.concatenate(cg_all) if cg_all else np.array([]),
            "relative_log_gain": np.concatenate(rg_all) if rg_all else np.array([]),
            "alpha_bcs_means": np.array(alpha_bcs_means),
            "alpha_acs_means": np.array(alpha_acs_means),
        }
        logger.info(
            f"SNR={snr}dB: cg.mean={results_per_snr[snr]['common_log_gain'].mean():.4f}, "
            f"rg.mean={results_per_snr[snr]['relative_log_gain'].mean():.4f}, "
            f"alpha_bcs={np.mean(alpha_bcs_means):.4f}"
        )

    for h in handles:
        h.remove()

    save_dict = {}
    for snr, d in results_per_snr.items():
        for k, v in d.items():
            save_dict[f"snr{snr}_{k}"] = v

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **save_dict)
    logger.info(f"Saved traces to {out}")


if __name__ == "__main__":
    main()
