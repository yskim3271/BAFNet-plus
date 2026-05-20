"""Dump BAFNet+ calibration and fusion internal tensors for paper figures.

Loads a trained BAFNetPlus checkpoint and runs forward on a set of TAPS test
utterances at SNR = 0 dB with deterministic noise/RIR mixing, applying
post-mix BCS gain perturbations Δ_bcs ∈ {-10, 0, +10} dB. For each (idx, Δ_bcs)
pair, captures the full set of internal tensors used by Fig. (calib) and
Fig. (fusion) in the TASLP manuscript:

    - phi_t (5 calibration features [B, 5, T])
    - c_t (scaled common log gain [B, 1, T])
    - Delta_t (scaled relative log gain [B, 1, T])
    - g_t^bcs, g_t^acs (per-frame branch gains [B, 1, T])
    - bcs_mag_cal, acs_mag_cal (post-calibration TF magnitudes [B, F, T])
    - acs_mask (ACS spectral mask [B, F, T])
    - alpha_bcs, alpha_acs (per-bin fusion weights [B, F, T])
    - bcs_wave, noisy_acs_wave, clean_acs_wave (1D waveforms for context)

Output: Paper_TASLP/figures/data/calib_fusion/idx{idx:03d}_dbcs{db:+d}.npz

Model code is NOT modified: internal tensors are reproduced by calling the
exact same intermediate functions BAFNetPlus.forward uses.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure src/ is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import load_checkpoint, load_model  # noqa: E402
from src.data import Noise_Augmented_Dataset  # noqa: E402
from src.runtime_common import load_test_dataset  # noqa: E402
from src.stft import mag_pha_stft  # noqa: E402
from src.utils import parse_file_list  # noqa: E402


# ---- Fixed paper protocol (TAPS default) ----
DEFAULT_NOISE_DIR = "/home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k"
DEFAULT_RIR_DIR = "/home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k"
DEFAULT_NOISE_TEST = "dataset/taps/noise_test.txt"
DEFAULT_RIR_TEST = "dataset/taps/rir_test.txt"

STFT_ARGS = dict(n_fft=400, hop_size=100, win_size=400, compress_factor=0.3)
TARGET_DB_FS = -25
SAMPLING_RATE = 16_000


def run_forward_capture(model, bcs_com, acs_com):
    """Reproduce BAFNetPlus.forward but expose all internal tensors.

    Mirrors BAFNetPlus.forward (src/models/bafnetplus.py:225) exactly; the only
    addition is collecting intermediates into a dict.
    """
    captured: dict[str, torch.Tensor] = {}

    bcs_mag, _, bcs_com_out = model.mapping(bcs_com)
    acs_mag, _, acs_com_out, acs_mask = model.masking(acs_com, return_mask=True)

    # --- Calibration ---
    if not model.use_calibration:
        raise RuntimeError("Checkpoint has calibration disabled; cannot dump.")
    calib_feat = model._build_calibration_features(bcs_mag, acs_mag, acs_mask)
    calib_hidden = model.calibration_encoder(calib_feat)
    common_log = torch.tanh(model.common_gain_head(calib_hidden))
    common_log = common_log * model.calibration_max_common_log_gain
    if model.use_relative_gain:
        rel_log = torch.tanh(model.relative_gain_head(calib_hidden))
        rel_log = rel_log * model.calibration_max_relative_log_gain
        bcs_gain = torch.exp(common_log - 0.5 * rel_log)
        acs_gain = torch.exp(common_log + 0.5 * rel_log)
    else:
        rel_log = torch.zeros_like(common_log)
        bcs_gain = acs_gain = torch.exp(common_log)

    bcs_gain_4d = bcs_gain.transpose(1, 2).unsqueeze(1)
    acs_gain_4d = acs_gain.transpose(1, 2).unsqueeze(1)
    bcs_com_cal = bcs_com_out * bcs_gain_4d
    acs_com_cal = acs_com_out * acs_gain_4d

    # --- Fusion ---
    if model.mask_only_alpha:
        raise RuntimeError("Checkpoint is mask-only ablation; cannot dump 3ch.")
    alpha_in = model._build_alpha_features(bcs_com_cal, acs_com_cal, acs_mask)
    alpha = alpha_in
    for block in model.alpha_convblocks:
        alpha = block(alpha)
    alpha = model.alpha_out(alpha)
    alpha = alpha.transpose(2, 3)
    alpha = torch.softmax(alpha, dim=1)
    alpha_bcs = alpha[:, 0]  # [B, F, T]
    alpha_acs = alpha[:, 1]

    eps = 1e-8
    bcs_mag_cal = torch.sqrt(bcs_com_cal[..., 0] ** 2 + bcs_com_cal[..., 1] ** 2 + eps)
    acs_mag_cal = torch.sqrt(acs_com_cal[..., 0] ** 2 + acs_com_cal[..., 1] ** 2 + eps)

    captured["phi_t"] = calib_feat.detach().cpu().numpy()
    captured["c_t"] = common_log.detach().cpu().numpy()
    captured["delta_t"] = rel_log.detach().cpu().numpy()
    captured["g_bcs"] = bcs_gain.detach().cpu().numpy()
    captured["g_acs"] = acs_gain.detach().cpu().numpy()
    captured["bcs_mag_cal"] = bcs_mag_cal.detach().cpu().numpy()
    captured["acs_mag_cal"] = acs_mag_cal.detach().cpu().numpy()
    captured["acs_mask"] = acs_mask.detach().cpu().numpy()
    captured["alpha_bcs"] = alpha_bcs.detach().cpu().numpy()
    captured["alpha_acs"] = alpha_acs.detach().cpu().numpy()
    return captured


def build_dataset(indices, noise_dir, noise_test, rir_dir, rir_test, snr=0):
    """Build a deterministic Noise_Augmented_Dataset over a subset of TAPS test."""
    testset = load_test_dataset("taps", 1)  # no augmentation, single pass
    # Keep only requested indices, in order. (load_test_dataset returns HF Dataset)
    indices = sorted(set(int(i) for i in indices))
    testset_sub = testset.select(indices)

    noise_list = parse_file_list(noise_dir, noise_test)
    rir_list = parse_file_list(rir_dir, rir_test)
    dataset = Noise_Augmented_Dataset(
        datapair_list=testset_sub,
        noise_list=noise_list,
        rir_list=rir_list,
        snr_range=[snr, snr],
        reverb_proportion=0.0,
        target_dB_FS=TARGET_DB_FS,
        target_dB_FS_floating_value=0,
        silence_length=0.2,
        sampling_rate=SAMPLING_RATE,
        deterministic=True,
        with_id=True,
        with_text=False,
        bcs_only=False,
        bcs_gain_perturbation_db=0,  # we apply perturbation manually
    )
    return dataset, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="results/experiments/bafnetplus_50ms/best.th")
    parser.add_argument("--model-config",
                        default="results/experiments/bafnetplus_50ms/.hydra/config.yaml")
    parser.add_argument("--output-dir",
                        default="../Paper_TASLP/figures/data/calib_fusion")
    parser.add_argument("--indices", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4, 5, 10, 20, 50, 100],
                        help="TAPS test split indices to dump.")
    parser.add_argument("--db-list", nargs="+", type=int, default=[-10, 0, 10],
                        help="Gain perturbation levels δ in dB (see --mode).")
    parser.add_argument("--mode", choices=["bcs_only", "anti_symmetric"],
                        default="anti_symmetric",
                        help="bcs_only: Δ_bcs=+δ. anti_symmetric: Δ_bcs=+δ, "
                             "Δ_acs=-δ (matches §VI.D gain-robust protocol).")
    parser.add_argument("--snr", type=int, default=0,
                        help="Fixed evaluation SNR in dB.")
    parser.add_argument("--noise-dir", default=DEFAULT_NOISE_DIR)
    parser.add_argument("--noise-test", default=DEFAULT_NOISE_TEST)
    parser.add_argument("--rir-dir", default=DEFAULT_RIR_DIR)
    parser.add_argument("--rir-test", default=DEFAULT_RIR_TEST)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dump] output: {out_dir}")

    # --- Load model ---
    print(f"[dump] loading model config: {args.model_config}")
    from omegaconf import OmegaConf
    conf = OmegaConf.load(args.model_config)
    model_args = conf.model
    model_params = OmegaConf.to_container(model_args.param, resolve=True)
    print(f"[dump] loading model {model_args.model_class} from {model_args.model_lib}")
    model = load_model(
        model_lib=model_args.model_lib,
        model_class_name=model_args.model_class,
        model_params=model_params,
        device=args.device,
    )
    print(f"[dump] loading checkpoint: {args.checkpoint}")
    chkpt_dir = os.path.dirname(args.checkpoint)
    chkpt_file = os.path.basename(args.checkpoint)
    load_checkpoint(model=model, chkpt_dir=chkpt_dir, chkpt_file=chkpt_file,
                    device=args.device)
    model.eval()

    # --- Build dataset ---
    dataset, indices = build_dataset(args.indices, args.noise_dir,
                                     args.noise_test, args.rir_dir,
                                     args.rir_test, snr=args.snr)
    print(f"[dump] dataset size: {len(dataset)} (idx={indices}) "
          f"SNR={args.snr}dB mode={args.mode}")

    # --- Iterate ---
    with torch.no_grad():
        for ds_idx, sample_idx in enumerate(indices):
            bcs, noisy_acs, clean_acs, utt_id = dataset[ds_idx]
            # Add batch dim, move to device
            bcs = bcs.unsqueeze(0).to(args.device)
            noisy_acs = noisy_acs.unsqueeze(0).to(args.device)
            clean_acs_np = clean_acs.detach().cpu().numpy().squeeze()
            print(f"[dump] idx={sample_idx} utt_id={utt_id} "
                  f"bcs.shape={tuple(bcs.shape)} noisy.shape={tuple(noisy_acs.shape)}")

            for db in args.db_list:
                bcs_scalar = 10.0 ** (db / 20.0)
                acs_scalar = (10.0 ** (-db / 20.0)
                              if args.mode == "anti_symmetric" else 1.0)
                bcs_pert = bcs * bcs_scalar
                acs_pert = noisy_acs * acs_scalar

                _, _, bcs_com = mag_pha_stft(bcs_pert, **STFT_ARGS)
                _, _, acs_com = mag_pha_stft(acs_pert, **STFT_ARGS)

                cap = run_forward_capture(model, bcs_com, acs_com)

                # Save with waveforms + metadata for downstream figures
                tag = f"snr{args.snr:+d}_{args.mode}"
                save_path = out_dir / f"idx{sample_idx:03d}_{tag}_d{db:+d}.npz"
                np.savez_compressed(
                    save_path,
                    idx=sample_idx,
                    db=db,
                    mode=args.mode,
                    snr=args.snr,
                    utt_id=str(utt_id),
                    n_fft=STFT_ARGS["n_fft"],
                    hop_size=STFT_ARGS["hop_size"],
                    win_size=STFT_ARGS["win_size"],
                    compress_factor=STFT_ARGS["compress_factor"],
                    sampling_rate=SAMPLING_RATE,
                    bcs_wave=bcs_pert.detach().cpu().numpy().squeeze(),
                    noisy_acs_wave=acs_pert.detach().cpu().numpy().squeeze(),
                    clean_acs_wave=clean_acs_np,
                    **cap,
                )
                print(f"[dump]   saved -> {save_path.name}")

    print("[dump] done")


if __name__ == "__main__":
    main()
