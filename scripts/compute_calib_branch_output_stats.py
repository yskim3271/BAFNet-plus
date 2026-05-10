"""Compute calibration-input statistics on **branch outputs** for Sec V.D MJ4.

For each TAPS train utterance (clean BCS+ACS pair, no noise/RIR), we run the
trained BAFNet+ checkpoint forward and capture the branch-output magnitudes
`bcs_mag`, `acs_mag` exactly as `_build_calibration_features` consumes them
(`bafnetplus.py:182-194`). We then compute the per-frame log-energy ratio
`phi^(3)_t = log(mean_k bcs_mag^2 + eps) - log(mean_k acs_mag^2 + eps)` and
report 5/25/50/75/95 percentiles in nats and the equivalent power-dB
thresholds, plus the saturation rate at `|phi^(3)| > 2 nats` (= the Δ_t bound
that perfectly equalises a frame, equivalent to ±8.69 dB inter-branch
amplitude ratio).

Outputs:
    Paper_TASLP/tables/calib_bound_stats_branch_output.{csv,md}
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import Noise_Augmented_Dataset
from src.models.bafnetplus import BAFNetPlus
from src.stft import mag_pha_stft

DEFAULT_CHKPT = ROOT / "results" / "experiments" / "bafnetplus_50ms" / "best.th"
DEFAULT_HYDRA = DEFAULT_CHKPT.parent / ".hydra" / "config.yaml"
TABLES_DIR = ROOT.parent / "Paper_TASLP" / "tables"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHKPT)
    p.add_argument("--hydra_config", type=Path, default=DEFAULT_HYDRA)
    p.add_argument("--num_utt", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--out_md", type=Path, default=TABLES_DIR / "calib_bound_stats_branch_output.md")
    p.add_argument("--out_csv", type=Path, default=TABLES_DIR / "calib_bound_stats_branch_output.csv")
    return p.parse_args()


def build_model(hydra_cfg_path: Path, ckpt_path: Path, device: torch.device) -> BAFNetPlus:
    cfg = OmegaConf.load(hydra_cfg_path)
    param = OmegaConf.to_container(cfg.model.param, resolve=True)
    # Keep checkpoint_mapping/masking paths so BAFNetPlus can auto-extract
    # the per-branch architecture from the saved checkpoints, but disable
    # eager pretrained-weight loads — the full BAFNet+ state_dict we load
    # below already contains the merged weights for both branches.
    param["load_pretrained_weights"] = False
    model = BAFNetPlus(**param)
    chkpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = chkpt.get("model", chkpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def collect_phi3(
    model: BAFNetPlus,
    dataset: Noise_Augmented_Dataset,
    cfg: OmegaConf,
    num_utt: int,
    device: torch.device,
) -> np.ndarray:
    """Return concatenated per-frame phi^(3) (nats) over `num_utt` utterances."""
    n_fft = cfg.n_fft
    hop = cfg.hop_size
    win = cfg.win_size
    cf = cfg.compress_factor

    captured: dict[str, torch.Tensor] = {}

    def cap_mapping(_module, _inputs, output):
        captured["bcs_mag"] = output[0].detach()  # (B, F, T)

    def cap_masking(_module, _inputs, output):
        captured["acs_mag"] = output[0].detach()  # (B, F, T)

    h1 = model.mapping.register_forward_hook(cap_mapping)
    h2 = model.masking.register_forward_hook(cap_masking)

    eps = 1e-8
    phis = []
    n_frames_total = 0
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    indices = indices[:num_utt]

    with torch.no_grad():
        for n, idx in enumerate(indices):
            bcs, _noisy_acs, clean_acs = dataset[int(idx)]
            bcs = bcs.to(device).unsqueeze(0)  # (1, T_samples)
            acs = clean_acs.to(device).unsqueeze(0)
            _, _, bcs_com = mag_pha_stft(bcs, n_fft, hop, win, compress_factor=cf)
            _, _, acs_com = mag_pha_stft(acs, n_fft, hop, win, compress_factor=cf)
            _ = model((bcs_com, acs_com))
            bcs_mag = captured["bcs_mag"]  # (1, F, T)
            acs_mag = captured["acs_mag"]
            bcs_log_e = torch.log(bcs_mag.pow(2).mean(dim=1) + eps)  # (1, T)
            acs_log_e = torch.log(acs_mag.pow(2).mean(dim=1) + eps)
            phi3 = (bcs_log_e - acs_log_e).squeeze(0).cpu().numpy()  # (T,)
            phis.append(phi3)
            n_frames_total += phi3.shape[0]
            if (n + 1) % 100 == 0:
                print(f"  [{n + 1}/{len(indices)}] frames so far: {n_frames_total}")

    h1.remove()
    h2.remove()
    return np.concatenate(phis, axis=0)


def power_dB(phi_nats: np.ndarray) -> np.ndarray:
    """phi^(3) in nats -> 10 log10(E_bcs / E_acs) in power dB."""
    return phi_nats * (10.0 / np.log(10.0))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[info] device={device} num_utt={args.num_utt}")
    cfg = OmegaConf.load(args.hydra_config)

    print("[info] loading TAPS train split (HuggingFace)")
    hf_ds = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset")
    train_split = hf_ds["train"]

    dataset = Noise_Augmented_Dataset(
        datapair_list=train_split,
        noise_list=[],
        rir_list=[],
        snr_range=[0, 0],
        reverb_proportion=0.0,
        target_dB_FS=-25,
        target_dB_FS_floating_value=0,
        silence_length=0.0,
        sampling_rate=cfg.sampling_rate,
        bcs_gain_perturbation_db=0,
        bypass_mixing=True,
    )
    print(f"[info] dataset size: {len(dataset)}")

    print(f"[info] loading checkpoint: {args.checkpoint}")
    model = build_model(args.hydra_config, args.checkpoint, device)
    print("[info] running forward + collecting phi^(3) on branch outputs ...")
    phi3 = collect_phi3(model, dataset, cfg, args.num_utt, device)
    print(f"[info] total frames: {phi3.shape[0]}")

    p5, p25, p50, p75, p95 = np.percentile(phi3, [5, 25, 50, 75, 95])
    sat_threshold_nats = 2.0  # |Δ| ≤ 1 fully equalises |phi^(3)| ≤ 2
    sat_rate = float(np.mean(np.abs(phi3) > sat_threshold_nats))
    p5_db, p25_db, p50_db, p75_db, p95_db = (power_dB(np.array([p5, p25, p50, p75, p95])))

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "value_nats", "value_power_dB"])
        for name, val in [
            ("p5", p5), ("p25", p25), ("p50", p50), ("p75", p75), ("p95", p95),
        ]:
            w.writerow([name, f"{val:.4f}", f"{power_dB(np.array([val]))[0]:.4f}"])
        w.writerow(["saturation_rate_abs_phi3_gt_2nats", f"{sat_rate:.6f}", ""])
        w.writerow(["num_frames", str(phi3.shape[0]), ""])
        w.writerow(["num_utt", str(args.num_utt), ""])

    md = (
        f"# Calibration bound statistics — branch outputs (BAFNet+ 50 ms checkpoint)\n\n"
        f"Source: TAPS train clean BCS+ACS pairs (no noise/RIR mixing), branch\n"
        f"outputs `bcs_mag`, `acs_mag` from `BAFNetPlus.mapping/masking` "
        f"forward hooks, exactly the tensors `_build_calibration_features` consumes.\n\n"
        f"- Number of utterances: **{args.num_utt}**\n"
        f"- Number of frames: **{phi3.shape[0]:,}**\n"
        f"- `phi^(3) = log(mean_k bcs_mag^2 + eps) - log(mean_k acs_mag^2 + eps)` (nats)\n\n"
        f"## Per-frame `phi^(3)` distribution (branch outputs)\n\n"
        f"| Percentile | nats | Power dB (10·log10(E_bcs/E_acs)) |\n"
        f"|---:|---:|---:|\n"
        f"| p5  | {p5:+.4f} | {p5_db:+.3f} |\n"
        f"| p25 | {p25:+.4f} | {p25_db:+.3f} |\n"
        f"| p50 | {p50:+.4f} | {p50_db:+.3f} |\n"
        f"| p75 | {p75:+.4f} | {p75_db:+.3f} |\n"
        f"| p95 | {p95:+.4f} | {p95_db:+.3f} |\n\n"
        f"## Saturation rate at the Δ_t bound\n\n"
        f"The relative-gain head `Δ_t ∈ [-G_max^Δ, +G_max^Δ] = [-1, +1]` nats\n"
        f"perfectly equalises any per-frame `|phi^(3)| ≤ 2` nats (equivalent\n"
        f"to ±8.69 dB inter-branch power ratio). Frames with\n"
        f"`|phi^(3)| > 2` nats cannot be fully equalised by the relative\n"
        f"gain alone.\n\n"
        f"- Fraction of frames with `|phi^(3)| > 2` nats: **{sat_rate*100:.3f}%**\n\n"
        f"## Generation\n\n"
        f"```\n"
        f"python -m scripts.compute_calib_branch_output_stats \\\n"
        f"  --checkpoint results/experiments/bafnetplus_50ms/best.th \\\n"
        f"  --num_utt {args.num_utt} --seed {args.seed} --device {args.device}\n"
        f"```\n"
    )
    args.out_md.write_text(md)

    print("\n=== summary ===")
    print(f"frames: {phi3.shape[0]:,}")
    print(f"p5/p25/p50/p75/p95 (nats):   {p5:+.4f} / {p25:+.4f} / {p50:+.4f} / {p75:+.4f} / {p95:+.4f}")
    print(f"p5/p25/p50/p75/p95 (powdB):  {p5_db:+.3f} / {p25_db:+.3f} / {p50_db:+.3f} / {p75_db:+.3f} / {p95_db:+.3f}")
    print(f"saturation rate |phi^(3)|>2: {sat_rate*100:.3f}%")
    print(f"\nwrote: {args.out_md}\nwrote: {args.out_csv}")


if __name__ == "__main__":
    main()
