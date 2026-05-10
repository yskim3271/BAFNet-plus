"""Analyze BAFNet+ training-time effective dataset length and step counts.

Computes per-dataset (TAPS / Vibravox) the segmentation expansion produced by
BAFNet+'s receptive-field auto policy (segment = 2 * RF samples, stride = RF
samples) so we can decide ``epochs`` for the Vibravox-native run.

Outputs per dataset:
- utterance count, mean / median / total audio duration
- examples after BAFNet+ Audioset segmentation
- steps per epoch at given batch_size
- audio time per epoch

Then prints the Vibravox / TAPS ratio and the audio-time-matched
``recommended_epochs`` (Option β from the plan):

    recommended = round(taps_total_audio / vibravox_audio_per_epoch)

Usage:
    python scripts/analyze_dataset_length.py
    python scripts/analyze_dataset_length.py --batch-size 8 --epochs-taps 150
    python scripts/analyze_dataset_length.py --datasets vibravox  # subset only
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Add project root to Python path so `from src...` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.receptive_field import compute_receptive_field, rf_to_segment


DATASETS: Dict[str, Dict[str, Optional[str]]] = {
    "taps": {
        "hf_id": "yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset",
        "config": None,
        "split": "train",
        "audio_key": "audio.acoustic_microphone",
    },
    "vibravox": {
        "hf_id": "yskim3271/vibravox_16k",
        "config": None,
        "split": "train",
        "audio_key": "audio.acoustic_microphone",
    },
}


@dataclass
class DatasetStats:
    name: str
    n_utterances: int
    mean_samples: float
    median_samples: float
    min_samples: int
    max_samples: int
    total_samples: int
    total_seconds: float
    sampling_rate: int

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600.0


@dataclass
class SegmentationStats:
    segment: int
    stride: int
    examples_total: int
    examples_per_utt_mean: float
    examples_per_utt_max: int
    audio_seconds_per_epoch: float
    steps_per_epoch: int

    @property
    def audio_hours_per_epoch(self) -> float:
        return self.audio_seconds_per_epoch / 3600.0


def _collect_lengths(hf_id: str, config: Optional[str], split: str, audio_key: str) -> List[int]:
    """Stream the dataset and collect per-utterance sample counts.

    Streaming keeps memory bounded when the train split is large.
    """
    from datasets import load_dataset

    if config is None:
        ds = load_dataset(hf_id, split=split, streaming=True)
    else:
        ds = load_dataset(hf_id, name=config, split=split, streaming=True)

    lengths: List[int] = []
    for item in ds:
        arr = item[audio_key]["array"]
        lengths.append(int(len(arr)))
    return lengths


def _summarize_dataset(name: str, lengths: List[int], sampling_rate: int) -> DatasetStats:
    total = sum(lengths)
    return DatasetStats(
        name=name,
        n_utterances=len(lengths),
        mean_samples=statistics.fmean(lengths),
        median_samples=statistics.median(lengths),
        min_samples=min(lengths),
        max_samples=max(lengths),
        total_samples=total,
        total_seconds=total / sampling_rate,
        sampling_rate=sampling_rate,
    )


def _segmentation_stats(
    lengths: Iterable[int],
    segment: int,
    stride: int,
    sampling_rate: int,
    batch_size: int,
) -> SegmentationStats:
    """Apply BAFNet+ Audioset segmentation formula (data.py:323-330)."""
    examples_per_utt: List[int] = []
    for wav_length in lengths:
        if segment is None or wav_length < segment:
            ex = 1
        else:
            ex = int(math.ceil((wav_length - segment) / stride) + 1)
        examples_per_utt.append(ex)

    examples_total = sum(examples_per_utt)
    audio_seconds_per_epoch = (examples_total * segment) / sampling_rate
    steps_per_epoch = math.ceil(examples_total / batch_size)
    return SegmentationStats(
        segment=segment,
        stride=stride,
        examples_total=examples_total,
        examples_per_utt_mean=statistics.fmean(examples_per_utt),
        examples_per_utt_max=max(examples_per_utt),
        audio_seconds_per_epoch=audio_seconds_per_epoch,
        steps_per_epoch=steps_per_epoch,
    )


def _print_dataset_block(stats: DatasetStats, seg: SegmentationStats, epochs: int) -> None:
    sr = stats.sampling_rate
    print(f"\n[{stats.name}] @ {sr} Hz")
    print(
        f"  utterances    : {stats.n_utterances:,d}"
        f"  mean={stats.mean_samples / sr:.2f}s"
        f"  median={stats.median_samples / sr:.2f}s"
        f"  min={stats.min_samples / sr:.2f}s"
        f"  max={stats.max_samples / sr:.2f}s"
        f"  total={stats.total_hours:.2f}h"
    )
    print(
        f"  segmentation  : segment={seg.segment} ({seg.segment / sr:.2f}s)"
        f"  stride={seg.stride} ({seg.stride / sr:.2f}s)"
    )
    print(
        f"  examples      : total={seg.examples_total:,d}"
        f"  mean/utt={seg.examples_per_utt_mean:.2f}"
        f"  max/utt={seg.examples_per_utt_max}"
    )
    print(
        f"  per-epoch     : steps={seg.steps_per_epoch:,d}"
        f"  audio={seg.audio_hours_per_epoch:.2f}h"
    )
    print(
        f"  total ({epochs} ep): steps={seg.steps_per_epoch * epochs:,d}"
        f"  audio={seg.audio_hours_per_epoch * epochs:,.0f}h"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets to analyze (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Effective batch size (default: 8).")
    parser.add_argument(
        "--epochs-taps", type=int, default=150,
        help="Reference TAPS epochs (default: 150). Used for the audio-time anchor.",
    )
    parser.add_argument(
        "--vibravox-config", type=str, default=None,
        help="Optional HF config name for yskim3271/vibravox_16k (e.g., speech_clean after A-4).",
    )
    # BAFNet+ default backbone params (must match conf/model/bafnetplus.yaml + backbone defaults)
    parser.add_argument("--dense-depth", type=int, default=4)
    parser.add_argument("--num-tsblock", type=int, default=4)
    parser.add_argument("--time-block-num", type=int, default=2)
    parser.add_argument("--time-dw-kernel-size", type=int, default=3)
    parser.add_argument("--time-block-kernel", type=int, nargs="+", default=[3, 11, 23, 31])
    parser.add_argument("--hop-size", type=int, default=100)
    parser.add_argument("--win-size", type=int, default=400)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    args = parser.parse_args()

    rf_param = {
        "dense_depth": args.dense_depth,
        "num_tsblock": args.num_tsblock,
        "time_block_num": args.time_block_num,
        "time_dw_kernel_size": args.time_dw_kernel_size,
        "time_block_kernel": args.time_block_kernel,
        "hop_size": args.hop_size,
        "win_size": args.win_size,
    }
    rf = compute_receptive_field(rf_param, sampling_rate=args.sampling_rate)
    rf_samples = rf_to_segment(rf_param, sampling_rate=args.sampling_rate)
    segment = rf_samples * 2
    stride = rf_samples

    print("=" * 78)
    print("BAFNet+ Receptive-Field Auto Segmentation")
    print("=" * 78)
    print(
        f"  RF: {rf.total_rf_frames} frames -> {rf.total_rf_samples} samples"
        f" ({rf.total_rf_ms:.1f} ms)"
    )
    print(
        f"  segment = 2 * RF_samples = {segment} ({segment / args.sampling_rate:.2f}s)"
        f" | stride = RF_samples = {stride} ({stride / args.sampling_rate:.2f}s)"
    )
    print(f"  batch_size = {args.batch_size}")
    print(f"  epochs_taps anchor = {args.epochs_taps}")

    results: Dict[str, SegmentationStats] = {}
    ds_summaries: Dict[str, DatasetStats] = {}

    for dset_name in args.datasets:
        meta = dict(DATASETS[dset_name])
        if dset_name == "vibravox" and args.vibravox_config:
            meta["config"] = args.vibravox_config

        print("\n" + "-" * 78)
        print(f"Loading {dset_name} ({meta['hf_id']}, config={meta['config']!r}, split={meta['split']})...")

        try:
            lengths = _collect_lengths(
                hf_id=meta["hf_id"],
                config=meta["config"],
                split=meta["split"],
                audio_key=meta["audio_key"],
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        ds_stats = _summarize_dataset(dset_name, lengths, args.sampling_rate)
        seg_stats = _segmentation_stats(
            lengths=lengths,
            segment=segment,
            stride=stride,
            sampling_rate=args.sampling_rate,
            batch_size=args.batch_size,
        )
        ds_summaries[dset_name] = ds_stats
        results[dset_name] = seg_stats
        _print_dataset_block(ds_stats, seg_stats, epochs=args.epochs_taps)

    if "taps" in results and "vibravox" in results:
        taps = results["taps"]
        vibr = results["vibravox"]
        print("\n" + "=" * 78)
        print("Ratio (Vibravox / TAPS)")
        print("=" * 78)
        print(f"  examples_total : {vibr.examples_total / taps.examples_total:.3f}x")
        print(f"  steps_per_epoch: {vibr.steps_per_epoch / taps.steps_per_epoch:.3f}x")
        print(f"  audio_per_epoch: {vibr.audio_hours_per_epoch / taps.audio_hours_per_epoch:.3f}x")

        taps_total_audio = taps.audio_hours_per_epoch * args.epochs_taps
        recommended_eps = round(taps_total_audio / vibr.audio_hours_per_epoch)
        # Proportional warmup / valid_start_epoch (from default 30 / 100 at 150 ep)
        scale = recommended_eps / args.epochs_taps
        warmup = max(1, round(30 * scale))
        valid_start = max(0, round(100 * scale))

        print(
            f"\n  Recommended (Option β, audio-time matched to TAPS {args.epochs_taps} ep):"
        )
        print(
            f"    epochs            = {recommended_eps}"
            f"  (so total_audio ≈ TAPS {taps_total_audio:,.0f}h)"
        )
        print(f"    warmup_epochs     = {warmup}  (= round(30 * {scale:.3f}))")
        print(f"    valid_start_epoch = {valid_start}  (= round(100 * {scale:.3f}))")
        print(
            f"\n  Other options:"
        )
        print(
            f"    Option α (epochs=150): wall ≈ {vibr.audio_hours_per_epoch / taps.audio_hours_per_epoch:.2f}x TAPS"
        )
        # Step-matched anchor
        taps_total_steps = taps.steps_per_epoch * args.epochs_taps
        step_matched_eps = round(taps_total_steps / vibr.steps_per_epoch)
        print(
            f"    Option γ (step-matched): epochs = {step_matched_eps}"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
