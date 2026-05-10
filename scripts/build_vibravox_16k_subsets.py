"""Build and publish multi-config subsets for ``yskim3271/vibravox_16k``.

Reads ``Cnam-LMSSC/vibravox`` (48 kHz, 5 BCS sensors + headset reference) and
emits three configs to ``yskim3271/vibravox_16k``:

- ``speech_clean`` (train/dev/test) — clean BCS+air paired speech
- ``speechless_noisy`` (train/dev) — BCS-recorded ambient (noise mixing source)
- ``speech_noisy`` (test) — real BCS+air recordings during noise broadcast

Per A-4 in the plan:
  - resample to 16 kHz via ``Audio(sampling_rate=16000)``
  - rename ``audio.headset_microphone`` -> ``audio.acoustic_microphone`` to
    match yskim3271/vibravox_16k legacy schema (so BAFNet+ ``data.py:84-85``
    works unchanged)
  - keep only throat + acoustic + speaker_id + sentence_id + text columns
  - push as separate ``config_name`` to preserve the existing ``default``
    (= ``speech_clean``-equivalent) config

Usage:
    huggingface-cli login   # one-time, write token
    python scripts/build_vibravox_16k_subsets.py
    python scripts/build_vibravox_16k_subsets.py --configs speechless_noisy
    python scripts/build_vibravox_16k_subsets.py --dry-run

Verify after:
    python -c "from datasets import load_dataset; \\
        ds = load_dataset('yskim3271/vibravox_16k', 'speechless_noisy', split='train'); \\
        print(list(ds.features.keys()), len(ds))"
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, List

from datasets import Audio, Value, load_dataset


logger = logging.getLogger(__name__)


SOURCE_HF_ID = "Cnam-LMSSC/vibravox"
TARGET_HF_ID = "yskim3271/vibravox_16k"
TARGET_SR = 16_000

KEEP_COLUMNS = [
    "audio.throat_microphone",
    "audio.acoustic_microphone",  # post-rename from audio.headset_microphone
    "speaker_id",
    "sentence_id",  # cast int -> str (BAFNet+ data.py:86 string concat)
    "text",         # post-rename from normalized_text
]

# Splits per config (target names after rename_map applied).
#
# NOTE: ``speech_clean`` is already published as the *default* config of
# ``yskim3271/vibravox_16k`` with the exact schema we want, so it is *not* in
# the default ``--configs`` set. Pass it explicitly only if you need to
# overwrite or republish the default content.
CONFIG_SPLITS = {
    "speech_clean":     ("train", "dev", "test"),
    "speechless_noisy": ("train", "dev"),
    "speech_noisy":     ("test",),
}

DEFAULT_CONFIGS = ["speechless_noisy", "speech_noisy"]


def _resample_and_rename(ds, sr: int = TARGET_SR):
    """Resample throat+headset to ``sr`` Hz and rename headset->acoustic.

    Keeps only the columns BAFNet+ needs (KEEP_COLUMNS), drops extra BCS sensors.
    """
    ds = ds.cast_column("audio.headset_microphone", Audio(sampling_rate=sr))
    ds = ds.cast_column("audio.throat_microphone",  Audio(sampling_rate=sr))
    ds = ds.rename_column("audio.headset_microphone", "audio.acoustic_microphone")
    available = set(ds.column_names if hasattr(ds, "column_names") and isinstance(ds.column_names, list)
                    else next(iter(ds.values())).column_names)
    keep = [c for c in KEEP_COLUMNS if c in available]
    drop = [c for c in available if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def build_and_push(config: str, *, splits: Iterable[str], dry_run: bool = False) -> None:
    logger.info("=" * 78)
    logger.info("Building config %s (splits=%s)", config, list(splits))
    logger.info("=" * 78)

    src = load_dataset(SOURCE_HF_ID, config)
    available_splits = list(src.keys())
    logger.info("  source splits: %s", available_splits)

    # Cnam-LMSSC repo uses split names "train"/"validation"/"test"; yskim3271 uses
    # "dev" instead of "validation". Map at push time.
    rename_map = {"validation": "dev"}
    dst = {}
    for src_split in available_splits:
        target_split = rename_map.get(src_split, src_split)
        if target_split not in splits:
            logger.info("  skip split %s (not in target splits)", src_split)
            continue
        ds_split = src[src_split]
        logger.info("  resampling %s (%d rows) -> %s @ %d Hz",
                    src_split, len(ds_split), target_split, TARGET_SR)
        # 1) Resample throat + headset to 16 kHz (lazy via cast_column)
        ds_split = ds_split.cast_column("audio.headset_microphone",
                                         Audio(sampling_rate=TARGET_SR))
        ds_split = ds_split.cast_column("audio.throat_microphone",
                                         Audio(sampling_rate=TARGET_SR))
        # 2) Rename to match yskim3271/vibravox_16k legacy schema
        ds_split = ds_split.rename_column("audio.headset_microphone",
                                          "audio.acoustic_microphone")
        if "normalized_text" in ds_split.column_names:
            ds_split = ds_split.rename_column("normalized_text", "text")
        elif "raw_text" in ds_split.column_names and "text" not in ds_split.column_names:
            ds_split = ds_split.rename_column("raw_text", "text")
        # 3) Cast sentence_id int -> str (BAFNet+ data.py:86 string concat).
        # Use cast_column with Value("string") for explicit Arrow schema cast —
        # ds.map(lambda) gets re-inferred to int64 on parquet push.
        if "sentence_id" in ds_split.column_names:
            ds_split = ds_split.cast_column("sentence_id", Value("string"))
        # 4) Drop extra BCS sensors / metadata not consumed by BAFNet+
        keep = [c for c in KEEP_COLUMNS if c in ds_split.column_names]
        drop = [c for c in ds_split.column_names if c not in keep]
        if drop:
            logger.info("  drop columns: %s", drop)
            ds_split = ds_split.remove_columns(drop)
        dst[target_split] = ds_split

    if not dst:
        logger.warning("  no splits produced for config %s, skipping push", config)
        return

    logger.info("  final splits: %s", {k: len(v) for k, v in dst.items()})
    logger.info("  final columns: %s", list(dst[next(iter(dst))].column_names))

    if dry_run:
        logger.info("  [DRY-RUN] skip push to %s, config_name=%s", TARGET_HF_ID, config)
        return

    from datasets import DatasetDict
    ddict = DatasetDict(dst)
    logger.info("  pushing to %s, config_name=%s ...", TARGET_HF_ID, config)
    ddict.push_to_hub(TARGET_HF_ID, config_name=config)
    logger.info("  done: %s/%s", TARGET_HF_ID, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIG_SPLITS.keys()),
        default=DEFAULT_CONFIGS,
        help=("Configs to build (default: %(default)s — speech_clean is already "
              "the default config of yskim3271/vibravox_16k, no republish needed)."),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not push to HF Hub; only resample + verify schema.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    for config in args.configs:
        splits = CONFIG_SPLITS[config]
        build_and_push(config, splits=splits, dry_run=args.dry_run)

    logger.info("All requested configs processed.")


if __name__ == "__main__":
    main()
