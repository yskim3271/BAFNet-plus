"""Split legacy ``gain-robustness/<experiment>.json`` aggregates into per-offset wrapped files.

The original aggregates use a nested ``{"snr=<v>dB": {"offset=<v>dB": <metrics>}}``
shape that combines many eval runs (one per offset). The current
``src.runtime_common.build_evaluation_output`` schema is a single-run wrapper
(``{"metadata": ..., "metrics": {"<snr>dB": ...}}``) where
``bcs_gain_db``/``acs_gain_db`` are scalars. To map aggregates onto the
current schema we split along the ``offset`` axis: one output file per offset,
with all SNRs flattened into a multi-snr-style ``metrics`` tree.

Offset convention (from the pre-refactor ``scripts/eval.sh``, line:
``bcs_gain=$(python3 -c "print($offset / 2)")`` and
``acs_gain=$(python3 -c "print(-$offset / 2)")``):

    bcs_gain_db = +offset / 2
    acs_gain_db = -offset / 2
    relative    = bcs_gain_db - acs_gain_db = offset

Input:  ``results/eval/gain-robustness/<experiment>.json``
Output: ``results/eval/gain-robustness/<experiment>/offset<v>dB.json`` (per offset)
Side effect on ``--apply``: the input aggregate file is removed after all its
per-offset splits succeed.

Idempotent: files that already have a ``metadata`` key at the top level are
recognized as wrapped. If the input aggregate is missing but the output
subdirectory already exists, the tool reports ``already-split`` and leaves
things alone.

Usage:
    # Preview (default)
    python -m src.analysis.split_gain_robustness

    # Apply in place
    python -m src.analysis.split_gain_robustness --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.wrap_eval_json import (  # noqa: E402
    DATASET_LANGUAGE_MAP,
    detect_eval_stt,
    detect_per_utterance,
)

SNR_NESTED_KEY = re.compile(r"^snr=(-?\d+)dB$")
OFFSET_KEY = re.compile(r"^offset=(-?\d+)dB$")

logger = logging.getLogger(__name__)


def parse_snr(key: str) -> Optional[int]:
    """Extract the SNR integer from a ``snr=<v>dB`` key."""
    m = SNR_NESTED_KEY.match(key)
    return int(m.group(1)) if m else None


def parse_offset(key: str) -> Optional[int]:
    """Extract the offset integer from an ``offset=<v>dB`` key."""
    m = OFFSET_KEY.match(key)
    return int(m.group(1)) if m else None


def gains_for_offset(offset: int) -> Tuple[float, float]:
    """Return (bcs_gain_db, acs_gain_db) for a given offset, per eval.sh convention."""
    return float(offset) / 2.0, -float(offset) / 2.0


def regroup_by_offset(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Reshape ``{snr=...: {offset=...: metrics}}`` into ``{offset: {<snr>dB: metrics}}``."""
    by_offset: Dict[int, Dict[str, Any]] = {}
    for snr_key, cell in data.items():
        snr = parse_snr(snr_key)
        if snr is None or not isinstance(cell, dict):
            raise ValueError(f"Unexpected top-level key in aggregate: {snr_key!r}")
        snr_label = f"{snr}dB"
        for offset_key, metrics in cell.items():
            offset = parse_offset(offset_key)
            if offset is None or not isinstance(metrics, dict):
                raise ValueError(f"Unexpected offset key under {snr_key}: {offset_key!r}")
            by_offset.setdefault(offset, {})[snr_label] = metrics
    return by_offset


def build_offset_metadata(
    cfg: Any,
    aggregate_path: Path,
    per_offset_metrics: Dict[str, Any],
    experiment: str,
    offset: int,
) -> Dict[str, Any]:
    """Reconstruct metadata for one split (per-offset) file."""
    dset = cfg.get("dset", {})
    test_noise = cfg.get("test_noise", {})
    dataset = str(dset.get("dataset", "")).lower()
    eval_stt = detect_eval_stt(per_offset_metrics)

    stt_language: Optional[str] = None
    if eval_stt:
        stt_language = DATASET_LANGUAGE_MAP.get(dataset)

    bcs_gain_db, acs_gain_db = gains_for_offset(offset)

    # Sort SNR steps numerically (metrics keys are "<int>dB").
    def snr_int(k: str) -> int:
        return int(k.rstrip("dB"))

    snr_step = sorted((snr_int(k) for k in per_offset_metrics.keys()))

    # model_config points at the shared .configs/<experiment>.yaml alongside the
    # aggregate root — one level above each per-offset split file.
    model_config = aggregate_path.parent / ".configs" / f"{experiment}.yaml"

    metadata: Dict[str, Any] = {
        "dataset": dataset,
        "model_config": str(model_config),
        "chkpt_dir": f"results/experiments/{experiment}",
        "chkpt_file": "best.th",
        "snr_step": snr_step,
        "noise_dir": str(dset.get("noise_dir", "")),
        "noise_test": str(dset.get("noise_test", "")),
        "rir_dir": str(dset.get("rir_dir", "")),
        "rir_test": str(dset.get("rir_test", "")),
        "test_augment_numb": int(dset.get("test_augment_numb", 0)),
        "reverb_proportion": float(test_noise.get("reverb_proportion", 0.0)),
        "target_dB_FS": float(test_noise.get("target_dB_FS", 0.0)),
        "target_dB_FS_floating_value": float(test_noise.get("target_dB_FS_floating_value", 0.0)),
        "silence_length": float(test_noise.get("silence_length", 0.0)),
        "bcs_gain_db": bcs_gain_db,
        "acs_gain_db": acs_gain_db,
        "eval_stt": eval_stt,
        "stt_language": stt_language,
        "save_per_utterance": detect_per_utterance(per_offset_metrics),
        "reconstructed": True,
    }
    return metadata


def split_file(aggregate_path: Path, apply: bool) -> Tuple[int, Optional[str]]:
    """Split one aggregate into per-offset files. Returns (written, skip_reason)."""
    experiment = aggregate_path.stem
    out_dir = aggregate_path.parent / experiment

    with aggregate_path.open() as f:
        data = json.load(f)

    if isinstance(data, dict) and "metadata" in data and "metrics" in data:
        return 0, "already-wrapped-single-run"

    config_path = aggregate_path.parent / ".configs" / f"{experiment}.yaml"
    if not config_path.exists():
        return 0, f"missing-config:{config_path}"

    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    by_offset = regroup_by_offset(data)

    written = 0
    for offset, per_offset_metrics in sorted(by_offset.items()):
        metadata = build_offset_metadata(
            cfg_dict, aggregate_path, per_offset_metrics, experiment, offset
        )
        wrapped = {"metadata": metadata, "metrics": per_offset_metrics}
        out_path = out_dir / f"offset{offset}dB.json"
        if apply:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                json.dump(wrapped, f, indent=2)
        written += 1

    if apply:
        aggregate_path.unlink()

    return written, None


def run(root: Path, apply: bool) -> Tuple[int, int, int, int]:
    """Split every aggregate directly under ``root``. Returns counts tuple."""
    aggregates: List[Path] = sorted(root.glob("*.json"))
    scanned = len(aggregates)
    total_written = 0
    processed = 0
    errors = 0

    verb = "split" if apply else "would split"
    for path in aggregates:
        written, skip = split_file(path, apply=apply)
        rel = path.relative_to(root) if path.is_absolute() else path
        if skip:
            errors += 1
            logger.warning("skip (%s) %s", skip, rel)
            continue
        processed += 1
        total_written += written
        logger.info("%s %s → %d per-offset files", verb, rel, written)

    return scanned, processed, total_written, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/eval/gain-robustness"),
        help="Directory containing aggregate <experiment>.json files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write splits and delete originals (default: dry-run).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.root.exists():
        logger.error("Root does not exist: %s", args.root)
        sys.exit(1)

    scanned, processed, written, errors = run(args.root, apply=args.apply)

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info(
        "[%s] scanned=%d split=%d per_offset_files=%d errors=%d",
        mode,
        scanned,
        processed,
        written,
        errors,
    )
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
