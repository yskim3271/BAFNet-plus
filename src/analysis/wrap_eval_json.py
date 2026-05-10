"""Wrap legacy flat ``multi-snr`` eval JSONs into the current ``{metadata, metrics}`` schema.

Combines the sibling ``.configs/<experiment>.yaml`` snapshot with content
inferred from the JSON body (SNR keys, ``cer``/``wer`` presence,
``per_utterance`` presence) to reconstruct the metadata dict produced by
``src.runtime_common.build_evaluation_output``. The result is written back in
place with a ``reconstructed: true`` marker on the metadata.

Scope is deliberately limited to ``results/eval/multi-snr/*.json`` (single-run
aggregates). ``results/eval/gain-robustness/*.json`` are aggregate sweeps over
multiple per-cell ``bcs_gain_db``/``acs_gain_db`` combinations and do not fit
the single-metadata wrapper; they are left as legacy flat nested files. See
``docs/wiki/concepts/eval-json-schema.md`` for the format contract.

Idempotent: files that already have a ``metadata`` key at the top level are
left untouched.

Usage:
    # Preview (default)
    python -m src.analysis.wrap_eval_json

    # Apply in place
    python -m src.analysis.wrap_eval_json --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_LANGUAGE_MAP = {
    "taps": "korean",
    "vibravox": "french",
}

SNR_KEY_PATTERN = re.compile(r"^(-?\d+)dB$")

logger = logging.getLogger(__name__)


def parse_snr_keys(metrics: Dict[str, Any]) -> List[Union[int, str]]:
    """Extract SNR cell keys from metric dict.

    Returns integer SNRs from keys like ``"-20dB"`` (multi-SNR) and the
    sentinel string ``"native"`` for Vibravox-native single-cell eval (Phase 2).
    Native cells sort after integer cells.
    """
    snrs: List[Union[int, str]] = []
    for key in metrics.keys():
        m = SNR_KEY_PATTERN.match(key)
        if m:
            snrs.append(int(m.group(1)))
        elif key == "native":
            snrs.append("native")
    return sorted(snrs, key=lambda x: (isinstance(x, str), x))


def detect_eval_stt(metrics: Dict[str, Any]) -> bool:
    """True when any SNR cell carries CER/WER fields."""
    for cell in metrics.values():
        if isinstance(cell, dict) and ("cer" in cell or "wer" in cell):
            return True
    return False


def detect_per_utterance(metrics: Dict[str, Any]) -> bool:
    """True when any SNR cell carries a ``per_utterance`` block."""
    for cell in metrics.values():
        if isinstance(cell, dict) and "per_utterance" in cell:
            return True
    return False


def build_metadata(
    cfg: Any,
    json_path: Path,
    metrics: Dict[str, Any],
    experiment: str,
) -> Dict[str, Any]:
    """Reconstruct the metadata dict from config + JSON content.

    ``chkpt_dir`` and ``chkpt_file`` follow the repository convention
    ``results/experiments/<experiment>/best.th``.
    """
    dset = cfg.get("dset", {})
    test_noise = cfg.get("test_noise", {})
    dataset = str(dset.get("dataset", "")).lower()
    eval_stt = detect_eval_stt(metrics)

    stt_language: Optional[str] = None
    if eval_stt:
        stt_language = DATASET_LANGUAGE_MAP.get(dataset)

    metadata: Dict[str, Any] = {
        "dataset": dataset,
        "model_config": str(json_path.parent / ".configs" / f"{experiment}.yaml"),
        "chkpt_dir": f"results/experiments/{experiment}",
        "chkpt_file": "best.th",
        "snr_step": parse_snr_keys(metrics),
        "noise_dir": str(dset.get("noise_dir", "")),
        "noise_test": str(dset.get("noise_test", "")),
        "rir_dir": str(dset.get("rir_dir", "")),
        "rir_test": str(dset.get("rir_test", "")),
        "test_augment_numb": int(dset.get("test_augment_numb", 0)),
        "reverb_proportion": float(test_noise.get("reverb_proportion", 0.0)),
        "target_dB_FS": float(test_noise.get("target_dB_FS", 0.0)),
        "target_dB_FS_floating_value": float(test_noise.get("target_dB_FS_floating_value", 0.0)),
        "silence_length": float(test_noise.get("silence_length", 0.0)),
        "bcs_gain_db": 0.0,
        "acs_gain_db": 0.0,
        "eval_stt": eval_stt,
        "stt_language": stt_language,
        "save_per_utterance": detect_per_utterance(metrics),
        "reconstructed": True,
    }
    return metadata


def wrap_file(json_path: Path, apply: bool) -> Tuple[bool, Optional[str]]:
    """Wrap one JSON file in place. Returns ``(changed, skip_reason)``."""
    with json_path.open() as f:
        data = json.load(f)

    if isinstance(data, dict) and "metadata" in data and "metrics" in data:
        return False, "already-wrapped"

    experiment = json_path.stem
    config_path = json_path.parent / ".configs" / f"{experiment}.yaml"
    if not config_path.exists():
        return False, f"missing-config:{config_path}"

    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(data, dict):
        return False, "non-dict-root"

    metadata = build_metadata(cfg_dict, json_path, data, experiment)
    wrapped = {"metadata": metadata, "metrics": data}

    if apply:
        with json_path.open("w") as f:
            json.dump(wrapped, f, indent=2)

    return True, None


def run(root: Path, apply: bool) -> Tuple[int, int, int, int]:
    """Wrap every JSON under ``root``. Returns (scanned, changed, already, errors)."""
    files: List[Path] = sorted(root.glob("*.json"))

    scanned = len(files)
    changed = 0
    already = 0
    errors = 0

    if scanned == 0:
        logger.warning("No JSON files under %s", root)
        return 0, 0, 0, 0

    verb = "wrapped" if apply else "would wrap"
    for path in files:
        did, skip = wrap_file(path, apply=apply)
        rel = path.relative_to(root) if path.is_absolute() else path
        if did:
            changed += 1
            logger.info("%s %s", verb, rel)
        elif skip == "already-wrapped":
            already += 1
            logger.info("skip (already wrapped) %s", rel)
        else:
            errors += 1
            logger.warning("skip (%s) %s", skip, rel)

    return scanned, changed, already, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/eval/multi-snr"),
        help="Directory containing flat multi-SNR JSONs and .configs/ (default: results/eval/multi-snr).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in place (default: dry-run).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.root.exists():
        logger.error("Root does not exist: %s", args.root)
        sys.exit(1)

    scanned, changed, already, errors = run(args.root, apply=args.apply)

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info(
        "[%s] scanned=%d changed=%d already_wrapped=%d errors=%d",
        mode,
        scanned,
        changed,
        already,
        errors,
    )
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
