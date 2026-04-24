"""Retarget ``metadata.model_config`` to the canonical ``.hydra/config.yaml``.

After the 2026-04-24 wrap/split retrofits, every wrapped eval JSON carries
a ``metadata.model_config`` pointer originally set to the sed-rewritten
snapshot ``results/eval/<scenario>/.configs/<experiment>.yaml``. The original
training config still lives under
``results/experiments/<experiment>/.hydra/config.yaml``, and the rewritten
snapshot is redundant (and even carries an unanchored-sed path duplication
bug for locally trained models). This utility updates ``model_config`` to
point at the canonical training config so the ``.configs/`` snapshot dirs
can be removed without dangling references.

Experiment name is derived from the JSON path:

- ``results/eval/multi-snr/<experiment>.json`` → ``<experiment>``
- ``results/eval/gain-robustness/<experiment>/offset<v>dB.json``
  → ``<experiment>``

Files under a ``.configs`` directory are skipped.

Idempotent: if ``metadata.model_config`` is already the canonical target,
the file is left untouched. Non-wrapped JSONs are silently skipped.

Usage:
    # Preview (default)
    python -m src.analysis.retarget_model_config

    # Apply in place
    python -m src.analysis.retarget_model_config --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def derive_experiment(json_path: Path, root: Path) -> Optional[str]:
    """Extract the experiment name from ``json_path`` relative to ``root``.

    Expects either ``multi-snr/<exp>.json`` or
    ``gain-robustness/<exp>/offset<v>dB.json`` under ``root``.
    """
    try:
        rel = json_path.relative_to(root)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 2:
        return None

    scenario = parts[0]
    if scenario == "multi-snr" and len(parts) == 2:
        return Path(parts[1]).stem
    if scenario == "gain-robustness" and len(parts) == 3:
        return parts[1]
    return None


def process_file(json_path: Path, root: Path, apply: bool) -> Tuple[bool, Optional[str]]:
    """Retarget one file. Returns ``(changed, skip_reason)``."""
    with json_path.open() as f:
        data = json.load(f)

    if not (isinstance(data, dict) and "metadata" in data and "metrics" in data):
        return False, "not-wrapped"

    experiment = derive_experiment(json_path, root)
    if not experiment:
        return False, "experiment-undetermined"

    new_target = f"results/experiments/{experiment}/.hydra/config.yaml"
    metadata = data["metadata"]
    if metadata.get("model_config") == new_target:
        return False, "already-retargeted"

    metadata["model_config"] = new_target
    if apply:
        with json_path.open("w") as f:
            json.dump(data, f, indent=2)

    return True, None


def run(root: Path, apply: bool) -> Tuple[int, int, int, int]:
    """Walk ``root`` and retarget every wrapped JSON. Returns counts tuple."""
    files: List[Path] = [
        p for p in sorted(root.rglob("*.json")) if ".configs" not in p.parts
    ]

    scanned = len(files)
    changed = 0
    already = 0
    errors = 0

    if scanned == 0:
        logger.warning("No JSON files under %s", root)
        return 0, 0, 0, 0

    verb = "retargeted" if apply else "would retarget"
    for path in files:
        did, skip = process_file(path, root, apply=apply)
        rel = path.relative_to(root) if path.is_absolute() else path
        if did:
            changed += 1
            logger.info("%s %s", verb, rel)
        elif skip == "already-retargeted":
            already += 1
            logger.info("skip (already retargeted) %s", rel)
        elif skip == "not-wrapped":
            logger.info("skip (not wrapped) %s", rel)
        else:
            errors += 1
            logger.warning("skip (%s) %s", skip, rel)

    return scanned, changed, already, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/eval"),
        help="Directory to walk (default: results/eval).",
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
        "[%s] scanned=%d changed=%d already_retargeted=%d errors=%d",
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
