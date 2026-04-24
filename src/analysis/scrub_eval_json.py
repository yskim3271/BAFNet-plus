"""Remove deprecated metric keys from ``results/eval/**/*.json``.

Walks every JSON file under the given root (default: ``results/eval``) and
removes any occurrence of the deprecated metric keys. The traversal is a
structural recursion over dicts and lists, so both the current wrapped format
(``{"metadata": ..., "metrics": {...}}``) and the legacy flat format
(``{"<snr>": {...}}``) are handled without special cases. Nested
``gain-robustness`` files (snr x offset) are also handled.

Idempotent: re-running on a clean tree reports zero modifications.

Usage:
    # Preview changes (default)
    python -m src.analysis.scrub_eval_json

    # Apply in place
    python -m src.analysis.scrub_eval_json --apply

    # Scrub a subdirectory
    python -m src.analysis.scrub_eval_json --root results/eval/multi-snr --apply

See ``docs/wiki/concepts/eval-json-schema.md`` for the canonical metric
schema.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPRECATED_KEYS = (
    "dnsmos_p808",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovr",
    "utmos",
)

logger = logging.getLogger(__name__)


def scrub(obj: Any, deprecated: Tuple[str, ...]) -> int:
    """Recursively remove deprecated keys. Returns the number of keys removed."""
    removed = 0
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in deprecated:
                del obj[key]
                removed += 1
            else:
                removed += scrub(obj[key], deprecated)
    elif isinstance(obj, list):
        for item in obj:
            removed += scrub(item, deprecated)
    return removed


def process_file(path: Path, apply: bool) -> int:
    """Load, scrub, and optionally write back one file. Returns keys removed."""
    with path.open() as f:
        data = json.load(f)

    removed = scrub(data, DEPRECATED_KEYS)
    if removed and apply:
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    return removed


def run(root: Path, apply: bool) -> Tuple[int, int, int]:
    """Scrub every JSON under ``root``. Returns (files_scanned, files_changed, keys_removed)."""
    files: List[Path] = sorted(root.rglob("*.json"))

    if not files:
        logger.warning("No JSON files under %s", root)
        return 0, 0, 0

    files_changed = 0
    keys_removed = 0

    for path in files:
        removed = process_file(path, apply=apply)
        if removed:
            files_changed += 1
            keys_removed += removed
            rel = path.relative_to(root) if path.is_absolute() else path
            verb = "removed" if apply else "would remove"
            logger.info("%s %d key(s) in %s", verb, removed, rel)

    return len(files), files_changed, keys_removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/eval"),
        help="Root directory to walk (default: results/eval).",
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

    scanned, changed, removed = run(args.root, apply=args.apply)

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info(
        "[%s] scanned=%d files_changed=%d keys_removed=%d deprecated=%s",
        mode,
        scanned,
        changed,
        removed,
        ",".join(DEPRECATED_KEYS),
    )


if __name__ == "__main__":
    main()
