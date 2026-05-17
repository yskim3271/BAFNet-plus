"""Cycle 18b — generate split-graph sidecar JSON for the QAT pilot trunk.

Reads the canonical D1 split sidecar (``bafnetplus_50ms_split_int8_t2.json``),
rewrites only the ``trunk`` section to point at the QAT-overridden INT8 ONNX
(``bafnetplus_50ms_int8_qat_pilot_trunk_t2.onnx``), keeps the ``head`` /
``checkpoint`` / ``geometry`` / ``t2_prelu_decompose`` blocks intact, and writes
the result to ``bafnetplus_50ms_split_int8_qat_pilot_t2.json``.

The output sidecar is the ``--combined-sidecar`` input to
``Android_projects/scripts/parity/check_onnx_parity.py --split --split-trunk-int8``
in the cycle 18b Step 10 directional host-parity check.

Usage::

    python -m scripts.build_qat_split_sidecar \\
        --base results/onnx/bafnetplus_50ms_split_int8_t2.json \\
        --qat-trunk-onnx results/onnx/bafnetplus_50ms_int8_qat_pilot_trunk_t2.onnx \\
        --output results/onnx/bafnetplus_50ms_split_int8_qat_pilot_t2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True,
                    help="D1 split sidecar JSON to clone from (trunk + head + checkpoint blocks).")
    ap.add_argument("--qat-trunk-onnx", type=Path, required=True,
                    help="Cycle 18b Option β output INT8 ONNX (QAT-overridden trunk).")
    ap.add_argument("--qat-trunk-sidecar", type=Path, default=None,
                    help="Sidecar JSON for the QAT trunk ONNX. Defaults to <qat-trunk-onnx>.json.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output combined split sidecar JSON path.")
    ap.add_argument("--schema-tag", type=str,
                    default="cycle18b-qat-pilot-split-trunk-t2",
                    help="schema_version token to stamp on the trunk block.")
    args = ap.parse_args()

    if not args.base.exists():
        raise SystemExit(f"[FATAL] base sidecar not found: {args.base}")
    if not args.qat_trunk_onnx.exists():
        raise SystemExit(f"[FATAL] qat-trunk-onnx not found: {args.qat_trunk_onnx}")

    qat_sidecar = args.qat_trunk_sidecar or args.qat_trunk_onnx.with_suffix(
        args.qat_trunk_onnx.suffix + ".json"
    )

    base = json.loads(args.base.read_text())
    if "trunk" not in base:
        raise SystemExit(f"[FATAL] base sidecar has no 'trunk' block: {args.base}")

    qat_size = args.qat_trunk_onnx.stat().st_size
    qat_md5 = hashlib.md5(args.qat_trunk_onnx.read_bytes()).hexdigest()

    trunk = dict(base["trunk"])
    trunk["onnx_file"] = args.qat_trunk_onnx.name
    trunk["sidecar_file"] = qat_sidecar.name
    trunk["onnx_size_bytes"] = qat_size
    trunk["onnx_md5"] = qat_md5
    trunk["schema_version"] = args.schema_tag
    trunk["produced_by"] = "scripts.build_qat_split_sidecar"
    trunk["qat_lineage"] = {
        "base_split_sidecar": args.base.name,
        "base_int8_md5": base["trunk"].get("onnx_md5"),
        "qat_trunk_path": str(args.qat_trunk_onnx),
        "qat_trunk_md5": qat_md5,
    }
    base["trunk"] = trunk
    base["schema_version"] = f"{base.get('schema_version', 's21-bafnetplus-split-v1')}+qat-pilot"
    base["produced_by"] = "scripts.build_qat_split_sidecar"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base, indent=2, sort_keys=False))
    print(f"Wrote {args.output} (trunk md5={qat_md5}, size={qat_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
