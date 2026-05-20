"""Cycle 18b-14 Round 2 — Split TSBlock 3 freq_stage (cycle 18b-13 seg_c) into 4 sub-op sub-graphs.

Plan B Round 2 sub-op bisection inside the freq_stage (root sub-block from
cycle 18b-13 Round 1 decision (β) ``freq_stage_is_root``). Splits the
cycle 18b-13 simplified ONNX (``bafnetplus_50ms_fp32_complex_cycle18b13_simplified.onnx``,
md5 ``cf5878f7…``) — NOT the production D3 ONNX directly — into 4 sub-segments
covering the freq_stage cascade:

* seg_c1: ``mapping/Add_48_output_0`` (= time_stage output, freq_stage input)
  → ``mapping/Add_49_output_0`` (= CAB_0 output, ``skip + x * cab.beta``).
  Sub-op: **CAB_0** (norm → pwconv1 → dwconv → sg → x * sca(x) → pwconv2 → +skip*beta).
* seg_c2: ``mapping/Add_49_output_0`` → ``mapping/Add_51_output_0`` (= GPKFFN_0
  output, ``proj_last * scale + shortcut``). Sub-op: **GPKFFN_0** (norm →
  proj_first → 4 attn_{ks}*conv_{ks} parallel branches → cat → proj_last*scale + shortcut).
* seg_c3: ``mapping/Add_51_output_0`` → ``mapping/Add_52_output_0`` (= CAB_1
  output). Sub-op: **CAB_1** (same shape as CAB_0).
* seg_c4: inputs ``mapping/Add_52_output_0`` + ``mapping/Add_48_output_0``
  → ``mapping/Add_55_output_0`` (= seg_c output, GPKFFN_1 + outer residual *
  beta_f). Sub-op: **GPKFFN_1 + outer residual addition**.
  (extract_model pulls Transpose_11 + Reshape_14 from the Add_48 dep into
  seg_c4 to compute Mul_188 = Reshape_14 * beta_f; 2 redundant shape ops
  with no numeric divergence vs seg_c1's identical computation.)

The four (8 in Round 2 chain w/ seg_a/b/d) cycle 18b-13 ONNX + 4 sidecars
(``cf5878f7…`` / ``bdfdc852…`` / ``2403cf2b…`` / ``60875da1…`` /
``0261430f…`` and ``40cc38e8…`` / ``f26da5b5…`` / ``a0dbc3c6…`` /
``f6a4211735…``) are NOT modified. The production D3 ONNX is NOT touched.
Only 4 new sub-graph ONNX files (``..._cycle18b14_seg_c{1,2,3,4}.onnx``)
+ 4 per-segment sidecar JSONs are emitted (schema
``cycle18b14-freq-stage-segment-c{1,2,3,4}-v1``).

Run:
  python -m scripts.export_bafnetplus_cycle18b14_split_freq_stage
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

import onnx
import onnx.utils


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ONNX = REPO_ROOT / "results/onnx/bafnetplus_50ms_fp32_complex_cycle18b13_simplified.onnx"
OUT_DIR = REPO_ROOT / "results/onnx"

# Boundary tensor names in the cycle 18b-13 simplified ONNX (mapping branch).
# Confirmed via Add-node graph inspection (see cycle 18b-14 plan G1 analysis):
#   Add_49 = Reshape_14 + Mul_167               (CAB_0 residual:  skip + x*beta)
#   Add_51 = Mul_176 + Add_49                   (GPKFFN_0 residual: proj_last*scale + shortcut)
#   Add_52 = Add_51 + Mul_178                   (CAB_1 residual)
#   Add_54 = Mul_187 + Add_52                   (GPKFFN_1 residual)
#   Add_55 = Add_54 + Mul_188                   (outer freq_stage residual; Mul_188 = Reshape_14 * beta_f)
TSBLOCK_3_TIME_STAGE_OUTPUT = "/mapping_core/Add_48_output_0"  # seg_c1 in, seg_c4 outer residual src
CAB_0_OUTPUT = "/mapping_core/Add_49_output_0"
GPKFFN_0_OUTPUT = "/mapping_core/Add_51_output_0"
CAB_1_OUTPUT = "/mapping_core/Add_52_output_0"
TSBLOCK_3_OUTPUT = "/mapping_core/Add_55_output_0"  # seg_c4 out (cycle 18b-13 seg_c output)


def md5sum(path: Path) -> str:
    """Return hex MD5 of a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_segment(
    src_path: Path,
    out_path: Path,
    input_names: List[str],
    output_names: List[str],
) -> dict:
    """Run ``onnx.utils.extract_model`` and return basic stats."""
    onnx.utils.extract_model(
        str(src_path),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        check_model=False,
    )
    m = onnx.load(str(out_path), load_external_data=False)
    onnx.checker.check_model(m)
    return {
        "num_nodes": len(m.graph.node),
        "num_inputs": len(m.graph.input),
        "num_outputs": len(m.graph.output),
        "input_names": [i.name for i in m.graph.input],
        "output_names": [o.name for o in m.graph.output],
    }


def write_sidecar(seg_id: str, out_onnx: Path, sidecar: dict) -> Path:
    """Write the per-segment sidecar JSON next to the ONNX."""
    sidecar_path = out_onnx.with_suffix(".onnx.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return sidecar_path


def main() -> None:
    if not SRC_ONNX.exists():
        raise SystemExit(f"missing simplified ONNX: {SRC_ONNX}")

    src_md5 = md5sum(SRC_ONNX)
    print(f"src ONNX md5: {src_md5}")
    if src_md5 != "cf5878f7e671dd2f4e9febca681603e1":
        raise SystemExit(
            f"SAFETY: cycle 18b-13 simplified ONNX md5 mismatch ({src_md5} != cf5878f7…) — refusing to proceed"
        )

    # Production D3 anchor cross-check (never written, only read).
    prod_src = REPO_ROOT / "results/onnx/bafnetplus_50ms_fp32_complex.onnx"
    if not prod_src.exists():
        raise SystemExit(f"missing production D3 ONNX: {prod_src}")
    prod_md5 = md5sum(prod_src)
    if prod_md5 != "980aa36308eba63232bb253c36249f38":
        raise SystemExit(f"SAFETY: production D3 ONNX md5 drifted ({prod_md5} != 980aa363…)")

    # ---- seg_c1: time_stage_output → cab_0_output (CAB_0) ----
    seg_c1_inputs = [TSBLOCK_3_TIME_STAGE_OUTPUT]
    seg_c1_outputs = [CAB_0_OUTPUT]
    seg_c1_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b14_seg_c1.onnx"
    print(f"\n[seg_c1 CAB_0]   in={seg_c1_inputs} → out={seg_c1_outputs}")
    seg_c1_info = extract_segment(SRC_ONNX, seg_c1_path, seg_c1_inputs, seg_c1_outputs)
    print(f"  nodes={seg_c1_info['num_nodes']}, file={seg_c1_path.name}")

    # ---- seg_c2: cab_0_output → gpkffn_0_output (GPKFFN_0) ----
    seg_c2_inputs = [CAB_0_OUTPUT]
    seg_c2_outputs = [GPKFFN_0_OUTPUT]
    seg_c2_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b14_seg_c2.onnx"
    print(f"\n[seg_c2 GPKFFN_0] in={seg_c2_inputs} → out={seg_c2_outputs}")
    seg_c2_info = extract_segment(SRC_ONNX, seg_c2_path, seg_c2_inputs, seg_c2_outputs)
    print(f"  nodes={seg_c2_info['num_nodes']}, file={seg_c2_path.name}")

    # ---- seg_c3: gpkffn_0_output → cab_1_output (CAB_1) ----
    seg_c3_inputs = [GPKFFN_0_OUTPUT]
    seg_c3_outputs = [CAB_1_OUTPUT]
    seg_c3_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b14_seg_c3.onnx"
    print(f"\n[seg_c3 CAB_1]   in={seg_c3_inputs} → out={seg_c3_outputs}")
    seg_c3_info = extract_segment(SRC_ONNX, seg_c3_path, seg_c3_inputs, seg_c3_outputs)
    print(f"  nodes={seg_c3_info['num_nodes']}, file={seg_c3_path.name}")

    # ---- seg_c4: cab_1_output + time_stage_output → tsblock_3_output (GPKFFN_1 + outer residual) ----
    # Add_48 is the outer freq_stage input (= time_stage output) needed for Mul_188 = Reshape_14*beta_f.
    # extract_model pulls Transpose_11 + Reshape_14 from the Add_48 dep — these are 2 deterministic
    # shape ops with no numeric divergence (FP32 byte-identical to seg_c1's Reshape_14).
    seg_c4_inputs = [CAB_1_OUTPUT, TSBLOCK_3_TIME_STAGE_OUTPUT]
    seg_c4_outputs = [TSBLOCK_3_OUTPUT]
    seg_c4_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b14_seg_c4.onnx"
    print(f"\n[seg_c4 GPKFFN_1+res] in={seg_c4_inputs} → out={seg_c4_outputs}")
    seg_c4_info = extract_segment(SRC_ONNX, seg_c4_path, seg_c4_inputs, seg_c4_outputs)
    print(f"  nodes={seg_c4_info['num_nodes']}, file={seg_c4_path.name}")

    # Sanity: total node count.
    total = sum(s["num_nodes"] for s in (seg_c1_info, seg_c2_info, seg_c3_info, seg_c4_info))
    full_seg_c = 113  # cycle 18b-13 seg_c node count
    print(
        f"\n--- Total nodes across 4 freq_stage sub-segments: {total} "
        f"(cycle 18b-13 seg_c base: {full_seg_c}; "
        f"+2 redundant Transpose+Reshape ops in seg_c4) ---"
    )

    # Per-segment sidecars (all stateless; freq_stage 4D Conv2d kernel=(1,K)).
    common = {
        "schema_version_root": "cycle18b14-freq-stage-segments-v1",
        "production_d3_md5": prod_md5,
        "production_d3_sidecar_md5": "804f7b1dbf7f29b1947d13e950013121",
        "cycle18b13_simplified_md5": src_md5,
        "boundary_tensors": {
            "time_stage_output": TSBLOCK_3_TIME_STAGE_OUTPUT,
            "cab_0_output": CAB_0_OUTPUT,
            "gpkffn_0_output": GPKFFN_0_OUTPUT,
            "cab_1_output": CAB_1_OUTPUT,
            "tsblock_3_output": TSBLOCK_3_OUTPUT,
        },
        "branch": "mapping",
        "tsblock_index": 3,
        "stage": "freq",
    }

    seg_info_map = {
        "c1": (seg_c1_path, seg_c1_info, seg_c1_inputs, seg_c1_outputs, "CAB_0 (first CAB inside freq_stage, ends at skip+x*beta)"),
        "c2": (seg_c2_path, seg_c2_info, seg_c2_inputs, seg_c2_outputs, "GPKFFN_0 (first GPKFFN inside freq_stage, ends at proj_last*scale+shortcut)"),
        "c3": (seg_c3_path, seg_c3_info, seg_c3_inputs, seg_c3_outputs, "CAB_1 (second CAB inside freq_stage)"),
        "c4": (seg_c4_path, seg_c4_info, seg_c4_inputs, seg_c4_outputs, "GPKFFN_1 + outer freq_stage residual (ends at +residual*beta_f)"),
    }
    for sid, (path, info, ins, outs, desc) in seg_info_map.items():
        sidecar = {
            **common,
            "schema_version": f"cycle18b14-freq-stage-segment-{sid}-v1",
            "segment_id": sid,
            "description": desc,
            "onnx_path": str(path.relative_to(REPO_ROOT.parent)),
            "onnx_md5": md5sum(path),
            "graph_stats": info,
            "core": {
                "num_states": 0,
                "non_state_inputs": list(ins),
                "non_state_outputs": list(outs),
            },
            "io": {
                "input_names": list(ins),
                "output_names": list(outs),
                "state_names": [],
                "state_shapes": [],
            },
        }
        sp = write_sidecar(sid, path, sidecar)
        print(f"  [seg_{sid}] sidecar={sp.name}, onnx md5={sidecar['onnx_md5']}, states=0")


if __name__ == "__main__":
    main()
