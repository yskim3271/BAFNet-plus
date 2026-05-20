"""Cycle 18b-15 Round 3 — Split GPKFFN_0/GPKFFN_1 into PRE/MID/POST sub-op sub-graphs.

Plan B Round 3 sub-op bisection inside the GPKFFN sub-blocks (root sub-op-type
from cycle 18b-14 Round 2 decision (γ) ``both_contribute`` with GPKFFN dominant).
Splits the cycle 18b-13 simplified ONNX
(``bafnetplus_50ms_fp32_complex_cycle18b13_simplified.onnx``, md5 ``cf5878f7…``)
— NOT the production D3 ONNX directly, and NOT cycle 18b-14's seg_c2/c4 — into
6 sub-segments covering the two GPKFFNs inside TSBlock 3 freq_stage:

* seg_g0_pre: ``mapping/Add_49_output_0`` (= CAB_0 output) →
  ``/mapping_core/proj_first/proj_first.0_14/Conv_output_0`` (= GPKFFN_0
  proj_first Conv output, BEFORE chunk).
  Sub-op: **GPKFFN_0 PRE** (norm + proj_first Conv 1x1).
* seg_g0_mid: ``/mapping_core/proj_first/proj_first.0_14/Conv_output_0`` →
  ``/mapping_core/Concat_33_output_0`` (= cat of 4 attn_ks * conv_ks results).
  Sub-op: **GPKFFN_0 MID** (4 Slice chunk + 4 parallel (attn_ks.0 + attn_ks.1
  + conv_ks + mul) + Concat).
* seg_g0_post: inputs ``/mapping_core/Concat_33_output_0`` +
  ``/mapping_core/Add_49_output_0`` (shortcut) → ``/mapping_core/Add_51_output_0``
  (= GPKFFN_0 output). Sub-op: **GPKFFN_0 POST** (proj_last Conv + scale Mul
  + shortcut Add).
* seg_g1_pre: ``/mapping_core/Add_52_output_0`` (= CAB_1 output) →
  ``/mapping_core/proj_first/proj_first.0_15/Conv_output_0`` (= GPKFFN_1
  proj_first Conv output). Sub-op: **GPKFFN_1 PRE**.
* seg_g1_mid: ``/mapping_core/proj_first/proj_first.0_15/Conv_output_0`` →
  ``/mapping_core/Concat_34_output_0``. Sub-op: **GPKFFN_1 MID**.
* seg_g1_post: inputs ``/mapping_core/Concat_34_output_0`` +
  ``/mapping_core/Add_52_output_0`` (shortcut) + ``/mapping_core/Add_48_output_0``
  (outer freq_stage residual src) → ``/mapping_core/Add_55_output_0``
  (= freq_stage output = TSBlock 3 output). Sub-op: **GPKFFN_1 POST + outer
  freq_stage residual** (proj_last + scale*x + shortcut + Reshape+Transpose+
  Reshape+Mul(beta_f) + Add).

The five cycle 18b-13 ONNX (``cf5878f7…`` simplified + 4 seg_{a,b,c,d}) and
the four cycle 18b-14 ONNX (``ad460471…/fccd9efc…/53141a0e…/dc856b24…``) +
sidecars are NOT modified. The production D3 ONNX is NOT touched. Only 6 new
sub-graph ONNX files (``..._cycle18b15_seg_g{0,1}_{pre,mid,post}.onnx``) +
6 per-segment sidecar JSONs are emitted (schema
``cycle18b15-gpkffn-segment-g{0,1}_{pre,mid,post}-v1``).

Run::

    python -m scripts.export_bafnetplus_cycle18b15_split_gpkffn
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
# Confirmed via direct ONNX graph inspection on cycle 18b-14 seg_c2 (GPKFFN_0,
# 34 nodes) and seg_c4 (GPKFFN_1+outer residual, 39 nodes) — all 10 tensors
# also present in cycle 18b-13 simplified.onnx (verified before extraction).
#
# GPKFFN_0 (cycle 18b-14 seg_c2): Add_49 → ... → Add_51
#   Add_49        = CAB_0 output  (seg_g0_pre input, also seg_g0_post shortcut)
#   proj_first.0_14/Conv = norm + Conv 1x1   (seg_g0_pre output, seg_g0_mid input)
#   Concat_33     = cat(4× attn_ks*conv_ks)  (seg_g0_mid output, seg_g0_post input)
#   Add_51        = proj_last*scale + shortcut  (seg_g0_post output, = GPKFFN_0 final)
#
# GPKFFN_1 (cycle 18b-14 seg_c4): Add_52 → ... → Add_55 (incl outer residual)
#   Add_52        = CAB_1 output  (seg_g1_pre input, also seg_g1_post shortcut)
#   proj_first.0_15/Conv = norm + Conv 1x1   (seg_g1_pre output, seg_g1_mid input)
#   Concat_34     = cat(4× attn_ks*conv_ks)  (seg_g1_mid output, seg_g1_post input)
#   Add_48        = time_stage output  (seg_g1_post outer-residual src; pulled
#                   in via Transpose+Reshape pair — matches seg_c4 pattern)
#   Add_55        = (proj_last*scale + shortcut) + Reshape_14 * beta_f
#                   (seg_g1_post output, = freq_stage output = TSBlock 3 output)
CAB_0_OUTPUT = "/mapping_core/Add_49_output_0"
GPKFFN_0_PROJ_FIRST = "/mapping_core/proj_first/proj_first.0_14/Conv_output_0"
GPKFFN_0_CONCAT = "/mapping_core/Concat_33_output_0"
GPKFFN_0_OUTPUT = "/mapping_core/Add_51_output_0"
CAB_1_OUTPUT = "/mapping_core/Add_52_output_0"
GPKFFN_1_PROJ_FIRST = "/mapping_core/proj_first/proj_first.0_15/Conv_output_0"
GPKFFN_1_CONCAT = "/mapping_core/Concat_34_output_0"
TSBLOCK_3_TIME_STAGE_OUTPUT = "/mapping_core/Add_48_output_0"  # outer-residual src
TSBLOCK_3_OUTPUT = "/mapping_core/Add_55_output_0"  # seg_g1_post out (= seg_c4 out)


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

    # Cycle 18b-14 seg_c2/c4 cross-check (never written, only read).
    for sid, expected_md5 in (
        ("c2", "fccd9efc76433d817fde1604492f67a0"),
        ("c4", "dc856b24552e47ad40c44d2551bff949"),
    ):
        p = REPO_ROOT / f"results/onnx/bafnetplus_50ms_fp32_complex_cycle18b14_seg_{sid}.onnx"
        m = md5sum(p)
        if m != expected_md5:
            raise SystemExit(f"SAFETY: cycle 18b-14 seg_{sid} md5 drifted ({m} != {expected_md5})")

    # ---- seg_g0_pre: CAB_0_out → proj_first.0_14 (GPKFFN_0 PRE: norm + proj_first) ----
    seg_g0_pre_inputs = [CAB_0_OUTPUT]
    seg_g0_pre_outputs = [GPKFFN_0_PROJ_FIRST]
    seg_g0_pre_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g0_pre.onnx"
    print(f"\n[seg_g0_pre GPKFFN_0 PRE] in={seg_g0_pre_inputs} → out={seg_g0_pre_outputs}")
    seg_g0_pre_info = extract_segment(SRC_ONNX, seg_g0_pre_path, seg_g0_pre_inputs, seg_g0_pre_outputs)
    print(f"  nodes={seg_g0_pre_info['num_nodes']}, file={seg_g0_pre_path.name}")

    # ---- seg_g0_mid: proj_first.0_14 → Concat_33 (GPKFFN_0 MID: chunk + 4× attn*conv + cat) ----
    seg_g0_mid_inputs = [GPKFFN_0_PROJ_FIRST]
    seg_g0_mid_outputs = [GPKFFN_0_CONCAT]
    seg_g0_mid_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g0_mid.onnx"
    print(f"\n[seg_g0_mid GPKFFN_0 MID] in={seg_g0_mid_inputs} → out={seg_g0_mid_outputs}")
    seg_g0_mid_info = extract_segment(SRC_ONNX, seg_g0_mid_path, seg_g0_mid_inputs, seg_g0_mid_outputs)
    print(f"  nodes={seg_g0_mid_info['num_nodes']}, file={seg_g0_mid_path.name}")

    # ---- seg_g0_post: Concat_33 + Add_49 (shortcut) → Add_51 (GPKFFN_0 POST: proj_last + scale + residual) ----
    seg_g0_post_inputs = [GPKFFN_0_CONCAT, CAB_0_OUTPUT]
    seg_g0_post_outputs = [GPKFFN_0_OUTPUT]
    seg_g0_post_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g0_post.onnx"
    print(f"\n[seg_g0_post GPKFFN_0 POST] in={seg_g0_post_inputs} → out={seg_g0_post_outputs}")
    seg_g0_post_info = extract_segment(SRC_ONNX, seg_g0_post_path, seg_g0_post_inputs, seg_g0_post_outputs)
    print(f"  nodes={seg_g0_post_info['num_nodes']}, file={seg_g0_post_path.name}")

    # ---- seg_g1_pre: CAB_1_out → proj_first.0_15 (GPKFFN_1 PRE) ----
    seg_g1_pre_inputs = [CAB_1_OUTPUT]
    seg_g1_pre_outputs = [GPKFFN_1_PROJ_FIRST]
    seg_g1_pre_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g1_pre.onnx"
    print(f"\n[seg_g1_pre GPKFFN_1 PRE] in={seg_g1_pre_inputs} → out={seg_g1_pre_outputs}")
    seg_g1_pre_info = extract_segment(SRC_ONNX, seg_g1_pre_path, seg_g1_pre_inputs, seg_g1_pre_outputs)
    print(f"  nodes={seg_g1_pre_info['num_nodes']}, file={seg_g1_pre_path.name}")

    # ---- seg_g1_mid: proj_first.0_15 → Concat_34 (GPKFFN_1 MID) ----
    seg_g1_mid_inputs = [GPKFFN_1_PROJ_FIRST]
    seg_g1_mid_outputs = [GPKFFN_1_CONCAT]
    seg_g1_mid_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g1_mid.onnx"
    print(f"\n[seg_g1_mid GPKFFN_1 MID] in={seg_g1_mid_inputs} → out={seg_g1_mid_outputs}")
    seg_g1_mid_info = extract_segment(SRC_ONNX, seg_g1_mid_path, seg_g1_mid_inputs, seg_g1_mid_outputs)
    print(f"  nodes={seg_g1_mid_info['num_nodes']}, file={seg_g1_mid_path.name}")

    # ---- seg_g1_post: Concat_34 + Add_52 (shortcut) + Add_48 (outer residual src) → Add_55 ----
    # GPKFFN_1 POST + outer freq_stage residual. Pulls Reshape+Transpose+Reshape
    # chain from Add_48 dep to compute Mul_188 = Reshape_14 * beta_f (same
    # pattern as cycle 18b-14 seg_c4 — 3 deterministic shape ops; FP32 byte-
    # identical to seg_c1's path).
    seg_g1_post_inputs = [GPKFFN_1_CONCAT, CAB_1_OUTPUT, TSBLOCK_3_TIME_STAGE_OUTPUT]
    seg_g1_post_outputs = [TSBLOCK_3_OUTPUT]
    seg_g1_post_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g1_post.onnx"
    print(f"\n[seg_g1_post GPKFFN_1 POST + outer res] in={seg_g1_post_inputs} → out={seg_g1_post_outputs}")
    seg_g1_post_info = extract_segment(SRC_ONNX, seg_g1_post_path, seg_g1_post_inputs, seg_g1_post_outputs)
    print(f"  nodes={seg_g1_post_info['num_nodes']}, file={seg_g1_post_path.name}")

    # Sanity: total node count vs cycle 18b-14 seg_c2 (34) + seg_c4 (39) = 73.
    total = sum(s["num_nodes"] for s in (
        seg_g0_pre_info, seg_g0_mid_info, seg_g0_post_info,
        seg_g1_pre_info, seg_g1_mid_info, seg_g1_post_info,
    ))
    print(
        f"\n--- Total nodes across 6 GPKFFN sub-segments: {total} "
        f"(cycle 18b-14 seg_c2 + seg_c4 base: 34 + 39 = 73; "
        f"per-segment may differ slightly due to extract_model re-emitting "
        f"deterministic Reshape/Transpose for outer residual) ---"
    )

    # Per-segment sidecars (all stateless; GPKFFN is functional-stateless).
    common = {
        "schema_version_root": "cycle18b15-gpkffn-segments-v1",
        "production_d3_md5": prod_md5,
        "production_d3_sidecar_md5": "804f7b1dbf7f29b1947d13e950013121",
        "cycle18b13_simplified_md5": src_md5,
        "cycle18b14_seg_c2_md5": "fccd9efc76433d817fde1604492f67a0",
        "cycle18b14_seg_c4_md5": "dc856b24552e47ad40c44d2551bff949",
        "boundary_tensors": {
            "cab_0_output": CAB_0_OUTPUT,
            "gpkffn_0_proj_first": GPKFFN_0_PROJ_FIRST,
            "gpkffn_0_concat": GPKFFN_0_CONCAT,
            "gpkffn_0_output": GPKFFN_0_OUTPUT,
            "cab_1_output": CAB_1_OUTPUT,
            "gpkffn_1_proj_first": GPKFFN_1_PROJ_FIRST,
            "gpkffn_1_concat": GPKFFN_1_CONCAT,
            "tsblock_3_time_stage_output": TSBLOCK_3_TIME_STAGE_OUTPUT,
            "tsblock_3_output": TSBLOCK_3_OUTPUT,
        },
        "branch": "mapping",
        "tsblock_index": 3,
        "stage": "freq",
    }

    seg_info_map = {
        "g0_pre": (
            seg_g0_pre_path, seg_g0_pre_info, seg_g0_pre_inputs, seg_g0_pre_outputs,
            "GPKFFN_0 PRE (norm + proj_first Conv 1x1, BEFORE chunk)",
        ),
        "g0_mid": (
            seg_g0_mid_path, seg_g0_mid_info, seg_g0_mid_inputs, seg_g0_mid_outputs,
            "GPKFFN_0 MID (4 Slice chunk + 4 parallel attn_ks*conv_ks + Concat)",
        ),
        "g0_post": (
            seg_g0_post_path, seg_g0_post_info, seg_g0_post_inputs, seg_g0_post_outputs,
            "GPKFFN_0 POST (proj_last + scale*x + shortcut Add)",
        ),
        "g1_pre": (
            seg_g1_pre_path, seg_g1_pre_info, seg_g1_pre_inputs, seg_g1_pre_outputs,
            "GPKFFN_1 PRE (norm + proj_first Conv 1x1, BEFORE chunk)",
        ),
        "g1_mid": (
            seg_g1_mid_path, seg_g1_mid_info, seg_g1_mid_inputs, seg_g1_mid_outputs,
            "GPKFFN_1 MID (4 Slice chunk + 4 parallel attn_ks*conv_ks + Concat)",
        ),
        "g1_post": (
            seg_g1_post_path, seg_g1_post_info, seg_g1_post_inputs, seg_g1_post_outputs,
            "GPKFFN_1 POST + outer freq_stage residual (proj_last + scale*x + shortcut + Reshape+Transpose+Reshape+Mul(beta_f) + Add → Add_55)",
        ),
    }
    for sid, (path, info, ins, outs, desc) in seg_info_map.items():
        sidecar = {
            **common,
            "schema_version": f"cycle18b15-gpkffn-segment-{sid}-v1",
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
