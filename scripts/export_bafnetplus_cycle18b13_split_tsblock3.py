"""Cycle 18b-13 Round 1 — Split production D3 ONNX into 4 sub-graphs around mapping TSBlock 3.

Plan B Round 1 sub-block bisection. Splits ``bafnetplus_50ms_fp32_complex.onnx``
(``980aa363…``) into:

* seg_A: model inputs → ``mapping/tsblock_3_input``
  (= ``/mapping_core/Reshape_12_output_0``; mapping branch up to and including
  TSBlock 2 + the reshape-into-TSBlock-3 layout shuffle; consumes
  ``bcs_mag``/``bcs_pha`` + mapping states 0..63).
* seg_B: ``mapping/tsblock_3_input`` → ``mapping/tsblock_3_time_stage_output``
  (= ``/mapping_core/Add_48_output_0``; TSBlock 3 time_stage + residual *
  beta_t; consumes mapping states 64..83 = 20 stateful conv states).
* seg_C: ``mapping/tsblock_3_time_stage_output`` →
  ``mapping/tsblock_3_output``
  (= ``/mapping_core/Add_55_output_0``; TSBlock 3 freq_stage + residual *
  beta_f; stateless — freq_stage is Conv2d(kernel=(1,K))).
* seg_D: ``mapping/tsblock_3_output`` → ``est_mag``/``phase_real``/``phase_imag``
  + remaining next_state outputs (mapping decoders, masking full branch,
  calibration, alpha, fusion).

The production ONNX is loaded but never written; the original
``bafnetplus_50ms_fp32_complex.onnx`` (md5 ``980aa363…``) is unchanged. Only
4 new sub-graph ONNX files are emitted alongside per-segment sidecar JSONs
(schema ``cycle18b13-tsblock3-segment-{a,b,c,d}-v1``).

Run:
  python -m scripts.export_bafnetplus_cycle18b13_split_tsblock3
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List

import onnx
import onnx.utils
import onnxsim


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ONNX = REPO_ROOT / "results/onnx/bafnetplus_50ms_fp32_complex.onnx"
OUT_DIR = REPO_ROOT / "results/onnx"
SIMPLIFIED_ONNX = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b13_simplified.onnx"

# Boundary tensor names in the production ONNX graph (mapping branch).
TSBLOCK_3_INPUT = "/mapping_core/Reshape_12_output_0"
TSBLOCK_3_TIME_STAGE_OUTPUT = "/mapping_core/Add_48_output_0"
TSBLOCK_3_OUTPUT = "/mapping_core/Add_55_output_0"

# Production model output names (cycle 18b-12 sidecar).
PROD_PRIMARY_OUTPUTS = ["est_mag", "phase_real", "phase_imag"]


def md5sum(path: Path) -> str:
    """Return hex MD5 of a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_names(branch: str, lo: int, hi: int, all_inputs: Iterable[str]) -> List[str]:
    """Return the ONNX state input names with state_<i> ∈ [lo, hi] for a branch.

    Args:
        branch: Branch prefix (``mapping``, ``masking``).
        lo: Inclusive lower bound on the state index.
        hi: Inclusive upper bound.
        all_inputs: Iterable of all input names from the production graph.
    """
    prefix = f"{branch}/state_"
    result: List[str] = []
    for name in all_inputs:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        try:
            sid = int(suffix.split("_", 1)[0])
        except ValueError:
            continue
        if lo <= sid <= hi:
            result.append(name)
    return sorted(result, key=lambda n: int(n.split("_", 2)[1].split("_", 1)[0]))


def collect_branch_states(branch: str, all_inputs: Iterable[str]) -> List[str]:
    """All state inputs for a given branch (alpha/calibration prefixes have no underscore)."""
    prefix = f"{branch}/"
    matched = [n for n in all_inputs if n.startswith(prefix) and "state_" in n]
    return sorted(matched, key=_state_sort_key)


def _state_sort_key(name: str) -> int:
    suffix = name.split("/", 1)[1]
    suffix = suffix.replace("state_", "")
    try:
        return int(suffix.split("_", 1)[0])
    except ValueError:
        return 99999


def next_names(state_inputs: List[str]) -> List[str]:
    """Map a list of state input names to their next_<name> output names."""
    return [f"next_{n}" for n in state_inputs]


def write_sidecar(seg_id: str, out_onnx: Path, sidecar: dict) -> Path:
    """Write the per-segment sidecar JSON next to the ONNX."""
    sidecar_path = out_onnx.with_suffix(".onnx.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return sidecar_path


def extract_segment(
    src_path: Path,
    out_path: Path,
    input_names: List[str],
    output_names: List[str],
    *,
    check_run: bool = False,
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


def main() -> None:
    if not SRC_ONNX.exists():
        raise SystemExit(f"missing production ONNX: {SRC_ONNX}")

    src_md5 = md5sum(SRC_ONNX)
    print(f"src ONNX md5: {src_md5}")
    if src_md5 != "980aa36308eba63232bb253c36249f38":
        raise SystemExit(f"SAFETY: production D3 ONNX md5 mismatch ({src_md5} != 980aa363…) — refusing to proceed")

    # Production D3 has 210 Shape ops + 218 Gather ops + 216 Reshape ops with
    # dynamic-shape Shape→Gather→Concat chains. These create transitive
    # dependencies that prevent clean ``onnx.utils.extract_model`` splits at
    # boundary tensors (seg_c would pull in bcs_mag + all mapping states 0..63
    # purely for shape inputs). Simplifying with onnxsim folds them into
    # concrete Constants without altering FP32 semantics: Conv count is
    # preserved exactly, Shape ops drop to 0, dynamic shape chains become
    # literal Constants. Conv weights, scales, biases are untouched.
    print("Loading + simplifying production ONNX (constant fold + shape elimination) ...")
    model = onnx.load(str(SRC_ONNX), load_external_data=False)
    print(f"  before: nodes={len(model.graph.node)}")
    model, ok = onnxsim.simplify(model)
    if not ok:
        raise SystemExit("onnxsim.simplify reported failure")
    print(f"  after:  nodes={len(model.graph.node)}")
    onnx.save(model, str(SIMPLIFIED_ONNX))
    simp_md5 = md5sum(SIMPLIFIED_ONNX)
    print(f"  simplified ONNX md5: {simp_md5} → {SIMPLIFIED_ONNX.name}")
    temp_inferred = SIMPLIFIED_ONNX

    all_inputs = [i.name for i in model.graph.input]
    all_outputs = [o.name for o in model.graph.output]

    # Verify boundary tensors exist in value_info.
    value_info_names = {vi.name for vi in model.graph.value_info}
    for tname in (TSBLOCK_3_INPUT, TSBLOCK_3_TIME_STAGE_OUTPUT, TSBLOCK_3_OUTPUT):
        if tname not in value_info_names:
            print(f"WARN: boundary tensor {tname} not in value_info (extract_model may still work)")

    mapping_state_0_63 = state_names("mapping", 0, 63, all_inputs)
    mapping_state_64_83 = state_names("mapping", 64, 83, all_inputs)
    mapping_state_84_91 = state_names("mapping", 84, 91, all_inputs)
    masking_state_all = collect_branch_states("masking", all_inputs)
    calibration_states = sorted([n for n in all_inputs if n.startswith("calibration/state_")])
    alpha_states = sorted([n for n in all_inputs if n.startswith("alpha/state_")])

    assert len(mapping_state_0_63) == 64, f"mapping 0..63 count = {len(mapping_state_0_63)}"
    assert len(mapping_state_64_83) == 20, f"mapping 64..83 count = {len(mapping_state_64_83)}"
    assert len(mapping_state_84_91) == 8, f"mapping 84..91 count = {len(mapping_state_84_91)}"
    assert len(masking_state_all) == 92, f"masking count = {len(masking_state_all)}"
    assert len(calibration_states) == 2, f"calibration count = {len(calibration_states)}"
    assert len(alpha_states) == 4, f"alpha count = {len(alpha_states)}"

    # ---- seg_A: model inputs → tsblock_3_input + mapping states 0..63 ----
    seg_a_inputs = ["bcs_mag", "bcs_pha"] + mapping_state_0_63
    seg_a_outputs = [TSBLOCK_3_INPUT] + next_names(mapping_state_0_63)
    seg_a_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b13_seg_a.onnx"
    print(f"\n[seg_a] in={len(seg_a_inputs)} out={len(seg_a_outputs)}")
    seg_a_info = extract_segment(temp_inferred, seg_a_path, seg_a_inputs, seg_a_outputs)
    print(f"  nodes={seg_a_info['num_nodes']}, file={seg_a_path.name}")

    # ---- seg_B: tsblock_3_input + mapping states 64..83 → tsblock_3_time_stage_output ----
    seg_b_inputs = [TSBLOCK_3_INPUT] + mapping_state_64_83
    seg_b_outputs = [TSBLOCK_3_TIME_STAGE_OUTPUT] + next_names(mapping_state_64_83)
    seg_b_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b13_seg_b.onnx"
    print(f"\n[seg_b] in={len(seg_b_inputs)} out={len(seg_b_outputs)}")
    seg_b_info = extract_segment(temp_inferred, seg_b_path, seg_b_inputs, seg_b_outputs)
    print(f"  nodes={seg_b_info['num_nodes']}, file={seg_b_path.name}")

    # ---- seg_C: tsblock_3_time_stage_output → tsblock_3_output (stateless) ----
    seg_c_inputs = [TSBLOCK_3_TIME_STAGE_OUTPUT]
    seg_c_outputs = [TSBLOCK_3_OUTPUT]
    seg_c_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b13_seg_c.onnx"
    print(f"\n[seg_c] in={len(seg_c_inputs)} out={len(seg_c_outputs)}")
    seg_c_info = extract_segment(temp_inferred, seg_c_path, seg_c_inputs, seg_c_outputs)
    print(f"  nodes={seg_c_info['num_nodes']}, file={seg_c_path.name}")

    # ---- seg_D: tsblock_3_output + everything else → final outputs + remaining states ----
    seg_d_inputs = (
        [TSBLOCK_3_OUTPUT, "acs_mag", "acs_pha"]
        + mapping_state_84_91
        + masking_state_all
        + calibration_states
        + alpha_states
    )
    seg_d_outputs = (
        list(PROD_PRIMARY_OUTPUTS)
        + next_names(mapping_state_84_91)
        + next_names(masking_state_all)
        + next_names(calibration_states)
        + next_names(alpha_states)
    )
    seg_d_path = OUT_DIR / "bafnetplus_50ms_fp32_complex_cycle18b13_seg_d.onnx"
    print(f"\n[seg_d] in={len(seg_d_inputs)} out={len(seg_d_outputs)}")
    seg_d_info = extract_segment(temp_inferred, seg_d_path, seg_d_inputs, seg_d_outputs)
    print(f"  nodes={seg_d_info['num_nodes']}, file={seg_d_path.name}")

    # Sanity: total node count of 4 segments.
    total = sum(s["num_nodes"] for s in (seg_a_info, seg_b_info, seg_c_info, seg_d_info))
    print(f"\n--- Total nodes across 4 segments: {total} (simplified base: {len(model.graph.node)}, production: 7460) ---")

    # Write sidecars.
    common = {
        "schema_version": "cycle18b13-tsblock3-segments-v1",
        "production_d3_md5": src_md5,
        "production_d3_sidecar_md5": "804f7b1dbf7f29b1947d13e950013121",
        "simplified_onnx_md5": simp_md5,
        "boundary_tensors": {
            "tsblock_3_input": TSBLOCK_3_INPUT,
            "tsblock_3_time_stage_output": TSBLOCK_3_TIME_STAGE_OUTPUT,
            "tsblock_3_output": TSBLOCK_3_OUTPUT,
        },
        "branch": "mapping",
        "tsblock_index": 3,
    }
    # Load production sidecar to inherit state_shapes for the state inputs used by each segment.
    prod_sidecar = json.loads((SRC_ONNX.with_suffix(".onnx.json")).read_text())
    prod_state_names = prod_sidecar["io"]["state_names"]
    prod_state_shapes = prod_sidecar["io"]["state_shapes"]
    shape_by_name = {n: s for n, s in zip(prod_state_names, prod_state_shapes)}

    NON_STATE = {"bcs_mag", "bcs_pha", "acs_mag", "acs_pha",
                 TSBLOCK_3_INPUT, TSBLOCK_3_TIME_STAGE_OUTPUT, TSBLOCK_3_OUTPUT}

    seg_info_map = {
        "a": (seg_a_path, seg_a_info, seg_a_inputs, seg_a_outputs, "pre-block-3 mapping branch"),
        "b": (seg_b_path, seg_b_info, seg_b_inputs, seg_b_outputs, "TSBlock 3 time_stage + residual*beta_t"),
        "c": (seg_c_path, seg_c_info, seg_c_inputs, seg_c_outputs, "TSBlock 3 freq_stage + residual*beta_f (stateless)"),
        "d": (seg_d_path, seg_d_info, seg_d_inputs, seg_d_outputs, "mapping decoders + masking branch + fusion"),
    }
    for sid, (path, info, ins, outs, desc) in seg_info_map.items():
        seg_state_names = [n for n in ins if n not in NON_STATE]
        seg_state_shapes = [shape_by_name[n] for n in seg_state_names]
        sidecar = {
            **common,
            "schema_version": f"cycle18b13-tsblock3-segment-{sid}-v1",
            "segment_id": sid,
            "description": desc,
            "onnx_path": str(path.relative_to(REPO_ROOT.parent)),
            "onnx_md5": md5sum(path),
            "graph_stats": info,
            "core": {
                "num_states": len(seg_state_names),
                "non_state_inputs": [n for n in ins if n in NON_STATE],
                "non_state_outputs": [n for n in outs if n in NON_STATE or not n.startswith("next_")],
            },
            "io": {
                "input_names": ins,
                "output_names": outs,
                "state_names": seg_state_names,
                "state_shapes": seg_state_shapes,
            },
        }
        sp = write_sidecar(sid, path, sidecar)
        print(f"  [seg_{sid}] sidecar={sp.name}, onnx md5={sidecar['onnx_md5']}, states={len(seg_state_names)}")


if __name__ == "__main__":
    main()
