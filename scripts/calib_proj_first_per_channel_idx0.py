"""Cycle 18b-17 Phase 0 — per-channel Conv output max-abs calibration for A1.

Re-uses cycle 18b-16 profile sub-graphs (`bafnetplus_50ms_fp32_complex_cycle18b16_profile_g{0,1}_pre.onnx`)
and the cycle 18b-13 / 18b-14 / 18b-15 carrier segment chain on TAPS idx 0
(213 steps) to compute *per-output-channel* (256 channels) max-abs of the
two mapping TSBlock-3 freq_stage GPKFFN proj_first Conv outputs (g0 / g1).

Distinct from cycle 18b-16's `profile_gpkffn_proj_first_idx0.py` which only
stored per-tensor scalar aggregates + top-5 channel summary + SmoothQuant
input-channel s-vector — this script saves the full *256-element* per-output-
channel max-abs vector required by A1's per-channel asymmetric Clip
(`hi[c] = 1.10 * max_abs_calib[c]`, `lo[c] = -hi[c]`).

The cycle 18b-16 profile sub-graphs (`94cd09f5…/57f7c5b7…`), cycle 18b-15
seg_g{0,1}_pre/mid/post (`6b18b7ab…/4...`), cycle 18b-14 seg_c1/c3, cycle 18b-13
seg_a/b/d, and production D3 ONNX are ALL anchor files — NOT modified.

The chain math is IDENTICAL to cycle 18b-16 profile (same 11 sub-graphs, same
idx0 input, same step order) — only the aggregation differs.

Usage::

    python BAFNetPlus/scripts/calib_proj_first_per_channel_idx0.py \\
        --seg-dir BAFNetPlus/results/onnx \\
        --inputs-bin Android_projects/results/parity/cycle18b7b_state_trajectory/inputs/idx0.bin \\
        --output-json Android_projects/results/parity/cycle18b17_proj_first_per_channel_calib_idx0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort


_HEADER_FMT = "<8s6i"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAGIC = b"PESQv1\x00\x00"

TSBLOCK_3_INPUT = "/mapping_core/Reshape_12_output_0"
TSBLOCK_3_TIME_STAGE_OUTPUT = "/mapping_core/Add_48_output_0"
CAB_0_OUTPUT = "/mapping_core/Add_49_output_0"
GPKFFN_0_PROJ_FIRST = "/mapping_core/proj_first/proj_first.0_14/Conv_output_0"
GPKFFN_0_LNORM_OUT = "/mapping_core/norm_29/Add_1_output_0"
GPKFFN_0_CONCAT = "/mapping_core/Concat_33_output_0"
GPKFFN_0_OUTPUT = "/mapping_core/Add_51_output_0"
CAB_1_OUTPUT = "/mapping_core/Add_52_output_0"
GPKFFN_1_PROJ_FIRST = "/mapping_core/proj_first/proj_first.0_15/Conv_output_0"
GPKFFN_1_LNORM_OUT = "/mapping_core/norm_31/Add_1_output_0"
GPKFFN_1_CONCAT = "/mapping_core/Concat_34_output_0"
TSBLOCK_3_OUTPUT = "/mapping_core/Add_55_output_0"

EXPECTED_G0_PRE_MD5 = "94cd09f509ef505d1d3fc2382f071e1a"
EXPECTED_G1_PRE_MD5 = "57f7c5b7f8776b161671ca4e77cb905f"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_inputs_bin(path: Path):
    raw = path.read_bytes()
    magic, num_steps, freq_size, t_export, chunk, n_tensors, _ = struct.unpack(_HEADER_FMT, raw[:_HEADER_SIZE])
    if magic != _MAGIC:
        raise ValueError(f"bad magic in {path}: {magic!r}")
    if n_tensors != 4:
        raise ValueError(f"expected n_tensors=4, got {n_tensors}")
    body = np.frombuffer(raw[_HEADER_SIZE:], dtype=np.float32)
    body = body.reshape(num_steps, 4, freq_size, t_export)
    return num_steps, freq_size, t_export, chunk, body


def state_shape_from_input(model: onnx.ModelProto, name: str):
    for inp in model.graph.input:
        if inp.name == name:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                if not d.dim_value:
                    raise ValueError(f"dynamic dim in {name}")
                dims.append(int(d.dim_value))
            return tuple(dims)
    raise KeyError(name)


def _make_session(p: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-dir", type=Path, default=Path("results/onnx"))
    parser.add_argument("--inputs-bin", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    seg_paths = {
        "a":       args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b13_seg_a.onnx",
        "b":       args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b13_seg_b.onnx",
        "d":       args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b13_seg_d.onnx",
        "c1":      args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b14_seg_c1.onnx",
        "c3":      args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b14_seg_c3.onnx",
        "g0_pre":  args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b16_profile_g0_pre.onnx",
        "g0_mid":  args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g0_mid.onnx",
        "g0_post": args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g0_post.onnx",
        "g1_pre":  args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b16_profile_g1_pre.onnx",
        "g1_mid":  args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g1_mid.onnx",
        "g1_post": args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g1_post.onnx",
    }
    seg_md5 = {sid: md5sum(p) for sid, p in seg_paths.items()}
    assert seg_md5["g0_pre"] == EXPECTED_G0_PRE_MD5, f"g0_pre profile md5 drift: {seg_md5['g0_pre']}"
    assert seg_md5["g1_pre"] == EXPECTED_G1_PRE_MD5, f"g1_pre profile md5 drift: {seg_md5['g1_pre']}"
    print("[calib] seg anchor md5s:")
    for sid, m in seg_md5.items():
        print(f"[calib]   {sid:8s} {m}")

    num_steps, freq_size, t_export, chunk, inputs = read_inputs_bin(args.inputs_bin)
    print(f"[calib] inputs: steps={num_steps} freq={freq_size} T={t_export} chunk={chunk}")

    sess: Dict[str, ort.InferenceSession] = {sid: _make_session(p) for sid, p in seg_paths.items()}
    models: Dict[str, onnx.ModelProto] = {sid: onnx.load(str(p), load_external_data=False) for sid, p in seg_paths.items()}

    NON_STATE_IN = {
        "bcs_mag", "bcs_pha", "acs_mag", "acs_pha",
        TSBLOCK_3_INPUT, TSBLOCK_3_TIME_STAGE_OUTPUT,
        CAB_0_OUTPUT, GPKFFN_0_PROJ_FIRST, GPKFFN_0_CONCAT, GPKFFN_0_OUTPUT,
        CAB_1_OUTPUT, GPKFFN_1_PROJ_FIRST, GPKFFN_1_CONCAT, TSBLOCK_3_OUTPUT,
        GPKFFN_0_LNORM_OUT, GPKFFN_1_LNORM_OUT,
    }
    seg_state_inputs = {sid: [i.name for i in m.graph.input if i.name not in NON_STATE_IN] for sid, m in models.items()}
    seg_state: Dict[str, Dict[str, np.ndarray]] = {sid: {} for sid in seg_state_inputs}
    for sid, names in seg_state_inputs.items():
        for n in names:
            seg_state[sid][n] = np.zeros(state_shape_from_input(models[sid], n), dtype=np.float32)
    seg_out_idx = {sid: {o.name: i for i, o in enumerate(models[sid].graph.output)} for sid in models}

    out_channels = 256
    # Per-step per-output-channel max-abs (213 x 256) for g0_conv_out / g1_conv_out.
    stats_g0_conv = np.zeros((num_steps, out_channels), dtype=np.float32)
    stats_g1_conv = np.zeros((num_steps, out_channels), dtype=np.float32)

    for step in range(num_steps):
        step_in = inputs[step]
        bcs_mag, bcs_pha = step_in[0:1], step_in[1:2]
        acs_mag, acs_pha = step_in[2:3], step_in[3:4]

        a_in = {"bcs_mag": bcs_mag, "bcs_pha": bcs_pha}
        for n in seg_state_inputs["a"]: a_in[n] = seg_state["a"][n]
        a_out = sess["a"].run(None, a_in)
        ts3_in = a_out[seg_out_idx["a"][TSBLOCK_3_INPUT]]

        b_in = {TSBLOCK_3_INPUT: ts3_in}
        for n in seg_state_inputs["b"]: b_in[n] = seg_state["b"][n]
        b_out = sess["b"].run(None, b_in)
        time_out = b_out[seg_out_idx["b"][TSBLOCK_3_TIME_STAGE_OUTPUT]]

        c1_out = sess["c1"].run(None, {TSBLOCK_3_TIME_STAGE_OUTPUT: time_out})
        cab0 = c1_out[seg_out_idx["c1"][CAB_0_OUTPUT]]

        g0_pre_out = sess["g0_pre"].run(None, {CAB_0_OUTPUT: cab0})
        g0_conv_out = g0_pre_out[seg_out_idx["g0_pre"][GPKFFN_0_PROJ_FIRST]]
        g0_mid_out = sess["g0_mid"].run(None, {GPKFFN_0_PROJ_FIRST: g0_conv_out})
        g0_concat = g0_mid_out[seg_out_idx["g0_mid"][GPKFFN_0_CONCAT]]
        g0_post_out = sess["g0_post"].run(None, {GPKFFN_0_CONCAT: g0_concat, CAB_0_OUTPUT: cab0})
        gpkffn0 = g0_post_out[seg_out_idx["g0_post"][GPKFFN_0_OUTPUT]]
        c3_out = sess["c3"].run(None, {GPKFFN_0_OUTPUT: gpkffn0})
        cab1 = c3_out[seg_out_idx["c3"][CAB_1_OUTPUT]]

        g1_pre_out = sess["g1_pre"].run(None, {CAB_1_OUTPUT: cab1})
        g1_conv_out = g1_pre_out[seg_out_idx["g1_pre"][GPKFFN_1_PROJ_FIRST]]
        g1_mid_out = sess["g1_mid"].run(None, {GPKFFN_1_PROJ_FIRST: g1_conv_out})
        g1_concat = g1_mid_out[seg_out_idx["g1_mid"][GPKFFN_1_CONCAT]]
        g1_post_out = sess["g1_post"].run(None, {
            GPKFFN_1_CONCAT: g1_concat,
            CAB_1_OUTPUT: cab1,
            TSBLOCK_3_TIME_STAGE_OUTPUT: time_out,
        })
        ts3_out = g1_post_out[seg_out_idx["g1_post"][TSBLOCK_3_OUTPUT]]

        d_in = {TSBLOCK_3_OUTPUT: ts3_out, "acs_mag": acs_mag, "acs_pha": acs_pha}
        for n in seg_state_inputs["d"]: d_in[n] = seg_state["d"][n]
        d_out = sess["d"].run(None, d_in)

        # Per-output-channel max-abs: g0_conv_out / g1_conv_out are [1, 256, T_export=14].
        stats_g0_conv[step, :] = np.max(np.abs(g0_conv_out), axis=(0, 2)).astype(np.float32)
        stats_g1_conv[step, :] = np.max(np.abs(g1_conv_out), axis=(0, 2)).astype(np.float32)

        # Roll seg_a / seg_b / seg_d states.
        for sid, seg_out_for in (("a", a_out), ("b", b_out), ("d", d_out)):
            for n in seg_state_inputs[sid]:
                seg_state[sid][n] = seg_out_for[seg_out_idx[sid][f"next_{n}"]].astype(np.float32, copy=False)

        if step % 50 == 0:
            print(f"[calib] step {step}/{num_steps} g0_max={stats_g0_conv[step].max():.3f} g1_max={stats_g1_conv[step].max():.3f}")

    # Aggregate per-channel max over all steps.
    per_ch_max_g0 = np.max(stats_g0_conv, axis=0)  # [256]
    per_ch_max_g1 = np.max(stats_g1_conv, axis=0)

    report = {
        "schema_version": "cycle18b17-proj-first-per-channel-calib-v1",
        "produced_by": "BAFNetPlus.scripts.calib_proj_first_per_channel_idx0",
        "inputs": {
            "seg_md5": seg_md5,
            "inputs_bin": str(args.inputs_bin),
            "inputs_bin_md5": md5sum(args.inputs_bin),
            "num_steps": num_steps,
            "out_channels": out_channels,
        },
        "g0_conv_out_per_channel_max_abs": per_ch_max_g0.tolist(),
        "g1_conv_out_per_channel_max_abs": per_ch_max_g1.tolist(),
        "summary": {
            "g0_max_overall": float(per_ch_max_g0.max()),
            "g0_min_per_ch": float(per_ch_max_g0.min()),
            "g0_p50_per_ch": float(np.percentile(per_ch_max_g0, 50)),
            "g0_p90_per_ch": float(np.percentile(per_ch_max_g0, 90)),
            "g1_max_overall": float(per_ch_max_g1.max()),
            "g1_min_per_ch": float(per_ch_max_g1.min()),
            "g1_p50_per_ch": float(np.percentile(per_ch_max_g1, 50)),
            "g1_p90_per_ch": float(np.percentile(per_ch_max_g1, 90)),
            "cross_check_with_cycle18b16_profile": {
                "g0_max_should_match_~12.638": float(per_ch_max_g0.max()),
                "g1_max_should_match_~13.072": float(per_ch_max_g1.max()),
            },
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"[calib] wrote {args.output_json} md5={md5sum(args.output_json)}")
    print(f"[calib] g0_max_abs={per_ch_max_g0.max():.3f} g0_p50={np.percentile(per_ch_max_g0, 50):.3f}")
    print(f"[calib] g1_max_abs={per_ch_max_g1.max():.3f} g1_p50={np.percentile(per_ch_max_g1, 50):.3f}")
    print("[calib] OK — per-channel calibration emitted for A1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
