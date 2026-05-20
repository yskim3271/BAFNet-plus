"""Cycle 18b-16 Phase 1 — host-side activation profile for GPKFFN proj_first.

Inserts the LayerNorm-output tensors (``norm_29/Add_1_output_0`` and
``norm_31/Add_1_output_0``) as additional graph outputs of the cycle 18b-15
``seg_g0_pre`` / ``seg_g1_pre`` sub-graphs (no node added, just extra
``ValueInfoProto`` exposed on ``graph.output``). The augmented sub-graphs are
saved as ``bafnetplus_50ms_fp32_complex_cycle18b16_profile_g{0,1}_pre.onnx``
(new anchors; cycle 18b-15 ONNX/sidecar set is NOT modified).

Then drives the cycle 18b-15 11-segment chain on ``cycle18b7b_state_trajectory/
inputs/idx0.bin`` (213 steps) but swaps in the two profile sub-graphs for
``g0_pre`` / ``g1_pre`` so that ``Conv input (= LayerNorm output)`` and
``Conv output (= proj_first output)`` are captured at every step for both
GPKFFNs.

Reports per-step + per-input-channel max-abs (channel j ∈ [0,63]) of:

* ``norm_29/Add_1`` (= GPKFFN_0 LayerNorm output = proj_first Conv input)
* ``proj_first.0_14/Conv`` (= GPKFFN_0 proj_first Conv output)
* ``norm_31/Add_1`` (= GPKFFN_1 LayerNorm output = proj_first Conv input)
* ``proj_first.0_15/Conv`` (= GPKFFN_1 proj_first Conv output)

Plus per-tensor aggregated stats over all 213 steps + all positions:

* overall max-abs, top-5 channel indices, max-abs over (step × pos)
* SmoothQuant candidate scale ``s[j] = max(|X[j]|)^α / max(|W[j]|)^α``
  with ``α = 0.5`` (canonical balanced) per input channel
* Clip threshold candidates ``T ∈ {Y_max·0.5, 256, 512, 1024}``
* FP16-safety: count steps where ``|Y_max|`` exceeds ``{128, 256, 512, 1024,
  4096, 65504}``

The cycle 18b-15 anchor ONNX/sidecars (``seg_g0_pre.onnx`` md5 ``6b18b7ab…``
etc.) and production D3 ONNX (``980aa363…``) are NOT modified.

Usage::

    python scripts/profile_gpkffn_proj_first_idx0.py \\
        --seg-dir results/onnx \\
        --inputs-bin ../Android_projects/results/parity/cycle18b7b_state_trajectory/inputs/idx0.bin \\
        --output-json ../Android_projects/results/parity/cycle18b16_proj_first_activation_profile_idx0.json \\
        --profile-out-dir results/onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnx.utils
import onnxruntime as ort
from onnx import numpy_helper


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

GPKFFN_SUB_SEG_IDS = ["g0_pre", "g0_mid", "g0_post", "g1_pre", "g1_mid", "g1_post"]

PROJ_FIRST_WEIGHT_NAMES = {
    "g0": "mapping_core.sequence_block.3.freq_stage.0.1.proj_first.0.weight",
    "g1": "mapping_core.sequence_block.3.freq_stage.1.1.proj_first.0.weight",
}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_inputs_bin(path: Path):
    raw = path.read_bytes()
    magic, num_steps, freq_size, t_export, chunk, n_tensors, _ = struct.unpack(
        _HEADER_FMT, raw[:_HEADER_SIZE]
    )
    if magic != _MAGIC:
        raise ValueError(f"bad magic in {path}: {magic!r}")
    if n_tensors != 4:
        raise ValueError(f"expected n_tensors=4, got {n_tensors}")
    body = np.frombuffer(raw[_HEADER_SIZE:], dtype=np.float32)
    body = body.reshape(num_steps, 4, freq_size, t_export)
    return num_steps, freq_size, t_export, chunk, body


def state_shape_from_input(model: onnx.ModelProto, name: str) -> Tuple[int, ...]:
    for inp in model.graph.input:
        if inp.name == name:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_value:
                    dims.append(int(d.dim_value))
                else:
                    raise ValueError(f"dynamic dim in {name}")
            return tuple(dims)
    raise KeyError(f"{name} not in model inputs")


def augment_seg_with_extra_output(src_onnx: Path, extra_tensor_name: str, dst_onnx: Path) -> str:
    """Add ``extra_tensor_name`` as an additional ``graph.output`` of ``src_onnx``.

    No nodes are added; ORT computes the intermediate tensor as a side-effect.
    The original output (Conv output) is preserved as the first output.
    Returns the dst md5.
    """
    m = onnx.load(str(src_onnx))
    # Sanity: the extra tensor must appear as an output of one of the existing nodes.
    found = False
    elem_type = None
    shape = None
    for n in m.graph.node:
        for out in n.output:
            if out == extra_tensor_name:
                found = True
    if not found:
        raise ValueError(f"{extra_tensor_name} not produced by any node in {src_onnx}")
    # Infer shape from the first input (LN output has same shape as Conv input).
    # Convolution preserves [B, C_in -> C_out=4*C_in, T]; LN output is [B, C_in, T].
    # We use the input ValueInfo as a template and override the type.
    in_vi = m.graph.input[0]
    elem_type = in_vi.type.tensor_type.elem_type  # FLOAT
    dims = []
    for d in in_vi.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        else:
            dims.append(None)
    new_vi = onnx.helper.make_tensor_value_info(extra_tensor_name, elem_type, dims)
    m.graph.output.insert(0, new_vi)  # LN output FIRST, Conv output SECOND
    onnx.checker.check_model(m)
    onnx.save(m, str(dst_onnx))
    return md5sum(dst_onnx)


def _make_session(onnx_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])


def _channel_max_abs(arr: np.ndarray) -> np.ndarray:
    """Return per-input-channel max-abs from a [B, C, T] tensor."""
    return np.max(np.abs(arr), axis=(0, 2)).astype(np.float32)


def _hist_buckets(maxabs: np.ndarray) -> Dict[str, int]:
    """Histogram of max-abs values into FP16-safety bins."""
    edges = [32.0, 128.0, 256.0, 512.0, 1024.0, 4096.0, 65504.0]
    counts = {f"le_{int(e)}": int(np.sum(maxabs <= e)) for e in edges}
    counts["gt_65504"] = int(np.sum(maxabs > 65504.0))
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-dir", type=Path, default=Path("results/onnx"))
    parser.add_argument("--src-onnx", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle18b13_simplified.onnx"),
                        help="Used as weight source for SmoothQuant s-vector.")
    parser.add_argument("--inputs-bin", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--profile-out-dir", type=Path,
                        default=Path("results/onnx"))
    args = parser.parse_args()

    src_md5 = md5sum(args.src_onnx)
    print(f"[profile] cycle 18b-13 simplified ONNX md5: {src_md5}")

    # ---- Build augmented profile sub-graphs.
    g0_pre_src = args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g0_pre.onnx"
    g1_pre_src = args.seg_dir / "bafnetplus_50ms_fp32_complex_cycle18b15_seg_g1_pre.onnx"
    g0_pre_md5_orig = md5sum(g0_pre_src)
    g1_pre_md5_orig = md5sum(g1_pre_src)
    expected_g0 = "6b18b7ab1d0cc24f8324953e228b3e6c"
    expected_g1 = "29ddc0315e97772abb17d41ea5938e59"
    assert g0_pre_md5_orig == expected_g0, f"seg_g0_pre md5 drift: {g0_pre_md5_orig}"
    assert g1_pre_md5_orig == expected_g1, f"seg_g1_pre md5 drift: {g1_pre_md5_orig}"
    args.profile_out_dir.mkdir(parents=True, exist_ok=True)
    g0_pre_dst = args.profile_out_dir / "bafnetplus_50ms_fp32_complex_cycle18b16_profile_g0_pre.onnx"
    g1_pre_dst = args.profile_out_dir / "bafnetplus_50ms_fp32_complex_cycle18b16_profile_g1_pre.onnx"
    g0_pre_md5 = augment_seg_with_extra_output(g0_pre_src, GPKFFN_0_LNORM_OUT, g0_pre_dst)
    g1_pre_md5 = augment_seg_with_extra_output(g1_pre_src, GPKFFN_1_LNORM_OUT, g1_pre_dst)
    print(f"[profile] wrote {g0_pre_dst.name} md5={g0_pre_md5}")
    print(f"[profile] wrote {g1_pre_dst.name} md5={g1_pre_md5}")

    # Re-verify cycle 18b-15 anchors UNCHANGED.
    assert md5sum(g0_pre_src) == expected_g0, "seg_g0_pre anchor mutated!"
    assert md5sum(g1_pre_src) == expected_g1, "seg_g1_pre anchor mutated!"

    # ---- Build chain sessions (cycle 18b-15 carryover + augmented g0_pre/g1_pre).
    seg_carry_paths = {sid: args.seg_dir / f"bafnetplus_50ms_fp32_complex_cycle18b13_seg_{sid}.onnx"
                       for sid in ("a", "b", "d")}
    seg_cab_paths = {sid: args.seg_dir / f"bafnetplus_50ms_fp32_complex_cycle18b14_seg_{sid}.onnx"
                     for sid in ("c1", "c3")}
    seg_gpkffn_carry = {sid: args.seg_dir / f"bafnetplus_50ms_fp32_complex_cycle18b15_seg_{sid}.onnx"
                        for sid in ("g0_mid", "g0_post", "g1_mid", "g1_post")}
    seg_profile = {"g0_pre": g0_pre_dst, "g1_pre": g1_pre_dst}
    all_paths = {**seg_carry_paths, **seg_cab_paths, **seg_gpkffn_carry, **seg_profile}
    for sid, p in all_paths.items():
        if not p.exists():
            raise SystemExit(f"missing seg_{sid}: {p}")

    num_steps, freq_size, t_export, chunk, inputs = read_inputs_bin(args.inputs_bin)
    print(f"[profile] inputs: steps={num_steps}, freq={freq_size}, T={t_export}, chunk={chunk}")

    print("[profile] loading 11 segment sessions ...")
    sess: Dict[str, ort.InferenceSession] = {sid: _make_session(p) for sid, p in all_paths.items()}
    seg_models: Dict[str, onnx.ModelProto] = {sid: onnx.load(str(p), load_external_data=False)
                                              for sid, p in all_paths.items()}

    NON_STATE_IN = {
        "bcs_mag", "bcs_pha", "acs_mag", "acs_pha",
        TSBLOCK_3_INPUT, TSBLOCK_3_TIME_STAGE_OUTPUT,
        CAB_0_OUTPUT, GPKFFN_0_PROJ_FIRST, GPKFFN_0_CONCAT, GPKFFN_0_OUTPUT,
        CAB_1_OUTPUT, GPKFFN_1_PROJ_FIRST, GPKFFN_1_CONCAT, TSBLOCK_3_OUTPUT,
        GPKFFN_0_LNORM_OUT, GPKFFN_1_LNORM_OUT,
    }
    seg_state_inputs: Dict[str, List[str]] = {
        sid: [i.name for i in m.graph.input if i.name not in NON_STATE_IN]
        for sid, m in seg_models.items()
    }
    seg_state: Dict[str, Dict[str, np.ndarray]] = {sid: {} for sid in seg_state_inputs}
    for sid, names in seg_state_inputs.items():
        for n in names:
            seg_state[sid][n] = np.zeros(state_shape_from_input(seg_models[sid], n), dtype=np.float32)
    seg_out_idx = {sid: {o.name: i for i, o in enumerate(seg_models[sid].graph.output)} for sid in seg_models}

    # ---- Stats accumulators.
    in_channels = 64
    # Per-step, per-channel max-abs (213, 64) for each tensor.
    stats: Dict[str, np.ndarray] = {
        "g0_norm_in":      np.zeros((num_steps, in_channels), dtype=np.float32),
        "g0_lnorm_out":    np.zeros((num_steps, in_channels), dtype=np.float32),
        "g0_conv_out":     np.zeros((num_steps, in_channels * 4), dtype=np.float32),  # 256 output channels
        "g1_norm_in":      np.zeros((num_steps, in_channels), dtype=np.float32),
        "g1_lnorm_out":    np.zeros((num_steps, in_channels), dtype=np.float32),
        "g1_conv_out":     np.zeros((num_steps, in_channels * 4), dtype=np.float32),
    }
    overall_max: Dict[str, float] = {k: 0.0 for k in stats}
    step_of_max: Dict[str, int] = {k: -1 for k in stats}

    for step in range(num_steps):
        step_in = inputs[step]
        bcs_mag, bcs_pha = step_in[0:1], step_in[1:2]
        acs_mag, acs_pha = step_in[2:3], step_in[3:4]

        # seg_a
        a_in = {"bcs_mag": bcs_mag, "bcs_pha": bcs_pha}
        for n in seg_state_inputs["a"]: a_in[n] = seg_state["a"][n]
        a_out = sess["a"].run(None, a_in)
        ts3_in = a_out[seg_out_idx["a"][TSBLOCK_3_INPUT]]
        # seg_b
        b_in = {TSBLOCK_3_INPUT: ts3_in}
        for n in seg_state_inputs["b"]: b_in[n] = seg_state["b"][n]
        b_out = sess["b"].run(None, b_in)
        time_out = b_out[seg_out_idx["b"][TSBLOCK_3_TIME_STAGE_OUTPUT]]
        # seg_c1
        c1_out = sess["c1"].run(None, {TSBLOCK_3_TIME_STAGE_OUTPUT: time_out})
        cab0 = c1_out[seg_out_idx["c1"][CAB_0_OUTPUT]]
        # PROFILE g0_pre (augmented: 2 outputs: LN_out, Conv_out)
        g0_pre_out = sess["g0_pre"].run(None, {CAB_0_OUTPUT: cab0})
        g0_ln_out = g0_pre_out[seg_out_idx["g0_pre"][GPKFFN_0_LNORM_OUT]]
        g0_conv_out = g0_pre_out[seg_out_idx["g0_pre"][GPKFFN_0_PROJ_FIRST]]
        # seg_g0_mid + seg_g0_post + seg_c3
        g0_mid_out = sess["g0_mid"].run(None, {GPKFFN_0_PROJ_FIRST: g0_conv_out})
        g0_concat = g0_mid_out[seg_out_idx["g0_mid"][GPKFFN_0_CONCAT]]
        g0_post_out = sess["g0_post"].run(None, {GPKFFN_0_CONCAT: g0_concat, CAB_0_OUTPUT: cab0})
        gpkffn0 = g0_post_out[seg_out_idx["g0_post"][GPKFFN_0_OUTPUT]]
        c3_out = sess["c3"].run(None, {GPKFFN_0_OUTPUT: gpkffn0})
        cab1 = c3_out[seg_out_idx["c3"][CAB_1_OUTPUT]]
        # PROFILE g1_pre
        g1_pre_out = sess["g1_pre"].run(None, {CAB_1_OUTPUT: cab1})
        g1_ln_out = g1_pre_out[seg_out_idx["g1_pre"][GPKFFN_1_LNORM_OUT]]
        g1_conv_out = g1_pre_out[seg_out_idx["g1_pre"][GPKFFN_1_PROJ_FIRST]]
        # finish chain so states roll correctly
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

        # Capture per-channel max-abs.
        for key, arr in (
            ("g0_norm_in",   cab0),
            ("g0_lnorm_out", g0_ln_out),
            ("g0_conv_out",  g0_conv_out),
            ("g1_norm_in",   cab1),
            ("g1_lnorm_out", g1_ln_out),
            ("g1_conv_out",  g1_conv_out),
        ):
            ch_max = _channel_max_abs(arr)
            stats[key][step, :] = ch_max
            step_max = float(np.max(ch_max))
            if step_max > overall_max[key]:
                overall_max[key] = step_max
                step_of_max[key] = step

        # Roll states (a/b/d only).
        for sid, seg_out_for in (("a", a_out), ("b", b_out), ("d", d_out)):
            for n in seg_state_inputs[sid]:
                seg_state[sid][n] = seg_out_for[seg_out_idx[sid][f"next_{n}"]].astype(np.float32, copy=False)

    # ---- Aggregate: per-channel max over all steps + SmoothQuant s candidates.
    # Load proj_first weights from cycle 18b-13 simplified.
    m_src = onnx.load(str(args.src_onnx), load_external_data=False)
    weights = {}
    for t in m_src.graph.initializer:
        if t.name in PROJ_FIRST_WEIGHT_NAMES.values():
            weights[t.name] = numpy_helper.to_array(t).astype(np.float32)
    assert PROJ_FIRST_WEIGHT_NAMES["g0"] in weights
    assert PROJ_FIRST_WEIGHT_NAMES["g1"] in weights
    W_g0 = weights[PROJ_FIRST_WEIGHT_NAMES["g0"]]  # [256, 64, 1]
    W_g1 = weights[PROJ_FIRST_WEIGHT_NAMES["g1"]]
    # per-input-channel max-abs of W: max over (out_dim, kernel)
    W_g0_in_ch_max = np.max(np.abs(W_g0), axis=(0, 2))  # [64]
    W_g1_in_ch_max = np.max(np.abs(W_g1), axis=(0, 2))  # [64]

    def aggregate(per_step_stats: np.ndarray) -> Dict[str, object]:
        flat = per_step_stats.reshape(-1)
        ch_max = np.max(per_step_stats, axis=0)  # max over steps, per channel
        step_max = np.max(per_step_stats, axis=1)  # max over channels, per step
        top5_ch = np.argsort(-ch_max)[:5].tolist()
        worst_step = int(np.argmax(step_max))
        return {
            "max_abs_overall": float(np.max(per_step_stats)),
            "max_abs_overall_position_step": worst_step,
            "max_abs_per_step_top5_steps": np.argsort(-step_max)[:5].tolist(),
            "top5_channel_max_abs": {int(c): float(ch_max[c]) for c in top5_ch},
            "mean_per_step_max": float(np.mean(step_max)),
            "std_per_step_max": float(np.std(step_max)),
            "channel_max_abs_summary": {
                "min": float(np.min(ch_max)),
                "p50": float(np.percentile(ch_max, 50)),
                "p90": float(np.percentile(ch_max, 90)),
                "max": float(np.max(ch_max)),
            },
            "fp16_safety_buckets_per_channel_max": _hist_buckets(ch_max),
        }

    # SmoothQuant s candidates: s[j] = max(|X[j]|)^α / max(|W[j]|)^α, α=0.5
    alpha = 0.5
    X_g0_ch_max = np.max(stats["g0_lnorm_out"], axis=0)  # [64], Conv input
    X_g1_ch_max = np.max(stats["g1_lnorm_out"], axis=0)
    eps = 1e-6  # avoid div by 0
    s_g0 = np.power(np.maximum(X_g0_ch_max, eps), alpha) / np.power(np.maximum(W_g0_in_ch_max, eps), alpha)
    s_g1 = np.power(np.maximum(X_g1_ch_max, eps), alpha) / np.power(np.maximum(W_g1_in_ch_max, eps), alpha)

    # Clip threshold candidates from g0/g1 Conv output max.
    Y_g0_max = float(np.max(stats["g0_conv_out"]))
    Y_g1_max = float(np.max(stats["g1_conv_out"]))
    Y_max = max(Y_g0_max, Y_g1_max)

    report = {
        "schema_version": "cycle18b16-proj-first-activation-profile-v1",
        "produced_by": "BAFNetPlus.scripts.profile_gpkffn_proj_first_idx0",
        "inputs": {
            "cycle18b13_simplified_md5": src_md5,
            "seg_g0_pre_orig_md5": g0_pre_md5_orig,
            "seg_g1_pre_orig_md5": g1_pre_md5_orig,
            "seg_g0_pre_profile_md5": g0_pre_md5,
            "seg_g1_pre_profile_md5": g1_pre_md5,
            "inputs_bin": str(args.inputs_bin),
            "inputs_bin_md5": md5sum(args.inputs_bin),
            "num_steps": num_steps,
            "in_channels": in_channels,
        },
        "per_tensor_aggregate": {k: aggregate(v) for k, v in stats.items()},
        "smoothquant_alpha_0_5_s_vector": {
            "g0": {
                "X_in_ch_max_top5_idx": np.argsort(-X_g0_ch_max)[:5].tolist(),
                "X_in_ch_max_top5_val": [float(X_g0_ch_max[i]) for i in np.argsort(-X_g0_ch_max)[:5]],
                "W_in_ch_max_top5_idx": np.argsort(-W_g0_in_ch_max)[:5].tolist(),
                "W_in_ch_max_top5_val": [float(W_g0_in_ch_max[i]) for i in np.argsort(-W_g0_in_ch_max)[:5]],
                "s_min": float(np.min(s_g0)),
                "s_max": float(np.max(s_g0)),
                "s_p50": float(np.percentile(s_g0, 50)),
                "s_p90": float(np.percentile(s_g0, 90)),
                "alpha": alpha,
            },
            "g1": {
                "X_in_ch_max_top5_idx": np.argsort(-X_g1_ch_max)[:5].tolist(),
                "X_in_ch_max_top5_val": [float(X_g1_ch_max[i]) for i in np.argsort(-X_g1_ch_max)[:5]],
                "W_in_ch_max_top5_idx": np.argsort(-W_g1_in_ch_max)[:5].tolist(),
                "W_in_ch_max_top5_val": [float(W_g1_in_ch_max[i]) for i in np.argsort(-W_g1_in_ch_max)[:5]],
                "s_min": float(np.min(s_g1)),
                "s_max": float(np.max(s_g1)),
                "s_p50": float(np.percentile(s_g1, 50)),
                "s_p90": float(np.percentile(s_g1, 90)),
                "alpha": alpha,
            },
        },
        "clip_threshold_candidates": {
            "Y_g0_max_abs": Y_g0_max,
            "Y_g1_max_abs": Y_g1_max,
            "Y_overall_max_abs": Y_max,
            "candidates_T_hard": [256.0, 512.0, 1024.0],
            "candidate_T_at_p99_global": None,  # compute below
        },
        "fp16_safety": {
            "Y_g0_steps_over_256": int(np.sum(np.max(stats["g0_conv_out"], axis=1) > 256.0)),
            "Y_g0_steps_over_512": int(np.sum(np.max(stats["g0_conv_out"], axis=1) > 512.0)),
            "Y_g1_steps_over_256": int(np.sum(np.max(stats["g1_conv_out"], axis=1) > 256.0)),
            "Y_g1_steps_over_512": int(np.sum(np.max(stats["g1_conv_out"], axis=1) > 512.0)),
            "X_g0_steps_over_256": int(np.sum(np.max(stats["g0_lnorm_out"], axis=1) > 256.0)),
            "X_g1_steps_over_256": int(np.sum(np.max(stats["g1_lnorm_out"], axis=1) > 256.0)),
        },
        "smoothquant_s_full_vectors": {
            "g0_s_64ch": s_g0.tolist(),
            "g1_s_64ch": s_g1.tolist(),
            "g0_inv_s_64ch": (1.0 / np.maximum(s_g0, eps)).tolist(),
            "g1_inv_s_64ch": (1.0 / np.maximum(s_g1, eps)).tolist(),
        },
    }
    # Compute T at p99 of Conv output max-abs.
    y_all = np.concatenate([stats["g0_conv_out"].reshape(-1), stats["g1_conv_out"].reshape(-1)])
    report["clip_threshold_candidates"]["candidate_T_at_p99_global"] = float(np.percentile(y_all, 99.9))
    report["clip_threshold_candidates"]["candidate_T_at_p999_global"] = float(np.percentile(y_all, 99.99))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"[profile] wrote {args.output_json}")
    print(f"[profile] G1 PASS — per-channel/per-step profile captured for both GPKFFN proj_firsts (idx 0).")
    print(f"[profile] g0 max-abs: X={overall_max['g0_lnorm_out']:.2f} Y={overall_max['g0_conv_out']:.2f}")
    print(f"[profile] g1 max-abs: X={overall_max['g1_lnorm_out']:.2f} Y={overall_max['g1_conv_out']:.2f}")
    print(f"[profile] suggested SmoothQuant alpha=0.5 g0 s p50={np.percentile(s_g0, 50):.3f} p90={np.percentile(s_g0, 90):.3f}")
    print(f"[profile] suggested SmoothQuant alpha=0.5 g1 s p50={np.percentile(s_g1, 50):.3f} p90={np.percentile(s_g1, 90):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
