"""Cycle 18b-17 Candidate B1 — chunked proj_first Conv (4-way split along input channel).

Plan C extension category: decomposition / partial-sum locality. Splits each
of the 2 target proj_first 1x1 Conv (64 input channels, 256 output channels)
into 4 sub-Convs each on 16 input channels, summed via an Add tree. The
intuition: HTP-FP16's Conv accumulator sees a partial sum of 16 input
channels per sub-Conv (1/4 the accumulator span), then the Add tree merges
in FP16 with bounded magnitude per intermediate. The total math is identical
in real arithmetic (Y[o] = sum_c W[o,c] * X[c] = sum_g sum_{c in chunk_g}
W[o,c] * X[c]); FP32 reordering noise is bounded at ~1e-5..5e-5.

Targets ONLY the 2 mapping TSBlock-3 freq_stage proj_first Conv nodes:

  * ``/mapping_core/proj_first/proj_first.0_14/Conv`` (g0, 64 -> 256)
  * ``/mapping_core/proj_first/proj_first.0_15/Conv`` (g1, 64 -> 256)

Structure inserted per target:

  Slice(z, [0:16])  -> z0  -> Conv(W[:, 0:16], no bias)  -> c0
  Slice(z, [16:32]) -> z1  -> Conv(W[:, 16:32], no bias) -> c1
  Slice(z, [32:48]) -> z2  -> Conv(W[:, 32:48], no bias) -> c2
  Slice(z, [48:64]) -> z3  -> Conv(W[:, 48:64], no bias) -> c3
  Add(c0, c1) -> t01;  Add(c2, c3) -> t23
  Add(t01, t23) -> t
  Add(t, b_broadcast[1, 256, 1]) -> y_chunked
  (all downstream consumers of the original Conv output are re-pointed to y_chunked.)

The original Conv node is REMOVED (its weight + bias initializers stay in the
file as dead initializers since they are no longer referenced by any node;
the new chunked Convs use new initializers of name ``..._chunk0/1/2/3.weight``).

Production D3 ONNX (md5 ``980aa363...``) is NOT modified; output is written to
``bafnetplus_50ms_fp32_complex_cycle18b17_b1_chunkconv.onnx`` with sidecar
schema ``s20-bafnetplus-functional-fp32-complex-cycle18b17-b1-chunkconv-v1``.

Usage::

    python BAFNetPlus/scripts/export_bafnetplus_cycle18b17_candidate_b1_chunkconv.py \\
        --src BAFNetPlus/results/onnx/bafnetplus_50ms_fp32_complex.onnx \\
        --dst BAFNetPlus/results/onnx/bafnetplus_50ms_fp32_complex_cycle18b17_b1_chunkconv.onnx
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


EXPECTED_SRC_MD5 = "980aa36308eba63232bb253c36249f38"

TARGETS = (
    {
        "key": "g0",
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_14/Conv",
        "weight_init":    "mapping_core.sequence_block.3.freq_stage.0.1.proj_first.0.weight",
        "bias_init":      "mapping_core.sequence_block.3.freq_stage.0.1.proj_first.0.bias",
    },
    {
        "key": "g1",
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_15/Conv",
        "weight_init":    "mapping_core.sequence_block.3.freq_stage.1.1.proj_first.0.weight",
        "bias_init":      "mapping_core.sequence_block.3.freq_stage.1.1.proj_first.0.bias",
    },
)

NUM_CHUNKS = 4
CHUNK_SIZE = 16  # 64 / 4
OUT_CHANNELS = 256


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def chunk_one(model: onnx.ModelProto, target: dict) -> dict:
    inits = {t.name: (i, t) for i, t in enumerate(model.graph.initializer)}
    W = numpy_helper.to_array(inits[target["weight_init"]][1]).astype(np.float32)  # [256, 64, 1]
    b = numpy_helper.to_array(inits[target["bias_init"]][1]).astype(np.float32)    # [256]
    assert W.shape == (OUT_CHANNELS, NUM_CHUNKS * CHUNK_SIZE, 1), f"W shape {W.shape}"
    assert b.shape == (OUT_CHANNELS,), f"b shape {b.shape}"

    target_node = None
    target_idx = None
    for i, n in enumerate(model.graph.node):
        if n.name == target["conv_node_name"]:
            target_node = n
            target_idx = i
            break
    if target_node is None:
        raise ValueError(f"Conv node {target['conv_node_name']} not found")
    z_input = target_node.input[0]   # LN Add_1_output_0 (= LN output)
    orig_output = target_node.output[0]  # /…/Conv_output_0
    chunked_output = orig_output + "_chunkconv_cycle18b17"

    prefix = target["conv_node_name"]
    new_inits: List = []
    new_nodes: List = []

    # Slice init constants (starts/ends/axes) - one set per chunk.
    starts_axes = {}
    for g in range(NUM_CHUNKS):
        starts_axes[g] = {
            "starts": np.array([g * CHUNK_SIZE], dtype=np.int64),
            "ends":   np.array([(g + 1) * CHUNK_SIZE], dtype=np.int64),
            "axes":   np.array([1], dtype=np.int64),
        }
        for k, arr in starts_axes[g].items():
            init = numpy_helper.from_array(arr, name=f"{prefix}_chunk{g}_{k}_cycle18b17")
            new_inits.append(init)

    # Chunk weight slices.
    chunk_weights = []
    for g in range(NUM_CHUNKS):
        W_g = W[:, g * CHUNK_SIZE : (g + 1) * CHUNK_SIZE, :]  # [256, 16, 1]
        w_init = numpy_helper.from_array(
            W_g.astype(np.float32),
            name=f"{prefix}_chunk{g}.weight_cycle18b17",
        )
        chunk_weights.append(w_init)
        new_inits.append(w_init)

    # Bias broadcast initializer [1, 256, 1].
    b_broadcast = numpy_helper.from_array(
        b.reshape(1, OUT_CHANNELS, 1).astype(np.float32),
        name=f"{prefix}_bias_broadcast_cycle18b17",
    )
    new_inits.append(b_broadcast)

    # 4 Slice nodes + 4 Conv nodes (no bias on each chunk Conv).
    chunk_conv_outs: List[str] = []
    for g in range(NUM_CHUNKS):
        slice_out = f"{z_input}_chunk{g}_slice_cycle18b17"
        slice_node = helper.make_node(
            "Slice",
            inputs=[
                z_input,
                f"{prefix}_chunk{g}_starts_cycle18b17",
                f"{prefix}_chunk{g}_ends_cycle18b17",
                f"{prefix}_chunk{g}_axes_cycle18b17",
            ],
            outputs=[slice_out],
            name=f"{prefix}_chunk{g}_Slice_cycle18b17",
        )
        new_nodes.append(slice_node)

        conv_out = f"{prefix}_chunk{g}_conv_out_cycle18b17"
        conv_node = helper.make_node(
            "Conv",
            inputs=[slice_out, chunk_weights[g].name],
            outputs=[conv_out],
            name=f"{prefix}_chunk{g}_Conv_cycle18b17",
            kernel_shape=[1],
            strides=[1],
            pads=[0, 0],
            dilations=[1],
            group=1,
        )
        new_nodes.append(conv_node)
        chunk_conv_outs.append(conv_out)

    # Add tree: t01 = c0 + c1, t23 = c2 + c3, t = t01 + t23
    t01_out = f"{prefix}_add_01_cycle18b17"
    t23_out = f"{prefix}_add_23_cycle18b17"
    t_out = f"{prefix}_add_t_cycle18b17"
    new_nodes.append(helper.make_node("Add", inputs=[chunk_conv_outs[0], chunk_conv_outs[1]],
                                       outputs=[t01_out], name=f"{prefix}_Add01_cycle18b17"))
    new_nodes.append(helper.make_node("Add", inputs=[chunk_conv_outs[2], chunk_conv_outs[3]],
                                       outputs=[t23_out], name=f"{prefix}_Add23_cycle18b17"))
    new_nodes.append(helper.make_node("Add", inputs=[t01_out, t23_out],
                                       outputs=[t_out], name=f"{prefix}_AddT_cycle18b17"))

    # Bias broadcast Add to produce y_chunked at the SAME tensor name as the original Conv output.
    new_nodes.append(helper.make_node(
        "Add", inputs=[t_out, b_broadcast.name],
        outputs=[orig_output],  # reuse original Conv output name -> all downstream consumers stay valid.
        name=f"{prefix}_AddBias_cycle18b17",
    ))

    # Remove the original Conv node; insert the new nodes at the same position.
    del model.graph.node[target_idx]
    for offset, nd in enumerate(new_nodes):
        model.graph.node.insert(target_idx + offset, nd)
    model.graph.initializer.extend(new_inits)

    # The original weight/bias initializers become unreferenced (dead). They remain in
    # the file but no node consumes them. onnx.checker tolerates unused initializers.

    return {
        "key": target["key"],
        "conv_node_name": target["conv_node_name"],
        "num_chunks": NUM_CHUNKS,
        "chunk_size_input_channels": CHUNK_SIZE,
        "output_channels": OUT_CHANNELS,
        "W_max_abs_per_chunk": [
            float(np.abs(W[:, g * CHUNK_SIZE : (g + 1) * CHUNK_SIZE, :]).max())
            for g in range(NUM_CHUNKS)
        ],
        "orig_W_max_abs": float(np.abs(W).max()),
        "orig_b_max_abs": float(np.abs(b).max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx"))
    parser.add_argument("--dst", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle18b17_b1_chunkconv.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
    args = parser.parse_args()

    src_md5 = md5sum(args.src)
    assert src_md5 == EXPECTED_SRC_MD5, f"production D3 md5 drift: {src_md5}"
    print(f"[b1] src md5: {src_md5} (production D3 OK)")

    model = onnx.load(str(args.src))
    per_target = [chunk_one(model, t) for t in TARGETS]

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    dst_md5 = md5sum(args.dst)
    print(f"[b1] wrote {args.dst.name} md5={dst_md5}")
    for pt in per_target:
        print(f"[b1] {pt['key']}: {pt['num_chunks']} chunks of {pt['chunk_size_input_channels']} input channels each, "
              f"W chunks max_abs={pt['W_max_abs_per_chunk']}")

    src_sidecar = json.loads(args.sidecar_src.read_text())
    side = copy.deepcopy(src_sidecar)
    side["schema_version"] = "s20-bafnetplus-functional-fp32-complex-cycle18b17-b1-chunkconv-v1"
    side["produced_by"] = "BAFNetPlus.scripts.export_bafnetplus_cycle18b17_candidate_b1_chunkconv"
    side["cycle18b17"] = {
        "candidate": "b1_chunkconv",
        "rationale": (
            "Decompose each target proj_first 1x1 Conv (64 -> 256) into 4 sub-Convs on "
            "16 input channels each, summed via an Add tree. HTP-FP16 accumulator span "
            "per sub-Conv is 1/4 the original; the Add tree merges in FP16 with bounded "
            "magnitude per intermediate. The structure cannot fuse back to a single Conv "
            "(distinct Conv nodes with distinct weight initializers), so the intervention "
            "is fold-resistant. FP32 host parity is bounded by reordering noise (~1e-5 to "
            "5e-5)."
        ),
        "num_chunks": NUM_CHUNKS,
        "chunk_size_input_channels": CHUNK_SIZE,
        "targets": per_target,
        "source_production_md5": src_md5,
    }
    side["onnx_file_md5"] = dst_md5
    dst_sidecar = args.dst.with_suffix(".onnx.json")
    dst_sidecar.write_text(json.dumps(side, indent=2))
    print(f"[b1] wrote {dst_sidecar.name} md5={md5sum(dst_sidecar)}")

    assert md5sum(args.src) == EXPECTED_SRC_MD5, "production source mutated!"
    print("[b1] G2 OK -- production D3 byte-identical, candidate B1 chunkconv emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
