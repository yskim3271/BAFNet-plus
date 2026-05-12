"""
BAFNetPlus ONNX Export Script.

Exports the dual-input BAFNetPlus streaming model to ONNX format (FP32 + QDQ INT8)
compatible with Android deployment (StatefulInference.kt, streaming_config.json).

Architecture (single-session, unified graph):

    Host (Android)                           ONNX Model (HTP)
    ─────────────                            ────────────────
    raw audio → STFT → bcs/acs mag/pha ──→  [BAFNetPlusStatefulExportableNNCore]
                       state_mapping_*  ──→  Mapping Backbone (encoder → TSBlocks → decoders)
                       state_masking_*  ──→  Masking Backbone (encoder → TSBlocks → decoders)
                       state_calibr.*   ──→  Calibration Encoder (stateful Causal Conv1d)
                       state_alpha_*    ──→  Alpha Conv Blocks (stateful Causal Conv2d)
                                       ←──   est_mag, est_com_real, est_com_imag (post fusion)
                                       ←──   next_state_* (166 tensors)
    iSTFT ← OLA ← mag + complex ←──────

Key design decisions:
    - Backbones (mapping/masking): encoder/decoder in NON-streaming mode
      (zero-padded), like LaCoSENet. Only TSBlock states are externalized
      (80 per backbone = 160 total).
    - Fusion (calibration + alpha): streaming mode with EXTERNALIZED state
      via inline stateful conv forward (mirrors StatefulCausalConv1d/2d).
      2 calibration states + 4 alpha states = 6 fusion states.
    - Total states: 160 + 6 = 166.
    - Phase handling: atan2 is EXCLUDED from the graph (QNN HTP precision
      concern). The graph emits ``est_mag, est_com_real, est_com_imag``; host
      can compute ``atan2(imag + eps, real + eps)`` to recover est_pha, or
      feed the complex tensor directly to iSTFT. Bypassing atan2 matches the
      LaCoSENet phase-output convention (phase_real/phase_imag on host).
    - No ablation mode support: only `full` (plan §Q1).

Usage:
    # From dual checkpoints (required)
    python -m src.models.streaming.onnx.export_bafnetplus_onnx \\
        --chkpt_dir_mapping results/experiments/bm_map_50ms \\
        --chkpt_dir_masking results/experiments/bm_mask_50ms \\
        --output_dir android/benchmark-app/src/main/assets \\
        --simplify --quantize_qdq \\
        --calibration_fixture_dir android/benchmark-app/src/androidTest/assets/bafnetplus_fixtures
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / naming
# ---------------------------------------------------------------------------

# TSBlock state-key rename map (shared with LaCoSENet export).
STATE_KEY_RENAME = {"sca_dwconv": "ema"}

# State name prefixes — alphabetically ordered so Python sorted() and Kotlin
# List.sort() produce identical ordering across platforms.
PREFIX_ALPHA = "state_alpha_conv"
PREFIX_CALIBRATION = "state_calibration_conv"
PREFIX_MAPPING_TSBLOCK = "state_mapping_rf"
PREFIX_MASKING_TSBLOCK = "state_masking_rf"

# Ablation mode supported by this exporter. Only `full` per plan §Q1.
SUPPORTED_ABLATION = "full"


# ---------------------------------------------------------------------------
# Step 1: State Registry Builder
# ---------------------------------------------------------------------------


def _collect_tsblock_states(
    streaming_tsblocks: nn.ModuleList,
    freq_size: int,
    prefix: str,
) -> List[Tuple[str, Tuple[int, ...], Tuple[str, int, int, str, str]]]:
    """Walk TSBlocks, emit (name, shape, locator) triples.

    locator = (category='tsblock', block_idx, tb_idx, section, raw_key)
    """
    entries: List[Tuple[str, Tuple[int, ...], Tuple[str, int, int, str, str]]] = []
    for blk_idx, block in enumerate(streaming_tsblocks):
        block_states = block.init_state(batch_size=1, freq_size=freq_size)
        for tb_idx, tb_state in enumerate(block_states):
            for section_key, section_dict in [("cab", tb_state["cab"]), ("gpkffn", tb_state["gpkffn"])]:
                for raw_key, tensor in section_dict.items():
                    display_key = STATE_KEY_RENAME.get(raw_key, raw_key)
                    name = f"{prefix}_{blk_idx}_tb{tb_idx}_{section_key}_{display_key}"
                    entries.append((name, tuple(tensor.shape), ("tsblock", blk_idx, tb_idx, section_key, raw_key)))
    return entries


def _collect_calibration_states(
    calibration_encoder: Optional[nn.Module],
    freq_bins_hint: int,  # unused; kept for symmetry
) -> List[Tuple[str, Tuple[int, ...], Tuple[str, int]]]:
    """Walk calibration_encoder Sequential, emit states for each StatefulCausalConv1d.

    Each calibration sub-module is nn.Sequential(StatefulCausalConv1d, PReLU).
    The state shape is [1, C_in, padding_size].
    """
    if calibration_encoder is None:
        return []

    from src.models.streaming.layers.stateful_conv import StatefulCausalConv1d

    entries: List[Tuple[str, Tuple[int, ...], Tuple[str, int]]] = []
    for i, seq in enumerate(calibration_encoder):
        stateful_conv = seq[0]
        if not isinstance(stateful_conv, StatefulCausalConv1d):
            raise TypeError(
                f"calibration_encoder[{i}][0] is {type(stateful_conv).__name__}, "
                "expected StatefulCausalConv1d (did you forget convert_to_stateful?)"
            )
        c_in = stateful_conv.conv.in_channels
        shape = (1, c_in, stateful_conv.padding_size)
        name = f"{PREFIX_CALIBRATION}_{i}"
        entries.append((name, shape, ("calibration", i)))
    return entries


def _collect_alpha_states(
    alpha_convblocks: nn.ModuleList,
    freq_bins: int,
) -> List[Tuple[str, Tuple[int, ...], Tuple[str, int]]]:
    """Walk alpha_convblocks, emit states for each StatefulCausalConv2d.

    Each alpha block is nn.Sequential(StatefulCausalConv2d, BN, PReLU).
    The state shape is [1, C_in, time_padding, F_padded] where F_padded is
    `freq_bins + 2 * freq_padding` (symmetric freq padding applied before state).
    """
    from src.models.streaming.layers.stateful_conv import StatefulCausalConv2d

    entries: List[Tuple[str, Tuple[int, ...], Tuple[str, int]]] = []
    for i, seq in enumerate(alpha_convblocks):
        stateful_conv = seq[0]
        if not isinstance(stateful_conv, StatefulCausalConv2d):
            raise TypeError(
                f"alpha_convblocks[{i}][0] is {type(stateful_conv).__name__}, "
                "expected StatefulCausalConv2d (did you forget convert_to_stateful?)"
            )
        c_in = stateful_conv.conv.in_channels
        f_padded = freq_bins + 2 * stateful_conv.freq_padding
        shape = (1, c_in, stateful_conv.time_padding, f_padded)
        name = f"{PREFIX_ALPHA}_{i}"
        entries.append((name, shape, ("alpha", i)))
    return entries


class BafnetPlusStateRegistry:
    """Holds (name, shape, locator) entries for every externalized state,
    pre-sorted alphabetically by name.

    Category locators:
        ("tsblock", branch, block_idx, tb_idx, section, raw_key)
          branch ∈ {"mapping", "masking"}
        ("calibration", conv_idx)
        ("alpha", conv_idx)
    """

    def __init__(self, entries: List[Tuple[str, Tuple[int, ...], tuple]]):
        entries_sorted = sorted(entries, key=lambda e: e[0])
        self.entries = entries_sorted

    @property
    def names(self) -> List[str]:
        return [e[0] for e in self.entries]

    @property
    def shapes(self) -> List[Tuple[int, ...]]:
        return [e[1] for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def as_registry_dict(self) -> "OrderedDict[str, Tuple[int, ...]]":
        d: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict()
        for name, shape, _ in self.entries:
            d[name] = shape
        return d


def build_bafnetplus_state_registry(
    bafnetplus: nn.Module,
    freq_size: int = 100,
    freq_bins: int = 201,
    export_time_frames: int = 11,
):
    """Forward-ordered state registry built by walking the converted bafnetplus.

    ``bafnetplus`` must have its StatefulCausalConv* already replaced by
    FunctionalStateful* (via :func:`convert_module_inplace`). Walks the four
    sub-modules of each backbone (dense_encoder, sequence_block, mask_decoder,
    phase_decoder) plus fusion (calibration_encoder, alpha_convblocks), in the
    same order that :func:`forward_backbone` invokes them.
    """
    from src.models.streaming.onnx.functional_core import (
        FunctionalStateRegistry,
        build_functional_state_entries,
    )
    entries = build_functional_state_entries(
        bafnetplus,
        freq_bins=freq_bins,
        freq_size_encoded=freq_size,
        export_time_frames=export_time_frames,
    )
    return FunctionalStateRegistry(entries)


# ---------------------------------------------------------------------------
# Step 2: BAFNetPlusStatefulExportableNNCore
# ---------------------------------------------------------------------------


class BAFNetPlusStatefulExportableNNCore(nn.Module):
    """BAFNet+ ONNX-exportable wrapper using FunctionalStateful + explicit state I/O.

    Routes state through every FunctionalStateful conv in forward order, mirroring
    the PyTorch streaming forward exactly (corr 1.0000 vs nonstream at FP32).

    Forward signature:
        (bcs_mag, bcs_pha, acs_mag, acs_pha, *flat_states)
            → (est_mag, est_com_real, est_com_imag, *next_flat_states)

    Pre-conditions:
        - ``bafnetplus`` has every StatefulCausalConv1d / StatefulCausalConv2d /
          StatefulAsymmetricConv2d replaced by FunctionalStateful* counterparts
          (see :func:`convert_module_inplace`).
        - ``state_registry`` (FunctionalStateRegistry) lists the states in the
          exact forward order used by :func:`forward_backbone` + fusion below.
    """

    def __init__(
        self,
        bafnetplus: nn.Module,
        freq_size: int = 100,
        chunk_size: int = 8,
        state_registry=None,
    ):
        super().__init__()
        self.bafnetplus = bafnetplus
        self.freq_size = freq_size
        self.chunk_size = chunk_size
        self.state_registry = state_registry
        self._state_names = list(state_registry.names) if state_registry is not None else []

    @property
    def state_names(self) -> List[str]:
        return list(self._state_names)

    # ------------------------------------------------------------------
    # Fusion helpers (calibration + alpha) with explicit state routing
    # ------------------------------------------------------------------
    def _calibration(self, bcs_est_mag, acs_est_mag, bcs_com_out, acs_com_out, acs_mask, state_iter):
        bf = self.bafnetplus
        eps = 1e-8
        bcs_log_E = torch.log(bcs_est_mag.pow(2).mean(dim=1, keepdim=True) + eps)
        acs_log_E = torch.log(acs_est_mag.pow(2).mean(dim=1, keepdim=True) + eps)
        log_E_diff = bcs_log_E - acs_log_E
        mask_mean = acs_mask.mean(dim=1, keepdim=True)
        mask_var = acs_mask.var(dim=1, keepdim=True, unbiased=False)
        cal_feat = torch.cat([bcs_log_E, acs_log_E, log_E_diff, mask_mean, mask_var], dim=1)

        x = cal_feat
        for seq in bf.calibration_encoder:
            conv = seq[0]
            prelu = seq[1]
            x, ns = conv(x, state_iter.take(), state_frames=self.chunk_size)
            state_iter.push(ns)
            x = prelu(x)
        calibration_hidden = x

        common_log_gain = torch.tanh(bf.common_gain_head(calibration_hidden))
        common_log_gain = common_log_gain * bf.calibration_max_common_log_gain
        if getattr(bf, "use_relative_gain", True):
            relative_log_gain = torch.tanh(bf.relative_gain_head(calibration_hidden))
            relative_log_gain = relative_log_gain * bf.calibration_max_relative_log_gain
            bcs_gain = torch.exp(common_log_gain - 0.5 * relative_log_gain)
            acs_gain = torch.exp(common_log_gain + 0.5 * relative_log_gain)
        else:
            bcs_gain = acs_gain = torch.exp(common_log_gain)
        bcs_gain_b = bcs_gain.transpose(1, 2).unsqueeze(1)
        acs_gain_b = acs_gain.transpose(1, 2).unsqueeze(1)
        bcs_com_cal = bcs_com_out * bcs_gain_b
        acs_com_cal = acs_com_out * acs_gain_b
        return bcs_com_cal, acs_com_cal

    def _alpha_fusion(self, bcs_com_cal, acs_com_cal, acs_mask, state_iter):
        bf = self.bafnetplus
        eps = 1e-8
        bcs_mag_cal = torch.sqrt(bcs_com_cal[..., 0] ** 2 + bcs_com_cal[..., 1] ** 2 + eps)
        acs_mag_cal = torch.sqrt(acs_com_cal[..., 0] ** 2 + acs_com_cal[..., 1] ** 2 + eps)
        alpha_feat = torch.stack([bcs_mag_cal, acs_mag_cal, acs_mask], dim=1).transpose(2, 3)

        x = alpha_feat
        for seq in bf.alpha_convblocks:
            conv = seq[0]
            bn = seq[1]
            prelu = seq[2]
            x, ns = conv(x, state_iter.take(), state_frames=self.chunk_size)
            state_iter.push(ns)
            x = prelu(bn(x))

        alpha = bf.alpha_out(x).transpose(2, 3)
        alpha = torch.softmax(alpha, dim=1)
        alpha_bcs = alpha[:, 0].unsqueeze(-1)
        alpha_acs = alpha[:, 1].unsqueeze(-1)
        est_com = bcs_com_cal * alpha_bcs + acs_com_cal * alpha_acs
        com_real, com_imag = est_com[..., 0], est_com[..., 1]
        est_mag = torch.sqrt(com_real ** 2 + com_imag ** 2 + eps)
        return est_mag, com_real, com_imag

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        bcs_mag: Tensor,
        bcs_pha: Tensor,
        acs_mag: Tensor,
        acs_pha: Tensor,
        *flat_states: Tensor,
    ):
        from src.models.streaming.onnx.functional_core import StateIterator, forward_backbone

        state_iter = StateIterator(list(flat_states))

        # Mapping branch — full layer-by-layer state routing (encoder + tsblocks + decoders)
        bcs_est_mag, _bcs_mask_unused, bcs_com_out = forward_backbone(
            self.bafnetplus.mapping, bcs_mag, bcs_pha, state_iter,
            chunk_size=self.chunk_size, infer_type="mapping",
        )
        # Masking branch — same routing pattern, distinct weights
        acs_est_mag, acs_mask, acs_com_out = forward_backbone(
            self.bafnetplus.masking, acs_mag, acs_pha, state_iter,
            chunk_size=self.chunk_size, infer_type="masking",
        )

        # Calibration (state-routed FunctionalStatefulConv1d)
        if getattr(self.bafnetplus, "use_calibration", True):
            bcs_com_cal, acs_com_cal = self._calibration(
                bcs_est_mag, acs_est_mag, bcs_com_out, acs_com_out, acs_mask, state_iter,
            )
        else:
            bcs_com_cal, acs_com_cal = bcs_com_out, acs_com_out

        # Alpha fusion (state-routed FunctionalStatefulCausalConv2d)
        est_mag, est_com_real, est_com_imag = self._alpha_fusion(
            bcs_com_cal, acs_com_cal, acs_mask, state_iter,
        )

        # Sanity: every registry slot must have been consumed exactly once.
        if state_iter.consumed != len(self._state_names):
            raise RuntimeError(
                f"State count mismatch: consumed {state_iter.consumed} but registry has "
                f"{len(self._state_names)} entries — check registry / forward order alignment."
            )

        return (est_mag, est_com_real, est_com_imag, *state_iter.next_states)


# ---------------------------------------------------------------------------
# Step 3: Model Preparation
# ---------------------------------------------------------------------------


def prepare_bafnetplus_from_checkpoints(
    chkpt_dir_mapping: str,
    chkpt_dir_masking: str,
    chkpt_file: str = "best.th",
    device: str = "cpu",
    ablation_mode: str = "full",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[nn.Module, nn.ModuleList, nn.ModuleList, Dict[str, Any]]:
    """Build BAFNetPlus + streaming TSBlocks + fusion stateful convs.

    Reuses :meth:`BAFNetPlusStreaming.from_checkpoint` to guarantee the Kaiming
    fusion-weight RNG state matches the Stage 2 fixture generator
    (scripts/make_bafnetplus_streaming_golden.py), then switches encoder/decoder
    back to non-streaming mode for ONNX export (LaCoSENet convention).

    The `seed` controls the Kaiming init of BAFNetPlus fusion weights
    (alpha_convblocks, calibration_encoder, alpha_out, *_gain_head). It must
    match Stage 2 fixture generator (default 42) for Stage 2 fixture parity.

    Returns (bafnetplus, model_info).
    """
    from omegaconf import OmegaConf

    from src.models.streaming.bafnetplus_streaming import BAFNetPlusStreaming

    if ablation_mode != SUPPORTED_ABLATION:
        raise NotImplementedError(
            f"Stage 3 only supports ablation_mode='{SUPPORTED_ABLATION}', got '{ablation_mode}'",
        )

    # Seed BEFORE BAFNetPlusStreaming.from_checkpoint so Kaiming init RNG state
    # matches scripts/make_bafnetplus_streaming_golden.py line 359.
    torch.manual_seed(seed)

    # 1. Build via streaming wrapper's factory (identical code path to Stage 2
    #    fixture generator — ensures byte-equal fusion weights under the same
    #    seed). Default chunk/lookahead matches Stage 2.
    streaming = BAFNetPlusStreaming.from_checkpoint(
        chkpt_dir_mapping=chkpt_dir_mapping,
        chkpt_dir_masking=chkpt_dir_masking,
        chkpt_file=chkpt_file,
        chunk_size=8,
        encoder_lookahead=3,
        decoder_lookahead=3,
        ablation_mode=ablation_mode,
        device=device,
        verbose=verbose,
    )

    bafnetplus = streaming.model

    # Convert all StatefulCausalConv* modules to FunctionalStateful* with
    # explicit state I/O. This makes wrapper.forward layer-by-layer state-routed
    # (mirroring BAFNetPlusStreaming.process_samples), so the ONNX graph captures
    # the same streaming forward as the PyTorch reference.
    from src.models.streaming.onnx.functional_stateful import convert_module_inplace
    n_converted = convert_module_inplace(bafnetplus)
    if verbose:
        print(f"  Converted {n_converted} StatefulCausalConv* → FunctionalStateful*")
    bafnetplus.eval()

    # 3. Read config params for streaming_config.json
    def _load_conf(d: str):
        return OmegaConf.load(os.path.join(d, ".hydra", "config.yaml"))

    conf_map = _load_conf(chkpt_dir_mapping)
    p_map = conf_map.model.param
    conf_mask = _load_conf(chkpt_dir_masking)
    p_mask = conf_mask.model.param

    model_info = {
        "chkpt_dir_mapping": chkpt_dir_mapping,
        "chkpt_dir_masking": chkpt_dir_masking,
        "chkpt_file": chkpt_file,
        "ablation_mode": ablation_mode,
        "n_fft": int(p_map.n_fft),
        "hop_size": int(p_map.hop_size),
        "win_size": int(p_map.win_size),
        "compress_factor": float(p_map.compress_factor),
        "dense_channel": int(p_map.dense_channel),
        "num_tsblock": int(p_map.num_tsblock),
        "encoder_padding_ratio": list(p_map.encoder_padding_ratio),
        "decoder_padding_ratio": list(p_map.decoder_padding_ratio),
        "sample_rate": int(getattr(conf_map, "sampling_rate", 16000)),
        "freq_size": int(streaming.freq_size),
        "infer_type_mapping": str(p_map.infer_type),
        "infer_type_masking": str(p_mask.infer_type),
    }
    return bafnetplus, model_info


# ---------------------------------------------------------------------------
# Step 4: ONNX Export / Simplify
# ---------------------------------------------------------------------------


def export_bafnetplus_onnx(
    wrapper: BAFNetPlusStatefulExportableNNCore,
    output_path: str,
    freq_bins: int,
    export_time_frames: int,
    verbose: bool = True,
) -> str:
    """torch.onnx.export wrapper.

    Inputs: bcs_mag/pha, acs_mag/pha, *state_* tensors.
    Outputs: est_mag, est_pha, *next_state_* tensors.
    """
    device = next(wrapper.parameters()).device
    state_names = wrapper.state_names
    shapes = wrapper.state_registry.shapes

    # Build dummy inputs
    bcs_mag = torch.randn(1, freq_bins, export_time_frames, device=device)
    bcs_pha = torch.randn(1, freq_bins, export_time_frames, device=device)
    acs_mag = torch.randn(1, freq_bins, export_time_frames, device=device)
    acs_pha = torch.randn(1, freq_bins, export_time_frames, device=device)

    flat_states = [torch.zeros(*shape, device=device) for shape in shapes]
    dummy_input = (bcs_mag, bcs_pha, acs_mag, acs_pha, *flat_states)

    input_names = ["bcs_mag", "bcs_pha", "acs_mag", "acs_pha"] + state_names
    output_names = ["est_mag", "est_com_real", "est_com_imag"] + [f"next_{n}" for n in state_names]

    if verbose:
        print("\nExporting ONNX model:")
        print(f"  Output: {output_path}")
        print(
            f"  Inputs : bcs_mag/pha[1,{freq_bins},{export_time_frames}], "
            f"acs_mag/pha[1,{freq_bins},{export_time_frames}], "
            f"{len(state_names)} states"
        )
        print(f"  Outputs: est_mag, est_com_real, est_com_imag, {len(state_names)} next_states")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")

    return output_path


def simplify_onnx(input_path: str, output_path: Optional[str] = None, verbose: bool = True) -> str:
    """Run onnx-simplifier (BN folding, redundant op removal)."""
    import onnx
    from onnxsim import simplify

    if output_path is None:
        output_path = input_path

    original_size = os.path.getsize(input_path) / (1024 * 1024)
    model = onnx.load(input_path)
    simplified_model, check = simplify(model)
    if not check:
        print("WARNING: onnx-simplifier validation failed, using original model")
        return input_path
    onnx.save(simplified_model, output_path)
    simplified_size = os.path.getsize(output_path) / (1024 * 1024)
    if verbose:
        print("\nONNX simplification:")
        print(f"  Input:  {input_path} ({original_size:.1f} MB)")
        print(f"  Output: {output_path} ({simplified_size:.1f} MB)")
        print(f"  Reduction: {(1 - simplified_size / original_size) * 100:.1f}%")
    return output_path


# ---------------------------------------------------------------------------
# Step 5: Verification (PyTorch wrapper vs ORT) — with optional fixture input
# ---------------------------------------------------------------------------


def _read_bin(path: str, shape: Tuple[int, ...]) -> "np.ndarray":
    import numpy as np

    return np.fromfile(path, dtype=np.float32).reshape(shape)


def _fixture_inputs_for_chunk(
    fixture_dir: str,
    chunk_idx: int,
    time_frames: Optional[int] = None,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Load (bcs_mag, bcs_pha, acs_mag, acs_pha) for one chunk from a fixture.

    The fixture's source ``T`` is inferred from the .bin file size
    (``bytes / 4 / 201``).

    If ``time_frames`` is supplied and smaller than the source T, the last
    dim is sliced — required for D5d's per-tier exports where
    ``export_time_frames`` ∈ {2, 4, 8, 12, 16}.
    """
    base = os.path.join(fixture_dir, f"chunk_{chunk_idx:03d}")
    bcs_mag_path = os.path.join(base, "bcs_mag.bin")
    src_t = os.path.getsize(bcs_mag_path) // 4 // 201
    if src_t * 201 * 4 != os.path.getsize(bcs_mag_path):
        raise ValueError(
            f"fixture chunk {bcs_mag_path}: size {os.path.getsize(bcs_mag_path)} "
            f"not divisible by 4*201; expected float32 [1,201,T] layout"
        )
    shape = (1, 201, src_t)
    bcs_mag = _read_bin(bcs_mag_path, shape)
    bcs_pha = _read_bin(os.path.join(base, "bcs_pha.bin"), shape)
    acs_mag = _read_bin(os.path.join(base, "acs_mag.bin"), shape)
    acs_pha = _read_bin(os.path.join(base, "acs_pha.bin"), shape)
    if time_frames is not None and time_frames != src_t:
        if time_frames > src_t:
            raise ValueError(
                f"time_frames={time_frames} exceeds fixture chunk length {src_t}; "
                f"regenerate fixture with --chunk_frames {time_frames} --lookahead 0 "
                "or pick chunk_size+lookahead within source T."
            )
        bcs_mag = bcs_mag[:, :, :time_frames]
        bcs_pha = bcs_pha[:, :, :time_frames]
        acs_mag = acs_mag[:, :, :time_frames]
        acs_pha = acs_pha[:, :, :time_frames]
    return bcs_mag, bcs_pha, acs_mag, acs_pha


def verify_onnx_multi(
    wrapper: BAFNetPlusStatefulExportableNNCore,
    onnx_path: str,
    freq_bins: int,
    export_time_frames: int,
    fixture_dir: Optional[str] = None,
    num_chunks: int = 3,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """Verify ONNX model: PyTorch wrapper vs ORT CPU EP, multi-chunk state carry-over.

    If fixture_dir is given, use chunk_000 .. chunk_{N-1} fixture bins as the
    mag/pha inputs (real BAFNet-style data). Otherwise use random inputs.

    The PyTorch wrapper and ONNX graph share the same implementation; mismatches
    indicate an ONNX export bug (tracing issue, op lowering, etc.).
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("WARNING: onnxruntime/numpy not available, skipping verification")
        return True

    if verbose:
        src_label = f"fixture={fixture_dir}" if fixture_dir else "random"
        print(f"\nVerifying ONNX model ({num_chunks} chunks, inputs={src_label}):")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    state_names = wrapper.state_names
    shapes = wrapper.state_registry.shapes

    pt_flat_states = [torch.zeros(*shape) for shape in shapes]
    ort_state_dict: Dict[str, "np.ndarray"] = {
        name: np.zeros(shape, dtype=np.float32) for name, shape in zip(state_names, shapes)
    }

    max_error = 0.0
    all_passed = True
    for chunk_idx in range(num_chunks):
        if fixture_dir is not None:
            bcs_mag_np, bcs_pha_np, acs_mag_np, acs_pha_np = _fixture_inputs_for_chunk(
                fixture_dir, chunk_idx,
            )
        else:
            rng = np.random.default_rng(chunk_idx)
            bcs_mag_np = rng.standard_normal((1, freq_bins, export_time_frames), dtype=np.float32)
            bcs_pha_np = rng.standard_normal((1, freq_bins, export_time_frames), dtype=np.float32)
            acs_mag_np = rng.standard_normal((1, freq_bins, export_time_frames), dtype=np.float32)
            acs_pha_np = rng.standard_normal((1, freq_bins, export_time_frames), dtype=np.float32)

        bcs_mag = torch.from_numpy(bcs_mag_np).contiguous()
        bcs_pha = torch.from_numpy(bcs_pha_np).contiguous()
        acs_mag = torch.from_numpy(acs_mag_np).contiguous()
        acs_pha = torch.from_numpy(acs_pha_np).contiguous()

        with torch.no_grad():
            pt_outputs = wrapper(bcs_mag, bcs_pha, acs_mag, acs_pha, *pt_flat_states)
        pt_est_mag = pt_outputs[0]
        pt_est_com_real = pt_outputs[1]
        pt_est_com_imag = pt_outputs[2]
        pt_next_states = pt_outputs[3:]

        ort_inputs = {
            "bcs_mag": bcs_mag_np,
            "bcs_pha": bcs_pha_np,
            "acs_mag": acs_mag_np,
            "acs_pha": acs_pha_np,
        }
        ort_inputs.update(ort_state_dict)
        ort_outputs = sess.run(None, ort_inputs)
        ort_est_mag = ort_outputs[0]
        ort_est_com_real = ort_outputs[1]
        ort_est_com_imag = ort_outputs[2]
        ort_next_states = ort_outputs[3:]

        for out_name, pt_out, ort_out in [
            ("est_mag", pt_est_mag, ort_est_mag),
            ("est_com_real", pt_est_com_real, ort_est_com_real),
            ("est_com_imag", pt_est_com_imag, ort_est_com_imag),
        ]:
            err = float(np.max(np.abs(pt_out.numpy() - ort_out)))
            rms = float(np.sqrt(np.mean((pt_out.numpy() - ort_out) ** 2)))
            max_error = max(max_error, err)
            ok = np.allclose(pt_out.numpy(), ort_out, atol=atol, rtol=rtol)
            if not ok:
                print(f"  FAIL: chunk {chunk_idx}, {out_name}, max_err={err:.2e}, rms={rms:.2e}")
                all_passed = False

        for i, name in enumerate(state_names):
            err = float(np.max(np.abs(pt_next_states[i].numpy() - ort_next_states[i])))
            max_error = max(max_error, err)
            if not np.allclose(pt_next_states[i].numpy(), ort_next_states[i], atol=atol, rtol=rtol):
                print(f"  FAIL: chunk {chunk_idx}, next_{name}, max_err={err:.2e}")
                all_passed = False

        pt_flat_states = list(pt_next_states)
        ort_state_dict = {name: ort_next_states[i] for i, name in enumerate(state_names)}

        if verbose:
            err_mag = float(np.max(np.abs(pt_est_mag.numpy() - ort_est_mag)))
            err_real = float(np.max(np.abs(pt_est_com_real.numpy() - ort_est_com_real)))
            err_imag = float(np.max(np.abs(pt_est_com_imag.numpy() - ort_est_com_imag)))
            print(
                f"  Chunk {chunk_idx}: est_mag max={err_mag:.2e}, "
                f"real max={err_real:.2e}, imag max={err_imag:.2e}, running max={max_error:.2e}"
            )

    status = "PASSED" if all_passed else "FAILED"
    print(f"  Verification {status}: max error = {max_error:.2e}")
    return all_passed


def verify_against_fixture(
    onnx_path: str,
    fixture_dir: str,
    state_names: List[str],
    state_shapes: List[Tuple[int, ...]],
    num_chunks: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Diagnostic-only: ORT on fixture inputs vs Stage 2 fixture `est_mag`.

    **This is informational only, not a parity gate.** The Stage 2 fixture's
    `chunk_000` is actually the SECOND model forward (post pre-warm) — the first
    forward's outputs are discarded by the fixture generator. Therefore ORT
    starting from zero state cannot match fixture `chunk_000` est_mag/est_pha;
    drift is the expected behavior. The authoritative parity check is
    ``verify_onnx_multi`` (PyTorch wrapper vs ORT, same initial state).

    Returns per-chunk drift magnitudes for the changelog / post-mortem.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_state = {name: np.zeros(shape, dtype=np.float32) for name, shape in zip(state_names, state_shapes)}

    drift_per_chunk: List[Dict[str, float]] = []
    if verbose:
        print("\nDiagnostic: ORT vs Stage 2 fixture est_mag (expected drift — pre-warm offset):")
    for i in range(num_chunks):
        bcs_mag, bcs_pha, acs_mag, acs_pha = _fixture_inputs_for_chunk(fixture_dir, i)
        ort_inputs = {"bcs_mag": bcs_mag, "bcs_pha": bcs_pha, "acs_mag": acs_mag, "acs_pha": acs_pha}
        ort_inputs.update(ort_state)
        outputs = sess.run(None, ort_inputs)
        ort_est_mag = outputs[0]
        ort_next_states = outputs[3:]

        fx_est_mag = _read_bin(os.path.join(fixture_dir, f"chunk_{i:03d}", "est_mag.bin"), (1, 201, 8))
        mag_max = float(np.max(np.abs(fx_est_mag - ort_est_mag)))
        mag_rms = float(np.sqrt(np.mean((fx_est_mag - ort_est_mag) ** 2)))
        drift_per_chunk.append({"chunk": i, "est_mag_max": mag_max, "est_mag_rms": mag_rms})
        if verbose:
            print(f"  chunk {i}: est_mag drift max={mag_max:.2e} rms={mag_rms:.2e}")
        ort_state = {name: ort_next_states[idx] for idx, name in enumerate(state_names)}

    return {"chunks": drift_per_chunk}


# ---------------------------------------------------------------------------
# Step 6: QDQ Quantization
# ---------------------------------------------------------------------------


def quantize_bafnetplus_qdq_for_htp(
    input_path: str,
    output_path: str,
    calibration_fixture_dir: Optional[str] = None,
    num_calibration_samples: int = 40,
    activation_type: str = "QUInt8",
    weight_type: str = "QUInt8",
    verbose: bool = True,
) -> str:
    """Apply QDQ static quantization optimized for QNN HTP.

    Similar to LaCoSENet's quantize_onnx_qdq_for_htp but uses BAFNetPlus fixture
    chunks (bcs_mag/pha + acs_mag/pha) as calibration data when available.
    """
    import numpy as np
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import (
        get_qnn_qdq_config,
        qnn_preprocess_model,
    )

    act_qtype = getattr(QuantType, activation_type)
    wt_qtype = getattr(QuantType, weight_type)

    if verbose:
        print("\nQDQ static quantization for HTP:")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Activation type: {activation_type}, weight type: {weight_type}")

    # Preprocess
    preprocessed_path = input_path.replace(".onnx", "_preproc.onnx")
    model_changed = qnn_preprocess_model(model_input=input_path, model_output=preprocessed_path)
    quant_input = preprocessed_path if model_changed else input_path
    if verbose and model_changed:
        print(f"  Preprocessed model: {preprocessed_path}")

    # Collect input shapes
    model = onnx.load(quant_input)
    input_shapes: "OrderedDict[str, Tuple[int, ...]]" = OrderedDict()
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else 1)
        input_shapes[inp.name] = tuple(shape)

    class BafnetPlusCalibrationReader(CalibrationDataReader):
        """Sequential-streaming calibration reader (H-2-1 fix).

        Previously each calibration sample initialised all 190 state inputs
        to zeros, regardless of position in the chunk stream. In real
        streaming inference the state buffers accumulate across chunks,
        so fusion-path activations (calibration_encoder, alpha_convblocks)
        see distributions the quantizer never calibrated for → INT8 range
        mismatch → RMS-blowup catastrophe (observed corr 0.07 / RMS 6.4x
        vs FP32 ONNX).

        Fix: maintain a persistent state_dict, run an FP32 ORT session
        on the *pre-quantization* graph for each calibration chunk to
        produce the next_state tensors, and feed those as the inputs to
        the subsequent calibration chunk. Optionally skip the first
        ``state_warmup_chunks`` chunks (states still near zero — not
        representative of steady state).
        """

        def __init__(self, fixture_dir, input_shapes, num_samples,
                     fp32_onnx_path: Optional[str] = None,
                     state_warmup_chunks: int = 4):
            self.samples: List[Dict[str, "np.ndarray"]] = []
            self.current = 0
            self.state_warmup_chunks = state_warmup_chunks

            state_names = [n for n in input_shapes if n.startswith("state_")]
            # Build initial state_dict (all zeros — first chunk's input)
            state_dict: Dict[str, "np.ndarray"] = {
                n: np.zeros(input_shapes[n], dtype=np.float32) for n in state_names
            }
            graph_time_frames = int(input_shapes["bcs_mag"][2])

            # Source of (bcs_mag, bcs_pha, acs_mag, acs_pha) per chunk
            if fixture_dir is not None and os.path.isdir(fixture_dir):
                manifest_path = os.path.join(fixture_dir, "manifest.json")
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                n_chunks = min(num_samples + state_warmup_chunks, len(manifest.get("chunks", [])))
                if verbose:
                    print(
                        f"  Calibration: sequential streaming over {n_chunks} fixture chunks "
                        f"(first {state_warmup_chunks} skipped for warmup) from {fixture_dir}"
                    )

                def get_chunk_inputs(chunk_idx):
                    return _fixture_inputs_for_chunk(fixture_dir, chunk_idx, time_frames=graph_time_frames)
            else:
                n_chunks = num_samples + state_warmup_chunks
                rng = np.random.default_rng(0)
                if verbose:
                    print(
                        f"  Calibration: sequential streaming over {n_chunks} random chunks "
                        f"(first {state_warmup_chunks} skipped for warmup)"
                    )

                def get_chunk_inputs(chunk_idx):
                    mag_shape = input_shapes["bcs_mag"]
                    pha_shape = input_shapes["bcs_pha"]
                    bcs_mag = np.abs(rng.standard_normal(mag_shape, dtype=np.float32)) * 0.5
                    bcs_pha = (rng.random(pha_shape, dtype=np.float32) * 2 - 1) * np.pi
                    acs_mag = np.abs(rng.standard_normal(mag_shape, dtype=np.float32)) * 0.5
                    acs_pha = (rng.random(pha_shape, dtype=np.float32) * 2 - 1) * np.pi
                    return bcs_mag, bcs_pha, acs_mag, acs_pha

            # FP32 ORT session for producing next_state tensors
            sess = None
            if fp32_onnx_path is not None and os.path.exists(fp32_onnx_path):
                import onnxruntime as ort
                sess = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])
                # Map output names to find next_state_* outputs
                output_names = [o.name for o in sess.get_outputs()]
            else:
                output_names = []

            for chunk_idx in range(n_chunks):
                bcs_mag, bcs_pha, acs_mag, acs_pha = get_chunk_inputs(chunk_idx)
                sample = {
                    "bcs_mag": bcs_mag.astype(np.float32),
                    "bcs_pha": bcs_pha.astype(np.float32),
                    "acs_mag": acs_mag.astype(np.float32),
                    "acs_pha": acs_pha.astype(np.float32),
                    **state_dict,
                }
                # Append (skip warmup chunks where state is still near zero)
                if chunk_idx >= state_warmup_chunks:
                    self.samples.append(sample)

                # Advance state via FP32 forward (if available)
                if sess is not None and chunk_idx < n_chunks - 1:
                    ort_outs = sess.run(None, sample)
                    new_state = {}
                    for out_name, out_val in zip(output_names, ort_outs):
                        if out_name.startswith("next_state_"):
                            in_name = out_name[len("next_"):]  # strip "next_" prefix
                            if in_name in state_dict:
                                new_state[in_name] = out_val.astype(np.float32)
                    if new_state:
                        state_dict = new_state
                    # else: no next_state outputs matched → keep zeros (degenerate)

            if verbose:
                print(f"  Calibration: collected {len(self.samples)} samples (post-warmup)")

        def get_next(self):
            if self.current >= len(self.samples):
                return None
            s = self.samples[self.current]
            self.current += 1
            return s

        def rewind(self):
            self.current = 0

    # NOTE on calibration strategy:
    #   Tried sequential-streaming calibration (fp32_onnx_path=quant_input,
    #   state_warmup_chunks=4) with both random and real-recapp fixtures.
    #   Both REGRESSED — RMS-ratio 19x→123x vs FP32 (worse than zero-state
    #   random which gave 3.35x). Hypothesis: the new wrapper exposes 24
    #   extra encoder/decoder state tensors (190 total vs prior 166), and
    #   sequential rollout pushes those into wider distributions than the
    #   QDQ minmax calibrator handles well. Real-audio fixture exposes
    #   low-freq voiced peaks that drive INT8 scale wider than typical
    #   values, hurting precision for the bulk distribution.
    #   Reverting to zero-state + random which matches the prior baseline
    #   (~3.35x RMS ratio).
    calib_reader = BafnetPlusCalibrationReader(
        calibration_fixture_dir, input_shapes, num_calibration_samples,
        fp32_onnx_path=None,
        state_warmup_chunks=0,
    )

    qnn_config = get_qnn_qdq_config(
        model_input=quant_input,
        calibration_data_reader=calib_reader,
        activation_type=act_qtype,
        weight_type=wt_qtype,
    )
    quantize(model_input=quant_input, model_output=output_path, quant_config=qnn_config)

    if model_changed and os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    if verbose:
        print(f"  Input size:  {input_size:.1f} MB")
        print(f"  Output size: {output_size:.1f} MB")
        print(f"  Size change: {(output_size / input_size - 1) * 100:+.1f}%")
    return output_path


# ---------------------------------------------------------------------------
# Step 7: streaming_config.json
# ---------------------------------------------------------------------------


def _get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _file_md5(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_bafnetplus_streaming_config(
    model_info: Dict[str, Any],
    state_registry: BafnetPlusStateRegistry,
    chunk_size: int,
    encoder_lookahead: int,
    decoder_lookahead: int,
    export_time_frames: int,
    output_path: str,
    quantization: str = "float32",
    fp32_onnx_path: Optional[str] = None,
    qdq_onnx_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Generate bafnetplus_streaming_config.json.

    Schema extends LaCoSENet's schema with:
      - model_info.name = "BAFNetPlus"
      - inputs: ["bcs_mag", "bcs_pha", "acs_mag", "acs_pha", ...state_names]
      - outputs: ["est_mag", "est_pha", ...next_state_names]
      - fusion_info: {use_calibration, use_relative_gain, alpha_input_channels, ...}
      - state_layout: per-state shape/bytes
    """
    n_fft = model_info["n_fft"]
    freq_bins = n_fft // 2 + 1
    state_names = state_registry.names
    state_shapes = state_registry.shapes

    state_layout = []
    total_state_bytes = 0
    for name, shape in zip(state_names, state_shapes):
        nbytes = 4
        for d in shape:
            nbytes *= d
        state_layout.append({"name": name, "shape": list(shape), "dtype": "float32", "bytes": nbytes})
        total_state_bytes += nbytes

    input_names = ["bcs_mag", "bcs_pha", "acs_mag", "acs_pha"] + state_names
    output_names = ["est_mag", "est_com_real", "est_com_imag"] + [f"next_{n}" for n in state_names]

    config = {
        "model_info": {
            "name": "BAFNetPlus",
            "version": "1.0.0",
            "export_format": "bafnetplus_stateful_nncore",
            "quantization": quantization,
            "phase_output_mode": "complex",
            "qnn_compatible": True,
            "supported_backends": ["qnn_htp", "nnapi", "cpu"],
            "model_type": "BAFNetPlus",
            "ablation_mode": model_info.get("ablation_mode", "full"),
            "infer_type_mapping": model_info.get("infer_type_mapping", "mapping"),
            "infer_type_masking": model_info.get("infer_type_masking", "masking"),
        },
        "stft_config": {
            "n_fft": n_fft,
            "hop_size": model_info["hop_size"],
            "win_length": model_info["win_size"],
            "sample_rate": model_info.get("sample_rate", 16000),
            "center": True,
            "compress_factor": model_info["compress_factor"],
        },
        "streaming_config": {
            "chunk_size_frames": chunk_size,
            "encoder_lookahead": encoder_lookahead,
            "decoder_lookahead": decoder_lookahead,
            "export_time_frames": export_time_frames,
            "freq_bins": freq_bins,
            "freq_bins_encoded": model_info["freq_size"],
        },
        "qnn_config": {
            "target_soc": "SM8550",
            "htp_performance_mode": "burst",
            "context_cache_enabled": True,
            "vtcm_mb": 8,
            "enable_htp_fp16_precision": False,
        },
        "fusion_info": {
            "use_calibration": bool(model_info.get("use_calibration", True)),
            "use_relative_gain": bool(model_info.get("use_relative_gain", True)),
            "alpha_input_channels": int(model_info.get("alpha_input_channels", 3)),
            "calibration_max_common_log_gain": float(
                model_info.get("calibration_max_common_log_gain", 0.5),
            ),
            "calibration_max_relative_log_gain": float(
                model_info.get("calibration_max_relative_log_gain", 1.0),
            ),
        },
        "io_info": {
            "input_names": input_names,
            "output_names": output_names,
        },
        "state_info": {
            "num_states": len(state_names),
            "state_names": state_names,
            "total_state_bytes": total_state_bytes,
            "state_layout": state_layout,
        },
        "export_info": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "checkpoint_md5_mapping": _file_md5(
                os.path.join(
                    model_info.get("chkpt_dir_mapping", ""),
                    model_info.get("chkpt_file", "best.th"),
                ),
            ),
            "checkpoint_md5_masking": _file_md5(
                os.path.join(
                    model_info.get("chkpt_dir_masking", ""),
                    model_info.get("chkpt_file", "best.th"),
                ),
            ),
            "onnx_fp32_md5": _file_md5(fp32_onnx_path) if fp32_onnx_path else None,
            "onnx_qdq_md5": _file_md5(qdq_onnx_path) if qdq_onnx_path else None,
            "git_commit": _get_git_commit(),
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    if verbose:
        print(f"\nGenerated streaming_config.json:")
        print(f"  Output: {output_path}")
        print(f"  States: {len(state_names)} ({total_state_bytes / 1024:.1f} KB total)")
    return output_path


# ---------------------------------------------------------------------------
# Step 8: CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export BAFNetPlus streaming model to ONNX (FP32 + QDQ INT8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1],
    )

    parser.add_argument("--chkpt_dir_mapping", required=True, type=str)
    parser.add_argument("--chkpt_dir_masking", required=True, type=str)
    parser.add_argument(
        "--chkpt_dir_unified",
        type=str,
        default=None,
        help="Optional unified BAFNet+ checkpoint dir (e.g., results/experiments/bafnetplus_50ms). "
             "If provided, fusion weights (alpha_convblocks, alpha_out, calibration_encoder, "
             "common_gain_head, relative_gain_head) are loaded from this checkpoint's `best.th` "
             "AFTER backbone init + Kaiming-random fusion init, overriding the random fusion "
             "with the trained end-to-end weights. REQUIRED for deployment-quality ONNX; "
             "without this flag the fusion stage is Kaiming-random (intended only for Stage 2 "
             "architectural fixture parity, NOT inference quality).",
    )
    parser.add_argument("--chkpt_file", type=str, default="best.th")
    parser.add_argument("--ablation_mode", type=str, default="full")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Kaiming init of BAFNetPlus fusion weights "
             "(must match Stage 2 fixture generator)",
    )

    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--encoder_lookahead", type=int, default=3)
    parser.add_argument("--decoder_lookahead", type=int, default=3)

    parser.add_argument("--output_dir", type=str, default="exports")
    parser.add_argument("--output_name", type=str, default="bafnetplus.onnx")
    parser.add_argument("--config_name", type=str, default="bafnetplus_streaming_config.json")

    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--quantize_qdq", action="store_true")
    parser.add_argument(
        "--calibration_fixture_dir",
        type=str,
        default=None,
        help="Path to Stage 2 fixture directory (used for QDQ calibration and fixture verify)",
    )
    parser.add_argument("--qdq_activation_type", type=str, default="QUInt8", choices=["QUInt8", "QUInt16"])
    parser.add_argument("--qdq_weight_type", type=str, default="QUInt8")

    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--verify_chunks", type=int, default=3)

    args = parser.parse_args()

    print("=" * 64)
    print("BAFNetPlus ONNX Export")
    print("=" * 64)

    device = "cpu"

    # --- Prepare model ---
    bafnetplus, model_info = prepare_bafnetplus_from_checkpoints(
        chkpt_dir_mapping=args.chkpt_dir_mapping,
        chkpt_dir_masking=args.chkpt_dir_masking,
        chkpt_file=args.chkpt_file,
        device=device,
        ablation_mode=args.ablation_mode,
        seed=args.seed,
    )

    # --- Overlay trained weights from unified checkpoint (2026-05-11 fix v2) ---
    # NOTE: previously this filtered to fusion-only prefixes, which left
    # mapping/masking backbones with the bm_map/bm_mask Stage-1 weights
    # (pre-joint-fine-tune). The unified checkpoint contains the post-Stage-2
    # *jointly* fine-tuned mapping/masking weights, which the wrapper MUST use
    # to be numerically equivalent to BAFNetPlusStreaming.process_samples
    # (which loads the full unified ckpt via streaming.model.load_state_dict).
    # Fusion-only filter was the root cause of wrapper-vs-nonstream corr 0.04
    # and FP32 ONNX vs PT streaming corr 0.77 (see 2026-05-11 cycle).
    if args.chkpt_dir_unified is not None:
        unified_path = os.path.join(args.chkpt_dir_unified, args.chkpt_file)
        print(f"\nLoading FULL unified weights (mapping + masking + fusion) from:\n  {unified_path}")
        unified_blob = torch.load(unified_path, map_location=device, weights_only=False)
        unified_sd = unified_blob["model"] if isinstance(unified_blob, dict) and "model" in unified_blob else unified_blob
        missing, unexpected = bafnetplus.load_state_dict(unified_sd, strict=False)
        if unexpected:
            raise RuntimeError(
                f"unified checkpoint has {len(unexpected)} unexpected keys vs wrapper.bafnetplus state_dict "
                f"(model does not declare them): {unexpected[:3]}",
            )
        if missing:
            raise RuntimeError(
                f"unified checkpoint left {len(missing)} keys un-overwritten in wrapper.bafnetplus: "
                f"{missing[:3]}",
            )
        print(f"  Loaded {len(unified_sd)} trained weight tensors (full unified ckpt overlay)")
        model_info["chkpt_dir_unified"] = args.chkpt_dir_unified
        model_info["fusion_source"] = "trained_unified_full"
    else:
        print("\nWARNING: --chkpt_dir_unified not provided — fusion weights are Kaiming-random "
              "(seed={}). This ONNX is for architectural fixture parity only, NOT deployment "
              "quality. Pass --chkpt_dir_unified <unified_ckpt_dir> for trained fusion."
              .format(args.seed))
        model_info["fusion_source"] = "kaiming_random_seed_{}".format(args.seed)

    n_fft = model_info["n_fft"]
    freq_bins = n_fft // 2 + 1
    freq_size = model_info["freq_size"]
    # Fix B (2026-05-11): export sees (chunk + encoder_la + decoder_la) frames per
    # call, matching the redesigned BAFNetPlusStreaming.process_samples single-pass
    # forward. Previously this was `chunk + max(enc_la, dec_la)`, which only worked
    # when one of the lookaheads was zero. With both > 0 (e.g. 50ms variant has
    # enc_la=3, dec_la=3), the wrapper now needs to see all 14 = 8 + 3 + 3 frames
    # so encoder and decoder share the same input window — byte-equivalent to
    # BAFNetPlus.forward via spec-streaming (corr 1.0000, SDR +104.5 dB anchor).
    export_time_frames = args.chunk_size + args.encoder_lookahead + args.decoder_lookahead

    print(f"\nDimensions:")
    print(f"  freq_bins (STFT): {freq_bins}")
    print(f"  freq_size (encoded): {freq_size}")
    print(f"  export_time_frames: {export_time_frames}")
    print(f"  ablation_mode: {args.ablation_mode}")

    # --- State registry ---
    state_registry = build_bafnetplus_state_registry(
        bafnetplus, freq_size=freq_size, freq_bins=freq_bins,
        export_time_frames=export_time_frames,
    )
    print(f"  num_states: {len(state_registry)}")

    # Fusion attribute summary
    model_info["use_calibration"] = bool(getattr(bafnetplus, "use_calibration", True))
    model_info["use_relative_gain"] = bool(getattr(bafnetplus, "use_relative_gain", True))
    model_info["alpha_input_channels"] = 1 if getattr(bafnetplus, "mask_only_alpha", False) else 3
    model_info["calibration_max_common_log_gain"] = float(
        getattr(bafnetplus, "calibration_max_common_log_gain", 0.5),
    )
    model_info["calibration_max_relative_log_gain"] = float(
        getattr(bafnetplus, "calibration_max_relative_log_gain", 1.0),
    )

    # --- Wrapper ---
    wrapper = BAFNetPlusStatefulExportableNNCore(
        bafnetplus=bafnetplus,
        freq_size=freq_size,
        chunk_size=args.chunk_size,
        state_registry=state_registry,
    ).eval()

    onnx_path = os.path.join(args.output_dir, args.output_name)
    config_path = os.path.join(args.output_dir, args.config_name)

    # --- Export FP32 ---
    export_bafnetplus_onnx(
        wrapper=wrapper,
        output_path=onnx_path,
        freq_bins=freq_bins,
        export_time_frames=export_time_frames,
    )

    # --- Simplify ---
    if args.simplify:
        simplify_onnx(input_path=onnx_path)

    # --- Verify FP32 vs wrapper ---
    if not args.skip_verify:
        verify_onnx_multi(
            wrapper=wrapper,
            onnx_path=onnx_path,
            freq_bins=freq_bins,
            export_time_frames=export_time_frames,
            fixture_dir=args.calibration_fixture_dir,
            num_chunks=args.verify_chunks,
            atol=1e-4,
            rtol=1e-4,
        )
        if args.calibration_fixture_dir:
            verify_against_fixture(
                onnx_path=onnx_path,
                fixture_dir=args.calibration_fixture_dir,
                state_names=state_registry.names,
                state_shapes=state_registry.shapes,
                num_chunks=args.verify_chunks,
            )

    # --- QDQ quantization ---
    quantization_label = "float32"
    qdq_path = None
    if args.quantize_qdq:
        qdq_path = onnx_path.replace(".onnx", "_qdq.onnx")
        quantize_bafnetplus_qdq_for_htp(
            input_path=onnx_path,
            output_path=qdq_path,
            calibration_fixture_dir=args.calibration_fixture_dir,
            activation_type=args.qdq_activation_type,
            weight_type=args.qdq_weight_type,
        )
        quantization_label = "qdq_int8"
        # Smoke check QDQ (loose tolerance)
        if not args.skip_verify:
            verify_onnx_multi(
                wrapper=wrapper,
                onnx_path=qdq_path,
                freq_bins=freq_bins,
                export_time_frames=export_time_frames,
                fixture_dir=args.calibration_fixture_dir,
                num_chunks=1,
                atol=0.5,
                rtol=0.5,
            )

    # --- Generate streaming_config ---
    generate_bafnetplus_streaming_config(
        model_info=model_info,
        state_registry=state_registry,
        chunk_size=args.chunk_size,
        encoder_lookahead=args.encoder_lookahead,
        decoder_lookahead=args.decoder_lookahead,
        export_time_frames=export_time_frames,
        output_path=config_path,
        quantization=quantization_label,
        fp32_onnx_path=onnx_path,
        qdq_onnx_path=qdq_path,
    )

    print(f"\nDone. Files:")
    print(f"  {onnx_path}")
    if qdq_path:
        print(f"  {qdq_path}")
    print(f"  {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
