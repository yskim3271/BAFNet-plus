"""
LaCoSENet ONNX Export Script.

Exports the streaming model to ONNX format compatible with Android deployment
(StatefulInference.kt, streaming_config.json).

Architecture:
    Host (Android)                    ONNX Model (HTP)
    ─────────────                    ────────────────
    raw audio → STFT → mag/pha ──→  [StatefulExportableNNCore]
                       state_* ──→  encoder → TSBlocks → decoders
                                ←── est_mask, phase_real/imag
                                ←── next_state_*
    iSTFT ← mask apply ←────────

Key design decisions:
    - Encoder/Decoder: NON-streaming mode (zero-padded). STFT windowing
      provides sufficient context for asymmetric padding.
    - TSBlock states only externalized: 80 state tensors
      (4 blocks × 2 time_blocks × 10 states each)
    - Phase output: "complex" mode (phase_real/imag separate, host does atan2)
      for QNN/HTP compatibility.
    - State naming: state_rf_{block}_{tb}_{section}_{key}, alphabetically sorted.
    - state_frames=chunk_size: passed to TSBlocks so lookahead frames
      don't contaminate streaming state.

Usage:
    # From checkpoint
    python scripts/export_onnx.py --chkpt_dir results/experiments/prk_0131_1 \\
        --chunk_size 32 --encoder_lookahead 7 --decoder_lookahead 7

    # Without checkpoint (testing)
    python scripts/export_onnx.py --no_checkpoint --chunk_size 32 \\
        --encoder_lookahead 7 --decoder_lookahead 7
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
from torch import Tensor

from src.models.backbone import Backbone
from src.models.streaming.converters.conv_converter import (
    convert_to_stateful,
    set_streaming_mode,
)
from src.models.streaming.layers.tsblock import StreamingTSBlock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PARAMS = dict(
    n_fft=400,
    hop_size=100,
    win_size=400,
    compress_factor=0.3,
    dense_channel=64,
    sigmoid_beta=2.0,
    num_tsblock=4,
    dense_depth=4,
    time_dw_kernel_size=3,
    time_block_kernel=[3, 5, 7, 11],
    freq_block_kernel=[3, 11, 23, 31],
    time_block_num=2,
    freq_block_num=2,
    causal_ts_block=True,
    encoder_padding_ratio=(1.0, 0.0),
    decoder_padding_ratio=(1.0, 0.0),
    sca_kernel_size=11,
    infer_type="masking",
)

# Rename map: code key → ONNX state name suffix
# Keeps compatibility with existing Android streaming_config.json convention.
STATE_KEY_RENAME = {
    "sca_dwconv": "ema",
}


# ---------------------------------------------------------------------------
# Step 1: State Registry Builder
# ---------------------------------------------------------------------------


def build_state_registry(
    streaming_tsblocks: nn.ModuleList,
    freq_size: int,
) -> OrderedDict:
    """Build an ordered registry of all TSBlock state tensors.

    Calls StreamingTSBlock.init_state() for each block, flattens the nested
    state structure, and returns an alphabetically sorted OrderedDict mapping
    canonical state names to their shapes.

    Args:
        streaming_tsblocks: Converted streaming TSBlock ModuleList.
        freq_size: Encoded frequency dimension.

    Returns:
        OrderedDict[str, tuple]: name → shape, sorted alphabetically.
    """
    registry: Dict[str, Tuple[int, ...]] = {}

    for blk_idx, block in enumerate(streaming_tsblocks):
        # init_state returns List[Dict[str, Tensor]] — one dict per time_block
        block_states = block.init_state(batch_size=1, freq_size=freq_size)

        for tb_idx, tb_state in enumerate(block_states):
            for section_key, section_dict in [("cab", tb_state["cab"]), ("gpkffn", tb_state["gpkffn"])]:
                for raw_key, tensor in section_dict.items():
                    key = STATE_KEY_RENAME.get(raw_key, raw_key)
                    name = f"state_rf_{blk_idx}_tb{tb_idx}_{section_key}_{key}"
                    registry[name] = tuple(tensor.shape)

    # Sort alphabetically (matches Android convention)
    sorted_registry = OrderedDict(sorted(registry.items()))
    return sorted_registry


def _build_state_map(
    streaming_tsblocks: nn.ModuleList,
    freq_size: int,
) -> List[Tuple[str, int, int, str, str]]:
    """Build a mapping from flat state index to nested structure location.

    Returns a list of (name, block_idx, tb_idx, section, raw_key) sorted
    alphabetically by name. raw_key is the original dict key (before rename).
    """
    entries = []

    for blk_idx, block in enumerate(streaming_tsblocks):
        block_states = block.init_state(batch_size=1, freq_size=freq_size)

        for tb_idx, tb_state in enumerate(block_states):
            for section_key, section_dict in [("cab", tb_state["cab"]), ("gpkffn", tb_state["gpkffn"])]:
                for raw_key in section_dict:
                    display_key = STATE_KEY_RENAME.get(raw_key, raw_key)
                    name = f"state_rf_{blk_idx}_tb{tb_idx}_{section_key}_{display_key}"
                    entries.append((name, blk_idx, tb_idx, section_key, raw_key))

    entries.sort(key=lambda e: e[0])
    return entries


# ---------------------------------------------------------------------------
# Step 2: StatefulExportableNNCore
# ---------------------------------------------------------------------------


class StatefulExportableNNCore(nn.Module):
    """ONNX-exportable wrapper with externalized TSBlock states.

    Forward signature:
        (mag, pha, *flat_states) → (est_mask, phase_real, phase_imag, *next_flat_states)

    Inputs:
        mag: [1, F, T]  — compressed magnitude spectrogram
        pha: [1, F, T]  — phase spectrogram
        flat_states: N tensors, alphabetically sorted by state name

    Outputs:
        est_mask: [1, F, T]  — estimated mask
        phase_real: [1, F, T]  — real part of phase estimate (host does atan2)
        phase_imag: [1, F, T]  — imaginary part of phase estimate
        next_flat_states: N tensors (same order as input states)
    """

    def __init__(
        self,
        model: nn.Module,
        streaming_tsblocks: nn.ModuleList,
        freq_size: int,
        chunk_size: int,
        infer_type: str = "masking",
    ):
        super().__init__()
        self.dense_encoder = model.dense_encoder
        self.mask_decoder = model.mask_decoder

        # Phase decoder sub-modules (skip atan2 for ONNX/QNN compatibility)
        self.phase_dense_block = model.phase_decoder.dense_block
        self.phase_conv = model.phase_decoder.phase_conv
        self.phase_conv_r = model.phase_decoder.phase_conv_r
        self.phase_conv_i = model.phase_decoder.phase_conv_i

        self.streaming_tsblocks = streaming_tsblocks
        self.freq_size = freq_size
        self.chunk_size = chunk_size
        self.infer_type = infer_type

        # Build state map for flatten/unflatten
        self._state_map = _build_state_map(streaming_tsblocks, freq_size)
        self._num_blocks = len(streaming_tsblocks)

    @property
    def state_names(self) -> List[str]:
        """Return alphabetically sorted state names."""
        return [entry[0] for entry in self._state_map]

    def _unflatten_states(
        self, flat_states: Tuple[Tensor, ...],
    ) -> List[List[Dict[str, Tensor]]]:
        """Convert flat state tuple to nested structure for TSBlocks.

        Returns:
            List[List[Dict[str, Tensor]]]: [block][time_block][section][key]
        """
        # Determine structure sizes
        num_blocks = self._num_blocks
        # Get time_block_num from first block
        time_block_num = self.streaming_tsblocks[0].time_block_num

        # Initialize nested structure
        nested: List[List[Dict[str, Dict[str, Tensor]]]] = []
        for _ in range(num_blocks):
            block_states = []
            for _ in range(time_block_num):
                block_states.append({"cab": {}, "gpkffn": {}})
            nested.append(block_states)

        # Fill from flat states using state map
        for i, (_, blk_idx, tb_idx, section, raw_key) in enumerate(self._state_map):
            nested[blk_idx][tb_idx][section][raw_key] = flat_states[i]

        return nested

    def _flatten_states(
        self, nested: List[List[Dict[str, Tensor]]],
    ) -> Tuple[Tensor, ...]:
        """Convert nested state structure to flat tuple, alphabetically sorted."""
        flat = []
        for _, blk_idx, tb_idx, section, raw_key in self._state_map:
            flat.append(nested[blk_idx][tb_idx][section][raw_key])
        return tuple(flat)

    def forward(self, mag: Tensor, pha: Tensor, *flat_states: Tensor):
        """Forward pass with explicit state I/O.

        Args:
            mag: Magnitude [1, F, T]
            pha: Phase [1, F, T]
            *flat_states: Flattened state tensors (alphabetically sorted)

        Returns:
            Tuple of (est_mask, phase_real, phase_imag, *next_flat_states)
        """
        # 1. Stack and permute: [1, 2, T, F]
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)

        # 2. Encoder (non-streaming, zero-padded)
        x = self.dense_encoder(x)  # [1, C, T, F_enc]

        # 3. Streaming TSBlocks with explicit state I/O
        nested_states = self._unflatten_states(flat_states)
        for i, block in enumerate(self.streaming_tsblocks):
            x, new_block_states = block(x, nested_states[i], state_frames=self.chunk_size)
            nested_states[i] = new_block_states
        next_flat_states = self._flatten_states(nested_states)

        # 4. Mask decoder (non-streaming, zero-padded)
        est_mask = self.mask_decoder(x).squeeze(1).transpose(1, 2)  # [1, F, T]

        # 5. Phase decoder — complex mode (skip atan2)
        p = self.phase_dense_block(x)
        p = self.phase_conv(p)
        phase_real = self.phase_conv_r(p).squeeze(1).transpose(1, 2)  # [1, F, T]
        phase_imag = self.phase_conv_i(p).squeeze(1).transpose(1, 2)  # [1, F, T]

        return (est_mask, phase_real, phase_imag, *next_flat_states)


# ---------------------------------------------------------------------------
# Step 3: Model Preparation
# ---------------------------------------------------------------------------


def prepare_from_checkpoint(
    chkpt_dir: str,
    chkpt_file: str = "best.th",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[nn.Module, nn.ModuleList, Dict[str, Any]]:
    """Prepare model from checkpoint for ONNX export.

    Returns:
        Tuple of (model, streaming_tsblocks, model_info)
    """
    from src.models.streaming.utils import prepare_streaming_model

    model, metadata = prepare_streaming_model(
        chkpt_dir=chkpt_dir,
        chkpt_file=chkpt_file,
        use_stateful_conv=True,
        device=device,
        verbose=verbose,
    )

    streaming_tsblocks = metadata["streaming_tsblocks"]
    model_args = metadata["model_args"]

    # Disable streaming mode for encoder/decoder (non-streaming, zero-padded)
    set_streaming_mode(model, False)

    model_info = {
        "n_fft": getattr(model_args, "n_fft", 400),
        "hop_size": getattr(model_args, "hop_size", 100),
        "win_size": getattr(model_args, "win_size", 400),
        "compress_factor": getattr(model_args, "compress_factor", 0.3),
        "dense_channel": getattr(model_args, "dense_channel", 64),
        "num_tsblock": getattr(model_args, "num_tsblock", 4),
        "infer_type": getattr(model_args, "infer_type", "masking"),
        "encoder_padding_ratio": tuple(getattr(model_args, "encoder_padding_ratio", [1.0, 0.0])),
        "decoder_padding_ratio": tuple(getattr(model_args, "decoder_padding_ratio", [1.0, 0.0])),
        "chkpt_dir": chkpt_dir,
        "chkpt_file": chkpt_file,
    }

    return model, streaming_tsblocks, model_info


def prepare_without_checkpoint(
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[nn.Module, nn.ModuleList, Dict[str, Any]]:
    """Prepare model without checkpoint (random weights, for testing).

    Returns:
        Tuple of (model, streaming_tsblocks, model_info)
    """
    if verbose:
        print("Preparing model without checkpoint (random weights)")

    model = Backbone(**MODEL_PARAMS)
    model = model.to(device).eval()

    # Convert TSBlocks → streaming
    streaming_tsblocks = StreamingTSBlock.convert_sequence_block(model.sequence_block)
    streaming_tsblocks = streaming_tsblocks.to(device).eval()

    # Convert encoder/decoder to stateful (for weight compatibility)
    model = convert_to_stateful(model, verbose=verbose, inplace=True)
    model.to(device).eval()

    # Disable streaming mode for encoder/decoder (non-streaming, zero-padded)
    set_streaming_mode(model, False)

    model_info = {
        "n_fft": MODEL_PARAMS["n_fft"],
        "hop_size": MODEL_PARAMS["hop_size"],
        "win_size": MODEL_PARAMS["win_size"],
        "compress_factor": MODEL_PARAMS["compress_factor"],
        "dense_channel": MODEL_PARAMS["dense_channel"],
        "num_tsblock": MODEL_PARAMS["num_tsblock"],
        "infer_type": MODEL_PARAMS["infer_type"],
        "encoder_padding_ratio": MODEL_PARAMS["encoder_padding_ratio"],
        "decoder_padding_ratio": MODEL_PARAMS["decoder_padding_ratio"],
        "chkpt_dir": None,
        "chkpt_file": None,
    }

    if verbose:
        print(f"  Model: Backbone ({MODEL_PARAMS['num_tsblock']} TSBlocks)")
        print(f"  Streaming TSBlocks: {len(streaming_tsblocks)} blocks")

    return model, streaming_tsblocks, model_info


def compute_freq_size(model: nn.Module, n_fft: int, device: str = "cpu") -> int:
    """Compute encoded frequency dimension via dummy forward."""
    freq_bins = n_fft // 2 + 1
    with torch.no_grad():
        dummy = torch.randn(1, 2, 4, freq_bins, device=device)
        freq_size = model.dense_encoder(dummy).shape[3]
    return freq_size


# ---------------------------------------------------------------------------
# Step 4: ONNX Export
# ---------------------------------------------------------------------------


def export_onnx(
    wrapper: StatefulExportableNNCore,
    output_path: str,
    freq_bins: int,
    export_time_frames: int,
    verbose: bool = True,
) -> str:
    """Export the wrapper to ONNX format.

    Args:
        wrapper: StatefulExportableNNCore instance.
        output_path: Path to save the .onnx file.
        freq_bins: Number of frequency bins (n_fft // 2 + 1).
        export_time_frames: Time dimension of the input tensors.
        verbose: Print export information.

    Returns:
        Path to the exported ONNX file.
    """
    device = next(wrapper.parameters()).device
    state_names = wrapper.state_names

    # Build dummy inputs
    mag = torch.randn(1, freq_bins, export_time_frames, device=device)
    pha = torch.randn(1, freq_bins, export_time_frames, device=device)

    state_registry = build_state_registry(wrapper.streaming_tsblocks, wrapper.freq_size)
    flat_states = []
    for name in state_names:
        shape = state_registry[name]
        flat_states.append(torch.zeros(*shape, device=device))

    dummy_input = (mag, pha, *flat_states)

    # I/O names
    input_names = ["mag", "pha"] + state_names
    output_names = ["est_mask", "phase_real", "phase_imag"] + [f"next_{n}" for n in state_names]

    if verbose:
        print(f"\nExporting ONNX model:")
        print(f"  Output: {output_path}")
        print(f"  Inputs: mag[1,{freq_bins},{export_time_frames}], "
              f"pha[1,{freq_bins},{export_time_frames}], "
              f"{len(state_names)} states")
        print(f"  Outputs: est_mask, phase_real, phase_imag, "
              f"{len(state_names)} next_states")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")

    return output_path


def simplify_onnx(
    input_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Apply onnx-simplifier to remove redundant ops (BN folding, reshape/transpose).

    Args:
        input_path: Path to the ONNX model.
        output_path: Path to save simplified model. Defaults to overwriting input.
        verbose: Print information.

    Returns:
        Path to the simplified model.
    """
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
        print(f"\nONNX simplification:")
        print(f"  Input:  {input_path} ({original_size:.1f} MB)")
        print(f"  Output: {output_path} ({simplified_size:.1f} MB)")
        print(f"  Reduction: {(1 - simplified_size / original_size) * 100:.1f}%")

    return output_path


# ---------------------------------------------------------------------------
# Step 5: Verification
# ---------------------------------------------------------------------------


def verify_onnx(
    wrapper: StatefulExportableNNCore,
    onnx_path: str,
    freq_bins: int,
    export_time_frames: int,
    num_chunks: int = 3,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """Verify ONNX model against PyTorch with multi-chunk state carry-over.

    Args:
        wrapper: PyTorch StatefulExportableNNCore.
        onnx_path: Path to the ONNX model.
        freq_bins: Number of frequency bins.
        export_time_frames: Time dimension of inputs.
        num_chunks: Number of chunks to verify (tests state carry-over).
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        verbose: Print verification details.

    Returns:
        True if verification passes.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("WARNING: onnxruntime not installed, skipping verification")
        return True

    if verbose:
        print(f"\nVerifying ONNX model ({num_chunks} chunks):")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    state_names = wrapper.state_names
    state_registry = build_state_registry(wrapper.streaming_tsblocks, wrapper.freq_size)

    # Initialize states (zeros)
    pt_flat_states = []
    ort_state_dict = {}
    for name in state_names:
        shape = state_registry[name]
        pt_flat_states.append(torch.zeros(*shape))
        ort_state_dict[name] = torch.zeros(*shape).numpy()

    max_error = 0.0
    all_passed = True

    for chunk_idx in range(num_chunks):
        # Random input
        mag = torch.randn(1, freq_bins, export_time_frames)
        pha = torch.randn(1, freq_bins, export_time_frames)

        # PyTorch forward
        with torch.no_grad():
            pt_outputs = wrapper(mag, pha, *pt_flat_states)

        pt_est_mask = pt_outputs[0]
        pt_phase_real = pt_outputs[1]
        pt_phase_imag = pt_outputs[2]
        pt_next_states = pt_outputs[3:]

        # ONNX Runtime forward
        ort_inputs = {"mag": mag.numpy(), "pha": pha.numpy()}
        ort_inputs.update(ort_state_dict)

        ort_outputs = sess.run(None, ort_inputs)

        ort_est_mask = ort_outputs[0]
        ort_phase_real = ort_outputs[1]
        ort_phase_imag = ort_outputs[2]
        ort_next_states = ort_outputs[3:]

        # Compare main outputs
        import numpy as np

        for out_name, pt_out, ort_out in [
            ("est_mask", pt_est_mask, ort_est_mask),
            ("phase_real", pt_phase_real, ort_phase_real),
            ("phase_imag", pt_phase_imag, ort_phase_imag),
        ]:
            err = np.max(np.abs(pt_out.numpy() - ort_out))
            max_error = max(max_error, err)
            if not np.allclose(pt_out.numpy(), ort_out, atol=atol, rtol=rtol):
                print(f"  FAIL: chunk {chunk_idx}, {out_name}, max_err={err:.2e}")
                all_passed = False

        # Compare state outputs
        for i, name in enumerate(state_names):
            err = np.max(np.abs(pt_next_states[i].numpy() - ort_next_states[i]))
            max_error = max(max_error, err)
            if not np.allclose(pt_next_states[i].numpy(), ort_next_states[i], atol=atol, rtol=rtol):
                print(f"  FAIL: chunk {chunk_idx}, next_{name}, max_err={err:.2e}")
                all_passed = False

        # Carry over states
        pt_flat_states = list(pt_next_states)
        ort_state_dict = {name: ort_next_states[i] for i, name in enumerate(state_names)}

        if verbose:
            print(f"  Chunk {chunk_idx}: max_err={max_error:.2e}")

    if all_passed:
        print(f"  Verification PASSED: max error = {max_error:.2e}")
    else:
        print(f"  Verification FAILED: max error = {max_error:.2e}")

    return all_passed


# ---------------------------------------------------------------------------
# Step 6: streaming_config.json
# ---------------------------------------------------------------------------


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _get_checkpoint_md5(chkpt_dir: Optional[str], chkpt_file: str = "best.th") -> Optional[str]:
    """Compute MD5 of checkpoint file."""
    if chkpt_dir is None:
        return None
    path = os.path.join(chkpt_dir, chkpt_file)
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def quantize_onnx_dynamic(
    input_path: str,
    output_path: str,
    verbose: bool = True,
) -> str:
    """Apply INT8 dynamic quantization to an ONNX model.

    Args:
        input_path: Path to the float32 ONNX model.
        output_path: Path to save the quantized model.
        verbose: Print information.

    Returns:
        Path to the quantized model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)

    if verbose:
        print(f"\nINT8 dynamic quantization:")
        print(f"  Input:  {input_path} ({input_size:.1f} MB)")
        print(f"  Output: {output_path} ({output_size:.1f} MB)")
        print(f"  Reduction: {(1 - output_size / input_size) * 100:.1f}%")

    return output_path


def quantize_onnx_qdq_for_htp(
    input_path: str,
    output_path: str,
    calibration_dir: Optional[str] = None,
    activation_type: str = "QUInt16",
    weight_type: str = "QUInt8",
    verbose: bool = True,
) -> str:
    """Apply QDQ static quantization optimized for QNN HTP execution.

    Uses ORT's QNN-specific quantization pipeline:
    1. Preprocess model for QNN compatibility
    2. Generate quantization config with CalibrationDataReader
    3. Apply QDQ quantization nodes

    Args:
        input_path: Path to the float32 ONNX model.
        output_path: Path to save the QDQ quantized model.
        calibration_dir: Directory with .npz calibration data (optional).
            Each .npz should contain 'mag' and 'pha' arrays.
            If None, uses random calibration data.
        activation_type: Activation quantization type ("QUInt8" or "QUInt16").
        weight_type: Weight quantization type ("QUInt8").
        verbose: Print information.

    Returns:
        Path to the quantized model.
    """
    import tempfile

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
        print(f"\nQDQ static quantization for HTP:")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Activation type: {activation_type}")
        print(f"  Weight type: {weight_type}")

    # Step 1: Preprocess model for QNN
    preprocessed_path = input_path.replace(".onnx", "_preproc.onnx")
    model_changed = qnn_preprocess_model(
        model_input=input_path,
        model_output=preprocessed_path,
    )
    if model_changed:
        if verbose:
            print(f"  Preprocessed model saved to: {preprocessed_path}")
        quant_input = preprocessed_path
    else:
        if verbose:
            print(f"  No preprocessing needed")
        quant_input = input_path

    # Step 2: Build CalibrationDataReader
    # Load model to get input shapes
    model = onnx.load(quant_input)
    input_shapes = {}
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else 1)
        input_shapes[inp.name] = shape

    class QnnCalibrationDataReader(CalibrationDataReader):
        def __init__(self, calibration_dir, input_shapes, num_samples=100):
            self.input_shapes = input_shapes
            self.num_samples = num_samples
            self.current = 0
            self.calibration_data = []

            if calibration_dir and os.path.isdir(calibration_dir):
                # Load real calibration data from .npz files
                npz_files = sorted([
                    f for f in os.listdir(calibration_dir) if f.endswith(".npz")
                ])
                for npz_file in npz_files[:num_samples]:
                    data = np.load(os.path.join(calibration_dir, npz_file))
                    sample = {}
                    for name, shape in input_shapes.items():
                        if name in data:
                            sample[name] = data[name].astype(np.float32)
                        elif name.startswith("state_"):
                            sample[name] = np.zeros(shape, dtype=np.float32)
                        else:
                            sample[name] = np.random.randn(*shape).astype(np.float32)
                    self.calibration_data.append(sample)
                if verbose:
                    print(f"  Loaded {len(self.calibration_data)} calibration samples from {calibration_dir}")
            else:
                # Random calibration data fallback
                for _ in range(num_samples):
                    sample = {}
                    for name, shape in input_shapes.items():
                        if name.startswith("state_"):
                            sample[name] = np.zeros(shape, dtype=np.float32)
                        elif name == "mag":
                            # Simulate compressed magnitude: positive values, typical range
                            sample[name] = np.abs(np.random.randn(*shape)).astype(np.float32) * 0.5
                        elif name == "pha":
                            # Phase: [-pi, pi]
                            sample[name] = (np.random.rand(*shape).astype(np.float32) * 2 - 1) * np.pi
                        else:
                            sample[name] = np.random.randn(*shape).astype(np.float32)
                    self.calibration_data.append(sample)
                if verbose:
                    print(f"  Using {num_samples} random calibration samples (no calibration_dir provided)")

        def get_next(self):
            if self.current >= len(self.calibration_data):
                return None
            sample = self.calibration_data[self.current]
            self.current += 1
            return sample

        def rewind(self):
            self.current = 0

    calib_reader = QnnCalibrationDataReader(calibration_dir, input_shapes)

    # Step 3: Get QNN QDQ config
    qnn_config = get_qnn_qdq_config(
        model_input=quant_input,
        calibration_data_reader=calib_reader,
        activation_type=act_qtype,
        weight_type=wt_qtype,
    )

    # Step 4: Quantize
    quantize(
        model_input=quant_input,
        model_output=output_path,
        quant_config=qnn_config,
    )

    # Cleanup preprocessed file
    if model_changed and os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)

    if verbose:
        print(f"  Input size:  {input_size:.1f} MB")
        print(f"  Output size: {output_size:.1f} MB")
        print(f"  Size change: {(output_size / input_size - 1) * 100:+.1f}%")

    return output_path


def generate_streaming_config(
    model_info: Dict[str, Any],
    state_registry: OrderedDict,
    chunk_size: int,
    encoder_lookahead: int,
    decoder_lookahead: int,
    export_time_frames: int,
    freq_size: int,
    output_path: str,
    quantization: str = "float32",
    verbose: bool = True,
) -> str:
    """Generate streaming_config.json compatible with Android infrastructure.

    Args:
        model_info: Model metadata from preparation step.
        state_registry: Ordered state name → shape mapping.
        chunk_size: Chunk size in STFT frames.
        encoder_lookahead: Encoder lookahead frames.
        decoder_lookahead: Decoder lookahead frames.
        export_time_frames: Time dimension of ONNX model inputs.
        freq_size: Encoded frequency dimension.
        output_path: Path to save the JSON file.
        verbose: Print information.

    Returns:
        Path to the generated config file.
    """
    n_fft = model_info["n_fft"]
    freq_bins = n_fft // 2 + 1
    state_names = list(state_registry.keys())

    config = {
        "model_info": {
            "name": os.path.basename(model_info.get("chkpt_dir") or "no_checkpoint"),
            "version": "1.0.0",
            "export_format": "stateful_nncore",
            "quantization": quantization,
            "phase_output_mode": "complex",
            "qnn_compatible": True,
            "supported_backends": ["qnn_htp", "nnapi", "cpu"],
            "infer_type": model_info["infer_type"],
        },
        "stft_config": {
            "n_fft": n_fft,
            "hop_size": model_info["hop_size"],
            "win_length": model_info["win_size"],
            "sample_rate": 16000,
            "center": True,
            "compress_factor": model_info["compress_factor"],
        },
        "streaming_config": {
            "chunk_size_frames": chunk_size,
            "encoder_lookahead": encoder_lookahead,
            "decoder_lookahead": decoder_lookahead,
            "export_time_frames": export_time_frames,
            "freq_bins": freq_bins,
            "freq_bins_encoded": freq_size,
        },
        "qnn_config": {
            "target_soc": "SM8550",
            "htp_performance_mode": "burst",
            "context_cache_enabled": True,
            "vtcm_mb": 8,
            "enable_htp_fp16_precision": False,
        },
        "state_info": {
            "num_states": len(state_names),
            "state_names": state_names,
        },
        "export_info": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "checkpoint_md5": _get_checkpoint_md5(
                model_info.get("chkpt_dir"), model_info.get("chkpt_file", "best.th"),
            ),
            "git_commit": _get_git_commit(),
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"\nGenerated streaming_config.json:")
        print(f"  Output: {output_path}")
        print(f"  States: {len(state_names)}")
        print(f"  Chunk size: {chunk_size} frames")
        print(f"  Export time frames: {export_time_frames}")

    return output_path


# ---------------------------------------------------------------------------
# Step 7: CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export LaCoSENet streaming model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From checkpoint
  python scripts/export_onnx.py --chkpt_dir results/experiments/prk_0131_1 \\
      --chunk_size 32 --encoder_lookahead 7 --decoder_lookahead 7

  # Without checkpoint (testing)
  python scripts/export_onnx.py --no_checkpoint --chunk_size 32 \\
      --encoder_lookahead 7 --decoder_lookahead 7
        """,
    )

    # Model source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--chkpt_dir", type=str, help="Checkpoint directory path")
    source_group.add_argument(
        "--no_checkpoint", action="store_true",
        help="Use random weights (for testing export pipeline)",
    )

    parser.add_argument("--chkpt_file", type=str, default="best.th", help="Checkpoint filename")

    # Streaming parameters
    parser.add_argument("--chunk_size", type=int, default=32, help="Chunk size in STFT frames")
    parser.add_argument("--encoder_lookahead", type=int, default=7, help="Encoder lookahead frames")
    parser.add_argument("--decoder_lookahead", type=int, default=7, help="Decoder lookahead frames")

    # Output
    parser.add_argument("--output_dir", type=str, default="exports", help="Output directory")
    parser.add_argument("--output_name", type=str, default=None, help="ONNX filename (auto-generated if not set)")

    # Quantization
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 dynamic quantization")
    parser.add_argument("--quantize_qdq", action="store_true", help="Apply QDQ static quantization for QNN HTP")
    parser.add_argument("--calibration_dir", type=str, default=None,
                        help="Directory with .npz calibration data for QDQ quantization (optional)")
    parser.add_argument("--qdq_activation_type", type=str, default="QUInt8",
                        choices=["QUInt8", "QUInt16"],
                        help="QDQ activation quantization type (default: QUInt8, QUInt16 may fail on HTP for PReLU)")
    parser.add_argument("--qdq_weight_type", type=str, default="QUInt8",
                        choices=["QUInt8"],
                        help="QDQ weight quantization type (default: QUInt8)")

    # Simplification
    parser.add_argument("--simplify", action="store_true", help="Apply onnx-simplifier (BN folding, redundant op removal)")

    # Verification
    parser.add_argument("--skip_verify", action="store_true", help="Skip ONNX verification")
    parser.add_argument("--verify_chunks", type=int, default=3, help="Number of chunks for verification")

    args = parser.parse_args()

    print("=" * 64)
    print("LaCoSENet ONNX Export")
    print("=" * 64)

    device = "cpu"  # ONNX export is always on CPU

    # --- Prepare model ---
    if args.no_checkpoint:
        model, streaming_tsblocks, model_info = prepare_without_checkpoint(device=device)
    else:
        model, streaming_tsblocks, model_info = prepare_from_checkpoint(
            chkpt_dir=args.chkpt_dir,
            chkpt_file=args.chkpt_file,
            device=device,
        )

    # --- Compute dimensions ---
    n_fft = model_info["n_fft"]
    freq_bins = n_fft // 2 + 1
    freq_size = compute_freq_size(model, n_fft, device=device)
    export_time_frames = args.chunk_size + max(args.encoder_lookahead, args.decoder_lookahead)

    print(f"\nDimensions:")
    print(f"  freq_bins (STFT): {freq_bins}")
    print(f"  freq_size (encoded): {freq_size}")
    print(f"  export_time_frames: {export_time_frames} "
          f"(chunk={args.chunk_size} + lookahead={max(args.encoder_lookahead, args.decoder_lookahead)})")

    # --- Build state registry ---
    state_registry = build_state_registry(streaming_tsblocks, freq_size)
    print(f"  num_states: {len(state_registry)}")

    # --- Create wrapper ---
    wrapper = StatefulExportableNNCore(
        model=model,
        streaming_tsblocks=streaming_tsblocks,
        freq_size=freq_size,
        chunk_size=args.chunk_size,
        infer_type=model_info["infer_type"],
    )
    wrapper.eval()

    # --- Determine output filename ---
    if args.output_name:
        onnx_filename = args.output_name
    else:
        if args.chkpt_dir:
            base = os.path.basename(args.chkpt_dir.rstrip("/"))
        else:
            base = "test_model"
        onnx_filename = f"{base}_c{args.chunk_size}.onnx"

    onnx_path = os.path.join(args.output_dir, onnx_filename)
    config_path = os.path.join(args.output_dir, "streaming_config.json")

    # --- Export ONNX ---
    export_onnx(
        wrapper=wrapper,
        output_path=onnx_path,
        freq_bins=freq_bins,
        export_time_frames=export_time_frames,
    )

    # --- Verify ---
    if not args.skip_verify:
        verify_onnx(
            wrapper=wrapper,
            onnx_path=onnx_path,
            freq_bins=freq_bins,
            export_time_frames=export_time_frames,
            num_chunks=args.verify_chunks,
        )

    # --- Simplify ---
    if args.simplify:
        simplify_onnx(input_path=onnx_path)

    # --- Quantization ---
    quantization_label = "float32"
    if args.quantize:
        quantize_onnx_dynamic(
            input_path=onnx_path,
            output_path=onnx_path,  # overwrite in-place
        )
        quantization_label = "int8_dynamic"

    if args.quantize_qdq:
        qdq_path = onnx_path.replace(".onnx", "_qdq.onnx")
        quantize_onnx_qdq_for_htp(
            input_path=onnx_path,
            output_path=qdq_path,
            calibration_dir=args.calibration_dir,
            activation_type=args.qdq_activation_type,
            weight_type=args.qdq_weight_type,
        )
        # Verify QDQ model with relaxed tolerance
        if not args.skip_verify:
            verify_onnx(
                wrapper=wrapper,
                onnx_path=qdq_path,
                freq_bins=freq_bins,
                export_time_frames=export_time_frames,
                num_chunks=args.verify_chunks,
                atol=0.1,
                rtol=0.1,
            )

    # --- Generate streaming_config.json ---
    generate_streaming_config(
        model_info=model_info,
        state_registry=state_registry,
        chunk_size=args.chunk_size,
        encoder_lookahead=args.encoder_lookahead,
        decoder_lookahead=args.decoder_lookahead,
        export_time_frames=export_time_frames,
        freq_size=freq_size,
        output_path=config_path,
        quantization=quantization_label,
    )

    print(f"\nDone. Files:")
    print(f"  {onnx_path}")
    if args.quantize_qdq:
        qdq_path = onnx_path.replace(".onnx", "_qdq.onnx")
        print(f"  {qdq_path}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()
