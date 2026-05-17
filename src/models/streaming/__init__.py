"""Streaming inference subsystem for BAFNet+ (stateful-conv chunk-by-chunk path).

Leaf modules built in session S2 of the streaming-ONNX rebuild:
    - ``layers.stateful_conv``: stateful ``CausalConv1d`` / ``AsymmetricConv2d`` /
      ``CausalConv2d``.
    - ``context``: ``StateFramesContext`` thread-local + stateful-layer helpers.
    - ``converters``: convert a ``Backbone`` to its stateful form, toggle
      streaming, reset state.
    - ``lookahead``: derive ``L_enc`` / ``L_dec`` from the configured backbone's
      ``AsymmetricConv2d`` right-padding fields.
    - ``onnx``: explicit-state (functional) versions for ONNX export.

S3 added:
    - ``backbone_streaming.BackboneStreaming``: the PyTorch streaming wrapper
      (spectrogram → spectrogram) for a single mapping/masking ``Backbone`` —
      host STFT-context buffer + decoder-lookahead feature buffer + ``process_*``.

S4 added the exportable single-Backbone ONNX path under ``onnx/``:
    - ``onnx.backbone_core.ExportableBackboneCore``: the graph-internal model
      with every conv state as an explicit forward arg/return (state buffers in
      the graph, host STFT/iSTFT/feature-buffer outside) + the matching
      ``torch.onnx.export`` driver, sidecar JSON and ORT-vs-PT verify harness
      in ``onnx.export``.

S5 added the BAFNet+ decomposed-core (PyTorch, non-streaming) under
``onnx.bafnetplus_core``:
    - :class:`BAFNetPlusCore`: re-expresses ``BAFNetPlus.forward`` via two
      :class:`ExportableBackboneCore` instances + shared-reference calibration /
      alpha modules. Same per-branch ``(mag, pha)`` boundary as S4.
    - :func:`prove_t_export`: one-chunk ``T_export`` proof harness.

S6 added the streaming wrapper for BAFNet+ — ``bafnetplus_streaming.BAFNetPlusStreaming``:
two ``BackboneStreaming`` instances + the BAFNet+ fusion algebra (calibration
+ alpha softmax blend, both non-streaming at S6 — turned functional in S8)
+ a shared iSTFT/OLA buffer for the monophonic fused output. The wrapper takes
paired ``(bcs_audio, acs_audio)`` streams and returns enhanced audio.

S8 will turn the fusion-path layers functional-stateful and export the BAFNet+
FP32 ONNX; S9 is the ORT host wrapper.
"""

from src.models.streaming.backbone_streaming import BackboneStreaming, SpectrogramChunk
from src.models.streaming.bafnetplus_streaming import BAFNetPlusStreaming
from src.models.streaming.context import StateFramesContext, get_state_frames_context
from src.models.streaming.converters import (
    convert_to_stateful,
    prepare_streaming_model,
    reset_streaming_state,
    set_streaming_mode,
)
from src.models.streaming.layers.stateful_conv import (
    StatefulAsymmetricConv2d,
    StatefulCausalConv1d,
    StatefulCausalConv2d,
)
from src.models.streaming.lookahead import LookaheadInfo, compute_lookahead
from src.models.streaming.onnx import (
    BAFNetPlusCore,
    BAFNetPlusOrtStreaming,
    ExportableBAFNetPlusCore,
    ExportableBackboneCore,
    ExportResult,
    audit_alpha_time_lookahead,
    audit_calibration_is_causal,
    export_backbone_core_to_onnx,
    export_backbone_to_onnx_from_checkpoint,
    export_bafnetplus_core_to_onnx,
    export_bafnetplus_to_onnx_from_checkpoint,
    load_backbone_from_checkpoint,
    load_bafnetplus_from_checkpoint,
    prove_t_export,
    run_bafnetplus_core_streaming,
    run_core_streaming,
    verify_backbone_core_multistep,
    verify_bafnetplus_core_multistep,
)

__all__ = [
    "StateFramesContext",
    "get_state_frames_context",
    "StatefulCausalConv1d",
    "StatefulAsymmetricConv2d",
    "StatefulCausalConv2d",
    "convert_to_stateful",
    "set_streaming_mode",
    "reset_streaming_state",
    "prepare_streaming_model",
    "compute_lookahead",
    "LookaheadInfo",
    "BackboneStreaming",
    "SpectrogramChunk",
    # S6 BAFNet+ PT streaming wrapper.
    "BAFNetPlusStreaming",
    # S4 exportable ONNX core + driver (re-exported from streaming.onnx).
    "ExportableBackboneCore",
    "ExportResult",
    "export_backbone_core_to_onnx",
    "export_backbone_to_onnx_from_checkpoint",
    "load_backbone_from_checkpoint",
    "run_core_streaming",
    "verify_backbone_core_multistep",
    # S5 BAFNet+ decomposed core + proof harness + audits.
    "BAFNetPlusCore",
    "prove_t_export",
    "audit_calibration_is_causal",
    "audit_alpha_time_lookahead",
    # S8 functional-stateful exportable BAFNet+ core + export driver.
    "ExportableBAFNetPlusCore",
    "export_bafnetplus_core_to_onnx",
    "export_bafnetplus_to_onnx_from_checkpoint",
    "load_bafnetplus_from_checkpoint",
    "run_bafnetplus_core_streaming",
    "verify_bafnetplus_core_multistep",
    # S9 ORT host wrapper.
    "BAFNetPlusOrtStreaming",
]
