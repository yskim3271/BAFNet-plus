"""ONNX-export helpers for the BAFNet+ streaming subsystem.

S2 built the leaf layers:
    - ``functional_stateful``: explicit-state ``FunctionalStatefulConv{1d,2d}`` /
      ``FunctionalStatefulCausalConv2d`` (no internal buffer — state is a forward
      arg/return) + ``convert_to_functional``.
    - ``state_registry``: ``StateInfo`` / ``StateRegistry`` /
      ``collect_states_from_model`` for wiring conv states as graph I/O.

S4 added the exportable single-Backbone path:
    - ``backbone_core``: :class:`ExportableBackboneCore` (the graph-internal
      model — encoder/TS/decoders with every conv state as an explicit forward
      arg/return) + :class:`StateIterator` + :func:`convert_stateful_to_functional`.
    - ``export``: ``torch.onnx.export`` driver, multi-chunk ORT-vs-PT verify
      harness, ``run_core_streaming`` PT chunk-driver, sidecar JSON.

A ``conv_transpose_wrapper`` (DepthToSpace + Pad + Conv2d decomposition of
``nn.ConvTranspose2d``) was intentionally **not** ported: the LaCoSENet version
is an INT8/QDQ + NNAPI helper, and the FP32-only rebuild uses
``torch.onnx.export(opset_version=17, dynamo=False)`` which exports the
``(1,3)`` / stride ``(1,2)`` ``ConvTranspose2d`` in ``MaskDecoder`` /
``PhaseDecoder`` natively as a single ``ConvTranspose`` op (verified: ORT FP32
vs PyTorch max abs diff ~1e-6 on torch 2.9.1 / onnxruntime 1.18.1).

S5 added the BAFNet+ decomposed-core (PyTorch, non-streaming) under
``bafnetplus_core``:
    - :class:`BAFNetPlusCore`: re-expresses ``BAFNetPlus.forward`` via two
      :class:`ExportableBackboneCore` instances (one per branch) + shared
      references to the calibration / alpha modules — same ``(mag, pha)``
      host boundary per branch.
    - :func:`prove_t_export`: one-chunk ``T_export`` proof harness comparing
      ``chunk + max(L_enc, L_dec)`` vs ``chunk + L_enc + L_dec`` (+
      ``L_alpha`` if non-zero) against ``BAFNetPlus.forward()``.
    - :func:`audit_calibration_is_causal` / :func:`audit_alpha_time_lookahead`:
      structural causality audits used by the S5 tests.

S6 will add the streaming wrapper (dual STFT contexts, paired host buffers);
S8 turns the calibration / alpha layers functional and exports the BAFNet+
FP32 ONNX; S9 is the ORT host wrapper.
"""

from src.models.streaming.onnx.backbone_core import (
    ExportableBackboneCore,
    StateIterator,
    convert_stateful_to_functional,
)
from src.models.streaming.onnx.bafnetplus_core import (
    BAFNetPlusCore,
    BAFNetPlusHeadCore,
    BAFNetPlusTrunkCore,
    ExportableBAFNetPlusCore,
    audit_alpha_time_lookahead,
    audit_calibration_is_causal,
    prove_t_export,
)
from src.models.streaming.onnx.export import (
    ExportResult,
    export_backbone_core_to_onnx,
    export_backbone_to_onnx_from_checkpoint,
    export_bafnetplus_core_to_onnx,
    export_bafnetplus_head_to_onnx,
    export_bafnetplus_split_to_onnx_from_checkpoint,
    export_bafnetplus_to_onnx_from_checkpoint,
    export_bafnetplus_trunk_to_onnx,
    load_backbone_from_checkpoint,
    load_bafnetplus_from_checkpoint,
    run_bafnetplus_core_streaming,
    run_bafnetplus_split_streaming,
    run_core_streaming,
    verify_backbone_core_multistep,
    verify_bafnetplus_core_multistep,
    verify_bafnetplus_split_multistep,
)
from src.models.streaming.onnx.functional_stateful import (
    FunctionalStatefulCausalConv2d,
    FunctionalStatefulConv1d,
    FunctionalStatefulConv2d,
    convert_to_functional,
)
from src.models.streaming.onnx.ort_wrapper import (
    BAFNetPlusOrtSplitStreaming,
    BAFNetPlusOrtStreaming,
)
from src.models.streaming.onnx.state_registry import (
    StateInfo,
    StateRegistry,
    collect_states_from_model,
)

__all__ = [
    # S2 functional layers + registry.
    "FunctionalStatefulConv1d",
    "FunctionalStatefulConv2d",
    "FunctionalStatefulCausalConv2d",
    "convert_to_functional",
    "StateInfo",
    "StateRegistry",
    "collect_states_from_model",
    # S4 exportable core + export driver.
    "ExportableBackboneCore",
    "StateIterator",
    "convert_stateful_to_functional",
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
    # S21 split core (trunk + head graph splitting for B2 deployable).
    "BAFNetPlusTrunkCore",
    "BAFNetPlusHeadCore",
    "export_bafnetplus_trunk_to_onnx",
    "export_bafnetplus_head_to_onnx",
    "export_bafnetplus_split_to_onnx_from_checkpoint",
    "run_bafnetplus_split_streaming",
    "verify_bafnetplus_split_multistep",
    "BAFNetPlusOrtSplitStreaming",
]
