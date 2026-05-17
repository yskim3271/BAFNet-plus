"""Unit tests for Phase C QAT scaffolding (cycle 17).

Coverage:

- :class:`QNNQuantizer` annotation: a Conv2d on a tiny model gets the
  per-channel int8 weight + per-tensor uint16 activation spec, and skip
  substrings (phase_conv, alpha_out, ...) keep matching nodes out of the
  annotated list.
- :func:`prepare_bafnetplus_for_qat`: synthetic 2-TSBlock BAFNet+ → PT2E prep
  → trainable GraphModule that forwards without crashing on a [1,F,T,2]
  example.
- Eval-mode bit-identity vs FP32 base: prepared model in eval mode (fake-quant
  inert) returns outputs within numerical tolerance of the FP32 forward.
- :class:`KDLoss`: output-level forward returns a finite scalar with a backward
  pass that flows gradients into the student.
- :class:`Solver` ``__init__``: builds with ``discriminator=None`` +
  ``teacher=None`` + ``kd_loss=None`` without crashing, and ``_serialize``
  on the resulting state survives the None guards.
- Warm-start: D1 ONNX trunk → ``warm_start_from_ptq`` extracts scales and
  reports a mapping rate. Skipped if the D1 asset is missing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn

from src.checkpoint import ConfigDict
from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.qat import (
    KDLoss,
    QNNQuantizer,
    prepare_bafnetplus_for_qat,
    warm_start_from_ptq,
)
from src.models.streaming.qat.kd import DEFAULT_FEATURE_LAYERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- shared

CHUNK_SIZE = 8
N_FFT, HOP, WIN, COMPRESS = 400, 100, 400, 0.3
FREQ_SIZE = N_FFT // 2 + 1  # 201

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
D1_TRUNK_ONNX = _PROJECT_ROOT / "results" / "onnx" / "bafnetplus_50ms_int8_qdq_trunk_t2.onnx"


def _synthetic_backbone_param(infer_type: str, dense_channel: int = 8) -> Dict:
    return {
        "n_fft": N_FFT,
        "hop_size": HOP,
        "win_size": WIN,
        "dense_channel": dense_channel,
        "sigmoid_beta": 2.0,
        "compress_factor": COMPRESS,
        "dense_depth": 4,
        "num_tsblock": 2,
        "time_dw_kernel_size": 3,
        "time_block_kernel": [3, 5, 7, 11],
        "freq_block_kernel": [3, 11, 23, 31],
        "time_block_num": 2,
        "freq_block_num": 2,
        "causal_ts_block": True,
        "encoder_padding_ratio": (0.9, 0.1),
        "decoder_padding_ratio": (0.9, 0.1),
        "sca_kernel_size": 11,
        "infer_type": infer_type,
    }


def _synthetic_bafnetplus(ablation_mode: str = "full", dense_channel: int = 8) -> BAFNetPlus:
    args_mapping = ConfigDict(
        {
            "model_lib": "backbone",
            "model_class": "Backbone",
            "param": _synthetic_backbone_param("mapping", dense_channel=dense_channel),
        }
    )
    args_masking = ConfigDict(
        {
            "model_lib": "backbone",
            "model_class": "Backbone",
            "param": _synthetic_backbone_param("masking", dense_channel=dense_channel),
        }
    )
    model = BAFNetPlus(
        args_mapping=args_mapping,
        args_masking=args_masking,
        ablation_mode=ablation_mode,
        load_pretrained_weights=False,
    )
    return model


def _example_inputs(time_frames: int = 32):
    bcs = torch.randn(1, FREQ_SIZE, time_frames, 2) * 0.3
    acs = torch.randn(1, FREQ_SIZE, time_frames, 2) * 0.3
    return bcs, acs


# ============================================================ Quantizer annotation

class _MicroNet(nn.Module):
    """One Conv2d + ReLU — minimal annotation target."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 4, 3, padding=1, bias=True)
        self.r = nn.ReLU()

    def forward(self, x):
        return self.r(self.c(x))


def test_quantizer_annotation_dummy_conv2d():
    """QNNQuantizer.annotate marks at least one Conv node with W8A16 specs."""
    from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e

    model = _MicroNet().train()
    example = (torch.randn(1, 3, 8, 8),)
    exported = torch.export.export(model, example, strict=False)
    quant = QNNQuantizer()
    prepared = prepare_qat_pt2e(exported.module(), quant)

    assert quant.annotated_node_names, "Expected at least one annotated Conv node"
    summary = quant.summary()
    assert summary["scheme"] == "w8a16_qnn"
    assert summary["weight"].startswith("per_channel_symmetric_int8")
    assert summary["activation"].startswith("per_tensor_affine_uint16")

    # Forward + backward should run.
    out = prepared(*example)
    loss = out.mean()
    loss.backward()
    assert out.shape == (1, 4, 8, 8)


def test_quantizer_skip_patterns_exclude_head_and_phase_conv():
    """Nodes whose name contains a skip substring are routed to skipped_node_names."""

    class _Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha_out = nn.Conv2d(2, 2, 1)
            self.regular = nn.Conv2d(2, 2, 3, padding=1)

        def forward(self, x):
            return self.regular(self.alpha_out(x))

    from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e

    model = _Mixed().train()
    example = (torch.randn(1, 2, 8, 8),)
    exported = torch.export.export(model, example, strict=False)
    quant = QNNQuantizer()
    _ = prepare_qat_pt2e(exported.module(), quant)

    skipped = quant.skipped_node_names
    annotated = quant.annotated_node_names
    # PT2E flattens FX node names to ``conv2d`` / ``conv2d_1``; the original
    # module hierarchy is recovered from nn_module_stack + weight-arg names.
    # The alpha_out conv runs first (order matters) so it gets the bare
    # ``conv2d`` name, and ``regular`` gets ``conv2d_1``.
    assert len(skipped) == 1 and len(annotated) == 1, (
        f"Expected 1 skipped + 1 annotated; got annotated={annotated} skipped={skipped}"
    )
    assert skipped[0] == "conv2d", f"alpha_out conv should be skipped (first); got skipped={skipped}"
    assert annotated[0] == "conv2d_1", f"regular conv should be annotated (second); got annotated={annotated}"


# ============================================================ prepare_bafnetplus_for_qat

@pytest.mark.slow
def test_prepare_synthetic_bafnetplus_forward():
    """Synthetic BAFNet+ → PT2E prep → forward returns 3 tensors of the right shape."""
    model = _synthetic_bafnetplus("full")
    bcs, acs = _example_inputs(time_frames=32)
    prepared = prepare_bafnetplus_for_qat(model, ((bcs, acs),))
    prepared.eval()  # eval mode → fake-quant collapses to identity
    out = prepared((bcs, acs))
    assert isinstance(out, tuple) and len(out) == 3
    est_mag, est_pha, est_com = out
    assert est_mag.shape == (1, FREQ_SIZE, 32)
    assert est_pha.shape == (1, FREQ_SIZE, 32)
    assert est_com.shape == (1, FREQ_SIZE, 32, 2)
    assert torch.isfinite(est_mag).all()


@pytest.mark.slow
def test_prepare_preserves_fp32_eval_within_tolerance():
    """Eval-mode prepared model output is close to FP32 base output.

    PT2E fake-quant in eval mode (observer disabled, fake_quant_enabled=0)
    operates as identity. The exact bit-identity claim depends on whether the
    QAT observers had any opportunity to record stats before eval — we set
    eval immediately so the round-trip should be within a small epsilon.
    """
    torch.manual_seed(123)
    model = _synthetic_bafnetplus("full").eval()
    bcs, acs = _example_inputs(time_frames=16)
    with torch.no_grad():
        fp32_mag, fp32_pha, fp32_com = model((bcs, acs))

    prepared = prepare_bafnetplus_for_qat(model, ((bcs, acs),))
    # Disable fake-quant explicitly to simulate "pure FP32 path through the
    # prepared graph" — without this, PT2E may still apply fake-quant in eval
    # if observers have collected stats.
    from torch.ao.quantization.fake_quantize import FakeQuantize

    for mod in prepared.modules():
        if isinstance(mod, FakeQuantize):
            mod.disable_fake_quant()
            mod.disable_observer()
    prepared.eval()
    with torch.no_grad():
        prep_mag, prep_pha, prep_com = prepared((bcs, acs))

    dmag = (fp32_mag - prep_mag).abs().max().item()
    dcom = (fp32_com - prep_com).abs().max().item()
    # Generous tolerance — PT2E's graph rewriting may introduce libm-level diffs.
    assert dmag < 1e-3, f"FP32 vs prepared eval |dmag|={dmag:.3e}"
    assert dcom < 1e-3, f"FP32 vs prepared eval |dcom|={dcom:.3e}"


# ============================================================ KDLoss

def test_kd_loss_forward_runs_and_grads_flow():
    """KDLoss returns a finite scalar; gradients flow back into student tensors."""
    torch.manual_seed(7)
    B, F_, T = 1, 16, 8
    student_mag = torch.randn(B, F_, T, requires_grad=True)
    student_pha = torch.randn(B, F_, T, requires_grad=True)
    student_com = torch.randn(B, F_, T, 2, requires_grad=True)
    teacher_mag = torch.randn(B, F_, T)
    teacher_pha = torch.randn(B, F_, T)
    teacher_com = torch.randn(B, F_, T, 2)

    kd = KDLoss()
    losses = kd(
        student_out=(student_mag, student_pha, student_com),
        teacher_out=(teacher_mag, teacher_pha, teacher_com),
    )
    assert set(losses.keys()) == {"output_mag", "output_pha", "output_com", "feature", "total"}
    assert torch.isfinite(losses["total"]).item()
    losses["total"].backward()
    assert student_mag.grad is not None
    assert torch.isfinite(student_mag.grad).all().item()
    assert student_com.grad is not None
    # Feature-level term collapses to zero when no hooks fire (PT2E student
    # has no compatible submodules and we did not register hooks here).
    assert losses["feature"].abs().item() < 1e-12


def test_kd_loss_register_hooks_reports_hit_counts():
    """Hooks register on the teacher eager model; PT2E-student-side is sparse."""
    teacher = _synthetic_bafnetplus("full").eval()
    student = teacher  # simulate eager-eager (both sides find the layers)

    kd = KDLoss(feature_layers=list(DEFAULT_FEATURE_LAYERS)[:2])  # use 2 layers for speed
    hits = kd.register_hooks(student, teacher)
    # Synthetic BAFNet+ has 2 TSBlocks → mapping.sequence_block.0 + .1 both hit.
    assert len(hits["student_hit"]) == 2
    assert len(hits["teacher_hit"]) == 2
    kd.remove_hooks()


# ============================================================ Solver None-gating

def test_solver_init_with_discriminator_none():
    """Solver(__init__) with discriminator=None, optim_disc=None succeeds."""
    from omegaconf import OmegaConf

    from src.solver import Solver

    model = nn.Linear(2, 2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    args = OmegaConf.create(
        {
            "loss": {"magnitude": 0.9, "phase": 0.3, "complex": 0.1, "consistency": 0.1, "metric": 0.0},
            "n_fft": 400,
            "hop_size": 100,
            "win_size": 400,
            "compress_factor": 0.3,
            "model": {"input_type": "acs+bcs"},
            "device": "cpu",
            "epochs": 1,
            "continue_from": None,
            "eval_every": 1,
            "valid_start_epoch": 0,
            "log_dir": "/tmp/qat_test_logdir",
            "num_prints": 1,
            "num_workers": 0,
            "kd_weight": 1.0,
            "fit_steps": None,
        }
    )
    data = {"tr_loader": [], "va_loader": [], "ev_loader_list": {}}
    solver = Solver(
        data=data,
        model=model,
        discriminator=None,
        optim=optim,
        optim_disc=None,
        scheduler=None,
        scheduler_disc=None,
        args=args,
        logger=logger,
        device=torch.device("cpu"),
        teacher=None,
        kd_loss=None,
    )
    # _serialize must not crash even with discriminator=None.
    os.chdir("/tmp")
    solver._serialize()  # writes /tmp/checkpoint.th + (no best.th since best_state is None)
    assert Path("/tmp/checkpoint.th").exists()


# ============================================================ warm-start (skipped if D1 missing)

@pytest.mark.skipif(
    not D1_TRUNK_ONNX.exists(),
    reason=f"D1 trunk ONNX not present at {D1_TRUNK_ONNX}; warm-start test skipped",
)
@pytest.mark.slow
def test_warm_start_extracts_d1_scales_with_reasonable_rate():
    """warm_start_from_ptq returns a non-zero mapping rate on D1 trunk."""
    model = _synthetic_bafnetplus("full")
    bcs, acs = _example_inputs(time_frames=32)
    prepared = prepare_bafnetplus_for_qat(model, ((bcs, acs),))
    summary = warm_start_from_ptq(prepared, str(D1_TRUNK_ONNX), verbose=False)
    assert summary["num_pt_total"] > 0, "Expected at least some PT fake-quant modules"
    assert summary["num_onnx_total"] > 0, "Expected ONNX QDQ scales to load"
    # Cycle 17 exit gate: ≥ 70 % is the GO criterion; we accept ≥ 1 match as
    # the test threshold (the synthetic 2-TSBlock model has very different
    # graph from the 4-TSBlock real D1, so substring matching may be partial).
    assert summary["num_matched"] >= 1, (
        f"Expected at least one PT→ONNX match; got summary={summary}"
    )
    logger.info(
        "warm-start mapping rate on synthetic vs real D1: %.1f%% (%d/%d)",
        summary["mapping_rate"] * 100,
        summary["num_matched"], summary["num_pt_total"],
    )
