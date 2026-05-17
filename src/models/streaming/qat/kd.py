"""Self-distillation loss against the FP32 D1 teacher (PESQ-GAN replacement).

The MetricGAN PESQ-prediction discriminator has been the per-step training
bottleneck (CPU ``batch_pesq`` via joblib at :func:`src.losses.batch_pesq`).
Phase C drops the discriminator and replaces its supervisory signal with a
self-KD loss against the frozen FP32 base (D1 teacher).

Loss form
---------
``L_kd = w_out_mag · MSE(s_mag, t_mag)
       + w_out_pha · phase_losses(t_pha, s_pha)
       + w_out_com · 2 · MSE(s_com, t_com)
       + w_feat    · Σ_{L in feature_layers} MSE(s_feat[L], t_feat[L])``

Output-level terms reuse :func:`src.losses.phase_losses` so the KD signal lives
in the same metric space as the FP32 base training loss; this avoids
distribution shift when warm-starting the QAT model from the FP32 weights.

Feature-level alignment (caveat)
--------------------------------
The PT2E-prepared student is a :class:`torch.fx.GraphModule` that retains the
original ``mapping.sequence_block.0``-style submodules in ``named_modules()``
(so the hook *registration* succeeds on the student), **but the FX-traced
forward routes through the rewritten graph and never invokes those eager
submodules**, so registered hooks never fire on the student side. The teacher
(eager :class:`BAFNetPlus`) DOES capture all TSBlock outputs.

Practical outcome: student-side ``_student_feats`` stays empty; the per-layer
loop in :meth:`forward` skips when either side is missing; the feature term
collapses to zero. Output-level KD (mag/pha/com) carries the supervisory
signal. Cycle 18 may instrument the FX graph with explicit trace points for
true feature alignment if output-level alone proves insufficient.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import phase_losses

logger = logging.getLogger(__name__)


DEFAULT_FEATURE_LAYERS: Tuple[str, ...] = (
    # Teacher (eager BAFNetPlus) module paths for TSBlock outputs.
    "mapping.sequence_block.0",
    "mapping.sequence_block.1",
    "mapping.sequence_block.2",
    "mapping.sequence_block.3",
    "masking.sequence_block.0",
    "masking.sequence_block.1",
    "masking.sequence_block.2",
    "masking.sequence_block.3",
)


DEFAULT_WEIGHTS: Dict[str, float] = {
    "output_mag": 1.0,
    "output_pha": 1.0,
    "output_com": 1.0,  # multiplied by 2 inside forward (matches src/solver.py:290)
    "feature_uniform": 0.5,
}


class KDLoss(nn.Module):
    """Self-KD loss assembly for QAT (output-level + optional feature-level).

    Usage::

        teacher = build_fp32_teacher()
        student = prepare_bafnetplus_for_qat(...)

        kd = KDLoss(feature_layers=DEFAULT_FEATURE_LAYERS, weights=DEFAULT_WEIGHTS)
        kd.register_hooks(student, teacher)

        # In training step:
        kd.clear()
        with torch.no_grad():
            t_out = teacher(model_input)
        s_out = student(model_input)
        loss_dict = kd(s_out, t_out)
        loss_kd = loss_dict["total"]
    """

    def __init__(
        self,
        feature_layers: Optional[Iterable[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.feature_layers: List[str] = list(feature_layers) if feature_layers is not None else list(DEFAULT_FEATURE_LAYERS)
        self.weights: Dict[str, float] = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)

        # Caches for per-step features. Cleared by ``clear()`` between steps.
        self._student_feats: Dict[str, torch.Tensor] = {}
        self._teacher_feats: Dict[str, torch.Tensor] = {}
        self._student_handles: List[Any] = []
        self._teacher_handles: List[Any] = []

    # ------------------------------------------------------------- hooks

    def register_hooks(self, student: nn.Module, teacher: nn.Module) -> Dict[str, List[str]]:
        """Register forward hooks for feature-level alignment.

        Returns a dict ``{'student_hit': [...], 'teacher_hit': [...]}`` listing
        which layers were actually found in each model. Cycle 17 expects the
        student-side list to be empty (PT2E flattens the hierarchy); the
        teacher-side list should contain all eight TSBlock paths.
        """
        self.remove_hooks()
        student_named = dict(student.named_modules())
        teacher_named = dict(teacher.named_modules())

        student_hit: List[str] = []
        teacher_hit: List[str] = []
        for layer in self.feature_layers:
            if layer in student_named:
                handle = student_named[layer].register_forward_hook(
                    self._make_hook(self._student_feats, layer)
                )
                self._student_handles.append(handle)
                student_hit.append(layer)
            if layer in teacher_named:
                handle = teacher_named[layer].register_forward_hook(
                    self._make_hook(self._teacher_feats, layer)
                )
                self._teacher_handles.append(handle)
                teacher_hit.append(layer)
        logger.info(
            "[KD hooks] student-side hit %d / %d layers; teacher-side hit %d / %d",
            len(student_hit), len(self.feature_layers),
            len(teacher_hit), len(self.feature_layers),
        )
        return {"student_hit": student_hit, "teacher_hit": teacher_hit}

    def remove_hooks(self) -> None:
        for h in self._student_handles:
            h.remove()
        for h in self._teacher_handles:
            h.remove()
        self._student_handles.clear()
        self._teacher_handles.clear()

    def clear(self) -> None:
        """Empty per-step feature caches. Call before every forward pass."""
        self._student_feats.clear()
        self._teacher_feats.clear()

    def _make_hook(self, target: Dict[str, torch.Tensor], layer: str) -> Callable:
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                target[layer] = tensor
        return _hook

    # ------------------------------------------------------------- forward

    def forward(
        self,
        student_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        teacher_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the KD loss dict.

        Args:
            student_out: ``(mag, pha, com)`` from the QAT student forward.
            teacher_out: ``(mag, pha, com)`` from the FP32 teacher forward (no_grad).

        Returns:
            Dict with per-term loss tensors (each scalar) and a ``total`` key.
        """
        s_mag, s_pha, s_com = student_out
        t_mag, t_pha, t_com = teacher_out

        w_mag = self.weights.get("output_mag", 1.0)
        w_pha = self.weights.get("output_pha", 1.0)
        w_com = self.weights.get("output_com", 1.0)
        w_feat = self.weights.get("feature_uniform", 0.5)

        loss_mag = F.mse_loss(s_mag, t_mag.detach()) * w_mag
        loss_pha = phase_losses(t_pha.detach(), s_pha) * w_pha
        loss_com = F.mse_loss(s_com, t_com.detach()) * 2 * w_com

        loss_feat = torch.zeros((), device=s_mag.device, dtype=s_mag.dtype)
        n_aligned = 0
        for layer in self.feature_layers:
            sf = self._student_feats.get(layer)
            tf = self._teacher_feats.get(layer)
            if sf is None or tf is None:
                continue
            if sf.shape != tf.shape:
                continue  # PT2E may transform shapes; skip with no error
            loss_feat = loss_feat + F.mse_loss(sf, tf.detach())
            n_aligned += 1
        if n_aligned > 0:
            loss_feat = loss_feat * (w_feat / max(1, n_aligned))

        total = loss_mag + loss_pha + loss_com + loss_feat
        return {
            "output_mag": loss_mag,
            "output_pha": loss_pha,
            "output_com": loss_com,
            "feature": loss_feat,
            "total": total,
        }
