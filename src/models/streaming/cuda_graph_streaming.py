"""CUDA-graph streaming driver for BAFNet+ (Local GPU low-latency inference).

The eager PyTorch streaming wrapper (:class:`~src.models.streaming.bafnetplus_streaming.BAFNetPlusStreaming`,
S6) is launch-overhead-bound on a GPU: each chunk issues hundreds of tiny conv
kernels through the Python eager dispatcher, pinning per-chunk latency at
~25 ms regardless of tier on an RTX 5090 (c09 Local GPU sweep). The GPU itself
is near-idle between launches. ``torch.compile(mode="reduce-overhead")`` cannot
fix it because the eager stateful convs reassign their persistent ``_state``
tensors every step (``stateful_conv.py``: ``self._state = ...clone()``), which
CUDA graphs forbid (graphs require static state addresses).

``BAFNetPlusCudaGraphStreaming`` removes the overhead by driving the
*functional* :class:`~src.models.streaming.onnx.bafnetplus_core.ExportableBAFNetPlusCore`
instead — explicit 190-state I/O, branch-free, the same core CP10 exports to
ONNX. One steady-state chunk-step (the forward plus the 190-state copy-back) is
captured once into a :class:`torch.cuda.CUDAGraph`; each subsequent chunk is a
single :meth:`torch.cuda.CUDAGraph.replay` — hundreds of kernel launches
collapsed into one.

Scope: GPU-only, isolated (spectrogram magnitude/phase in, spectrogram out — no
host STFT/iSTFT, no audio buffering). It is the Local GPU measurement vehicle
for the c09 cross-platform RTF matrix, not a drop-in replacement for the
production ``BAFNetPlusStreaming`` audio-streaming API.

The driver follows the ``CUDA_VISIBLE_DEVICES``-pinned single-GPU convention of
the c09 benchmark scripts (one visible CUDA device per process).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch import Tensor

from src.models.streaming.onnx.bafnetplus_core import ExportableBAFNetPlusCore
from src.models.streaming.onnx.export import load_bafnetplus_from_checkpoint

logger = logging.getLogger(__name__)


class BAFNetPlusCudaGraphStreaming:
    """Capture-and-replay CUDA-graph streaming driver for ``ExportableBAFNetPlusCore``.

    Build one per latency tier via :meth:`from_checkpoint`. Construction loads
    the per-tier checkpoint, builds the functional core, allocates static
    input/state buffers (fixed addresses — mandatory for graph capture) and
    captures one chunk-step. Steady-state inference is then :meth:`step` (copy
    inputs in, one replay) or :meth:`replay` (replay with pre-loaded inputs).

    Attributes:
        core: The functional exportable core (on the GPU, ``eval()``, grad off).
        chunk_size: STFT frames produced per chunk-step.
        t_export: Frames consumed per chunk-step (``chunk_size + total_lookahead``).
        freq_size: ``n_fft // 2 + 1``.
        num_states: State tensors threaded across chunks (190 for the 50 ms anchor).
        device: The CUDA device.
    """

    def __init__(
        self,
        core: ExportableBAFNetPlusCore,
        chunk_size: int,
        *,
        device: Union[str, torch.device] = "cuda",
        warmup_iters: int = 11,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"BAFNetPlusCudaGraphStreaming is GPU-only, got device={device}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.device = device
        self.chunk_size = int(chunk_size)
        self.core = core.to(device).eval()
        self.core.requires_grad_(False)
        self.core.set_state_frames_for_update(self.chunk_size)

        self.num_states = self.core.num_states
        self.t_export = self.chunk_size + self.core.total_lookahead
        self.freq_size = self.core.n_fft // 2 + 1

        # Static buffers — fixed addresses are mandatory for CUDA-graph capture.
        shape = (1, self.freq_size, self.t_export)
        self._bcs_mag = torch.zeros(shape, device=device)
        self._bcs_pha = torch.zeros(shape, device=device)
        self._acs_mag = torch.zeros(shape, device=device)
        self._acs_pha = torch.zeros(shape, device=device)
        self._states: List[Tensor] = self.core.init_states(
            batch_size=1, freq_size=self.freq_size, time_frames=self.t_export, device=device
        )
        if len(self._states) != self.num_states:
            raise RuntimeError(
                f"init_states returned {len(self._states)} tensors, expected num_states={self.num_states}"
            )

        self._graph = torch.cuda.CUDAGraph()
        self._outs: Tuple[Tensor, ...] = ()
        self._capture(warmup_iters)
        self.reset_state()

    @classmethod
    def from_checkpoint(
        cls,
        chkpt_dir: Union[str, Path],
        chunk_size: int,
        *,
        chkpt_file: str = "best.th",
        device: Union[str, torch.device] = "cuda",
        phase_output_mode: str = "complex",
        warmup_iters: int = 11,
        verbose: bool = True,
    ) -> "BAFNetPlusCudaGraphStreaming":
        """Load a per-tier unified BAFNet+ checkpoint and build the CUDA-graph driver.

        Args:
            chkpt_dir: A ``results/experiments/bafnetplus_<T>ms`` directory.
            chunk_size: Tier chunk size (c09 LOCKED convention: the 12.5 / 25 /
                50 / 75 / 100 ms tiers use chunk 2 / 4 / 8 / 12 / 16).
            phase_output_mode: ``'complex'`` (the deployed s20 graph CP10
                exports) or ``'atan2'``.
        """
        bafnet, _info = load_bafnetplus_from_checkpoint(
            chkpt_dir, chkpt_file=chkpt_file, device="cpu", verbose=verbose
        )
        core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet, phase_output_mode=phase_output_mode)
        return cls(core, chunk_size, device=device, warmup_iters=warmup_iters)

    @torch.no_grad()
    def _capture(self, warmup_iters: int) -> None:
        """Warm up on a side stream, then capture forward + the 190-state copy-back."""
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup_iters):
                self.core(self._bcs_mag, self._bcs_pha, self._acs_mag, self._acs_pha, *self._states)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        with torch.cuda.graph(self._graph):
            outs = self.core(self._bcs_mag, self._bcs_pha, self._acs_mag, self._acs_pha, *self._states)
            # Fold the state copy-back into the graph: the forward reads the
            # static state buffers first and the copy-back writes them last, so
            # one replay advances state correctly with zero per-chunk launch
            # cost. outs[:3] = (est_mag, phase_a, phase_b); outs[3:] = next states.
            for buf, nxt in zip(self._states, outs[3:]):
                buf.copy_(nxt)
        self._outs = outs

    @torch.no_grad()
    def reset_state(self) -> None:
        """Zero the streaming state in place (start of a new utterance)."""
        for st in self._states:
            st.zero_()

    @torch.no_grad()
    def step(
        self,
        bcs_mag: Tensor,
        bcs_pha: Tensor,
        acs_mag: Tensor,
        acs_pha: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run one streaming chunk-step: load inputs, then one CUDA-graph replay.

        Args:
            bcs_mag / bcs_pha / acs_mag / acs_pha: ``[1, freq_size, t_export]``
                per-branch magnitude / phase chunks (on any device — copied into
                the static GPU buffers).

        Returns:
            The core's first three outputs (``est_mag`` plus two phase/complex
            tensors), each a view into the graph's static output buffers —
            valid only until the next :meth:`step` / :meth:`replay` (clone to
            retain).
        """
        self._bcs_mag.copy_(bcs_mag)
        self._bcs_pha.copy_(bcs_pha)
        self._acs_mag.copy_(acs_mag)
        self._acs_pha.copy_(acs_pha)
        self._graph.replay()
        return self._outs[0], self._outs[1], self._outs[2]

    @torch.no_grad()
    def replay(self) -> None:
        """Replay the captured chunk-step against the pre-loaded static inputs.

        For the isolated zero-input latency benchmark: the static input buffers
        already hold the deterministic input, so the measured unit is exactly
        one ``graph.replay()`` (no input marshalling).
        """
        self._graph.replay()

    def load_inputs(self, bcs_mag: Tensor, bcs_pha: Tensor, acs_mag: Tensor, acs_pha: Tensor) -> None:
        """Copy a chunk into the static input buffers without replaying."""
        self._bcs_mag.copy_(bcs_mag)
        self._bcs_pha.copy_(bcs_pha)
        self._acs_mag.copy_(acs_mag)
        self._acs_pha.copy_(acs_pha)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(chunk_size={self.chunk_size}, t_export={self.t_export}, "
            f"freq_size={self.freq_size}, num_states={self.num_states}, device={self.device})"
        )


__all__ = ["BAFNetPlusCudaGraphStreaming"]
