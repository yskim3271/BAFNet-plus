"""
BAFNetPlus Streaming Wrapper with Lookahead Buffering.

Dual-input streaming inference for BAFNetPlus (mapping + masking + fusion).
Structurally mirrors :class:`src.models.streaming.lacosenet.LaCoSENet` but:
- Accepts two synchronized input streams (bcs_samples, acs_samples).
- Runs two Backbones (mapping/masking) sequentially per chunk, sharing a
  single ``StateFramesContext`` block for state-update guarding.
- Adds a fusion step (calibration + alpha 2D conv) whose submodules are
  converted to stateful variants via :func:`convert_to_stateful`.
- Produces a single enhanced waveform stream via a single iSTFT OLA.

Stage 1 scope:
- Only ablation_mode='full' is supported; other modes raise NotImplementedError.
- Fusion weights come from BAFNetPlus.__init__ (Kaiming random); for
  offline-streaming parity the caller must keep a pre-streaming-conversion
  deepcopy of BAFNetPlus (see ``from_checkpoint(return_offline=True)``).
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.stft import complex_to_mag_pha, mag_pha_stft, mag_pha_to_complex, manual_istft_ola

logger = logging.getLogger(__name__)


class BAFNetPlusStreaming(nn.Module):
    """
    Streaming wrapper for BAFNetPlus (two Backbones + fusion).

    Attributes:
        chunk_size: STFT frames emitted per output chunk.
        encoder_lookahead: Future frames buffered for the encoder (input lookahead).
        decoder_lookahead: Future frames buffered for the decoder.
        total_lookahead: encoder_lookahead + decoder_lookahead.
        latency_ms: Total algorithmic latency in ms.
    """

    def __init__(
        self,
        model: nn.Module,
        streaming_tsblocks_mapping: nn.ModuleList,
        streaming_tsblocks_masking: nn.ModuleList,
        chunk_size: int = 8,
        encoder_lookahead: int = 3,
        decoder_lookahead: int = 3,
        hop_size: int = 100,
        n_fft: int = 400,
        win_size: int = 400,
        compress_factor: float = 0.3,
        sample_rate: int = 16000,
        freq_size: int = 100,
        stft_center: bool = True,
        disable_state_guard: bool = False,
    ):
        super().__init__()

        if getattr(model, "ablation_mode", None) != "full":
            raise NotImplementedError(
                f"Stage 1 only supports ablation_mode='full', got " f"'{getattr(model, 'ablation_mode', None)}'"
            )
        if streaming_tsblocks_mapping is None or streaming_tsblocks_masking is None:
            raise ValueError("streaming_tsblocks_mapping and streaming_tsblocks_masking are required")

        self.model = model
        self.model.eval()

        self.streaming_tsblocks_mapping = streaming_tsblocks_mapping
        self.streaming_tsblocks_masking = streaming_tsblocks_masking

        # STFT parameters
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.sample_rate = sample_rate
        self.stft_center = stft_center
        if stft_center:
            self.stft_future_samples = self.win_size // 2
            self.stft_center_delay_samples = self.stft_future_samples
        else:
            self.stft_future_samples = 0
            self.stft_center_delay_samples = 0

        # Streaming parameters
        self.chunk_size = chunk_size
        self.encoder_lookahead = encoder_lookahead
        self.decoder_lookahead = decoder_lookahead
        self.input_lookahead_frames = int(encoder_lookahead)
        self.total_lookahead = self.input_lookahead_frames + decoder_lookahead

        self.total_frames_needed = chunk_size + self.input_lookahead_frames
        if stft_center:
            self.samples_per_chunk = (self.total_frames_needed - 1) * hop_size + self.stft_future_samples
        else:
            self.samples_per_chunk = (self.total_frames_needed - 1) * hop_size + n_fft

        self.output_frames_per_chunk = chunk_size
        self.output_samples_per_chunk = chunk_size * hop_size
        self.ola_tail_size = win_size - hop_size

        self.latency_samples = self.total_lookahead * hop_size + self.stft_center_delay_samples
        self.latency_ms = self.latency_samples / sample_rate * 1000

        self.freq_size = freq_size
        self.disable_state_guard = disable_state_guard

        # Buffers (initialized on CPU, moved to device lazily)
        self._reset_buffers()

        self._streaming_config: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Buffer / state management
    # ------------------------------------------------------------------
    def _reset_buffers(self) -> None:
        """Reset all internal buffers (input, STFT context, feature, OLA, TSBlock states)."""
        self.bcs_input_buffer = torch.tensor([], dtype=torch.float32)
        self.acs_input_buffer = torch.tensor([], dtype=torch.float32)

        self.bcs_feature_buffer: List[Dict[str, Any]] = []
        self.acs_feature_buffer: List[Dict[str, Any]] = []
        self._buffered_frames = 0

        if self.stft_center:
            self.bcs_stft_context = torch.zeros(self.win_size // 2, dtype=torch.float32)
            self.acs_stft_context = torch.zeros(self.win_size // 2, dtype=torch.float32)
        else:
            self.bcs_stft_context = None
            self.acs_stft_context = None

        self._ola_buffer = torch.zeros(self.ola_tail_size, dtype=torch.float32)
        self._ola_norm = torch.zeros(self.ola_tail_size, dtype=torch.float32)

        self._tsblock_states_mapping = self._init_tsblock_states(self.streaming_tsblocks_mapping)
        self._tsblock_states_masking = self._init_tsblock_states(self.streaming_tsblocks_masking)

    def _init_tsblock_states(self, tsblocks: nn.ModuleList) -> List[List[Dict[str, Tensor]]]:
        device = self.device
        dtype = next(self.model.parameters()).dtype
        all_states: List[List[Dict[str, Tensor]]] = []
        for block in tsblocks:
            block_states = block.init_state(
                batch_size=1,
                freq_size=self.freq_size,
                device=device,
                dtype=dtype,
            )
            all_states.append(block_states)
        return all_states

    def reset_state(self) -> None:
        """Reset all streaming state for a new audio stream."""
        self._reset_buffers()
        from src.models.streaming.converters.conv_converter import reset_streaming_state

        reset_streaming_state(self.model.mapping)
        reset_streaming_state(self.model.masking)
        if self.model.use_calibration:
            reset_streaming_state(self.model.calibration_encoder)
        reset_streaming_state(self.model.alpha_convblocks)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def streaming_config(self) -> Dict[str, Any]:
        return {
            **self._streaming_config,
            "chunk_size_frames": self.chunk_size,
            "encoder_lookahead": self.encoder_lookahead,
            "decoder_lookahead": self.decoder_lookahead,
            "input_lookahead_frames": self.input_lookahead_frames,
            "total_lookahead": self.total_lookahead,
            "stft_center": self.stft_center,
            "stft_center_delay_samples": self.stft_center_delay_samples,
            "output_frames_per_chunk": self.output_frames_per_chunk,
            "samples_per_chunk": self.samples_per_chunk,
            "output_samples_per_chunk": self.output_samples_per_chunk,
            "latency_samples": self.latency_samples,
            "latency_ms": self.latency_ms,
            "hop_size": self.hop_size,
            "sample_rate": self.sample_rate,
            "freq_size": self.freq_size,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        chkpt_dir_mapping: str,
        chkpt_dir_masking: str,
        chkpt_file: str = "best.th",
        chunk_size: int = 8,
        encoder_lookahead: int = 3,
        decoder_lookahead: int = 3,
        ablation_mode: str = "full",
        device: Optional[str] = None,
        verbose: bool = True,
        return_offline: bool = False,
    ):
        """
        Build a BAFNetPlusStreaming from two Backbone checkpoint directories.

        The two checkpoints must share num_tsblock, time_block_kernel, padding ratios,
        STFT params, dense_channel, and dense_depth. Otherwise a ValueError is raised.

        Args:
            return_offline: If True, also return the pre-streaming-conversion
                :class:`BAFNetPlus` instance (for offline-streaming parity testing).
        """
        from omegaconf import OmegaConf

        from src.models.bafnetplus import BAFNetPlus
        from src.models.streaming.converters.conv_converter import convert_to_stateful, set_streaming_mode
        from src.models.streaming.utils import apply_stateful_conv, apply_streaming_tsblock

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Compatibility check on params from both checkpoints
        def _load_conf(d: str):
            return OmegaConf.load(os.path.join(d, ".hydra", "config.yaml"))

        conf_map = _load_conf(chkpt_dir_mapping)
        conf_mask = _load_conf(chkpt_dir_masking)
        p_map = conf_map.model.param
        p_mask = conf_mask.model.param

        compat_fields = [
            "num_tsblock",
            "time_block_kernel",
            "encoder_padding_ratio",
            "decoder_padding_ratio",
            "causal_ts_block",
            "dense_channel",
            "dense_depth",
            "n_fft",
            "hop_size",
            "win_size",
            "compress_factor",
        ]
        for f in compat_fields:
            v_map = p_map.get(f)
            v_mask = p_mask.get(f)
            if v_map != v_mask:
                raise ValueError(f"checkpoint config mismatch on '{f}': mapping={v_map} vs masking={v_mask}")

        # 2. Build BAFNetPlus (loads both backbones + Kaiming-inits fusion)
        bafnetplus_original = BAFNetPlus(
            checkpoint_mapping=os.path.join(chkpt_dir_mapping, chkpt_file),
            checkpoint_masking=os.path.join(chkpt_dir_masking, chkpt_file),
            ablation_mode=ablation_mode,
        )
        bafnetplus_original.eval()
        bafnetplus_original.to(device)

        # 2b. Deepcopy so streaming conversion leaves offline model untouched
        bafnetplus = copy.deepcopy(bafnetplus_original)

        # 3. Streaming conversions on Backbones (apply_stateful_conv handles eval + set_streaming_mode)
        _, tsblocks_map, count_ts_map = apply_streaming_tsblock(bafnetplus.mapping, verbose=verbose)
        apply_stateful_conv(bafnetplus.mapping, device=device, verbose=verbose)
        _, tsblocks_mask, count_ts_mask = apply_streaming_tsblock(bafnetplus.masking, verbose=verbose)
        apply_stateful_conv(bafnetplus.masking, device=device, verbose=verbose)

        # 4. Convert fusion modules to stateful (must .eval() before set_streaming_mode)
        if bafnetplus.use_calibration:
            convert_to_stateful(bafnetplus.calibration_encoder, inplace=True, verbose=verbose)
            bafnetplus.calibration_encoder.eval()
            set_streaming_mode(bafnetplus.calibration_encoder, True)
        convert_to_stateful(bafnetplus.alpha_convblocks, inplace=True, verbose=verbose)
        bafnetplus.alpha_convblocks.eval()
        set_streaming_mode(bafnetplus.alpha_convblocks, True)

        # 5. Infer freq_size from each branch; verify agreement
        stft_freq = p_map.n_fft // 2 + 1
        with torch.no_grad():
            dummy = torch.randn(1, 2, 4, stft_freq, device=next(bafnetplus.parameters()).device)
            fs_map = bafnetplus.mapping.dense_encoder(dummy).shape[3]
            fs_mask = bafnetplus.masking.dense_encoder(dummy).shape[3]
        if fs_map != fs_mask:
            raise ValueError(f"freq_size mismatch: mapping={fs_map} vs masking={fs_mask}")

        # 6. Instantiate wrapper
        sample_rate = int(conf_map.sampling_rate)
        stft_center = True  # conv defaults; checkpoint configs don't override this for 50ms tier

        instance = cls(
            model=bafnetplus,
            streaming_tsblocks_mapping=tsblocks_map,
            streaming_tsblocks_masking=tsblocks_mask,
            chunk_size=chunk_size,
            encoder_lookahead=encoder_lookahead,
            decoder_lookahead=decoder_lookahead,
            hop_size=int(p_map.hop_size),
            n_fft=int(p_map.n_fft),
            win_size=int(p_map.win_size),
            compress_factor=float(p_map.compress_factor),
            sample_rate=sample_rate,
            freq_size=fs_map,
            stft_center=stft_center,
        )
        instance._streaming_config = {
            "chkpt_dir_mapping": chkpt_dir_mapping,
            "chkpt_dir_masking": chkpt_dir_masking,
            "encoder_padding_ratio": list(p_map.encoder_padding_ratio),
            "decoder_padding_ratio": list(p_map.decoder_padding_ratio),
            "tsblock_count_mapping": count_ts_map,
            "tsblock_count_masking": count_ts_mask,
            "ablation_mode": ablation_mode,
        }

        if verbose:
            logger.info(
                "BAFNetPlusStreaming loaded: chunk_size=%d, encoder_lookahead=%d, "
                "decoder_lookahead=%d, latency_ms=%.1f",
                chunk_size,
                encoder_lookahead,
                decoder_lookahead,
                instance.latency_ms,
            )

        if return_offline:
            return instance, bafnetplus_original
        return instance

    # ------------------------------------------------------------------
    # STFT / encoder / decoder helpers (shared by per-chunk and fast paths)
    # ------------------------------------------------------------------
    def _stft(self, audio: Tensor) -> Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        _, _, com = mag_pha_stft(
            audio,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            center=False,
        )
        return com

    def _process_encoder(
        self,
        spectrogram: Tensor,
        dense_encoder: nn.Module,
        tsblocks: nn.ModuleList,
        tsblock_states: List[List[Dict[str, Tensor]]],
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Encoder path for one branch: complex spec -> dense_encoder -> streaming TSBlocks.

        Returns (mag, ts_out, valid_frames).
        The caller is responsible for wrapping this in a StateFramesContext (for the
        StatefulAsymmetricConv2d layers inside dense_encoder). StreamingTSBlock does NOT
        auto-read thread-local, so we pass valid_frames explicitly to each block.
        """
        _, _, T, _ = spectrogram.shape
        mag, pha = complex_to_mag_pha(spectrogram, stack_dim=-1)
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)  # [B, 2, T, F]
        encoded = dense_encoder(x)
        ts_out = encoded
        valid_frames = min(T, self.chunk_size)
        sf = None if self.disable_state_guard else valid_frames
        for i, block in enumerate(tsblocks):
            ts_out, new_states = block(ts_out, tsblock_states[i], state_frames=sf)
            tsblock_states[i] = new_states
        return mag, ts_out, valid_frames

    def _buffer_features(
        self,
        ts_out: Tensor,
        mag: Tensor,
        valid_frames: int,
        feature_buffer: List[Dict[str, Any]],
    ) -> None:
        feature_buffer.append(
            {
                "features": ts_out[:, :, :valid_frames, :],
                "mag": mag[:, :, :valid_frames],
                "frames": valid_frames,
            }
        )

    def _extract_extended(self, feature_buffer: List[Dict[str, Any]]) -> Tuple[Tensor, Tensor]:
        total_needed = self.chunk_size + self.decoder_lookahead
        all_features = torch.cat([buf["features"] for buf in feature_buffer], dim=2)
        all_mag = torch.cat([buf["mag"] for buf in feature_buffer], dim=2)
        return all_features[:, :, :total_needed, :], all_mag[:, :, :total_needed]

    def _slide_feature_buffer(self, feature_buffer: List[Dict[str, Any]], frames_to_remove: int) -> int:
        removed = 0
        while removed < frames_to_remove and feature_buffer:
            buf = feature_buffer[0]
            if buf["frames"] <= (frames_to_remove - removed):
                removed += buf["frames"]
                feature_buffer.pop(0)
            else:
                keep_frames = buf["frames"] - (frames_to_remove - removed)
                buf["features"] = buf["features"][:, :, -keep_frames:, :]
                buf["mag"] = buf["mag"][:, :, -keep_frames:]
                buf["frames"] = keep_frames
                removed = frames_to_remove
        return removed

    def _decode_branch(
        self,
        extended_features: Tensor,
        extended_mag: Tensor,
        mask_decoder: nn.Module,
        phase_decoder: nn.Module,
        infer_type: str,
        return_mask: bool,
    ) -> Tuple[Tensor, ...]:
        """
        Decoder pass for one branch. Runs both mask_decoder and phase_decoder on the
        extended (chunk + lookahead) features, then trims to chunk_size.
        Caller is responsible for StateFramesContext wrapping.

        Returns:
            (est_mag, est_pha, est_com) for infer_type='mapping' (return_mask=False).
            (est_mag, est_pha, est_com, mask) for infer_type='masking' (return_mask=True).
        """
        mask_out = mask_decoder(extended_features).squeeze(1).transpose(1, 2)  # [B, F, T_ext]
        est_pha = phase_decoder(extended_features).squeeze(1).transpose(1, 2)
        if infer_type == "masking":
            est_mag = extended_mag * mask_out
        else:
            est_mag = mask_out

        est_mag = est_mag[:, :, : self.chunk_size]
        est_pha = est_pha[:, :, : self.chunk_size]
        mask_out = mask_out[:, :, : self.chunk_size]
        est_com = mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)
        if return_mask:
            return est_mag, est_pha, est_com, mask_out
        return est_mag, est_pha, est_com

    # ------------------------------------------------------------------
    # Fusion (calibration + alpha)
    # ------------------------------------------------------------------
    def _apply_calibration_streaming(
        self,
        bcs_com_out: Tensor,
        acs_com_out: Tensor,
        bcs_mag: Tensor,
        acs_mag: Tensor,
        acs_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Streaming calibration — same math as BAFNetPlus._apply_calibration but
        calibration_encoder is stateful (called within StateFramesContext)."""
        model = self.model
        eps = 1e-8
        bcs_log_energy = torch.log(bcs_mag.pow(2).mean(dim=1, keepdim=True) + eps)
        acs_log_energy = torch.log(acs_mag.pow(2).mean(dim=1, keepdim=True) + eps)
        log_energy_diff = bcs_log_energy - acs_log_energy
        acs_mask_mean = acs_mask.mean(dim=1, keepdim=True)
        acs_mask_var = acs_mask.var(dim=1, keepdim=True, unbiased=False)
        calibration_feat = torch.cat(
            [bcs_log_energy, acs_log_energy, log_energy_diff, acs_mask_mean, acs_mask_var],
            dim=1,
        )  # [B, 5, T]

        calibration_hidden = model.calibration_encoder(calibration_feat)  # [B, 16, T]
        common_log_gain = torch.tanh(model.common_gain_head(calibration_hidden))
        common_log_gain = common_log_gain * model.calibration_max_common_log_gain

        if model.use_relative_gain:
            relative_log_gain = torch.tanh(model.relative_gain_head(calibration_hidden))
            relative_log_gain = relative_log_gain * model.calibration_max_relative_log_gain
            bcs_gain = torch.exp(common_log_gain - 0.5 * relative_log_gain)
            acs_gain = torch.exp(common_log_gain + 0.5 * relative_log_gain)
        else:
            bcs_gain = acs_gain = torch.exp(common_log_gain)

        bcs_gain = bcs_gain.transpose(1, 2).unsqueeze(1)  # [B, 1, T, 1]
        acs_gain = acs_gain.transpose(1, 2).unsqueeze(1)

        bcs_com_cal = bcs_com_out * bcs_gain
        acs_com_cal = acs_com_out * acs_gain
        return bcs_com_cal, acs_com_cal

    def _fuse(self, bcs_com_cal: Tensor, acs_com_cal: Tensor, acs_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Alpha fusion — replicates BAFNetPlus.forward lines 258-274 for ablation='full'."""
        model = self.model
        eps = 1e-8

        bcs_mag_cal = torch.sqrt(bcs_com_cal[:, :, :, 0] ** 2 + bcs_com_cal[:, :, :, 1] ** 2 + eps)
        acs_mag_cal = torch.sqrt(acs_com_cal[:, :, :, 0] ** 2 + acs_com_cal[:, :, :, 1] ** 2 + eps)
        alpha_feat = torch.stack([bcs_mag_cal, acs_mag_cal, acs_mask], dim=1).transpose(2, 3)  # [B, 3, T, F]

        for block in model.alpha_convblocks:
            alpha_feat = block(alpha_feat)
        alpha = model.alpha_out(alpha_feat)  # [B, 2, T, F]
        alpha = alpha.transpose(2, 3)  # [B, 2, F, T]
        alpha = torch.softmax(alpha, dim=1)
        alpha_bcs = alpha[:, 0].unsqueeze(-1)
        alpha_acs = alpha[:, 1].unsqueeze(-1)
        est_com = bcs_com_cal * alpha_bcs + acs_com_cal * alpha_acs

        est_mag, est_pha = complex_to_mag_pha(est_com, stack_dim=-1)
        return est_mag, est_pha

    def _manual_istft_ola(self, est_mag: Tensor, est_pha: Tensor) -> Tensor:
        """Cross-chunk OLA iSTFT, consuming/updating self._ola_buffer + self._ola_norm."""
        if self._ola_buffer.device != est_mag.device:
            self._ola_buffer = self._ola_buffer.to(est_mag.device)
            self._ola_norm = self._ola_norm.to(est_mag.device)
        output, new_buf, new_norm = manual_istft_ola(
            est_mag,
            est_pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            ola_buffer=self._ola_buffer,
            ola_norm=self._ola_norm,
        )
        self._ola_buffer = new_buf
        self._ola_norm = new_norm
        return output[: self.output_samples_per_chunk]

    # ------------------------------------------------------------------
    # Per-chunk streaming
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_samples(self, bcs_samples: Tensor, acs_samples: Tensor) -> Optional[Tensor]:
        """
        Stream two synchronized audio buffers. Returns an enhanced chunk when available.

        Args:
            bcs_samples: [T] or [1, T] — BCS input chunk.
            acs_samples: [T] or [1, T] — ACS input chunk (same length as bcs_samples).

        Returns:
            Enhanced waveform [output_samples_per_chunk] when a chunk is ready, else None.
        """
        from src.models.streaming.utils import StateFramesContext

        if bcs_samples.dim() == 2:
            if bcs_samples.shape[0] != 1:
                raise ValueError("Batch size must be 1 for streaming")
            bcs_samples = bcs_samples.squeeze(0)
        if acs_samples.dim() == 2:
            if acs_samples.shape[0] != 1:
                raise ValueError("Batch size must be 1 for streaming")
            acs_samples = acs_samples.squeeze(0)
        if bcs_samples.shape != acs_samples.shape:
            raise ValueError(f"bcs/acs sample length mismatch: {bcs_samples.shape} vs {acs_samples.shape}")

        device = self.device
        bcs_samples = bcs_samples.to(device)
        acs_samples = acs_samples.to(device)

        if self.stft_center and self.bcs_stft_context.device != device:
            self.bcs_stft_context = self.bcs_stft_context.to(device)
            self.acs_stft_context = self.acs_stft_context.to(device)

        self.bcs_input_buffer = torch.cat([self.bcs_input_buffer.to(device), bcs_samples])
        self.acs_input_buffer = torch.cat([self.acs_input_buffer.to(device), acs_samples])

        if len(self.bcs_input_buffer) < self.samples_per_chunk:
            return None

        bcs_chunk = self.bcs_input_buffer[: self.samples_per_chunk]
        acs_chunk = self.acs_input_buffer[: self.samples_per_chunk]

        if self.stft_center:
            context_size = self.win_size // 2
            bcs_with_ctx = torch.cat([self.bcs_stft_context, bcs_chunk])
            acs_with_ctx = torch.cat([self.acs_stft_context, acs_chunk])
            bcs_spec = self._stft(bcs_with_ctx)
            acs_spec = self._stft(acs_with_ctx)

            advance = self.output_samples_per_chunk
            if advance >= context_size:
                self.bcs_stft_context = self.bcs_input_buffer[advance - context_size : advance].clone()
                self.acs_stft_context = self.acs_input_buffer[advance - context_size : advance].clone()
            else:
                need_from_prev = context_size - advance
                self.bcs_stft_context = torch.cat(
                    [
                        self.bcs_stft_context[len(self.bcs_stft_context) - need_from_prev :],
                        self.bcs_input_buffer[:advance],
                    ]
                ).clone()
                self.acs_stft_context = torch.cat(
                    [
                        self.acs_stft_context[len(self.acs_stft_context) - need_from_prev :],
                        self.acs_input_buffer[:advance],
                    ]
                ).clone()
        else:
            bcs_spec = self._stft(bcs_chunk)
            acs_spec = self._stft(acs_chunk)

        # Encoder: process both branches within one StateFramesContext
        _, _, T_spec, _ = bcs_spec.shape
        valid_frames = min(T_spec, self.chunk_size)
        with StateFramesContext(None if self.disable_state_guard else valid_frames):
            bcs_mag, bcs_ts_out, vf_bcs = self._process_encoder(
                bcs_spec,
                self.model.mapping.dense_encoder,
                self.streaming_tsblocks_mapping,
                self._tsblock_states_mapping,
            )
            acs_mag, acs_ts_out, vf_acs = self._process_encoder(
                acs_spec,
                self.model.masking.dense_encoder,
                self.streaming_tsblocks_masking,
                self._tsblock_states_masking,
            )
        assert vf_bcs == vf_acs == valid_frames

        self._buffer_features(bcs_ts_out, bcs_mag, valid_frames, self.bcs_feature_buffer)
        self._buffer_features(acs_ts_out, acs_mag, valid_frames, self.acs_feature_buffer)
        self._buffered_frames += valid_frames

        total_needed = self.chunk_size + self.decoder_lookahead
        if self._buffered_frames < total_needed:
            self.bcs_input_buffer = self.bcs_input_buffer[self.output_samples_per_chunk :]
            self.acs_input_buffer = self.acs_input_buffer[self.output_samples_per_chunk :]
            return None

        bcs_ext_feat, bcs_ext_mag = self._extract_extended(self.bcs_feature_buffer)
        acs_ext_feat, acs_ext_mag = self._extract_extended(self.acs_feature_buffer)

        # Decoder + fusion: share one StateFramesContext for all stateful decoder/fusion convs
        with StateFramesContext(None if self.disable_state_guard else self.chunk_size):
            bcs_est_mag, _bcs_est_pha, bcs_com_out = self._decode_branch(
                bcs_ext_feat,
                bcs_ext_mag,
                self.model.mapping.mask_decoder,
                self.model.mapping.phase_decoder,
                infer_type="mapping",
                return_mask=False,
            )
            acs_est_mag, _acs_est_pha, acs_com_out, acs_mask = self._decode_branch(
                acs_ext_feat,
                acs_ext_mag,
                self.model.masking.mask_decoder,
                self.model.masking.phase_decoder,
                infer_type="masking",
                return_mask=True,
            )

            if self.model.use_calibration:
                bcs_com_cal, acs_com_cal = self._apply_calibration_streaming(
                    bcs_com_out,
                    acs_com_out,
                    bcs_est_mag,
                    acs_est_mag,
                    acs_mask,
                )
            else:
                bcs_com_cal, acs_com_cal = bcs_com_out, acs_com_out

            est_mag, est_pha = self._fuse(bcs_com_cal, acs_com_cal, acs_mask)

        # Slide feature buffers (drop consumed chunk_size frames)
        self._slide_feature_buffer(self.bcs_feature_buffer, self.chunk_size)
        removed = self._slide_feature_buffer(self.acs_feature_buffer, self.chunk_size)
        self._buffered_frames -= removed

        valid_output = self._manual_istft_ola(est_mag, est_pha)

        self.bcs_input_buffer = self.bcs_input_buffer[self.output_samples_per_chunk :]
        self.acs_input_buffer = self.acs_input_buffer[self.output_samples_per_chunk :]
        return valid_output

    # ------------------------------------------------------------------
    # Fast pipeline (batched STFT + sequential model + batched iSTFT)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_audio_fast(self, bcs_audio: Tensor, acs_audio: Tensor) -> Tensor:
        """
        Full-audio dual-input pipeline with batched STFT/iSTFT and sequential
        per-chunk model forward. Mirrors :meth:`LaCoSENet.process_audio_fast`.
        """
        from src.models.streaming.utils import StateFramesContext

        if bcs_audio.dim() == 2:
            bcs_audio = bcs_audio.squeeze(0)
        if acs_audio.dim() == 2:
            acs_audio = acs_audio.squeeze(0)
        if bcs_audio.shape != acs_audio.shape:
            raise ValueError(f"bcs/acs audio length mismatch: {bcs_audio.shape} vs {acs_audio.shape}")

        audio_length = len(bcs_audio)
        device = self.device
        bcs_audio = bcs_audio.to(device)
        acs_audio = acs_audio.to(device)

        self.reset_state()

        # --- Phase 1: Batched STFT for both branches ---
        flush_size = self.samples_per_chunk * (self.total_lookahead + 2)
        bcs_padded = torch.cat([bcs_audio, torch.zeros(flush_size, device=device)])
        acs_padded = torch.cat([acs_audio, torch.zeros(flush_size, device=device)])

        osp = self.output_samples_per_chunk
        if self.stft_center:
            context_size = self.win_size // 2
            bcs_pre = torch.cat([torch.zeros(context_size, device=device), bcs_padded])
            acs_pre = torch.cat([torch.zeros(context_size, device=device), acs_padded])
            stft_input_len = context_size + self.samples_per_chunk
        else:
            bcs_pre = bcs_padded
            acs_pre = acs_padded
            stft_input_len = self.samples_per_chunk

        n_calls = (len(bcs_pre) - stft_input_len) // osp + 1

        bcs_batch = bcs_pre.unfold(0, stft_input_len, osp)[:n_calls]
        acs_batch = acs_pre.unfold(0, stft_input_len, osp)[:n_calls]

        _, _, bcs_batch_com = mag_pha_stft(
            bcs_batch,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            center=False,
        )
        _, _, acs_batch_com = mag_pha_stft(
            acs_batch,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            center=False,
        )

        # --- Phase 2: Sequential fused model forward ---
        all_mag: List[Tensor] = []
        all_pha: List[Tensor] = []

        for j in range(n_calls):
            bcs_spec = bcs_batch_com[j : j + 1]
            acs_spec = acs_batch_com[j : j + 1]

            _, _, T_spec, _ = bcs_spec.shape
            valid_frames = min(T_spec, self.chunk_size)

            with StateFramesContext(None if self.disable_state_guard else valid_frames):
                bcs_mag, bcs_ts_out, vf_bcs = self._process_encoder(
                    bcs_spec,
                    self.model.mapping.dense_encoder,
                    self.streaming_tsblocks_mapping,
                    self._tsblock_states_mapping,
                )
                acs_mag, acs_ts_out, _ = self._process_encoder(
                    acs_spec,
                    self.model.masking.dense_encoder,
                    self.streaming_tsblocks_masking,
                    self._tsblock_states_masking,
                )

            self._buffer_features(bcs_ts_out, bcs_mag, vf_bcs, self.bcs_feature_buffer)
            self._buffer_features(acs_ts_out, acs_mag, vf_bcs, self.acs_feature_buffer)
            self._buffered_frames += vf_bcs

            if self._buffered_frames < self.chunk_size + self.decoder_lookahead:
                continue

            bcs_ext_feat, bcs_ext_mag = self._extract_extended(self.bcs_feature_buffer)
            acs_ext_feat, acs_ext_mag = self._extract_extended(self.acs_feature_buffer)

            with StateFramesContext(None if self.disable_state_guard else self.chunk_size):
                bcs_est_mag, _, bcs_com_out = self._decode_branch(
                    bcs_ext_feat,
                    bcs_ext_mag,
                    self.model.mapping.mask_decoder,
                    self.model.mapping.phase_decoder,
                    infer_type="mapping",
                    return_mask=False,
                )
                acs_est_mag, _, acs_com_out, acs_mask = self._decode_branch(
                    acs_ext_feat,
                    acs_ext_mag,
                    self.model.masking.mask_decoder,
                    self.model.masking.phase_decoder,
                    infer_type="masking",
                    return_mask=True,
                )

                if self.model.use_calibration:
                    bcs_com_cal, acs_com_cal = self._apply_calibration_streaming(
                        bcs_com_out,
                        acs_com_out,
                        bcs_est_mag,
                        acs_est_mag,
                        acs_mask,
                    )
                else:
                    bcs_com_cal, acs_com_cal = bcs_com_out, acs_com_out

                est_mag, est_pha = self._fuse(bcs_com_cal, acs_com_cal, acs_mask)

            self._slide_feature_buffer(self.bcs_feature_buffer, self.chunk_size)
            removed = self._slide_feature_buffer(self.acs_feature_buffer, self.chunk_size)
            self._buffered_frames -= removed

            all_mag.append(est_mag)
            all_pha.append(est_pha)

        if not all_mag:
            return torch.tensor([], device=device)

        # --- Phase 3: Batched iSTFT (stateless OLA) ---
        cat_mag = torch.cat(all_mag, dim=2)
        cat_pha = torch.cat(all_pha, dim=2)

        output, _, _ = manual_istft_ola(
            cat_mag,
            cat_pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            ola_buffer=None,
            ola_norm=None,
        )

        if len(output) > audio_length:
            output = output[:audio_length]
        return output

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use BAFNetPlusStreaming.process_audio_fast(bcs, acs) or .process_samples(bcs, acs).")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  ablation_mode={self._streaming_config.get('ablation_mode', '?')},\n"
            f"  chunk_size={self.chunk_size},\n"
            f"  encoder_lookahead={self.encoder_lookahead},\n"
            f"  decoder_lookahead={self.decoder_lookahead},\n"
            f"  total_lookahead={self.total_lookahead},\n"
            f"  latency_ms={self.latency_ms:.2f},\n"
            f"  tsblocks_mapping={len(self.streaming_tsblocks_mapping)},\n"
            f"  tsblocks_masking={len(self.streaming_tsblocks_masking)},\n"
            f")"
        )
