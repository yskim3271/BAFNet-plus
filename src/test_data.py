"""Unit tests for ``Noise_Augmented_Dataset`` mixing strategies.

Covers:
- ``simple_add_mix`` length matching (longer / shorter noise)
- ``simple_add_mix`` no-rescale (RMS ratio preserved up to mix)
- vibravox_native pipeline smoke (mock HF dataset, skip_db_fs_normalize=True)
- bypass_mixing test path (Vibravox speech_noisy single-cell)
- snr_random regression (default path returns same length tuple as before)
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import Noise_Augmented_Dataset


# ----------------------------------------------------------------------------
# Mock HF-style row helpers
# ----------------------------------------------------------------------------

def _make_speech_pair_row(speaker_id: str, sentence_id: str, length: int, seed: int = 0):
    """Mimic yskim3271/vibravox_16k row shape."""
    rng = np.random.default_rng(seed)
    return {
        "audio.throat_microphone": {
            "array": rng.standard_normal(length, dtype=np.float32) * 0.1,
            "sampling_rate": 16000,
        },
        "audio.acoustic_microphone": {
            "array": rng.standard_normal(length, dtype=np.float32) * 0.3,
            "sampling_rate": 16000,
        },
        "speaker_id": speaker_id,
        "sentence_id": sentence_id,
        "text": "test sentence",
    }


class _MockHFDataset:
    """Minimal HF-Dataset stand-in: indexable + len()."""

    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


# ----------------------------------------------------------------------------
# simple_add_mix
# ----------------------------------------------------------------------------

class TestSimpleAddMix:
    def test_length_match_when_noise_longer(self):
        clean = torch.zeros(16000)
        noise = torch.ones(64000)
        out = Noise_Augmented_Dataset.simple_add_mix(clean, noise)
        assert out.shape == clean.shape

    def test_length_match_when_noise_shorter(self):
        clean = torch.zeros(16000)
        noise = torch.ones(4000)
        out = Noise_Augmented_Dataset.simple_add_mix(clean, noise)
        assert out.shape == clean.shape

    def test_length_match_when_equal(self):
        clean = torch.zeros(16000)
        noise = torch.ones(16000)
        out = Noise_Augmented_Dataset.simple_add_mix(clean, noise)
        assert out.shape == clean.shape

    def test_no_rescale_zero_clean(self):
        """When clean is zero, output equals (cropped/repeated) noise — no scaling."""
        random.seed(42)
        clean = torch.zeros(16000)
        noise = torch.ones(16000) * 0.5
        out = Noise_Augmented_Dataset.simple_add_mix(clean, noise)
        torch.testing.assert_close(out, torch.ones(16000) * 0.5)

    def test_no_rescale_known_signals(self):
        """Verify ``out = clean + noise_aligned`` at every sample."""
        random.seed(0)
        clean = torch.full((1000,), 0.3)
        noise = torch.full((1000,), 0.2)
        out = Noise_Augmented_Dataset.simple_add_mix(clean, noise)
        torch.testing.assert_close(out, torch.full((1000,), 0.5))


# ----------------------------------------------------------------------------
# vibravox_native pipeline smoke
# ----------------------------------------------------------------------------

class TestVibravoxNativePipeline:
    def test_returns_paired_tuple(self):
        speech_rows = [_make_speech_pair_row("p0", "u0", 64000, seed=1)]
        noise_rows = [_make_speech_pair_row("noise", "n0", 32000, seed=2)]
        noise_ds = _MockHFDataset(noise_rows)

        ds = Noise_Augmented_Dataset(
            datapair_list=speech_rows,
            noise_list=[],
            rir_list=[],
            snr_range=[0, 0],
            reverb_proportion=0.0,
            target_dB_FS=-25,
            target_dB_FS_floating_value=0,
            silence_length=0.2,
            sampling_rate=16000,
            segment=64000,
            stride=32000,
            mix_strategy="vibravox_native",
            noise_dataset=noise_ds,
            noise_sensor="throat_microphone",
            skip_db_fs_normalize=True,
        )

        assert len(ds) >= 1
        bcs, noisy_acs, clean_acs = ds[0]

        assert isinstance(bcs, torch.Tensor)
        assert bcs.shape == noisy_acs.shape == clean_acs.shape
        assert bcs.shape[-1] == 64000
        # Vibravox-native: no dB FS normalization → raw float32 magnitudes
        # (sanity: not all zero, not infinite)
        assert torch.isfinite(bcs).all()
        assert torch.isfinite(noisy_acs).all()
        assert torch.isfinite(clean_acs).all()

    def test_init_requires_noise_dataset(self):
        with pytest.raises(ValueError, match="vibravox_native"):
            Noise_Augmented_Dataset(
                datapair_list=[_make_speech_pair_row("p0", "u0", 16000)],
                noise_list=[],
                rir_list=[],
                snr_range=[0, 0],
                reverb_proportion=0.0,
                target_dB_FS=-25,
                target_dB_FS_floating_value=0,
                silence_length=0.2,
                mix_strategy="vibravox_native",
                noise_dataset=None,
                bypass_mixing=False,
            )


# ----------------------------------------------------------------------------
# bypass_mixing (Phase 2 single-cell test)
# ----------------------------------------------------------------------------

class TestBypassMixing:
    def test_noisy_acs_equals_clean_acs(self):
        rows = [_make_speech_pair_row("p0", "u0", 32000, seed=3)]

        ds = Noise_Augmented_Dataset(
            datapair_list=rows,
            noise_list=[],
            rir_list=[],
            snr_range=[0, 0],
            reverb_proportion=0.0,
            target_dB_FS=-25,
            target_dB_FS_floating_value=0,
            silence_length=0.2,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
            mix_strategy="vibravox_native",
            bypass_mixing=True,
            bcs_gain_perturbation_db=0,  # disable gain perturb for exact equality
        )

        bcs, noisy_acs, clean_acs, uid, text = ds[0]
        assert noisy_acs.shape == clean_acs.shape
        # noisy_acs is a clone of clean_acs; gain perturb only affects bcs
        torch.testing.assert_close(noisy_acs, clean_acs)
        assert uid == "p0_u0"
        assert text == "test sentence"


# ----------------------------------------------------------------------------
# snr_random regression (TAPS path unchanged)
# ----------------------------------------------------------------------------

class TestSnrRandomRegression:
    def test_default_mix_strategy_is_snr_random(self):
        ds = Noise_Augmented_Dataset(
            datapair_list=[_make_speech_pair_row("p0", "u0", 16000)],
            noise_list=[],
            rir_list=[],
            snr_range=[0, 0],
            reverb_proportion=0.0,
            target_dB_FS=-25,
            target_dB_FS_floating_value=0,
            silence_length=0.2,
            bcs_only=True,  # avoid needing real noise files
        )
        assert ds.mix_strategy == "snr_random"

    def test_invalid_mix_strategy_raises(self):
        with pytest.raises(ValueError, match="Unsupported mix_strategy"):
            Noise_Augmented_Dataset(
                datapair_list=[_make_speech_pair_row("p0", "u0", 16000)],
                noise_list=[],
                rir_list=[],
                snr_range=[0, 0],
                reverb_proportion=0.0,
                target_dB_FS=-25,
                target_dB_FS_floating_value=0,
                silence_length=0.2,
                mix_strategy="bogus",
                bcs_only=True,
            )

    def test_bcs_only_path_unchanged(self):
        """Default snr_random + bcs_only returns dummy_acs zeros (regression)."""
        rows = [_make_speech_pair_row("p0", "u0", 16000, seed=7)]
        ds = Noise_Augmented_Dataset(
            datapair_list=rows,
            noise_list=[],
            rir_list=[],
            snr_range=[0, 0],
            reverb_proportion=0.0,
            target_dB_FS=-25,
            target_dB_FS_floating_value=0,
            silence_length=0.2,
            sampling_rate=16000,
            bcs_only=True,
        )
        bcs, dummy_acs, clean_acs = ds[0]
        torch.testing.assert_close(dummy_acs, torch.zeros_like(dummy_acs))
        assert bcs.shape == clean_acs.shape == dummy_acs.shape
