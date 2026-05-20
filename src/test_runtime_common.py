"""Unit tests for ``runtime_common`` evaluation-loader helpers.

Covers:
- ``parse_snr_keys`` "native" sentinel (Vibravox-native single-cell)
- ``build_evaluation_output`` metadata snr_step formatting in both modes
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime_common import RuntimeBundle, build_evaluation_output
from src.analysis.wrap_eval_json import parse_snr_keys


class TestParseSnrKeys:
    def test_multi_snr(self):
        metrics = {"-20dB": {}, "0dB": {}, "10dB": {}}
        assert parse_snr_keys(metrics) == [-20, 0, 10]

    def test_native_only(self):
        metrics = {"native": {}}
        assert parse_snr_keys(metrics) == ["native"]

    def test_mixed(self):
        metrics = {"-20dB": {}, "0dB": {}, "native": {}}
        # ints sort first, "native" last
        assert parse_snr_keys(metrics) == [-20, 0, "native"]

    def test_unknown_keys_skipped(self):
        metrics = {"per_utterance_summary": {}, "0dB": {}}
        assert parse_snr_keys(metrics) == [0]


class TestBuildEvaluationOutput:
    def _make_runtime(self, save_per_utterance: bool = False) -> RuntimeBundle:
        conf = SimpleNamespace(save_per_utterance=save_per_utterance)
        return RuntimeBundle(
            conf=conf,
            model=None,
            model_args=None,
            data_loader_list={},
            stft_args={},
            bcs_only=False,
            stt_language="korean",
        )

    def test_multi_snr_metadata(self):
        args = SimpleNamespace(
            dataset="taps",
            model_config="conf.yaml",
            chkpt_dir="ckpt",
            chkpt_file="best.th",
            snr_step=[-20, 0, 10],
            noise_dir="/n",
            noise_test="n.txt",
            rir_dir="/r",
            rir_test="r.txt",
            test_augment_numb=2,
            reverb_proportion=0.0,
            target_dB_FS=-25.0,
            target_dB_FS_floating_value=0.0,
            silence_length=0.2,
            bcs_gain_db=0.0,
            acs_gain_db=0.0,
            eval_stt=False,
        )
        out = build_evaluation_output(metrics={}, args=args, runtime=self._make_runtime())
        md = out["metadata"]
        assert md["snr_step"] == [-20, 0, 10]
        assert md["test_bypass_mixing"] is False
        assert md["test_hf_dataset"] == ""
        assert md["test_subset"] == ""

    def test_vibravox_native_metadata(self):
        args = SimpleNamespace(
            dataset="vibravox",
            model_config="conf.yaml",
            chkpt_dir="ckpt",
            chkpt_file="best.th",
            snr_step=[],
            test_augment_numb=1,
            reverb_proportion=0.0,
            target_dB_FS=-25.0,
            target_dB_FS_floating_value=0.0,
            silence_length=0.2,
            bcs_gain_db=0.0,
            acs_gain_db=0.0,
            eval_stt=False,
            test_bypass_mixing=True,
            test_hf_dataset="yskim3271/vibravox_16k",
            test_subset="speech_noisy",
        )
        out = build_evaluation_output(metrics={}, args=args, runtime=self._make_runtime())
        md = out["metadata"]
        assert md["snr_step"] == ["native"]
        assert md["test_bypass_mixing"] is True
        assert md["test_hf_dataset"] == "yskim3271/vibravox_16k"
        assert md["test_subset"] == "speech_noisy"
