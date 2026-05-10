"""Shared helpers for runtime entry points."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

import torch
from datasets import concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data import Noise_Augmented_Dataset
from src.checkpoint import load_checkpoint, load_model
from src.stft import mag_pha_stft
from src.utils import get_stft_args_from_config, parse_file_list


DATASET_LANGUAGE_MAP = {
    "taps": "korean",
    "vibravox": "french",
}

DATASET_SPLITS = {
    "taps": "yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset",
    "vibravox": "yskim3271/vibravox_16k",
}


@dataclass
class RuntimeBundle:
    """Prepared objects needed by runtime entry points."""

    conf: Any
    model: torch.nn.Module
    model_args: Any
    data_loader_list: Dict[str, DataLoader]
    stft_args: Dict[str, Any]
    bcs_only: bool
    stt_language: str


EvaluationRuntime = RuntimeBundle


def resolve_stt_language(
    dataset: str,
    requested_language: str | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """Return explicit STT language or dataset-based default."""
    if requested_language:
        if logger:
            logger.info(f"Using user-specified STT language: {requested_language}")
        return requested_language

    language = DATASET_LANGUAGE_MAP.get(dataset.lower(), "korean")
    if logger:
        logger.info(f"Auto-detected STT language from dataset: {language}")
    return language


def load_test_dataset(dataset: str, test_augment_numb: int, *, hf_id: str | None = None,
                      config: str | None = None):
    """Load and repeat the requested public test split.

    ``hf_id`` and ``config`` override the default ``DATASET_SPLITS`` lookup —
    used by Vibravox-native single-cell test (Phase 2) to point at a different
    multi-config subset (e.g., ``yskim3271/vibravox_16k``,
    ``config="speech_noisy"``).
    """
    if hf_id is None:
        dataset_key = dataset.lower()
        if dataset_key not in DATASET_SPLITS:
            raise ValueError(f"Unknown dataset: {dataset}")
        hf_id = DATASET_SPLITS[dataset_key]

    if config is not None:
        testset = load_dataset(hf_id, name=config, split="test")
    else:
        testset = load_dataset(hf_id, split="test")
    return concatenate_datasets([testset] * test_augment_numb)


def get_eval_stft_args(conf: Any) -> Dict[str, Any]:
    """Build STFT args from model config with root-level overrides."""
    stft_args = get_stft_args_from_config(conf.model)
    if hasattr(conf, "compress_factor"):
        stft_args["compress_factor"] = conf.compress_factor
    return stft_args


def build_eval_loaders(args: Any, bcs_only: bool) -> Dict[str, DataLoader]:
    """Build evaluation loaders.

    Two modes:
    - **multi-SNR (default)**: one loader per ``args.snr_step`` value, deterministic
      noise/RIR mixing via ``Noise_Augmented_Dataset(snr_range=[snr,snr],
      deterministic=True)``. Loader keys are ``"-20"``, ``"0"``, etc.
    - **single-cell vibravox-native** (``args.test_bypass_mixing=True``): one
      loader at key ``"native"`` consumes the paired real-recorded
      ``speech_noisy`` subset; mixing is bypassed in
      :class:`Noise_Augmented_Dataset` (Phase 2).
    """
    test_bypass_mixing = bool(getattr(args, "test_bypass_mixing", False))

    if test_bypass_mixing:
        testset = load_test_dataset(
            args.dataset,
            args.test_augment_numb,
            hf_id=getattr(args, "test_hf_dataset", None),
            config=getattr(args, "test_subset", None),
        )
        dataset = Noise_Augmented_Dataset(
            datapair_list=testset,
            noise_list=[],
            rir_list=[],
            snr_range=[0, 0],
            reverb_proportion=0.0,
            target_dB_FS=args.target_dB_FS,
            target_dB_FS_floating_value=args.target_dB_FS_floating_value,
            silence_length=args.silence_length,
            deterministic=True,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
            bcs_only=bcs_only,
            mix_strategy="vibravox_native",
            bypass_mixing=True,
        )
        return {
            "native": DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        }

    testset = load_test_dataset(args.dataset, args.test_augment_numb)

    if bcs_only:
        noise_test_list, rir_test_list = [], []
    else:
        noise_test_list = parse_file_list(args.noise_dir, args.noise_test)
        rir_test_list = parse_file_list(args.rir_dir, args.rir_test)

    loaders: Dict[str, DataLoader] = {}
    for fixed_snr in args.snr_step:
        dataset = Noise_Augmented_Dataset(
            datapair_list=testset,
            noise_list=noise_test_list,
            rir_list=rir_test_list,
            snr_range=[fixed_snr, fixed_snr],
            reverb_proportion=args.reverb_proportion,
            target_dB_FS=args.target_dB_FS,
            target_dB_FS_floating_value=args.target_dB_FS_floating_value,
            silence_length=args.silence_length,
            deterministic=True,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
            bcs_only=bcs_only,
        )
        loaders[f"{fixed_snr}"] = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    return loaders


def prepare_evaluation_runtime(
    args: Any,
    logger: logging.Logger | None = None,
    *,
    eval_stt: bool | None = None,
    save_per_utterance: bool | None = None,
) -> RuntimeBundle:
    """Load config, model, STFT args, and evaluation loaders for a CLI namespace."""
    conf = OmegaConf.load(args.model_config)
    conf.device = args.device
    conf.eval_stt = bool(getattr(args, "eval_stt", False) if eval_stt is None else eval_stt)
    conf.bcs_gain_db = float(getattr(args, "bcs_gain_db", 0.0))
    conf.acs_gain_db = float(getattr(args, "acs_gain_db", 0.0))
    conf.save_per_utterance = bool(
        getattr(args, "save_per_utterance_json", False)
        if save_per_utterance is None
        else save_per_utterance
    )
    conf.test_bypass_mixing = bool(getattr(args, "test_bypass_mixing", False))
    conf.test_hf_dataset = str(getattr(args, "test_hf_dataset", "") or "")
    conf.test_subset = str(getattr(args, "test_subset", "") or "")

    stt_language = resolve_stt_language(
        args.dataset,
        getattr(args, "stt_language", None),
        logger,
    )
    conf.stt_language = stt_language

    model_args = conf.model
    model = load_model(
        model_args.model_lib,
        model_args.model_class,
        model_args.param,
        args.device,
    )
    model = load_checkpoint(model, args.chkpt_dir, args.chkpt_file, args.device)

    bcs_only = "acs" not in model_args.input_type
    stft_args = get_eval_stft_args(conf)
    data_loader_list = build_eval_loaders(args, bcs_only)

    return RuntimeBundle(
        conf=conf,
        model=model,
        model_args=model_args,
        data_loader_list=data_loader_list,
        stft_args=stft_args,
        bcs_only=bcs_only,
        stt_language=stt_language,
    )


def prepare_enhancement_runtime(
    args: Any,
    logger: logging.Logger | None = None,
) -> RuntimeBundle:
    """Prepare config, model, loader, and STFT args for one-SNR enhancement."""
    runtime_args = SimpleNamespace(**vars(args))
    runtime_args.model_config = str(Path(args.chkpt_dir) / ".hydra" / "config.yaml")
    runtime_args.snr_step = [args.snr]
    runtime_args.reverb_proportion = args.reverb_prop
    runtime_args.target_dB_FS_floating_value = getattr(args, "target_dB_FS_floating_value", 0)
    runtime_args.silence_length = getattr(args, "silence_length", 0.2)
    runtime_args.test_augment_numb = getattr(args, "test_augment_numb", 1)
    runtime_args.num_workers = getattr(args, "num_workers", 0)
    runtime_args.eval_stt = False
    runtime_args.save_per_utterance_json = False
    runtime_args.bcs_gain_db = 0.0
    runtime_args.acs_gain_db = 0.0

    return prepare_evaluation_runtime(
        runtime_args,
        logger=None,
        eval_stt=False,
        save_per_utterance=False,
    )


def build_evaluation_output(metrics: Any, args: Any, runtime: RuntimeBundle) -> Dict[str, Any]:
    """Wrap aggregate evaluation metrics with reproducibility metadata."""
    test_bypass_mixing = bool(getattr(args, "test_bypass_mixing", False))
    snr_step_raw = getattr(args, "snr_step", []) or []
    if test_bypass_mixing:
        snr_step: List[Any] = ["native"]
    else:
        snr_step = [int(snr) for snr in snr_step_raw]

    metadata = {
        "dataset": str(getattr(args, "dataset", "")),
        "model_config": str(getattr(args, "model_config", "")),
        "chkpt_dir": str(getattr(args, "chkpt_dir", "")),
        "chkpt_file": str(getattr(args, "chkpt_file", "")),
        "snr_step": snr_step,
        "noise_dir": str(getattr(args, "noise_dir", "")),
        "noise_test": str(getattr(args, "noise_test", "")),
        "rir_dir": str(getattr(args, "rir_dir", "")),
        "rir_test": str(getattr(args, "rir_test", "")),
        "test_augment_numb": int(getattr(args, "test_augment_numb", 0)),
        "reverb_proportion": float(getattr(args, "reverb_proportion", 0.0)),
        "target_dB_FS": float(getattr(args, "target_dB_FS", 0.0)),
        "target_dB_FS_floating_value": float(getattr(args, "target_dB_FS_floating_value", 0.0)),
        "silence_length": float(getattr(args, "silence_length", 0.0)),
        "bcs_gain_db": float(getattr(args, "bcs_gain_db", 0.0)),
        "acs_gain_db": float(getattr(args, "acs_gain_db", 0.0)),
        "eval_stt": bool(getattr(args, "eval_stt", False)),
        "stt_language": runtime.stt_language,
        "save_per_utterance": bool(getattr(runtime.conf, "save_per_utterance", False)),
        "test_bypass_mixing": test_bypass_mixing,
        "test_hf_dataset": str(getattr(args, "test_hf_dataset", "") or ""),
        "test_subset": str(getattr(args, "test_subset", "") or ""),
    }
    return {
        "metadata": metadata,
        "metrics": metrics,
    }


def write_json(data: Any, output_path: str | Path, logger: logging.Logger | None = None) -> Path:
    """Write JSON data and return the normalized output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    if logger:
        logger.info(f"Results saved to {path}")
    return path


ModelInput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def get_model_input(
    bcs: torch.Tensor,
    noisy_acs: torch.Tensor,
    input_type: str,
    device: torch.device,
    stft_args: Dict[str, Any],
) -> ModelInput:
    """Build the model input tensor(s) for a given input modality.

    Centralizes the ``acs`` / ``bcs`` / ``acs+bcs`` dispatch so train / evaluate
    / enhance paths share one source of truth.

    Returns:
        Single complex spectrogram for ``acs`` / ``bcs``; a ``(bcs, acs)`` tuple
        for ``acs+bcs``.

    Raises:
        ValueError: If ``input_type`` is not recognized.
    """
    if input_type == "acs":
        return mag_pha_stft(noisy_acs, **stft_args)[2].to(device)
    if input_type == "bcs":
        return mag_pha_stft(bcs, **stft_args)[2].to(device)
    if input_type == "acs+bcs":
        return (
            mag_pha_stft(bcs, **stft_args)[2].to(device),
            mag_pha_stft(noisy_acs, **stft_args)[2].to(device),
        )
    raise ValueError(f"Invalid model input type: {input_type}")


def add_runtime_common_args(
    parser: argparse.ArgumentParser,
    *,
    require_model_config: bool = True,
) -> None:
    """Add checkpoint / dataset / device flags shared by evaluation-style CLIs.

    Used by ``src.evaluate``, ``src.analysis.forward_traces`` and any CLI that
    loads a trained checkpoint against a public test split.
    """
    if require_model_config:
        parser.add_argument("--model_config", type=str, required=True,
                            help="Path to the model config file.")
    parser.add_argument("--chkpt_dir", type=str, required=True,
                        help="Path to the checkpoint directory.")
    parser.add_argument("--chkpt_file", type=str, default="best.th",
                        help="Checkpoint file name. default is best.th")
    parser.add_argument("--dataset", type=str, default="taps",
                        choices=["taps", "vibravox"],
                        help="Dataset to use: taps or vibravox.")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu).",
    )
    parser.add_argument("--num_workers", type=int, default=5,
                        help="DataLoader workers. default is 5")


def add_eval_augmentation_args(parser: argparse.ArgumentParser) -> None:
    """Add noise/RIR/SNR augmentation flags shared by multi-SNR eval CLIs.

    All flags are optional at parse time so Vibravox-native single-cell mode
    (``--test_bypass_mixing``) can omit noise/RIR/SNR entirely. Multi-SNR mode
    enforces the required-flag check in
    :func:`validate_eval_augmentation_args` after parsing.
    """
    parser.add_argument("--noise_dir", type=str, default=None,
                        help="Path to the noise directory (multi-SNR only).")
    parser.add_argument("--noise_test", type=str, default=None,
                        help="List of noise files for testing (multi-SNR only).")
    parser.add_argument("--rir_dir", type=str, default=None,
                        help="Path to the RIR directory (multi-SNR only).")
    parser.add_argument("--rir_test", type=str, default=None,
                        help="List of RIR files for testing (multi-SNR only).")
    parser.add_argument("--snr_step", nargs="+", type=int, default=None,
                        help="One or more SNR values in dB to evaluate (multi-SNR only).")
    parser.add_argument("--test_augment_numb", type=int, default=2,
                        help="Number of test augmentations. default is 2")
    parser.add_argument("--reverb_proportion", type=float, default=0.0,
                        help="Reverberation proportion. default is 0.0")
    parser.add_argument("--target_dB_FS", type=float, default=-25,
                        help="Target dB FS. default is -25")
    parser.add_argument("--target_dB_FS_floating_value", type=float, default=0,
                        help="Target dB FS floating value. default is 0")
    parser.add_argument("--silence_length", type=float, default=0.2,
                        help="Silence length. default is 0.2")
    parser.add_argument("--test_bypass_mixing", action="store_true",
                        help="Vibravox-native single-cell test: load paired real-recorded "
                             "speech_noisy directly, bypass synthetic noise mixing. "
                             "Output JSON uses 'native' SNR key.")
    parser.add_argument("--test_hf_dataset", type=str, default=None,
                        help="HF dataset id for --test_bypass_mixing (e.g., yskim3271/vibravox_16k).")
    parser.add_argument("--test_subset", type=str, default=None,
                        help="HF config name for --test_bypass_mixing (e.g., speech_noisy).")


def validate_eval_augmentation_args(args: argparse.Namespace) -> None:
    """Enforce required flags depending on eval mode (multi-SNR vs vibravox-native).

    Multi-SNR mode requires ``--noise_dir`` / ``--noise_test`` / ``--rir_dir`` /
    ``--rir_test`` / ``--snr_step``. Vibravox-native (``--test_bypass_mixing``)
    requires ``--test_hf_dataset`` / ``--test_subset``.
    """
    if getattr(args, "test_bypass_mixing", False):
        missing = [
            name for name in ("test_hf_dataset", "test_subset")
            if not getattr(args, name, None)
        ]
        if missing:
            raise SystemExit(
                f"--test_bypass_mixing requires the following flags: {missing}"
            )
        return

    multi_snr_required = ("noise_dir", "noise_test", "rir_dir", "rir_test", "snr_step")
    missing = [name for name in multi_snr_required if not getattr(args, name, None)]
    if missing:
        raise SystemExit(
            f"Multi-SNR eval requires the following flags: {missing}. "
            f"Use --test_bypass_mixing for Vibravox-native single-cell eval instead."
        )
