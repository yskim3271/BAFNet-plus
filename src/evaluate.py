# Copyright (c) POSTECH, and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: yunsik kim
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import nlptutti as sarmetric
import logging
from typing import Dict, Optional, Any, List, Tuple
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.compute_metrics import compute_metrics
from src.runtime_common import build_evaluation_output, prepare_evaluation_runtime, write_json
from src.stft import mag_pha_stft, mag_pha_istft
from src.utils import bold, LogProgress


class STTEvaluator:
    """
    Speech-to-Text (STT) evaluator

    Loads Whisper model once and reuses it for multiple evaluations

    Example:
        >>> evaluator = STTEvaluator(device="cuda")
        >>> cer1, wer1 = evaluator.evaluate(samples1)
        >>> cer2, wer2 = evaluator.evaluate(samples2)  # No model reload
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str = "cuda",
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model_id: HuggingFace model ID
            device: Device (cuda/cpu)
            logger: Logger instance
        """
        self.model_id = model_id
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Load model once
        self.pipe = self._load_model()

        self.logger.info(f"✅ STTEvaluator initialized with {model_id}")

    def _load_model(self):
        """Load Whisper model"""
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.logger.info(f"Loading STT model: {self.model_id}")
        self.logger.info(f"Using dtype: {torch_dtype}")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            torch_dtype=torch_dtype,
            device=self.device,
        )

        return pipe

    def evaluate(
        self,
        enhanced: List[Tuple[np.ndarray, str]],
        language: str = "korean"
    ) -> Tuple[float, float]:
        """
        Perform STT evaluation

        Args:
            enhanced: List of (audio, reference_text) tuples
            language: Language hint (korean, english, etc.)

        Returns:
            (cer, wer) tuple

        Example:
            >>> samples = [(audio_array, "안녕하세요"), ...]
            >>> cer, wer = evaluator.evaluate(samples, language="korean")
            >>> print(f"CER: {cer:.4f}, WER: {wer:.4f}")
        """
        cer, wer, _, _ = self._evaluate_internal(enhanced, language)
        return cer, wer

    def evaluate_with_per_utterance(
        self,
        enhanced: List[Tuple[np.ndarray, str]],
        language: str = "korean"
    ) -> Tuple[float, float, List[float], List[float]]:
        """Same as :meth:`evaluate` but also returns per-utterance CER/WER lists.

        Used by the ``--save_per_utterance_json`` path to populate the
        per-utterance JSON block.
        """
        return self._evaluate_internal(enhanced, language)

    def _evaluate_internal(
        self,
        enhanced: List[Tuple[np.ndarray, str]],
        language: str,
    ) -> Tuple[float, float, List[float], List[float]]:
        if not enhanced:
            self.logger.warning("No samples to evaluate")
            return 0.0, 0.0, [], []

        self.logger.info(f"Evaluating {len(enhanced)} samples with STT...")

        cer_list: List[float] = []
        wer_list: List[float] = []

        for wav, reference_text in enhanced:
            with torch.no_grad():
                transcription = self.pipe(
                    wav,
                    generate_kwargs={
                        "language": language,
                        "num_beams": 1,
                        "max_length": 100
                    }
                )['text']

            cer_list.append(sarmetric.get_cer(
                reference_text, transcription, rm_punctuation=True
            )['cer'])
            wer_list.append(sarmetric.get_wer(
                reference_text, transcription, rm_punctuation=True
            )['wer'])

        cer = sum(cer_list) / len(cer_list)
        wer = sum(wer_list) / len(wer_list)

        self.logger.info(f"STT Evaluation complete: CER={cer:.4f}, WER={wer:.4f}")

        return cer, wer, cer_list, wer_list


def evaluate(
    args: DictConfig,
    model: torch.nn.Module,
    data_loader_list: Dict[str, DataLoader],
    logger: logging.Logger,
    epoch: Optional[int] = None,
    stft_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Model evaluation

    Args:
        args: Evaluation settings (OmegaConf)
        model: Model to evaluate
        data_loader_list: Dictionary of dataloaders by SNR {"0": loader, "5": loader, ...}
        logger: Logger instance
        epoch: Current epoch (during training)
        stft_args: STFT parameters

    Returns:
        Dictionary of metrics by SNR
        {
            "0dB": {"pesq": 2.5, "stoi": 0.85, "cer": 0.02, "wer": 0.05, ...},
            "5dB": {"pesq": 2.8, "stoi": 0.90, "cer": 0.01, "wer": 0.03, ...}
        }
    """
    prefix = f"Epoch {epoch+1}, " if epoch is not None else ""

    metrics = {}
    model.eval()

    # Initialize STT evaluator once (reuse across SNRs)
    stt_evaluator = None
    if args.eval_stt:
        stt_evaluator = STTEvaluator(
            model_id="openai/whisper-large-v3",
            device=args.device,
            logger=logger
        )

    save_per_utt = bool(getattr(args, 'save_per_utterance', False))

    for snr, data_loader in data_loader_list.items():
        iterator = LogProgress(logger, data_loader, name=f"Evaluate on {snr}dB")
        enhanced = []
        results  = []
        utt_ids: List[str] = [] if save_per_utt else []
        bcs_gain_db = getattr(args, 'bcs_gain_db', 0.0)
        acs_gain_db = getattr(args, 'acs_gain_db', 0.0)
        bcs_scalar = 10 ** (bcs_gain_db / 20) if bcs_gain_db != 0.0 else 1.0
        acs_scalar = 10 ** (acs_gain_db / 20) if acs_gain_db != 0.0 else 1.0

        with torch.no_grad():
            for data in iterator:
                bcs, noisy_acs, clean_acs, utt_id, text = data

                # Apply independent gain perturbation to BCS and ACS
                if bcs_scalar != 1.0:
                    bcs = bcs * bcs_scalar
                if acs_scalar != 1.0:
                    noisy_acs = noisy_acs * acs_scalar
                    clean_acs = clean_acs * acs_scalar

                if args.model.input_type == "acs":
                    input = mag_pha_stft(noisy_acs, **stft_args)[2].to(args.device)
                elif args.model.input_type == "bcs":
                    input = mag_pha_stft(bcs, **stft_args)[2].to(args.device)
                elif args.model.input_type == "acs+bcs":
                    input = mag_pha_stft(bcs, **stft_args)[2].to(args.device), mag_pha_stft(noisy_acs, **stft_args)[2].to(args.device)

                clean_mag_hat, clean_pha_hat, _ = model(input)

                clean_acs_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)

                clean_acs = clean_acs.squeeze().detach().cpu().numpy()
                clean_acs_hat = clean_acs_hat.squeeze().detach().cpu().numpy()
                if len(clean_acs) != len(clean_acs_hat):
                    length = min(len(clean_acs), len(clean_acs_hat))
                    clean_acs = clean_acs[0:length]
                    clean_acs_hat = clean_acs_hat[0:length]

                enhanced.append((clean_acs_hat, text[0]))
                results.append(compute_metrics(clean_acs, clean_acs_hat))
                if save_per_utt:
                    utt_ids.append(utt_id[0])

        pesq, csig, cbak, covl, segSNR, stoi = np.mean(results, axis=0)
        snr_key = f'{snr}dB'
        metrics[snr_key] = {
            "pesq": pesq,
            "stoi": stoi,
            "csig": csig,
            "cbak": cbak,
            "covl": covl,
            "segSNR": segSNR
        }
        logger.info(bold(f"{prefix}Performance on {snr}dB: PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}"))

        # STT evaluation (model reused!)
        per_utt_cer: Optional[List[float]] = None
        per_utt_wer: Optional[List[float]] = None
        if stt_evaluator:
            language = getattr(args, 'stt_language', 'korean')
            if save_per_utt:
                cer, wer, per_utt_cer, per_utt_wer = stt_evaluator.evaluate_with_per_utterance(enhanced, language=language)
            else:
                cer, wer = stt_evaluator.evaluate(enhanced, language=language)
            metrics[snr_key]['cer'] = cer
            metrics[snr_key]['wer'] = wer
            logger.info(bold(f"{prefix}Performance on {snr}dB: CER={cer:.4f}, WER={wer:.4f}"))

        # Per-utterance block for downstream statistical analysis.
        if save_per_utt:
            per_utt_block: Dict[str, Dict[str, float]] = {}
            for i, uid in enumerate(utt_ids):
                rec = {
                    "pesq": float(results[i][0]),
                    "csig": float(results[i][1]),
                    "cbak": float(results[i][2]),
                    "covl": float(results[i][3]),
                    "segSNR": float(results[i][4]),
                    "stoi": float(results[i][5]),
                }
                if per_utt_cer is not None:
                    rec["cer"] = float(per_utt_cer[i])
                    rec["wer"] = float(per_utt_wer[i])
                per_utt_block[uid] = rec
            metrics[snr_key]["per_utterance"] = per_utt_block

    return metrics



if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--chkpt_dir", type=str, default='.', help="Path to the checkpoint directory. default is current directory")
    parser.add_argument("--chkpt_file", type=str, default="best.th", help="Checkpoint file name. default is best.th")
    parser.add_argument("--dataset", type=str, default="taps", choices=["taps", "vibravox"], help="Dataset to use: taps or vibravox. default is taps")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to the noise directory.")
    parser.add_argument("--noise_test", type=str, required=True, help="List of noise files for testing.")
    parser.add_argument("--rir_dir", type=str, required=True, help="Path to the RIR directory.")
    parser.add_argument("--rir_test", type=str, required=True, help="List of RIR files for testing.")
    parser.add_argument("--test_augment_numb", type=int, default=2, help="Number of test augmentations. default is 2")
    parser.add_argument("--snr_step", nargs="+", type=int, required=True, help="Signal to noise ratio. default is 0 dB")
    parser.add_argument("--reverb_proportion", type=float, default=0.0, help="Reverberation proportion. default is 0.5")
    parser.add_argument("--target_dB_FS", type=float, default=-25, help="Target dB FS. default is -25")
    parser.add_argument("--target_dB_FS_floating_value", type=float, default=0, help="Target dB FS floating value. default is 0")
    parser.add_argument("--silence_length", type=float, default=0.2, help="Silence length. default is 0.2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Specifies the device (cuda or cpu).")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers. default is 5")
    parser.add_argument("--log_file", type=str, default="output.log", help="Log file name. default is output.log")
    parser.add_argument("--eval_stt", default=False, action="store_true", help="Evaluate STT performance")
    parser.add_argument("--stt_language", type=str, default=None, help="STT language (korean, french, english, etc.). Auto-detected from dataset if not specified.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save evaluation results as JSON.")
    parser.add_argument("--bcs_gain_db", type=float, default=0.0, help="Gain perturbation for BCS input in dB. default is 0.0")
    parser.add_argument("--acs_gain_db", type=float, default=0.0, help="Gain perturbation for ACS input in dB. default is 0.0")
    parser.add_argument("--save_per_utterance_json", default=False, action="store_true",
                        help="Include per-utterance metrics (pesq/stoi/csig/cbak/covl/segSNR, and CER/WER if --eval_stt) under a 'per_utterance' key in each SNR block of --output_json.")

    args = parser.parse_args()

    log_file = args.log_file
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)

    runtime = prepare_evaluation_runtime(args, logger=logger)

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {runtime.model_args.model_class}")
    logger.info(f"Input type: {runtime.model_args.input_type}")
    logger.info(f"Checkpoint: {args.chkpt_dir}")
    logger.info(f"Device: {args.device}")
    if args.bcs_gain_db != 0.0 or args.acs_gain_db != 0.0:
        logger.info(f"Gain perturbation: BCS={args.bcs_gain_db:+.1f}dB, ACS={args.acs_gain_db:+.1f}dB (relative={args.bcs_gain_db - args.acs_gain_db:+.1f}dB)")

    metrics = evaluate(args=runtime.conf,
                       model=runtime.model,
                       data_loader_list=runtime.data_loader_list,
                       logger=logger,
                       epoch=None,
                       stft_args=runtime.stft_args)

    if args.output_json:
        write_json(build_evaluation_output(metrics, args, runtime), args.output_json, logger=logger)
