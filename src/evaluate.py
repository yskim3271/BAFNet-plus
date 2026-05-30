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
import torchaudio
import numpy as np
import nlptutti as sarmetric
import logging
from typing import Dict, Optional, Any, List, Tuple
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.compute_metrics import compute_metrics
from src.runtime_common import (
    add_eval_augmentation_args,
    add_runtime_common_args,
    build_evaluation_output,
    get_model_input,
    prepare_evaluation_runtime,
    validate_eval_augmentation_args,
    write_json,
)
from src.stft import mag_pha_istft
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


class SquimEvaluator:
    """Squim-Objective non-intrusive evaluator (STOI / PESQ / SI-SDR estimates).

    Loads ``torchaudio.pipelines.SQUIM_OBJECTIVE`` once and reuses it. Used for
    Vibravox ``speech_noisy`` test where headset reference is noisy and
    intrusive PESQ is unreliable — matches Vibravox paper (T1) Table V
    reporting convention (Squim-STOI / Noresqa-MOS). Returns three predicted
    metrics per utterance; no clean reference required.
    """

    def __init__(self, device: str = "cuda", logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Loading Squim-Objective model")
        bundle = torchaudio.pipelines.SQUIM_OBJECTIVE
        self.model = bundle.get_model().to(device).eval()
        self.sample_rate = bundle.sample_rate
        self.logger.info(f"SquimEvaluator initialized (sample_rate={self.sample_rate})")

    @torch.no_grad()
    def evaluate_one(self, waveform: np.ndarray) -> Tuple[float, float, float]:
        """Predict (Squim-STOI, Squim-PESQ, Squim-SI-SDR) for one mono utterance."""
        w = torch.as_tensor(waveform, dtype=torch.float32, device=self.device).reshape(1, -1)
        stoi, pesq, sisdr = self.model(w)
        return float(stoi.item()), float(pesq.item()), float(sisdr.item())


class DNSMOSEvaluator:
    """DNSMOS P.835 non-intrusive evaluator (ovrl / sig / bak / p808 MOS).

    Wraps ``speechmos.dnsmos`` (Microsoft DNS-Challenge ONNX model). Used
    alongside ``SquimEvaluator`` for Vibravox ``speech_noisy`` test reporting
    where the headset reference is noisy (T1 §V.B). DNSMOS is CPU-only ONNX
    inference (~450 ms/utt on the bundled model); ``device`` is accepted for
    API parity with ``SquimEvaluator`` but ignored.
    """

    def __init__(self, device: str = "cuda", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        from speechmos import dnsmos as _dnsmos
        self._dnsmos = _dnsmos
        self.sample_rate = 16000
        self.logger.info("DNSMOSEvaluator initialized (sample_rate=16000, ONNX CPU)")

    def evaluate_one(self, waveform: np.ndarray) -> Tuple[float, float, float, float]:
        """Predict (ovrl_mos, sig_mos, bak_mos, p808_mos) for one mono utterance."""
        w = np.asarray(waveform, dtype=np.float32).reshape(-1)
        out = self._dnsmos.run(w, sr=self.sample_rate)
        return (
            float(out["ovrl_mos"]),
            float(out["sig_mos"]),
            float(out["bak_mos"]),
            float(out["p808_mos"]),
        )


class PEREvaluator:
    """Phoneme Error Rate evaluator using HF wav2vec2 phonemizer.

    Matches T4 (Mimi WASPAA 2025) / T1 (Vibravox §VI) protocol: the same
    phonemizer model is applied to both the enhanced output and the headset
    reference; PER is the character-level edit-distance ratio between the
    two IPA phoneme strings. Default model is
    ``Cnam-LMSSC/phonemizer_headset_microphone`` (Wav2Vec2ForCTC fine-tuned
    on Vibravox headset, IPA token vocab, 16 kHz).
    """

    def __init__(
        self,
        model_id: str = "Cnam-LMSSC/phonemizer_headset_microphone",
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
    ):
        from transformers import AutoProcessor, AutoModelForCTC
        self.logger = logger or logging.getLogger(__name__)
        self.device = device
        self.model_id = model_id
        self.logger.info(f"Loading phonemizer: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()
        self.sample_rate = 16000
        self.logger.info(f"PEREvaluator initialized (sample_rate={self.sample_rate})")

    @torch.no_grad()
    def _phonemize(self, waveform: np.ndarray) -> str:
        w = np.asarray(waveform, dtype=np.float32).reshape(-1)
        inputs = self.processor(
            w, sampling_rate=self.sample_rate, return_tensors="pt",
        ).input_values.to(self.device)
        logits = self.model(inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0]

    def evaluate_one(
        self, enhanced: np.ndarray, reference: np.ndarray,
    ) -> Tuple[float, str, str]:
        """Compute PER for one (enhanced, reference) pair using shared phonemizer.

        Returns ``(per, pred_phonemes, ref_phonemes)``. ``per`` is the
        character-level edit distance (IPA tokens are single characters in
        this tokenizer) of the predicted phoneme string against the reference,
        normalised by reference length. If the reference is empty, PER is 0.
        """
        import jiwer
        pred = self._phonemize(enhanced)
        ref = self._phonemize(reference)
        if len(ref) == 0:
            return 0.0, pred, ref
        per = float(jiwer.cer(ref, pred))
        return per, pred, ref


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

    # Initialize Squim-Objective evaluator (non-intrusive STOI/PESQ/SI-SDR estimates)
    squim_evaluator = None
    if bool(getattr(args, 'eval_squim', False)):
        squim_evaluator = SquimEvaluator(device=args.device, logger=logger)

    # Initialize DNSMOS evaluator (non-intrusive P.835 ovrl/sig/bak/p808 MOS)
    dnsmos_evaluator = None
    if bool(getattr(args, 'eval_dnsmos', False)):
        dnsmos_evaluator = DNSMOSEvaluator(device=args.device, logger=logger)

    # Initialize PER evaluator (HF wav2vec2 phonemizer; T4-style shared phonemizer
    # protocol — enhanced vs reference IPA edit distance).
    per_evaluator = None
    if bool(getattr(args, 'eval_per', False)):
        per_model_id = str(getattr(args, 'per_model_id',
                                   'Cnam-LMSSC/phonemizer_headset_microphone'))
        per_evaluator = PEREvaluator(model_id=per_model_id, device=args.device, logger=logger)

    save_per_utt = bool(getattr(args, 'save_per_utterance', False))

    for snr, data_loader in data_loader_list.items():
        iterator = LogProgress(logger, data_loader, name=f"Evaluate on {snr}dB")
        enhanced = []
        results  = []
        squim_results: List[Tuple[float, float, float]] = [] if squim_evaluator else []
        dnsmos_results: List[Tuple[float, float, float, float]] = [] if dnsmos_evaluator else []
        per_results: List[Tuple[float, str, str]] = [] if per_evaluator else []
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

                model_input = get_model_input(
                    bcs, noisy_acs, args.model.input_type, args.device, stft_args
                )

                clean_mag_hat, clean_pha_hat, _ = model(model_input)

                clean_acs_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)

                clean_acs = clean_acs.squeeze().detach().cpu().numpy()
                clean_acs_hat = clean_acs_hat.squeeze().detach().cpu().numpy()
                if len(clean_acs) != len(clean_acs_hat):
                    length = min(len(clean_acs), len(clean_acs_hat))
                    clean_acs = clean_acs[0:length]
                    clean_acs_hat = clean_acs_hat[0:length]

                enhanced.append((clean_acs_hat, text[0]))
                results.append(compute_metrics(clean_acs, clean_acs_hat))
                if squim_evaluator:
                    squim_results.append(squim_evaluator.evaluate_one(clean_acs_hat))
                if dnsmos_evaluator:
                    dnsmos_results.append(dnsmos_evaluator.evaluate_one(clean_acs_hat))
                if per_evaluator:
                    per_results.append(per_evaluator.evaluate_one(clean_acs_hat, clean_acs))
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

        if squim_evaluator:
            squim_stoi, squim_pesq, squim_sisdr = np.mean(squim_results, axis=0)
            metrics[snr_key]['squim_stoi'] = float(squim_stoi)
            metrics[snr_key]['squim_pesq'] = float(squim_pesq)
            metrics[snr_key]['squim_sisdr'] = float(squim_sisdr)
            logger.info(bold(f"{prefix}Squim-Objective on {snr}dB: STOI={squim_stoi:.4f}, PESQ={squim_pesq:.4f}, SI-SDR={squim_sisdr:.4f}"))

        if dnsmos_evaluator:
            dns_ovrl, dns_sig, dns_bak, dns_p808 = np.mean(dnsmos_results, axis=0)
            metrics[snr_key]['dnsmos_ovrl'] = float(dns_ovrl)
            metrics[snr_key]['dnsmos_sig'] = float(dns_sig)
            metrics[snr_key]['dnsmos_bak'] = float(dns_bak)
            metrics[snr_key]['dnsmos_p808'] = float(dns_p808)
            logger.info(bold(f"{prefix}DNSMOS on {snr}dB: ovrl={dns_ovrl:.4f}, sig={dns_sig:.4f}, bak={dns_bak:.4f}, p808={dns_p808:.4f}"))

        if per_evaluator:
            per_mean = float(np.mean([p[0] for p in per_results]))
            metrics[snr_key]['per'] = per_mean
            logger.info(bold(f"{prefix}PER on {snr}dB: {per_mean:.4f} (model={per_evaluator.model_id})"))

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
                if squim_evaluator:
                    rec["squim_stoi"] = float(squim_results[i][0])
                    rec["squim_pesq"] = float(squim_results[i][1])
                    rec["squim_sisdr"] = float(squim_results[i][2])
                if dnsmos_evaluator:
                    rec["dnsmos_ovrl"] = float(dnsmos_results[i][0])
                    rec["dnsmos_sig"] = float(dnsmos_results[i][1])
                    rec["dnsmos_bak"] = float(dnsmos_results[i][2])
                    rec["dnsmos_p808"] = float(dnsmos_results[i][3])
                if per_evaluator:
                    rec["per"] = float(per_results[i][0])
                    rec["phoneme_pred"] = str(per_results[i][1])
                    rec["phoneme_ref"] = str(per_results[i][2])
                per_utt_block[uid] = rec
            metrics[snr_key]["per_utterance"] = per_utt_block

    return metrics


def run_per_utterance_evaluation(
    conf: DictConfig,
    model: torch.nn.Module,
    data_loader_list: Dict[str, DataLoader],
    stft_args: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run ``evaluate()`` in per-utterance-only mode for statistical analysis.

    Forces ``eval_stt=False`` and ``save_per_utterance=True`` on ``conf`` and
    returns only the per-utterance block, keyed by SNR then utterance id.

    Output shape::

        {
            "-10dB": {"<spk>_<sent>": {"pesq": ..., "stoi": ..., ...}, ...},
            "0dB":   {...},
        }
    """
    conf.eval_stt = False
    conf.save_per_utterance = True

    metrics = evaluate(
        args=conf,
        model=model,
        data_loader_list=data_loader_list,
        logger=logger or logging.getLogger(__name__),
        epoch=None,
        stft_args=stft_args,
    )
    return {
        snr_key: snr_metrics.get("per_utterance", {})
        for snr_key, snr_metrics in metrics.items()
    }


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_runtime_common_args(parser)
    add_eval_augmentation_args(parser)
    parser.add_argument("--log_file", type=str, default="output.log",
                        help="Log file name. default is output.log")
    parser.add_argument("--eval_stt", default=False, action="store_true",
                        help="Evaluate STT performance")
    parser.add_argument("--eval_squim", default=False, action="store_true",
                        help="Compute Squim-Objective non-intrusive STOI/PESQ/SI-SDR estimates "
                             "(torchaudio.pipelines.SQUIM_OBJECTIVE). Recommended for Vibravox "
                             "speech_noisy where intrusive PESQ is unreliable due to noisy reference.")
    parser.add_argument("--eval_dnsmos", default=False, action="store_true",
                        help="Compute DNSMOS P.835 non-intrusive ovrl/sig/bak/p808 MOS "
                             "(speechmos.dnsmos, ONNX). Complements --eval_squim for Vibravox "
                             "speech_noisy T1 reporting alignment.")
    parser.add_argument("--eval_per", default=False, action="store_true",
                        help="Compute Phoneme Error Rate via HF wav2vec2 phonemizer "
                             "(default: Cnam-LMSSC/phonemizer_headset_microphone). Applies the "
                             "same phonemizer to enhanced output and clean_acs reference per "
                             "T4 (Mimi WASPAA 2025) / T1 (Vibravox §VI) protocol.")
    parser.add_argument("--per_model_id", type=str,
                        default="Cnam-LMSSC/phonemizer_headset_microphone",
                        help="HF model id for --eval_per phonemizer.")
    parser.add_argument("--stt_language", type=str, default=None,
                        help="STT language (korean, french, english, etc.). "
                             "Auto-detected from dataset if not specified.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save evaluation results as JSON.")
    parser.add_argument("--bcs_gain_db", type=float, default=0.0,
                        help="Gain perturbation for BCS input in dB. default is 0.0")
    parser.add_argument("--acs_gain_db", type=float, default=0.0,
                        help="Gain perturbation for ACS input in dB. default is 0.0")
    parser.add_argument("--save_per_utterance_json", default=False, action="store_true",
                        help="Include per-utterance metrics (pesq/stoi/csig/cbak/covl/segSNR, "
                             "and CER/WER if --eval_stt) under a 'per_utterance' key in each "
                             "SNR block of --output_json.")

    args = parser.parse_args()
    validate_eval_augmentation_args(args)

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
