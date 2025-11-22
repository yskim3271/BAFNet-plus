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
        if not enhanced:
            self.logger.warning("No samples to evaluate")
            return 0.0, 0.0

        self.logger.info(f"Evaluating {len(enhanced)} samples with STT...")

        cer_sum = 0.0
        wer_sum = 0.0

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

            cer_sum += sarmetric.get_cer(
                reference_text, transcription, rm_punctuation=True
            )['cer']
            wer_sum += sarmetric.get_wer(
                reference_text, transcription, rm_punctuation=True
            )['wer']

        cer = cer_sum / len(enhanced)
        wer = wer_sum / len(enhanced)

        self.logger.info(f"STT Evaluation complete: CER={cer:.4f}, WER={wer:.4f}")

        return cer, wer


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

    for snr, data_loader in data_loader_list.items():
        iterator = LogProgress(logger, data_loader, name=f"Evaluate on {snr}dB")
        enhanced = []
        results  = []
        with torch.no_grad():
            for data in iterator:
                bcs, noisy_acs, clean_acs, _, text = data
                
                if args.model.input_type == "acs":
                    input = mag_pha_stft(noisy_acs, **stft_args)[2].to(args.device)
                elif args.model.input_type == "bcs":
                    input = mag_pha_stft(bcs, **stft_args)[2].to(args.device)
                elif args.model.input_type == "acs+bcs":
                    input = mag_pha_stft(bcs, **stft_args)[2].to(args.device), mag_pha_stft(noisy_acs, **stft_args)[2].to(args.device)

                clean_mag_hat, clean_pha_hat, _ = model(input)

                clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)

                clean_acs = clean_acs.squeeze().detach().cpu().numpy()
                clean_hat = clean_hat.squeeze().detach().cpu().numpy()
                if len(clean_acs) != len(clean_hat):
                    length = min(len(clean_acs), len(clean_hat))
                    clean_acs = clean_acs[0:length]
                    clean_hat = clean_hat[0:length]

                enhanced.append((clean_hat, text[0]))
                results.append(compute_metrics(clean_acs, clean_hat))
        
        pesq, csig, cbak, covl, segSNR, stoi = np.mean(results, axis=0)
        metrics[f'{snr}dB'] = {
            "pesq": pesq,
            "stoi": stoi,
            "csig": csig,
            "cbak": cbak,
            "covl": covl,
            "segSNR": segSNR
        }
        logger.info(bold(f"{prefix}Performance on {snr}dB: PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}"))

        # STT evaluation (model reused!)
        if stt_evaluator:
            language = getattr(args, 'stt_language', 'korean')
            cer, wer = stt_evaluator.evaluate(enhanced, language=language)
            metrics[f'{snr}dB']['cer'] = cer
            metrics[f'{snr}dB']['wer'] = wer
            logger.info(bold(f"{prefix}Performance on {snr}dB: CER={cer:.4f}, WER={wer:.4f}"))
   
    return metrics



if __name__=="__main__":
    import argparse
    from src.data import Noise_Augmented_Dataset
    from src.utils import load_model, load_checkpoint, parse_file_list, get_stft_args_from_config
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from datasets import load_dataset, concatenate_datasets
    
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
    
    args = parser.parse_args()
    chkpt_dir = args.chkpt_dir
    chkpt_file = args.chkpt_file
    device = args.device

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
    
    conf = OmegaConf.load(args.model_config)
    conf.device = device
    conf.eval_stt = args.eval_stt
    # Note: stt_language will be set after dataset is determined (see below)
    
    model_args = conf.model
    model_lib = model_args.model_lib
    model_class_name = model_args.model_class

    # Load model and checkpoint using utility functions
    model = load_model(model_lib, model_class_name, model_args.param, device)
    model = load_checkpoint(model, chkpt_dir, chkpt_file, device)
    tm_only = model_args.input_type == "tm"

    # Dataset-to-language mapping for STT
    DATASET_LANGUAGE_MAP = {
        "taps": "korean",
        "vibravox": "french"
    }

    # Load dataset based on user selection
    if args.dataset.lower() == "taps":
        testset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")
    elif args.dataset.lower() == "vibravox":
        testset = load_dataset("yskim3271/vibravox_16k", split="test")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Configure STT language (CLI override or auto-detection)
    if args.stt_language:
        stt_language = args.stt_language
        logger.info(f"Using user-specified STT language: {stt_language}")
    else:
        stt_language = DATASET_LANGUAGE_MAP.get(args.dataset.lower(), "korean")
        logger.info(f"Auto-detected STT language from dataset: {stt_language}")

    conf.stt_language = stt_language

    testset_list = [testset] * args.test_augment_numb
    testset = concatenate_datasets(testset_list)
    
    ev_loader_list = {}

    # Parse file lists using utility function
    noise_test_list = parse_file_list(args.noise_dir, args.noise_test)
    rir_test_list = parse_file_list(args.rir_dir, args.rir_test)

    # Prepare STFT args using utility function
    stft_args = get_stft_args_from_config(model_args)

    for fixed_snr in args.snr_step:
        ev_dataset = Noise_Augmented_Dataset(
            datapair_list= testset,
            noise_list= noise_test_list,
            rir_list= rir_test_list,
            snr_range= [fixed_snr, fixed_snr],
            reverb_proportion=args.reverb_proportion,
            target_dB_FS=args.target_dB_FS,
            target_dB_FS_floating_value=args.target_dB_FS_floating_value,
            silence_length=args.silence_length,
            deterministic=True,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
            tm_only=tm_only,
        )

        ev_loader = DataLoader(
            dataset=ev_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True
        )

        ev_loader_list[f"{fixed_snr}"] = ev_loader

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {model_class_name}")
    logger.info(f"Input type: {model_args.input_type}")
    logger.info(f"Checkpoint: {chkpt_dir}")
    logger.info(f"Device: {device}")
    
    evaluate(args=conf,
            model=model,
            data_loader_list=ev_loader_list,
            logger=logger,
            epoch=None,
            stft_args=stft_args)
