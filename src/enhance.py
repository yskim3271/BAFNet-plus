import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchaudio
import logging
from typing import Dict, Optional, Any, Tuple
from torch.utils.data import DataLoader

from src.runtime_common import get_model_input, prepare_enhancement_runtime
from src.utils import LogProgress
from src.stft import mag_pha_istft

# Constants
DEFAULT_SAMPLE_RATE = 16_000


def save_wavs(wavs_dict: Dict[str, torch.Tensor], filepath: str, sr: int = DEFAULT_SAMPLE_RATE) -> None:
    """Save multiple waveforms to separate files.

    Args:
        wavs_dict: Dictionary mapping suffixes to waveform tensors
        filepath: Base filepath (suffixes will be appended)
        sr: Sample rate in Hz
    """
    for key, wav in wavs_dict.items():
        try:
            torchaudio.save(filepath + f"_{key}.wav", wav, sr)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save {filepath}_{key}.wav: {e}")


def enhance(
    args: Any,
    model: torch.nn.Module,
    data_loader: DataLoader,
    logger: logging.Logger,
    snr: int,
    epoch: Optional[int] = None,
    local_out_dir: str = "samples",
    stft_args: Optional[Dict[str, Any]] = None
) -> None:
    """Run enhancement on a dataset and save results.

    Args:
        args: Configuration object with model and device settings
        model: Enhancement model
        data_loader: DataLoader for test dataset
        logger: Logger instance
        snr: Signal-to-noise ratio in dB
        epoch: Training epoch number (for naming outputs)
        local_out_dir: Output directory for enhanced samples
        stft_args: STFT parameters for frequency-domain models

    Raises:
        ValueError: If stft_args is None for frequency-domain models
    """
    model.eval()

    if stft_args is None:
        raise ValueError("stft_args must be provided for frequency-domain models")

    suffix = f"_epoch{epoch+1}" if epoch is not None else ""
    outdir_wavs = os.path.join(local_out_dir, f"wavs" + suffix + f"_{snr}dB")
    os.makedirs(outdir_wavs, exist_ok=True)

    failed_samples = []

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Generate enhanced files")
        for batch_idx, data in enumerate(iterator):
            try:
                # Get batch data (batch, channel, time)
                bcs, noisy_acs, clean_acs, id, _ = data

                # Prepare model input using shared dispatcher
                input_data = get_model_input(
                    bcs, noisy_acs, args.model.input_type, args.device, stft_args
                )

                # Run model inference. BAFNet/BAFNetPlus expect tuple input as
                # one argument, matching train/evaluate behavior.
                clean_mag_hat, clean_pha_hat, _ = model(input_data)
                clean_acs_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)

                # Move to CPU and squeeze channel dimension
                bcs = bcs.squeeze(1).cpu()
                clean_acs = clean_acs.squeeze(1).cpu()
                noisy_acs = noisy_acs.squeeze(1).cpu()
                clean_acs_hat = clean_acs_hat.squeeze(1).cpu()

                wavs_dict = {
                    "bcs": bcs,
                    "noisy_acs": noisy_acs,
                    "clean_acs": clean_acs,
                    "clean_acs_hat": clean_acs_hat,
                }

                save_wavs(wavs_dict, os.path.join(outdir_wavs, id[0]))

            except Exception as e:
                logger.error(f"Failed to process sample {batch_idx} (id={id[0] if id else 'unknown'}): {e}")
                failed_samples.append((batch_idx, id[0] if id else 'unknown', str(e)))
                continue

    # Report failures if any
    if failed_samples:
        logger.warning(f"Failed to process {len(failed_samples)}/{len(data_loader)} samples")
        failure_log_path = os.path.join(outdir_wavs, "failed_samples.txt")
        try:
            with open(failure_log_path, 'w') as f:
                for batch_idx, sample_id, error in failed_samples:
                    f.write(f"Batch {batch_idx} | ID: {sample_id} | Error: {error}\n")
            logger.info(f"Failure log saved to {failure_log_path}")
        except Exception as e:
            logger.error(f"Failed to save failure log: {e}")


if __name__ == "__main__":
    import logging
    import logging.config
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--chkpt_dir", type=str, default='.', help="Path to the checkpoint directory. default is current directory")
    parser.add_argument("--chkpt_file", type=str, default="best.th", help="Checkpoint file name. default is best.th")
    parser.add_argument("--dataset", type=str, default="taps", choices=["taps", "vibravox"], help="Dataset to use: taps or vibravox. default is taps")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to the noise directory.")
    parser.add_argument("--noise_test", type=str, required=True, help="List of noise files for testing.")
    parser.add_argument("--rir_dir", type=str, required=True, help="Path to the RIR directory.")
    parser.add_argument("--rir_test", type=str, required=True, help="List of RIR files for testing.")
    parser.add_argument("--snr", type=int, default=0, help="Signal to noise ratio. default is 0 dB")
    parser.add_argument("--reverb_prop", type=float, default=0, help="Reverberation proportion. default is 0")
    parser.add_argument("--target_dB_FS", type=float, default=-26, help="Target dB FS. default is -26")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory for enhanced samples. default is samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Specifies the device (cuda or cpu).")

    args = parser.parse_args()
    local_out_dir = args.output_dir

    hydra_conf = OmegaConf.load(os.path.join(args.chkpt_dir, '.hydra', "hydra.yaml"))
    del hydra_conf.hydra.job_logging.handlers.file
    hydra_conf.hydra.job_logging.root.handlers = ['console']
    logging_conf = OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True)

    logging.config.dictConfig(logging_conf)
    logger = logging.getLogger(__name__)

    runtime = prepare_enhancement_runtime(args, logger=logger)
    model = runtime.model

    tt_loader = runtime.data_loader_list[str(args.snr)]

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {runtime.model_args.model_class}")
    logger.info(f"Checkpoint: {args.chkpt_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {local_out_dir}")
    os.makedirs(local_out_dir, exist_ok=True)

    enhance(model=model,
            data_loader=tt_loader,
            args=runtime.conf,
            snr=args.snr,
            epoch=None,
            logger=logger,
            local_out_dir=local_out_dir,
            stft_args=runtime.stft_args
            )
