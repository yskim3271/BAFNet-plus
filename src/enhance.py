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

from src.utils import LogProgress
from src.stft import mag_pha_stft, mag_pha_istft

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


def write(wav: torch.Tensor, filename: str, sr: int = DEFAULT_SAMPLE_RATE) -> None:
    """Write a single waveform to file with normalization.

    Args:
        wav: Waveform tensor
        filename: Output filename
        sr: Sample rate in Hz
    """
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    try:
        torchaudio.save(filename, wav.cpu(), sr)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save {filename}: {e}")


def enhance_multiple_snr(
    args: Any,
    model: torch.nn.Module,
    dataloader_list: Dict[int, DataLoader],
    logger: logging.Logger,
    epoch: Optional[int] = None,
    local_out_dir: str = "samples"
) -> None:
    """Run enhancement on multiple SNR levels.

    Args:
        args: Configuration object
        model: Enhancement model
        dataloader_list: Dictionary mapping SNR values to dataloaders
        logger: Logger instance
        epoch: Training epoch number (for naming)
        local_out_dir: Output directory
    """
    for snr, data_loader in dataloader_list.items():
        enhance(args, model, data_loader, logger, snr, epoch, local_out_dir)


def _get_model_input(
    bcs: torch.Tensor,
    noisy_acs: torch.Tensor,
    input_type: str,
    device: torch.device,
    stft_args: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, ...]:
    """Prepare model input based on input type.

    Args:
        bcs: Body-conducted speech input [B, C, T]
        noisy_acs: Noisy air-conducted speech input [B, C, T]
        input_type: One of ["acs", "bcs", "acs+bcs"]
        device: Target device
        stft_args: STFT parameters for frequency-domain models

    Returns:
        Model input tensor(s)

    Raises:
        ValueError: If input_type is invalid or stft_args is missing for freq-domain models
    """
    # Frequency-domain handlers
    freq_handlers = {
        "acs": lambda: mag_pha_stft(noisy_acs, **stft_args)[2].to(device),
        "bcs": lambda: mag_pha_stft(bcs, **stft_args)[2].to(device),
        "acs+bcs": lambda: (
            mag_pha_stft(bcs, **stft_args)[2].to(device),
            mag_pha_stft(noisy_acs, **stft_args)[2].to(device)
        ),
    }

    # Check if frequency-domain model has stft_args
    if input_type in freq_handlers and stft_args is None:
        raise ValueError("stft_args must be provided for frequency-domain models")

    # Dispatch to appropriate handler
    if input_type not in freq_handlers:
        raise ValueError(f"Invalid model input type: {input_type}")

    return freq_handlers[input_type]()


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

    # Validate stft_args before processing (moved from inside loop)
    if args.model.input_type in ["acs", "bcs", "acs+bcs"] and stft_args is None:
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

                # Prepare model input using dispatcher
                input_data = _get_model_input(
                    bcs, noisy_acs, args.model.input_type, args.device, stft_args
                )

                # Run model inference
                if isinstance(input_data, tuple):
                    clean_mag_hat, clean_pha_hat, _ = model(*input_data)
                else:
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
    from src.data import Noise_Augmented_Dataset
    from src.utils import load_model, load_checkpoint, parse_file_list, get_stft_args_from_config
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from datasets import load_dataset

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

    # Frozen/Folded normalization options
    parser.add_argument(
        "--norm_infer_mode",
        type=str,
        choices=["instance", "frozen", "folded"],
        default="instance",
        help="Normalization inference mode: "
             "instance (default, original IN), "
             "frozen (fixed stats from calibration), "
             "folded (IN absorbed into Conv)"
    )
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default=None,
        help="Path to *.normstats.pt file for frozen/folded mode. "
             "If not specified, looks for model.normstats.pt in chkpt_dir"
    )

    # SCA (Squeeze-Channel-Attention) streaming options
    parser.add_argument(
        "--sca_mode",
        type=str,
        choices=["global", "ema", "local"],
        default="global",
        help="SCA squeeze mode: "
             "global (default, AdaptiveAvgPool1d equivalent), "
             "ema (exponential moving average for streaming), "
             "local (last W frames only)"
    )
    parser.add_argument(
        "--sca_ema_alpha",
        type=float,
        default=0.99,
        help="EMA decay factor for sca_mode=ema (default: 0.99). "
             "Higher values = slower adaptation, more stable."
    )
    parser.add_argument(
        "--sca_local_window",
        type=int,
        default=32,
        help="Window size for sca_mode=local (default: 32 frames)"
    )
    parser.add_argument(
        "--sca_stateful_eval",
        action="store_true",
        default=True,
        help="Maintain EMA state across chunks in eval mode (default: True)"
    )

    # Stateful Convolution options
    parser.add_argument(
        "--use_stateful_conv",
        action="store_true",
        default=False,
        help="Use stateful convolutions for streaming inference. "
             "Eliminates zero-padding discontinuity at chunk boundaries."
    )

    args = parser.parse_args()
    chkpt_dir = args.chkpt_dir
    chkpt_file = args.chkpt_file
    device = args.device
    local_out_dir = args.output_dir

    conf = OmegaConf.load(os.path.join(chkpt_dir, '.hydra', "config.yaml"))
    hydra_conf = OmegaConf.load(os.path.join(chkpt_dir, '.hydra', "hydra.yaml"))
    del hydra_conf.hydra.job_logging.handlers.file
    hydra_conf.hydra.job_logging.root.handlers = ['console']
    logging_conf = OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True)


    logging.config.dictConfig(logging_conf)
    logger = logging.getLogger(__name__)
    conf.device = device

    model_args = conf.model
    model_lib = model_args.model_lib
    model_class_name = model_args.model_class

    # Load model and checkpoint using utility functions
    model = load_model(model_lib, model_class_name, model_args.param, device)
    model = load_checkpoint(model, chkpt_dir, chkpt_file, device)

    # Apply frozen/folded normalization if requested
    if args.norm_infer_mode != "instance":
        from pathlib import Path
        from src.calibration.calibrator import INCalibrator

        # Find stats file
        if args.norm_stats_path is not None:
            stats_path = args.norm_stats_path
        else:
            stats_path = os.path.join(chkpt_dir, "model.normstats.pt")

        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Calibration stats not found at {stats_path}. "
                f"Run 'python -m src.calibrate --chkpt_dir {chkpt_dir}' first, "
                f"or specify --norm_stats_path"
            )

        logger.info(f"Loading calibration stats from: {stats_path}")
        stats, meta = INCalibrator.load(stats_path)
        logger.info(f"  Modules: {len(stats)}, Model: {meta.get('model_class', 'unknown')}")

        if args.norm_infer_mode == "frozen":
            from src.models.streaming.converters import replace_in_with_frozen
            logger.info("Applying frozen normalization...")
            model = replace_in_with_frozen(model, stats, verbose=True)

        elif args.norm_infer_mode == "folded":
            from src.calibration.folding import fold_model
            logger.info("Applying conv folding...")
            model = fold_model(model, stats, verbose=True)

        model.to(device)

    # Apply SCA replacement if not using global mode
    if args.sca_mode != "global":
        from src.models.streaming.converters import replace_sca_with_streaming, get_sca_info

        logger.info(f"Replacing SCA with streaming mode: {args.sca_mode}")
        logger.info(f"  EMA alpha: {args.sca_ema_alpha}")
        logger.info(f"  Local window: {args.sca_local_window}")
        logger.info(f"  Stateful eval: {args.sca_stateful_eval}")

        model = replace_sca_with_streaming(
            model,
            mode=args.sca_mode,
            ema_alpha=args.sca_ema_alpha,
            local_window=args.sca_local_window,
            stateful_eval=args.sca_stateful_eval,
            verbose=True,
        )

        sca_info = get_sca_info(model)
        logger.info(f"SCA replacement complete: {sca_info['streaming_sca_count']} modules")

    # Apply stateful convolutions if requested
    if args.use_stateful_conv:
        from src.models.streaming.converters import (
            convert_to_stateful,
            set_streaming_mode,
            get_stateful_layer_count,
        )

        logger.info("Applying stateful convolutions...")
        model = convert_to_stateful(model, verbose=False, inplace=True)
        model.to(device)
        model.eval()
        set_streaming_mode(model, True)

        layer_counts = get_stateful_layer_count(model)
        logger.info(f"Stateful conversion complete: {layer_counts['total']} layers")
        if layer_counts["StatefulCausalConv1d"] > 0:
            logger.info(f"  - StatefulCausalConv1d: {layer_counts['StatefulCausalConv1d']}")
        if layer_counts["StatefulAsymmetricConv2d"] > 0:
            logger.info(f"  - StatefulAsymmetricConv2d: {layer_counts['StatefulAsymmetricConv2d']}")
        if layer_counts["StatefulCausalConv2d"] > 0:
            logger.info(f"  - StatefulCausalConv2d: {layer_counts['StatefulCausalConv2d']}")

    bcs_only = "acs" not in model_args.input_type

    # Load dataset based on user selection
    if args.dataset.lower() == "taps":
        testset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")
    elif args.dataset.lower() == "vibravox":
        testset = load_dataset("yskim3271/vibravox_16k", split="test")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Parse file lists using utility function
    noise_test_list = parse_file_list(args.noise_dir, args.noise_test)
    rir_test_list = parse_file_list(args.rir_dir, args.rir_test)

    tt_dataset = Noise_Augmented_Dataset(datapair_list=testset,
                                         noise_list=noise_test_list,
                                         rir_list=rir_test_list,
                                         snr_range=[args.snr, args.snr],
                                         reverb_proportion=args.reverb_prop,
                                         target_dB_FS=args.target_dB_FS,
                                         target_dB_FS_floating_value=0,
                                         silence_length=0.2,
                                         deterministic=True,
                                         sampling_rate=16000,
                                         with_id=True,
                                         with_text=True,
                                         bcs_only=bcs_only)


    tt_loader = DataLoader(
        dataset=tt_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    # Prepare STFT args for frequency-domain models using utility function
    stft_args = get_stft_args_from_config(model_args)

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {model_class_name}")
    logger.info(f"Checkpoint: {chkpt_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Norm inference mode: {args.norm_infer_mode}")
    logger.info(f"SCA mode: {args.sca_mode}" + (f" (alpha={args.sca_ema_alpha})" if args.sca_mode == "ema" else "") + (f" (window={args.sca_local_window})" if args.sca_mode == "local" else ""))
    logger.info(f"Stateful conv: {args.use_stateful_conv}")
    logger.info(f"Output directory: {local_out_dir}")
    os.makedirs(local_out_dir, exist_ok=True)

    enhance(model=model,
            data_loader=tt_loader,
            args=conf,
            snr=args.snr,
            epoch=None,
            logger=logger,
            local_out_dir=local_out_dir,
            stft_args=stft_args
            )
