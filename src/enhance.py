import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchaudio

from matplotlib import pyplot as plt
from src.utils import LogProgress
from src.stft import mag_pha_stft, mag_pha_istft

def save_wavs(wavs_dict, filepath, sr=16_000):
    for i, (key, wav) in enumerate(wavs_dict.items()):
        torchaudio.save(filepath + f"_{key}.wav", wav, sr)
        
def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)

def enhance_multiple_snr(args, model, dataloader_list, logger, epoch=None, local_out_dir="samples"):
    for snr, data_loader in dataloader_list.items():
        enhance(args, model, data_loader, logger, snr, epoch, local_out_dir)

def enhance(args, model, data_loader, logger, snr, epoch=None, local_out_dir="samples", stft_args=None):
    model.eval()

    suffix = f"_epoch{epoch+1}" if epoch is not None else ""

    iterator = LogProgress(logger, data_loader, name=f"Enhance on {snr}dB")
    outdir_wavs= os.path.join(local_out_dir, f"wavs" + suffix + f"_{snr}dB")
    os.makedirs(outdir_wavs, exist_ok=True)

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Generate enhanced files")
        for data in iterator:
            # Get batch data (batch, channel, time)
            tm, noisy_am, clean_am, id, text = data

            # Handle frequency-domain models (acs, bcs, acs+bcs)
            if args.model.input_type in ["acs", "bcs", "acs+bcs"]:
                if stft_args is None:
                    raise ValueError("stft_args must be provided for frequency-domain models")

                if args.model.input_type == "acs":
                    input = mag_pha_stft(noisy_am, **stft_args)[2].to(args.device)
                elif args.model.input_type == "bcs":
                    input = mag_pha_stft(tm, **stft_args)[2].to(args.device)
                elif args.model.input_type == "acs+bcs":
                    input = mag_pha_stft(tm, **stft_args)[2].to(args.device), mag_pha_stft(noisy_am, **stft_args)[2].to(args.device)

                clean_mag_hat, clean_pha_hat, clean_com_hat = model(input)
                clean_am_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)
            # Handle time-domain models (am, tm, am+tm)
            elif args.model.input_type == "am":
                clean_am_hat = model(noisy_am.to(args.device))
            elif args.model.input_type == "tm":
                clean_am_hat = model(tm.to(args.device))
            elif args.model.input_type == "am+tm":
                clean_am_hat = model(tm.to(args.device), noisy_am.to(args.device))
            else:
                raise ValueError(f"Invalid model input type: {args.model.input_type}")

            tm = tm.squeeze(1).cpu()
            clean_am = clean_am.squeeze(1).cpu()
            noisy_am = noisy_am.squeeze(1).cpu()
            clean_am_hat = clean_am_hat.squeeze(1).cpu()
                        
            wavs_dict = {
                "tm": tm,
                "noisy_am": noisy_am,
                "clean_am": clean_am,
                "clean_am_hat": clean_am_hat,
            }
            
            save_wavs(wavs_dict, os.path.join(outdir_wavs, id[0]))


if __name__=="__main__":
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
    tm_only = model_args.input_type == "tm"

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
                                         tm_only=tm_only)

    
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
        