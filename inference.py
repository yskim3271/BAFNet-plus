import os
import torch
import logging
import argparse
from omegaconf import OmegaConf

from models.tamenet import tamenet
from data import TAPSnoisytdataset, StepSampler
from torch.utils.data import DataLoader
from enhance import enhance
from evaluate import evaluate


def main(args):

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    device = args.device

    va_dataset = TAPSnoisytdataset(
        filelist=args.dset.valid,
        noise_dir = args.dset.noise_dir,
        noise_list=args.dset.noise_valid,
        rir_dir = args.dset.rir_dir,
        rir_list=args.dset.rir_valid,
        snr_range=args.valid_noise.snr_range,
        reverb_proportion=args.valid_noise.reverb_proportion,
        target_dB_FS=args.valid_noise.target_dB_FS,
        target_dB_FS_floating_value=args.valid_noise.target_dB_FS_floating_value,
        silence_length=args.valid_noise.silence_length,
        deterministic=args.valid_noise.deterministic,
        sampling_rate=args.sampling_rate,
        fileidx=True
    )
    
    va_loader = DataLoader(
        dataset=va_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    tt_dataset = TAPSnoisytdataset(
        filelist=args.dset.valid,
        noise_dir= args.dset.noise_dir,
        noise_list=args.dset.noise_valid,
        rir_dir = args.dset.rir_dir,
        rir_list=args.dset.rir_valid,
        snr_range=args.valid_noise.snr_range,
        reverb_proportion=args.valid_noise.reverb_proportion,
        target_dB_FS=args.valid_noise.target_dB_FS,
        target_dB_FS_floating_value=args.valid_noise.target_dB_FS_floating_value,
        silence_length=args.valid_noise.silence_length,
        deterministic=args.valid_noise.deterministic,
        sampling_rate=args.sampling_rate,
        fileidx=True
    )
    tt_loader = DataLoader(
        dataset=tt_dataset, 
        batch_size=1,
        sampler=StepSampler(len(tt_dataset), 100),
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    checkpoint_path = args.path
    
    checkpoint_pkg_best = os.path.join(checkpoint_path, "best.th")
    checkpoint_pkg_last = os.path.join(checkpoint_path, "checkpoint.th")
    
    model = tamenet(**args.tamenet).to(device)

    ### ------ best model ------ ###
    
    if args.best:
        model = model.load_state_dict(torch.load(checkpoint_pkg_best)['model'])
        
        with torch.no_grad():
            enhance(args, model, tt_loader, 0, logger, "best")
            evaluate(args, model, va_loader, 0, logger, "best")

    ### ------ last model ------ ###
    
    if args.last:
        model.load_state_dict(torch.load(checkpoint_pkg_last)['model'])
        
        with torch.no_grad():
            enhance(args, model, tt_loader, 0, logger, "last")
            evaluate(args, model, va_loader, 0, logger, "last")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance noisy speech")
    
    parser.add_argument("--best", action="store_true", help="Use best model")
    parser.add_argument("--last", action="store_true", help="Use last model")
    parser.add_argument("--path", type=str, help="Path to the checkpoint")
    option = parser.parse_args()

    config_path = os.path.join(option.path, ".hydra", "config.yaml")
    args = OmegaConf.load(config_path)
    args = OmegaConf.merge(args, OmegaConf.create(vars(option)))
    main(args)