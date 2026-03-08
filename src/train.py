import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import psutil
import hydra
import random
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from src.models.discriminator import MetricGANDiscriminator
import shutil
from src.data import Noise_Augmented_Dataset, StepSampler
from src.receptive_field import compute_receptive_field, rf_to_segment
from src.solver import Solver
from src.utils import load_model, load_model_config_from_checkpoint, parse_file_list

torch.backends.cudnn.benchmark = True


def _resolve_rf_param(args, logger):
    """Resolve the param dict to use for receptive field computation.

    For BAFNet (or any model with checkpoint_mapping/checkpoint_masking),
    load backbone params from each checkpoint, verify they produce the same RF,
    and return the shared param. For other models, return args.model.param directly.
    """
    param = args.model.param
    checkpoint_mapping = getattr(param, "checkpoint_mapping", None)
    checkpoint_masking = getattr(param, "checkpoint_masking", None)

    if checkpoint_mapping is None and checkpoint_masking is None:
        return param

    # Load backbone params from checkpoints
    configs = {}
    for name, ckpt_path in [("mapping", checkpoint_mapping), ("masking", checkpoint_masking)]:
        if ckpt_path is None:
            continue
        config = load_model_config_from_checkpoint(ckpt_path)
        configs[name] = config["param"]
        logger.info(f"[RF] Loaded {name} backbone param from: {ckpt_path}")

    if len(configs) < 2:
        # Only one backbone checkpoint provided, use it directly
        rf_param = next(iter(configs.values()))
        return rf_param

    # Both checkpoints provided — compute RF for each and verify they match
    names = list(configs.keys())
    rf_a = compute_receptive_field(configs[names[0]], sampling_rate=args.sampling_rate)
    rf_b = compute_receptive_field(configs[names[1]], sampling_rate=args.sampling_rate)

    if rf_a.total_rf_frames != rf_b.total_rf_frames:
        raise ValueError(
            f"Receptive field mismatch between backbone models: "
            f"{names[0]}={rf_a.total_rf_frames} frames ({rf_a.total_rf_ms:.1f}ms) vs "
            f"{names[1]}={rf_b.total_rf_frames} frames ({rf_b.total_rf_ms:.1f}ms). "
            f"Both backbones must have the same receptive field for BAFNet training."
        )

    logger.info(
        f"[RF] Both backbones have matching RF: {rf_a.total_rf_frames} frames ({rf_a.total_rf_ms:.1f}ms)"
    )
    return configs[names[0]]


def terminate_child_processes():
    current_process = psutil.Process(os.getpid())
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

def setup_logger(name):
    """Set up logger"""
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return logging.getLogger(name)

def run(args):
        
    # Create and initialize logger
    logger = setup_logger("train")

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_args = args.model
    model_lib = model_args.model_lib
    model_class_name = model_args.model_class

    # Prepare model params (skip pretrained loading if resuming from checkpoint)
    model_params = OmegaConf.to_container(model_args.param, resolve=True)
    if args.continue_from is not None and 'load_pretrained_weights' in model_params:
        model_params['load_pretrained_weights'] = False
        logger.info("[Resume] Skipping pretrained weights loading (will load from checkpoint)")

    # Load model
    model = load_model(model_lib, model_class_name, model_params, device)

    # Calculate and log the total number of parameters and model size
    logger.info(f"Selected model: {model_lib}.{model_class_name}")
    total_params = sum(p.numel() for p in model.parameters())
    model_params = (total_params) / 1000000
    logger.info(f"Model's parameters: {model_params:.2f} M")

    if args.save_code:
        # Use hydra.utils.to_absolute_path to get the correct path
        scripts_dir = os.path.dirname(hydra.utils.to_absolute_path(__file__))
        project_root = os.path.dirname(scripts_dir)
        src = os.path.join(project_root, "src", "models", f"{model_lib}.py")
        dest = f"./{model_lib}.py"
        
        if os.path.exists(src):
            shutil.copy2(src, dest)
            logger.info(f"Copied {src} to {dest}")
        else:
            logger.warning(f"Model file not found: {src}")

    if args.optim == "adam":
        optim_class = torch.optim.Adam
    elif args.optim == "adamW" or args.optim == "adamw":
        optim_class = torch.optim.AdamW

    discriminator = MetricGANDiscriminator().to(device)

    # optimizer
    if hasattr(args.model, 'finetune') and args.model.finetune.get('enabled', False):
        # ============================================================
        # Finetune: Differential LR for pretrained models
        # ============================================================
        finetune_cfg = args.model.finetune
        pretrained_lr = finetune_cfg.get('pretrained_lr', 0.0001)
        weight_decay = finetune_cfg.get('weight_decay', 0.01)

        # Collect pretrained parameters (mapping + masking)
        pretrained_params = list(model.mapping.parameters()) + list(model.masking.parameters())

        # Collect fusion module parameters (CNN-based fusion)
        fusion_params = []
        for i in range(model.conv_depth):
            fusion_params.extend(getattr(model, f"convblock_{i}").parameters())
        fusion_params.extend(model.sigmoid.parameters())

        # Build param_groups
        fusion_lr = finetune_cfg.get('fusion_lr', args.lr)
        param_groups = [
            {
                'params': pretrained_params,
                'lr': pretrained_lr,
                'weight_decay': 0.0,  # No decay for pretrained (preserve features)
                'name': 'pretrained'
            },
            {
                'params': fusion_params,
                'lr': fusion_lr,
                'weight_decay': weight_decay,
                'name': 'fusion'
            }
        ]

        optim = optim_class(param_groups, betas=args.betas)

        # Log configuration
        logger.info(f"[Finetune] Created {args.optim} with differential learning rates:")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            logger.info(f"  {group['name']:15s}: LR={group['lr']:.2e}, WD={group['weight_decay']:.4f}, Params={num_params:,}")
    else:
        # Standard optimizer
        optim = optim_class(model.parameters(), lr=args.lr, betas=args.betas)

    optim_disc = optim_class(discriminator.parameters(), lr=args.lr, betas=args.betas)
    
    # scheduler
    scheduler = None
    scheduler_disc = None

    if args.lr_decay is not None:
        # Determine warmup settings from finetune config
        warmup_epochs = 0
        warmup_start_factor = 0.2

        if hasattr(args.model, 'finetune') and args.model.finetune.get('enabled', False):
            warmup_epochs = args.model.finetune.get('warmup_epochs', 0)
            warmup_start_factor = args.model.finetune.get('warmup_start_factor', 0.2)

        if warmup_epochs > 0:
            logger.info(f"[Scheduler] Warmup enabled: {warmup_epochs} epochs (start_factor={warmup_start_factor})")

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=warmup_start_factor, total_iters=warmup_epochs
            )
            exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optim, gamma=args.lr_decay, last_epoch=-1
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optim, schedulers=[warmup_scheduler, exp_scheduler], milestones=[warmup_epochs]
            )
        else:
            # Standard scheduler without warmup
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.lr_decay, last_epoch=-1)

        # Discriminator scheduler (no warmup)
        scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(optim_disc, gamma=args.lr_decay, last_epoch=-1)

    # Load dataset from Huggingface
    dset = args.dset.get("dataset", "TAPS")
    logger.info(f"Dataset selected: {dset}")
    if dset == "TAPS":
        _dataset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset")
    elif dset == "Vibravox":
        _dataset = load_dataset("yskim3271/vibravox_16k")

    trainset = _dataset['train']
    validset = _dataset['dev']
    testset = _dataset['test']

    testset_list = [testset] * args.dset.test_augment_numb
    testset = concatenate_datasets(testset_list)
    
    bcs_only = "acs" not in args.model.input_type

    if bcs_only:
        noise_train_list, noise_valid_list, noise_test_list = [], [], []
        rir_train_list, rir_valid_list, rir_test_list = [], [], []
    else:
        noise_train_list = parse_file_list(args.dset.noise_dir, args.dset.noise_train)
        noise_valid_list = parse_file_list(args.dset.noise_dir, args.dset.noise_valid)
        noise_test_list = parse_file_list(args.dset.noise_dir, args.dset.noise_test)
        rir_train_list = parse_file_list(args.dset.rir_dir, args.dset.rir_train)
        rir_valid_list = parse_file_list(args.dset.rir_dir, args.dset.rir_valid)
        rir_test_list = parse_file_list(args.dset.rir_dir, args.dset.rir_test)

    # Determine training segment size
    raw_segment = args.segment
    if str(raw_segment) == "auto":
        rf_param = _resolve_rf_param(args, logger)
        rf_samples = rf_to_segment(rf_param, sampling_rate=args.sampling_rate)
        segment = rf_samples * 2
        rf = compute_receptive_field(rf_param, sampling_rate=args.sampling_rate)
        logger.info(
            f"Auto segment from RF: {rf.total_rf_frames} frames = "
            f"{rf.total_rf_samples} samples -> aligned RF={rf_samples} samples, "
            f"segment={segment} samples ({segment / args.sampling_rate * 1000:.1f} ms)"
        )
    else:
        segment = int(raw_segment)

    if str(args.stride) == "auto":
        stride = rf_samples
    else:
        stride = int(args.stride)

    if str(args.shift) == "auto":
        shift = rf_samples
    else:
        shift = int(args.shift)

    logger.info(f"Training segment={segment}, stride={stride}, shift={shift}")

    # Set up dataset and dataloader
    tr_dataset = Noise_Augmented_Dataset(
        datapair_list=trainset,
        noise_list=noise_train_list,
        rir_list=rir_train_list,
        snr_range=args.train_noise.snr_range,
        reverb_proportion=args.train_noise.reverb_proportion,
        target_dB_FS=args.train_noise.target_dB_FS,
        target_dB_FS_floating_value=args.train_noise.target_dB_FS_floating_value,
        silence_length=args.train_noise.silence_length,
        sampling_rate=args.sampling_rate,
        segment=segment,
        stride=stride,
        shift=shift,
        bcs_only=bcs_only
    )

    tr_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

        
    # Set up validation and test dataset and dataloader
    va_dataset = Noise_Augmented_Dataset(
        datapair_list= validset,
        noise_list= noise_valid_list,
        rir_list= rir_valid_list,
        snr_range=args.valid_noise.snr_range,
        reverb_proportion=args.valid_noise.reverb_proportion,
        target_dB_FS=args.valid_noise.target_dB_FS,
        target_dB_FS_floating_value=args.valid_noise.target_dB_FS_floating_value,
        silence_length=args.valid_noise.silence_length,
        deterministic=args.valid_noise.deterministic,
        sampling_rate=args.sampling_rate,
        bcs_only=bcs_only,
    )
    va_loader = DataLoader(
        dataset=va_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    ev_loader_list = {}
    tt_loader_list = {}
    
    for fixed_snr in args.test_noise.snr_step:
        ev_dataset = Noise_Augmented_Dataset(
            datapair_list= testset,
            noise_list= noise_test_list,
            rir_list= rir_test_list,
            snr_range= [fixed_snr, fixed_snr],
            reverb_proportion=args.test_noise.reverb_proportion,
            target_dB_FS=args.test_noise.target_dB_FS,
            target_dB_FS_floating_value=args.test_noise.target_dB_FS_floating_value,
            silence_length=args.test_noise.silence_length,
            deterministic=args.test_noise.deterministic,
            sampling_rate=args.sampling_rate,
            with_id=True,
            with_text=True,
            bcs_only=bcs_only,
        )

        ev_loader = DataLoader(
            dataset=ev_dataset, 
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
        tt_loader = DataLoader(
            dataset=ev_dataset, 
            batch_size=1,
            sampler=StepSampler(len(ev_dataset), 100),
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        ev_loader_list[f"{fixed_snr}"] = ev_loader
        tt_loader_list[f"{fixed_snr}"] = tt_loader
    
    dataloader = {
        "tr_loader": tr_loader,
        "va_loader": va_loader,
        "ev_loader_list": ev_loader_list,
        "tt_loader_list": tt_loader_list,
    }

    # Solver
    solver = Solver(
        data=dataloader,
        model=model,
        discriminator=discriminator,
        optim=optim,
        optim_disc=optim_disc,
        scheduler=scheduler,
        scheduler_disc=scheduler_disc,
        args=args,
        logger=logger,
        device=device
    )
    solver.train()
    sys.exit(0)

def _main(args):
    global __file__

    logger = setup_logger("main")

    for key, value in args.dset.items():
        if isinstance(value, str) and key != "dataset":
            args.dset[key] = hydra.utils.to_absolute_path(value)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)

    run(args)

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(args):
    logger = setup_logger("main")
    try:
        _main(args)
    except KeyboardInterrupt:
        logger.info("Training stopped by user")
        terminate_child_processes()
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error occurred in main: {str(e)}")
        terminate_child_processes()
        sys.exit(1)

if __name__ == "__main__":
    main()