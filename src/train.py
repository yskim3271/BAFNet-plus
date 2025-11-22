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
from src.models.discriminator import MetricGAN_Discriminator
import shutil
from src.data import Noise_Augmented_Dataset, StepSampler
from src.solver import Solver
from src.utils import load_model, parse_file_list

torch.backends.cudnn.benchmark = True

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

    # Load model using utility function
    model = load_model(model_lib, model_class_name, model_args.param, device)

    # LoRA injection (if enabled in config)
    if hasattr(args.model, 'lora') and args.model.lora.get('enabled', False):
        from src.models.lora import inject_lora_into_model, get_lora_parameters

        logger.info(f"[LoRA] Injecting LoRA adapters (rank={args.model.lora.rank})")

        # Inject LoRA into the model
        num_lora_params = inject_lora_into_model(
            model,
            target_modules=args.model.lora.get('target_modules', None),
            rank=args.model.lora.rank,
            alpha=args.model.lora.alpha,
            dropout=args.model.lora.get('dropout', 0.0)
        )
        logger.info(f"[LoRA] Added {num_lora_params:,} trainable LoRA parameters")

        # Freeze all non-LoRA parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters
        for param in get_lora_parameters(model):
            param.requires_grad = True

        # Unfreeze CNN layers (alpha blending network in BAFNet)
        if hasattr(model, 'conv_depth'):
            for i in range(model.conv_depth):
                for param in getattr(model, f"convblock_{i}").parameters():
                    param.requires_grad = True
            for param in model.sigmoid.parameters():
                param.requires_grad = True
            logger.info(f"[LoRA] Unfroze {model.conv_depth} CNN blocks for training")

        # Log parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[LoRA] Total params: {total_params:,}, Trainable: {trainable_params:,} "
                    f"({100*trainable_params/total_params:.2f}%)")

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

    discriminator = MetricGAN_Discriminator().to(device)

    # optimizer
    if hasattr(args.model, 'lora') and args.model.lora.get('enabled', False):
        # Different learning rates for LoRA vs CNN
        from src.models.lora import get_lora_parameters

        lora_params = get_lora_parameters(model)
        cnn_params = []

        # Collect CNN parameters (if BAFNet)
        if hasattr(model, 'conv_depth'):
            for i in range(model.conv_depth):
                cnn_params.extend(getattr(model, f"convblock_{i}").parameters())
            cnn_params.extend(model.sigmoid.parameters())

        param_groups = [
            {'params': lora_params, 'lr': args.model.lora.lr, 'name': 'lora'},
            {'params': cnn_params, 'lr': args.lr, 'name': 'cnn'}
        ]

        optim = optim_class(param_groups, betas=args.betas)
        logger.info(f"[Optimizer] Created {args.optim} with different LRs:")
        logger.info(f"  LoRA LR: {args.model.lora.lr}")
        logger.info(f"  CNN LR: {args.lr}")
    else:
        # Standard optimizer for non-LoRA models
        optim = optim_class(model.parameters(), lr=args.lr, betas=args.betas)

    optim_disc = optim_class(discriminator.parameters(), lr=args.lr, betas=args.betas)
    
    # scheduler
    scheduler = None
    scheduler_disc = None

    if args.lr_decay is not None:
        # Add warmup for LoRA fine-tuning
        if hasattr(args.model, 'lora') and args.model.lora.get('enabled', False):
            warmup_epochs = args.model.lora.get('warmup_epochs', 5)
            warmup_start_factor = args.model.lora.get('warmup_start_factor', 0.2)

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
            # Standard scheduler for non-LoRA models
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
    
    noise_train_list = parse_file_list(args.dset.noise_dir, args.dset.noise_train)
    noise_valid_list = parse_file_list(args.dset.noise_dir, args.dset.noise_valid)
    noise_test_list = parse_file_list(args.dset.noise_dir, args.dset.noise_test)
    rir_train_list = parse_file_list(args.dset.rir_dir, args.dset.rir_train)
    rir_valid_list = parse_file_list(args.dset.rir_dir, args.dset.rir_valid)
    rir_test_list = parse_file_list(args.dset.rir_dir, args.dset.rir_test)
    
    tm_only = args.model.input_type == "tm"

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
        segment=args.segment, 
        stride=args.stride, 
        shift=args.shift,
        tm_only=tm_only
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
        tm_only=tm_only,
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
            tm_only=tm_only,
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