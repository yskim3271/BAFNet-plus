import os
import sys
import logging
import psutil
import importlib
import hydra
import random
import torch
import torch.distributed as dist
import numpy as np
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from models.discriminator import MetricGAN_Discriminator

from data import TAPSnoisytdataset, StepSampler
from solver import Solver

def kill_child_processes():
    """kill child processes"""
    current_process = psutil.Process(os.getpid())
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

def setup_logger(name, rank=None):
    """Set up logger"""
    if rank == 0:
        hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    else:
        logging.basicConfig(level=logging.ERROR)
        
    return logging.getLogger(name)

def setup_distributed(rank, world_size, args):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = str(args.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(args.ddp.master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_list(dir, file):
    return [os.path.join(dir, line.strip()) for line in open(file, "r")]

def run(rank, world_size, args):
        
    # Create and initialize logger
    logger = setup_logger("train", rank)

    # Set up distributed training environment
    if world_size > 1:
        setup_distributed(rank, world_size, args)
    
    if rank == 0:
        logger.info(f"Training with {world_size} GPUs")
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    
    model_args = args.model
    model_lib = model_args.model_lib
    model_class = model_args.model_class
    
    # import model library
    module = importlib.import_module("models." + model_lib)
    model_class = getattr(module, model_class)
    
    model = model_class(**model_args.param)
    model = model.to(args.device)

    if rank == 0:
        # Calculate and log the total number of parameters and model size
        logger.info(f"Selected model: {model_lib}.{model_class}")
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = (total_params) / (1024 * 1024)
        logger.info(f"Model's size: {model_size_mb:.2f} MB")

    discriminator = None
    optim_disc = None

    if args.optim == "adam":
        optim_class = torch.optim.Adam
    elif args.optim == "adamW" or args.optim == "adamw":
        optim_class = torch.optim.AdamW
    

    metricganloss_cfg = args.loss.get("metricgan_loss")

    if metricganloss_cfg is not None:
        if world_size > 1 and not metricganloss_cfg.get("use_torch_pesq", False):
            # Traditional PESQ calculation can occur deadlock with DDP. Use torch_pesq instead.
            logger.error("MetricGAN Loss with traditional PESQ cannot be used with DDP. Please use a single GPU.")
            cleanup()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            sys.exit(0)
            
        discriminator = MetricGAN_Discriminator(ndf=metricganloss_cfg.ndf)
        discriminator = discriminator.to(args.device)
        del metricganloss_cfg.ndf
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
        if discriminator is not None:
            discriminator = DDP(discriminator, device_ids=[rank])
    
    # optimizer
    optim = optim_class(model.parameters(), lr=args.lr, betas=args.betas)
    if discriminator is not None:
        optim_disc = optim_class(discriminator.parameters(), lr=args.lr, betas=args.betas)

    scheduler = None
    scheduler_disc = None

    if args.lr_decay is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.lr_decay, last_epoch=-1)
        if discriminator is not None:
            # Use a separate scheduler for the discriminator
            scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(optim_disc, gamma=args.lr_decay, last_epoch=-1)

    # Load dataset
    if rank == 0:
        taps_dataset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset")
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        taps_dataset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset")
    
    trainset = taps_dataset['train']
    validset = taps_dataset['dev']
    testset = taps_dataset['test']
    testset_list = [testset] * args.dset.test_augment_numb
    testset = concatenate_datasets(testset_list)
        
    noise_train_list = parse_list(args.dset.noise_dir, args.dset.noise_train)
    noise_valid_list = parse_list(args.dset.noise_dir, args.dset.noise_valid)
    noise_test_list = parse_list(args.dset.noise_dir, args.dset.noise_test)
    rir_train_list = parse_list(args.dset.rir_dir, args.dset.rir_train)
    rir_valid_list = parse_list(args.dset.rir_dir, args.dset.rir_valid)
    rir_test_list = parse_list(args.dset.rir_dir, args.dset.rir_test)
    
    tm_only = args.model.input_type == "tm"
        
    # Set up dataset and dataloader
    tr_dataset = TAPSnoisytdataset(
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
    
    # Set up distributed sampler
    tr_sampler = DistributedSampler(tr_dataset) if world_size > 1 else None
    tr_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        sampler=tr_sampler,
        shuffle=(tr_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True
    )
        
    # Set up validation and test dataset and dataloader
    va_dataset = TAPSnoisytdataset(
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
    va_sampler = DistributedSampler(va_dataset, shuffle=False) if world_size > 1 else None
    va_loader = DataLoader(
        dataset=va_dataset, 
        batch_size=1,
        sampler=va_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    ev_loader_list = {}
    tt_loader_list = {}
    
    for fixed_snr in args.test_noise.snr_step:
        ev_dataset = TAPSnoisytdataset(
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
        "tr_sampler": tr_sampler,
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
        rank=rank,
        world_size=world_size,
        device=device
    )
    solver.train()
    
    cleanup()
        

def _main(args):
    global __file__

    logger = setup_logger("main")
    
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
            
    __file__ = hydra.utils.to_absolute_path(__file__)
    
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        import torch.multiprocessing as mp
        try:
            mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
        except KeyboardInterrupt:
            logger.info("Training stopped by user")
            kill_child_processes()
        except Exception as e:
            logger.exception(f"Error occurred in spawn: {str(e)}")
            kill_child_processes()
    else:
        run(0, 1, args)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(args):
    logger = setup_logger("main")
    try:
        _main(args)
    except KeyboardInterrupt:
        logger.info("Training stopped by user")
        kill_child_processes()
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error occurred in main: {str(e)}")
        kill_child_processes()
        sys.exit(1)

if __name__ == "__main__":
    main()