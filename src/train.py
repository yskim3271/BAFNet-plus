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
from src.utils import load_model, parse_file_list

torch.backends.cudnn.benchmark = True


def create_kfold_splits(dataset, fold_index, num_folds=5,
                        held_out_size=5587, dev_ratio=0.4516,
                        shuffle=True, seed=2039):
    """Speaker-disjoint K-Fold Cross Validation 데이터 분할.

    Sorted-greedy bin packing 알고리즘으로 화자 단위 분할을 수행하여,
    동일 화자가 train과 held-out에 동시에 존재하는 data leakage를 방지한다.

    Args:
        dataset: 전체 HuggingFace dataset (train+dev+test 합친 것)
        fold_index: 현재 fold (1-based, 1~num_folds)
        num_folds: 전체 fold 수
        held_out_size: fold 1~(num_folds-1)의 target held-out 크기
        dev_ratio: held-out 내 dev 비율
        shuffle: 화자 리스트 셔플 여부
        seed: 랜덤 시드

    Returns:
        trainset, devset, testset, fold_info (재현성 검증용 정보)
    """
    from collections import defaultdict

    total = len(dataset)
    speaker_ids = dataset["speaker_id"]

    # 1. speaker → [indices] 매핑 구축
    speaker_to_indices = defaultdict(list)
    for idx, spk in enumerate(speaker_ids):
        speaker_to_indices[spk].append(idx)

    speakers = list(speaker_to_indices.keys())
    total_speakers = len(speakers)

    # 2. seed 기반 셔플 (tie-breaking용)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(speakers)

    # 3. 샘플 수 내림차순 stable sort (셔플 순서 보존)
    speakers.sort(key=lambda s: len(speaker_to_indices[s]), reverse=True)

    # 4. Fold별 target 크기 계산
    fold_targets = []
    for i in range(num_folds):
        if i < num_folds - 1:
            fold_targets.append(held_out_size)
        else:
            fold_targets.append(total - held_out_size * (num_folds - 1))

    # 5. Sorted-greedy bin packing: 각 화자를 (target - current) 차이가 가장 큰 bin에 배정
    fold_speakers = [[] for _ in range(num_folds)]
    fold_sizes = [0] * num_folds

    for spk in speakers:
        spk_size = len(speaker_to_indices[spk])
        # (target - current) 차이가 가장 큰 bin 선택
        best_fold = max(range(num_folds), key=lambda f: fold_targets[f] - fold_sizes[f])
        fold_speakers[best_fold].append(spk)
        fold_sizes[best_fold] += spk_size

    # 6. 현재 fold의 held-out / train 분리
    held_out_spks = fold_speakers[fold_index - 1]
    train_spks = []
    for i in range(num_folds):
        if i != fold_index - 1:
            train_spks.extend(fold_speakers[i])

    # held-out 인덱스 수집
    held_out_indices = []
    for spk in held_out_spks:
        held_out_indices.extend(speaker_to_indices[spk])

    train_indices = []
    for spk in train_spks:
        train_indices.extend(speaker_to_indices[spk])

    # 7. held-out 화자를 dev/test로 speaker 단위 분할
    held_out_total = len(held_out_indices)
    dev_target = int(held_out_total * dev_ratio)

    dev_spks = []
    test_spks = []
    dev_count = 0

    # 화자 리스트를 seed 기반으로 셔플 후 누적 할당
    held_out_spks_shuffled = list(held_out_spks)
    rng.shuffle(held_out_spks_shuffled)

    for spk in held_out_spks_shuffled:
        spk_size = len(speaker_to_indices[spk])
        if dev_count + spk_size <= dev_target or len(dev_spks) == 0:
            dev_spks.append(spk)
            dev_count += spk_size
        else:
            test_spks.append(spk)

    dev_indices = []
    for spk in dev_spks:
        dev_indices.extend(speaker_to_indices[spk])

    test_indices = []
    for spk in test_spks:
        test_indices.extend(speaker_to_indices[spk])

    # Dataset 생성
    trainset = dataset.select(train_indices)
    devset = dataset.select(dev_indices)
    testset = dataset.select(test_indices)

    # speaker_to_fold 매핑 생성
    speaker_to_fold = {}
    for fi in range(num_folds):
        for spk in fold_speakers[fi]:
            speaker_to_fold[spk] = fi + 1  # 1-based

    # 재현성 검증용 정보
    fold_info = {
        "fold_index": fold_index,
        "num_folds": num_folds,
        "seed": seed,
        "shuffle": shuffle,
        "total_samples": total,
        "total_speakers": total_speakers,
        "train_size": len(trainset),
        "dev_size": len(devset),
        "test_size": len(testset),
        "held_out_size_config": held_out_size,
        "held_out_size_actual": len(held_out_indices),
        "dev_ratio": dev_ratio,
        "all_fold_sizes": fold_sizes,
        "held_out_speakers": sorted(held_out_spks),
        "dev_speakers": sorted(dev_spks),
        "test_speakers": sorted(test_spks),
        "speaker_to_fold": speaker_to_fold,
    }

    return trainset, devset, testset, fold_info


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

        # Collect fusion module parameters (BAFNet v1 or v2)
        fusion_params = []
        if hasattr(model, 'conv_depth'):
            # BAFNet v1: CNN-based fusion
            for i in range(model.conv_depth):
                fusion_params.extend(getattr(model, f"convblock_{i}").parameters())
            fusion_params.extend(model.sigmoid.parameters())
            fusion_name = 'bafnet_cnn'
        elif hasattr(model, 'fusion'):
            # BAFNet v2: Mamba-based fusion
            fusion_params.extend(model.fusion.parameters())
            fusion_name = 'bafnet_fusion'
        else:
            raise ValueError("Model must have either 'conv_depth' (v1) or 'fusion' (v2) attribute")

        # Build param_groups
        fusion_lr = finetune_cfg.get('fusion_lr', finetune_cfg.get('cnn_lr', args.lr))
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
                'name': fusion_name
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

    # K-Fold Cross Validation
    cv_config = args.get("cv", {})
    if cv_config.get("enabled", False):
        # 전체 데이터 합치기
        all_data = concatenate_datasets([
            _dataset['train'],
            _dataset['dev'],
            _dataset['test']
        ])
        logger.info(f"[K-Fold CV] Fold {cv_config.fold_index}/{cv_config.num_folds}")
        logger.info(f"[K-Fold CV] Total samples: {len(all_data)}")

        # Fold별 분할
        trainset, validset, testset, fold_info = create_kfold_splits(
            dataset=all_data,
            fold_index=cv_config.fold_index,
            num_folds=cv_config.num_folds,
            held_out_size=cv_config.get("held_out_size", 5587),
            dev_ratio=cv_config.get("dev_ratio", 0.4516),
            shuffle=cv_config.get("shuffle", True),
            seed=args.seed
        )

        logger.info(f"[K-Fold CV] Train: {len(trainset)}, Dev: {len(validset)}, Test: {len(testset)}")

        # fold 정보 저장 (재현성 검증용)
        fold_info_path = os.path.join(os.getcwd(), "fold_info.yaml")
        OmegaConf.save(OmegaConf.create(fold_info), fold_info_path)
        logger.info(f"[K-Fold CV] Fold info saved to: {fold_info_path}")
    else:
        # 기존 방식 (원본 스플릿 사용)
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
        segment = rf_to_segment(args.model.param, sampling_rate=args.sampling_rate)
        rf = compute_receptive_field(args.model.param, sampling_rate=args.sampling_rate)
        logger.info(
            f"Auto segment from RF: {rf.total_rf_frames} frames = "
            f"{rf.total_rf_samples} samples -> aligned to {segment} samples "
            f"({segment / args.sampling_rate * 1000:.1f} ms)"
        )
    else:
        segment = int(raw_segment)

    if str(args.stride) == "auto":
        stride = segment // 2
    else:
        stride = int(args.stride)

    if str(args.shift) == "auto":
        shift = segment // 2
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