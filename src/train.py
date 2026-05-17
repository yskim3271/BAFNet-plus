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
from src.receptive_field import compute_receptive_field, resolve_rf_param, rf_to_segment
from src.solver import Solver
from src.checkpoint import load_model
from src.utils import parse_file_list

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


def _get_optim_class(name: str):
    if name == "adam":
        return torch.optim.Adam
    if name in ("adamW", "adamw"):
        return torch.optim.AdamW
    raise ValueError(f"Unsupported optim: {name}")


def _build_optimizer(args, model, optim_class, logger):
    """Create the main optimizer with optional pretrained/fusion differential LRs."""
    finetune_enabled = hasattr(model, 'mapping') and hasattr(model, 'masking')
    if finetune_enabled:
        ft = args.finetune
        pretrained_ids = {id(p) for p in model.mapping.parameters()} \
                       | {id(p) for p in model.masking.parameters()}
        pretrained_params = [p for p in model.parameters() if id(p) in pretrained_ids]
        fusion_params = [p for p in model.parameters() if id(p) not in pretrained_ids]
        if not fusion_params:
            logger.warning("[Finetune] No fusion parameters found — all params belong to pretrained submodules.")

        param_groups = [
            {'params': pretrained_params, 'lr': ft.pretrained_lr,
             'weight_decay': ft.pretrained_weight_decay, 'name': 'pretrained'},
            {'params': fusion_params, 'lr': ft.fusion_lr,
             'weight_decay': ft.fusion_weight_decay, 'name': 'fusion'},
        ]
        optim = optim_class(param_groups, betas=args.betas)

        logger.info(f"[Finetune] Created {args.optim} with differential learning rates:")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            logger.info(f"  {group['name']:15s}: LR={group['lr']:.2e}, WD={group['weight_decay']:.4f}, Params={num_params:,}")
        return optim

    return optim_class(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=args.betas,
    )


def _build_cosine_warmup_scheduler(optimizer, args):
    """Cosine Annealing scheduler with optional linear warmup prefix."""
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=args.warmup_start_factor, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.eta_min)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min)


def _resolve_segment_and_stride(args, logger):
    """Resolve (segment, stride, shift) samples, honoring ``auto`` via RF."""
    rf_samples = None
    raw_segment = args.segment
    if str(raw_segment) == "auto":
        rf_param = resolve_rf_param(args, logger)
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

    stride = rf_samples if str(args.stride) == "auto" else int(args.stride)
    shift = rf_samples if str(args.shift) == "auto" else int(args.shift)
    logger.info(f"Training segment={segment}, stride={stride}, shift={shift}")
    return segment, stride, shift


def _build_dataloaders(args, trainset, validset, testset, segment, stride, shift, bcs_only):
    """Build train / valid / per-SNR evaluation DataLoaders."""
    mix_strategy = args.dset.get("mix_strategy", "snr_random")
    is_vibravox_native = (mix_strategy == "vibravox_native")

    if bcs_only or is_vibravox_native:
        noise_train_list = noise_valid_list = noise_test_list = []
        rir_train_list = rir_valid_list = rir_test_list = []
    else:
        noise_train_list = parse_file_list(args.dset.noise_dir, args.dset.noise_train)
        noise_valid_list = parse_file_list(args.dset.noise_dir, args.dset.noise_valid)
        noise_test_list = parse_file_list(args.dset.noise_dir, args.dset.noise_test)
        rir_train_list = parse_file_list(args.dset.rir_dir, args.dset.rir_train)
        rir_valid_list = parse_file_list(args.dset.rir_dir, args.dset.rir_valid)
        rir_test_list = parse_file_list(args.dset.rir_dir, args.dset.rir_test)

    tr_noise_dataset = va_noise_dataset = None
    if is_vibravox_native:
        tr_noise_dataset = load_dataset(
            args.dset.noise_hf_dataset,
            name=args.dset.noise_subset_train,
            split="train",
        )
        va_noise_dataset = load_dataset(
            args.dset.noise_hf_dataset,
            name=args.dset.noise_subset_valid,
            split="dev",
        )

    noise_sensor = args.dset.get("noise_sensor", "throat_microphone")
    skip_db_fs_normalize = bool(args.dset.get("skip_db_fs_normalize", False))

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
        bcs_only=bcs_only,
        bcs_gain_perturbation_db=getattr(args.train_noise, 'bcs_gain_perturbation_db', 0),
        mix_strategy=mix_strategy,
        noise_dataset=tr_noise_dataset,
        noise_sensor=noise_sensor,
        skip_db_fs_normalize=skip_db_fs_normalize,
    )
    tr_loader = DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    va_dataset = Noise_Augmented_Dataset(
        datapair_list=validset,
        noise_list=noise_valid_list,
        rir_list=rir_valid_list,
        snr_range=args.valid_noise.snr_range,
        reverb_proportion=args.valid_noise.reverb_proportion,
        target_dB_FS=args.valid_noise.target_dB_FS,
        target_dB_FS_floating_value=args.valid_noise.target_dB_FS_floating_value,
        silence_length=args.valid_noise.silence_length,
        deterministic=args.valid_noise.deterministic,
        sampling_rate=args.sampling_rate,
        bcs_only=bcs_only,
        mix_strategy=mix_strategy,
        noise_dataset=va_noise_dataset,
        noise_sensor=noise_sensor,
        skip_db_fs_normalize=skip_db_fs_normalize,
    )
    va_loader = DataLoader(
        dataset=va_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    ev_loader_list, tt_loader_list = {}, {}
    for fixed_snr in args.test_noise.snr_step:
        ev_dataset = Noise_Augmented_Dataset(
            datapair_list=testset,
            noise_list=noise_test_list,
            rir_list=rir_test_list,
            snr_range=[fixed_snr, fixed_snr],
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
        ev_loader_list[f"{fixed_snr}"] = DataLoader(
            dataset=ev_dataset, batch_size=1,
            num_workers=args.num_workers, pin_memory=True,
        )
        tt_loader_list[f"{fixed_snr}"] = DataLoader(
            dataset=ev_dataset, batch_size=1,
            sampler=StepSampler(len(ev_dataset), 100),
            num_workers=args.num_workers, pin_memory=True,
        )

    return {
        "tr_loader": tr_loader,
        "va_loader": va_loader,
        "ev_loader_list": ev_loader_list,
        "tt_loader_list": tt_loader_list,
    }


def _copy_model_source(model_lib, logger):
    """Copy the model source file next to Hydra's run directory."""
    scripts_dir = os.path.dirname(hydra.utils.to_absolute_path(__file__))
    project_root = os.path.dirname(scripts_dir)
    src = os.path.join(project_root, "src", "models", f"{model_lib}.py")
    dest = f"./{model_lib}.py"
    if os.path.exists(src):
        shutil.copy2(src, dest)
        logger.info(f"Copied {src} to {dest}")
    else:
        logger.warning(f"Model file not found: {src}")


def _build_qat_components(args, model, device, logger):
    """Build QAT-mode (model, teacher, kd_loss) by wrapping ``model`` via PT2E.

    Returns:
        (prepared_student, teacher_or_None, kd_loss_or_None,
         warm_start_summary_or_None) — ``teacher`` and ``kd_loss`` are None
        when ``args.kd.enabled`` is False (KD can be disabled for ablation).
    """
    from src.models.streaming.qat import (
        KDLoss,
        prepare_bafnetplus_for_qat,
        warm_start_from_ptq,
        QNNQuantizer,
    )
    from src.models.streaming.qat.kd import DEFAULT_FEATURE_LAYERS, DEFAULT_WEIGHTS

    qat_cfg = args.qat
    kd_cfg = getattr(args, "kd", None)

    # Build example inputs matching BAFNet+ forward signature: tuple of two
    # complex spectrograms shaped [B, F, T, 2]. The PT2E exported FX graph
    # specializes B to the example batch size (Dim.AUTO collapses internal
    # ``view(B*F, …)`` reshapes), so ``example_batch_size`` MUST equal the
    # training ``batch_size`` — otherwise the first batch crashes with a
    # shape mismatch. T is dynamic (cycle 18a fix in ``prepare.py``).
    import torch as _torch
    n_fft = int(getattr(args, "n_fft", 400))
    freq = n_fft // 2 + 1
    time_frames = int(getattr(qat_cfg, "example_time_frames", 400))
    example_batch = getattr(qat_cfg, "example_batch_size", None)
    if example_batch is None:
        example_batch = args.batch_size
    example_batch = int(example_batch)
    bcs_example = _torch.zeros(example_batch, freq, time_frames, 2, device=device)
    acs_example = _torch.zeros(example_batch, freq, time_frames, 2, device=device)
    example_inputs = ((bcs_example, acs_example),)
    logger.info(f"[QAT] PT2E example_inputs: batch={example_batch}, time_frames={time_frames}")

    quantizer = QNNQuantizer(
        skip_phase_conv=bool(getattr(qat_cfg, "skip_phase_conv", True)),
        skip_head=bool(getattr(qat_cfg, "skip_head", True)),
    )
    logger.info(f"[QAT] Quantizer config: {quantizer.summary()}")

    prepared_student = prepare_bafnetplus_for_qat(
        model=model,
        example_inputs=example_inputs,
        quantizer=quantizer,
    )
    logger.info(
        f"[QAT] PT2E prep: annotated={len(quantizer.annotated_node_names)} "
        f"skipped={len(quantizer.skipped_node_names)} (Conv/Linear ops)"
    )

    warm_start_summary = None
    ptq_path = getattr(qat_cfg, "ptq_onnx_path", None)
    if ptq_path:
        ptq_abs = hydra.utils.to_absolute_path(ptq_path)
        if os.path.exists(ptq_abs):
            warm_start_summary = warm_start_from_ptq(
                prepared_student, ptq_abs,
                verbose=bool(getattr(qat_cfg, "warm_start_verbose", False)),
            )
            logger.info(
                f"[QAT warm-start] mapping_rate={warm_start_summary['mapping_rate']:.1%} "
                f"({warm_start_summary['num_matched']}/{warm_start_summary['num_pt_total']} "
                f"PT fake-quants matched; {len(warm_start_summary['unused_onnx'])} ONNX keys unused)"
            )
        else:
            logger.warning(f"[QAT warm-start] PTQ ONNX not found at {ptq_abs}; skipping warm-start (cold init)")

    teacher = None
    kd_loss = None
    if kd_cfg is not None and bool(getattr(kd_cfg, "enabled", False)):
        teacher_args = kd_cfg.get("teacher_model_args", None)
        teacher_ckpt = getattr(kd_cfg, "teacher_ckpt", None)
        if teacher_args is None:
            # Fallback: re-use the student's model spec (BAFNet+ class + same params).
            teacher_args = OmegaConf.create({
                "model_lib": args.model.model_lib,
                "model_class": args.model.model_class,
                "param": args.model.param,
            })
        teacher_params = OmegaConf.to_container(teacher_args.param, resolve=True)
        teacher_params.setdefault("load_pretrained_weights", False)
        teacher = load_model(
            teacher_args.model_lib, teacher_args.model_class, teacher_params, device
        )
        if teacher_ckpt is not None:
            teacher_ckpt_abs = hydra.utils.to_absolute_path(teacher_ckpt)
            ckpt_pkg = torch.load(teacher_ckpt_abs, map_location=device, weights_only=False)
            state_dict = ckpt_pkg["model"] if isinstance(ckpt_pkg, dict) and "model" in ckpt_pkg else ckpt_pkg
            teacher.load_state_dict(state_dict)
            logger.info(f"[QAT KD] Teacher loaded from {teacher_ckpt_abs}")
        else:
            logger.warning("[QAT KD] No teacher_ckpt; teacher uses freshly-built FP32 weights (likely wrong)")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        feature_layers = list(kd_cfg.get("feature_layers", DEFAULT_FEATURE_LAYERS))
        weights = OmegaConf.to_container(kd_cfg.get("weights", OmegaConf.create(DEFAULT_WEIGHTS)), resolve=True)
        kd_loss = KDLoss(feature_layers=feature_layers, weights=weights).to(device)
        hook_hits = kd_loss.register_hooks(prepared_student, teacher)
        logger.info(
            f"[QAT KD] Hook hits: student={len(hook_hits['student_hit'])}, "
            f"teacher={len(hook_hits['teacher_hit'])} (feature-level KD will be sparse "
            f"on PT2E student — output-level dominates)"
        )

    return prepared_student, teacher, kd_loss, warm_start_summary


def run(args):
    logger = setup_logger("train")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_args = args.model
    model_lib = model_args.model_lib
    model_class_name = model_args.model_class

    model_params = OmegaConf.to_container(model_args.param, resolve=True)
    if args.continue_from is not None:
        model_params['load_pretrained_weights'] = False
        logger.info("[Resume] Skipping pretrained weights loading (will load from checkpoint)")

    model = load_model(model_lib, model_class_name, model_params, device)

    logger.info(f"Selected model: {model_lib}.{model_class_name}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model's parameters: {total_params / 1_000_000:.2f} M")

    if args.save_code:
        _copy_model_source(model_lib, logger)

    # QAT mode (cycle 17): training_mode='qat' wraps model via PT2E, drops the
    # MetricGAN discriminator (CPU batch_pesq bottleneck), and adds self-KD.
    training_mode = str(getattr(args, "training_mode", "gan")).lower()
    teacher = None
    kd_loss = None
    if training_mode == "qat":
        logger.info("[Training mode] QAT (PT2E + KD; discriminator dropped)")
        model, teacher, kd_loss, _ = _build_qat_components(args, model, device, logger)
        discriminator = None
    else:
        logger.info("[Training mode] GAN (PESQ-GAN MetricGANDiscriminator)")
        discriminator = MetricGANDiscriminator().to(device)

    optim_class = _get_optim_class(args.optim)
    optim = _build_optimizer(args, model, optim_class, logger)
    if discriminator is not None:
        optim_disc = optim_class(
            discriminator.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=args.betas,
        )
    else:
        optim_disc = None

    logger.info(
        f"[Scheduler] Cosine Annealing (warmup={args.warmup_epochs} epochs, eta_min={args.eta_min})"
    )
    scheduler = _build_cosine_warmup_scheduler(optim, args)
    scheduler_disc = _build_cosine_warmup_scheduler(optim_disc, args) if optim_disc is not None else None

    dset = args.dset.get("dataset", "TAPS")
    logger.info(f"Dataset selected: {dset}")
    if dset == "TAPS":
        _dataset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset")
    elif dset == "Vibravox":
        speech_subset = args.dset.get("speech_subset", None)
        if speech_subset:
            logger.info(f"[Vibravox] loading multi-config subset: {speech_subset}")
            _dataset = load_dataset("yskim3271/vibravox_16k", name=speech_subset)
        else:
            # Legacy default config (pre A-4 publish)
            _dataset = load_dataset("yskim3271/vibravox_16k")

    trainset = _dataset['train']
    validset = _dataset['dev']
    testset = concatenate_datasets([_dataset['test']] * args.dset.test_augment_numb)

    bcs_only = "acs" not in args.model.input_type
    segment, stride, shift = _resolve_segment_and_stride(args, logger)
    dataloader = _build_dataloaders(
        args, trainset, validset, testset, segment, stride, shift, bcs_only,
    )

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
        device=device,
        teacher=teacher,
        kd_loss=kd_loss,
    )
    solver.train()
    sys.exit(0)

_DSET_PATH_KEYS = {
    "noise_dir", "noise_train", "noise_valid", "noise_test",
    "rir_dir",   "rir_train",   "rir_valid",   "rir_test",
}


def _main(args):
    global __file__

    logger = setup_logger("main")

    # Convert filesystem path entries to absolute paths. Other string entries
    # (e.g., mix_strategy, noise_hf_dataset, speech_subset) are left untouched
    # so vibravox_native dispatch and HF dataset ids stay valid.
    for key, value in args.dset.items():
        if isinstance(value, str) and key in _DSET_PATH_KEYS:
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