import os
import time
import torch
import numpy as np
import logging
import re
import importlib
from typing import Dict, List, Optional, Union, Any
from joblib import Parallel, delayed
from contextlib import contextmanager
import atexit
from pesq import pesq

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss + gd_loss + iaf_loss

# Reusable joblib Parallel pool (loky backend)
_JOBLIB_PARALLEL = None
_JOBLIB_WORKERS = None

def _get_joblib_parallel(workers: int):
    """Return a reusable joblib Parallel instance; recreate if worker size changed."""
    global _JOBLIB_PARALLEL, _JOBLIB_WORKERS
    if _JOBLIB_PARALLEL is None or _JOBLIB_WORKERS != workers:
        # Terminate existing pool if present (best-effort; uses private API)
        if _JOBLIB_PARALLEL is not None:
            try:
                _JOBLIB_PARALLEL._terminate_pool()
            except Exception:
                pass
        _JOBLIB_PARALLEL = Parallel(n_jobs=workers, backend="loky", prefer="processes")
        _JOBLIB_WORKERS = workers
    return _JOBLIB_PARALLEL

def batch_pesq(clean, noisy, workers=8, normalize=True):
    # Reuse a single loky process pool to avoid frequent creation/cleanup cycles
    parallel = _get_joblib_parallel(workers)
    pesq_score = parallel(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    if normalize:
        pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)

def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def serialize_model(model):
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out

def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))

def basename(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    return filename, ext



class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    Args:
        - logger: logger obtained from `logging.getLogger`,
        - iterable: iterable object to wrap
        - updates (int): number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length.
        - total (int): length of the iterable, in case it does not support
            `len`.
        - name (str): prefix to use in the log.
        - level: logging level (like `logging.INFO`).
    """
    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos
    
    def append(self, **infos):
        self._infos.update(**infos)
        
    def _append(self, info):
        self._infos.update(info)

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            # logging is delayed by 1 it, in order to have the metrics from update
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1/self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")


def remove_ansi_codes(text: str) -> str:
    """
    Remove ANSI escape codes from text.

    Args:
        text: Text potentially containing ANSI escape codes

    Returns:
        Text with ANSI codes removed
    """
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\[\[?[0-9;]*m')
    return ansi_escape.sub('', text)


def parse_performance_metrics(line: str) -> Optional[Dict[str, Union[int, float, str]]]:
    """
    Parse performance metrics from a trainer log line.

    Supports two formats:
    1. "Performance on 0dB: PESQ=2.4788, STOI=0.8973, CSIG=1.8814, CBAK=2.9560, COVL=2.2210"
    2. "Performance on 0dB: CER=0.2174, WER=0.5025"

    Args:
        line: Log line containing performance metrics

    Returns:
        Dictionary with parsed metrics including 'epoch', 'snr', and metric values,
        or None if the line doesn't match the expected format

    Example:
        >>> line = "Epoch 100, Performance on 0dB: PESQ=3.2368, STOI=0.9464"
        >>> parse_performance_metrics(line)
        {'epoch': 100, 'snr': '0dB', 'PESQ': 3.2368, 'STOI': 0.9464}
    """
    # Remove ANSI codes
    clean_line = remove_ansi_codes(line)

    # Check if line contains performance metrics
    if "Performance on" not in clean_line:
        return None

    # Extract epoch number if present
    epoch_match = re.search(r'Epoch\s+(\d+)', clean_line)
    epoch = int(epoch_match.group(1)) if epoch_match else None

    # Extract SNR level (e.g., "0dB", "5dB")
    snr_match = re.search(r'Performance on\s+([-\d]+dB)', clean_line)
    snr = snr_match.group(1) if snr_match else None

    # Extract all metric key-value pairs (e.g., "PESQ=3.2368")
    metrics = {}
    metric_pattern = re.compile(r'([A-Z]+)=([\d.]+)')

    for match in metric_pattern.finditer(clean_line):
        metric_name = match.group(1)
        metric_value = float(match.group(2))
        metrics[metric_name] = metric_value

    # Build result dictionary
    result = {}
    if epoch is not None:
        result['epoch'] = epoch
    if snr is not None:
        result['snr'] = snr
    result.update(metrics)

    return result if metrics else None


def parse_log_metrics(log_path: str, snr_filter: Optional[str] = None) -> List[Dict[str, Union[int, float, str]]]:
    """
    Parse all performance metrics from a trainer log file.

    Args:
        log_path: Path to the trainer log file
        snr_filter: Optional SNR level to filter (e.g., "0dB"). If None, returns all.

    Returns:
        List of dictionaries containing parsed metrics

    Example:
        >>> metrics = parse_log_metrics("outputs/prk_1026_1/trainer.log", snr_filter="0dB")
        >>> print(metrics[0])
        {'epoch': 100, 'snr': '0dB', 'PESQ': 3.2368, 'STOI': 0.9464, ...}
    """
    metrics_list = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_performance_metrics(line)
            if parsed is not None:
                # Apply SNR filter if specified
                if snr_filter is None or parsed.get('snr') == snr_filter:
                    metrics_list.append(parsed)

    return metrics_list


# ============================================================================
# Model and Checkpoint Utilities
# ============================================================================

def load_model(model_lib: str, model_class_name: str, model_params: Dict[str, Any], device: str = 'cuda'):
    """
    Load model dynamically from models directory.

    Args:
        model_lib: Model library name (e.g., "primeknet", "primeknet_gru")
        model_class_name: Model class name (e.g., "PrimeKnet")
        model_params: Model parameters dictionary
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Model instance loaded on specified device

    Example:
        >>> model = load_model("primeknet", "PrimeKnet", params, "cuda")
    """
    module = importlib.import_module(f"src.models.{model_lib}")
    model_class = getattr(module, model_class_name)
    model = model_class(**model_params)
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, chkpt_dir: str, chkpt_file: str, device: str = 'cuda'):
    """
    Load model checkpoint.

    Args:
        model: Model instance to load checkpoint into
        chkpt_dir: Directory containing checkpoint file
        chkpt_file: Checkpoint filename (e.g., "best.th", "checkpoint.th")
        device: Device for loading checkpoint

    Returns:
        Model with loaded checkpoint

    Example:
        >>> model = load_checkpoint(model, "outputs/exp", "best.th", "cuda")
    """
    chkpt_path = os.path.join(chkpt_dir, chkpt_file)
    chkpt = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    return model


def parse_file_list(directory: str, list_file: str) -> List[str]:
    """
    Parse file list and return full paths.

    Args:
        directory: Base directory path
        list_file: Path to file containing list of filenames (one per line)

    Returns:
        List of full file paths

    Example:
        >>> noise_files = parse_file_list("/data/noise", "dataset/noise_train.txt")
    """
    with open(list_file, 'r') as f:
        return [os.path.join(directory, line.strip()) for line in f]


def get_stft_args_from_config(model_args) -> Dict[str, Any]:
    """
    Extract STFT arguments from model configuration.

    Args:
        model_args: Model configuration object with param attributes

    Returns:
        Dictionary of STFT arguments

    Example:
        >>> stft_args = get_stft_args_from_config(config.model)
        >>> # Returns: {"n_fft": 400, "hop_size": 100, "win_size": 400, "compress_factor": 0.3}
    """
    fft_len = model_args.param.fft_len
    return {
        "n_fft": fft_len,
        "hop_size": model_args.param.get("hop_len", fft_len // 4),
        "win_size": model_args.param.get("win_len", fft_len),
        "compress_factor": model_args.param.get("compress_factor", 1.0)
    }
