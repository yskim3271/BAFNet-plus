"""Miscellaneous utilities: progress logger, ANSI colors, config/file parsing.

Heavier concerns were extracted into dedicated modules:
    - ``src.losses``     : phase/PESQ losses and joblib pool
    - ``src.checkpoint`` : model loading, checkpoint I/O, ``ConfigDict``
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List


# ============================================================================
# Progress logging
# ============================================================================

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
    """Display text with some ANSI color in the terminal."""
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    """Display text in bold in the terminal."""
    return colorize(text, "1")


# ============================================================================
# History / Config / File list helpers
# ============================================================================

def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out


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
    import os

    with open(list_file, 'r') as f:
        return [os.path.join(directory, line.strip()) for line in f]


def get_stft_args_from_config(model_args) -> Dict[str, Any]:
    """
    Extract STFT arguments from model configuration.

    Supports both new (n_fft/hop_size/win_size) and legacy (fft_len/hop_len/win_len)
    parameter names for backward compatibility with old checkpoint configs.

    Args:
        model_args: Model configuration object with param attributes

    Returns:
        Dictionary of STFT arguments

    Example:
        >>> stft_args = get_stft_args_from_config(config.model)
        >>> # Returns: {"n_fft": 400, "hop_size": 100, "win_size": 400, "compress_factor": 0.3}
    """
    param = model_args.param
    n_fft = param.get("n_fft", None) or param.get("fft_len", 400)
    hop_size = param.get("hop_size", None) or param.get("hop_len", n_fft // 4)
    win_size = param.get("win_size", None) or param.get("win_len", n_fft)
    return {
        "n_fft": n_fft,
        "hop_size": hop_size,
        "win_size": win_size,
        "compress_factor": param.get("compress_factor", 1.0)
    }
