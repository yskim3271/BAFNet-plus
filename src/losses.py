"""Training loss and signal-quality scorer utilities.

Contains phase-based losses and a joblib-backed batch PESQ scorer used by the
MetricGAN loss path. The joblib process pool is wrapped so workers are cleaned
up on interpreter shutdown.
"""

from __future__ import annotations

import atexit
from typing import Optional

import numpy as np
import torch
from joblib import Parallel, delayed
from pesq import pesq


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss + gd_loss + iaf_loss


class PesqPool:
    """Lazily-created joblib Parallel pool reused across batch_pesq calls.

    The pool is recreated only when the requested worker count changes. An
    ``atexit`` hook attempts to terminate the pool cleanly on interpreter
    shutdown to avoid lingering loky subprocesses.
    """

    def __init__(self) -> None:
        self._parallel: Optional[Parallel] = None
        self._workers: Optional[int] = None
        atexit.register(self.close)

    def get(self, workers: int) -> Parallel:
        if self._parallel is None or self._workers != workers:
            self.close()
            self._parallel = Parallel(n_jobs=workers, backend="loky", prefer="processes")
            self._workers = workers
        return self._parallel

    def close(self) -> None:
        if self._parallel is None:
            return
        try:
            self._parallel._terminate_pool()
        except Exception:
            pass
        finally:
            self._parallel = None
            self._workers = None


_PESQ_POOL = PesqPool()


def batch_pesq(clean, noisy, workers: int = 8, normalize: bool = True):
    parallel = _PESQ_POOL.get(workers)
    pesq_score = parallel(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    if normalize:
        pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)


def pesq_loss(clean, noisy, sr: int = 16000):
    try:
        return pesq(sr, clean, noisy, "wb")
    except Exception:
        return -1
