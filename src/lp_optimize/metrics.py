"""Quality metrics: standard deviation / RMS / Welch PSD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class StdRms:
    std: float
    rms: float
    peak2peak: float


def std_rms(x: np.ndarray) -> StdRms:
    """Return std, RMS and peak-to-peak of ``x`` (NaNs are ignored)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return StdRms(float("nan"), float("nan"), float("nan"))
    return StdRms(
        std=float(np.std(x, ddof=1) if x.size > 1 else 0.0),
        rms=float(np.sqrt(np.mean(x ** 2))),
        peak2peak=float(np.ptp(x)),
    )


def welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD estimate.  Returns ``(f, Pxx)`` in Hz and units²/Hz."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if nperseg is None:
        nperseg = min(len(x), 1 << 14)
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    return f, Pxx
