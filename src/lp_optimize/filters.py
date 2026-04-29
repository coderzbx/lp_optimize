"""Zero-phase low/high/band-pass filters and Savitzky-Golay smoother."""

from __future__ import annotations

import numpy as np
from scipy import signal


def butter_filter(
    x: np.ndarray,
    fs: float,
    cutoff: float | tuple[float, float],
    btype: str = "lowpass",
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth filter (``filtfilt``).

    ``cutoff`` is a scalar for low/high-pass and a 2-tuple for band-pass /
    band-stop.  Frequencies are in Hz.
    """
    nyq = 0.5 * fs
    if isinstance(cutoff, (tuple, list, np.ndarray)):
        wn = (np.asarray(cutoff, dtype=float) / nyq).clip(1e-6, 0.999999)
    else:
        wn = float(cutoff) / nyq
        wn = min(max(wn, 1e-6), 0.999999)
    b, a = signal.butter(order, wn, btype=btype)
    return signal.filtfilt(b, a, np.asarray(x, dtype=float))


def savgol(x: np.ndarray, window: int = 51, order: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing; preserves waveform shape better than a low-pass."""
    if window % 2 == 0:
        window += 1
    window = max(window, order + 2 + (order % 2 == 0))
    return signal.savgol_filter(np.asarray(x, dtype=float), window, order)


def antialias_decimate(x: np.ndarray, fs: float, fs_out: float) -> tuple[np.ndarray, float]:
    """Anti-alias low-pass + decimate to ``fs_out``.

    Returns ``(y, fs_out_actual)``.  Uses :func:`scipy.signal.resample_poly`
    so the new sample rate is ``fs * up / down`` with small integer ratios.
    """
    if fs_out >= fs:
        return np.asarray(x, dtype=float), fs
    from math import gcd

    # rational approximation
    up, down = 1, int(round(fs / fs_out))
    g = gcd(up, down)
    up //= g
    down //= g
    y = signal.resample_poly(np.asarray(x, dtype=float), up, down)
    return y, fs * up / down
