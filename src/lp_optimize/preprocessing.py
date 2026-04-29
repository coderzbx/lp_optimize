"""Pre-processing utilities: outlier removal, gap interpolation, detrend."""

from __future__ import annotations

import numpy as np
from scipy import signal


def hampel_filter(x: np.ndarray, window: int = 11, n_sigma: float = 3.0) -> np.ndarray:
    """Remove impulsive outliers (spikes) using a Hampel filter.

    For each sample the local median ``m`` and median absolute deviation
    ``MAD`` are computed inside a window of length ``window``.  Samples for
    which ``|x - m| > n_sigma * 1.4826 * MAD`` are replaced by ``m``.

    Parameters
    ----------
    x : np.ndarray
        1-D input signal.
    window : int, default 11
        Window length (must be odd).  ~10 ms at 1 kHz is a good default.
    n_sigma : float, default 3.0
        Threshold in robust-sigma units.

    Returns
    -------
    np.ndarray
        Cleaned copy of ``x``.
    """
    x = np.asarray(x, dtype=float)
    if window % 2 == 0:
        window += 1
    k = 1.4826  # MAD -> sigma for Gaussian data
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    out = x.copy()
    # vectorised rolling median / MAD
    shape = (x.size, window)
    strides = (padded.strides[0], padded.strides[0])
    win = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1)
    threshold = n_sigma * k * mad
    bad = np.abs(x - med) > threshold
    out[bad] = med[bad]
    return out


def interpolate_gaps(x: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Linearly interpolate NaN gaps; gaps longer than ``max_gap`` are kept as NaN."""
    x = np.asarray(x, dtype=float).copy()
    nan = np.isnan(x)
    if not nan.any():
        return x
    idx = np.arange(x.size)
    good = ~nan
    if good.sum() < 2:
        return x
    x_interp = x.copy()
    x_interp[nan] = np.interp(idx[nan], idx[good], x[good])
    if max_gap is not None and max_gap > 0:
        # restore NaN where the gap is too long
        edges = np.diff(np.concatenate(([0], nan.astype(np.int8), [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            if e - s > max_gap:
                x_interp[s:e] = np.nan
    return x_interp


def detrend_polynomial(x: np.ndarray, order: int = 1) -> np.ndarray:
    """Subtract a least-squares polynomial trend (default: linear)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    t = np.arange(n)
    coeffs = np.polyfit(t, x, order)
    trend = np.polyval(coeffs, t)
    return x - trend


def highpass_detrend(x: np.ndarray, fs: float, fc: float = 0.05) -> np.ndarray:
    """Remove very-low-frequency drift using a zero-phase Butterworth high-pass."""
    if fc <= 0:
        return np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    wn = min(fc / nyq, 0.99)
    b, a = signal.butter(2, wn, btype="highpass")
    return signal.filtfilt(b, a, x)
