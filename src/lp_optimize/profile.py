"""Convert a time-domain elevation series into an evenly-spaced spatial
profile using vehicle speed (or GNSS-derived distance)."""

from __future__ import annotations

import numpy as np


def time_to_space(
    elevation: np.ndarray,
    speed: np.ndarray,
    fs: float,
    ds: float = 0.1,
    min_speed: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an elevation time-series into space domain.

    Parameters
    ----------
    elevation : np.ndarray
        Elevation values sampled at ``fs`` Hz.
    speed : np.ndarray
        Vehicle speed in m/s sampled at ``fs`` Hz, same length as
        ``elevation``.
    fs : float
        Sample rate in Hz.
    ds : float, default 0.1
        Output spatial sample spacing in metres (0.1 m = 10 cm is typical
        for IRI work).
    min_speed : float, default 5.0
        Samples acquired below this speed are dropped (their per-sample
        spacing becomes ill-defined and IMU compensation is unreliable).

    Returns
    -------
    s : np.ndarray
        Distance axis in metres, evenly spaced at ``ds``.
    h : np.ndarray
        Elevation profile resampled onto ``s``.
    """
    h = np.asarray(elevation, dtype=float)
    v = np.asarray(speed, dtype=float)
    if h.shape != v.shape:
        raise ValueError("elevation and speed must have the same shape")

    keep = v >= min_speed
    if keep.sum() < 2:
        raise ValueError("not enough samples above min_speed")
    h = h[keep]
    v = v[keep]
    dt = 1.0 / fs
    # cumulative distance, starting from 0
    s_t = np.concatenate(([0.0], np.cumsum(0.5 * (v[:-1] + v[1:]) * dt)))
    s_uniform = np.arange(0.0, s_t[-1], ds)
    h_uniform = np.interp(s_uniform, s_t, h)
    return s_uniform, h_uniform
