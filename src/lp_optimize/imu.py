"""Accelerometer / IMU based body-motion compensation.

Two complementary tools are provided:

* :func:`accel_to_displacement` -- numerically robust double integration of
  a vertical acceleration trace using FFT division by ``-(2 pi f)**2`` with
  a band-pass mask.  This avoids the runaway drift of time-domain double
  integration.
* :func:`lms_anc` -- a normalised LMS adaptive noise canceller.  Given a
  primary signal (laser elevation contaminated by body vibration) and a
  reference signal (e.g. the integrated body displacement, or the raw
  acceleration), it learns the unknown coupling FIR filter on-line and
  returns the cleaned signal.
"""

from __future__ import annotations

import numpy as np


def accel_to_displacement(
    accel: np.ndarray,
    fs: float,
    band: tuple[float, float] = (0.5, 80.0),
) -> np.ndarray:
    """Convert a vertical acceleration trace (m/s²) to displacement (m).

    Implementation: subtract the mean (remove gravity / static bias), then
    perform the double integration in the frequency domain as
    ``X(f) = A(f) / (-(2 pi f)**2)``, with the spectrum forced to zero
    outside ``band`` (and at DC) to suppress the unbounded ``1/f**2`` gain
    near 0 Hz.

    Parameters
    ----------
    accel : np.ndarray
        Vertical acceleration in m/s² (positive up).
    fs : float
        Sample rate in Hz.
    band : (f_lo, f_hi)
        Pass-band in Hz.  ``f_lo`` should be > 0; values around 0.3–1 Hz
        work well for vehicle body motion.  ``f_hi`` should be safely below
        the Nyquist frequency.
    """
    a = np.asarray(accel, dtype=float)
    a = a - np.nanmean(a)
    n = a.size
    # rfft for real signals
    A = np.fft.rfft(a)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    # band-pass mask; explicitly kill DC
    f_lo, f_hi = band
    mask = (f >= f_lo) & (f <= min(f_hi, 0.5 * fs * 0.999))
    omega2 = (2.0 * np.pi * f) ** 2
    # avoid 0/0 at f=0
    omega2[0] = 1.0
    X = np.where(mask, -A / omega2, 0.0)
    x = np.fft.irfft(X, n=n)
    return x


def lms_anc(
    primary: np.ndarray,
    reference: np.ndarray,
    n_taps: int = 128,
    mu: float = 0.05,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised LMS adaptive noise canceller.

    Learns an FIR mapping ``w`` from ``reference`` to the noise component of
    ``primary`` and subtracts it.  Returns ``(cleaned, weights)``.

    Parameters
    ----------
    primary : np.ndarray
        Signal to clean (laser elevation here).
    reference : np.ndarray
        Reference correlated with the noise but not with the desired signal
        (e.g. body vertical displacement integrated from IMU).
    n_taps : int
        FIR length.  128 taps at 2 kHz cover ~64 ms of memory, enough for
        the dominant 1–15 Hz body modes.
    mu : float
        Step size (0–1 for NLMS).  Smaller = more stable, slower.
    eps : float
        Regularisation for the input-power normalisation.
    """
    d = np.asarray(primary, dtype=float).copy()
    u = np.asarray(reference, dtype=float).copy()
    if d.shape != u.shape:
        raise ValueError("primary and reference must have the same shape")
    n = d.size
    w = np.zeros(n_taps)
    y = np.zeros(n)
    e = np.zeros(n)
    # Pre-allocated tap buffer
    x_buf = np.zeros(n_taps)
    for k in range(n):
        # shift in newest reference sample
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = u[k]
        y_k = float(np.dot(w, x_buf))
        e_k = d[k] - y_k
        norm = float(np.dot(x_buf, x_buf)) + eps
        w += (mu / norm) * e_k * x_buf
        y[k] = y_k
        e[k] = e_k
    return e, w
