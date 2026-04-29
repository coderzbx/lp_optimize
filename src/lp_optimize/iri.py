"""International Roughness Index (IRI) via the Golden Car quarter-car model.

Parameters and procedure follow ASTM E1926 / Sayers 1995 ("On the
Calculation of International Roughness Index from Longitudinal Road
Profile").  The standard simulation speed is 80 km/h.

Golden Car parameters (per unit sprung mass):
    k1 = 653.0   (tyre stiffness)
    k2 = 63.3    (suspension stiffness)
    c  = 6.0     (suspension damping)
    mu = 0.15    (unsprung / sprung mass ratio)
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _signal

K1 = 653.0
K2 = 63.3
C = 6.0
MU = 0.15
SIM_SPEED_KMH = 80.0
SIM_SPEED_MS = SIM_SPEED_KMH * 1000.0 / 3600.0  # 22.2222 m/s


def _state_space() -> _signal.StateSpace:
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-K2, -C, K2, C],
        [0.0, 0.0, 0.0, 1.0],
        [K2 / MU, C / MU, -(K1 + K2) / MU, -C / MU],
    ])
    B = np.array([[0.0], [0.0], [0.0], [K1 / MU]])
    # output the slope difference (zs' - zu')
    Cm = np.array([[0.0, 1.0, 0.0, -1.0]])
    D = np.array([[0.0]])
    return _signal.StateSpace(A, B, Cm, D)


def compute_iri(
    s: np.ndarray,
    h: np.ndarray,
    speed_ms: float = SIM_SPEED_MS,
    settle_distance: float = 11.0,
) -> float:
    """Compute IRI (m/km) from an evenly-spaced road profile.

    Parameters
    ----------
    s : np.ndarray
        Distance axis in metres, evenly spaced.
    h : np.ndarray
        Road elevation in metres, same length as ``s``.
    speed_ms : float
        Simulation speed in m/s.  ``80 km/h`` is the standard.
    settle_distance : float
        Initial distance over which the integrator is allowed to settle
        (the standard recommends an 11 m lead-in).  This portion is
        excluded from the IRI average.

    Returns
    -------
    float
        IRI in m/km.
    """
    s = np.asarray(s, dtype=float)
    h = np.asarray(h, dtype=float)
    if s.size != h.size or s.size < 4:
        raise ValueError("s and h must have the same length (>= 4)")
    ds = float(np.mean(np.diff(s)))
    if ds <= 0:
        raise ValueError("s must be strictly increasing")

    # Time vector for the simulation
    t = s / speed_ms
    sys = _state_space()
    # Initial state: in steady state on the first elevation sample.
    y0 = h[0]
    x0 = np.array([y0, 0.0, y0, 0.0])
    _, slope_diff, _ = _signal.lsim(sys, U=h, T=t, X0=x0)
    slope_diff = np.asarray(slope_diff).ravel()

    # Drop the lead-in
    n_skip = int(np.ceil(settle_distance / ds))
    n_skip = min(n_skip, s.size - 2)
    iri_slope = float(np.mean(np.abs(slope_diff[n_skip:])))
    # |slope| is in (m/s) / (m/s) = dimensionless = m/m -> *1000 -> m/km
    return iri_slope * 1000.0
