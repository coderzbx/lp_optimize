"""Vehicle attitude estimation (pitch / roll) from IMU.

A simple complementary filter that fuses gyro integration (good
short-term) with accelerometer-derived tilt (good long-term).  This is
sufficient for pitch/roll compensation of a downward-looking laser
range finder; full quaternion / Madgwick / EKF solutions can be plugged
in later if heading is also required.
"""

from __future__ import annotations

import numpy as np


def complementary_attitude(
    accel: np.ndarray,
    gyro: np.ndarray,
    fs: float,
    alpha: float = 0.98,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate pitch and roll (radians) from 3-axis accel and gyro.

    Parameters
    ----------
    accel : np.ndarray, shape (N, 3)
        Body-frame accelerations (m/s²), columns = (ax, ay, az).
    gyro : np.ndarray, shape (N, 3)
        Body-frame angular rates (rad/s), columns = (wx, wy, wz).
    fs : float
        Sample rate in Hz.
    alpha : float, default 0.98
        Complementary blend.  Closer to 1 -> more gyro, less accel.

    Returns
    -------
    pitch, roll : np.ndarray, shape (N,)
        Pitch (rotation about y) and roll (rotation about x), radians.
    """
    a = np.asarray(accel, dtype=float)
    g = np.asarray(gyro, dtype=float)
    if a.shape != g.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("accel and gyro must have shape (N, 3)")
    n = a.shape[0]
    dt = 1.0 / fs

    # Per-sample tilt from accel (assuming low specific force apart from g)
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    pitch_acc = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))
    roll_acc = np.arctan2(ay, az)

    pitch = np.empty(n)
    roll = np.empty(n)
    pitch[0] = pitch_acc[0]
    roll[0] = roll_acc[0]
    for k in range(1, n):
        pitch[k] = alpha * (pitch[k - 1] + g[k, 1] * dt) + (1 - alpha) * pitch_acc[k]
        roll[k] = alpha * (roll[k - 1] + g[k, 0] * dt) + (1 - alpha) * roll_acc[k]
    return pitch, roll


def project_to_vertical(distance: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Project a slant range measured by a body-fixed downward laser onto
    the world vertical axis: ``h = d * cos(pitch) * cos(roll)``.
    """
    d = np.asarray(distance, dtype=float)
    return d * np.cos(pitch) * np.cos(roll)
