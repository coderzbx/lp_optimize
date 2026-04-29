"""Synthetic-data demo of the three pipelines.

Run with:  python -m examples.demo  (after `pip install -e .`).
"""

from __future__ import annotations

import numpy as np

from lp_optimize import pipeline_idle_off, pipeline_idle_on, pipeline_driving


FS = 1000.0
DURATION = 60.0  # 1 minute is enough for a demo


def _synth_idle_off(rng: np.random.Generator) -> np.ndarray:
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    h0 = 1.500  # nominal stand-off (m)
    drift = 0.0005 * np.sin(2 * np.pi * 0.005 * t)  # 5 mHz, ±0.5 mm
    noise = rng.normal(0.0, 5e-5, n)  # 50 µm white
    spikes = np.zeros(n)
    idx = rng.choice(n, size=20, replace=False)
    spikes[idx] = rng.normal(0.0, 5e-3, idx.size)
    return h0 + drift + noise + spikes


def _synth_idle_on(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    h0 = 1.500
    # body modes: 2 Hz (suspension) + 12 Hz (engine)
    z_body = (
        0.0008 * np.sin(2 * np.pi * 2.0 * t)
        + 0.0002 * np.sin(2 * np.pi * 12.0 * t + 0.7)
    )
    a_z = (
        -0.0008 * (2 * np.pi * 2.0) ** 2 * np.sin(2 * np.pi * 2.0 * t)
        - 0.0002 * (2 * np.pi * 12.0) ** 2 * np.sin(2 * np.pi * 12.0 * t + 0.7)
    )
    a_z += rng.normal(0.0, 0.02, n)
    laser = h0 - z_body + rng.normal(0.0, 5e-5, n)
    return laser, a_z


def _synth_driving(rng: np.random.Generator) -> dict:
    duration = 30.0
    n = int(FS * duration)
    t = np.arange(n) / FS
    v = 22.222  # 80 km/h
    speed = np.full(n, v)
    s = v * t

    # Synthetic road: smooth-ish profile (ISO C-class-like)
    rough = np.zeros_like(s)
    for k in range(20):
        wl = rng.uniform(0.5, 30.0)  # wavelength m
        amp = 0.002 * (wl / 5.0)
        rough += amp * np.sin(2 * np.pi * s / wl + rng.uniform(0, 2 * np.pi))

    pitch = 0.005 * np.sin(2 * np.pi * 0.7 * t)
    roll = 0.003 * np.sin(2 * np.pi * 0.4 * t + 1.0)
    z_body = 0.005 * np.sin(2 * np.pi * 1.5 * t) + 0.001 * np.sin(2 * np.pi * 10.0 * t)

    # Laser slant distance: nominal stand-off - true road - body z, then 1/cos
    h0 = 1.5
    laser = (h0 - rough - z_body) / (np.cos(pitch) * np.cos(roll))
    laser += rng.normal(0.0, 5e-5, n)

    # Accelerometer: gravity in body frame (rotated) + body z accel + noise
    g = 9.81
    a_x = -g * np.sin(pitch) + rng.normal(0.0, 0.02, n)
    a_y = g * np.cos(pitch) * np.sin(roll) + rng.normal(0.0, 0.02, n)
    a_z_dyn = (
        -0.005 * (2 * np.pi * 1.5) ** 2 * np.sin(2 * np.pi * 1.5 * t)
        - 0.001 * (2 * np.pi * 10.0) ** 2 * np.sin(2 * np.pi * 10.0 * t)
    )
    a_z = g * np.cos(pitch) * np.cos(roll) + a_z_dyn + rng.normal(0.0, 0.02, n)

    gyro_x = np.gradient(roll, 1.0 / FS) + rng.normal(0.0, 0.001, n)
    gyro_y = np.gradient(pitch, 1.0 / FS) + rng.normal(0.0, 0.001, n)
    gyro_z = rng.normal(0.0, 0.001, n)

    accel = np.column_stack([a_x, a_y, a_z])
    gyro = np.column_stack([gyro_x, gyro_y, gyro_z])
    return dict(elevation=laser, accel=accel, gyro=gyro, speed=speed, true_road=rough, s=s)


def main() -> None:
    rng = np.random.default_rng(0)

    # Problem 1
    laser_off = _synth_idle_off(rng)
    r1 = pipeline_idle_off(laser_off, fs=FS)
    print("[problem 1] engine off, vehicle still")
    print(f"  std={r1.metrics.std*1e3:.4f} mm  rms={r1.metrics.rms:.4f} m  pp={r1.metrics.peak2peak*1e3:.3f} mm")

    # Problem 2
    laser_on, a_z = _synth_idle_on(rng)
    r2 = pipeline_idle_on(laser_on, a_z, fs=FS)
    print("[problem 2] engine on, vehicle still + accelerometer")
    print(f"  std={r2.metrics.std*1e3:.4f} mm  rms={r2.metrics.rms:.4f} m  pp={r2.metrics.peak2peak*1e3:.3f} mm")

    # Problem 3
    drv = _synth_driving(rng)
    r3 = pipeline_driving(
        drv["elevation"],
        drv["accel"],
        drv["gyro"],
        drv["speed"],
        fs=FS,
    )
    print("[problem 3] driving + IMU")
    print(f"  profile length={r3.profile_s[-1]:.1f} m  IRI={r3.iri_m_per_km:.3f} m/km")
    print(f"  profile std={r3.metrics.std*1e3:.3f} mm")


if __name__ == "__main__":
    main()
