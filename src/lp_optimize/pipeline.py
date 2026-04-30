"""High-level processing pipelines for the three operating conditions
described in `需求文档`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attitude import complementary_attitude, project_to_vertical
from .filters import antialias_decimate, butter_filter, median_filter, savgol
from .imu import accel_to_displacement, lms_anc
from .iri import compute_iri
from .metrics import std_rms, welch_psd, StdRms
from .preprocessing import hampel_filter, highpass_detrend, interpolate_gaps
from .profile import time_to_space


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class IdleResult:
    """Result of an "idle" (stationary vehicle) processing pipeline."""

    elevation: np.ndarray
    fluctuation: np.ndarray
    fs: float
    metrics: StdRms
    psd_freq: np.ndarray = field(repr=False)
    psd: np.ndarray = field(repr=False)


@dataclass
class DrivingResult:
    """Result of the full driving pipeline."""

    profile_s: np.ndarray  # distance axis (m)
    profile_h: np.ndarray  # elevation profile (m), spatial domain
    elevation_t: np.ndarray  # time-domain elevation after compensation (m)
    fs: float  # time-domain sample rate
    iri_m_per_km: float
    metrics: StdRms


# ---------------------------------------------------------------------------
# Problem 1 -- engine off, vehicle still
# ---------------------------------------------------------------------------


def _idle_residual_cleanup(
    x: np.ndarray,
    fs: float,
    *,
    residual_hampel_window: int,
    residual_hampel_sigma: float,
    median_window: int,
    smooth_cutoff: float,
    savgol_window: int,
    savgol_order: int,
) -> np.ndarray:
    """Extra robust smoothing for stationary idle traces."""
    x = hampel_filter(x, window=residual_hampel_window, n_sigma=residual_hampel_sigma)
    x = median_filter(x, window=median_window)
    x = butter_filter(x, fs=fs, cutoff=smooth_cutoff, btype="lowpass", order=4)
    x = savgol(x, window=savgol_window, order=savgol_order)
    return x


def pipeline_idle_off(
    elevation: np.ndarray,
    fs: float = 2000.0,
    *,
    hampel_window: int = 21,
    hampel_sigma: float = 3.0,
    residual_hampel_window: int = 61,
    residual_hampel_sigma: float = 2.5,
    median_window: int = 9,
    drift_cutoff: float = 0.05,
    smooth_cutoff: float = 40.0,
    savgol_window: int = 41,
    savgol_order: int = 3,
    decimate_to: float | None = 100.0,
) -> IdleResult:
    """Process problem-1 data (engine off, vehicle still).

    Pipeline: outlier removal -> gap interpolation -> low-frequency
    de-trend -> low-pass smoothing -> optional decimation.  The returned
    ``elevation`` keeps a robust absolute baseline; the zero-mean dynamic
    component is exposed separately as ``fluctuation``.
    """
    x = np.asarray(elevation, dtype=float)
    x = interpolate_gaps(x)
    x = hampel_filter(x, window=hampel_window, n_sigma=hampel_sigma)
    baseline = float(np.median(x))
    x = highpass_detrend(x, fs=fs, fc=drift_cutoff)
    x = _idle_residual_cleanup(
        x,
        fs,
        residual_hampel_window=residual_hampel_window,
        residual_hampel_sigma=residual_hampel_sigma,
        median_window=median_window,
        smooth_cutoff=smooth_cutoff,
        savgol_window=savgol_window,
        savgol_order=savgol_order,
    )
    fs_out = fs
    if decimate_to is not None and decimate_to < fs:
        x, fs_out = antialias_decimate(x, fs=fs, fs_out=decimate_to)
    metrics = std_rms(x)
    f, p = welch_psd(x, fs=fs_out)
    return IdleResult(
        elevation=baseline + x,
        fluctuation=x,
        fs=fs_out,
        metrics=metrics,
        psd_freq=f,
        psd=p,
    )


# ---------------------------------------------------------------------------
# Problem 2 -- engine on, vehicle still, with body accelerometer
# ---------------------------------------------------------------------------


def pipeline_idle_on(
    elevation: np.ndarray,
    accel_z: np.ndarray,
    fs: float = 2000.0,
    *,
    hampel_window: int = 21,
    hampel_sigma: float = 3.0,
    residual_hampel_window: int = 61,
    residual_hampel_sigma: float = 2.5,
    median_window: int = 9,
    drift_cutoff: float = 0.05,
    accel_band: tuple[float, float] = (0.5, 80.0),
    use_anc: bool = True,
    anc_taps: int = 128,
    anc_mu: float = 0.05,
    smooth_cutoff: float = 40.0,
    savgol_window: int = 41,
    savgol_order: int = 3,
    decimate_to: float | None = 100.0,
) -> IdleResult:
    """Process problem-2 data (idle, stationary, with vertical accelerometer).

    Steps:

    1. Hampel + gap interpolation on the laser elevation (problem-1 cleanup).
    2. Integrate ``accel_z`` to body vertical displacement ``z_body`` via
       band-limited frequency-domain double integration.
    3. Subtract ``z_body`` from the laser reading; this is the deterministic
       compensation step.
    4. Optionally run an LMS adaptive noise canceller using ``z_body`` as
       the reference, to mop up residual coupling that the deterministic
       step missed (gain / phase mismatch, mounting flexure, ...).
    5. Light low-pass + de-trend + optional decimation, identical to
       problem 1, so the two outputs are directly comparable.
    """
    x = np.asarray(elevation, dtype=float)
    a = np.asarray(accel_z, dtype=float)
    if x.shape != a.shape:
        raise ValueError("elevation and accel_z must have the same shape")
    x = interpolate_gaps(x)
    x = hampel_filter(x, window=hampel_window, n_sigma=hampel_sigma)
    z_body = accel_to_displacement(a, fs=fs, band=accel_band)
    baseline = float(np.median(x + z_body))
    # Remove DC / drift first so the (zero-mean) reference and the
    # primary live on comparable scales -- otherwise LMS becomes unstable.
    x = highpass_detrend(x, fs=fs, fc=drift_cutoff)
    # Laser reads (true ground - body z); compensate: add body displacement back.
    x_comp = x + z_body

    if use_anc:
        # The remaining noise should still correlate with body motion;
        # let the LMS filter learn the leftover transfer function.
        x_comp, _ = lms_anc(x_comp, z_body, n_taps=anc_taps, mu=anc_mu)

    x_comp = _idle_residual_cleanup(
        x_comp,
        fs,
        residual_hampel_window=residual_hampel_window,
        residual_hampel_sigma=residual_hampel_sigma,
        median_window=median_window,
        smooth_cutoff=smooth_cutoff,
        savgol_window=savgol_window,
        savgol_order=savgol_order,
    )

    fs_out = fs
    if decimate_to is not None and decimate_to < fs:
        x_comp, fs_out = antialias_decimate(x_comp, fs=fs, fs_out=decimate_to)

    metrics = std_rms(x_comp)
    f, p = welch_psd(x_comp, fs=fs_out)
    return IdleResult(
        elevation=baseline + x_comp,
        fluctuation=x_comp,
        fs=fs_out,
        metrics=metrics,
        psd_freq=f,
        psd=p,
    )


# ---------------------------------------------------------------------------
# Problem 3 -- vehicle driving, full IMU
# ---------------------------------------------------------------------------


def pipeline_driving(
    elevation: np.ndarray,
    accel_xyz: np.ndarray,
    gyro_xyz: np.ndarray,
    speed: np.ndarray,
    fs: float = 2000.0,
    *,
    hampel_window: int = 21,
    hampel_sigma: float = 3.0,
    accel_band: tuple[float, float] = (0.3, 40.0),
    use_anc: bool = True,
    anc_taps: int = 128,
    anc_mu: float = 0.02,
    profile_band: tuple[float, float] = (1.0 / 50.0, 1.0 / 0.5),
    ds: float = 0.1,
    min_speed: float = 5.0,
    compute_iri_index: bool = True,
) -> DrivingResult:
    """Process problem-3 data (driving) with full IMU.

    Pipeline overview
    -----------------
    1. **Cleanup** the laser series (Hampel + gap interpolation).
    2. **Attitude estimation** with a complementary filter on the 3-axis
       accel and gyro -> pitch / roll.  Project the slant range onto the
       world vertical axis via ``h = d * cos(pitch) * cos(roll)``.
    3. **Body vertical displacement** is integrated from ``a_z`` with the
       same band-limited FFT method as problem 2; gravity is removed by
       the de-mean step inside :func:`accel_to_displacement`.  Adding it
       back to the projected laser reading recovers the true ground
       elevation under the wheel.
    4. **Adaptive residual cleanup** with NLMS, identical to problem 2.
    5. **Time -> space resampling** using ``speed`` (samples below
       ``min_speed`` are dropped).
    6. **Spatial band-pass** keeps wavelengths between 0.5 m and 50 m
       (the IRI-relevant range).
    7. **IRI** computed with the Golden Car quarter-car model at
       80 km/h.
    """
    x = np.asarray(elevation, dtype=float)
    a = np.asarray(accel_xyz, dtype=float)
    g = np.asarray(gyro_xyz, dtype=float)
    v = np.asarray(speed, dtype=float)
    n = x.size
    if a.shape != (n, 3) or g.shape != (n, 3):
        raise ValueError("accel_xyz and gyro_xyz must have shape (N, 3)")
    if v.shape != (n,):
        raise ValueError("speed must have shape (N,)")

    # 1. cleanup laser
    x = interpolate_gaps(x)
    x = hampel_filter(x, window=hampel_window, n_sigma=hampel_sigma)

    # 2. attitude -> vertical projection
    pitch, roll = complementary_attitude(a, g, fs=fs)
    h_proj = project_to_vertical(x, pitch, roll)
    # Remove DC / drift before adaptive cleanup (see pipeline_idle_on).
    h_proj = highpass_detrend(h_proj, fs=fs, fc=0.05)

    # 3. body vertical displacement compensation
    z_body = accel_to_displacement(a[:, 2], fs=fs, band=accel_band)
    h_comp = h_proj + z_body

    # 4. residual ANC
    if use_anc:
        h_comp, _ = lms_anc(h_comp, z_body, n_taps=anc_taps, mu=anc_mu)

    # 5. time -> space
    s, h_s = time_to_space(h_comp, v, fs=fs, ds=ds, min_speed=min_speed)

    # 6. spatial band-pass: wavelengths 0.5--50 m -> spatial freq 0.02--2 cyc/m.
    fs_space = 1.0 / ds  # samples per metre
    lam_long, lam_short = 1.0 / profile_band[0], 1.0 / profile_band[1]
    # band pass in cycles/metre
    f_lo, f_hi = profile_band  # already in cyc/m
    # ensure inside Nyquist
    f_hi = min(f_hi, 0.49 * fs_space)
    h_bp = butter_filter(h_s, fs=fs_space, cutoff=(f_lo, f_hi), btype="bandpass", order=2)

    # 7. IRI
    iri = compute_iri(s, h_bp) if compute_iri_index else float("nan")

    return DrivingResult(
        profile_s=s,
        profile_h=h_bp,
        elevation_t=h_comp,
        fs=fs,
        iri_m_per_km=iri,
        metrics=std_rms(h_bp),
    )
