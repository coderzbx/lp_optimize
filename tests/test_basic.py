import numpy as np

from lp_optimize.preprocessing import hampel_filter, interpolate_gaps
from lp_optimize.imu import accel_to_displacement, lms_anc
from lp_optimize.attitude import complementary_attitude, project_to_vertical
from lp_optimize.iri import compute_iri
from lp_optimize.profile import time_to_space


def test_hampel_removes_spike():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1e-3, 1000)
    x[500] = 1.0  # huge spike
    y = hampel_filter(x, window=11, n_sigma=3.0)
    assert abs(y[500]) < 1e-2


def test_interpolate_gaps_linear():
    x = np.array([0.0, np.nan, np.nan, 3.0])
    y = interpolate_gaps(x)
    assert np.allclose(y, [0.0, 1.0, 2.0, 3.0])


def test_accel_to_displacement_recovers_sine():
    fs = 1000.0
    t = np.arange(0, 10.0, 1.0 / fs)
    f0 = 5.0
    z = 0.01 * np.sin(2 * np.pi * f0 * t)
    a = -0.01 * (2 * np.pi * f0) ** 2 * np.sin(2 * np.pi * f0 * t)
    z_hat = accel_to_displacement(a, fs=fs, band=(0.5, 50.0))
    # Skip edges (FFT band-pass has transient)
    err = np.std(z_hat[1000:-1000] - z[1000:-1000])
    assert err < 5e-4, err


def test_lms_anc_cancels_correlated_noise():
    rng = np.random.default_rng(1)
    n = 5000
    ref = rng.normal(0.0, 1.0, n)
    # noise = ref delayed by 2 samples and scaled
    noise = np.concatenate(([0, 0], 0.7 * ref[:-2]))
    desired = rng.normal(0.0, 0.01, n)
    primary = desired + noise
    cleaned, _ = lms_anc(primary, ref, n_taps=16, mu=0.1)
    # after convergence the residual should be much smaller than primary
    assert np.std(cleaned[2000:]) < np.std(primary[2000:]) * 0.3


def test_attitude_zero_motion():
    fs = 100.0
    n = 200
    a = np.tile(np.array([0.0, 0.0, 9.81]), (n, 1))
    g = np.zeros((n, 3))
    pitch, roll = complementary_attitude(a, g, fs=fs)
    assert np.max(np.abs(pitch)) < 1e-3
    assert np.max(np.abs(roll)) < 1e-3
    h = project_to_vertical(np.full(n, 1.5), pitch, roll)
    assert np.allclose(h, 1.5, atol=1e-3)


def test_time_to_space_constant_speed():
    fs = 100.0
    n = 1000
    v = np.full(n, 10.0)  # 10 m/s
    h = np.linspace(0.0, 1.0, n)
    s, hs = time_to_space(h, v, fs=fs, ds=0.1, min_speed=1.0)
    # total distance ~ 10 m/s * 9.99 s = 99.9 m
    assert s[-1] > 95.0 and s[-1] < 100.5
    assert np.all(np.diff(s) > 0)


def test_iri_flat_road_is_zero():
    s = np.arange(0.0, 200.0, 0.25)
    h = np.zeros_like(s)
    iri = compute_iri(s, h)
    assert iri < 1e-6


def test_iri_sinusoidal_road_is_positive():
    s = np.arange(0.0, 200.0, 0.1)
    h = 0.005 * np.sin(2 * np.pi * s / 5.0)  # 5 m wavelength, 5 mm amp
    iri = compute_iri(s, h)
    assert iri > 0.5  # should be a clearly non-trivial value
