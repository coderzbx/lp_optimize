import os

import numpy as np

from lp_optimize.plotting import plot_elevation, plot_acceleration, plot_comparison


def test_plot_elevation_creates_file(tmp_path):
    fs = 100.0
    n = 500
    y = 0.001 * np.sin(2 * np.pi * 2.0 * np.arange(n) / fs)
    out = tmp_path / "elev.png"
    p = plot_elevation(y, fs=fs, path=str(out))
    assert p == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_acceleration_handles_1d_and_3axis(tmp_path):
    fs = 100.0
    n = 200
    a1 = np.random.default_rng(0).normal(0.0, 0.1, n)
    out1 = tmp_path / "a1.png"
    plot_acceleration(a1, fs=fs, path=str(out1))
    assert out1.exists() and out1.stat().st_size > 0

    a3 = np.random.default_rng(1).normal(0.0, 0.1, (n, 3))
    out3 = tmp_path / "a3.png"
    plot_acceleration(a3, fs=fs, path=str(out3))
    assert out3.exists() and out3.stat().st_size > 0


def test_plot_comparison_truncates_to_shorter(tmp_path):
    fs = 50.0
    a = np.linspace(0.0, 1.0, 300)
    b = np.linspace(0.0, 1.0, 250)  # different length on purpose
    out = tmp_path / "cmp.png"
    plot_comparison(a, b, fs=fs, path=str(out), label_a="A", label_b="B")
    assert out.exists() and out.stat().st_size > 0


def test_plot_creates_parent_dirs(tmp_path):
    fs = 100.0
    y = np.zeros(50)
    out = tmp_path / "nested" / "dir" / "elev.png"
    plot_elevation(y, fs=fs, path=str(out))
    assert out.exists()
