"""Tests for the CSV IO helpers and the CLI subcommands."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from lp_optimize.cli import align_to_reference, main as cli_main
from lp_optimize.io_csv import (
    read_input_csv,
    read_result_csv,
    write_result_csv,
)


FS = 1000.0


def _write_csv(path: Path, rows: list[list[object]], header: list[str] | None) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if header is not None:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def _synth_idle_off(n: int, rng: np.random.Generator) -> np.ndarray:
    """Engine-off-like signal in mm."""
    t = np.arange(n) / FS
    elev_m = 1.500 + 0.0005 * np.sin(2 * np.pi * 0.005 * t) + rng.normal(0, 5e-5, n)
    return elev_m * 1e3  # mm


def _synth_idle_on(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / FS
    z_body = 0.0008 * np.sin(2 * np.pi * 2.0 * t) + 0.0002 * np.sin(
        2 * np.pi * 12.0 * t + 0.7
    )
    a_z = (
        -0.0008 * (2 * np.pi * 2.0) ** 2 * np.sin(2 * np.pi * 2.0 * t)
        - 0.0002 * (2 * np.pi * 12.0) ** 2 * np.sin(2 * np.pi * 12.0 * t + 0.7)
    )
    a_z += rng.normal(0.0, 0.02, n)
    laser_m = 1.5 - z_body + rng.normal(0.0, 5e-5, n)
    return laser_m * 1e3, a_z


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def test_read_input_csv_with_header(tmp_path: Path) -> None:
    p = tmp_path / "C001.csv"
    _write_csv(
        p,
        rows=[[0, 1500.5, 0.01], [1, 1500.4, -0.02], [2, 1500.6, 0.0]],
        header=["idx", "elev_mm", "a_z"],
    )
    elev_m, _ = read_input_csv(p, need_accel=False)
    assert elev_m.shape == (3,)
    np.testing.assert_allclose(elev_m, [1.5005, 1.5004, 1.5006])


def test_read_input_csv_without_header_with_accel(tmp_path: Path) -> None:
    p = tmp_path / "C040.csv"
    _write_csv(
        p,
        rows=[[0, 1500.0, 0.1], [1, 1501.0, -0.2]],
        header=None,
    )
    elev_m, a = read_input_csv(p, need_accel=True)
    assert a is not None
    np.testing.assert_allclose(elev_m, [1.5, 1.501])
    np.testing.assert_allclose(a, [0.1, -0.2])


def test_read_input_csv_missing_accel_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    _write_csv(p, rows=[[0, 1500.0]], header=None)
    with pytest.raises(ValueError):
        read_input_csv(p, need_accel=True)


def test_write_then_read_result_csv_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "Result.csv"
    elev_m = np.linspace(0.0, 0.001, 20)
    write_result_csv(p, elev_m, fs=100.0)
    elev_back, fs_back = read_result_csv(p)
    np.testing.assert_allclose(elev_back, elev_m, atol=1e-9)
    assert abs(fs_back - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# align_to_reference
# ---------------------------------------------------------------------------


def test_align_to_reference_matches_std() -> None:
    rng = np.random.default_rng(0)
    fs = 100.0
    n = 4096
    ref = rng.normal(0.0, 1e-4, n)
    tgt = rng.normal(0.0, 5e-4, n)  # 5x noisier
    aligned = align_to_reference(tgt, ref, fs=fs)
    assert abs(np.std(aligned) - np.std(ref)) / np.std(ref) < 0.05


# ---------------------------------------------------------------------------
# End-to-end CLI invocation
# ---------------------------------------------------------------------------


def test_cli_idle_off_then_idle_on_then_align(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = int(FS * 5)  # 5 s of data is plenty for a smoke test

    c001 = tmp_path / "C001.csv"
    c040 = tmp_path / "C040.csv"
    elev_off_mm = _synth_idle_off(n, rng)
    elev_on_mm, a_z = _synth_idle_on(n, rng)
    _write_csv(
        c001,
        rows=[[i, elev_off_mm[i]] for i in range(n)],
        header=["idx", "elev_mm"],
    )
    _write_csv(
        c040,
        rows=[[i, elev_on_mm[i], a_z[i]] for i in range(n)],
        header=["idx", "elev_mm", "a_z"],
    )

    r1 = tmp_path / "Result_C001.csv"
    r2 = tmp_path / "Result_C040.csv"
    r3 = tmp_path / "Result_C040_aligned.csv"

    rc = cli_main(["idle-off", str(c001), str(r1), "--fs", "1000"])
    assert rc == 0 and r1.exists()

    rc = cli_main(["idle-on", str(c040), str(r2), "--fs", "1000"])
    assert rc == 0 and r2.exists()

    rc = cli_main(["align", str(r1), str(r2), str(r3)])
    assert rc == 0 and r3.exists()

    ref_m, fs_ref = read_result_csv(r1)
    aligned_m, fs_aligned = read_result_csv(r3)
    assert abs(fs_aligned - fs_ref) < 1e-6
    # After alignment the std should match the reference within ~5 %.
    assert abs(np.std(aligned_m) - np.std(ref_m)) / np.std(ref_m) < 0.1
