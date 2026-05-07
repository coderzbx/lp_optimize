"""CSV input / output helpers for the command-line interface.

The on-vehicle laser logger writes CSV files where:

* column 1 is some kind of index / timestamp,
* column 2 is the laser elevation reading in **millimetres**,
* column 3 (when present) is the vertical accelerometer reading
  (``a_z``) at the same instant.

The processing pipelines in :mod:`lp_optimize.pipeline` work in **metres**,
so :func:`read_input_csv` converts ``mm -> m`` on read and
:func:`write_result_csv` converts back ``m -> mm`` on write.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


__all__ = [
    "read_input_csv",
    "read_input_csv_with_timestamps",
    "read_result_csv",
    "read_result_csv_full",
    "resample_series_axis",
    "write_result_csv",
]


def _is_number(token: str) -> bool:
    try:
        float(token)
    except (TypeError, ValueError):
        return False
    return True


def _read_csv_rows(path: str | Path) -> tuple[list[str] | None, list[list[str]]]:
    """Read a CSV file, returning ``(header, rows)``.

    The first non-empty row is treated as a header if its second field is not
    numeric. This keeps the readers tolerant of files with or without a
    header line.
    """
    header: list[str] | None = None
    rows: list[list[str]] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        first = True
        for row in reader:
            if not row:
                continue
            # strip whitespace
            row = [c.strip() for c in row]
            if not any(row):
                continue
            if first:
                first = False
                # header row: 2nd column not numeric
                if len(row) >= 2 and not _is_number(row[1]):
                    header = row
                    continue
            rows.append(row)
    return header, rows


def _iter_data_rows(path: str | Path) -> Iterable[list[str]]:
    """Yield non-empty CSV rows, automatically skipping a header line."""
    _, rows = _read_csv_rows(path)
    yield from rows


def resample_series_axis(values: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample a 1-D axis array to length ``n_out``."""
    x = np.asarray(values, dtype=float).ravel()
    if n_out < 0:
        raise ValueError("n_out must be >= 0")
    if x.size == n_out:
        return x.copy()
    if n_out == 0:
        return np.empty(0, dtype=float)
    if x.size == 0:
        return np.zeros(n_out, dtype=float)
    if x.size == 1:
        return np.full(n_out, x[0], dtype=float)
    src = np.linspace(0.0, 1.0, x.size)
    dst = np.linspace(0.0, 1.0, n_out)
    return np.interp(dst, src, x)


def read_input_csv(
    path: str | Path,
    *,
    need_accel: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read an input CSV (``C001.csv`` / ``C040.csv``).

    Parameters
    ----------
    path:
        Path to the CSV file.
    need_accel:
        When ``True``, column 3 is required and returned as the second
        element of the tuple.  When ``False`` (default), only the
        elevation column is parsed and ``None`` is returned for the
        accelerometer.

    Returns
    -------
    elevation_m, accel_z
        ``elevation_m`` is the column-2 reading converted from mm to m.
        ``accel_z`` is the column-3 reading (assumed to be in m/s^2),
        or ``None`` when ``need_accel=False``.
    """
    elev_mm: list[float] = []
    accel: list[float] = []
    for row in _iter_data_rows(path):
        if len(row) < 2:
            raise ValueError(
                f"{path}: expected at least 2 columns, got {len(row)}: {row!r}"
            )
        elev_mm.append(float(row[1]))
        if need_accel:
            if len(row) < 3:
                raise ValueError(
                    f"{path}: expected at least 3 columns (col3 = accel), "
                    f"got {len(row)}: {row!r}"
                )
            accel.append(float(row[2]))

    elev_m = np.asarray(elev_mm, dtype=float) * 1e-3
    if need_accel:
        return elev_m, np.asarray(accel, dtype=float)
    return elev_m, None


def read_input_csv_with_timestamps(
    path: str | Path,
    *,
    need_accel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read an input CSV and also return the first-column sample axis.

    The first column is passed through as ``timestamps`` when numeric.  When
    it is absent or non-numeric, a simple sample index is emitted instead so
    downstream code always receives an axis array of matching length.
    """
    timestamps: list[float] = []
    elev_mm: list[float] = []
    accel: list[float] = []

    for i, row in enumerate(_iter_data_rows(path)):
        if len(row) < 2:
            raise ValueError(
                f"{path}: expected at least 2 columns, got {len(row)}: {row!r}"
            )
        timestamps.append(float(row[0]) if _is_number(row[0]) else float(i))
        elev_mm.append(float(row[1]))
        if need_accel:
            if len(row) < 3:
                raise ValueError(
                    f"{path}: expected at least 3 columns (col3 = accel), "
                    f"got {len(row)}: {row!r}"
                )
            accel.append(float(row[2]))

    timestamp_arr = np.asarray(timestamps, dtype=float)
    elev_m = np.asarray(elev_mm, dtype=float) * 1e-3
    if need_accel:
        return timestamp_arr, elev_m, np.asarray(accel, dtype=float)
    return timestamp_arr, elev_m, None


def write_result_csv(
    path: str | Path,
    elevation_m: np.ndarray,
    fs: float,
    *,
    fluctuation_m: np.ndarray | None = None,
    timestamps_s: np.ndarray | None = None,
) -> None:
    """Write a processed elevation series to ``path``.

    The output file always contains the source timestamp axis
    ``timestamp_s`` first, followed by the absolute processed elevation
    ``elevation_mm``. Optional columns for the zero-mean fluctuation
    component and the processing time axis ``t_s`` are emitted when
    provided.
    """
    elevation_m = np.asarray(elevation_m, dtype=float)
    n = elevation_m.size
    t = np.arange(n, dtype=float) / float(fs)
    fluct = None if fluctuation_m is None else np.asarray(fluctuation_m, dtype=float)
    if fluct is not None and fluct.shape != elevation_m.shape:
        raise ValueError("fluctuation_m must have the same shape as elevation_m")
    ts = None if timestamps_s is None else np.asarray(timestamps_s, dtype=float)
    if ts is not None and ts.shape != elevation_m.shape:
        raise ValueError("timestamps_s must have the same shape as elevation_m")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["elevation_mm"]
        if ts is not None:
            header.insert(0, "timestamp_s")
        if fluct is not None:
            header.append("fluctuation_mm")
        header.append("t_s")
        writer.writerow(header)
        for i, (ti, hi) in enumerate(zip(t, elevation_m * 1e3)):
            row = [f"{hi:.6f}"]
            if ts is not None:
                row.insert(0, f"{ts[i]:.6f}")
            if fluct is not None:
                row.append(f"{fluct[i] * 1e3:.6f}")
            row.append(f"{ti:.6f}")
            writer.writerow(row)


def read_result_csv(path: str | Path) -> tuple[np.ndarray, float]:
    """Read a ``Result_*.csv`` file written by :func:`write_result_csv`.

    Returns
    -------
    elevation_m, fs
        ``elevation_m`` is the elevation column converted from mm to m.
        ``fs`` is inferred from the median spacing of the time column
        (``1 / median(dt)``).
    """
    t, elev_m, _, _ = read_result_csv_full(path)
    if len(t) < 2:
        raise ValueError(f"{path}: need at least 2 rows to infer sample rate")
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError(f"{path}: non-positive time step {dt}")
    fs = 1.0 / dt
    return elev_m, fs


def read_result_csv_full(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read a processed result CSV including optional extra columns.

    Returns
    -------
    t_s, elevation_m, fluctuation_m, timestamp_s
        ``t_s`` is the processing time axis, ``elevation_m`` is the absolute
        processed elevation in metres, ``fluctuation_m`` is the optional
        zero-mean component in metres, and ``timestamp_s`` is the optional
        source timestamp column.
    """
    header, rows = _read_csv_rows(path)
    if header is not None:
        header_map = {name.strip(): i for i, name in enumerate(header)}
        t_idx = header_map.get("t_s", 0)
        h_idx = header_map.get("elevation_mm", 1)
        fluct_idx = header_map.get("fluctuation_mm")
        timestamp_idx = header_map.get("timestamp_s")
    else:
        t_idx = 0
        h_idx = 1
        fluct_idx = 2 if rows and len(rows[0]) >= 3 else None
        timestamp_idx = 3 if rows and len(rows[0]) >= 4 else None

    t_list: list[float] = []
    h_mm: list[float] = []
    fluct_mm: list[float] = []
    ts_list: list[float] = []

    for row in rows:
        need_cols = max(
            t_idx,
            h_idx,
            -1 if fluct_idx is None else fluct_idx,
            -1 if timestamp_idx is None else timestamp_idx,
        ) + 1
        if len(row) < need_cols:
            raise ValueError(
                f"{path}: expected at least {need_cols} columns, got {len(row)}: {row!r}"
            )
        t_list.append(float(row[t_idx]))
        h_mm.append(float(row[h_idx]))
        if fluct_idx is not None:
            fluct_mm.append(float(row[fluct_idx]))
        if timestamp_idx is not None:
            ts_list.append(float(row[timestamp_idx]))

    fluct = None
    if fluct_idx is not None:
        fluct = np.asarray(fluct_mm, dtype=float) * 1e-3
    timestamps = None
    if timestamp_idx is not None:
        timestamps = np.asarray(ts_list, dtype=float)
    return (
        np.asarray(t_list, dtype=float),
        np.asarray(h_mm, dtype=float) * 1e-3,
        fluct,
        timestamps,
    )
