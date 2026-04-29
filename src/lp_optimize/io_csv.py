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
    "read_result_csv",
    "write_result_csv",
]


def _is_number(token: str) -> bool:
    try:
        float(token)
    except (TypeError, ValueError):
        return False
    return True


def _iter_data_rows(path: str | Path) -> Iterable[list[str]]:
    """Yield non-empty CSV rows, automatically skipping a header line.

    A row is treated as a header if its second field is not parseable as a
    number.  This makes the reader tolerant of files that do or do not
    include a header.
    """
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
                    continue
            yield row


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


def write_result_csv(
    path: str | Path,
    elevation_m: np.ndarray,
    fs: float,
) -> None:
    """Write a processed elevation series to ``path``.

    The output file has a header ``t_s,elevation_mm`` with elevation
    converted from metres back to millimetres.
    """
    elevation_m = np.asarray(elevation_m, dtype=float)
    n = elevation_m.size
    t = np.arange(n, dtype=float) / float(fs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_s", "elevation_mm"])
        for ti, hi in zip(t, elevation_m * 1e3):
            writer.writerow([f"{ti:.6f}", f"{hi:.6f}"])


def read_result_csv(path: str | Path) -> tuple[np.ndarray, float]:
    """Read a ``Result_*.csv`` file written by :func:`write_result_csv`.

    Returns
    -------
    elevation_m, fs
        ``elevation_m`` is the elevation column converted from mm to m.
        ``fs`` is inferred from the median spacing of the time column
        (``1 / median(dt)``).
    """
    t_list: list[float] = []
    h_mm: list[float] = []
    for row in _iter_data_rows(path):
        if len(row) < 2:
            raise ValueError(
                f"{path}: expected at least 2 columns, got {len(row)}: {row!r}"
            )
        t_list.append(float(row[0]))
        h_mm.append(float(row[1]))
    if len(t_list) < 2:
        raise ValueError(f"{path}: need at least 2 rows to infer sample rate")
    t = np.asarray(t_list, dtype=float)
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError(f"{path}: non-positive time step {dt}")
    fs = 1.0 / dt
    return np.asarray(h_mm, dtype=float) * 1e-3, fs
