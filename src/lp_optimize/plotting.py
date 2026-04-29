"""Line-plot helpers for elevation / acceleration data.

These utilities turn the (processed) 1-D arrays produced by the
:mod:`lp_optimize.pipeline` functions into line plots saved to disk.

Three helpers are exposed:

* :func:`plot_elevation` -- single elevation series vs. time.
* :func:`plot_acceleration` -- single acceleration series vs. time.
* :func:`plot_comparison` -- two series overlaid on the same axes,
  for the typical "engine-off baseline vs. engine-on compensated"
  comparison described in problem 2 of ``需求文档``.

The module uses Matplotlib's non-interactive ``Agg`` backend so it works
in headless environments (CI, servers, ...).
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib

# Force a non-interactive backend before pyplot is imported so that the
# helpers can be used in headless environments such as CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np  # noqa: E402


__all__ = [
    "plot_elevation",
    "plot_acceleration",
    "plot_comparison",
]


def _time_axis(n: int, fs: float) -> np.ndarray:
    """Return a time axis (seconds) of length ``n`` for sample rate ``fs``."""
    if fs <= 0:
        raise ValueError("fs must be > 0")
    return np.arange(n) / float(fs)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_line_plot(
    x: np.ndarray,
    series: Sequence[tuple[np.ndarray, str]],
    *,
    path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (10.0, 4.0),
    dpi: int = 120,
) -> str:
    """Render one or more ``(y, label)`` series against ``x`` and save to ``path``."""
    _ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=figsize)
    try:
        for y, label in series:
            y = np.asarray(y)
            if y.shape != x.shape:
                raise ValueError(
                    f"series '{label}' length {y.shape} does not match x length {x.shape}"
                )
            ax.plot(x, y, label=label, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if len(series) > 1 or series[0][1]:
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
    return path


def plot_elevation(
    elevation: np.ndarray,
    fs: float,
    path: str,
    *,
    title: str = "Processed elevation",
    label: str = "elevation",
    ylabel: str = "elevation (mm)",
    unit_scale: float = 1e3,
    figsize: tuple[float, float] = (18.0, 8.0),
    dpi: int = 180,
) -> str:
    """Save a line plot of a (processed) elevation time series.

    Parameters
    ----------
    elevation
        1-D array of elevation samples.
    fs
        Sample rate in Hz, used to build the time axis.
    path
        Output image path (file extension determines the format, e.g. ``.png``).
    """
    y = np.asarray(elevation, dtype=float).ravel() * float(unit_scale)
    t = _time_axis(y.size, fs)
    return _save_line_plot(
        t,
        [(y, label)],
        path=path,
        title=title,
        xlabel="time (s)",
        ylabel=ylabel,
        figsize=figsize,
        dpi=dpi,
    )


def plot_acceleration(
    accel: np.ndarray,
    fs: float,
    path: str,
    *,
    title: str = "Acceleration",
    labels: Sequence[str] | None = None,
    ylabel: str = "acceleration (m/s²)",
    figsize: tuple[float, float] = (18.0, 8.0),
    dpi: int = 180,
) -> str:
    """Save a line plot of an acceleration time series.

    ``accel`` may be a 1-D array (single axis, typically vertical) or a 2-D
    array of shape ``(N, k)`` with one column per axis (e.g. ``(N, 3)`` for
    a 3-axis accelerometer).  ``labels`` overrides the per-axis legend
    labels; defaults to ``"a"`` / ``"a_x"``, ``"a_y"``, ``"a_z"``.
    """
    a = np.asarray(accel, dtype=float)
    if a.ndim == 1:
        cols = a[:, None]
    elif a.ndim == 2:
        cols = a
    else:
        raise ValueError("accel must be 1-D or 2-D")

    n_axes = cols.shape[1]
    if labels is None:
        if n_axes == 1:
            labels = ["a"]
        elif n_axes == 3:
            labels = ["a_x", "a_y", "a_z"]
        else:
            labels = [f"a{i}" for i in range(n_axes)]
    if len(labels) != n_axes:
        raise ValueError(
            f"labels length {len(labels)} does not match number of axes {n_axes}"
        )

    t = _time_axis(cols.shape[0], fs)
    series = [(cols[:, i], labels[i]) for i in range(n_axes)]
    return _save_line_plot(
        t,
        series,
        path=path,
        title=title,
        xlabel="time (s)",
        ylabel=ylabel,
        figsize=figsize,
        dpi=dpi,
    )


def plot_comparison(
    series_a: np.ndarray,
    series_b: np.ndarray,
    fs: float,
    path: str,
    *,
    label_a: str = "series A",
    label_b: str = "series B",
    title: str = "Comparison",
    ylabel: str = "elevation (mm)",
    unit_scale: float = 1e3,
    figsize: tuple[float, float] = (18.0, 8.0),
    dpi: int = 180,
) -> str:
    """Save a line plot overlaying two series sampled at the same rate.

    The two arrays may have different lengths -- they are aligned to a
    common time axis starting at ``t = 0`` and truncated to the shorter
    of the two so the comparison is well-defined.
    """
    a = np.asarray(series_a, dtype=float).ravel() * float(unit_scale)
    b = np.asarray(series_b, dtype=float).ravel() * float(unit_scale)
    n = int(min(a.size, b.size))
    if n == 0:
        raise ValueError("series_a and series_b must be non-empty")
    a = a[:n]
    b = b[:n]
    t = _time_axis(n, fs)
    return _save_line_plot(
        t,
        [(a, label_a), (b, label_b)],
        path=path,
        title=title,
        xlabel="time (s)",
        ylabel=ylabel,
        figsize=figsize,
        dpi=dpi,
    )
