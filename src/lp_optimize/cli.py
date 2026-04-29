"""Command-line interface for the three CSV processing scenarios.

Run ``python -m lp_optimize --help`` (or, after install, ``lp-optimize
--help``) for usage.

The three subcommands map onto the three scenarios from ``需求文档``:

* ``idle-off``  -- engine off (problem 1).  Wraps :func:`pipeline_idle_off`.
* ``idle-on``   -- engine on, vehicle still + accelerometer (problem 2).
  Wraps :func:`pipeline_idle_on`.
* ``align``     -- post-process a Result_*.csv produced by ``idle-on`` so
  its fluctuation level matches a reference Result_*.csv produced by
  ``idle-off``.  This implements the third bullet of the task: "通过程序
  将 Result_C040.csv 的高程值优化到与 Result_C001.csv 类似".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .filters import butter_filter
from .io_csv import read_input_csv, read_result_csv, write_result_csv
from .pipeline import pipeline_idle_off, pipeline_idle_on
from .preprocessing import highpass_detrend


# ---------------------------------------------------------------------------
# Scenarios 1 & 2
# ---------------------------------------------------------------------------


def _print_metrics(label: str, elevation_m: np.ndarray) -> None:
    std = float(np.std(elevation_m))
    rms = float(np.sqrt(np.mean(elevation_m ** 2)))
    pp = float(np.ptp(elevation_m))
    print(
        f"[{label}] n={elevation_m.size}  "
        f"std={std * 1e3:.4f} mm  "
        f"rms={rms * 1e3:.4f} mm  "
        f"pp={pp * 1e3:.3f} mm"
    )


def cmd_idle_off(args: argparse.Namespace) -> int:
    elev_m, _ = read_input_csv(args.input, need_accel=False)
    decimate_to = None if args.no_decimate else args.decimate_to
    result = pipeline_idle_off(elev_m, fs=args.fs, decimate_to=decimate_to)
    write_result_csv(args.output, result.elevation, fs=result.fs)
    _print_metrics("idle-off", result.elevation)
    print(f"wrote {args.output}  (fs_out={result.fs:g} Hz)")
    return 0


def cmd_idle_on(args: argparse.Namespace) -> int:
    elev_m, accel = read_input_csv(args.input, need_accel=True)
    assert accel is not None  # for type checkers
    decimate_to = None if args.no_decimate else args.decimate_to
    result = pipeline_idle_on(
        elev_m,
        accel,
        fs=args.fs,
        use_anc=not args.no_anc,
        decimate_to=decimate_to,
    )
    write_result_csv(args.output, result.elevation, fs=result.fs)
    _print_metrics("idle-on", result.elevation)
    print(f"wrote {args.output}  (fs_out={result.fs:g} Hz)")
    return 0


# ---------------------------------------------------------------------------
# Scenario 3 -- align Result_C040 to Result_C001
# ---------------------------------------------------------------------------


def align_to_reference(
    target_m: np.ndarray,
    reference_m: np.ndarray,
    fs: float,
    *,
    drift_cutoff: float = 0.05,
    smooth_cutoff: float | None = None,
    match_std: bool = True,
) -> np.ndarray:
    """Optimise ``target_m`` so its fluctuation level matches ``reference_m``.

    The two series do **not** need to have the same length: the reference
    is only used to derive a target standard deviation (= "波动率").

    Steps
    -----
    1. Re-apply the same low-frequency de-trend used by the idle pipelines
       so any residual drift in the target is removed.
    2. Optionally low-pass at ``smooth_cutoff`` Hz so the spectral content
       cannot extend beyond the reference's band.  Defaults to the higher
       of 40 Hz and ``0.45 * fs`` (whichever is smaller), which is a safe
       sub-Nyquist cap.
    3. If ``match_std``, scale the target so that
       ``std(target) == std(reference)``.  This is the "fluctuation-rate
       alignment" step.
    """
    target = np.asarray(target_m, dtype=float).copy()
    reference = np.asarray(reference_m, dtype=float)

    target = highpass_detrend(target, fs=fs, fc=drift_cutoff)

    if smooth_cutoff is None:
        smooth_cutoff = min(40.0, 0.45 * fs)
    if smooth_cutoff is not None and smooth_cutoff < 0.5 * fs:
        target = butter_filter(
            target, fs=fs, cutoff=smooth_cutoff, btype="lowpass", order=4
        )

    if match_std:
        s_ref = float(np.std(reference))
        s_tgt = float(np.std(target))
        if s_tgt > 1e-12 and s_ref > 0:
            target = target * (s_ref / s_tgt)

    return target


def cmd_align(args: argparse.Namespace) -> int:
    ref_m, fs_ref = read_result_csv(args.reference)
    tgt_m, fs_tgt = read_result_csv(args.target)
    if abs(fs_ref - fs_tgt) > 1e-6:
        print(
            f"warning: reference fs={fs_ref:g} Hz != target fs={fs_tgt:g} Hz; "
            f"using target fs for filtering",
            file=sys.stderr,
        )
    aligned = align_to_reference(
        tgt_m,
        ref_m,
        fs=fs_tgt,
        match_std=not args.no_match_std,
    )
    write_result_csv(args.output, aligned, fs=fs_tgt)

    _print_metrics("reference ", ref_m)
    _print_metrics("input     ", tgt_m)
    _print_metrics("aligned   ", aligned)
    print(f"wrote {args.output}  (fs_out={fs_tgt:g} Hz)")
    return 0


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------


def _add_idle_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="input CSV (col2 = elevation in mm)")
    p.add_argument("output", type=Path, help="output Result_*.csv path")
    p.add_argument(
        "--fs",
        type=float,
        default=1000.0,
        help="input sample rate in Hz (default: 1000)",
    )
    p.add_argument(
        "--decimate-to",
        type=float,
        default=100.0,
        help="output sample rate in Hz after anti-alias decimation "
        "(default: 100)",
    )
    p.add_argument(
        "--no-decimate",
        action="store_true",
        help="keep the original input sample rate (overrides --decimate-to)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lp-optimize",
        description=(
            "Process laser elevation CSV files for road roughness "
            "computation."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser(
        "idle-off",
        help="problem 1: engine-off elevation cleanup",
        description=(
            "Read an engine-off CSV (column 2 = elevation in mm) and "
            "write the processed Result_*.csv (column = elevation_mm)."
        ),
    )
    _add_idle_common(p1)
    p1.set_defaults(func=cmd_idle_off)

    p2 = sub.add_parser(
        "idle-on",
        help="problem 2: engine-on elevation + accelerometer compensation",
        description=(
            "Read an engine-on CSV (col 2 = elevation in mm, col 3 = "
            "vertical accelerometer in m/s^2) and write the compensated "
            "Result_*.csv."
        ),
    )
    _add_idle_common(p2)
    p2.add_argument(
        "--no-anc",
        action="store_true",
        help="disable the LMS adaptive noise canceller stage",
    )
    p2.set_defaults(func=cmd_idle_on)

    p3 = sub.add_parser(
        "align",
        help="problem 3: align an idle-on result to an idle-off reference",
        description=(
            "Re-process a Result_*.csv produced by `idle-on` so its "
            "fluctuation level matches the Result_*.csv produced by "
            "`idle-off`."
        ),
    )
    p3.add_argument(
        "reference",
        type=Path,
        help="reference Result_*.csv (e.g. Result_C001.csv from idle-off)",
    )
    p3.add_argument(
        "target",
        type=Path,
        help="target Result_*.csv to align (e.g. Result_C040.csv from idle-on)",
    )
    p3.add_argument(
        "output",
        type=Path,
        help="output path for the aligned CSV",
    )
    p3.add_argument(
        "--no-match-std",
        action="store_true",
        help="skip the std rescale step (only re-detrend / low-pass)",
    )
    p3.set_defaults(func=cmd_align)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover - module run as script
    raise SystemExit(main())
