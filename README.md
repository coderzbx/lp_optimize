# lp_optimize

Elevation-data optimization for road roughness (IRI) computation, addressing
the three questions in `需求文档`.

> **设计思路（中文）**: 详见 [`docs/思路.md`](docs/思路.md)，文档中按问题
> 1 / 2 / 3 系统说明了物理背景、信号特征、处理流程、关键参数、对应代码入口
> 与工程经验。代码实现与该文档一一对应。

## 三个管线概要

1. **熄火静止** (`pipeline_idle_off`) – cleans a 30-min idle-off laser trace
   via Hampel outlier removal, low-frequency de-trending, low-pass smoothing
   and decimation.  Produces the **noise-floor baseline** (std / RMS / PSD).
2. **点火静止 + 加速度计** (`pipeline_idle_on`) – band-limited frequency-domain
   double-integration of the vertical accelerometer recovers body vertical
   displacement, which is added back to the laser reading to remove engine /
   suspension vibration.  An optional NLMS adaptive noise canceller mops up
   residual coupling.  Output should reach the problem-1 baseline.
3. **行驶 + IMU** (`pipeline_driving`) – complementary-filter attitude
   estimation projects the slant range onto the vertical axis; vertical body
   displacement compensation is applied as in problem 2; the time-domain
   profile is then re-sampled to a uniform spatial grid using vehicle speed
   and band-limited to the IRI wavelength range (0.5 – 50 m).  IRI is
   computed with the standard Golden Car quarter-car model at 80 km/h
   (ASTM E1926).

## Install

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Run the demo

```bash
python -m examples.demo
```

The demo also auto-generates line plots of the processed elevation,
the acceleration data, and an overlay comparison of the engine-off
(problem 1) vs. engine-on compensated (problem 2) elevation series
into `examples/plots/`.

## Process your own CSV files (CLI)

The package ships a thin CSV CLI that wraps the three pipelines.  Input
files are expected to have the elevation reading (in **mm**) in column 2
and, for the engine-on case, the vertical accelerometer reading
(m/s²) in column 3.  A header row is auto-detected.

```bash
# 1. engine off  -> Result_C001.csv
python -m lp_optimize idle-off C001.csv Result_C001.csv --fs 2000

# 2. engine on  + accelerometer  -> Result_C040.csv
python -m lp_optimize idle-on  C040.csv Result_C040.csv --fs 2000

# 3. optimise Result_C040.csv to match Result_C001.csv
python -m lp_optimize align Result_C001.csv Result_C040.csv Result_C040_aligned.csv
```

After ``pip install -e .`` the same commands are available as
``lp-optimize idle-off ...`` etc.  Output ``Result_*.csv`` files keep the
source ``timestamp_s`` plus the absolute optimized elevation
``elevation_mm``.  When available they also include the zero-mean
``fluctuation_mm`` column and the processing-time ``t_s`` column for later
alignment / comparison with other result files.  By default the output is
decimated to 100 Hz; pass ``--no-decimate`` to keep the input rate.


## Generate line plots from your own data

The plotting helpers are exposed at the package top level:

```python
from lp_optimize import plot_elevation, plot_acceleration, plot_comparison

plot_elevation(result1.elevation, fs=result1.fs, path="elev1.png")
plot_acceleration(a_z, fs=2000.0, path="accel.png")
plot_comparison(result1.elevation, result2.elevation, fs=result1.fs,
                path="compare.png", label_a="engine off", label_b="engine on")
```

## Run the tests

```bash
pip install pytest
pytest -q
```

## Module map

| Module                      | Role                                                    |
| --------------------------- | ------------------------------------------------------- |
| `lp_optimize.preprocessing` | Hampel filter, gap interpolation, low-frequency detrend |
| `lp_optimize.filters`       | Zero-phase Butterworth, Savitzky-Golay, decimation      |
| `lp_optimize.imu`           | Accel → displacement (FFT), NLMS adaptive canceller     |
| `lp_optimize.attitude`      | Complementary-filter pitch / roll, vertical projection  |
| `lp_optimize.profile`       | Time → space resampling (uniform Δs)                    |
| `lp_optimize.iri`           | Golden Car quarter-car IRI (ASTM E1926)                 |
| `lp_optimize.metrics`       | std / RMS / Welch PSD                                   |
| `lp_optimize.plotting`      | Line plots for elevation / acceleration + comparison    |
| `lp_optimize.pipeline`      | The three end-to-end pipelines                          |
| `lp_optimize.io_csv`        | CSV read/write helpers (mm ↔ m conversion)              |
| `lp_optimize.cli`           | `lp-optimize` CLI: `idle-off` / `idle-on` / `align`     |
