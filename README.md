# lp_optimize

Elevation-data optimization for road roughness (IRI) computation, addressing
the three questions in `需求文档`:

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
| `lp_optimize.pipeline`      | The three end-to-end pipelines                          |
