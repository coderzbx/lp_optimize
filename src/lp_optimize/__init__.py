"""lp_optimize: elevation data optimization for road roughness computation.

Three pipelines are exposed, matching the three questions in `需求文档`:

* :func:`pipeline_idle_off`   -- engine off, vehicle still (problem 1)
* :func:`pipeline_idle_on`    -- engine on, vehicle still + accelerometer (problem 2)
* :func:`pipeline_driving`    -- vehicle driving + IMU (problem 3)

The lower-level building blocks live in their own modules and can be reused
independently.
"""

from .pipeline import pipeline_idle_off, pipeline_idle_on, pipeline_driving
from .metrics import std_rms, welch_psd
from .plotting import plot_elevation, plot_acceleration, plot_comparison

__all__ = [
    "pipeline_idle_off",
    "pipeline_idle_on",
    "pipeline_driving",
    "std_rms",
    "welch_psd",
    "plot_elevation",
    "plot_acceleration",
    "plot_comparison",
]
