"""End-to-end wiring: raw telemetry -> windows -> deviations -> hidden state.

This module is the single place that decides what the model is allowed to see.
Ground-truth tables (``passes``, ``disturbances``, ``defects``) are accepted only
by the evaluation helpers, never by the inference path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..factory.simulator import LineSimulator, SimResult
from ..factory.scenarios import Scenario
from ..factory.topology import LineTopology
from ..features.baseline import NominalBaseline
from ..features.windows import WindowSpec, aggregate_windows, attach_environment
from .shadow import ShadowConfig, ShadowSensor


@dataclass
class TwinContext:
    """A fitted twin: topology, frozen baseline, windowing and calibration."""

    line: LineTopology
    baseline: NominalBaseline
    spec: WindowSpec
    shadow_cfg: ShadowConfig
    calibration: dict = None

    def new_sensor(self) -> ShadowSensor:
        return ShadowSensor(self.line, self.shadow_cfg)


def simulate(line: LineTopology, scenario: Scenario, seed: int) -> SimResult:
    """Run one scenario against the line."""
    sim = LineSimulator(line, seed=seed)
    return sim.run(
        scenario.n_vehicles, scenario.disturbances, run_id=scenario.scenario_id
    )


def build_windows(
    res: SimResult, line: LineTopology, spec: WindowSpec
) -> pd.DataFrame:
    """Aggregate a run's *observed* telemetry into scored-ready windows."""
    w = aggregate_windows(res.telemetry, res.vehicles, line, spec)
    return attach_environment(w, res.environment)


def fit_context(
    line: LineTopology,
    nominal: SimResult,
    calibration_run: SimResult | None = None,
    spec: WindowSpec | None = None,
    shadow_cfg: ShadowConfig | None = None,
    target_window_fpr: float = 0.01,
) -> TwinContext:
    """Fit the nominal baseline and calibrate the detector.

    ``nominal`` fits the expectation of normal behaviour. ``calibration_run`` is
    a *second, independent* disturbance-free run used to set the correlation
    correction and the detection threshold. Using a held-out run matters: fitting
    the baseline and reading the null distribution off the same data understates
    the false-alarm rate, because the baseline has already absorbed that run's
    particular noise.
    """
    spec = spec or WindowSpec.for_line(line)
    w = build_windows(nominal, line, spec)
    baseline = NominalBaseline.fit(w, nominal.telemetry, line)
    ctx = TwinContext(
        line=line,
        baseline=baseline,
        spec=spec,
        shadow_cfg=shadow_cfg or ShadowConfig(),
    )

    cal_src = calibration_run if calibration_run is not None else nominal
    cal_windows = build_windows(cal_src, line, spec)
    cal_scored = baseline.score(cal_windows, line)
    sensor = ctx.new_sensor()
    ctx.calibration = sensor.calibrate(
        cal_scored,
        baseline.sigma_blocked,
        baseline.sigma_starved,
        target_window_fpr=target_window_fpr,
    )
    # calibrate() mutates the config object the context holds, so the settings
    # propagate to every sensor built from this context.
    ctx.shadow_cfg = sensor.cfg
    ctx.calibration["held_out_calibration"] = calibration_run is not None
    return ctx


def infer(ctx: TwinContext, res: SimResult) -> tuple:
    """Run the full inference path on a simulated run.

    Returns ``(scored_windows, shadow_frame, sensor)``. The sensor is returned so
    callers can reach its per-window ``ShadowResult`` objects for explanation.
    """
    w = build_windows(res, ctx.line, ctx.spec)
    scored = ctx.baseline.score(w, ctx.line)
    sensor = ctx.new_sensor()
    shadow = sensor.run(
        scored, ctx.baseline.sigma_blocked, ctx.baseline.sigma_starved
    )
    return scored, shadow, sensor
