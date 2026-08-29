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
from ..ingest.plant_data import PlantData
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
    #: Opt-in flow+quality evidence fusion for ambiguous blind-station groups
    #: (Plan C / Hybrid 2, see twin/evidence_fusion.py). Defaults False, so
    #: every existing caller of infer() is unaffected. Setting this True
    #: without also setting ``quality_baseline`` still has zero effect, since
    #: the fusion step requires both.
    enable_evidence_fusion: bool = False
    #: A twin.genealogy.QualityBaseline fitted for this same observability
    #: view, required (alongside enable_evidence_fusion=True) for the fusion
    #: step to run. None -- the default, and every caller that has not
    #: explicitly opted in -- means fusion never fires.
    quality_baseline: object = None

    def new_sensor(self) -> ShadowSensor:
        return ShadowSensor(self.line, self.shadow_cfg)


def simulate(line: LineTopology, scenario: Scenario, seed: int) -> SimResult:
    """Run one scenario against the line."""
    sim = LineSimulator(line, seed=seed)
    return sim.run(
        scenario.n_vehicles, scenario.disturbances, run_id=scenario.scenario_id
    )


def as_plant_data(source) -> PlantData:
    """Coerce whatever a caller has into the observable-only view.

    This is the *only* place a ``SimResult`` may enter the inference path, and
    it leaves ground truth behind at the door: everything downstream is typed to
    ``PlantData``, which has no field that could hold the answer. A historian
    export and a simulated run are indistinguishable past this line, which is
    what makes the pilot path testable before real data exists.
    """
    if isinstance(source, PlantData):
        return source
    if hasattr(source, "as_plant_data"):
        return source.as_plant_data()
    raise TypeError(
        f"expected PlantData or SimResult, got {type(source).__name__}"
    )


def build_windows(
    source, line: LineTopology, spec: WindowSpec
) -> pd.DataFrame:
    """Aggregate observed telemetry into scored-ready windows.

    Ambient conditions are optional in the data contract, so they are attached
    only when the plant actually supplied them.
    """
    data = as_plant_data(source)
    w = aggregate_windows(data.telemetry, data.vehicles, line, spec)
    if data.has_environment:
        w = attach_environment(w, data.environment)
    return w


def fit_context(
    line: LineTopology,
    nominal,
    calibration_run=None,
    spec: WindowSpec | None = None,
    shadow_cfg: ShadowConfig | None = None,
    target_window_fpr: float = 0.01,
    watch_target_fpr: float = 0.05,
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
    nominal = as_plant_data(nominal)
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
        watch_target_fpr=watch_target_fpr,
    )
    # calibrate() mutates the config object the context holds, so the settings
    # propagate to every sensor built from this context.
    ctx.shadow_cfg = sensor.cfg
    ctx.calibration["held_out_calibration"] = calibration_run is not None
    return ctx


def infer(ctx: TwinContext, res) -> tuple:
    """Run the full inference path on observable data.

    Returns ``(scored_windows, shadow_frame, sensor)``. The sensor is returned so
    callers can reach its per-window ``ShadowResult`` objects for explanation.
    """
    w = build_windows(res, ctx.line, ctx.spec)
    scored = ctx.baseline.score(w, ctx.line)
    sensor = ctx.new_sensor()
    shadow = sensor.run(
        scored, ctx.baseline.sigma_blocked, ctx.baseline.sigma_starved
    )

    # --- optional evidence fusion for flagged-ambiguous groups (Plan C / Hybrid 2) ---
    # Opt-in only: never overwrites top_station. See twin/evidence_fusion.py.
    # Both enable_evidence_fusion and quality_baseline must be set for this to
    # do anything, so every existing caller -- whose TwinContext has these at
    # their dataclass defaults (False / None) -- is byte-identical to before.
    if getattr(ctx, "enable_evidence_fusion", False) and getattr(ctx, "quality_baseline", None) is not None:
        from . import evidence_fusion as EF
        from . import genealogy as GN
        from .placement import ambiguity as compute_ambiguity

        data = as_plant_data(res)
        if data.has_quality_path and not shadow.empty:
            amb_df = compute_ambiguity(
                ctx.line, ctx.line.observed_indices, ctx.shadow_cfg,
                ctx.baseline.sigma_blocked, ctx.baseline.sigma_starved,
            )
            groups = EF.ambiguous_groups(amb_df)
            if groups:
                wb = GN.window_bounds_from(scored)
                defects = GN.explode_defects(data.inspections)
                quality_state_df = GN.quality_state(
                    ctx.line, defects, wb, ctx.quality_baseline, pool_vehicles=200
                )
                fused_stations, fused_margins = [], []
                for _, row in shadow.iterrows():
                    group = next((g for g in groups if int(row["top_station"]) in g), None)
                    if group is None or quality_state_df.empty:
                        fused_stations.append(row["top_station"])
                        fused_margins.append(np.nan)
                        continue
                    result = EF.fuse_ambiguous_group(
                        row, quality_state_df, group,
                        row.get("t_lo_s", row.get("t_mid_s", 0.0)),
                        row.get("t_hi_s", row.get("t_mid_s", 0.0)), wb,
                    )
                    if result is None:
                        fused_stations.append(row["top_station"])
                        fused_margins.append(np.nan)
                    else:
                        fused_stations.append(result["fused_top_station"])
                        fused_margins.append(result["fused_llr_margin"])
                shadow = shadow.assign(fused_top_station=fused_stations, fused_llr_margin=fused_margins)

    return scored, shadow, sensor
