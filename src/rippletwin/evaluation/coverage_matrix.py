"""Sensor coverage matrix (Round 2 brief §16): 100/75/50/25/10% coverage,
crossed with which stations go dark -- random vs. the most valuable ones
first (``factory.topology.apply_coverage``'s ``strategy="critical"``, built
in Phase 4 on ``twin.placement``'s value-of-information ranking).

This is a separate, smaller experiment from ``evaluation.experiments``'s
own coverage sweep (which stays untouched -- it backs the flagship,
already-published README numbers at 4 coverage levels and random-only
demotion) rather than an edit to it, so extending the coverage grid here
carries no risk to those numbers. It reuses the same lower-level building
blocks (``fit_context``, ``infer``, ``episode_truth``,
``evaluate_localization``, ``evaluate_false_alarms``) at a smaller episode
count, purpose-built for the specific random-vs-critical comparison.

Every number here is a SIMULATED PROTOTYPE RESULT on synthetic data.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..factory import scenarios as SC
from ..factory.topology import apply_coverage, build_line
from ..features.windows import WindowSpec
from . import metrics as M
from .views import full_observability, telemetry_view
from ..twin.pipeline import build_windows, fit_context, infer, simulate

#: The brief asks for a 10% level; on line_42 that floor is not reachable --
#: 4 inspection gates + station 0 are always instrumented (see
#: apply_coverage's "protected" set), which alone is 5/42 = 11.9% of the
#: line. 0.12 is the lowest coverage this config can actually produce, and
#: that floor is itself a real, reportable finding: a plant's minimum
#: achievable coverage is bounded below by the points nobody would ever
#: leave uninstrumented, not by an arbitrary choice.
DEFAULT_COVERAGES = (1.00, 0.75, 0.50, 0.25, 0.12)
STRATEGIES = ("random", "critical")


@dataclass
class CoverageMatrixConfig:
    line_config: str = "configs/line_42.yaml"
    line_seed: int = 7
    coverages: Sequence[float] = DEFAULT_COVERAGES
    strategies: Sequence[str] = STRATEGIES
    n_episodes: int = 10
    episode_vehicles: int = 1200
    nominal_vehicles: int = 2200
    calibration_vehicles: int = 1800
    target_window_fpr: float = 0.02
    episode_seed_base: int = 8800


def run_coverage_matrix(
    cfg: CoverageMatrixConfig | None = None,
    out_dir: str | Path = "results",
    verbose: bool = True,
) -> dict:
    cfg = cfg or CoverageMatrixConfig()
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    configured = build_line(cfg.line_config, seed=cfg.line_seed)
    sim_line = full_observability(configured)
    spec = WindowSpec.for_line(configured)

    nominal_full = simulate(sim_line, SC.nominal_run(cfg.nominal_vehicles), seed=1)
    calib_full = simulate(sim_line, SC.nominal_run(cfg.calibration_vehicles), seed=2)

    rows = []
    for coverage in cfg.coverages:
        strategies = ["random"] if coverage >= 0.999 else list(cfg.strategies)
        for strategy in strategies:
            view = (
                full_observability(configured) if coverage >= 0.999
                else apply_coverage(configured, coverage, seed=11, strategy=strategy)
            )
            nom_v = telemetry_view(nominal_full, view, sim_line)
            cal_v = telemetry_view(calib_full, view, sim_line)
            ctx = fit_context(
                view, nom_v, calibration_run=cal_v, spec=spec,
                target_window_fpr=cfg.target_window_fpr,
            )

            for i in range(cfg.n_episodes):
                seed = cfg.episode_seed_base + i
                scen = SC.random_episode(configured, seed=seed, n_vehicles=cfg.episode_vehicles)
                res_full = simulate(sim_line, scen, seed=seed + 77)
                res_v = telemetry_view(res_full, view, sim_line)
                scored, shadow, sensor = infer(ctx, res_v)
                # shadow already carries t_mid_s/v_start/v_end per row (it is
                # ShadowSensor.run()'s own output); evaluate_localization/
                # evaluate_false_alarms merge window_times in on "window" and
                # would suffix any overlapping column names, so window_times
                # here supplies only the join key.
                wt = shadow[["window"]].drop_duplicates()

                truth = M.episode_truth(res_v, configured, view_line=view)
                row = {
                    "coverage": coverage, "strategy": strategy, "seed": seed,
                    "has_fault": truth.has_fault,
                    "source_hidden": truth.source_is_hidden if truth.has_fault else None,
                }
                if truth.has_fault:
                    m = M.evaluate_localization(shadow, wt, truth)
                    row.update({f"loc_{k}": v for k, v in m.items()})
                f = M.evaluate_false_alarms(shadow, truth, wt)
                if f:
                    row.update({f"fa_{k}": v for k, v in f.items()})
                det = shadow[shadow["detected"]]
                row["mean_confidence"] = float(det["group_prob"].mean()) if len(det) else np.nan
                rows.append(row)
            if verbose:
                print(f"  coverage={coverage:.2f} strategy={strategy}: {cfg.n_episodes} episodes done")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "coverage_matrix_raw.csv", index=False)

    agg = {}
    for c in df.columns:
        if c in ("coverage", "strategy", "seed", "has_fault", "source_hidden"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            agg[c] = "mean"
    summary = df.groupby(["coverage", "strategy"]).agg(agg).reset_index()
    summary["n_episodes"] = df.groupby(["coverage", "strategy"]).size().to_numpy()
    summary.to_csv(out_dir / "tables" / "coverage_matrix_summary.csv", index=False)

    manifest = {
        "generated_by": "rippletwin.evaluation.coverage_matrix.run_coverage_matrix",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "config": asdict(cfg),
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "tables" / "coverage_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    if verbose:
        print(summary.to_string(index=False))
        print(f"done in {time.time() - t0:.0f}s")
    return {"raw": df, "summary": summary, "manifest": manifest}
