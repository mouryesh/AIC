"""Decision-vs-outcome stress testing, oracle vs. sensor-corrupted views.

Following Saad Saoud (2026), "Ground-Truth-Aware Stress Testing of a
Closed-Loop Digital Twin Under Sensor Drift and Missing Data"
(arXiv:2608.14917): separate the question "did sensor corruption change
the decision" from "did it change the consequence." RippleTwin's existing
architecture already supplies every precondition this needs --
evaluation.experiments.run_experiment's own docstring states coverage
levels are VIEWS over one physics run, which is exactly the paired /
common-random-numbers design the paper argues for. This module adds
nothing to the estimator; it is a read-only consumer of the existing
simulator, twin.pipeline.infer, and factory.sensor_health's fault
injection.

Decision proxy: (top_station, detected) -- see runbook step B0.3 for why
the simpler proxy was chosen over recommend.dispatch's full payload for
this first version.

Fault-injection wiring (runbook step B0.2): confirmed against
tests/test_sensor_dynamics.py's real end-to-end usage --
``apply_sensor_faults`` is applied directly to an already-``telemetry_view``-
projected ``SimResult``'s ``.telemetry`` attribute, and the mutated
``SimResult`` is then fed straight to ``infer(ctx, res_v)`` (which itself
calls ``build_windows`` internally). No separate re-windowing step is
needed by the caller.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..factory import scenarios as SC
from ..factory.sensor_health import SensorFault, apply_sensor_faults
from ..factory.topology import LineTopology, apply_coverage, build_line
from ..features.windows import WindowSpec
from ..twin.pipeline import build_windows, fit_context, infer, simulate
from .experiments import ExperimentConfig
from .views import full_observability, telemetry_view


@dataclass
class StressCondition:
    """One point in the stress grid: a coverage level plus an optional
    sensor-fault severity applied on top of it."""

    coverage: float
    fault_kind: str | None = None       # None, or one of sensor_health's DROPOUT/INTERMITTENT/NOISY/STALE
    fault_fraction_of_run: float = 0.0  # what fraction of the episode's duration the fault covers
    label: str = ""


def _oracle_condition(coverage: float) -> StressCondition:
    return StressCondition(coverage=coverage, fault_kind=None, label=f"oracle_cov{coverage:.2f}")


def decision_mismatch(oracle_shadow: pd.DataFrame, stress_shadow: pd.DataFrame) -> pd.DataFrame:
    """Per-window comparison of (top_station, detected) between two paired runs.

    Both frames must come from the SAME episode/seed (paired by construction
    -- see run_stress_test). Returns one row per window with a boolean
    ``decision_mismatch`` column.
    """
    o = oracle_shadow[["window", "top_station", "detected"]].rename(
        columns={"top_station": "oracle_station", "detected": "oracle_detected"}
    )
    s = stress_shadow[["window", "top_station", "detected"]].rename(
        columns={"top_station": "stress_station", "detected": "stress_detected"}
    )
    merged = o.merge(s, on="window", how="inner")
    merged["decision_mismatch"] = (
        (merged["oracle_station"] != merged["stress_station"])
        | (merged["oracle_detected"] != merged["stress_detected"])
    )
    return merged


def run_stress_test(
    cfg: ExperimentConfig | None = None,
    stress_grid: Sequence[StressCondition] | None = None,
    out_dir: str | Path = "results_stress_test",
    verbose: bool = True,
) -> dict:
    """Run the oracle-vs-stress paired comparison over the existing episode set.

    Reuses cfg's episode/seed generation from ExperimentConfig so results
    are directly cross-referenceable against the flagship coverage-sweep
    tables. Writes to out_dir, never to the existing results/ directory.
    """
    cfg = cfg or ExperimentConfig()
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    configured = build_line(cfg.line_config, seed=cfg.line_seed)
    sim_line = full_observability(configured)
    spec = WindowSpec.for_line(configured, width=cfg.window_width, stride=cfg.window_stride)

    oracle_line = full_observability(configured)

    stress_grid = stress_grid or [
        StressCondition(coverage=0.75, fault_kind=None, label="cov75_clean"),
        StressCondition(coverage=0.50, fault_kind=None, label="cov50_clean"),
        StressCondition(coverage=0.75, fault_kind="DROPOUT", fault_fraction_of_run=0.30, label="cov75_dropout30pct"),
        StressCondition(coverage=0.50, fault_kind="NOISY", fault_fraction_of_run=0.30, label="cov50_noisy30pct"),
        StressCondition(coverage=0.75, fault_kind="STALE", fault_fraction_of_run=0.30, label="cov75_stale30pct"),
    ]

    nominal_full = simulate(sim_line, SC.nominal_run(cfg.nominal_vehicles), seed=1)
    calib_full = simulate(sim_line, SC.nominal_run(cfg.calibration_vehicles), seed=2)

    oracle_nom_v = telemetry_view(nominal_full, oracle_line, sim_line)
    oracle_cal_v = telemetry_view(calib_full, oracle_line, sim_line)
    oracle_ctx = fit_context(
        oracle_line, oracle_nom_v, calibration_run=oracle_cal_v, spec=spec,
        target_window_fpr=cfg.target_window_fpr,
    )

    coverage_ctx: Dict[float, object] = {}
    coverage_views: Dict[float, LineTopology] = {}
    for cond in stress_grid:
        if cond.coverage not in coverage_views:
            v = apply_coverage(configured, cond.coverage, seed=int(round(cond.coverage * 1000)))
            coverage_views[cond.coverage] = v
            nom_v = telemetry_view(nominal_full, v, sim_line)
            cal_v = telemetry_view(calib_full, v, sim_line)
            coverage_ctx[cond.coverage] = fit_context(
                v, nom_v, calibration_run=cal_v, spec=spec,
                target_window_fpr=cfg.target_window_fpr,
            )

    def episode_seeds(split: str) -> List[int]:
        if split == "tune":
            return [cfg.tune_seed_base + i for i in range(cfg.n_tune_episodes)]
        return [cfg.test_seed_base + i for i in range(cfg.n_test_episodes)]

    rows: List[dict] = []
    self_check_rows: List[dict] = []

    for split in ("test",):  # stress testing runs on the held-out split only
        for seed in episode_seeds(split):
            scen = SC.random_episode(configured, seed=seed, n_vehicles=cfg.episode_vehicles)
            res_full = simulate(sim_line, scen, seed=seed + 77)

            # --- oracle pass: full observability, no sensor faults -------
            oracle_res_v = telemetry_view(res_full, oracle_line, sim_line)
            _, oracle_shadow, _ = infer(oracle_ctx, oracle_res_v)

            # Mandatory self-check: oracle vs itself must be zero mismatch.
            self_mismatch = decision_mismatch(oracle_shadow, oracle_shadow)
            self_check_rows.append(
                {"seed": seed, "self_mismatch_rate": float(self_mismatch["decision_mismatch"].mean())}
            )

            for cond in stress_grid:
                v = coverage_views[cond.coverage]
                ctx = coverage_ctx[cond.coverage]
                res_v = telemetry_view(res_full, v, sim_line)

                if cond.fault_kind is not None:
                    # Apply an existing sensor_health fault to one observed
                    # station, covering fault_fraction_of_run of the episode's
                    # duration. Wiring confirmed against
                    # tests/test_sensor_dynamics.py: mutate res_v.telemetry in
                    # place with apply_sensor_faults, then feed res_v straight
                    # into infer() -- no separate rebuild-windows step needed.
                    if v.observed_indices:
                        target_station = int(v.observed_indices[len(v.observed_indices) // 2])
                        t_span = float(
                            res_v.telemetry["t_start_s"].max() - res_v.telemetry["t_start_s"].min()
                        )
                        fault = SensorFault(
                            station=target_station,
                            kind=cond.fault_kind,
                            t_start_s=0.0,
                            t_end_s=t_span * cond.fault_fraction_of_run,
                        )
                        res_v.telemetry = apply_sensor_faults(res_v.telemetry, [fault], seed=seed)

                _, stress_shadow, _ = infer(ctx, res_v)

                dm = decision_mismatch(oracle_shadow, stress_shadow)
                dmr = float(dm["decision_mismatch"].mean()) if len(dm) else np.nan

                # Outcome gap: median station-distance between oracle's and
                # stress condition's named station, on windows where the
                # oracle detected something. A cheap, already-available proxy
                # for "did the consequence change" -- see runbook B0.3.
                paired = oracle_shadow[["window", "top_station", "detected"]].merge(
                    stress_shadow[["window", "top_station"]], on="window", suffixes=("_oracle", "_stress")
                )
                detected_rows = paired[paired["detected"]]
                if len(detected_rows):
                    outcome_gap = float(
                        (detected_rows["top_station_oracle"] - detected_rows["top_station_stress"]).abs().median()
                    )
                else:
                    outcome_gap = np.nan

                rows.append(
                    {
                        "seed": seed,
                        "condition": cond.label,
                        "coverage": cond.coverage,
                        "fault_kind": cond.fault_kind,
                        "decision_mismatch_rate": dmr,
                        "outcome_gap_station_distance": outcome_gap,
                        "n_windows": len(dm),
                    }
                )
            if verbose:
                print(f"[stress_test] seed={seed} done ({time.time() - t_start:.0f}s)")

    result_df = pd.DataFrame(rows)
    self_check_df = pd.DataFrame(self_check_rows)
    result_df.to_csv(out_dir / "tables" / "decision_outcome_gap.csv", index=False)
    self_check_df.to_csv(out_dir / "tables" / "oracle_self_check.csv", index=False)

    summary = (
        # dropna=False: a clean condition's fault_kind is None/NaN, and
        # pandas groupby drops NaN-keyed groups by default -- which would
        # silently vanish every "clean" row (fault_kind=None) from the
        # summary. Confirmed against test_stress_test.py's clean-vs-dropout
        # comparison during B3 validation.
        result_df.groupby(["condition", "coverage", "fault_kind"], dropna=False)
        .agg(
            n=("seed", "size"),
            mean_dmr=("decision_mismatch_rate", "mean"),
            median_outcome_gap=("outcome_gap_station_distance", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "tables" / "decision_outcome_gap_summary.csv", index=False)

    manifest = {
        "generated_by": "rippletwin.evaluation.stress_test.run_stress_test",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "methodology_reference": "arXiv:2608.14917 (Saad Saoud, 2026)",
        "config": asdict(cfg),
        "stress_grid": [cond.__dict__ for cond in stress_grid],
        "oracle_self_check_max_mismatch": float(self_check_df["self_mismatch_rate"].max()) if len(self_check_df) else None,
        "runtime_s": round(time.time() - t_start, 1),
    }
    (out_dir / "tables" / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    if verbose:
        print(f"\n[stress_test] done in {time.time() - t_start:.0f}s")
        print(f"[stress_test] oracle self-check max mismatch: {manifest['oracle_self_check_max_mismatch']}")

    return {"detail": result_df, "summary": summary, "self_check": self_check_df, "manifest": manifest}


if __name__ == "__main__":
    run_stress_test()
