"""Distribution-shift robustness (Round 2 brief §15): does the twin still
work when the simulated factory it is *evaluated* on differs from the one
it was *calibrated* on?

Every other experiment in this repository fits ``TwinContext`` and draws its
test episodes from the same simulator distribution -- appropriate for
measuring what the method can do, but silent on whether it has overfit its
calibration (``NominalBaseline``'s expectations, ``ShadowSensor``'s
``tau``/``detect_llr``) to one particular noise regime.

This experiment fits the context once, on the standard line, then evaluates
it -- with NO re-fitting, NO re-calibration -- against episodes drawn from a
``perturb_line``-shifted version of the same topology: higher process noise
and a higher micro-stop rate. This is deliberately the harder, more honest
test: a plant's calibration does not get refreshed the instant conditions
drift, and the useful question is how much that costs, not whether a
freshly-refit model still works (of course it would).

Every number here is a SIMULATED PROTOTYPE RESULT on synthetic data.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..factory import scenarios as SC
from ..factory.topology import LineTopology, build_line
from ..features.windows import WindowSpec
from . import metrics as M
from .views import full_observability, telemetry_view
from ..twin.pipeline import fit_context, infer, simulate


def perturb_line(
    line: LineTopology,
    noise_mult: float = 1.6,
    microstop_mult: float = 1.8,
    fault_magnitude_mult: float = 1.0,
) -> LineTopology:
    """A structurally identical copy of ``line`` (same stations, tiers,
    buffers, takt -- everything a fitted ``TwinContext`` depends on
    structurally) with higher process-noise and micro-stop rates -- the
    "the shop floor has gotten noisier since we calibrated" scenario.

    ``fault_magnitude_mult`` is returned for the caller's convenience (it
    scales injected-disturbance magnitude at evaluation time, not a station
    property) rather than applied here.
    """
    new = copy.deepcopy(line)
    for s in new.stations:
        s.process_noise_cv = float(np.clip(s.process_noise_cv * noise_mult, 0.01, 0.5))
        s.microstop_rate = float(np.clip(s.microstop_rate * microstop_mult, 0.0, 0.2))
    return new


@dataclass
class DistributionShiftConfig:
    line_config: str = "configs/line_42.yaml"
    line_seed: int = 7
    nominal_vehicles: int = 2600
    calibration_vehicles: int = 2200
    n_episodes: int = 12
    episode_vehicles: int = 1200
    target_window_fpr: float = 0.01
    noise_mult: float = 1.6
    microstop_mult: float = 1.8
    fault_magnitude_mult: float = 1.15
    episode_seed_base: int = 9500


def run_distribution_shift_experiment(
    cfg: DistributionShiftConfig | None = None,
    out_dir: str | Path = "results",
    verbose: bool = True,
) -> dict:
    cfg = cfg or DistributionShiftConfig()
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    configured = build_line(cfg.line_config, seed=cfg.line_seed)
    sim_line = full_observability(configured)
    spec = WindowSpec.for_line(configured)

    # Fit ONCE, on the matched (unperturbed) distribution -- this context is
    # reused unchanged for both the matched and shifted evaluation below.
    nominal_full = simulate(sim_line, SC.nominal_run(cfg.nominal_vehicles), seed=1)
    calib_full = simulate(sim_line, SC.nominal_run(cfg.calibration_vehicles), seed=2)
    nom_v = telemetry_view(nominal_full, configured, sim_line)
    cal_v = telemetry_view(calib_full, configured, sim_line)
    ctx = fit_context(configured, nom_v, calibration_run=cal_v, spec=spec,
                       target_window_fpr=cfg.target_window_fpr)

    shifted_sim_line = full_observability(
        perturb_line(configured, cfg.noise_mult, cfg.microstop_mult)
    )

    rows = []
    for regime, phys_line in (("matched", sim_line), ("shifted", shifted_sim_line)):
        for i in range(cfg.n_episodes):
            seed = cfg.episode_seed_base + i
            scen = SC.random_episode(configured, seed=seed, n_vehicles=cfg.episode_vehicles)
            if regime == "shifted" and cfg.fault_magnitude_mult != 1.0:
                for d in scen.disturbances:
                    if d.kind != "MATERIAL_DELAY":
                        d.magnitude = 1.0 + (d.magnitude - 1.0) * cfg.fault_magnitude_mult

            res_full = simulate(phys_line, scen, seed=seed + 77)
            res_v = telemetry_view(res_full, configured, phys_line)
            scored, shadow, sensor = infer(ctx, res_v)
            wt = shadow[["window"]].drop_duplicates()

            truth = M.episode_truth(res_v, configured, view_line=configured)
            row: Dict = {"regime": regime, "seed": seed, "has_fault": truth.has_fault}
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
            print(f"  regime={regime}: {cfg.n_episodes} episodes done")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "distribution_shift_raw.csv", index=False)

    numeric = [c for c in df.columns if c not in ("regime", "seed", "has_fault")
               and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("regime")[numeric].mean().reset_index()
    summary["n_episodes"] = df.groupby("regime").size().to_numpy()
    summary.to_csv(out_dir / "tables" / "distribution_shift_summary.csv", index=False)

    manifest = {
        "generated_by": "rippletwin.evaluation.distribution_shift.run_distribution_shift_experiment",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "config": asdict(cfg),
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "tables" / "distribution_shift_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    if verbose:
        print(summary.to_string(index=False))
        print(f"done in {time.time() - t0:.0f}s")
    return {"raw": df, "summary": summary, "manifest": manifest}
