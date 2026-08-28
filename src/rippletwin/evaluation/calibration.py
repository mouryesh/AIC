"""Calibration (Round 2 brief §26): do the probabilities RippleTwin reports
mean what they claim?

If the system reports 80% probability, roughly 80% of comparable cases
should occur. Checked here for the two genuinely probabilistic,
forward-looking outputs in this repository -- ``twin.predict``'s per-window
bottleneck risk and ``twin.defect_risk``'s per-vehicle defect risk -- via a
reliability table (predicted-probability bucket vs. empirical positive
rate), a Brier score, and an expected calibration error computed only over
bins with enough samples to mean anything.

``ShadowSensor``'s station posterior (``group_prob``) is not re-checked
here: it is already calibrated by construction against a stated
false-alarm target (``ShadowSensor.calibrate``), which is a different,
already-answered question ("how often does this fire on nothing") from the
one this module asks ("when this says 70%, does it happen 70% of the time").

If a bin has too few samples, that is reported explicitly (``reliable``),
not smoothed into a headline number -- see docs/RESULTS.md.

Every number here is a SIMULATED PROTOTYPE RESULT on synthetic data, and on
a corpus this size the calibration numbers below are illustrative of the
method, not a statistically powered claim -- stated plainly rather than
found later.
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
from ..factory.topology import build_line
from ..twin import predict as PR
from ..twin.defect_risk import DefectRiskConfig, RawProcessBaseline, score_vehicles
from ..twin.pipeline import fit_context, infer, simulate
from . import metrics as M
from .defect_prediction import _label_true_defects


def brier_score(y_true: Sequence[bool], p: Sequence[float]) -> float:
    pr = np.clip(np.nan_to_num(np.asarray(p, dtype=float), nan=0.0), 0.0, 1.0)
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((pr - y) ** 2))


def reliability_table(
    y_true: Sequence[bool], p: Sequence[float], n_bins: int = 5, min_bin_n: int = 20,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=bool)
    pr = np.clip(np.nan_to_num(np.asarray(p, dtype=float), nan=0.0), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(pr, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bin_idx == b
        n = int(m.sum())
        mean_pred = float(pr[m].mean()) if n else float("nan")
        emp_rate = float(y[m].mean()) if n else float("nan")
        rows.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]), "n": n,
            "mean_predicted": mean_pred, "empirical_rate": emp_rate,
            "abs_gap": abs(mean_pred - emp_rate) if n else float("nan"),
            "reliable": n >= min_bin_n,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> dict:
    rel = table[table["reliable"]]
    total_n = int(table["n"].sum())
    if rel.empty or total_n == 0:
        return {"ece": float("nan"), "coverage_frac": 0.0, "n_reliable_bins": 0}
    ece = float((rel["abs_gap"] * rel["n"]).sum() / rel["n"].sum())
    return {
        "ece": ece,
        "coverage_frac": float(rel["n"].sum() / total_n),
        "n_reliable_bins": int(len(rel)),
    }


@dataclass
class CalibrationConfig:
    line_config: str = "configs/line_42.yaml"
    line_seed: int = 7
    nominal_vehicles: int = 2200
    calibration_vehicles: int = 1800
    n_episodes: int = 14
    episode_vehicles: int = 1000
    episode_seed_base: int = 9900


def _predict_calibration_pairs(line, ctx, cfg: CalibrationConfig):
    y_true, p = [], []
    for i in range(cfg.n_episodes):
        seed = cfg.episode_seed_base + i
        scen = SC.random_episode(line, seed=seed, n_vehicles=cfg.episode_vehicles)
        res = simulate(line, scen, seed=seed + 77)
        scored, shadow, sensor = infer(ctx, res)
        pred = PR.run_predictor(shadow, line, res.telemetry, scored, ctx.shadow_cfg)
        if pred.empty:
            continue
        truth = M.episode_truth(res, line)
        for _, r in pred.iterrows():
            station = r["station"]
            correctly_localised = (
                truth.has_fault
                and truth.t_start_s <= r["t_mid_s"] <= truth.t_end_s
                and pd.notna(station)
                and abs(float(station) - truth.station) <= 1
            )
            y_true.append(bool(correctly_localised))
            p.append(float(r["risk"]))
    return y_true, p


def _defect_calibration_pairs(line, ctx, raw_bl, model_cfg, cfg: CalibrationConfig):
    y_true, p = [], []
    for i in range(cfg.n_episodes):
        seed = cfg.episode_seed_base + 1000 + i
        scen = SC.random_episode(line, seed=seed, n_vehicles=cfg.episode_vehicles, p_fault=0.8)
        res = simulate(line, scen, seed=seed + 33)
        scored = score_vehicles(line, res.telemetry, ctx.baseline, raw_bl, model_cfg)
        y = _label_true_defects(scored, res.defects)
        y_true.extend(y.tolist())
        p.extend(np.nan_to_num(scored["risk"].to_numpy(dtype=float), nan=0.0).tolist())
    return y_true, p


def run_calibration_experiment(
    cfg: CalibrationConfig | None = None,
    out_dir: str | Path = "results",
    verbose: bool = True,
) -> dict:
    cfg = cfg or CalibrationConfig()
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    line = build_line(cfg.line_config, seed=cfg.line_seed)
    nominal = simulate(line, SC.nominal_run(cfg.nominal_vehicles), seed=1)
    calib = simulate(line, SC.nominal_run(cfg.calibration_vehicles), seed=2)
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)

    raw_bl = RawProcessBaseline.fit(nominal.telemetry, line)
    model_cfg = DefectRiskConfig()
    nom_scored = score_vehicles(line, nominal.telemetry, ctx.baseline, raw_bl, model_cfg)
    model_cfg.fit_scale(nom_scored["_combined"].to_numpy())

    yt_pred, p_pred = _predict_calibration_pairs(line, ctx, cfg)
    yt_def, p_def = _defect_calibration_pairs(line, ctx, raw_bl, model_cfg, cfg)

    results = {}
    for name, (yt, p) in (("bottleneck_risk", (yt_pred, p_pred)), ("defect_risk", (yt_def, p_def))):
        table = reliability_table(yt, p)
        table.to_csv(out_dir / "tables" / f"calibration_{name}_reliability.csv", index=False)
        ece = expected_calibration_error(table)
        results[name] = {
            "n": len(yt),
            "n_positive": int(np.sum(yt)),
            "brier": brier_score(yt, p),
            **ece,
        }
        if verbose:
            print(f"  {name}: n={len(yt)} n_positive={int(np.sum(yt))} "
                  f"brier={results[name]['brier']:.4f} ece={ece['ece']}")

    summary_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "head"})
    summary_df.to_csv(out_dir / "tables" / "calibration_summary.csv", index=False)

    manifest = {
        "generated_by": "rippletwin.evaluation.calibration.run_calibration_experiment",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "config": asdict(cfg),
        "results": results,
        "runtime_s": round(time.time() - t0, 1),
        "caveat": (
            "Small held-out sample: treat calibration numbers as illustrative "
            "of the method, not a statistically powered claim. Bins with too "
            "few samples are marked reliable=False in the reliability tables "
            "and excluded from the ECE, not smoothed over."
        ),
    }
    (out_dir / "tables" / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    if verbose:
        print(f"done in {time.time() - t0:.0f}s")
    return {"summary": summary_df, "results": results, "manifest": manifest}
