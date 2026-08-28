"""Defect-prediction experiment: precision/recall/F1/FPR/Brier score of
``twin.defect_risk`` against a naive historical-rate baseline.

Ground truth for a (vehicle, station) pair comes from ``res.defects`` --
the simulator's own record of each defect's true source station, eval-only
and never given to the model. A pair counts as a true positive class member
if that vehicle really acquired a defect there, whether or not it was ever
caught at a gate: this evaluates *prediction*, which by construction looks
earlier than any inspection result, so gating on "was it caught" would
evaluate something else.

Defects are rare (see ``factory/topology.py``'s ``base_defect_rate``, on the
order of 1e-3 per vehicle-visit), so precision at any operating point that
also catches a useful fraction of real defects is expected to be weak. That
is reported, not hidden -- see docs/RESULTS.md.

Every number here is a SIMULATED PROTOTYPE RESULT on synthetic data.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..factory import scenarios as SC
from ..factory.topology import LineTopology, build_line
from ..twin.defect_risk import (
    DefectRiskConfig,
    RawProcessBaseline,
    coverage_gap_report,
    score_vehicles,
)
from ..twin.pipeline import fit_context, simulate


@dataclass
class DefectPredictionConfig:
    line_config: str = "configs/line_42.yaml"
    line_seed: int = 7
    nominal_vehicles: int = 2600
    calibration_vehicles: int = 2200
    n_episodes: int = 14
    episode_vehicles: int = 1200
    risk_threshold: float = 0.5
    episode_seed_base: int = 8000


def _label_true_defects(scored: pd.DataFrame, defects: pd.DataFrame) -> np.ndarray:
    if defects.empty:
        return np.zeros(len(scored), dtype=bool)
    key = set(
        zip(defects["vehicle_id"].astype(int), defects["source_station"].astype(int))
    )
    pairs = zip(scored["vehicle_id"].astype(int), scored["station"].astype(int))
    return np.array([p in key for p in pairs], dtype=bool)


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0)
    return float(np.mean((p - y_true.astype(float)) ** 2))


def _naive_rate(line: LineTopology, nominal_defects: pd.DataFrame, n_vehicles: int) -> Dict[int, float]:
    """Historical per-station defect rate baseline: the same score for every
    vehicle at a station regardless of current evidence -- the "we already
    know station X is generally worse" detector."""
    if len(nominal_defects):
        counts = nominal_defects["source_station"].value_counts()
        return {s.index: float(counts.get(s.index, 0)) / max(1, n_vehicles) for s in line.stations}
    return {s.index: float(s.base_defect_rate) for s in line.stations}


def run_defect_prediction_experiment(
    cfg: DefectPredictionConfig | None = None,
    out_dir: str | Path = "results",
    verbose: bool = True,
) -> dict:
    cfg = cfg or DefectPredictionConfig()
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

    gaps = coverage_gap_report(line)
    gaps.to_csv(out_dir / "tables" / "defect_prediction_coverage_gaps.csv", index=False)

    naive_rate = _naive_rate(line, nominal.defects, len(nominal.vehicles))

    rows = []
    for i in range(cfg.n_episodes):
        seed = cfg.episode_seed_base + i
        scen = SC.random_episode(line, seed=seed, n_vehicles=cfg.episode_vehicles, p_fault=0.8)
        res = simulate(line, scen, seed=seed + 33)
        scored = score_vehicles(line, res.telemetry, ctx.baseline, raw_bl, model_cfg)
        y_true = _label_true_defects(scored, res.defects)
        y_score = np.nan_to_num(scored["risk"].to_numpy(dtype=float), nan=0.0)
        y_pred = y_score >= cfg.risk_threshold

        naive_score = scored["station"].map(naive_rate).to_numpy(dtype=float)
        naive_thr = float(np.quantile(naive_score, 0.995)) if naive_score.size else 1.0
        naive_pred = naive_score >= naive_thr

        m = _prf(y_true, y_pred)
        mn = _prf(y_true, naive_pred)
        rows.append({
            "seed": seed, "scenario": scen.scenario_id,
            "n_rows": len(scored), "n_true_defects": int(y_true.sum()),
            "model_brier": _brier(y_true, y_score),
            **{f"model_{k}": v for k, v in m.items()},
            **{f"naive_{k}": v for k, v in mn.items()},
        })
        if verbose:
            print(
                f"  episode seed={seed}: n_true={int(y_true.sum())} "
                f"model P={m['precision']:.3f} R={m['recall']:.3f} | "
                f"naive P={mn['precision']:.3f} R={mn['recall']:.3f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "defect_prediction_raw.csv", index=False)

    numeric_cols = [c for c in df.columns if c not in ("seed", "scenario")
                    and pd.api.types.is_numeric_dtype(df[c])]
    summary = {c: float(df[c].mean()) for c in numeric_cols}
    summary["n_episodes"] = int(len(df))
    summary["n_episodes_with_true_defect"] = int((df["n_true_defects"] > 0).sum())
    pd.DataFrame([summary]).to_csv(out_dir / "tables" / "defect_prediction_summary.csv", index=False)

    manifest = {
        "generated_by": "rippletwin.evaluation.defect_prediction.run_defect_prediction_experiment",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "config": asdict(cfg),
        "n_manual_stations_no_coverage": int(len(gaps)),
        "summary": summary,
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "tables" / "defect_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    if verbose:
        print(
            f"defect prediction: model P={summary.get('model_precision', float('nan')):.3f} "
            f"R={summary.get('model_recall', float('nan')):.3f} vs naive "
            f"P={summary.get('naive_precision', float('nan')):.3f} "
            f"R={summary.get('naive_recall', float('nan')):.3f} "
            f"(done in {time.time() - t0:.0f}s)"
        )
    return {"raw": df, "summary": summary, "coverage_gaps": gaps, "manifest": manifest}
