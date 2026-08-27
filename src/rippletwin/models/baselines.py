"""Baselines RippleTwin has to beat.

Each baseline is a real approach a plant could actually deploy today, not a
strawman. Together they isolate exactly which ingredient is doing the work:

    B0  SPC on observed stations
        Statistical process control on each instrumented station's own cycle
        time. This is what most plants genuinely run. It cannot name a station
        it does not measure -- not because it is badly implemented, but because
        nothing in it connects one station to another.

    B1  Anomaly detection, no line structure
        Isolation Forest over each observed station's feature vector. Modern,
        unsupervised, and completely topology-blind: it can tell you a station
        looks unusual, but not whether that station is a cause or a victim.

    B2  Observed-only twin
        RippleTwin's own flow model, restricted to hypotheses about stations
        that have sensors. This is the sharpest comparison in the set: same
        propagation physics, same likelihood, same calibration -- the *only*
        difference is whether un-instrumented stations are allowed to be
        candidates. It isolates the contribution of shadow-sensing itself and
        rules out the objection that RippleTwin's advantage comes from better
        features or better tuning rather than from the idea.

Every baseline emits the same output contract as RippleTwin -- a predicted
source station per window plus a detection flag -- so the metrics are computed
identically for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..factory.topology import LineTopology
from ..twin.shadow import ShadowConfig, ShadowSensor


@dataclass
class BaselineResult:
    """Per-window predictions in the common output contract."""

    name: str
    frame: pd.DataFrame  # window, top_station, detected, score


# ------------------------------------------------------------------- B0: SPC


def spc_baseline(
    scored: pd.DataFrame, line: LineTopology, z_threshold: float = 3.0
) -> BaselineResult:
    """Flag the observed station whose own cycle time deviates most.

    This is Western Electric rule 1 applied per station: the standard control
    chart a plant already has on the wall.
    """
    rows = []
    for w, g in scored.groupby("window", sort=True):
        z = g["z_proc"].to_numpy(dtype=float)
        st = g["station"].to_numpy(dtype=int)
        z = np.where(np.isfinite(z), z, -np.inf)
        if len(z) == 0:
            continue
        j = int(np.argmax(z))
        rows.append(
            {
                "window": int(w),
                "top_station": int(st[j]),
                "score": float(z[j]),
                "detected": bool(z[j] >= z_threshold),
            }
        )
    return BaselineResult("B0_SPC_observed", pd.DataFrame(rows))


# ------------------------------------------- B1: anomaly detection, no topology


def isolation_forest_baseline(
    nominal_scored: pd.DataFrame,
    scored: pd.DataFrame,
    line: LineTopology,
    contamination: float = 0.01,
    seed: int = 0,
) -> BaselineResult:
    """Isolation Forest per observed station, fitted on nominal data.

    Deliberately given the *same* features RippleTwin sees at observed stations,
    so any difference in result comes from the use of line structure rather than
    from an information advantage.
    """
    feats = ["z_proc", "d_blocked", "d_starved"]
    fit_df = nominal_scored.dropna(subset=feats)
    if fit_df.empty:
        return BaselineResult("B1_IsolationForest", pd.DataFrame())

    models: Dict[int, IsolationForest] = {}
    for st, g in fit_df.groupby("station"):
        if len(g) < 30:
            continue
        m = IsolationForest(
            contamination=contamination, random_state=seed, n_estimators=200
        )
        m.fit(g[feats].to_numpy())
        models[int(st)] = m

    # Score every row in one call per station rather than one call per row:
    # sklearn's per-call overhead dominates otherwise, and this baseline is
    # evaluated across every episode and every coverage level.
    work = scored[["window", "station"] + feats].copy()
    work["_score"] = np.nan
    for st, m in models.items():
        sel = (work["station"] == st) & np.isfinite(work[feats]).all(axis=1)
        if not sel.any():
            continue
        X = work.loc[sel, feats].to_numpy(dtype=float)
        # Negated so that larger means more anomalous, matching the others.
        work.loc[sel, "_score"] = -m.score_samples(X)

    work = work.dropna(subset=["_score"])
    if work.empty:
        return BaselineResult("B1_IsolationForest", pd.DataFrame())

    idx = work.groupby("window")["_score"].idxmax()
    best = work.loc[idx]
    frame = pd.DataFrame(
        {
            "window": best["window"].to_numpy(dtype=int),
            "top_station": best["station"].to_numpy(dtype=int),
            "score": best["_score"].to_numpy(dtype=float),
            # score_samples sits near -0.5 at the nominal boundary for this fit,
            # so the negated score crosses ~0.62 for a clear outlier.
            "detected": best["_score"].to_numpy(dtype=float) >= 0.62,
        }
    )
    return BaselineResult("B1_IsolationForest", frame.reset_index(drop=True))


# ------------------------------------------------- B2: observed-only flow twin


class ObservedOnlyShadowSensor(ShadowSensor):
    """RippleTwin's flow model with hidden stations excluded as hypotheses.

    Everything else -- propagation matrices, likelihood, calibration, temporal
    smoothing -- is inherited unchanged. The single behavioural difference is
    that a station without a sensor can never be named. That makes this the
    honest stand-in for a conventional digital twin: one that models the part of
    the line it can measure and is silent about the rest.
    """

    def _hypothesis_scores(self, d_blocked, d_starved, z_proc, sigma_b, sigma_s):
        ll_st, ll_null, ll_line, a_b, a_s = super()._hypothesis_scores(
            d_blocked, d_starved, z_proc, sigma_b, sigma_s
        )
        ll_st = np.array(ll_st, dtype=float)
        ll_st[~self._obs_mask] = -np.inf
        return ll_st, ll_null, ll_line, a_b, a_s


def observed_only_twin(
    line: LineTopology,
    scored: pd.DataFrame,
    cfg: ShadowConfig,
    sigma_b: float,
    sigma_s: float,
) -> BaselineResult:
    """Run the flow twin restricted to instrumented stations."""
    import copy

    sensor = ObservedOnlyShadowSensor(line, copy.deepcopy(cfg))
    out = sensor.run(scored, sigma_b, sigma_s)
    if out.empty:
        return BaselineResult("B2_observed_only_twin", out)
    frame = out[["window", "top_station", "detected"]].copy()
    frame["score"] = out["llr"].to_numpy()
    return BaselineResult("B2_observed_only_twin", frame)


def rippletwin_result(shadow: pd.DataFrame) -> BaselineResult:
    """Wrap RippleTwin's own output in the common contract."""
    if shadow.empty:
        return BaselineResult("RippleTwin", shadow)
    frame = shadow[["window", "top_station", "detected"]].copy()
    frame["score"] = shadow["llr"].to_numpy()
    return BaselineResult("RippleTwin", frame)


# ------------------------------------------------------ matched-FPR comparison


def apply_detection_rule(
    frame: pd.DataFrame, threshold: float, persistence: int = 2
) -> pd.DataFrame:
    """Turn continuous scores into detections under one common rule.

    Every method -- RippleTwin included -- goes through this same function with
    its own threshold. Without that, the comparison is meaningless: a detector
    with a low threshold looks more sensitive simply because it fires more
    often, and its cost shows up in a false-alarm column nobody reads next to
    the detection column.

    The persistence requirement (the same candidate, within one station, for
    ``persistence`` consecutive windows) is applied identically to all methods,
    so no method gets credit for temporal smoothing another was denied.
    """
    if frame.empty:
        return frame.assign(detected=False)
    out = frame.sort_values("window").reset_index(drop=True).copy()
    above = out["score"].to_numpy(dtype=float) >= threshold
    top = out["top_station"].to_numpy(dtype=int)

    det = np.zeros(len(out), dtype=bool)
    run = 0
    for i in range(len(out)):
        if above[i] and (i > 0 and abs(int(top[i]) - int(top[i - 1])) <= 1):
            run += 1
        elif above[i]:
            run = 1
        else:
            run = 0
        det[i] = run >= persistence
    out["detected"] = det
    return out


def calibrate_threshold(nominal_frame: pd.DataFrame, target_window_fpr: float) -> float:
    """Threshold giving ``target_window_fpr`` on disturbance-free reference data.

    Read off the empirical null distribution of the method's own score, so each
    method is placed at the same operating point on its own scale.
    """
    if nominal_frame.empty:
        return float("inf")
    s = nominal_frame["score"].to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("inf")
    return float(np.quantile(s, 1.0 - target_window_fpr))


def build_methods(
    line: LineTopology,
    scored: pd.DataFrame,
    shadow: pd.DataFrame,
    nominal_scored: pd.DataFrame,
    cfg: ShadowConfig,
    sigma_b: float,
    sigma_s: float,
    if_models: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """Run every method on one episode and return raw (uncalibrated) scores."""
    out: Dict[str, pd.DataFrame] = {}
    out["RippleTwin"] = rippletwin_result(shadow).frame
    out["B2_observed_only_twin"] = observed_only_twin(
        line, scored, cfg, sigma_b, sigma_s
    ).frame
    out["B0_SPC_observed"] = spc_baseline(scored, line).frame
    out["B1_IsolationForest"] = isolation_forest_baseline(
        nominal_scored, scored, line
    ).frame
    return {k: v for k, v in out.items() if not v.empty}
