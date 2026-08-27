"""Shadow-sensing for quality: attributing defects back to unmeasured stations.

Why a second mechanism is needed
--------------------------------
The flow model in ``shadow.py`` localises a station by the blocking and
starvation it induces. That works when a station gets *slower*. It is blind to a
station that keeps perfect takt while quietly producing bad work -- a drifting
fixture, a mis-set torque limit, a worn sealer nozzle. Nothing about the timing
of the line changes, so there is nothing for the flow model to see.

Measured on this line, the flow model detects a pure quality drift in 0% of
windows. That is the correct result for that mechanism, and it is why the twin
carries a second path rather than one over-stretched model.

The mechanism
-------------
Three pieces of structure make attribution tractable without a sensor:

1. **Vehicle genealogy.** Every vehicle passes every station in a known order.
   Where a station is instrumented we know exactly when a vehicle passed it;
   where it is not, we interpolate between the nearest observed stations either
   side. That gives a complete "which vehicle was at station k, when" table.

2. **Failure-mode propensity.** A plant knows from process FMEA which stations
   can physically produce which defect. A sealer station cannot cause a torque
   fault. This collapses the candidate set for a given defect from "everything
   upstream" to a handful of stations.

3. **Detection lag.** A defect surfaces at a gate some distance downstream of
   where it was made. Back-projecting through the genealogy puts the defect at a
   specific *time window* at each candidate station -- and a station that is
   actually drifting shows elevated attributed defects concentrated in the
   windows when it was drifting, not spread uniformly.

Attribution is Bayesian and produces a distribution, not a verdict. When the
evidence cannot separate two adjacent candidates, it says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..factory.topology import LineTopology


# ----------------------------------------------------------------- genealogy


def build_genealogy(
    line: LineTopology, telemetry: pd.DataFrame, vehicles: pd.DataFrame
) -> pd.DataFrame:
    """Estimate when each vehicle passed each station.

    Observed stations give exact departure times. Hidden stations are linearly
    interpolated between the nearest observed stations either side, which is
    accurate to well within one window because the line runs to takt.

    Returns a wide frame indexed by ``vehicle_id`` with one column per station.
    """
    piv = telemetry.pivot_table(
        index="vehicle_id", columns="station", values="t_depart_s", aggfunc="last"
    )
    n = line.n_stations
    for i in range(n):
        if i not in piv.columns:
            piv[i] = np.nan
    piv = piv[sorted(piv.columns)]

    # Interpolate across station index, then fill the ends from the release log.
    arr = piv.to_numpy(dtype=float)
    idx = np.arange(n, dtype=float)
    for r in range(arr.shape[0]):
        row = arr[r]
        good = np.isfinite(row)
        if good.sum() >= 2:
            arr[r] = np.interp(idx, idx[good], row[good])
        elif good.sum() == 1:
            arr[r] = np.where(np.isfinite(row), row, row[good][0])
    out = pd.DataFrame(arr, index=piv.index, columns=[int(c) for c in piv.columns])
    out.index.name = "vehicle_id"
    return out


def assign_windows(
    genealogy: pd.DataFrame, window_bounds: pd.DataFrame, station: int
) -> pd.Series:
    """Map each vehicle to the window in which it passed ``station``.

    ``window_bounds`` must carry ``window``, ``t_lo``, ``t_hi`` columns.
    """
    t = genealogy[station].to_numpy(dtype=float)
    lo = window_bounds["t_lo"].to_numpy()
    pos = np.searchsorted(lo, t, side="right") - 1
    pos = np.clip(pos, 0, len(window_bounds) - 1)
    return pd.Series(
        window_bounds["window"].to_numpy()[pos], index=genealogy.index, name="window"
    )


# --------------------------------------------------------------- attribution


def explode_defects(inspections: pd.DataFrame) -> pd.DataFrame:
    """One row per defect found, from the gate-level inspection log.

    This is the plant's own quality record. It knows the defect type and where
    it was *found*. It does not know where it was made -- that is what we infer.
    """
    if inspections.empty:
        return pd.DataFrame(
            columns=["vehicle_id", "gate_station", "gate_id", "defect_type", "t_found_s"]
        )
    fails = inspections[inspections["result"] == "FAIL"].copy()
    if fails.empty:
        return pd.DataFrame(
            columns=["vehicle_id", "gate_station", "gate_id", "defect_type", "t_found_s"]
        )
    fails["defect_type"] = fails["defect_types"].astype(str).str.split("|")
    ex = fails.explode("defect_type")
    ex = ex[ex["defect_type"].astype(str).str.len() > 0]
    return pd.DataFrame(
        {
            "vehicle_id": ex["vehicle_id"].to_numpy(),
            "gate_station": ex["station"].to_numpy(),
            "gate_id": ex["gate_id"].to_numpy(),
            "defect_type": ex["defect_type"].to_numpy(),
            "t_found_s": ex["t_s"].to_numpy(),
        }
    )


def candidate_prior(line: LineTopology) -> pd.DataFrame:
    """P(source station | defect type), from failure-mode propensity and base rate.

    This is the structural prior. It uses only process knowledge the plant
    already has: which station can make which defect, and how defect-prone each
    station is at nominal. No production data is involved.
    """
    rows = []
    for s in line.stations:
        for dtype, w in (s.defect_profile or {}).items():
            if w <= 0:
                continue
            rows.append(
                {
                    "station": s.index,
                    "station_id": s.station_id,
                    "zone": s.zone,
                    "defect_type": dtype,
                    "weight": float(w) * float(s.base_defect_rate),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["prior"] = df.groupby("defect_type")["weight"].transform(lambda x: x / x.sum())
    return df


def attribute_defects(
    line: LineTopology, defects_found: pd.DataFrame, prior: pd.DataFrame
) -> pd.DataFrame:
    """Distribute each found defect over its candidate source stations.

    A defect is only attributable to a station that is upstream of the gate that
    found it and that can physically produce that failure mode. The result is a
    soft assignment: each defect contributes fractional mass to several
    candidates, summing to one.
    """
    if defects_found.empty or prior.empty:
        return pd.DataFrame(
            columns=["vehicle_id", "station", "defect_type", "mass", "gate_station"]
        )

    merged = defects_found.merge(prior, on="defect_type", how="inner")
    # Causality: the defect must have been made before it was found.
    merged = merged[merged["station"] < merged["gate_station"]]
    if merged.empty:
        return pd.DataFrame(
            columns=["vehicle_id", "station", "defect_type", "mass", "gate_station"]
        )

    key = ["vehicle_id", "gate_station", "defect_type"]
    merged["mass"] = merged["prior"] / merged.groupby(key)["prior"].transform("sum")
    return merged[key + ["station", "station_id", "zone", "mass"]]




# ------------------------------------------------------- quality state (Poisson)


@dataclass
class QualityBaseline:
    """Nominal *observed* defect rate and failure-mode shape per station.

    ``lam`` is the expected number of defects per vehicle processed that will be
    caught at some downstream gate and are attributable to this station. It is
    estimated from a disturbance-free run, so it already absorbs the fact that
    gates are imperfect and that stations near the end of the line have fewer
    gates left to catch their mistakes.

    ``profile`` is the failure-mode shape from process FMEA. It is structural,
    not learned, and it is the discriminative part of the signal.
    """

    lam: Dict[int, float]
    profile: Dict[int, Dict[str, float]]
    types: List[str]

    @classmethod
    def fit(
        cls,
        line: LineTopology,
        attribution: pd.DataFrame,
        n_vehicles: int,
    ) -> "QualityBaseline":
        mass = (
            attribution.groupby("station")["mass"].sum()
            if not attribution.empty
            else pd.Series(dtype=float)
        )
        lam, prof = {}, {}
        types: List[str] = []
        for s in line.stations:
            m = float(mass.get(s.index, 0.0))
            # Shrinkage so a station that happened to produce nothing during
            # baseline does not later show an infinite rate ratio.
            lam[s.index] = (m + 0.5) / (n_vehicles + 1.0)
            prof[s.index] = dict(s.defect_profile or {})
            for t in prof[s.index]:
                if t not in types:
                    types.append(t)
        return cls(lam=lam, profile=prof, types=sorted(types))

    def expected_matrix(self, line: LineTopology) -> np.ndarray:
        """``M[k, t]`` -- expected defects of type t per vehicle, from station k."""
        M = np.zeros((line.n_stations, len(self.types)))
        t_index = {t: j for j, t in enumerate(self.types)}
        for k in range(line.n_stations):
            for t, w in self.profile.get(k, {}).items():
                M[k, t_index[t]] = self.lam[k] * w
        return M


def quality_state(
    line: LineTopology,
    defects_found: pd.DataFrame,
    window_bounds: pd.DataFrame,
    baseline: QualityBaseline,
    pool_vehicles: int = 200,
) -> pd.DataFrame:
    """Inferred quality state per station per window, via a Poisson mixture LLR.

    Why a mixture rather than soft assignment
    -----------------------------------------
    Splitting each defect across its candidate stations in proportion to a prior
    and then testing each station's total is weak: an eleven-fold drift at one
    station gets diluted across every station that shares its failure mode, and
    the true source ranked 11th out of 42 in testing.

    The discriminative signal is not the count, it is the **shape of the
    failure-mode histogram**. So we model the pooled defect-type counts as a sum
    of per-station Poisson contributions,

        E[O_t]  =  N * sum_k  lambda_k * p_k(t)

    and for each candidate k fit a single multiplier ``m_k >= 1`` on that
    station's contribution alone, scoring it by Poisson log-likelihood ratio
    against ``m_k = 1``. A station whose characteristic failure mode is the one
    actually over-represented gets a high ratio; a station that merely shares the
    zone does not.

    Alignment is in vehicle-index space: a defect found at the end of the line is
    counted against the pool containing the vehicle that carries it, which is
    exactly the pool that was passing through the source station when the defect
    was made. No time interpolation is involved, so hidden stations are handled
    on identical footing to instrumented ones.

    ``pool_vehicles`` sets how many vehicles are pooled before testing. Defects
    are two orders of magnitude rarer than flow deviations, so quality alerts
    need more material before they can say anything. That is a real limitation
    and it is why quality warnings arrive later than bottleneck warnings.
    """
    types = baseline.types
    t_index = {t: j for j, t in enumerate(types)}
    M = baseline.expected_matrix(line)          # (n_stations, n_types)
    n_types = len(types)

    if defects_found.empty:
        dv = np.array([], dtype=int)
        dt = np.array([], dtype=int)
    else:
        keep = defects_found["defect_type"].isin(t_index)
        dv = defects_found.loc[keep, "vehicle_id"].to_numpy(dtype=int)
        dt = np.array(
            [t_index[t] for t in defects_found.loc[keep, "defect_type"]], dtype=int
        )

    # Multiplier grid: dense where drifts usually sit, sparse in the tail.
    grid = np.concatenate([np.linspace(1.0, 6.0, 41), np.linspace(6.5, 30.0, 30)])

    rows = []
    for _, wb in window_bounds.iterrows():
        v_hi = int(wb["v_end"])
        v_lo = max(0, v_hi - pool_vehicles)
        n_veh = v_hi - v_lo
        if n_veh < 20:
            continue

        sel = (dv >= v_lo) & (dv < v_hi)
        O = np.bincount(dt[sel], minlength=n_types).astype(float)
        E0 = n_veh * M.sum(axis=0)                      # baseline expectation
        E0 = np.maximum(E0, 1e-9)

        # Profile the Poisson log-likelihood over the multiplier m >= 1, for
        # every station at once. One free parameter on a well-behaved 1-D
        # surface, so a grid beats depending on an optimiser -- and vectorising
        # over stations matters because this runs for every window of every
        # episode at every coverage level.
        contrib = n_veh * M                              # (n_stations, n_types)
        # E[station, m, type]
        E = E0[None, None, :] + (grid[None, :, None] - 1.0) * contrib[:, None, :]
        E = np.maximum(E, 1e-9)
        ll = (O[None, None, :] * np.log(E) - E).sum(axis=2)   # (n_stations, n_grid)
        best = np.argmax(ll, axis=1)
        ll_best = ll[np.arange(line.n_stations), best]
        ll0 = float((O * np.log(E0) - E0).sum())
        llr = np.maximum(0.0, ll_best - ll0)
        m_hat = grid[best]

        # A station that cannot produce any defect carries no evidence either way.
        inert = contrib.sum(axis=1) < 1e-6
        llr = np.where(inert, 0.0, llr)
        m_hat = np.where(inert, 1.0, m_hat)

        rows.append(
            pd.DataFrame(
                {
                    "window": int(wb["window"]),
                    "station": np.arange(line.n_stations),
                    "llr": llr,
                    "m_hat": m_hat,
                    "observed": float(O.sum()),
                    "expected": float(E0.sum()),
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=["window", "station", "station_id", "zone", "tier", "is_hidden",
                     "llr", "m_hat", "observed", "expected", "quality_score"]
        )

    out = pd.concat(rows, ignore_index=True)
    meta = {s.index: s for s in line.stations}
    out["station_id"] = out["station"].map(lambda k: meta[k].station_id)
    out["zone"] = out["station"].map(lambda k: meta[k].zone)
    out["tier"] = out["station"].map(lambda k: meta[k].tier)
    out["is_hidden"] = out["station"].map(lambda k: meta[k].is_hidden)
    out["quality_score"] = out["llr"]
    return out


def quality_alerts(
    qstate: pd.DataFrame,
    llr_threshold: float = 6.0,
    min_multiplier: float = 1.8,
    persistence: int = 3,
) -> pd.DataFrame:
    """Flag stations whose failure-mode signature is persistently over-represented.

    Three gates, each earning its place:

    * ``llr_threshold`` -- the evidence must beat the no-drift model by a
      meaningful margin.
    * ``min_multiplier`` -- a statistically detectable 15% rate rise is not worth
      pulling a supervisor off the floor for.
    * ``persistence`` -- a drifting fixture stays elevated pool after pool;
      a run of bad luck does not.
    """
    if qstate.empty:
        return qstate.assign(quality_alert=False)
    out = qstate.sort_values(["station", "window"]).copy()
    flag = (out["llr"] >= llr_threshold) & (out["m_hat"] >= min_multiplier)
    out["_f"] = flag.astype(float)
    roll = (
        out.groupby("station")["_f"]
        .rolling(persistence, min_periods=persistence)
        .sum()
        .reset_index(level=0, drop=True)
    )
    out["quality_alert"] = (roll >= persistence).fillna(False)
    return out.drop(columns=["_f"])


def window_bounds_from(scored_windows: pd.DataFrame) -> pd.DataFrame:
    """Extract (window, v_start, v_end, t_lo, t_hi) bounds from scored windows."""
    g = (
        scored_windows.groupby("window")
        .agg(
            t_lo=("t_depart_s_min", "min"),
            t_hi=("t_depart_s_max", "max"),
            v_start=("v_start", "first"),
            v_end=("v_end", "first"),
        )
        .reset_index()
    )
    return g.sort_values("window").reset_index(drop=True)
