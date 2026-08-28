"""Where to put the next sensor.

The obvious challenge to this whole project is "why not just instrument the
blind stations?" The honest answer is that you should — but a plant can only
retrofit during a few maintenance windows a year, so the question is not
*whether* to add sensors, it is **which ones buy the most**.

That question has an answer that falls directly out of the model, and it turns
the objection into a feature: RippleTwin is not competing with the
instrumentation budget, it is telling you how to spend it.

The idea
--------
A blind station is locatable only if its predicted flow signature is
distinguishable, using the stations we *can* see, from the signatures of its
neighbours. Two blind stations side by side with no sensor between them produce
almost the same pattern at every observing station, so no amount of data will
separate them -- that is a structural fact about the sensor layout, not a
shortcoming of the estimator, and it is knowable in advance.

So we measure, for each blind station, how similar its response pattern is to
its best-matching rival, restricted to the stations that are actually observed.
That is its **ambiguity**. Then, for each candidate sensor position, we ask how
much total ambiguity would fall if a sensor were placed there.

This is a value-of-information calculation over the sensor layout, and it is
cheap: it needs only the propagation matrices and the observed set. No
production data is required to run it, which means a plant can run it *before*
committing to a retrofit.

Validation
----------
The metric is only worth anything if the stations it calls ambiguous are the
ones the twin actually gets wrong. ``validate_against_outcomes`` checks exactly
that against measured localisation error, rather than asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..factory.topology import LineTopology
from .shadow import ShadowConfig, propagation_matrices


def _response_matrix(
    line: LineTopology,
    cfg: ShadowConfig,
    observed: Sequence[int],
    sigma_b: float,
    sigma_s: float,
) -> np.ndarray:
    """Predicted response of every candidate, as seen by the observed stations.

    Row ``k`` is what station ``k`` being the constraint would look like across
    the observing stations, with the two channels stacked and each scaled by its
    own noise level -- so similarity is measured in units of detectability
    rather than raw seconds.
    """
    B, S = propagation_matrices(line, cfg)
    obs = np.asarray(sorted(observed), dtype=int)
    if obs.size == 0:
        return np.zeros((line.n_stations, 0))
    return np.hstack([B[:, obs] / max(sigma_b, 1e-9), S[:, obs] / max(sigma_s, 1e-9)])


def ambiguity(
    line: LineTopology,
    observed: Sequence[int],
    cfg: ShadowConfig | None = None,
    sigma_b: float = 0.03,
    sigma_s: float = 0.05,
) -> pd.DataFrame:
    """How separable each station is from its best-matching rival.

    Two quantities are returned, and the distinction between them matters.

    ``ambiguity`` is the cosine similarity to the nearest-looking alternative.
    It is the intuitive, interpretable number -- "these two stations look 97%
    alike" -- and it is what the UI shows.

    ``separability`` is the quantity the placement decision actually uses: the
    magnitude of the residual left when the best-matching rival is fitted to
    this station's response, expressed in units of measurement noise. For
    candidates ``h`` and ``j`` with response vectors ``P``::

        separability(h) = min_j  ||P_h - proj_{P_j}(P_h)||
                        = min_j  |P_h| * sqrt(1 - cos^2(P_h, P_j))

    Why not just use cosine
    -----------------------
    Because cosine is **not monotone** under adding sensors, and a placement
    tool built on it will occasionally tell you that installing a sensor made
    the line harder to reason about. Adding an observing station that responds
    similarly to two rival hypotheses adds a large common-mode component to both
    response vectors, which pulls their angle together even though no
    information was lost.

    The residual magnitude does not have that defect. Writing ``c = P_h·P_j``,
    ``D = |P_j|^2``, and adding one observer contributing ``a`` and ``b`` to the
    two vectors, the change in squared residual reduces to ``(aD - cb)^2 >= 0``.
    So separability is provably non-decreasing when a sensor is added, which is
    the behaviour any honest value-of-information metric has to have. This was
    caught by ``test_instrumenting_a_station_reduces_ambiguity``.
    """
    cfg = cfg or ShadowConfig()
    R = _response_matrix(line, cfg, observed, sigma_b, sigma_s)
    norms = np.linalg.norm(R, axis=1)
    safe = np.where(norms > 1e-12, norms, 1.0)
    Rn = R / safe[:, None]

    C = Rn @ Rn.T
    np.fill_diagonal(C, -np.inf)
    # A station with no visible response at all cannot be distinguished from
    # anything: nothing about it reaches a sensor.
    invisible = norms <= 1e-12
    C[invisible, :] = 1.0
    C[:, invisible] = -np.inf

    best_cos = np.max(C, axis=1)
    best_cos = np.clip(np.where(np.isfinite(best_cos), best_cos, 1.0), 0.0, 1.0)
    partner = np.argmax(np.where(np.isfinite(C), C, -np.inf), axis=1)

    # Residual separation against EVERY rival, then take the closest one.
    sin2 = np.clip(1.0 - C**2, 0.0, 1.0)
    sep_all = norms[:, None] * np.sqrt(sin2)
    sep_all[~np.isfinite(C)] = np.inf
    separability = np.min(sep_all, axis=1)
    separability = np.where(invisible, 0.0, separability)

    obs_set = set(int(i) for i in observed)
    rows = []
    for s in line.stations:
        rows.append(
            {
                "station": s.index,
                "station_id": s.station_id,
                "zone": s.zone,
                "tier": s.tier,
                "is_hidden": s.index not in obs_set,
                "ambiguity": float(best_cos[s.index]),
                "resolvability": float(1.0 - best_cos[s.index]),
                "separability": float(separability[s.index]),
                "confusable_with": line.stations[int(partner[s.index])].station_id,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class PlacementRecommendation:
    """One candidate sensor position and what it would buy."""

    station: int
    station_id: str
    zone: str
    #: Total ambiguity removed across all still-blind stations.
    total_gain: float
    #: The blind stations this sensor would most help disambiguate.
    unlocks: List[str]
    #: Ambiguity of this station itself before instrumenting it.
    own_ambiguity_before: float
    rank: int = 0


def recommend_sensors(
    line: LineTopology,
    observed: Optional[Sequence[int]] = None,
    n_recommend: int = 5,
    cfg: ShadowConfig | None = None,
    sigma_b: float = 0.03,
    sigma_s: float = 0.05,
    suspicion: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Rank blind stations by how much instrumenting each would buy.

    ``suspicion`` optionally weights each blind station by how often the twin
    has actually suspected it, so the recommendation reflects where problems
    really occur on this line rather than treating every station as equally
    likely to fail. Without it the ranking is purely structural, which is what
    a plant can compute before it has any history.

    This is a greedy single-sensor ranking, not a jointly optimal set. Greedy is
    the right shape for the decision it supports: retrofits happen a few at a
    time, in whichever maintenance window comes next.
    """
    cfg = cfg or ShadowConfig()
    observed = list(observed if observed is not None else line.observed_indices)
    obs_set = set(int(i) for i in observed)
    blind = [s.index for s in line.stations if s.index not in obs_set]
    if not blind:
        return pd.DataFrame(
            columns=["station", "station_id", "zone", "total_gain", "unlocks",
                     "own_ambiguity_before", "rank"]
        )

    base = ambiguity(line, observed, cfg, sigma_b, sigma_s).set_index("station")
    w = suspicion or {}

    # Normaliser so gains are comparable across lines with different noise
    # scales; the ranking itself is unaffected.
    scale = float(np.median(base["separability"].replace(0.0, np.nan).dropna()) or 1.0)

    rows = []
    for c in blind:
        after = ambiguity(
            line, sorted(obs_set | {c}), cfg, sigma_b, sigma_s
        ).set_index("station")
        gains = {}
        for h in blind:
            if h == c:
                continue
            # Gain in SEPARABILITY, which is provably non-decreasing when a
            # sensor is added. Using cosine here would let a new sensor score a
            # negative gain, which is not a thing that can happen.
            g = float(after.loc[h, "separability"] - base.loc[h, "separability"])
            gains[h] = max(0.0, g / max(scale, 1e-9)) * float(w.get(h, 1.0))

        # Instrumenting a station also removes it from the inference problem
        # entirely: it stops being something we have to infer at all. That is
        # worth most where the station was least separable to begin with.
        own = float(base.loc[c, "ambiguity"]) * float(w.get(c, 1.0))
        total = own + sum(gains.values())

        helped = sorted(gains.items(), key=lambda kv: -kv[1])[:3]
        rows.append(
            {
                "station": c,
                "station_id": line.stations[c].station_id,
                "zone": line.stations[c].zone,
                "total_gain": total,
                "unlocks": ", ".join(
                    line.stations[h].station_id for h, g in helped if g > 1e-6
                ) or "(itself only)",
                "own_ambiguity_before": float(base.loc[c, "ambiguity"]),
            }
        )

    out = pd.DataFrame(rows).sort_values("total_gain", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(n_recommend).reset_index(drop=True)


def suspicion_from_shadow(shadow_frames: Sequence[pd.DataFrame]) -> Dict[int, float]:
    """How often each station has been the twin's leading suspect.

    Used to weight the placement ranking toward stations that actually cause
    trouble on this line, rather than treating all blind stations alike.
    """
    counts: Dict[int, float] = {}
    for f in shadow_frames:
        if f is None or f.empty or "detected" not in f.columns:
            continue
        d = f[f["detected"]]
        for st, n in d["top_station"].value_counts().items():
            counts[int(st)] = counts.get(int(st), 0.0) + float(n)
    if not counts:
        return {}
    total = sum(counts.values())
    # Keep a floor so a station that has never been suspected is not treated as
    # worthless to instrument -- absence of evidence, on a blind station, is
    # exactly what we cannot interpret.
    return {k: 0.25 + 0.75 * (v / total) for k, v in counts.items()}


def validate_against_outcomes(
    line: LineTopology,
    localization_raw: pd.DataFrame,
    coverage: float,
    observed: Sequence[int],
    cfg: ShadowConfig | None = None,
) -> dict:
    """Do the stations flagged ambiguous actually get localised worse?

    The placement metric is structural, so it is worth nothing unless it
    predicts real error. This compares the predicted ambiguity of each true
    source station against RippleTwin's measured localisation error on it.
    """
    amb = ambiguity(line, observed, cfg).set_index("station")["ambiguity"]

    df = localization_raw[
        (localization_raw["method"] == "RippleTwin")
        & (localization_raw["split"] == "test")
        & (np.isclose(localization_raw["coverage"], coverage))
    ]
    if df.empty or "true_station" not in df.columns:
        return {"n": 0, "note": "no per-episode true-station column available"}

    g = df.groupby("true_station").agg(
        err=("median_station_error", "mean"), n=("seed", "size")
    ).reset_index()
    g["ambiguity"] = g["true_station"].map(amb)
    g = g.dropna(subset=["ambiguity", "err"])
    if len(g) < 4:
        return {"n": int(len(g)), "note": "too few distinct source stations"}

    r = float(np.corrcoef(g["ambiguity"], g["err"])[0, 1])
    return {
        "n_stations": int(len(g)),
        "corr_ambiguity_vs_error": r,
        "interpretation": (
            "positive correlation means the structural metric predicts where the "
            "twin actually struggles"
        ),
    }
