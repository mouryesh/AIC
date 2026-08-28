"""Metric definitions.

Every metric here is defined against ground truth the model never sees, and each
one is stated precisely enough to be argued with. Where a metric needs a
reference point -- "earlier than what?" -- the reference is an observable event
on the plant floor, not a model-internal quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..factory.topology import LineTopology


def production_board_moment(
    passes: pd.DataFrame,
    line: LineTopology,
    window_vehicles: int = 60,
    threshold: float = 0.90,
    after_t_s: Optional[float] = None,
    sustain_vehicles: int = 25,
    reference_rate_vph: Optional[float] = None,
) -> Optional[float]:
    """When a human would notice the line is under-producing.

    This is the reference point for lead time. It is deliberately generous to
    the status quo: the moment the *rolling actual output rate* falls below
    ``threshold`` of the takt rate, which is when a shortfall becomes visible on
    the production board with no analytics at all.

    Claiming warning lead time against event onset would be meaningless -- the
    twin cannot detect a disturbance before it starts. Claiming it against the
    moment the plant would otherwise have reacted is the number that matters.

    ``after_t_s`` restricts the search to the period after the disturbance
    began. Without it the function returns the *first* dip anywhere in the run,
    which on a long shift is usually an ordinary early-run fluctuation hours
    before the fault -- and the resulting "lead time" comes out negative and
    meaningless.

    ``reference_rate_vph`` is what the shortfall is measured against, and it
    must be the line's **own observed nominal output rate**, not the theoretical
    takt rate. This 42-station line sustains about 82% of takt under completely
    normal conditions -- losses to blocking, starving and micro-stops accumulate
    across a long line, which is ordinary and is why real plants track OEE
    rather than takt compliance. Measured against theoretical takt the reference
    fires permanently, on clean runs included, and the metric means nothing.
    Measured against what the line normally does, it means "output fell
    materially below plan and stayed there", which is what a supervisor reacts
    to. Falling back to takt rate is deliberately noted as a weaker reference.

    Returns simulation time in seconds, or None if output never dropped.
    """
    last = line.n_stations - 1
    end = passes[passes["station"] == last].sort_values("vehicle_id")
    if len(end) < window_vehicles + sustain_vehicles + 5:
        return None
    t = end["t_depart_s"].to_numpy()
    span = t[window_vehicles:] - t[:-window_vehicles]
    rate = window_vehicles / np.maximum(span, 1e-9)
    nominal = (
        reference_rate_vph / 3600.0
        if reference_rate_vph is not None
        else 1.0 / line.takt_s
    )
    t_at = t[window_vehicles:]

    below = rate < threshold * nominal
    if after_t_s is not None:
        below = below & (t_at >= after_t_s)

    # Require the shortfall to be SUSTAINED. A single dip in a short rolling
    # window happens constantly under normal variation -- model mix, a micro
    # stop, an operator break -- and nobody walks onto the floor because of one.
    # Without this the reference moment lands within seconds of the disturbance
    # starting, and the resulting "lead time" is negative and meaningless.
    if sustain_vehicles > 1:
        run = 0
        for i, b in enumerate(below):
            run = run + 1 if b else 0
            if run >= sustain_vehicles:
                return float(t_at[i])
        return None

    idx = np.where(below)[0]
    return float(t_at[idx[0]]) if idx.size else None


@dataclass
class EpisodeTruth:
    """Ground truth for one episode, assembled for evaluation only."""

    has_fault: bool
    station: Optional[int]
    kind: Optional[str]
    t_start_s: Optional[float]
    t_end_s: Optional[float]
    ramp_s: float
    magnitude: float
    source_is_hidden: bool
    board_moment_s: Optional[float]


def episode_truth(
    res,
    line: LineTopology,
    view_line: Optional[LineTopology] = None,
    reference_rate_vph: Optional[float] = None,
) -> EpisodeTruth:
    """Extract ground truth from a simulated run.

    ``view_line`` decides whether the true source counts as hidden, because
    hiddenness is a property of the observability view, not of the physics.
    """
    view = view_line or line
    d = res.disturbances
    real = d[d["kind"] != "MATERIAL_DELAY"] if len(d) else d
    if len(real) == 0:
        return EpisodeTruth(False, None, None, None, None, 0.0, 0.0, False, None)
    r = real.iloc[0]
    st = int(r["station"])
    # The board moment is only meaningful once the disturbance is under way.
    board = production_board_moment(
        res.passes, line, after_t_s=float(r["t_start_s"]),
        reference_rate_vph=reference_rate_vph,
    )
    return EpisodeTruth(
        has_fault=True,
        station=st,
        kind=str(r["kind"]),
        t_start_s=float(r["t_start_s"]),
        t_end_s=float(r["t_end_s"]),
        ramp_s=float(r["ramp_s"]),
        magnitude=float(r["magnitude"]),
        source_is_hidden=bool(view.stations[st].is_hidden),
        board_moment_s=board,
    )


def evaluate_localization(
    pred: pd.DataFrame,
    window_times: pd.DataFrame,
    truth: EpisodeTruth,
    settle_fraction: float = 1.0,
) -> dict:
    """Score one method on one faulted episode.

    Windows are restricted to the period after the disturbance has fully ramped
    in. Scoring a detector during the ramp would reward and punish essentially
    at random, because for part of that period the disturbance is genuinely too
    small to see.
    """
    if not truth.has_fault or pred.empty:
        return {}

    p = pred.merge(window_times, on="window", how="left")
    t0 = truth.t_start_s + settle_fraction * truth.ramp_s
    during = p[(p["t_mid_s"] >= t0) & (p["t_mid_s"] <= truth.t_end_s)]
    if during.empty:
        return {}

    det = during[during["detected"]]
    n_during = len(during)
    out = {
        "n_windows_during": n_during,
        "detection_rate": len(det) / n_during if n_during else np.nan,
        # Episode-level: did the method ever catch this fault at all? This is
        # the question a plant actually asks. Window-level rate answers a
        # different one -- how continuously it held the alert -- and averaging
        # only that understates a method that fires once, correctly, and then
        # relaxes as the buffers re-equilibrate.
        "detected_episode": float(len(det) > 0),
    }
    if det.empty:
        out.update(
            {"top1": np.nan, "within1": np.nan, "median_station_error": np.nan,
             "lead_time_min": np.nan, "top1_episode": 0.0, "within1_episode": 0.0}
        )
        return out

    err = np.abs(det["top_station"].to_numpy() - truth.station)
    out["top1"] = float((err == 0).mean())
    out["within1"] = float((err <= 1).mean())
    out["median_station_error"] = float(np.median(err))
    # Episode-level localisation: was the modal named station correct? A
    # supervisor is dispatched once, to the station the system kept naming.
    modal = int(pd.Series(det["top_station"]).mode().iloc[0])
    out["top1_episode"] = float(modal == truth.station)
    out["within1_episode"] = float(abs(modal - truth.station) <= 1)

    # Lead time against the production-board moment.
    #
    # The alert must be CORRECTLY LOCALISED to count. Timing any alert, right or
    # wrong, rewards whichever detector is noisiest: a method that fires
    # constantly will always fire early, and on a first pass of this metric the
    # topology-blind anomaly baseline scored the *longest* lead time precisely
    # because it was wrong more often. A warning that names the wrong station
    # does not buy a supervisor any time, so it earns no credit here.
    #
    # The window is taken from disturbance ONSET rather than from the end of the
    # ramp: warning while the fault is still growing is exactly the value being
    # claimed, and scoring only after the ramp would discard the most useful
    # detections.
    from_onset = p[
        (p["t_mid_s"] >= truth.t_start_s)
        & (p["t_mid_s"] <= truth.t_end_s)
        & p["detected"]
    ]
    correct = from_onset[
        np.abs(from_onset["top_station"].to_numpy() - truth.station) <= 1
    ]
    if len(correct) and truth.board_moment_s is not None:
        first_correct_t = float(correct["t_mid_s"].min())
        out["lead_time_min"] = (truth.board_moment_s - first_correct_t) / 60.0
        out["first_correct_alert_t_s"] = first_correct_t
    else:
        # No correctly-localised alert means no lead time was bought at all.
        out["lead_time_min"] = np.nan
    out["first_alert_t_s"] = (
        float(from_onset["t_mid_s"].min()) if len(from_onset) else np.nan
    )
    return out


def evaluate_false_alarms(pred: pd.DataFrame, truth: EpisodeTruth,
                          window_times: pd.DataFrame) -> dict:
    """False-alarm rate on the parts of an episode with no active disturbance."""
    if pred.empty:
        return {}
    p = pred.merge(window_times, on="window", how="left")
    if truth.has_fault:
        quiet = p[(p["t_mid_s"] < truth.t_start_s) | (p["t_mid_s"] > truth.t_end_s)]
    else:
        quiet = p
    if quiet.empty:
        return {}
    return {
        "n_quiet_windows": len(quiet),
        "false_alarm_rate": float(quiet["detected"].mean()),
    }


def summarise(rows: List[dict], by: List[str]) -> pd.DataFrame:
    """Aggregate per-episode metrics, reporting means and counts."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    num = [
        c
        for c in df.columns
        if c not in by and pd.api.types.is_numeric_dtype(df[c])
    ]
    g = df.groupby(by, dropna=False)[num].mean().reset_index()
    g["n_episodes"] = df.groupby(by, dropna=False).size().to_numpy()
    return g


def true_bottleneck_onset(line: LineTopology, truth: "EpisodeTruth") -> Optional[float]:
    """The simulation time at which the injected disturbance's ground-truth
    processing time first exceeds takt -- i.e. the moment the constraint
    genuinely starts limiting output, from the ``Disturbance`` object itself
    rather than from an observed shortfall.

    This is a tighter, model-internal reference than
    ``production_board_moment`` (which needs a *sustained* line-level
    shortfall to fire, and often never fires at all -- see RESULTS.md §4).
    It exists specifically to evaluate a predictive layer that claims to warn
    *before* a constraint binds: "before what, precisely" needs an answer
    that does not itself depend on the detector being evaluated.

    Only defined for SLOWDOWN/COMBINED disturbances, which act on processing
    time. Uses the station's nominal ``base_cycle_s`` as the deterministic
    reference (per-vehicle noise averages out around it) -- the same
    quantity ``RippleForecast.is_binding`` compares an *estimate* of against
    takt at evaluation time.
    """
    if not truth.has_fault or truth.kind not in ("SLOWDOWN", "COMBINED"):
        return None
    if truth.station is None or truth.magnitude <= 1.0:
        return None
    base_cycle = float(line.stations[truth.station].base_cycle_s)
    takt = float(line.takt_s)
    health_needed = takt / base_cycle
    if health_needed <= 1.0:
        # Already at or above takt at zero disturbance intensity: binds
        # immediately once the disturbance starts.
        return float(truth.t_start_s)
    intensity_needed = (health_needed - 1.0) / (truth.magnitude - 1.0)
    if intensity_needed > 1.0:
        return None  # magnitude too weak to ever bind, even fully ramped in
    intensity_needed = max(0.0, intensity_needed)
    return float(truth.t_start_s + intensity_needed * truth.ramp_s)


def evaluate_early_warning(
    pred: pd.DataFrame,
    truth: "EpisodeTruth",
    true_onset_s: Optional[float],
) -> dict:
    """Score the predictive layer (``twin.predict``) on one episode.

    ``pred`` is the output of ``twin.predict.run_predictor``: one row per
    window with ``state``, ``t_mid_s`` and ``station``. A prediction only
    counts if it is *correctly localised* (within 1 station of the true
    source), for the same reason ``evaluate_localization`` requires it: an
    early warning naming the wrong station has not bought anyone real lead
    time.

    Lead time is reported against ``true_onset_s`` (see
    ``true_bottleneck_onset``) when available. Missed and false-alarm cases
    are reported explicitly, not folded into an average that would hide them.
    """
    from .metrics import EpisodeTruth  # local import avoids a cycle at module load

    if pred.empty:
        return {}

    elevated = pred[pred["state"].isin(("WATCH", "PREDICTED_CONSTRAINT", "ACTIVE_BOTTLENECK"))]

    if not truth.has_fault:
        # No fault this episode: any elevated state anywhere is a false alarm.
        return {
            "n_windows": int(len(pred)),
            "false_alarm": float(len(elevated) > 0),
            "n_false_alarm_windows": int(len(elevated)),
        }

    # Diagnostic only, never the headline: the first elevated state at ANY
    # station, not required to name the right one. Comparing this against
    # the correctly-localised lead time below separates two different
    # claims -- "the twin senses something is changing" vs. "the twin can
    # already say where" -- rather than letting one silently stand in for
    # the other.
    lo = truth.t_start_s - 3600.0
    hi = truth.t_end_s
    elevated_in_window = elevated[(elevated["t_mid_s"] >= lo) & (elevated["t_mid_s"] <= hi)]
    first_any_t = float(elevated_in_window["t_mid_s"].min()) if len(elevated_in_window) else None
    lead_time_any_station_min = (
        (true_onset_s - first_any_t) / 60.0 if (true_onset_s is not None and first_any_t is not None) else np.nan
    )

    correctly_localised = elevated[
        elevated["station"].notna()
        & (np.abs(elevated["station"].astype(float) - truth.station) <= 1)
    ]
    # Restrict to the episode's own fault window (plus a little slack before
    # onset for the pre-alarm to count) so a WATCH raised during an unrelated
    # part of a long run isn't credited to this fault.
    if len(correctly_localised):
        correctly_localised = correctly_localised[
            (correctly_localised["t_mid_s"] >= lo) & (correctly_localised["t_mid_s"] <= hi)
        ]

    if correctly_localised.empty:
        return {
            "n_windows": int(len(pred)),
            "missed": 1.0,
            "false_alarm": 0.0,
            "lead_time_min": np.nan,
            "lead_time_any_station_min": lead_time_any_station_min,
        }

    first_t = float(correctly_localised["t_mid_s"].min())
    out = {
        "n_windows": int(len(pred)),
        "missed": 0.0,
        "false_alarm": 0.0,
        "first_warning_t_s": first_t,
        "lead_time_any_station_min": lead_time_any_station_min,
    }
    if true_onset_s is not None:
        out["lead_time_min"] = (true_onset_s - first_t) / 60.0
        out["true_onset_s"] = true_onset_s
    else:
        out["lead_time_min"] = np.nan
    return out


def cycle_time_error(
    inferred_s: Optional[float], passes: pd.DataFrame, station: int,
    v_start: int, v_end: int,
) -> Optional[dict]:
    """Error of the inferred cycle time for a station with no sensor.

    This is the sharpest falsifiable claim RippleTwin makes: it estimates a
    number that, in the view given to the model, is unmeasurable. Ground truth
    exists in the simulator, so the estimate is checked rather than asserted.
    """
    if inferred_s is None:
        return None
    seg = passes[
        (passes["station"] == station)
        & (passes["vehicle_id"] >= v_start)
        & (passes["vehicle_id"] < v_end)
    ]
    if seg.empty:
        return None
    true_proc = float(seg["proc_time_s"].mean())
    if true_proc <= 0:
        return None
    return {
        "inferred_cycle_s": float(inferred_s),
        "true_cycle_s": true_proc,
        "abs_error_s": abs(float(inferred_s) - true_proc),
        "abs_error_pct": 100.0 * abs(float(inferred_s) - true_proc) / true_proc,
    }
