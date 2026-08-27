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
