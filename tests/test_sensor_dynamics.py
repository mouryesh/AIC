"""Tests for dynamic sensor failure (factory.sensor_health), the
critical-vs-random coverage strategy (factory.topology.apply_coverage), and
the ALLOW/WATCH/INVESTIGATE/ESCALATE/ABSTAIN taxonomy read-out
(recommend.engine.taxonomy_label).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rippletwin.factory import scenarios as SC
from rippletwin.factory.sensor_health import (
    DROPOUT,
    INTERMITTENT,
    NOISY,
    STALE,
    SensorFault,
    apply_sensor_faults,
    flag_stale_windows,
    mask_suspect_rows,
    summarize_data_quality,
)
from rippletwin.factory.topology import apply_coverage, build_line
from rippletwin.recommend.engine import (
    ACTION_ESCALATE,
    ACTION_INSPECT,
    ACTION_MONITOR,
    PRIORITY_LOW,
    Recommendation,
    TAXONOMY_ABSTAIN,
    TAXONOMY_ESCALATE,
    TAXONOMY_INVESTIGATE,
    TAXONOMY_WATCH,
    taxonomy_label,
)
from rippletwin.twin.pipeline import build_windows, fit_context, infer, simulate

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


def _telemetry(line, n_vehicles=400, seed=1):
    res = simulate(line, SC.nominal_run(n_vehicles), seed=seed)
    return res.telemetry, res


# ------------------------------------------------------------------- dropout


def test_dropout_removes_rows_only_in_interval_and_station(line):
    tel, _ = _telemetry(line)
    st = tel["station"].iloc[0]
    lo, hi = tel["t_start_s"].quantile([0.3, 0.6])
    fault = SensorFault(station=int(st), kind=DROPOUT, t_start_s=float(lo), t_end_s=float(hi))
    out = apply_sensor_faults(tel, [fault], seed=0)

    still_there = out[(out["station"] == st) & (out["t_start_s"] >= lo) & (out["t_start_s"] < hi)]
    assert len(still_there) == 0
    other_station_rows_before = len(tel[tel["station"] != st])
    other_station_rows_after = len(out[out["station"] != st])
    assert other_station_rows_before == other_station_rows_after
    assert set(out["data_quality"].unique()) <= {"OBSERVED"}


def test_intermittent_drops_only_a_fraction(line):
    tel, _ = _telemetry(line, n_vehicles=1200)
    st = int(tel["station"].iloc[0])
    lo, hi = float(tel["t_start_s"].min()), float(tel["t_start_s"].max())
    fault = SensorFault(
        station=st, kind=INTERMITTENT, t_start_s=lo, t_end_s=hi,
        burst_on_s=200.0, burst_off_s=200.0,
    )
    out = apply_sensor_faults(tel, [fault], seed=0)
    before = len(tel[tel["station"] == st])
    after = len(out[out["station"] == st])
    assert 0 < after < before, "intermittent dropout should remove some but not all rows"


# --------------------------------------------------------------------- noisy


def test_noisy_increases_variance_in_interval(line):
    tel, _ = _telemetry(line, n_vehicles=800)
    st = int(tel["station"].iloc[0])
    lo, hi = float(tel["t_start_s"].quantile(0.3)), float(tel["t_start_s"].quantile(0.7))
    fault = SensorFault(station=st, kind=NOISY, t_start_s=lo, t_end_s=hi, noise_frac=1.5)
    out = apply_sensor_faults(tel, [fault], seed=0)

    before_std = tel[(tel["station"] == st) & (tel["t_start_s"] >= lo) & (tel["t_start_s"] < hi)][
        "proc_time_s"
    ].std()
    after_std = out[(out["station"] == st) & (out["t_start_s"] >= lo) & (out["t_start_s"] < hi)][
        "proc_time_s"
    ].std()
    assert after_std > before_std * 1.3
    tagged = out[(out["station"] == st) & (out["t_start_s"] >= lo) & (out["t_start_s"] < hi)]
    assert (tagged["data_quality"] == "NOISY").all()


# --------------------------------------------------------------------- stale


def test_stale_freezes_value_and_collapses_variance(line):
    tel, _ = _telemetry(line, n_vehicles=800)
    st = int(tel["station"].iloc[0])
    lo, hi = float(tel["t_start_s"].quantile(0.3)), float(tel["t_start_s"].quantile(0.7))
    fault = SensorFault(station=st, kind=STALE, t_start_s=lo, t_end_s=hi)
    out = apply_sensor_faults(tel, [fault], seed=0)

    frozen = out[(out["station"] == st) & (out["t_start_s"] >= lo) & (out["t_start_s"] < hi)]
    assert frozen["proc_time_s"].nunique() == 1, "stale rows should all read the same value"
    assert (frozen["data_quality"] == "STALE").all()


# ---------------------------------------------------------- stale detection


def test_flag_stale_windows_catches_zero_variance_and_spares_normal():
    windows = pd.DataFrame(
        {
            "window": [0, 0, 1, 1],
            "station": [1, 2, 1, 2],
            "proc_time_s_std": [0.0, 1.4, 1.1, 1.6],
            "proc_time_s_count": [20, 20, 20, 20],
        }
    )
    flagged = flag_stale_windows(windows)
    assert flagged.loc[(flagged.window == 0) & (flagged.station == 1), "stale_suspect"].iloc[0]
    assert not flagged.loc[(flagged.window == 0) & (flagged.station == 2), "stale_suspect"].iloc[0]
    masked = mask_suspect_rows(flagged)
    assert len(masked) == 3


def test_flag_stale_windows_respects_min_count():
    windows = pd.DataFrame(
        {"window": [0], "station": [1], "proc_time_s_std": [0.0], "proc_time_s_count": [2]}
    )
    flagged = flag_stale_windows(windows, min_count=6)
    assert not flagged["stale_suspect"].iloc[0], "too few vehicles to call this stale, not just quiet"


# -------------------------------------------------------------- data quality summary


def test_summarize_data_quality(line):
    tel, _ = _telemetry(line, n_vehicles=800)
    st = int(tel["station"].iloc[0])
    lo, hi = float(tel["t_start_s"].quantile(0.3)), float(tel["t_start_s"].quantile(0.7))
    out = apply_sensor_faults(tel, [SensorFault(station=st, kind=NOISY, t_start_s=lo, t_end_s=hi)], seed=0)
    summary = summarize_data_quality(out)
    assert (summary[(summary.station == st) & (summary.data_quality == "NOISY")]["fraction"] > 0).any()


# ------------------------------------------------------------ critical coverage


def test_critical_strategy_differs_from_random(line):
    random_view = apply_coverage(line, 0.5, seed=11, strategy="random")
    critical_view = apply_coverage(line, 0.5, seed=11, strategy="critical")
    assert set(random_view.hidden_indices) != set(critical_view.hidden_indices)
    assert len(random_view.hidden_indices) == len(critical_view.hidden_indices)


def test_unknown_strategy_raises(line):
    with pytest.raises(ValueError):
        apply_coverage(line, 0.5, strategy="bogus")


# ------------------------------------------------------------------ taxonomy


def _rec(action, abstained, priority=PRIORITY_LOW):
    return Recommendation(
        action=action, priority=priority, target_stations=["S05"], title="t",
        detail="d", rationale="because", units_at_stake=1.0, confidence=0.5,
        abstained=abstained,
    )


def test_taxonomy_escalate_vs_abstain():
    label, _ = taxonomy_label(_rec(ACTION_ESCALATE, abstained=True))
    assert label == TAXONOMY_ESCALATE
    label2, _ = taxonomy_label(_rec("CHECK_INBOUND_MATERIAL", abstained=True))
    assert label2 == TAXONOMY_ABSTAIN


def test_taxonomy_watch_and_investigate():
    label, _ = taxonomy_label(_rec(ACTION_MONITOR, abstained=False))
    assert label == TAXONOMY_WATCH
    label2, _ = taxonomy_label(_rec(ACTION_INSPECT, abstained=False))
    assert label2 == TAXONOMY_INVESTIGATE


# ------------------------------------------------------------------ end to end


def test_confidence_degrades_when_the_nearest_evidence_drops_out(line):
    """Direct evidence for the F acceptance criterion: confidence must
    respond to evidence quality, not stay artificially constant.

    Dropping out the observed stations nearest the true constraint removes
    the sharpest evidence available (their d_blocked/d_starved deviation is
    largest near the boundary). Confidence should fall -- and, just as
    important, it should fall *honestly*: among the windows that still clear
    the detection threshold, localisation accuracy must not collapse. A
    system that goes from confidently right to confidently wrong when
    evidence degrades is far more dangerous than one that goes quiet.
    """
    from rippletwin.factory.sensor_health import DROPOUT

    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)

    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    truth = res.disturbances.iloc[0]
    k = int(truth["station"])

    _, shadow_clean, _ = infer(ctx, res)
    during_clean = shadow_clean[
        (shadow_clean.t_mid_s >= truth.t_start_s) & (shadow_clean.t_mid_s <= truth.t_end_s)
    ]

    neighbours = line.nearest_observed_upstream(k, 2) + line.nearest_observed_downstream(k, 2)
    faults = [
        SensorFault(station=s, kind=DROPOUT, t_start_s=truth.t_start_s, t_end_s=truth.t_end_s)
        for s in neighbours
    ]
    res.telemetry = apply_sensor_faults(res.telemetry, faults, seed=0)
    _, shadow_deg, _ = infer(ctx, res)
    during_deg = shadow_deg[
        (shadow_deg.t_mid_s >= truth.t_start_s) & (shadow_deg.t_mid_s <= truth.t_end_s)
    ]

    assert during_deg["group_prob"].mean() < during_clean["group_prob"].mean() * 0.75
    assert during_deg["confident"].mean() < during_clean["confident"].mean()
    # Among windows that still clear the bar, accuracy should not collapse --
    # the system is allowed to go quiet, not allowed to go confidently wrong.
    still_confident = during_deg[during_deg["confident"]]
    if len(still_confident) > 5:
        acc = (still_confident["top_station"] == k).mean()
        assert acc > 0.6


def test_stale_windows_are_flagged_precisely_at_the_faulted_station(line):
    """A stale sensor keeps *looking* instrumented -- that is what makes it
    dangerous. The detector must catch it from the data's own shape (zero
    variance where a running station never actually holds still) with no
    false positives elsewhere on a clean line."""
    from rippletwin.factory.sensor_health import STALE

    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)

    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    truth = res.disturbances.iloc[0]
    k = int(truth["station"])
    neighbours = line.nearest_observed_upstream(k, 2) + line.nearest_observed_downstream(k, 2)
    faults = [
        SensorFault(station=s, kind=STALE, t_start_s=truth.t_start_s, t_end_s=truth.t_end_s)
        for s in neighbours
    ]
    res.telemetry = apply_sensor_faults(res.telemetry, faults, seed=0)

    w = build_windows(res, ctx.line, ctx.spec)
    flagged = flag_stale_windows(w)
    n_flagged = int(flagged["stale_suspect"].sum())
    assert n_flagged > 0, "the STALE fault should be detectable from collapsed variance"

    at_faulted_station = flagged["station"].isin(neighbours)
    false_positives = int((flagged["stale_suspect"] & ~at_faulted_station).sum())
    assert false_positives == 0, (
        f"{false_positives} windows flagged stale at a station that was never faulted"
    )


def test_pipeline_survives_a_dropout_overlapping_a_real_disturbance(line):
    """The system must keep operating -- not crash, not silently mislead --
    when a sensor drops out during an actual fault."""
    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    truth = res.disturbances.iloc[0]

    downstream = line.nearest_observed_downstream(int(truth["station"]), k=1)
    assert downstream, "scenario picker guarantees an observed downstream neighbour"
    faulted_station = downstream[0]

    fault = SensorFault(
        station=faulted_station, kind=DROPOUT,
        t_start_s=float(truth["t_start_s"]), t_end_s=float(truth["t_end_s"]),
    )
    res.telemetry = apply_sensor_faults(res.telemetry, [fault], seed=0)

    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)
    # Should not raise despite a real gap in an otherwise-instrumented station.
    scored, shadow, sensor = infer(ctx, res)
    assert len(shadow) > 0
