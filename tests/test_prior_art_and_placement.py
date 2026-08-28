"""Tests for the Turning Point baseline and the sensor-placement model.

These cover the two things added after reviewing the manufacturing-science
literature: an implementation of the published method our mechanism derives
from, and the sensor-placement guidance that falls out of the same model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rippletwin.factory import scenarios as SC
from rippletwin.factory.topology import build_line
from rippletwin.models.baselines import (
    apply_detection_rule,
    calibrate_threshold,
    turning_point_baseline,
)
from rippletwin.twin.pipeline import fit_context, infer, simulate
from rippletwin.twin.placement import (
    ambiguity,
    recommend_sensors,
    suspicion_from_shadow,
)

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


@pytest.fixture(scope="module")
def ctx(line):
    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    return fit_context(line, nominal, calibration_run=calib)


# --------------------------------------------------- B3: Turning Point Method


def test_turning_point_finds_the_flip_in_a_synthetic_profile(line):
    """The published rule: blocked-dominant flips to starved-dominant."""
    n_obs = len(line.observed_indices)
    # Blocked upstream of index 20, starved downstream of it.
    rows = []
    for i in line.observed_indices:
        rows.append({
            "window": 0, "station": i,
            "d_blocked": 0.3 if i < 20 else 0.0,
            "d_starved": 0.0 if i < 20 else 0.3,
        })
    frame = turning_point_baseline(pd.DataFrame(rows), line).frame
    assert len(frame) == 1
    named = int(frame["top_station"].iloc[0])
    # It must name the first station on the starved side of the flip.
    assert named >= 20
    assert named == min(i for i in line.observed_indices if i >= 20)


def test_turning_point_special_cases_are_implemented_as_published(line):
    """All-starved names the first station; all-blocked names the last."""
    for blocked, starved, expect in [(0.0, 0.3, "first"), (0.3, 0.0, "last")]:
        rows = [
            {"window": 0, "station": i, "d_blocked": blocked, "d_starved": starved}
            for i in line.observed_indices
        ]
        frame = turning_point_baseline(pd.DataFrame(rows), line).frame
        named = int(frame["top_station"].iloc[0])
        if expect == "first":
            assert named == min(line.observed_indices)
        else:
            assert named == max(line.observed_indices)


def test_turning_point_can_never_name_an_uninstrumented_station(line, ctx):
    """The decisive limitation, asserted rather than asserted-about.

    This is the whole reason RippleTwin exists: the published method scans the
    stations it can measure, so a turning point inside a sensor gap is outside
    its output space.
    """
    res = simulate(line, SC.scenario_hidden_bottleneck(line), seed=20260301)
    scored, _, _ = infer(ctx, res)
    frame = turning_point_baseline(scored, line).frame
    assert len(frame) > 0
    hidden = set(line.hidden_indices)
    assert not set(frame["top_station"].unique()) & hidden


def test_turning_point_misses_a_hidden_source_that_rippletwin_finds(line, ctx):
    """Head-to-head on the flagship scenario, at a matched false-alarm rate."""
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    cal_scored, _, _ = infer(ctx, calib)
    thr = calibrate_threshold(turning_point_baseline(cal_scored, line).frame, 0.01)

    res = simulate(line, SC.scenario_hidden_bottleneck(line), seed=20260301)
    scored, shadow, _ = infer(ctx, res)
    truth = res.disturbances.iloc[0]
    k = int(truth["station"])
    assert line.stations[k].is_hidden

    t0 = float(truth["t_start_s"]) + float(truth["ramp_s"])
    t1 = float(truth["t_end_s"])

    tp = apply_detection_rule(turning_point_baseline(scored, line).frame, thr)
    wt = scored.groupby("window").agg(t_mid_s=("t_depart_s_min", "min")).reset_index()
    tp = tp.merge(wt, on="window")
    tp_active = tp[tp["detected"] & (tp["t_mid_s"] >= t0) & (tp["t_mid_s"] <= t1)]

    rt_active = shadow[
        shadow["detected"] & (shadow["t_mid_s"] >= t0) & (shadow["t_mid_s"] <= t1)
    ]

    assert len(tp_active) > 0, "the baseline must actually detect, or it is a strawman"
    assert (tp_active["top_station"] == k).mean() == 0.0
    assert (rt_active["top_station"] == k).mean() > 0.5


# ------------------------------------------------------- sensor placement


def test_adjacent_blind_stations_are_flagged_as_mutually_confusable(line):
    """Two blind stations side by side cannot be separated by flow evidence."""
    amb = ambiguity(line, line.observed_indices).set_index("station")
    adjacent = [
        (i, i + 1) for i in line.hidden_indices if (i + 1) in set(line.hidden_indices)
    ]
    if not adjacent:
        pytest.skip("this line has no adjacent blind pair")
    isolated = [
        i for i in line.hidden_indices
        if (i + 1) not in set(line.hidden_indices)
        and (i - 1) not in set(line.hidden_indices)
    ]
    for a, b in adjacent:
        assert amb.loc[a, "ambiguity"] > 0.9
        assert amb.loc[b, "ambiguity"] > 0.9
        # and each should name the other as its confusable partner
        assert amb.loc[a, "confusable_with"] == line.stations[b].station_id
    if isolated:
        worst_adjacent = max(amb.loc[a, "ambiguity"] for a, _ in adjacent)
        best_isolated = min(amb.loc[i, "ambiguity"] for i in isolated)
        assert worst_adjacent > best_isolated


def test_ambiguity_is_bounded_and_defined_for_every_station(line):
    amb = ambiguity(line, line.observed_indices)
    assert len(amb) == line.n_stations
    assert amb["ambiguity"].between(0.0, 1.0).all()
    assert np.allclose(amb["ambiguity"] + amb["resolvability"], 1.0)


@pytest.mark.parametrize("which", [0, 1, -1])
def test_adding_a_sensor_never_reduces_separability(line, which):
    """Information must never hurt — the property the placement metric needs.

    This is the metric the placement ranking uses, and monotonicity is exactly
    why it uses it: a value-of-information score that can go *down* when you add
    a sensor would tell a plant to skip a retrofit that actually helps.
    """
    base = ambiguity(line, line.observed_indices).set_index("station")["separability"]
    c = line.hidden_indices[which]
    after = ambiguity(
        line, sorted(set(line.observed_indices) | {c})
    ).set_index("station")["separability"]
    assert (after >= base - 1e-9).all(), "separability must be non-decreasing"


def test_cosine_ambiguity_is_deliberately_not_used_for_placement(line):
    """Documents *why* the placement metric is not the intuitive one.

    Cosine similarity between two candidate response patterns can *rise* when a
    sensor is added, because a new observer that responds similarly to both adds
    a large common-mode component and pulls their angle together. That is a real
    property of an angle-based measure, not a bug — but it makes cosine unfit
    for a value-of-information decision, which is why ``recommend_sensors``
    ranks on separability instead.

    If this test ever starts failing, the non-monotonicity has gone away and the
    comment above should be revisited.
    """
    base = ambiguity(line, line.observed_indices).set_index("station")["ambiguity"]
    c = line.hidden_indices[0]
    after = ambiguity(
        line, sorted(set(line.observed_indices) | {c})
    ).set_index("station")["ambiguity"]
    assert (after > base + 1e-9).any(), (
        "expected cosine ambiguity to be non-monotone under adding a sensor"
    )


def test_placement_prioritises_the_adjacent_blind_pairs(line):
    rec = recommend_sensors(line, n_recommend=5)
    assert len(rec) > 0
    assert rec["total_gain"].is_monotonic_decreasing
    adjacent = {
        i for i in line.hidden_indices if (i + 1) in set(line.hidden_indices)
    } | {
        i for i in line.hidden_indices if (i - 1) in set(line.hidden_indices)
    }
    if adjacent:
        top2 = set(rec["station"].head(2))
        assert top2 & adjacent, "adjacent blind stations should rank first"


def test_placement_needs_no_production_data(line):
    """A plant must be able to run this before committing to a retrofit."""
    rec = recommend_sensors(line)
    assert len(rec) > 0
    assert {"station_id", "total_gain", "unlocks"} <= set(rec.columns)


def test_suspicion_weighting_accepts_empty_history():
    assert suspicion_from_shadow([]) == {}
    assert suspicion_from_shadow([pd.DataFrame()]) == {}
