"""Tests for evaluation.bottleneck_diagnosis (Plan A, RESEARCH_EVALUATION.md #1)."""

import pandas as pd
import pytest

from rippletwin.evaluation.bottleneck_diagnosis import (
    bottleneck_frequency,
    bottleneck_shift_severity,
)
from rippletwin.factory.topology import build_line


LINE = build_line("configs/line_42.yaml", seed=7)


def test_bottleneck_frequency_sums_to_one_when_detections_exist():
    shadow = pd.DataFrame(
        {
            "window": [0, 1, 2, 3],
            "top_station": [2, 2, 5, 2],
            "detected": [True, True, True, False],
        }
    )
    rbf = bottleneck_frequency(shadow, LINE)
    assert rbf["rbf"].sum() == pytest.approx(1.0)
    assert rbf.loc[rbf["station"] == 2, "rbf"].iloc[0] == pytest.approx(2 / 3)
    assert rbf.loc[rbf["station"] == 5, "rbf"].iloc[0] == pytest.approx(1 / 3)


def test_bottleneck_frequency_empty_shadow_returns_empty_frame():
    empty = pd.DataFrame(columns=["window", "top_station", "detected"])
    rbf = bottleneck_frequency(empty, LINE)
    assert rbf.empty or rbf["rbf"].sum() == 0


def test_shift_severity_is_one_at_leaders_own_row_by_construction():
    shadow = pd.DataFrame(
        {
            "window": [0],
            "top_station": [3],
            "runner_up_station": [4],
            "runner_up_prob": [0.6],
            "group_prob": [0.6],
        }
    )
    sev = bottleneck_shift_severity(shadow)
    assert sev["severity_ratio"].iloc[0] == pytest.approx(1.0)


def test_shift_severity_rises_before_a_bottleneck_shift():
    # Synthetic two-station alternating-bottleneck sequence: station 3 leads
    # early, station 4's runner-up mass climbs, then station 4 takes over.
    shadow = pd.DataFrame(
        {
            "window": [0, 1, 2, 3],
            "top_station": [3, 3, 3, 4],
            "runner_up_station": [4, 4, 4, 3],
            "runner_up_prob": [0.10, 0.30, 0.55, 0.65],
            "group_prob": [0.80, 0.70, 0.60, 0.65],
        }
    )
    sev = bottleneck_shift_severity(shadow)
    ratios = sev["severity_ratio"].tolist()
    assert ratios[0] < ratios[1] < ratios[2]  # rising before the shift
    assert ratios[3] == pytest.approx(1.0)     # new leader's own row


def test_shift_severity_raises_on_missing_columns():
    incomplete = pd.DataFrame({"window": [0], "top_station": [1]})
    with pytest.raises(KeyError):
        bottleneck_shift_severity(incomplete)
