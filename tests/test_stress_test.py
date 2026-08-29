"""Tests for evaluation.stress_test (Plan B, RESEARCH_EVALUATION.md #11)."""

import pytest

from rippletwin.evaluation.experiments import ExperimentConfig
from rippletwin.evaluation.stress_test import StressCondition, run_stress_test


def _tiny_cfg() -> ExperimentConfig:
    # Small enough to run in a test suite: 2 test episodes, small vehicle counts.
    return ExperimentConfig(
        n_tune_episodes=0,
        n_test_episodes=2,
        episode_vehicles=300,
        nominal_vehicles=600,
        calibration_vehicles=600,
    )


def test_oracle_self_check_is_always_zero(tmp_path):
    cfg = _tiny_cfg()
    grid = [StressCondition(coverage=0.75, fault_kind=None, label="cov75_clean")]
    result = run_stress_test(cfg, stress_grid=grid, out_dir=tmp_path, verbose=False)
    assert result["self_check"]["self_mismatch_rate"].max() == pytest.approx(0.0)


def test_decision_mismatch_rate_nondecreasing_with_fault_severity(tmp_path):
    cfg = _tiny_cfg()
    grid = [
        StressCondition(coverage=0.75, fault_kind=None, fault_fraction_of_run=0.0, label="clean"),
        StressCondition(coverage=0.75, fault_kind="DROPOUT", fault_fraction_of_run=0.10, label="dropout10"),
        StressCondition(coverage=0.75, fault_kind="DROPOUT", fault_fraction_of_run=0.50, label="dropout50"),
    ]
    result = run_stress_test(cfg, stress_grid=grid, out_dir=tmp_path, verbose=False)
    means = result["summary"].set_index("condition")["mean_dmr"]
    assert means["clean"] <= means["dropout10"] + 1e-9
    assert means["dropout10"] <= means["dropout50"] + 1e-9


def test_outcome_gap_metric_has_no_nan_crash(tmp_path):
    cfg = _tiny_cfg()
    grid = [StressCondition(coverage=0.50, fault_kind="NOISY", fault_fraction_of_run=0.20, label="cov50_noisy")]
    result = run_stress_test(cfg, stress_grid=grid, out_dir=tmp_path, verbose=False)
    assert not result["detail"].empty
    assert "outcome_gap_station_distance" in result["detail"].columns
