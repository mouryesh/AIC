"""Tests for Phase 6: the sensor coverage matrix (random vs critical),
distribution-shift robustness, and calibration (Brier/reliability) of the
two probabilistic heads.

These are smoke/shape tests at small episode counts -- the point is that
each experiment runs end to end and produces the right structure, not that
a handful of episodes prove a statistically powered trend.
"""

from __future__ import annotations

import numpy as np
import pytest

from rippletwin.evaluation.calibration import (
    CalibrationConfig,
    brier_score,
    expected_calibration_error,
    reliability_table,
    run_calibration_experiment,
)
from rippletwin.evaluation.coverage_matrix import CoverageMatrixConfig, run_coverage_matrix
from rippletwin.evaluation.distribution_shift import (
    DistributionShiftConfig,
    perturb_line,
    run_distribution_shift_experiment,
)
from rippletwin.factory.topology import build_line

CONFIG = "configs/line_42.yaml"


# ------------------------------------------------------------------ coverage matrix


def test_coverage_matrix_runs_and_covers_random_and_critical(tmp_path):
    cfg = CoverageMatrixConfig(n_episodes=2, coverages=(1.0, 0.5, 0.12))
    out = run_coverage_matrix(cfg, out_dir=tmp_path, verbose=False)
    summary = out["summary"]
    # coverage 1.0 only ever runs "random" (there is nothing to hide), the
    # rest should carry both strategies.
    assert set(summary[summary["coverage"] < 0.99]["strategy"]) == {"random", "critical"}
    for col in ("mean_confidence",):
        assert col in summary.columns


def test_coverage_10pct_is_infeasible_on_line_42_and_12pct_is_the_floor():
    from rippletwin.factory.topology import apply_coverage

    line = build_line(CONFIG, seed=7)
    with pytest.raises(ValueError):
        apply_coverage(line, 0.10, strategy="random")
    # 0.12 must succeed -- it is the documented floor (4 inspection gates +
    # station 0 are always instrumented).
    view = apply_coverage(line, 0.12, strategy="random")
    assert view.coverage <= 0.20


# ------------------------------------------------------------ distribution shift


def test_perturb_line_keeps_structure_but_raises_variability():
    line = build_line(CONFIG, seed=7)
    shifted = perturb_line(line, noise_mult=2.0, microstop_mult=2.0)
    assert shifted.n_stations == line.n_stations
    assert [s.tier for s in shifted.stations] == [s.tier for s in line.stations]
    assert [s.out_buffer for s in shifted.stations] == [s.out_buffer for s in line.stations]
    for a, b in zip(line.stations, shifted.stations):
        assert b.process_noise_cv >= a.process_noise_cv
        assert b.microstop_rate >= a.microstop_rate


def test_distribution_shift_experiment_runs_end_to_end(tmp_path):
    cfg = DistributionShiftConfig(n_episodes=3, episode_vehicles=700)
    out = run_distribution_shift_experiment(cfg, out_dir=tmp_path, verbose=False)
    summary = out["summary"]
    assert set(summary["regime"]) == {"matched", "shifted"}
    assert (summary["n_episodes"] == 3).all()


# -------------------------------------------------------------------- calibration


def test_reliability_table_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = rng.random(2000) < p  # by construction, well-calibrated
    table = reliability_table(y, p, n_bins=5, min_bin_n=20)
    assert len(table) == 5
    reliable = table[table["reliable"]]
    assert len(reliable) > 0
    assert (reliable["abs_gap"] < 0.15).all()


def test_reliability_table_flags_small_bins_as_unreliable():
    p = [0.05] * 100 + [0.95] * 3  # top bin has almost no data
    y = [False] * 100 + [True] * 3
    table = reliability_table(y, p, n_bins=5, min_bin_n=20)
    top_bin = table.iloc[-1]
    assert top_bin["n"] == 3
    assert not top_bin["reliable"]


def test_brier_score_perfect_vs_worst():
    assert brier_score([True, False], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([True, False], [0.0, 1.0]) == pytest.approx(1.0)


def test_expected_calibration_error_excludes_unreliable_bins():
    table = reliability_table([False] * 100 + [True] * 3, [0.05] * 100 + [0.95] * 3,
                               n_bins=5, min_bin_n=20)
    ece = expected_calibration_error(table)
    assert ece["n_reliable_bins"] < 5
    assert ece["coverage_frac"] < 1.0


def test_calibration_experiment_runs_end_to_end(tmp_path):
    cfg = CalibrationConfig(n_episodes=2, episode_vehicles=500)
    out = run_calibration_experiment(cfg, out_dir=tmp_path, verbose=False)
    assert "bottleneck_risk" in out["results"]
    assert "defect_risk" in out["results"]
    for head in ("bottleneck_risk", "defect_risk"):
        assert out["results"][head]["n"] > 0
        assert 0.0 <= out["results"][head]["brier"] <= 1.0
