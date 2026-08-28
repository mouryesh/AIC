"""Tests for the predictive defect-risk layer (twin.defect_risk).

Checks: RawProcessBaseline scores held-out nominal data near zero; a RICH
station under a real quality-drift disturbance scores measurably higher risk
than the same station nominally; MANUAL stations are reported as an explicit
coverage gap rather than silently skipped; the evaluation experiment runs
end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from rippletwin.evaluation.defect_prediction import (
    DefectPredictionConfig,
    run_defect_prediction_experiment,
)
from rippletwin.factory import scenarios as SC
from rippletwin.factory.topology import build_line
from rippletwin.twin.defect_risk import (
    DefectRiskConfig,
    RawProcessBaseline,
    coverage_gap_report,
    predict_defect_risk,
    score_vehicles,
)
from rippletwin.twin.pipeline import fit_context, simulate

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


@pytest.fixture(scope="module")
def nominal(line):
    return simulate(line, SC.nominal_run(1800), seed=1)


@pytest.fixture(scope="module")
def ctx(line, nominal):
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    return fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)


@pytest.fixture(scope="module")
def raw_baseline(nominal, line):
    return RawProcessBaseline.fit(nominal.telemetry, line)


@pytest.fixture(scope="module")
def model_cfg(line, nominal, ctx, raw_baseline):
    cfg = DefectRiskConfig()
    scored = score_vehicles(line, nominal.telemetry, ctx.baseline, raw_baseline, cfg)
    cfg.fit_scale(scored["_combined"].to_numpy())
    return cfg


def test_coverage_gap_report_lists_every_manual_station(line):
    gaps = coverage_gap_report(line)
    assert len(gaps) == len(line.hidden_indices)
    assert set(gaps["station_id"]) == {line.stations[i].station_id for i in line.hidden_indices}


def test_held_out_nominal_scores_low_risk(line, ctx, raw_baseline, model_cfg):
    heldout = simulate(line, SC.nominal_run(1400), seed=99)
    scored = score_vehicles(line, heldout.telemetry, ctx.baseline, raw_baseline, model_cfg)
    r = scored["risk"].dropna()
    assert r.mean() < 0.15, f"nominal risk should sit low, got mean={r.mean():.3f}"
    assert np.percentile(r, 99) < 0.85


def test_quality_drift_raises_risk_at_the_true_station(line, ctx, raw_baseline, model_cfg):
    scen = SC.scenario_observed_station(line)  # combined kind, observed station
    res = simulate(line, scen, seed=20260301)
    truth = res.disturbances.iloc[0]
    k = int(truth["station"])
    assert not line.stations[k].is_hidden

    scored = score_vehicles(line, res.telemetry, ctx.baseline, raw_baseline, model_cfg)
    during = scored[
        (scored["station"] == k)
        & (scored["t_start_s"] >= float(truth["t_start_s"]) + float(truth["ramp_s"]))
        & (scored["t_start_s"] <= float(truth["t_end_s"]))
    ]
    before = scored[
        (scored["station"] == k) & (scored["t_start_s"] < float(truth["t_start_s"]))
    ]
    assert len(during) > 20 and len(before) > 20
    assert during["risk"].mean() > before["risk"].mean(), (
        "risk should rise at the disturbed station relative to its own baseline period"
    )


def test_predict_defect_risk_shape(line, ctx, raw_baseline, model_cfg):
    scen = SC.scenario_observed_station(line)
    res = simulate(line, scen, seed=20260301)
    scored = score_vehicles(line, res.telemetry.head(500), ctx.baseline, raw_baseline, model_cfg)
    preds = predict_defect_risk(line, scored, model_cfg)
    assert len(preds) == len(scored)
    for p in preds:
        assert not p.is_hidden  # telemetry never carries MANUAL rows
        assert 0.0 <= p.risk <= 1.0
        assert p.likely_origin_station_id == p.station_id
        assert p.confidence <= max(model_cfg.confidence_rich, model_cfg.confidence_basic)


def test_defect_prediction_experiment_runs_end_to_end(tmp_path):
    cfg = DefectPredictionConfig(n_episodes=3, episode_vehicles=600)
    out = run_defect_prediction_experiment(cfg, out_dir=tmp_path, verbose=False)
    summary = out["summary"]
    for key in ("model_precision", "model_recall", "model_brier", "naive_precision"):
        assert key in summary
    assert len(out["coverage_gaps"]) == 10  # matches line_42's manual_fraction
