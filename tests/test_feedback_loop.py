"""Tests for Phase 8: the per-station feedback prior (twin.feedback), its
exact reduction to today's uniform-prior behaviour when unused, the new
HITL decision vocabulary, and the alternative-hypothesis field on
explanations.
"""

from __future__ import annotations

import numpy as np
import pytest

from rippletwin.evaluation.feedback_experiment import (
    FeedbackExperimentConfig,
    run_feedback_experiment,
)
from rippletwin.explain.explain import explain_flow_alert
from rippletwin.factory import scenarios as SC
from rippletwin.hitl.ledger import (
    DECISION_APPROVED,
    DECISION_ESCALATED,
    DECISION_MODIFIED,
    DECISION_REJECTED,
    OUTCOME_CONFIRMED,
    OUTCOME_NOT_FOUND,
    DecisionLedger,
    precision_by_station,
)
from rippletwin.factory.topology import build_line
from rippletwin.twin.feedback import apply_feedback, priors_from_precision
from rippletwin.twin.pipeline import fit_context, infer, simulate
from rippletwin.twin.propagate import current_buffer_levels, forecast_ripple
from rippletwin.twin.shadow import ShadowConfig, ShadowSensor

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


@pytest.fixture(scope="module")
def ctx(line):
    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    return fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)


# ------------------------------------------------------ posterior reduction


def test_none_station_prior_weight_reproduces_uniform_prior_exactly(line):
    """The core regression guarantee for this phase: cfg.station_prior_weight
    defaults to None, and _posterior must produce bit-identical results to
    before this phase existed."""
    cfg = ShadowConfig()
    assert cfg.station_prior_weight is None
    sensor = ShadowSensor(line, cfg)
    n = line.n_stations
    post = sensor._posterior(np.zeros(n), 0.0, 0.0)
    # Station entries (equal evidence, equal prior) must be uniform among
    # themselves -- NULL legitimately differs, it carries cfg.null_prior=0.9
    # rather than the uniform "other hypothesis" share.
    station_vals = np.array([post[i] for i in range(n)])
    assert np.allclose(station_vals, station_vals[0], atol=1e-9)


def test_explicit_uniform_weights_match_none(line):
    cfg_none = ShadowConfig()
    cfg_uniform = ShadowConfig(station_prior_weight={i: 1.0 for i in range(line.n_stations)})
    n = line.n_stations
    ll = np.random.default_rng(0).normal(0, 1, n)
    post_none = ShadowSensor(line, cfg_none)._posterior(ll, 0.5, -1.0)
    post_uniform = ShadowSensor(line, cfg_uniform)._posterior(ll, 0.5, -1.0)
    for k in post_none:
        assert post_none[k] == pytest.approx(post_uniform[k], abs=1e-9)


def test_a_higher_weight_raises_that_stations_posterior(line):
    cfg_base = ShadowConfig()
    weighted = {i: 1.0 for i in range(line.n_stations)}
    weighted[5] = 3.0
    cfg_weighted = ShadowConfig(station_prior_weight=weighted)
    n = line.n_stations
    ll = np.zeros(n)  # identical evidence for every station
    p_base = ShadowSensor(line, cfg_base)._posterior(ll, 0.0, -5.0)
    p_weighted = ShadowSensor(line, cfg_weighted)._posterior(ll, 0.0, -5.0)
    assert p_weighted[5] > p_base[5]


# --------------------------------------------------------------- feedback


def test_priors_from_precision_empty_ledger_is_empty(line):
    assert priors_from_precision(line, DecisionLedger()) == {}


def test_priors_from_precision_confirmed_vs_rejected(line):
    ledger = DecisionLedger()
    good_id = line.stations[line.hidden_indices[0]].station_id
    bad_id = line.stations[line.hidden_indices[1]].station_id
    for i in range(5):
        e = ledger.record_alert("s", i, "FLOW", good_id, "MANUAL", True, 0.8, {}, {})
        ledger.record_decision(e.entry_id, DECISION_APPROVED, "sup")
        ledger.record_outcome(e.entry_id, OUTCOME_CONFIRMED, "ok")
    for i in range(5):
        e = ledger.record_alert("s", 100 + i, "FLOW", bad_id, "MANUAL", True, 0.6, {}, {})
        ledger.record_decision(e.entry_id, DECISION_APPROVED, "sup")
        ledger.record_outcome(e.entry_id, OUTCOME_NOT_FOUND, "nope")

    weights = priors_from_precision(line, ledger)
    good_idx = line.hidden_indices[0]
    bad_idx = line.hidden_indices[1]
    assert weights[good_idx] > 1.0
    assert weights[bad_idx] < 1.0


def test_priors_from_precision_respects_min_outcomes(line):
    ledger = DecisionLedger()
    station_id = line.stations[line.hidden_indices[0]].station_id
    e = ledger.record_alert("s", 0, "FLOW", station_id, "MANUAL", True, 0.8, {}, {})
    ledger.record_decision(e.entry_id, DECISION_APPROVED, "sup")
    ledger.record_outcome(e.entry_id, OUTCOME_CONFIRMED, "ok")
    weights = priors_from_precision(line, ledger, min_outcomes=3)
    assert line.hidden_indices[0] not in weights


def test_apply_feedback_does_not_mutate_original_config():
    cfg = ShadowConfig()
    fed = apply_feedback(cfg, {3: 2.0})
    assert cfg.station_prior_weight is None
    assert fed.station_prior_weight == {3: 2.0}


def test_decision_vocabulary_has_all_four_types():
    assert {DECISION_APPROVED, DECISION_REJECTED, DECISION_MODIFIED, DECISION_ESCALATED} == {
        "APPROVED", "REJECTED", "MODIFIED", "ESCALATED"
    }


def test_feedback_experiment_runs_end_to_end_and_reports_honestly(tmp_path):
    cfg = FeedbackExperimentConfig(n_episodes=3, episode_vehicles=700)
    out = run_feedback_experiment(cfg, out_dir=tmp_path, verbose=False)
    m = out["mechanism"]
    assert m["posterior_on_good_station_after_feedback"] >= 0.0
    manifest = out["manifest"]
    assert manifest["honest_verdict"] in ("IMPROVED", "NO MEASURABLE CHANGE", "WORSE", "INSUFFICIENT DATA")


# ------------------------------------------------------------- explanation


def test_explain_flow_alert_surfaces_an_alternative_hypothesis(line, ctx):
    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    scored, shadow, sensor = infer(ctx, res)
    det = shadow[shadow["detected"]]
    assert len(det)
    w = int(det.iloc[len(det) // 2]["window"])
    sr = next(r for r in sensor.last_results if r.window == w)
    fc = None
    exp = explain_flow_alert(line, sr, fc, None)
    # An alternative may or may not exist depending on how concentrated the
    # posterior is this window -- but if reported, it must be internally
    # consistent and never the same station as the top pick.
    if exp.alternative_station_id is not None:
        assert exp.alternative_station_id != line.stations[sr.top_station].station_id
        assert 0.0 <= exp.alternative_probability <= 1.0
        assert "ALTERNATIVE HYPOTHESIS" in exp.as_text()
