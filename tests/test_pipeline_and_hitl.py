"""Tests for genealogy attribution, propagation, explanations, HITL and baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rippletwin.explain.explain import INFERRED, OBSERVED, PREDICTED, explain_flow_alert
from rippletwin.factory import scenarios as SC
from rippletwin.factory.topology import apply_coverage, build_line
from rippletwin.hitl.ledger import (
    DECISION_APPROVED,
    OUTCOME_CONFIRMED,
    OUTCOME_NOT_FOUND,
    DecisionLedger,
    precision_by_station,
)
from rippletwin.models import baselines as BL
from rippletwin.recommend.engine import ACTION_ESCALATE, recommend_flow
from rippletwin.twin import genealogy as GN
from rippletwin.twin.pipeline import fit_context, infer, simulate
from rippletwin.twin.propagate import defect_exposure, forecast_ripple
from rippletwin.evaluation.views import full_observability, telemetry_view

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


@pytest.fixture(scope="module")
def ctx(line):
    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    return fit_context(line, nominal, calibration_run=calib)


# ------------------------------------------------------------------ genealogy


def test_genealogy_reconstructs_hidden_pass_times(line):
    """Interpolated pass times for blind stations must be within a takt or so."""
    res = simulate(line, SC.nominal_run(600), seed=5)
    gen = GN.build_genealogy(line, res.telemetry, res.vehicles)
    truth = res.passes.pivot_table(
        index="vehicle_id", columns="station", values="t_depart_s"
    )
    for h in line.hidden_indices:
        err = np.abs(gen[h].to_numpy() - truth[h].to_numpy())
        assert np.mean(err) < line.takt_s, (
            f"station {h} genealogy error {np.mean(err):.1f}s exceeds takt"
        )


def test_defect_attribution_respects_causality(line):
    res = simulate(line, SC.nominal_run(800), seed=6)
    prior = GN.candidate_prior(line)
    found = GN.explode_defects(res.inspections)
    att = GN.attribute_defects(line, found, prior)
    if len(att):
        # a defect can only be attributed upstream of the gate that found it
        assert (att["station"] < att["gate_station"]).all()
        # soft assignment must be a distribution
        g = att.groupby(["vehicle_id", "gate_station", "defect_type"])["mass"].sum()
        assert np.allclose(g.to_numpy(), 1.0, atol=1e-6)


def test_candidate_prior_is_normalised_per_defect_type(line):
    prior = GN.candidate_prior(line)
    s = prior.groupby("defect_type")["prior"].sum()
    assert np.allclose(s.to_numpy(), 1.0, atol=1e-9)


def test_quality_path_ranks_a_hidden_drift_highly(line, ctx):
    """The flow model is blind to a pure quality drift; this path is not."""
    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    prior = GN.candidate_prior(line)
    qbase = GN.QualityBaseline.fit(
        line, GN.attribute_defects(line, GN.explode_defects(nominal.inspections), prior),
        n_vehicles=len(nominal.vehicles),
    )
    scen = SC.scenario_hidden_quality(line)
    res = simulate(line, scen, seed=20260301)
    scored, shadow, _ = infer(ctx, res)
    wb = GN.window_bounds_from(scored)
    qs = GN.quality_state(line, GN.explode_defects(res.inspections), wb, qbase)
    assert len(qs) > 0

    truth = res.disturbances.iloc[0]
    k = int(truth["station"])
    sel = wb[(wb["t_lo"] > float(truth["t_start_s"]) + float(truth["ramp_s"]))
             & (wb["t_hi"] < float(truth["t_end_s"]))]["window"]
    during = qs[qs["window"].isin(sel)]
    rank = during.groupby("station")["llr"].mean().sort_values(ascending=False)
    pos = list(rank.index).index(k) + 1
    # Attribution is a shortlist, not a verdict: top-10 of 42 is the honest claim.
    assert pos <= 10, f"true source ranked {pos} of {len(rank)}"


def test_quality_path_is_quiet_on_a_clean_line(line, ctx):
    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    prior = GN.candidate_prior(line)
    qbase = GN.QualityBaseline.fit(
        line, GN.attribute_defects(line, GN.explode_defects(nominal.inspections), prior),
        n_vehicles=len(nominal.vehicles),
    )
    res = simulate(line, SC.scenario_normal_variation(line), seed=1234)
    scored, _, _ = infer(ctx, res)
    wb = GN.window_bounds_from(scored)
    qs = GN.quality_state(line, GN.explode_defects(res.inspections), wb, qbase)
    qa = GN.quality_alerts(qs)
    assert qa["quality_alert"].mean() < 0.02


# ---------------------------------------------------------------- propagation


def test_forecast_is_arithmetically_consistent(line):
    fc = forecast_ripple(line, station=10, constraint_cycle_s=75.0, horizon_min=60.0)
    assert fc.is_binding
    assert fc.sustained_rate_vph == pytest.approx(3600 / 75.0)
    expected_loss = (1 / line.takt_s - 1 / 75.0) * 3600.0
    assert fc.units_lost_at_horizon == pytest.approx(expected_loss, rel=1e-6)
    assert 0 < fc.throughput_loss_pct < 1


def test_forecast_reports_no_loss_inside_takt(line):
    fc = forecast_ripple(line, station=10, constraint_cycle_s=50.0)
    assert not fc.is_binding
    assert fc.throughput_loss_pct == 0.0
    assert fc.units_lost_at_horizon == 0.0


def test_defect_exposure_scales_with_multiplier(line):
    a = defect_exposure(line, 10, 1.0, 0.01, 100)
    b = defect_exposure(line, 10, 5.0, 0.01, 100)
    assert a["expected_extra_defective_units"] == pytest.approx(0.0)
    assert b["expected_extra_defective_units"] > a["expected_extra_defective_units"]


# --------------------------------------------------------------- explanation


def test_explanation_is_grounded_in_model_evidence(line, ctx):
    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    scored, shadow, sensor = infer(ctx, res)
    det = shadow[shadow["detected"]]
    assert len(det)
    w = int(det.iloc[len(det) // 2]["window"])
    sr = next(r for r in sensor.last_results if r.window == w)
    fc = forecast_ripple(line, sr.top_station, 75.0)
    exp = explain_flow_alert(line, sr, fc, 75.0)

    assert exp.station_id == line.stations[sr.top_station].station_id
    assert exp.evidence, "an explanation with no evidence is not an explanation"
    tags = {e.provenance for e in exp.evidence}
    assert tags <= {OBSERVED, INFERRED, PREDICTED}
    # a hidden station must be flagged as inferred and must carry a caveat
    if line.stations[sr.top_station].is_hidden:
        assert exp.is_inferred
        assert any("inferred" in c.lower() for c in exp.caveats)
    text = exp.as_text()
    assert "WHAT CHANGED" in text and "CONFIDENCE" in text


def test_recommendation_never_names_the_target_as_its_own_fallback(line, ctx):
    """The fallback must send a technician somewhere else, not back again."""
    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    _, shadow, sensor = infer(ctx, res)
    det = shadow[shadow["detected"]]
    assert len(det)
    w = int(det.iloc[len(det) // 2]["window"])
    sr = next(r for r in sensor.last_results if r.window == w)
    fc = forecast_ripple(line, sr.top_station, 76.0)
    rec = recommend_flow(line, sr, fc)
    top_id = line.stations[sr.top_station].station_id
    for alt in rec.alternatives:
        head, _, tail = alt.partition("is clear, check")
        if tail:
            assert top_id not in tail, f"fallback points back at {top_id}: {alt}"


def test_abstains_when_the_posterior_is_spread(line, ctx):
    """Low confidence must escalate, not guess."""
    from rippletwin.twin.shadow import ShadowResult

    sr = ShadowResult(
        window=1, posterior={}, top_station=10, top_prob=0.12,
        group=[8, 9, 10, 11, 12], group_prob=0.30, llr=99.0,
        amp_starve=0.2, amp_block=0.1, top_is_hidden=True,
        detected=True, confident=False,
        evidence={"p_null": 0.1, "p_line_supply": 0.05,
                  "d_blocked": np.zeros(line.n_stations),
                  "d_starved": np.zeros(line.n_stations),
                  "z_proc": np.zeros(line.n_stations),
                  "station_post": np.zeros(line.n_stations)},
    )
    rec = recommend_flow(line, sr, forecast_ripple(line, 10, 75.0))
    assert rec.abstained
    assert rec.action == ACTION_ESCALATE


# ---------------------------------------------------------------------- HITL


def test_ledger_chain_detects_tampering():
    led = DecisionLedger()
    e = led.record_alert("run", 1, "FLOW", "S08", "MANUAL", True, 0.9, {}, {})
    led.record_decision(e.entry_id, DECISION_APPROVED, "supervisor")
    led.record_outcome(e.entry_id, OUTCOME_CONFIRMED, "found")
    assert led.verify()["valid"]

    # An alert quietly rewritten after the fact must break the chain.
    led.entries[0].station_id = "S09"
    v = led.verify()
    assert not v["valid"]
    assert v["broken_at"] == 1


def test_ledger_roundtrips_through_disk(tmp_path):
    led = DecisionLedger()
    e = led.record_alert("run", 1, "FLOW", "S08", "MANUAL", True, 0.9, {"a": 1}, {"b": 2})
    led.record_decision(e.entry_id, DECISION_APPROVED, "supervisor")
    p = led.save(tmp_path / "ledger.jsonl")

    led2 = DecisionLedger(p)
    assert len(led2.entries) == len(led.entries)
    assert led2.verify()["valid"]


def test_precision_feedback_reflects_outcomes():
    led = DecisionLedger()
    for _ in range(4):
        e = led.record_alert("r", 1, "FLOW", "S08", "MANUAL", True, 0.9, {}, {})
        led.record_outcome(e.entry_id, OUTCOME_CONFIRMED)
    e = led.record_alert("r", 1, "FLOW", "S20", "BASIC", False, 0.9, {}, {})
    led.record_outcome(e.entry_id, OUTCOME_NOT_FOUND)

    prec = precision_by_station(led)
    p8 = prec[prec["station_id"] == "S08"]["precision"].iloc[0]
    p20 = prec[prec["station_id"] == "S20"]["precision"].iloc[0]
    assert p8 > p20


# ----------------------------------------------------------------- baselines


def test_observed_only_twin_never_names_a_hidden_station(line, ctx):
    """B2 is a conventional twin: it must be structurally unable to see blind stations."""
    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    scored, _, _ = infer(ctx, res)
    br = BL.observed_only_twin(
        line, scored, ctx.shadow_cfg,
        ctx.baseline.sigma_blocked, ctx.baseline.sigma_starved,
    )
    hidden = set(line.hidden_indices)
    assert not set(br.frame["top_station"].unique()) & hidden


def test_detection_rule_requires_persistence():
    frame = pd.DataFrame({
        "window": [0, 1, 2, 3, 4],
        "top_station": [5, 5, 5, 30, 5],
        "score": [10.0, 0.0, 10.0, 10.0, 10.0],
    })
    out = BL.apply_detection_rule(frame, threshold=5.0, persistence=2)
    # window 0 alone cannot satisfy a 2-window persistence requirement
    assert not out.loc[0, "detected"]
    # window 3 jumps 25 stations away, breaking the run
    assert not out.loc[3, "detected"]


def test_threshold_calibration_hits_its_target():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "window": np.arange(2000),
        "top_station": 0,
        "score": rng.normal(0, 1, 2000),
    })
    thr = BL.calibrate_threshold(frame, 0.01)
    assert (frame["score"] >= thr).mean() == pytest.approx(0.01, abs=0.005)


# ---------------------------------------------------------------------- views


def test_coverage_view_holds_physics_fixed(line):
    """A restricted view must change only what is visible, never the ground truth."""
    sim_line = full_observability(line)
    res = simulate(sim_line, SC.nominal_run(400), seed=8)
    v25 = apply_coverage(line, 0.25, seed=3)
    view = telemetry_view(res, v25, sim_line)

    pd.testing.assert_frame_equal(view.passes, res.passes)
    assert set(view.telemetry["station"].unique()) == set(v25.observed_indices)
    assert len(view.telemetry) < len(res.telemetry)


# ------------------------------------------------- head-of-line vs supply


def _shadow_result_at(line, station, d_blocked, d_starved):
    from rippletwin.twin.shadow import ShadowResult

    n = line.n_stations
    db = np.zeros(n)
    ds = np.zeros(n)
    for i, v in d_blocked.items():
        db[i] = v
    for i, v in d_starved.items():
        ds[i] = v
    return ShadowResult(
        window=1, posterior={}, top_station=station, top_prob=0.85,
        group=[station], group_prob=0.85, llr=200.0,
        amp_starve=0.5, amp_block=0.1,
        top_is_hidden=line.stations[station].is_hidden,
        detected=True, confident=True,
        evidence={"p_null": 0.0, "p_line_supply": 0.0,
                  "d_blocked": db, "d_starved": ds,
                  "z_proc": np.zeros(n), "station_post": np.zeros(n)},
    )


def test_head_of_line_supply_is_not_blamed_on_a_station(line):
    """A material delay must not be attributed to the second station.

    Regression test for a real failure: a supply interruption was attributed to
    S02 with 85% confidence while S01 sat starved at 179% of takt -- which
    cannot happen if S02 is the constraint, because S01 would be *blocked*.
    """
    from rippletwin.recommend.engine import ACTION_CHECK_SUPPLY

    sr = _shadow_result_at(
        line, station=1,
        d_blocked={0: -0.05},   # S01 is NOT backing up
        d_starved={0: 1.79, 2: 1.5, 3: 1.3},  # S01 is starved: nothing arriving
    )
    rec = recommend_flow(line, sr, forecast_ripple(line, 1, 78.0))
    assert rec.action == ACTION_CHECK_SUPPLY
    assert rec.abstained


def test_head_of_line_station_is_blamed_when_upstream_is_blocked(line):
    """The same position, but with S01 blocked, IS a station fault."""
    from rippletwin.recommend.engine import ACTION_INSPECT

    sr = _shadow_result_at(
        line, station=1,
        d_blocked={0: 0.12},    # S01 has work it cannot hand over
        d_starved={2: 0.35, 3: 0.30, 4: 0.28},
    )
    rec = recommend_flow(line, sr, forecast_ripple(line, 1, 78.0))
    assert rec.action == ACTION_INSPECT
    assert not rec.abstained


def test_line_supply_hypothesis_decays_from_the_head(line):
    """Supply starvation must be modelled as decaying, not uniform.

    Modelled as uniform, a decaying station hypothesis always fits real supply
    starvation better, and the material delay gets blamed on a station.
    """
    from rippletwin.twin.shadow import ShadowConfig, ShadowSensor

    s = ShadowSensor(line, ShadowConfig())
    assert hasattr(s, "head_starve")
    hs = s.head_starve
    assert hs[0] == pytest.approx(1.0)
    assert np.all(np.diff(hs) <= 1e-12), "must decay monotonically from the head"
    assert hs[-1] < 0.5, "must actually attenuate along the line"


def test_warmup_skips_the_line_fill_transient(line):
    """Windows must not start before the line has filled.

    During a cold start every station is starved simply because production has
    not reached it, which is not a fault and must not be scored as one.
    """
    from rippletwin.features.windows import WindowSpec

    spec = WindowSpec.for_line(line)
    assert spec.warmup >= line.n_stations
    wins = spec.windows(1200)
    assert wins[0][0] == spec.warmup
