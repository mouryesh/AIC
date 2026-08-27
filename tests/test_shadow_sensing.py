"""Tests for the shadow-sensing estimator, its calibration, and its silence.

These check three separable things:
  1. the propagation matrices encode the right physical directionality,
  2. the estimator finds a hidden station and stays quiet on a clean line,
  3. the deviation channels are actually calibrated on held-out nominal data.
"""

from __future__ import annotations

import numpy as np
import pytest

from rippletwin.factory import scenarios as SC
from rippletwin.factory.topology import build_line
from rippletwin.twin.pipeline import build_windows, fit_context, infer, simulate
from rippletwin.twin.shadow import (
    ShadowConfig,
    ShadowSensor,
    buffer_distance_matrix,
    infer_hidden_cycle_time,
    propagation_matrices,
)

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


@pytest.fixture(scope="module")
def ctx(line):
    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    return fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)


# ------------------------------------------------------------------ structure


def test_buffer_distance_is_metric_like(line):
    D = buffer_distance_matrix(line)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert (D >= 0).all()


def test_inter_zone_buffer_decouples(line):
    """Crossing a large inter-zone buffer must attenuate far more than a step."""
    D = buffer_distance_matrix(line)
    # within body shop, three stations apart
    within = D[3, 6]
    # across the body -> paint boundary
    across = D[12, 15]
    assert across > within * 2


def test_propagation_directionality(line):
    """Upstream blocks, downstream starves, and the constraint itself does neither."""
    B, S = propagation_matrices(line, ShadowConfig())
    k = 20
    assert (B[k, :k] > 0).all(), "upstream should carry blocking"
    assert (B[k, k:] == 0).all(), "no blocking at or downstream of the constraint"
    assert (S[k, k + 1:] > 0).all(), "downstream should carry starvation"
    assert (S[k, : k + 1] == 0).all(), "no starvation at or upstream of the constraint"


def test_propagation_decays_with_distance(line):
    B, S = propagation_matrices(line, ShadowConfig())
    k = 20
    # monotone decay moving away from the constraint
    up = B[k, max(0, k - 6):k][::-1]
    assert np.all(np.diff(up) <= 1e-12)


# ---------------------------------------------------------------- calibration


def test_deviation_channels_are_calibrated_on_heldout_nominal(line, ctx):
    """Held-out nominal data must score near zero, or everything else is noise.

    This is the test that caught the original design error: a MAD-based z-score
    on zero-inflated starvation time produced a mean of +3 sigma and a 99th
    percentile of 27 on data with nothing wrong with it.
    """
    heldout = simulate(line, SC.nominal_run(1400), seed=99)
    scored = ctx.baseline.score(build_windows(heldout, line, ctx.spec), line)

    for col, tol_mean, tol_p99 in [
        ("z_proc", 0.5, 4.0),
        ("d_blocked", 0.08, 0.45),
        ("d_starved", 0.08, 0.45),
    ]:
        v = scored[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        assert abs(np.mean(v)) < tol_mean, f"{col} biased: mean={np.mean(v):.3f}"
        assert np.percentile(v, 99) < tol_p99, f"{col} tail too heavy"


def test_calibration_applies_correlation_correction(ctx):
    cal = ctx.calibration
    assert 0.0 < cal["tau"] < 1.0, "tau must shrink the evidence, not inflate it"
    assert cal["mean_pairwise_corr"] > 0.1, "stations on a line are correlated"
    assert cal["n_eff"] < cal["n_observed"]
    assert cal["held_out_calibration"] is True


# ----------------------------------------------------------------- behaviour


def test_stays_quiet_on_a_clean_line(line, ctx):
    """Silence on a normal line is half the product."""
    clean = simulate(line, SC.scenario_normal_variation(line), seed=1234)
    _, shadow, _ = infer(ctx, clean)
    assert len(shadow) > 50
    assert shadow["detected"].mean() < 0.05, (
        f"false-alarm rate too high: {shadow['detected'].mean():.3f}"
    )


def test_does_not_blame_a_station_for_a_supply_delay(line, ctx):
    """A line-wide material delay must not be attributed to a station."""
    scen = SC.scenario_variant_shift(line)
    res = simulate(line, scen, seed=1234)
    _, shadow, _ = infer(ctx, res)
    assert shadow["detected"].mean() < 0.10


def test_localises_a_hidden_bottleneck(line, ctx):
    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    _, shadow, _ = infer(ctx, res)

    truth = res.disturbances.iloc[0]
    true_station = int(truth["station"])
    assert line.stations[true_station].is_hidden, "this scenario must target a blind station"

    t0 = float(truth["t_start_s"]) + float(truth["ramp_s"])
    active = shadow[
        shadow["detected"]
        & (shadow["t_mid_s"] >= t0)
        & (shadow["t_mid_s"] <= float(truth["t_end_s"]))
    ]
    assert len(active) > 0, "failed to detect a hidden bottleneck at all"
    err = np.abs(active["top_station"].to_numpy() - true_station)
    assert (err <= 1).mean() >= 0.80, f"within-1 accuracy only {(err <= 1).mean():.2f}"
    assert active["top_is_hidden"].mean() > 0.5


def test_infers_cycle_time_of_an_unmeasured_station(line, ctx):
    """The sharpest falsifiable claim: estimate a number we cannot measure."""
    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    _, shadow, _ = infer(ctx, res)
    truth = res.disturbances.iloc[0]
    k = int(truth["station"])
    t0 = float(truth["t_start_s"]) + float(truth["ramp_s"])
    active = shadow[
        shadow["detected"]
        & (shadow["t_mid_s"] >= t0)
        & (shadow["t_mid_s"] <= float(truth["t_end_s"]))
        & (shadow["top_station"] == k)
    ]
    assert len(active) > 0
    r = active.iloc[len(active) // 2]
    est = infer_hidden_cycle_time(
        line, res.telemetry, k, int(r["v_start"]), int(r["v_end"])
    )
    assert est is not None
    seg = res.passes[
        (res.passes["station"] == k)
        & (res.passes["vehicle_id"] >= int(r["v_start"]))
        & (res.passes["vehicle_id"] < int(r["v_end"]))
    ]
    true_cycle = float(seg["proc_time_s"].mean())
    err_pct = abs(est - true_cycle) / true_cycle * 100
    assert err_pct < 15.0, f"inferred cycle time off by {err_pct:.1f}%"


def test_agrees_with_the_sensor_when_a_sensor_exists(line, ctx):
    scen = SC.scenario_observed_station(line)
    res = simulate(line, scen, seed=20260301)
    _, shadow, _ = infer(ctx, res)
    truth = res.disturbances.iloc[0]
    k = int(truth["station"])
    t0 = float(truth["t_start_s"]) + float(truth["ramp_s"])
    active = shadow[
        shadow["detected"]
        & (shadow["t_mid_s"] >= t0)
        & (shadow["t_mid_s"] <= float(truth["t_end_s"]))
    ]
    assert len(active) > 0
    err = np.abs(active["top_station"].to_numpy() - k)
    assert (err <= 1).mean() >= 0.80


def test_reset_clears_temporal_state(line):
    s = ShadowSensor(line, ShadowConfig())
    s._ewma = np.ones(line.n_stations)
    s._lead_history = [1, 2, 3]
    s.reset()
    assert s._ewma is None
    assert s._lead_history == []
