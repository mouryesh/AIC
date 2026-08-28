"""Tests for the early bottleneck prediction layer (twin.predict).

Checks, in order:
  1. calibrate() produces a watch threshold strictly below the detect
     threshold, on the same held-out null distribution.
  2. the state machine responds correctly to synthetic llr trajectories
     (rising -> elevated, flat -> normal, falling from elevated -> recovering).
  3. on the dedicated gradual-ramp scenario, the predictor reaches an elevated
     state (WATCH/DEGRADING/PREDICTED_CONSTRAINT) before the constraint
     actually binds, i.e. genuine lead time exists to measure.
  4. true_bottleneck_onset agrees with common sense on edge cases.
  5. the early-warning experiment runs end to end and produces a summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rippletwin.evaluation import metrics as M
from rippletwin.evaluation.early_warning import EarlyWarningConfig, run_early_warning_experiment
from rippletwin.factory import scenarios as SC
from rippletwin.factory.topology import build_line
from rippletwin.twin import predict as PR
from rippletwin.twin.pipeline import fit_context, infer, simulate

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


@pytest.fixture(scope="module")
def ctx(line):
    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    return fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)


# ------------------------------------------------------------------ calibration


def test_watch_threshold_is_looser_than_detect_threshold(ctx):
    cal = ctx.calibration
    assert cal["watch_llr"] < cal["detect_llr"]
    assert cal["watch_llr"] > 0
    assert ctx.shadow_cfg.watch_llr == cal["watch_llr"]
    assert ctx.shadow_cfg.llr_noise_std >= 0


# ------------------------------------------------------------------ state machine


def _fake_shadow_df(llrs, detect_llr=8.0, watch_llr=4.0, station=5, hidden=True):
    n = len(llrs)
    return pd.DataFrame(
        {
            "window": np.arange(n),
            "t_mid_s": np.arange(n) * 100.0,
            "v_start": np.arange(n) * 20,
            "v_end": np.arange(n) * 20 + 20,
            "top_station": [station] * n,
            "top_station_id": [f"S{station + 1:02d}"] * n,
            "top_prob": [0.5] * n,
            "group_prob": [0.5] * n,
            "llr": llrs,
            "amp_starve": [0.1] * n,
            "amp_block": [0.0] * n,
            "top_is_hidden": [hidden] * n,
            "detected": [l >= detect_llr for l in llrs],
            "confident": [l >= detect_llr for l in llrs],
        }
    )


def _empty_telemetry_and_scored():
    return pd.DataFrame(columns=["station", "vehicle_id", "t_depart_s"]), pd.DataFrame(
        columns=["window", "station", "proc_time_s_mean"]
    )


def _cfg(detect_llr=8.0, watch_llr=4.0, noise=0.7):
    from rippletwin.twin.shadow import ShadowConfig

    return ShadowConfig(detect_llr=detect_llr, watch_llr=watch_llr, llr_noise_std=noise)


def test_flat_low_llr_stays_normal():
    shadow = _fake_shadow_df([0.1, 0.2, 0.1, 0.15, 0.1, 0.2, 0.1])
    tel, scored = _empty_telemetry_and_scored()
    out = PR.run_predictor(shadow, _line_stub(), tel, scored, _cfg())
    assert (out["state"] == PR.STATE_NORMAL).all()


def test_rising_llr_below_watch_reaches_degrading():
    shadow = _fake_shadow_df([0.1, 0.5, 1.2, 2.0, 2.8, 3.5])
    tel, scored = _empty_telemetry_and_scored()
    out = PR.run_predictor(shadow, _line_stub(), tel, scored, _cfg(watch_llr=4.0))
    assert PR.STATE_DEGRADING in set(out["state"])
    assert (out["state"] == PR.STATE_NORMAL).sum() < len(out)


def test_llr_crossing_watch_then_detect_reaches_predicted_or_active():
    shadow = _fake_shadow_df([0.1, 1.0, 3.0, 4.5, 6.0, 9.0, 9.5])
    tel, scored = _empty_telemetry_and_scored()
    out = PR.run_predictor(shadow, _line_stub(), tel, scored, _cfg(detect_llr=8.0, watch_llr=4.0))
    states = list(out["state"])
    # monotonically escalating evidence should pass through WATCH before
    # ever reaching a confident/detected state
    assert PR.STATE_WATCH in states
    watch_idx = states.index(PR.STATE_WATCH)
    detected_idx = next(
        (i for i, s in enumerate(states) if s in (PR.STATE_PREDICTED_CONSTRAINT, PR.STATE_ACTIVE_BOTTLENECK)),
        None,
    )
    if detected_idx is not None:
        assert watch_idx < detected_idx


def test_declining_after_elevated_reaches_recovering():
    shadow = _fake_shadow_df(
        [0.1, 3.0, 5.0, 6.0, 6.0, 5.5, 4.5, 3.0, 1.8, 1.0, 0.5, 0.3]
    )
    tel, scored = _empty_telemetry_and_scored()
    out = PR.run_predictor(shadow, _line_stub(), tel, scored, _cfg(detect_llr=8.0, watch_llr=4.0))
    assert PR.STATE_RECOVERING in set(out["state"])


def _line_stub():
    """A minimal object satisfying the .stations[i].is_hidden / .takt_s
    surface run_predictor actually touches, without paying for build_line."""

    class _Station:
        def __init__(self, hidden):
            self.is_hidden = hidden

    class _Line:
        takt_s = 60.0
        stations = [_Station(True)] * 50

        def nearest_observed_downstream(self, index, k=1):
            return []  # no observed neighbour -> infer_hidden_cycle_time returns None cleanly

    return _Line()


# ------------------------------------------------------------------ true onset


def test_true_bottleneck_onset_weak_magnitude_never_binds(line):
    truth = M.EpisodeTruth(
        has_fault=True, station=5, kind="SLOWDOWN", t_start_s=1000.0, t_end_s=5000.0,
        ramp_s=1000.0, magnitude=1.001, source_is_hidden=True, board_moment_s=None,
    )
    assert M.true_bottleneck_onset(line, truth) is None


def test_true_bottleneck_onset_strong_magnitude_binds_within_ramp(line):
    truth = M.EpisodeTruth(
        has_fault=True, station=5, kind="SLOWDOWN", t_start_s=1000.0, t_end_s=90_000.0,
        ramp_s=1800.0, magnitude=1.5, source_is_hidden=True, board_moment_s=None,
    )
    onset = M.true_bottleneck_onset(line, truth)
    assert onset is not None
    assert 1000.0 <= onset <= 1000.0 + 1800.0


# ------------------------------------------------------------------ integration


def test_predictor_warns_before_the_constraint_binds_on_s6(line, ctx):
    scen = SC.scenario_gradual_bottleneck(line, seed=606)
    res = simulate(line, scen, seed=6000)
    scored, shadow, _ = infer(ctx, res)
    pred = PR.run_predictor(shadow, line, res.telemetry, scored, ctx.shadow_cfg)
    assert len(pred) > 0

    truth = M.episode_truth(res, line)
    onset = M.true_bottleneck_onset(line, truth)
    assert onset is not None, "S6 must be parameterised to eventually bind"

    elevated = pred[pred["state"].isin(("WATCH", "DEGRADING", "PREDICTED_CONSTRAINT"))]
    elevated_before_onset = elevated[elevated["t_mid_s"] < onset]
    assert len(elevated_before_onset) > 0, (
        "predictor never showed elevated risk before the constraint actually bound"
    )


def test_early_warning_experiment_runs_end_to_end(tmp_path):
    cfg = EarlyWarningConfig(n_random_episodes=3)
    out = run_early_warning_experiment(cfg, out_dir=tmp_path, verbose=False)
    summary = out["summary"]
    assert summary["n_episodes"] == 6 + 3
    for key in ("miss_rate", "false_alarm_rate", "median_lead_time_min"):
        assert key in summary
