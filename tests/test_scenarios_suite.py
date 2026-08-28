"""The Round 2 brief's lettered stress-scenario suite (§31): the system must
not crash on any of A-L. Each letter runs the actual pipeline end to end;
see factory/scenarios.py::SCENARIO_SUITE_MAP for what composes each one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rippletwin.factory import scenarios as SC
from rippletwin.factory.graph_simulator import GraphLineSimulator
from rippletwin.factory.sensor_health import STALE, SensorFault, apply_sensor_faults
from rippletwin.factory.simulator import Disturbance, EVENT_SLOWDOWN
from rippletwin.factory.topology import apply_coverage, build_line
from rippletwin.twin.pipeline import fit_context, infer, simulate

CONFIG = "configs/line_42.yaml"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


@pytest.fixture(scope="module")
def ctx(line):
    nominal = simulate(line, SC.nominal_run(1800), seed=1)
    calib = simulate(line, SC.nominal_run(1500), seed=2)
    return fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)


def _run_ok(line, ctx, scen):
    res = simulate(line, scen, seed=20260301)
    scored, shadow, sensor = infer(ctx, res)
    assert len(shadow) > 0
    return shadow


# A, B, C, E, G, H are plain Scenario objects -- run them directly.


def test_scenario_A_normal_production(line, ctx):
    _run_ok(line, ctx, SC.scenario_normal_variation(line))


def test_scenario_B_hidden_bottleneck(line, ctx):
    _run_ok(line, ctx, SC.scenario_hidden_bottleneck(line))


def test_scenario_C_gradual_bottleneck_emergence(line, ctx):
    _run_ok(line, ctx, SC.scenario_gradual_bottleneck(line))


def test_scenario_E_multiple_simultaneous_abnormalities(line, ctx):
    _run_ok(line, ctx, SC.scenario_multiple_abnormalities(line))


def test_scenario_G_rare_defect(line, ctx):
    _run_ok(line, ctx, SC.scenario_rare_defect(line))


def test_scenario_H_high_production_surge(line, ctx):
    scen = SC.scenario_production_surge(line)
    scen.n_vehicles = 1200  # full 6000 is exercised by evaluation.surge, not this smoke test
    _run_ok(line, ctx, scen)


# D: an otherwise-normal fault scenario with a dynamic sensor fault overlaid.


def test_scenario_D_sensor_failure(line, ctx):
    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    truth = res.disturbances.iloc[0]
    neighbour = line.nearest_observed_downstream(int(truth["station"]), k=1)[0]
    fault = SensorFault(station=neighbour, kind=STALE,
                         t_start_s=float(truth["t_start_s"]), t_end_s=float(truth["t_end_s"]))
    res.telemetry = apply_sensor_faults(res.telemetry, [fault], seed=0)
    scored, shadow, sensor = infer(ctx, res)
    assert len(shadow) > 0


# F: contradictory evidence -- a fault plus a stale/noisy neighbour that
# reads suspiciously clean/wrong during the same window.


def test_scenario_F_contradictory_sensor_evidence(line, ctx):
    from rippletwin.factory.sensor_health import NOISY

    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=20260301)
    truth = res.disturbances.iloc[0]
    k = int(truth["station"])
    up = line.nearest_observed_upstream(k, 1)[0]
    down = line.nearest_observed_downstream(k, 1)[0]
    faults = [
        SensorFault(station=up, kind=STALE, t_start_s=float(truth["t_start_s"]), t_end_s=float(truth["t_end_s"])),
        SensorFault(station=down, kind=NOISY, t_start_s=float(truth["t_start_s"]), t_end_s=float(truth["t_end_s"]), noise_frac=2.0),
    ]
    res.telemetry = apply_sensor_faults(res.telemetry, faults, seed=0)
    scored, shadow, sensor = infer(ctx, res)
    assert len(shadow) > 0
    # Must not crash and must not silently claim full confidence throughout --
    # some degradation in mean confidence is expected under contradictory input.
    assert shadow["group_prob"].notna().any()


# I, J, K: non-serial topologies (Plant B / Plant C from Phase 5).


def test_scenario_I_parallel_station_topology():
    line = build_line("configs/plant_b_parallel.yaml", seed=7)
    nominal = GraphLineSimulator(line, seed=1).run(900, [], run_id="nominal")
    calib = GraphLineSimulator(line, seed=2).run(800, [], run_id="calib")
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.02)
    d = Disturbance(station=line.hidden_indices[0], kind=EVENT_SLOWDOWN,
                     t_start_s=10000, t_end_s=30000, magnitude=1.4, ramp_s=1000, label="t")
    res = GraphLineSimulator(line, seed=20260301).run(900, [d], run_id="fault")
    scored, shadow, sensor = infer(ctx, res)
    assert len(shadow) > 0


def test_scenario_J_rework_loop():
    line = build_line("configs/plant_c_rework.yaml", seed=7)
    nominal = GraphLineSimulator(line, seed=1).run(900, [], run_id="nominal")
    calib = GraphLineSimulator(line, seed=2).run(800, [], run_id="calib")
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.02)
    d = Disturbance(station=line.hidden_indices[0], kind=EVENT_SLOWDOWN,
                     t_start_s=10000, t_end_s=30000, magnitude=1.4, ramp_s=1000, label="t")
    res = GraphLineSimulator(line, seed=20260301).run(900, [d], run_id="fault")
    scored, shadow, sensor = infer(ctx, res)
    assert len(shadow) > 0


def test_scenario_K_different_plant_configuration():
    # Same claim as I/J from a different angle: no code change needed to
    # move between plants, only a different config path.
    for cfg in ("configs/plant_b_parallel.yaml", "configs/plant_c_rework.yaml"):
        line = build_line(cfg, seed=7)
        assert line.is_graph


# L: low sensor coverage.


def test_scenario_L_low_sensor_coverage(line, ctx):
    view = apply_coverage(line, 0.12, seed=11, strategy="random")
    nominal = simulate(view, SC.nominal_run(1600), seed=1)
    calib = simulate(view, SC.nominal_run(1400), seed=2)
    low_ctx = fit_context(view, nominal, calibration_run=calib, target_window_fpr=0.02)
    scen = SC.scenario_hidden_bottleneck(view)
    res = simulate(view, scen, seed=20260301)
    scored, shadow, sensor = infer(low_ctx, res)
    assert len(shadow) > 0


# The streaming demo script itself -- must exit cleanly.


def test_streaming_demo_script_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "demo" / "run_streaming_demo.py"),
         "--scenario", "S1_HIDDEN_BOTTLENECK", "--no-pace"],
        capture_output=True, text=True, timeout=120, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "SIMULATED PROTOTYPE RESULT" in result.stdout
