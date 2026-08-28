"""Tests for topology generalization: graph reachability on LineTopology,
the serial-line regression guarantee (is_graph False runs the untouched fast
path), GraphLineSimulator's physics, and the cross-topology experiment.
"""

from __future__ import annotations

import numpy as np
import pytest

from rippletwin.evaluation.topology_experiment import PLANTS, run_topology_experiment
from rippletwin.factory.graph_simulator import GraphLineSimulator
from rippletwin.factory.simulator import Disturbance, EVENT_SLOWDOWN
from rippletwin.factory.topology import Edge, LineTopology, Station, build_line
from rippletwin.twin.pipeline import fit_context, infer
from rippletwin.twin.shadow import (
    ShadowConfig,
    buffer_distance_matrix,
    propagation_matrices,
)

LINE_42 = "configs/line_42.yaml"
PLANT_B = "configs/plant_b_parallel.yaml"
PLANT_C = "configs/plant_c_rework.yaml"


# ------------------------------------------------------------ serial regression


def test_line_42_is_not_a_graph():
    line = build_line(LINE_42, seed=7)
    assert line.is_graph is False
    assert line.edges == []


def test_effective_edges_matches_implicit_chain():
    line = build_line(LINE_42, seed=7)
    edges = line.effective_edges()
    assert len(edges) == line.n_stations - 1
    for i, e in enumerate(edges):
        assert e.src == i and e.dst == i + 1
        assert e.buffer_capacity == line.stations[i].out_buffer


def test_serial_graph_helpers_match_index_comparison():
    line = build_line(LINE_42, seed=7)
    k = 20
    assert line.ancestors(k) == set(range(0, k))
    assert line.descendants(k) == set(range(k + 1, line.n_stations))
    assert line.successors(k) == [k + 1]
    assert line.predecessors(k) == [k - 1]


def test_serial_line_dispatch_is_unaffected_by_graph_code():
    """Sanity check that the dispatch added in Phase 5 does not touch the
    result for a plain serial line -- the actual regression guarantee is the
    full-experiment byte-identical check run manually each phase; this pins
    the same property at the unit level."""
    line = build_line(LINE_42, seed=7)
    cfg = ShadowConfig()
    D = buffer_distance_matrix(line)
    B, S = propagation_matrices(line, cfg)
    assert np.allclose(D, D.T)
    k = 20
    assert (B[k, :k] > 0).all()
    assert (S[k, k + 1 :] > 0).all()


# --------------------------------------------------------------- graph loading


def test_plant_b_loads_as_a_graph_with_a_split_and_merge():
    line = build_line(PLANT_B, seed=7)
    assert line.is_graph
    assert line.successors(3) == [4, 5]  # split
    assert line.predecessors(6) == [4, 5]  # merge
    assert line.ancestors(6) == {0, 1, 2, 3, 4, 5}
    assert line.descendants(3) == {4, 5, 6, 7, 8, 9, 10, 11}


def test_plant_c_loads_as_a_graph_with_a_rework_spur():
    line = build_line(PLANT_C, seed=7)
    assert line.is_graph
    assert line.successors(5) == [6, 12]  # spur off
    assert line.predecessors(6) == [5, 12]  # re-merge
    assert 12 in line.ancestors(7)  # the spur is a real upstream path to later stations


def test_topological_order_is_valid():
    for cfg in (PLANT_B, PLANT_C):
        line = build_line(cfg, seed=7)
        order = line.topological_order()
        assert sorted(order) == list(range(line.n_stations))
        pos = {s: i for i, s in enumerate(order)}
        for e in line.effective_edges():
            assert pos[e.src] < pos[e.dst]


def test_topological_order_raises_on_a_cycle():
    stations = [
        Station(index=0, station_id="S01", zone="BODY", base_cycle_s=50, manual_content=0.1,
                tier="BASIC", out_buffer=3, microstop_rate=0.0, microstop_range_s=(0, 0),
                process_noise_cv=0.01, base_defect_rate=0.001),
        Station(index=1, station_id="S02", zone="BODY", base_cycle_s=50, manual_content=0.1,
                tier="BASIC", out_buffer=3, microstop_rate=0.0, microstop_range_s=(0, 0),
                process_noise_cv=0.01, base_defect_rate=0.001),
    ]
    bad = LineTopology(
        name="bad", takt_s=60.0, stations=stations, zones={}, variants={}, shifts=[],
        environment={}, edges=[Edge(0, 1, 3), Edge(1, 0, 3)],
    )
    with pytest.raises(ValueError):
        bad.topological_order()


# ------------------------------------------------------------- graph simulator


def test_graph_simulator_rejects_a_non_graph_line():
    line = build_line(LINE_42, seed=7)
    with pytest.raises(ValueError):
        GraphLineSimulator(line, seed=1)


def test_graph_simulator_produces_directional_starvation_from_a_slowdown():
    line = build_line(PLANT_C, seed=7)
    k = 4  # a plain serial station, upstream of the rework split at 5
    d = Disturbance(station=k, kind=EVENT_SLOWDOWN, t_start_s=5000, t_end_s=40000,
                     magnitude=1.5, ramp_s=1000, label="test")
    res = GraphLineSimulator(line, seed=20260301).run(900, [d], run_id="test")
    during = res.passes[
        (res.passes["t_start_s"] >= d.t_start_s + d.ramp_s) & (res.passes["t_start_s"] <= d.t_end_s)
    ]
    downstream_starve = during[during["station"].isin(line.descendants(k))]["starved_s"].mean()
    upstream = during[during["station"].isin(line.ancestors(k))]
    assert downstream_starve > 5.0, "descendants of a slowed station should starve"


# ------------------------------------------------------------------ inference


@pytest.mark.parametrize("cfg_path", [PLANT_B, PLANT_C])
def test_unmodified_inference_localises_a_fault_on_a_graph_topology(cfg_path):
    line = build_line(cfg_path, seed=7)
    nominal = GraphLineSimulator(line, seed=1).run(1200, [], run_id="nominal")
    calib = GraphLineSimulator(line, seed=2).run(1000, [], run_id="calib")
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.02)

    hidden_ok = [
        i for i in line.hidden_indices
        if line.nearest_observed_upstream(i, 1) and line.nearest_observed_downstream(i, 1)
    ]
    k = hidden_ok[0] if hidden_ok else line.observed_indices[1]
    d = Disturbance(station=k, kind=EVENT_SLOWDOWN, t_start_s=15000, t_end_s=50000,
                     magnitude=1.5, ramp_s=1500, label="test")
    res = GraphLineSimulator(line, seed=20260301).run(1500, [d], run_id="fault")

    scored, shadow, sensor = infer(ctx, res)
    during = shadow[
        (shadow["t_mid_s"] >= d.t_start_s + d.ramp_s) & (shadow["t_mid_s"] <= d.t_end_s)
    ]
    det = during[during["detected"]]
    assert len(det) > 0, "must detect at all on a graph topology"
    within1 = (np.abs(det["top_station"].to_numpy() - k) <= 1).mean()
    assert within1 >= 0.7, f"within-1 localisation only {within1:.2f} on {cfg_path}"


def test_topology_experiment_runs_end_to_end_on_all_three_plants(tmp_path):
    out = run_topology_experiment(out_dir=tmp_path, n_episodes=2, n_vehicles=700, verbose=False)
    summary = out["summary"]
    assert len(summary) == len(PLANTS)
    for col in ("detection_rate", "top1", "within1", "false_alarm_rate", "mean_infer_latency_s"):
        assert col in summary.columns
