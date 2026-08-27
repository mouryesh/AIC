"""Tests for the simulator's flow physics.

The entire shadow-sensing claim rests on one physical fact: a slow station
blocks everything upstream of it and starves everything downstream. If the
simulator does not reproduce that, nothing built on top of it means anything.
These tests check the physics directly rather than checking that the model
agrees with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from rippletwin.factory.scenarios import nominal_run
from rippletwin.factory.simulator import (
    Disturbance,
    EVENT_SLOWDOWN,
    LineSimulator,
)
from rippletwin.factory.topology import (
    TIER_MANUAL,
    apply_coverage,
    build_line,
)

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


def test_line_matches_brief_parameters(line):
    """30-50 stations across body, paint and final, with uneven coverage."""
    assert 30 <= line.n_stations <= 50
    assert {s.zone for s in line.stations} == {"BODY", "PAINT", "FINAL"}
    # "a majority well-instrumented, a meaningful minority reliant on manual checks"
    assert 0.5 < line.coverage < 0.9
    assert len(line.hidden_indices) >= 5
    assert len(line.inspection_indices) >= 2


def test_stations_are_ordered_and_unique(line):
    assert [s.index for s in line.stations] == list(range(line.n_stations))
    assert len({s.station_id for s in line.stations}) == line.n_stations


def test_flow_recursion_is_causal(line):
    """A vehicle cannot start at a station before it left the previous one."""
    sim = LineSimulator(line, seed=3)
    res = sim.run(300, [], run_id="causality")
    p = res.passes.sort_values(["vehicle_id", "station"])
    for v in [10, 100, 250]:
        row = p[p["vehicle_id"] == v].sort_values("station")
        starts = row["t_start_s"].to_numpy()
        deps = row["t_depart_s"].to_numpy()
        # start at station i must be >= departure from station i-1
        assert np.all(starts[1:] >= deps[:-1] - 1e-6)
        # departure is never before the work finished
        assert np.all(row["t_depart_s"].to_numpy() >= row["t_end_s"].to_numpy() - 1e-6)


def test_blocked_and_starved_are_non_negative(line):
    sim = LineSimulator(line, seed=4)
    res = sim.run(300, [], run_id="signs")
    assert (res.passes["blocked_s"] >= -1e-6).all()
    assert (res.passes["starved_s"] >= -1e-6).all()


def test_a_station_cannot_process_two_vehicles_at_once(line):
    sim = LineSimulator(line, seed=5)
    res = sim.run(300, [], run_id="exclusion")
    for st in [0, 7, 20, 41]:
        g = res.passes[res.passes["station"] == st].sort_values("vehicle_id")
        starts = g["t_start_s"].to_numpy()
        deps = g["t_depart_s"].to_numpy()
        # vehicle v cannot start before vehicle v-1 has departed
        assert np.all(starts[1:] >= deps[:-1] - 1e-6)


@pytest.mark.parametrize("station", [4, 9, 20, 30])
def test_slowdown_blocks_upstream_and_starves_downstream(line, station):
    """The core physical claim, checked directly on ground truth.

    This is the test that matters most: if it fails, shadow-sensing has no
    mechanism to exploit and the whole approach is unsound.
    """
    sim = LineSimulator(line, seed=11)
    d = Disturbance(
        station=station, kind=EVENT_SLOWDOWN,
        t_start_s=20_000, t_end_s=70_000, magnitude=1.40, ramp_s=600,
    )
    res = sim.run(1400, [d], run_id="physics")
    p = res.passes
    pre = p[(p["t_start_s"] > 6_000) & (p["t_start_s"] < 18_000)]
    dur = p[(p["t_start_s"] > 30_000) & (p["t_start_s"] < 68_000)]

    gp = pre.groupby("station")[["blocked_s", "starved_s"]].mean()
    gd = dur.groupby("station")[["blocked_s", "starved_s"]].mean()
    delta = gd - gp

    zone = line.stations[station].zone
    same_zone = [s.index for s in line.stations if s.zone == zone]
    up = [i for i in same_zone if i < station][-4:]
    dn = [i for i in same_zone if i > station][:4]

    if up:
        assert delta.loc[up, "blocked_s"].mean() > 1.0, (
            f"upstream of {station} did not block"
        )
    if dn:
        assert delta.loc[dn, "starved_s"].mean() > 1.0, (
            f"downstream of {station} did not starve"
        )
    if up and dn:
        # The asymmetry the estimator relies on: upstream blocks, downstream starves.
        assert (
            delta.loc[up, "blocked_s"].mean() > delta.loc[up, "starved_s"].mean()
        )
        assert (
            delta.loc[dn, "starved_s"].mean() > delta.loc[dn, "blocked_s"].mean()
        )


def test_slowdown_reduces_throughput(line):
    sim_a = LineSimulator(line, seed=13)
    clean = sim_a.run(900, [], run_id="clean")
    sim_b = LineSimulator(line, seed=13)
    slow = sim_b.run(
        900,
        [Disturbance(station=10, kind=EVENT_SLOWDOWN, t_start_s=10_000,
                     t_end_s=60_000, magnitude=1.45, ramp_s=300)],
        run_id="slow",
    )
    assert slow.meta["throughput_vph"] < clean.meta["throughput_vph"]


def test_hidden_stations_emit_no_telemetry(line):
    """The blinding must be real: a hidden station leaks nothing."""
    sim = LineSimulator(line, seed=17)
    res = sim.run(400, [], run_id="blind")
    observed = set(res.telemetry["station"].unique())
    for h in line.hidden_indices:
        assert h not in observed, f"hidden station {h} leaked telemetry"
    assert observed == set(line.observed_indices)


def test_ground_truth_is_separate_from_telemetry(line):
    """Ground-truth-only columns must never appear in the model's input."""
    sim = LineSimulator(line, seed=19)
    res = sim.run(300, [], run_id="leak")
    forbidden = {"true_health", "true_defect_mult"}
    assert not (forbidden & set(res.telemetry.columns))


def test_apply_coverage_protects_inspection_gates(line):
    v = apply_coverage(line, 0.25, seed=3)
    assert v.coverage == pytest.approx(0.25, abs=0.03)
    for i in line.inspection_indices:
        assert v.stations[i].tier != TIER_MANUAL
    assert v.stations[0].tier != TIER_MANUAL


def test_defect_profiles_are_valid_distributions(line):
    for s in line.stations:
        assert s.defect_profile
        assert sum(s.defect_profile.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(w >= 0 for w in s.defect_profile.values())


def test_simulation_is_deterministic(line):
    a = LineSimulator(line, seed=42).run(200, [], run_id="a")
    b = LineSimulator(line, seed=42).run(200, [], run_id="b")
    assert a.meta["throughput_vph"] == pytest.approx(b.meta["throughput_vph"])
    assert np.allclose(
        a.passes["t_depart_s"].to_numpy(), b.passes["t_depart_s"].to_numpy()
    )
