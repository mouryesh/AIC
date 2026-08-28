"""Tests for the path a real plant's data takes into the twin.

Every bug these cover was found by running the pilot against a
historian-shaped export rather than against the simulator's own output. None
of them raised an exception at the time -- they produced confident, wrong
answers, which is the failure mode that matters for a system a plant is asked
to trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rippletwin.factory.simulator import Disturbance, LineSimulator
from rippletwin.factory.topology import build_line
from rippletwin.ingest.csv_adapter import normalise_identifiers
from rippletwin.ingest.plant_data import BLOCKER, PlantData
from rippletwin.ingest.states import (
    attribute_states_to_vehicles,
    close_state_intervals,
    discover_topology,
    normalise_state,
    unknown_state_labels,
)
from rippletwin.ingest.topology_infer import infer_line
from rippletwin.twin.pipeline import as_plant_data, fit_context, infer

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


@pytest.fixture(scope="module")
def run(line):
    return LineSimulator(line, seed=11).run(400, [], run_id="ing")


def _historian(tel):
    """Re-express telemetry as the state log + VIN scans a plant would hold."""
    ev = []
    for r in tel.itertuples():
        if r.starved_s > 0:
            ev.append((r.station, "Starved", r.t_start_s - r.starved_s))
        ev.append((r.station, "Running", r.t_start_s))
        if r.blocked_s > 0:
            ev.append((r.station, "Blocked", r.t_depart_s - r.blocked_s))
    states = pd.DataFrame(ev, columns=["station", "state", "t_s"])
    scans = tel[["vehicle_id", "station", "t_depart_s"]].rename(
        columns={"t_depart_s": "t_s"}
    )
    return states, scans


# ------------------------------------------------------- the model boundary


def test_plant_data_cannot_carry_ground_truth(run):
    """The separation is structural, not a convention maintained by care."""
    pd_ = run.as_plant_data()
    fields = set(pd_.__dataclass_fields__)
    assert not fields & {"passes", "disturbances", "defects"}


def test_as_plant_data_is_the_only_door(run):
    assert isinstance(as_plant_data(run), PlantData)
    assert as_plant_data(run.as_plant_data()) is not None
    with pytest.raises(TypeError):
        as_plant_data({"telemetry": None})


# --------------------------------------------------------- the state log join


def test_state_log_round_trip_is_exact(run):
    """A historian export must reconstruct dwell times, not approximate them."""
    tel = run.telemetry
    states, scans = _historian(tel)
    rec = attribute_states_to_vehicles(close_state_intervals(states), scans)

    m = tel[["vehicle_id", "station", "blocked_s", "starved_s"]].merge(
        rec[["vehicle_id", "station", "blocked_s", "starved_s"]],
        on=["vehicle_id", "station"], suffixes=("_t", "_r"),
    )
    # The first unit at each station has no predecessor departure, so its
    # occupancy window is undefined by construction.
    first = m.groupby("station")["vehicle_id"].transform("min")
    m = m[m["vehicle_id"] > first]
    assert len(m) > 5000
    for c in ("blocked_s", "starved_s"):
        assert (m[f"{c}_r"] - m[f"{c}_t"]).abs().max() < 1e-6


def test_processing_time_excludes_starvation(run):
    """Occupancy is starved + processing + blocked, not processing + blocked.

    Subtracting only ``blocked`` inflated processing time by exactly the
    starvation -- residual correlation 1.000 against ground truth.
    """
    tel = run.telemetry
    states, scans = _historian(tel)
    rec = attribute_states_to_vehicles(close_state_intervals(states), scans)
    data = PlantData.from_frames(telemetry=rec, vehicles=run.vehicles)

    m = tel[["vehicle_id", "station", "proc_time_s"]].merge(
        data.telemetry[["vehicle_id", "station", "proc_time_s"]],
        on=["vehicle_id", "station"], suffixes=("_t", "_d"),
    )
    first = m.groupby("station")["vehicle_id"].transform("min")
    m = m[m["vehicle_id"] > first]
    assert (m["proc_time_s_d"] - m["proc_time_s_t"]).abs().max() < 1e-6


def test_unterminated_state_is_truncated_not_extended():
    """An open interval at the end of an export must not invent dwell time."""
    states = pd.DataFrame({
        "station": [0, 0], "state": ["RUNNING", "BLOCKED"], "t_s": [0.0, 10.0],
    })
    closed = close_state_intervals(states, horizon_s=20.0)
    assert closed["t1"].max() == 20.0


def test_unknown_state_labels_are_reported_not_dropped():
    states = pd.DataFrame({
        "station": [0, 0], "state": ["RUNNING", "WEIRD_VENDOR_CODE"],
        "t_s": [0.0, 1.0],
    })
    assert unknown_state_labels(states) == ["WEIRD_VENDOR_CODE"]
    assert normalise_state("Outfeed-Full") == "blocked_s"
    assert normalise_state("INFEED_EMPTY") == "starved_s"


# --------------------------------------------------------------- identifiers


def test_identifiers_map_to_positions_not_sort_order(run):
    """VINs are strings; sorting them alphabetically would scramble build order."""
    tel = run.telemetry.copy()
    tel["vehicle_id"] = [f"VIN{v:06d}" for v in tel["vehicle_id"]]
    tel["station"] = [f"EQ-{s:03d}" for s in tel["station"]]
    veh = run.vehicles.copy()
    veh["vehicle_id"] = [f"VIN{v:06d}" for v in veh["vehicle_id"]]

    t, v, _, smap, vmap = normalise_identifiers(tel, veh)
    assert t["vehicle_id"].dtype.kind == "i"
    assert t["station"].dtype.kind == "i"
    # Originals are preserved: a work order must name the plant's own equipment.
    assert "station_key" in t.columns and "vehicle_key" in t.columns
    assert t["vehicle_id"].min() == 0


def test_supplied_station_order_places_blind_stations(run, line):
    """The station list is what puts a blind station in its real position."""
    tel = run.telemetry.copy()
    tel["station"] = [f"EQ-{s:03d}" for s in tel["station"]]
    veh = run.vehicles
    order = [f"EQ-{i:03d}" for i in range(line.n_stations)]
    t, _, _, smap, _ = normalise_identifiers(tel, veh, station_order=order)
    # A station's index is its position on the line, so the observed indices
    # are the true ones rather than a dense 0..31 ranking.
    assert set(t["station"]) == set(line.observed_indices)


def test_station_absent_from_the_supplied_order_is_an_error(run):
    tel = run.telemetry.copy()
    tel["station"] = [f"EQ-{s:03d}" for s in tel["station"]]
    with pytest.raises(ValueError, match="not in line.stations"):
        normalise_identifiers(tel, run.vehicles, station_order=["EQ-000"])


def test_topology_discovery_recovers_process_order(run, line):
    topo = discover_topology(run.telemetry)
    assert topo["station"].tolist() == sorted(set(run.telemetry["station"]))


# ---------------------------------------------------------------- validation


def test_missing_required_column_is_a_blocker(run):
    tel = run.telemetry.drop(columns=["blocked_s"])
    data = PlantData(telemetry=tel, vehicles=run.vehicles)
    rep = data.validate()
    assert not rep.ok
    assert any(i.code == "MISSING_COLUMNS" for i in rep.blockers)


def test_milliseconds_are_caught(run):
    tel = run.telemetry.copy()
    for c in ("proc_time_s", "blocked_s", "starved_s"):
        tel[c] = tel[c] * 1000.0
    data = PlantData.from_frames(telemetry=tel, vehicles=run.vehicles)
    rep = data.validate(n_stations=42, takt_s=60.0)
    assert any(i.code == "UNITS" for i in rep.issues)


def test_clock_skew_is_a_blocker_when_large(run):
    data = run.as_plant_data()
    rep = data.validate(n_stations=42, takt_s=60.0, clock_sync_s=60.0)
    assert any(i.code == "CLOCK_SKEW" and i.severity == BLOCKER for i in rep.issues)


def test_full_coverage_is_reported_as_not_needing_us(run):
    data = run.as_plant_data()
    n = data.telemetry["station"].nunique()
    rep = data.validate(n_stations=n)
    assert any(i.code == "FULL_COVERAGE" for i in rep.issues)


def test_contract_signals_reflect_what_is_actually_present(run):
    minimal = PlantData.from_frames(
        telemetry=run.telemetry[
            ["vehicle_id", "station", "t_start_s", "t_depart_s",
             "blocked_s", "starved_s"]
        ],
        vehicles=run.vehicles[["vehicle_id"]],
    )
    sig = minimal.contract_signals()
    assert "station_state" in sig
    assert "inspection_results" not in sig
    assert "process_channels" not in sig


# ------------------------------------------------------- degrading gracefully


def test_a_plant_with_no_optional_data_still_runs(line):
    """No conveyor counters, no gate results, no ambient -- the contract says
    all three are optional, so the pipeline must not require them."""
    def minimal(res):
        t = res.telemetry[["vehicle_id", "station", "t_start_s", "t_depart_s",
                           "blocked_s", "starved_s"]].copy()
        v = res.vehicles[["vehicle_id", "variant", "shift"]].copy()
        return PlantData.from_frames(telemetry=t, vehicles=v)

    nom = minimal(LineSimulator(line, seed=1).run(400, [], run_id="n"))
    cal = minimal(LineSimulator(line, seed=2).run(400, [], run_id="c"))
    assert not nom.has_quality_path and not nom.has_environment

    ctx = fit_context(line, nom, calibration_run=cal)
    scored, shadow, sensor = infer(ctx, cal)
    assert len(shadow) > 0
    assert np.isfinite(ctx.shadow_cfg.tau)


def test_inferred_line_reports_its_assumptions(run):
    data = run.as_plant_data()
    _, assumptions = infer_line(data, n_stations=42, takt_s=60.0)
    joined = " ".join(assumptions).lower()
    assert "buffer" in joined
    assert "serial" in joined


def test_supplying_the_station_order_removes_the_placement_guess(run, line):
    data = run.as_plant_data()
    _, guessed = infer_line(data, n_stations=42, takt_s=60.0)
    assert any("evenly" in a for a in guessed)

    tel = run.telemetry.copy()
    order = [f"EQ-{i:03d}" for i in range(line.n_stations)]
    tel["station"] = [f"EQ-{s:03d}" for s in tel["station"]]
    t, v, _, _, _ = normalise_identifiers(tel, run.vehicles, station_order=order)
    d2 = PlantData.from_frames(telemetry=t, vehicles=v)
    line2, named = infer_line(d2, takt_s=60.0, station_order=order)
    assert not any("evenly" in a for a in named)
    assert line2.n_stations == 42
