"""Generate a historian-shaped export, so the pilot path can be run end to end.

Why this exists
---------------
``rippletwin.pilot`` is the command a plant runs on its own data. Until a plant
gives us data, the only way to know whether that command actually works is to
manufacture an export that looks like theirs rather than like ours.

So this writes out what an OEE historian and an MES traceability view would
actually hold -- state *changes*, VIN reads, a build sequence, gate results --
with deliberately awkward column names (``Equipment``, ``SerialNo``,
``EventTime``), timestamps as strings rather than floats, and no pre-joined
per-unit dwell times. Nothing here is in RippleTwin's own vocabulary.

A slowdown is injected at a station with **no telemetry**, and that station is
then dropped from the export entirely, exactly as it would be absent from a
real plant's historian. The pilot has to name it without ever seeing a row
from it.

    python demo/make_plant_export.py
    python -m rippletwin.pilot --export demo/plant_export/mapping.yaml

The ground-truth answer is written to ``ANSWER.txt`` in the export directory,
which the pilot never reads. It is there so a judge can check the result rather
than take our word for it.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rippletwin.factory.simulator import Disturbance, LineSimulator  # noqa: E402
from rippletwin.factory.topology import build_line  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant_export")
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "configs", "line_42.yaml")
SEED = 20260315
N_VEHICLES = 6000
T0 = datetime(2026, 3, 2, 6, 0, 0)


def _stamp(t_s: np.ndarray) -> pd.Series:
    """Seconds -> the ISO strings a historian actually exports."""
    return pd.to_datetime(T0) + pd.to_timedelta(np.asarray(t_s, dtype=float), unit="s")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    line = build_line(CONFIG, seed=7)

    # Pick a blind station away from the head of the line, where the
    # head-of-line/supply ambiguity would otherwise dominate.
    blind = [i for i in line.hidden_indices if 8 <= i <= 34]
    target = blind[len(blind) // 2]
    truth = line.stations[target]

    # A gradual 35% slowdown: the shape of tooling wear or a fixture going out
    # of adjustment, not a hard failure.
    #
    # Placement is deliberate. The pilot splits an export chronologically into
    # baseline / calibration / scored, so a fault has to sit inside the scored
    # portion to be a fair test. An earlier draft of this file started the fault
    # at 55% of the run, which put it almost entirely inside the calibration
    # window -- the twin found nothing, correctly, because it had been told that
    # period was normal. Keep FAULT_START_FRAC above the default split of 0.60.
    FAULT_START_FRAC = 0.72
    FAULT_HOURS = 14
    t_start = N_VEHICLES * line.takt_s * FAULT_START_FRAC
    dist = Disturbance(
        station=target,
        kind="SLOWDOWN",
        t_start_s=t_start,
        t_end_s=t_start + 3600 * FAULT_HOURS,
        magnitude=1.35,
        ramp_s=1800.0,
        label="progressive fixture binding",
    )

    print(f"simulating {N_VEHICLES:,} units, fault at station "
          f"{target} ({truth.station_id}, tier={truth.tier})")
    res = LineSimulator(line, seed=SEED).run(N_VEHICLES, [dist], run_id="pilot")

    tel = res.telemetry
    # The historian only holds instrumented stations. The faulty one is manual,
    # so it is already absent -- assert that rather than assume it.
    assert target not in set(tel["station"]), (
        "the injected station appears in telemetry; pick a MANUAL station"
    )

    # ---- PLC / OEE state-change log -------------------------------------
    ev = []
    for r in tel.itertuples():
        if r.starved_s > 0:
            ev.append((r.station, "STARVED", r.t_start_s - r.starved_s))
        ev.append((r.station, "RUNNING", r.t_start_s))
        if r.blocked_s > 0:
            ev.append((r.station, "BLOCKED", r.t_depart_s - r.blocked_s))
    states = pd.DataFrame(ev, columns=["station", "state", "t_s"])
    states = states.sort_values("t_s", kind="mergesort")
    states = pd.DataFrame({
        "Equipment": [f"EQ-{s:03d}" for s in states["station"]],
        "StateCode": states["state"].to_numpy(),
        "EventTime": _stamp(states["t_s"].to_numpy()),
    })

    # ---- MES VIN traceability -------------------------------------------
    scans = pd.DataFrame({
        "Equipment": [f"EQ-{s:03d}" for s in tel["station"]],
        "SerialNo": [f"WVW{v:08d}" for v in tel["vehicle_id"]],
        "ReadTime": _stamp(tel["t_depart_s"].to_numpy()),
    })

    # ---- Build sequence --------------------------------------------------
    veh = res.vehicles
    vehicles = pd.DataFrame({
        "SerialNo": [f"WVW{v:08d}" for v in veh["vehicle_id"]],
        "ModelCode": veh["variant"].to_numpy(),
        "ShiftName": veh["shift"].to_numpy(),
    })

    # ---- Gate results ----------------------------------------------------
    insp = res.inspections
    gates = pd.DataFrame({
        "SerialNo": [f"WVW{v:08d}" for v in insp["vehicle_id"]],
        "GateCode": insp["gate_id"].to_numpy(),
        "Result": insp["result"].to_numpy(),
        "DefectCodes": insp["defect_types"].fillna("").to_numpy(),
    })

    for name, df in (("plc_state_log.csv", states), ("vin_scans.csv", scans),
                     ("build_sequence.csv", vehicles), ("gate_results.csv", gates)):
        df.to_csv(os.path.join(OUT, name), index=False)
        print(f"  wrote {name:<22} {len(df):>9,} rows")

    # Identifiers in the export are opaque strings, exactly as they would be in
    # a real MES. The mapping file is the only place they are interpreted.
    station_list = "\n".join(f"    - EQ-{i:03d}" for i in range(line.n_stations))
    mapping = f"""\
# Generated by demo/make_plant_export.py -- a stand-in for a real plant export.
# Column names on the left are the PLANT's; RippleTwin's code never changes.
plant_name: "Synthetic Assembly Plant (demo export)"

line:
  takt_s: {line.takt_s}
  n_stations: {line.n_stations}
  clock_sync_s: 0.5
  # Every station in process order, instrumented or not. A plant can answer
  # this from its equipment list; the blind ones simply never appear in the
  # historian. Without it the twin has to guess where the gaps are.
  stations:
{station_list}

time_scale: 1.0

files:
  states: plc_state_log.csv
  scans: vin_scans.csv
  vehicles: build_sequence.csv
  inspections: gate_results.csv

columns:
  states:
    Equipment: station
    StateCode: state
    EventTime: t_s
  scans:
    Equipment: station
    SerialNo: vehicle_id
    ReadTime: t_s
  vehicles:
    SerialNo: vehicle_id
    ModelCode: variant
    ShiftName: shift
  inspections:
    SerialNo: vehicle_id
    GateCode: gate_id
    Result: result
    DefectCodes: defect_types
"""
    with open(os.path.join(OUT, "mapping.yaml"), "w") as fh:
        fh.write(mapping)

    with open(os.path.join(OUT, "ANSWER.txt"), "w") as fh:
        fh.write(
            "GROUND TRUTH -- the pilot never reads this file.\n\n"
            f"Injected fault : station index {target} ({truth.station_id})\n"
            f"Tier           : {truth.tier}  (no telemetry -- absent from the export)\n"
            f"Kind           : {dist.kind}, {dist.magnitude:.2f}x processing time\n"
            f"Label          : {dist.label}\n"
            f"Active         : {dist.t_start_s / 3600:.1f}h to "
            f"{dist.t_end_s / 3600:.1f}h into the run\n"
            f"Ramp           : {dist.ramp_s / 60:.0f} min\n\n"
            "The export contains no row from this station. Naming it is the task.\n"
        )

    print(f"\nexport written to {OUT}")
    print("run:  python -m rippletwin.pilot --export demo/plant_export/mapping.yaml")


if __name__ == "__main__":
    main()
