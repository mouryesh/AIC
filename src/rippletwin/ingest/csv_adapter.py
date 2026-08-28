"""Reading a plant's own export into ``PlantData``, without touching our code.

Why a mapping file rather than a schema
---------------------------------------
No plant names its columns the way we do. A historian export has ``Equipment``,
``TagName``, ``EventTime``; an MES traceability view has ``SERIAL_NO`` and
``OP_CODE``. The naive answer is "send us data in this format", which pushes an
ETL project onto the customer before they have seen any value, and is a good way
for a pilot to die during procurement.

So the plant's column names live in a YAML file and our code never changes. Two
input shapes are supported, because plants have one or the other:

**Shape A -- state log + VIN scans (the common case).** A PLC or OEE historian
holds state changes; MES holds VIN reads. We join them with
``rippletwin.ingest.states``. This is the realistic path.

**Shape B -- pre-joined telemetry.** Some plants already compute per-unit dwell
in a reporting layer. If they hand us that, we skip the join.

Either way the output is a ``PlantData``, and everything downstream is identical
to a simulated run.

A note on what this does not do
-------------------------------
This reads files. It does not talk to OPC UA, MQTT Sparkplug or a historian API
directly, and pretending otherwise would be dishonest about the state of the
prototype. A file drop is deliberately the *first* integration step: it needs no
firewall change, no OT credentials and no vendor involvement, so a plant can
evaluate the method from an export before anyone opens a port. Streaming is the
Phase 2 conversation, not the Phase 0 one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import yaml

from .plant_data import PlantData
from .states import attribute_states_to_vehicles, close_state_intervals


@dataclass
class PlantExportSpec:
    """How one plant's files map onto what the twin needs."""

    root: str
    files: Dict[str, str] = field(default_factory=dict)
    columns: Dict[str, Dict[str, str]] = field(default_factory=dict)
    takt_s: Optional[float] = None
    n_stations: Optional[int] = None
    clock_sync_s: Optional[float] = None
    #: Multiply every duration/timestamp by this to reach seconds. A historian
    #: exporting milliseconds is common and silently catastrophic.
    time_scale: float = 1.0
    plant_name: str = ""
    #: Every station on the line, in process order, instrumented or not.
    #: The single most valuable optional input: it removes the guesswork about
    #: WHERE the blind stations sit, and a plant can answer it from its
    #: equipment list without a survey.
    station_order: Optional[List[str]] = None

    @classmethod
    def from_yaml(cls, path: str) -> "PlantExportSpec":
        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        line = cfg.get("line", {}) or {}
        return cls(
            root=cfg.get("root") or os.path.dirname(os.path.abspath(path)),
            files=cfg.get("files", {}) or {},
            columns=cfg.get("columns", {}) or {},
            takt_s=line.get("takt_s"),
            n_stations=line.get("n_stations"),
            clock_sync_s=line.get("clock_sync_s"),
            time_scale=float(cfg.get("time_scale", 1.0)),
            plant_name=cfg.get("plant_name", ""),
            station_order=(
                [str(x) for x in line["stations"]] if line.get("stations") else None
            ),
        )

    def path(self, key: str) -> Optional[str]:
        name = self.files.get(key)
        if not name:
            return None
        p = name if os.path.isabs(name) else os.path.join(self.root, name)
        return p if os.path.exists(p) else None

    def read(self, key: str) -> Optional[pd.DataFrame]:
        """Read one file and rename its columns into our vocabulary."""
        p = self.path(key)
        if p is None:
            return None
        df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        mapping = self.columns.get(key, {}) or {}
        # The YAML is written plant-column -> our-column, which is the direction
        # a plant engineer finds natural to fill in.
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        return df


def _epoch_of(df: Optional[pd.DataFrame], col: str) -> Optional[pd.Timestamp]:
    """Earliest timestamp in one frame's time column, if it is a timestamp."""
    if df is None or col not in df.columns:
        return None
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return None
    s = pd.to_datetime(s, errors="coerce", format="mixed")
    return s.min() if s.notna().any() else None


def _to_seconds(
    df: pd.DataFrame,
    col: str,
    scale: float,
    epoch: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Normalise a time column to float seconds from a *shared* epoch.

    The shared epoch is the whole point. An earlier version subtracted each
    file's own minimum timestamp, which looks reasonable and is badly wrong: a
    state log and a VIN-scan log do not begin at the same instant, so each frame
    was shifted by a different amount and the two were silently knocked out of
    alignment with each other. Durations still looked plausible -- they are
    differences, so a constant offset cancels -- but the interval join that
    attributes a BLOCKED state to a specific unit was reading from the wrong
    occupancy window.

    It cost 8 seconds of mean error per row on ``blocked_s`` and moved the
    localisation two stations. Nothing raised an exception at any point, which
    is exactly why the epoch is now computed once across every file and passed
    in.
    """
    if col not in df.columns:
        return df
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_datetime(s, errors="coerce", format="mixed")
        base = epoch if epoch is not None else s.min()
        s = (s - base).dt.total_seconds()
    else:
        s = s.astype(float) * scale
    out = df.copy()
    out[col] = s
    return out


def normalise_identifiers(
    telemetry: pd.DataFrame,
    vehicles: pd.DataFrame,
    inspections: Optional[pd.DataFrame] = None,
    station_order: Optional[List[str]] = None,
) -> tuple:
    """Replace plant identifiers with the positional indices the model needs.

    A plant identifies a unit by VIN (``WVW00005999``) and a station by an
    equipment code (``EQ-031``). The twin needs a *build-sequence position* and
    a *process-order position*, because windowing walks the build sequence and
    the propagation model walks the line. Those are different things, and
    conflating them is a real integration bug rather than a cosmetic one: sorting
    VINs alphabetically would silently scramble the build order.

    So the original identifiers are preserved as ``vehicle_key`` and
    ``station_key`` -- a work order has to name the equipment the plant calls it,
    not our index -- and integer positions are derived:

    * ``vehicle_id`` -- rank by first appearance on the line, i.e. build order
    * ``station`` -- rank by median departure time, i.e. process order

    When ``station_order`` lists every station on the line -- instrumented or
    not -- a station's index is its position in that list, so the blind stations
    occupy their real positions. Without it, only the instrumented stations can
    be ranked, and the blind ones have to be guessed at; that guess moved a
    localisation by twelve stations in testing, which is why the pilot report
    prints it as an assumption to confirm.

    Returns ``(telemetry, vehicles, inspections, station_map, vehicle_map)``.
    """
    tel = telemetry.copy()

    if station_order:
        station_map = {k: i for i, k in enumerate(station_order)}
        seen = set(tel["station"].astype(str).unique())
        unknown = sorted(seen - set(station_map))
        if unknown:
            raise ValueError(
                "these stations appear in the export but not in line.stations: "
                + ", ".join(unknown[:10])
            )
        tel["station"] = tel["station"].astype(str)
    else:
        # Median departure time along a serial line recovers the order of the
        # stations we can see. Verified exactly by round-trip against a known
        # line -- but it can only place the instrumented ones.
        order = (
            tel.groupby("station", observed=True)["t_depart_s"].median().sort_values()
        )
        station_map = {k: i for i, k in enumerate(order.index)}
    tel["station_key"] = tel["station"]
    tel["station"] = tel["station_key"].map(station_map)

    # Build order: when each unit first appears anywhere on the line.
    first_seen = (
        tel.groupby("vehicle_key" if "vehicle_key" in tel.columns else "vehicle_id",
                    observed=True)["t_depart_s"].min().sort_values()
    )
    vehicle_map = {k: i for i, k in enumerate(first_seen.index)}
    tel["vehicle_key"] = tel["vehicle_id"]
    tel["vehicle_id"] = tel["vehicle_key"].map(vehicle_map)

    veh = vehicles.copy()
    veh["vehicle_key"] = veh["vehicle_id"]
    veh["vehicle_id"] = veh["vehicle_key"].map(vehicle_map)
    # A unit in the build sequence that never appeared on the line (not yet
    # started, or a scan gap) has no position and cannot be windowed.
    veh = veh[veh["vehicle_id"].notna()].copy()
    veh["vehicle_id"] = veh["vehicle_id"].astype(int)

    insp = None
    if inspections is not None and not inspections.empty:
        insp = inspections.copy()
        insp["vehicle_key"] = insp["vehicle_id"]
        insp["vehicle_id"] = insp["vehicle_key"].map(vehicle_map)
        insp = insp[insp["vehicle_id"].notna()].copy()
        insp["vehicle_id"] = insp["vehicle_id"].astype(int)
        if "station" in insp.columns:
            insp["station_key"] = insp["station"]
            insp["station"] = insp["station_key"].map(station_map)

    tel = tel[tel["vehicle_id"].notna()].copy()
    tel["vehicle_id"] = tel["vehicle_id"].astype(int)
    tel["station"] = tel["station"].astype(int)
    return tel, veh, insp, station_map, vehicle_map


def load_plant_export(spec: PlantExportSpec) -> PlantData:
    """Build a ``PlantData`` from a plant's files.

    Prefers pre-joined telemetry when present, otherwise joins the state log to
    the VIN scans. Missing optional inputs are left absent rather than faked --
    ``PlantData.validate()`` reports what that costs.
    """
    vehicles = spec.read("vehicles")
    inspections = spec.read("inspections")
    environment = spec.read("environment")
    telemetry = spec.read("telemetry")
    states = spec.read("states")
    scans = spec.read("scans")

    # One epoch for the whole export. Every file must be measured from the same
    # zero or the cross-file joins below are meaningless -- see _to_seconds.
    candidates = [
        _epoch_of(telemetry, "t_depart_s"), _epoch_of(telemetry, "t_start_s"),
        _epoch_of(states, "t_s"), _epoch_of(scans, "t_s"),
        _epoch_of(environment, "t_s"),
    ]
    candidates = [c for c in candidates if c is not None]
    epoch = min(candidates) if candidates else None

    if telemetry is not None:
        telemetry = _to_seconds(telemetry, "t_depart_s", spec.time_scale, epoch)
        telemetry = _to_seconds(telemetry, "t_start_s", spec.time_scale, epoch)
        for c in ("blocked_s", "starved_s", "proc_time_s"):
            if c in telemetry.columns:
                telemetry[c] = telemetry[c].astype(float) * spec.time_scale
    else:
        if states is None or scans is None:
            raise FileNotFoundError(
                "need either a 'telemetry' file, or both 'states' and 'scans'. "
                f"found: {sorted(k for k in spec.files if spec.path(k))}"
            )
        states = _to_seconds(states, "t_s", spec.time_scale, epoch)
        scans = _to_seconds(scans, "t_s", spec.time_scale, epoch)
        closed = close_state_intervals(states)
        telemetry = attribute_states_to_vehicles(closed, scans)

    if vehicles is None:
        # Derivable: the build sequence is implied by the order units entered
        # the line. Weaker than an MES view (no variant, no shift) but enough
        # to run the flow path, and better than refusing to start.
        first = (
            telemetry.sort_values("t_depart_s")
            .groupby("vehicle_id", observed=True)["t_depart_s"]
            .min()
            .sort_values()
        )
        vehicles = pd.DataFrame({"vehicle_id": first.index.to_numpy()})

    if environment is not None:
        environment = _to_seconds(environment, "t_s", spec.time_scale)

    telemetry, vehicles, inspections, station_map, vehicle_map = (
        normalise_identifiers(telemetry, vehicles, inspections,
                              station_order=spec.station_order)
    )

    return PlantData.from_frames(
        telemetry=telemetry,
        vehicles=vehicles,
        inspections=inspections,
        environment=environment,
        meta={
            "source": "plant_export",
            "plant_name": spec.plant_name,
            "root": spec.root,
            "takt_s": spec.takt_s,
            "n_stations": spec.n_stations,
            # Kept so a finding can be reported using the plant's own equipment
            # code rather than our positional index.
            "station_map": station_map,
            "station_order_supplied": bool(spec.station_order),
            "n_units": len(vehicle_map),
        },
    )


#: A worked example, written out by ``rippletwin.pilot --emit-template``.
#: Deliberately uses the ugly column names a real export actually has.
TEMPLATE_YAML = """\
# RippleTwin plant export mapping.
#
# Fill in the left-hand side with YOUR column names. Do not rename your files
# or your columns -- that is what this file is for.
plant_name: "Example Assembly Plant, Line 3"

line:
  takt_s: 60           # planned takt, seconds per unit
  n_stations: 42       # total stations on the line, instrumented or not
  clock_sync_s: 1.0    # worst-case clock skew between station time sources

# Set to 0.001 if your durations/timestamps are in milliseconds.
time_scale: 1.0

files:
  # Shape A (usual): PLC/OEE state changes + MES VIN reads.
  states: plc_state_log.csv
  scans: vin_scans.csv
  # Shape B (if your reporting layer already joins them, use this instead
  # and delete states/scans above):
  # telemetry: station_cycle_report.csv
  vehicles: build_sequence.csv
  inspections: gate_results.csv     # optional -- enables defect attribution
  # environment: ambient.csv        # optional

columns:
  states:
    Equipment: station        # your station identifier column
    StateCode: state          # RUNNING / BLOCKED / STARVED / FAULT
    EventTime: t_s            # timestamp or seconds
  scans:
    Equipment: station
    SerialNo: vehicle_id      # VIN or serial
    ReadTime: t_s
  vehicles:
    SerialNo: vehicle_id
    ModelCode: variant
    ShiftName: shift
  inspections:
    SerialNo: vehicle_id
    GateCode: gate_id
    Result: result            # PASS / FAIL
    DefectCodes: defect_types # pipe-separated, e.g. "TORQUE_LOW|GAP"
"""
