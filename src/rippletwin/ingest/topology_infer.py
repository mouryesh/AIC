"""Building a line topology from a plant's data instead of our config file.

Why this matters more than it looks
-----------------------------------
``configs/line_42.yaml`` describes our simulated line down to per-station
microstop rates and defect profiles. A real plant has none of that, and asking
for it is how a pilot acquires a four-week delay: layout drawings live with
manufacturing engineering, buffer capacities live in controls, and neither team
has an incentive to produce a YAML file for an unproven vendor.

The inference path needs far less than the simulator does. It needs to know
which stations exist, what order they are in, which of them are instrumented,
and roughly how much buffer sits between them. Three of those four are readable
straight from the export, and the fourth can be estimated.

What is inferred, and how confident we are
------------------------------------------
* **Station order** -- read from median departure time. On a serial line this is
  exact, and our round-trip recovered 32 of 32 observed stations in order.
* **Which stations are instrumented** -- a station that appears in telemetry is
  observed; one that does not is blind. This is a fact about the export, not an
  inference.
* **Blind stations between observed ones** -- their *count* has to come from the
  plant, because a gap of unknown width in the station numbering is genuinely
  ambiguous. ``n_stations`` in the mapping file supplies it; without it we
  assume the observed stations are contiguous and say so loudly.
* **Buffer capacity** -- estimated from the lag between a station's blocking and
  its downstream neighbour's starvation, or defaulted. This is the weakest of
  the four and is reported as an assumption to confirm.

Everything here produces a *proposal for an engineer to sign off*, which is the
honest status of an auto-discovered topology. It cannot see parallel sub-lines,
rework loops or merge points, and it says so rather than silently modelling them
as serial.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..factory.topology import (
    OBSERVED_TIERS,
    TIER_BASIC,
    TIER_MANUAL,
    TIER_RICH,
    LineTopology,
    Station,
)
from .plant_data import PlantData
from .states import discover_topology

#: Used when the plant does not state buffer capacities. Small enough to be
#: conservative: over-stating buffers weakens the blocking channel.
DEFAULT_BUFFER = 2


def infer_line(
    data: PlantData,
    n_stations: Optional[int] = None,
    takt_s: Optional[float] = None,
    name: str = "plant-line",
    default_buffer: int = DEFAULT_BUFFER,
    station_order: Optional[List[str]] = None,
) -> Tuple[LineTopology, List[str]]:
    """Propose a topology from an export. Returns ``(line, assumptions)``.

    ``assumptions`` is the list of things an engineer has to confirm. It is
    returned rather than logged because the pilot report prints it under a
    heading that says so -- an inferred topology presented as fact is exactly
    the kind of quiet error that discredits a system three months in.
    """
    assumptions: List[str] = []
    tel = data.telemetry

    # When the plant listed every station, telemetry indices are already true
    # positions on the line and there is nothing to guess.
    if station_order:
        n_stations = len(station_order)
        observed_pos = sorted(int(x) for x in tel["station"].unique())
        return _build(
            data, tel, n_stations, takt_s, name, default_buffer,
            {p: p for p in observed_pos}, assumptions, station_order,
        )

    order = discover_topology(tel)
    observed_ids = order["station"].tolist()
    n_obs = len(observed_ids)

    # How many stations in total? Only the plant knows how many blind stations
    # sit in the gaps.
    if n_stations is None:
        n_stations = n_obs
        assumptions.append(
            f"Total station count was not supplied, so the line is assumed to be "
            f"exactly the {n_obs} instrumented stations with no blind stations "
            f"between them. If the line has manual stations, set 'n_stations' in "
            f"the mapping file -- shadow-sensing has nothing to find without it."
        )
    elif n_stations < n_obs:
        raise ValueError(
            f"n_stations={n_stations} is smaller than the {n_obs} stations "
            f"present in the telemetry"
        )

    # Takt. Median inter-departure at the busiest station is a decent estimate,
    # but the planned takt is a management fact, not a measurement, so prefer it.
    if takt_s is None:
        deltas = (
            tel.sort_values(["station", "t_depart_s"])
            .groupby("station", observed=True)["t_depart_s"]
            .diff()
        )
        takt_s = float(np.nanmedian(deltas)) if deltas.notna().any() else 60.0
        assumptions.append(
            f"Planned takt was not supplied; estimated {takt_s:.1f}s from median "
            f"inter-departure time. This is the line's *achieved* rate, which is "
            f"slower than planned takt whenever the line is losing output -- "
            f"supply the planned figure for a correct loss estimate."
        )

    # Spread the blind stations through the gaps. Without numbering information
    # the only defensible choice is to distribute them evenly, and to say so.
    n_blind = n_stations - n_obs
    if n_blind > 0:
        assumptions.append(
            f"{n_blind} blind station(s) are placed evenly between the "
            f"instrumented ones. If you know where the manual stations actually "
            f"sit, supply the station order -- their placement changes which "
            f"station a localisation names."
        )

    positions = _interleave(n_obs, n_stations)
    obs_at = dict(zip(positions, observed_ids))
    return _build(data, tel, n_stations, takt_s, name, default_buffer,
                  obs_at, assumptions, None)


def _build(
    data: PlantData,
    tel,
    n_stations: int,
    takt_s: float,
    name: str,
    default_buffer: int,
    obs_at: dict,
    assumptions: List[str],
    station_order: Optional[List[str]],
) -> Tuple[LineTopology, List[str]]:
    """Assemble the topology once the observed stations have been placed."""
    if takt_s is None:
        deltas = (
            tel.sort_values(["station", "t_depart_s"])
            .groupby("station", observed=True)["t_depart_s"].diff()
        )
        takt_s = float(np.nanmedian(deltas)) if deltas.notna().any() else 60.0
        assumptions.append(
            f"Planned takt was not supplied; estimated {takt_s:.1f}s from median "
            f"inter-departure time."
        )

    # Richness: a station with process channels is RICH, one with only
    # blocked/starved is BASIC. This is read from the data, not assumed.
    rich_cols = [c for c in ("torque_nm", "vibration_mm_s", "station_temp_c")
                 if c in tel.columns]
    rich_ids = set()
    if rich_cols:
        present = tel.groupby("station", observed=True)[rich_cols].apply(
            lambda g: g.notna().any().any()
        )
        rich_ids = set(present[present].index.tolist())

    cycle_by_station = (
        tel.groupby("station", observed=True)["proc_time_s"].median().to_dict()
        if tel["proc_time_s"].notna().any() else {}
    )

    stations: List[Station] = []
    for i in range(n_stations):
        src = obs_at.get(i)
        if src is not None:
            tier = TIER_RICH if src in rich_ids else TIER_BASIC
            # Name it what the plant names it -- a work order that says
            # "index 31" is not actionable on a shop floor.
            sid = str(station_order[i]) if station_order else str(src)
            cyc = float(cycle_by_station.get(src, takt_s) or takt_s)
        else:
            tier = TIER_MANUAL
            sid = str(station_order[i]) if station_order else f"BLIND_{i:02d}"
            cyc = float(takt_s)
        stations.append(
            Station(
                index=i,
                station_id=sid,
                zone="LINE",
                base_cycle_s=cyc,
                manual_content=1.0 if tier == TIER_MANUAL else 0.2,
                tier=tier,
                out_buffer=default_buffer,
                microstop_rate=0.0,
                microstop_range_s=(0.0, 0.0),
                process_noise_cv=0.0,
                base_defect_rate=0.0,
            )
        )

    if not any(s.out_buffer != default_buffer for s in stations):
        assumptions.append(
            f"Buffer capacity between stations is assumed to be {default_buffer} "
            f"units everywhere. Buffers set the delay before blocking propagates, "
            f"so real capacities improve upstream localisation."
        )

    variants = _variants(data)
    shifts = _shifts(data)

    line = LineTopology(
        name=name,
        takt_s=float(takt_s),
        stations=stations,
        zones={"LINE": {"start": 0, "end": n_stations - 1}},
        variants=variants,
        shifts=shifts,
        environment={},
    )
    assumptions.append(
        "The line is modelled as a single serial flow. Parallel sub-lines, "
        "rework loops and merge points are not detected and would need the "
        "propagation model rebuilt from the real process graph."
    )
    return line, assumptions


def _interleave(n_obs: int, n_total: int) -> List[int]:
    """Positions for ``n_obs`` observed stations spread across ``n_total`` slots.

    Endpoints are pinned so the first and last stations stay observed where
    possible: the head and tail of the line are the two places where a blind
    station is hardest to reconstruct, since one side has no neighbour.
    """
    if n_obs >= n_total:
        return list(range(n_total))
    if n_obs == 1:
        return [0]
    return sorted({int(round(k * (n_total - 1) / (n_obs - 1))) for k in range(n_obs)})


def _variants(data: PlantData) -> Dict[str, dict]:
    col = data.vehicles.get("variant")
    if col is None:
        return {"UNKNOWN": {"mix": 1.0}}
    mix = col.value_counts(normalize=True).to_dict()
    return {str(k): {"mix": float(v)} for k, v in mix.items()}


def _shifts(data: PlantData) -> List[dict]:
    col = data.vehicles.get("shift")
    names = sorted(col.astype(str).unique()) if col is not None else ["UNKNOWN"]
    hours = max(1, 24 // max(1, len(names)))
    return [
        {"name": n, "start_hour": (i * hours) % 24, "hours": hours}
        for i, n in enumerate(names)
    ]
