"""Forward propagation: what this disturbance is about to do to the line.

Detection says "S08 is the constraint". That alone does not tell a supervisor
whether to act now or at the end of the shift. This module answers the operative
question -- how much production is at stake, and how long until it hurts -- by
propagating the inferred constraint forward through the same flow physics the
line actually obeys.

Nothing here is learned. Given a constraint cycle time and the buffer capacities
either side of it, the arithmetic of a serial line fixes the answer. That is a
deliberate design choice: a forecast a plant engineer can check on paper is worth
more than a regression they have to take on faith, and it cannot drift.

Everything returned carries an explicit horizon and is expressed in units a plant
already uses -- vehicles, minutes, percent of takt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..factory.topology import LineTopology


@dataclass
class RippleForecast:
    """Predicted downstream consequence of an inferred constraint."""

    station: int
    station_id: str
    #: Estimated cycle time of the constraint, in seconds.
    constraint_cycle_s: float
    takt_s: float
    #: Vehicles per hour the line can sustain while this constraint holds.
    sustained_rate_vph: float
    nominal_rate_vph: float
    #: Fractional throughput loss, 0..1.
    throughput_loss_pct: float
    #: Vehicles not built over the forecast horizon.
    units_lost_at_horizon: float
    horizon_min: float
    #: Minutes until the buffer immediately downstream runs dry.
    minutes_to_downstream_starve: Optional[float]
    #: Minutes until the buffer immediately upstream backs up to full.
    minutes_to_upstream_block: Optional[float]
    #: Stations predicted to be starved, in the order they will be affected.
    downstream_affected: List[str] = field(default_factory=list)
    #: Stations predicted to be blocked.
    upstream_affected: List[str] = field(default_factory=list)
    #: True when the constraint is slower than takt and so genuinely limits output.
    is_binding: bool = False

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["downstream_affected"] = list(self.downstream_affected)
        d["upstream_affected"] = list(self.upstream_affected)
        return d


def forecast_ripple(
    line: LineTopology,
    station: int,
    constraint_cycle_s: float,
    horizon_min: float = 60.0,
    buffer_levels: Optional[Dict[int, float]] = None,
    max_affected: int = 6,
) -> RippleForecast:
    """Propagate an inferred constraint forward through the flow physics.

    ``constraint_cycle_s`` is the estimated processing time at the constraint --
    from ``infer_hidden_cycle_time`` for a station with no sensor, or from direct
    telemetry where one exists.

    ``buffer_levels`` maps station index to current outbound buffer occupancy.
    Where a level is unknown (both endpoints must be observed to derive it) we
    fall back to half capacity, which is the least informative assumption rather
    than the most alarming one.
    """
    takt = float(line.takt_s)
    P = float(constraint_cycle_s)
    stn = line.stations[station]

    nominal_rate = 3600.0 / takt
    sustained_rate = 3600.0 / max(P, 1e-6)
    is_binding = P > takt

    if is_binding:
        loss_pct = 1.0 - (takt / P)
        # Vehicles per second of deficit between what the line is fed and what
        # the constraint can pass.
        deficit_per_s = (1.0 / takt) - (1.0 / P)
    else:
        loss_pct = 0.0
        deficit_per_s = 0.0

    units_lost = deficit_per_s * horizon_min * 60.0

    def level_of(i: int) -> float:
        cap = float(min(line.stations[i].out_buffer, 10**4))
        if buffer_levels and i in buffer_levels and np.isfinite(buffer_levels[i]):
            return float(np.clip(buffer_levels[i], 0.0, cap))
        return cap / 2.0

    mins_starve = mins_block = None
    downstream: List[str] = []
    upstream: List[str] = []

    if is_binding and deficit_per_s > 0:
        if line.is_graph:
            # Graph case: walk the dominant (first, by station index) branch
            # at each split/merge rather than enumerating every branch --
            # a documented simplification (see docs/LIMITATIONS.md), not an
            # attempt at an exhaustive multi-branch forecast.
            nxt = line.successors(station)
            if nxt:
                lvl = level_of(station)
                mins_starve = float(lvl / deficit_per_s / 60.0)
                cur = nxt[0]
                for _ in range(max_affected):
                    downstream.append(line.stations[cur].station_id)
                    nxt2 = line.successors(cur)
                    if not nxt2:
                        break
                    cur = nxt2[0]

            prev = line.predecessors(station)
            if prev:
                p0 = prev[0]
                cap = float(min(line.stations[p0].out_buffer, 10**4))
                room = max(0.0, cap - level_of(p0))
                mins_block = float(room / deficit_per_s / 60.0)
                cur = p0
                for _ in range(max_affected):
                    upstream.append(line.stations[cur].station_id)
                    prev2 = line.predecessors(cur)
                    if not prev2:
                        break
                    cur = prev2[0]
        else:
            # Downstream: the buffer out of the constraint drains at the deficit rate.
            if station < line.n_stations - 1:
                lvl = level_of(station)
                mins_starve = float(lvl / deficit_per_s / 60.0)
                cum = lvl
                for i in range(station + 1, min(line.n_stations, station + 1 + max_affected)):
                    downstream.append(line.stations[i].station_id)
                    if i < line.n_stations - 1:
                        cum += level_of(i)

            # Upstream: the buffer into the constraint fills at the same deficit rate.
            if station > 0:
                cap = float(min(line.stations[station - 1].out_buffer, 10**4))
                room = max(0.0, cap - level_of(station - 1))
                mins_block = float(room / deficit_per_s / 60.0)
                for i in range(station - 1, max(-1, station - 1 - max_affected), -1):
                    upstream.append(line.stations[i].station_id)

    return RippleForecast(
        station=station,
        station_id=stn.station_id,
        constraint_cycle_s=P,
        takt_s=takt,
        sustained_rate_vph=sustained_rate,
        nominal_rate_vph=nominal_rate,
        throughput_loss_pct=loss_pct,
        units_lost_at_horizon=units_lost,
        horizon_min=horizon_min,
        minutes_to_downstream_starve=mins_starve,
        minutes_to_upstream_block=mins_block,
        downstream_affected=downstream,
        upstream_affected=upstream,
        is_binding=is_binding,
    )


def current_buffer_levels(
    scored_windows: pd.DataFrame, window: int
) -> Dict[int, float]:
    """Read observed buffer occupancy for one window, where it is derivable."""
    g = scored_windows[scored_windows["window"] == window]
    out: Dict[int, float] = {}
    if "buffer_level_mean" not in g.columns:
        return out
    for _, r in g.iterrows():
        v = r.get("buffer_level_mean", np.nan)
        if np.isfinite(v):
            out[int(r["station"])] = float(v)
    return out


def defect_exposure(
    line: LineTopology,
    station: int,
    multiplier: float,
    baseline_rate_per_vehicle: float,
    vehicles_in_flight: int,
) -> dict:
    """How many in-flight vehicles are likely carrying a defect from this station.

    ``vehicles_in_flight`` is the count of vehicles that have already passed the
    suspect station but have not yet reached the gate that would catch it. Those
    are the units at risk *right now*, and the number that makes a quality alert
    actionable rather than merely interesting.
    """
    excess = max(0.0, multiplier - 1.0) * baseline_rate_per_vehicle
    return {
        "vehicles_in_flight": int(vehicles_in_flight),
        "baseline_rate_per_vehicle": float(baseline_rate_per_vehicle),
        "estimated_multiplier": float(multiplier),
        "expected_extra_defective_units": float(excess * vehicles_in_flight),
        "expected_total_defective_units": float(
            multiplier * baseline_rate_per_vehicle * vehicles_in_flight
        ),
    }


def vehicles_between(
    line: LineTopology, station: int, genealogy: pd.DataFrame, now_vehicle: int
) -> int:
    """Vehicles that have passed ``station`` but not yet reached the next gate."""
    gate = line.next_inspection_after(station + 1)
    if gate is None:
        gate = line.n_stations - 1
    # In vehicle-index space the transit from station to gate is simply the
    # number of stations between them, since the line advances one vehicle per
    # takt at every station.
    return int(max(0, min(now_vehicle, gate - station)))
