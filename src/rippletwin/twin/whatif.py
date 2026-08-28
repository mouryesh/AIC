"""What-if / counterfactual projections (Round 2 brief §30).

Every function here is a **simulation-based projection**, not a measurement
and not a causal guarantee -- it re-runs the same deterministic physics
(``twin.propagate.forecast_ripple``) or the same structural
value-of-information calculation (``twin.placement.ambiguity``) that
already produces every other number in this repository, with one input
changed. Nothing here is a new model; it is the existing ones asked a
different, hypothetical question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..factory.topology import LineTopology
from .placement import ambiguity
from .propagate import RippleForecast, forecast_ripple


@dataclass
class WhatIfResult:
    label: str
    #: Always simulation-based -- never a causal certainty. Every caller
    #: (dashboard, CLI) must surface this alongside the numbers.
    disclaimer: str = "Simulation-based projection, not a measurement or a guarantee."


@dataclass
class RepairWhatIf(WhatIfResult):
    before: Optional[RippleForecast] = None
    after: Optional[RippleForecast] = None


def whatif_repair(
    line: LineTopology,
    station: int,
    current_cycle_s: float,
    horizon_min: float = 60.0,
    buffer_levels: Optional[dict] = None,
) -> RepairWhatIf:
    """"What if this station were repaired right now?" -- the same forecast
    physics, with the constraint's cycle time reset to the station's own
    nominal ``base_cycle_s`` instead of its currently estimated value."""
    stn = line.stations[station]
    before = forecast_ripple(line, station, current_cycle_s, horizon_min, buffer_levels)
    after = forecast_ripple(line, station, stn.base_cycle_s, horizon_min, buffer_levels)
    return RepairWhatIf(
        label=f"What if {stn.station_id} were repaired now?",
        before=before, after=after,
    )


@dataclass
class CycleTimeWhatIf(WhatIfResult):
    before: Optional[RippleForecast] = None
    after: Optional[RippleForecast] = None
    pct_improvement: float = 0.0


def whatif_cycle_time_improvement(
    line: LineTopology,
    station: int,
    current_cycle_s: float,
    pct_improvement: float,
    horizon_min: float = 60.0,
    buffer_levels: Optional[dict] = None,
) -> CycleTimeWhatIf:
    """"What if this station's cycle time improved by X%?" -- a partial
    counterfactual, for a supervisor asking about a process tweak rather
    than a full repair."""
    stn = line.stations[station]
    improved_cycle = current_cycle_s * (1.0 - pct_improvement)
    before = forecast_ripple(line, station, current_cycle_s, horizon_min, buffer_levels)
    after = forecast_ripple(line, station, improved_cycle, horizon_min, buffer_levels)
    return CycleTimeWhatIf(
        label=f"What if {stn.station_id}'s cycle time improved by {pct_improvement:.0%}?",
        before=before, after=after, pct_improvement=pct_improvement,
    )


@dataclass
class SensorWhatIf(WhatIfResult):
    station_id: str = ""
    ambiguity_before: float = 0.0
    ambiguity_after: float = 0.0
    separability_before: float = 0.0
    separability_after: float = 0.0


def whatif_add_sensor(line: LineTopology, station: int) -> SensorWhatIf:
    """"What if we added a sensor at this station?" -- the same
    separability/ambiguity calculation ``twin.placement.recommend_sensors``
    already uses, applied to one named candidate rather than ranking all of
    them. Uncertainty before vs. after, not a claim about localisation
    accuracy on a specific future fault (that is what
    ``twin.placement.validate_against_outcomes`` checks empirically)."""
    stn = line.stations[station]
    observed = set(line.observed_indices)
    before = ambiguity(line, observed).set_index("station")
    after = ambiguity(line, observed | {station}).set_index("station")
    return SensorWhatIf(
        label=f"What if we added a sensor at {stn.station_id}?",
        station_id=stn.station_id,
        ambiguity_before=float(before.loc[station, "ambiguity"]),
        ambiguity_after=float(after.loc[station, "ambiguity"]) if station in after.index else 0.0,
        separability_before=float(before.loc[station, "separability"]),
        separability_after=float(after.loc[station, "separability"]) if station in after.index else 0.0,
    )
