"""What RippleTwin needs from a real plant, and whether a given plant has it.

Why this module exists
----------------------
The dominant documented failure mode for industrial digital twins is not the
model -- it is the data layer. Reviews of failed projects describe "beautiful
replicas that don't support decision-making", built on fragmented, partially
integrated, low-quality data. A prototype that only works against its own
simulator is precisely that failure waiting to happen.

So this module states, signal by signal, exactly what the twin consumes, where
that signal lives in a real plant, what standard exposes it, and -- most
importantly -- **what RippleTwin can still do when it is missing**. Then it
turns that into a runnable Phase 0 assessment: point it at a plant's tag list
and it reports whether the twin can run at all, at what capability, and what
would have to be fixed first.

The point is to be able to answer, in a meeting, "what do you need from my
plant?" with a specific list rather than "real-time production data".

Grounding
---------
The signals below are not invented. Blocked and starved are standard equipment
states in OEE systems, conventionally derived at the PLC as::

    STARVED = motor running AND infeed empty
    BLOCKED = motor running AND outfeed full

from infeed/outfeed photocells that most conveyor-linked stations already carry
for interlock purposes. PackML (ISA-TR88.00.02) standardises machine states in
modern Siemens, Allen-Bradley, Beckhoff and B&R PLCs. VIN-level traceability --
station, timestamp, operator, torque/angle results per vehicle -- is routine in
automotive for recall and warranty reasons.

One industry quirk matters to us. Because starved and blocked are *line* losses
rather than station losses, plants usually exclude them from a station's own OEE
and file them as a nuisance category; penalising a machine for being starved is
a good way to start an argument with its operators. That means this data is
frequently collected and rarely used. RippleTwin's core input is, in a real
sense, waste data the plant is already paying to store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd


class Capability(str, Enum):
    """What the twin can do with a given data situation."""

    FULL = "FULL"
    FLOW_ONLY = "FLOW_ONLY"
    QUALITY_ONLY = "QUALITY_ONLY"
    NOT_VIABLE = "NOT_VIABLE"


#: Purdue reference levels, used to say where a signal is read from.
PURDUE = {
    0: "sensors and actuators",
    1: "PLC / basic control",
    2: "SCADA / HMI / line control",
    3: "MES / manufacturing operations",
    3.5: "industrial DMZ",
    4: "enterprise IT",
}


@dataclass
class Signal:
    """One input RippleTwin consumes, and where it comes from in reality."""

    key: str
    name: str
    #: What the twin uses it for.
    purpose: str
    #: True when the twin cannot run at all without it.
    required: bool
    #: Which capability is lost without it.
    enables: str
    purdue_level: float
    typical_source: str
    #: How it is usually exposed. Real protocol names, not aspiration.
    interface: str
    example_address: str
    #: What we do when it is absent. Never "the project stops" unless it must.
    if_missing: str
    #: Rough per-vehicle-per-station data rate, for sizing conversations.
    cardinality: str = "1 row per vehicle per station"


#: The complete input contract. Nothing else is read.
DATA_CONTRACT: List[Signal] = [
    Signal(
        key="station_state",
        name="Station state with timestamps (running / blocked / starved / fault)",
        purpose=(
            "The core signal. The boundary between blocked-dominant and "
            "starved-dominant stations is what localises a constraint."
        ),
        required=True,
        enables="everything in the flow path",
        purdue_level=1,
        typical_source="PLC state word, or OEE/downtime system already deployed",
        interface="OPC UA (PackML state), MQTT Sparkplug B, or historian tag",
        example_address="ns=2;s=Line1.St07.PackML.CurrentState",
        if_missing=(
            "Derivable at the PLC from motor-run plus infeed/outfeed photocells: "
            "STARVED = running AND infeed empty, BLOCKED = running AND outfeed "
            "full. If neither exists at a station, that station is simply one of "
            "the blind ones the twin is built to infer."
        ),
    ),
    Signal(
        key="station_cycle_time",
        name="Per-vehicle processing time at instrumented stations",
        purpose=(
            "Separates a station that is the constraint from one that is a "
            "victim of it: blocking and starving do not change a station's own "
            "tool cycle."
        ),
        required=False,
        enables="discriminating cause from victim at instrumented stations",
        purdue_level=1,
        typical_source="PLC cycle start/complete timestamps",
        interface="OPC UA, historian",
        example_address="ns=2;s=Line1.St07.CycleTime",
        if_missing=(
            "Localisation falls back to flow evidence alone. Accuracy drops at "
            "instrumented stations; the hidden-station capability is unaffected."
        ),
    ),
    Signal(
        key="vehicle_identity",
        name="VIN / serial read at each scanned station",
        purpose=(
            "Aligns windows in vehicle-index space rather than wall-clock, and "
            "carries defect attribution back to a source station."
        ),
        required=True,
        enables="window alignment and the whole quality path",
        purdue_level=2,
        typical_source="barcode / RFID reader, MES traceability record",
        interface="MES API, OPC UA, scanner middleware",
        example_address="MES.Traceability.UnitHistory[VIN]",
        if_missing=(
            "Flow path can run on wall-clock windows with degraded alignment. "
            "The quality path cannot run at all."
        ),
    ),
    Signal(
        key="build_sequence",
        name="Build sequence with model variant per VIN",
        purpose=(
            "Expectations are formed from the actual model mix in each window, "
            "so an SUV-heavy block is not mistaken for a fault."
        ),
        required=True,
        enables="mix-aware baselines",
        purdue_level=3,
        typical_source="MES / production scheduling",
        interface="MES API, database view",
        example_address="MES.Schedule.BuildSequence",
        if_missing=(
            "Baselines become mix-blind and false alarms rise sharply on a "
            "mixed-model line. Viable only on a single-variant line."
        ),
        cardinality="1 row per vehicle",
    ),
    Signal(
        key="inspection_results",
        name="Gate results with defect codes",
        purpose="Input to defect attribution and the source of outcome feedback.",
        required=False,
        enables="the quality path",
        purdue_level=3,
        typical_source="MES quality module, end-of-line test system",
        interface="MES API, database view",
        example_address="MES.Quality.InspectionResult",
        if_missing="Flow path runs normally; no defect attribution.",
        cardinality="1 row per vehicle per gate",
    ),
    Signal(
        key="shift_calendar",
        name="Shift pattern and planned downtime",
        purpose=(
            "Night shift is legitimately slower. Without this the twin flags "
            "the shift change as a fault, and breaks as a line stoppage."
        ),
        required=True,
        enables="correct baselines",
        purdue_level=3,
        typical_source="MES / ERP calendar",
        interface="database view, CSV export",
        example_address="ERP.Calendar.ShiftPattern",
        if_missing="Baselines are biased by shift and break structure.",
        cardinality="static configuration",
    ),
    Signal(
        key="line_topology",
        name="Station order and buffer capacities",
        purpose=(
            "Propagation distance is measured in buffer slots, not station "
            "adjacency. This is the structural half of the twin."
        ),
        required=True,
        enables="the propagation model",
        purdue_level=3,
        typical_source="layout drawings, conveyor design, controls documentation",
        interface="one-time engineering input (YAML)",
        example_address="configs/line_42.yaml",
        if_missing=(
            "No twin. This is a day of work with a controls engineer, not a "
            "data-collection project."
        ),
        cardinality="static configuration",
    ),
    Signal(
        key="failure_mode_map",
        name="Station-to-defect-type map (process FMEA)",
        purpose=(
            "Narrows defect attribution from 'somewhere upstream' to a handful "
            "of candidates. A sealer station cannot cause a torque fault."
        ),
        required=False,
        enables="defect attribution to a specific station",
        purdue_level=3,
        typical_source="process FMEA, control plan",
        interface="one-time engineering input",
        example_address="configs/line_42.yaml (defect_profile)",
        if_missing=(
            "Quality attribution degrades to zone level rather than station "
            "level. Still useful, much less specific."
        ),
        cardinality="static configuration",
    ),
    Signal(
        key="process_channels",
        name="Torque / vibration / temperature at rich stations",
        purpose="Corroborating evidence at stations that already have it.",
        required=False,
        enables="stronger evidence at instrumented stations",
        purdue_level=1,
        typical_source="tightening controllers, condition monitoring",
        interface="OPC UA, vendor API",
        example_address="ns=2;s=Line1.St07.Torque_Nm",
        if_missing="No effect on the hidden-station capability.",
    ),
    Signal(
        key="buffer_level",
        name="Buffer occupancy between stations",
        purpose="Sharpens the ripple forecast's time-to-starvation estimate.",
        required=False,
        enables="tighter forecast timing",
        purdue_level=2,
        typical_source="conveyor part counters, AGV system",
        interface="OPC UA, SCADA",
        example_address="ns=2;s=Line1.Buf07.Count",
        if_missing=(
            "Forecast falls back to half-capacity, which is the least "
            "informative assumption rather than the most alarming one."
        ),
    ),
]

#: Timing tolerance. The method compares event ordering across stations, so
#: clock skew between PLCs directly corrupts the signal. Industry guidance on
#: OEE architecture is notably silent on clock sync, which makes this a real
#: deployment risk rather than a theoretical one.
CLOCK_SYNC_REQUIREMENT_S = 1.0
CLOCK_SYNC_NOTE = (
    "Station events must share a time base to within about a second -- roughly "
    "a sixtieth of takt. NTP against a plant time server is sufficient; "
    "free-running PLC clocks are not. Where a station's events are timestamped "
    "on arrival at a historian rather than at the PLC, queueing delay shows up "
    "as apparent cycle-time variation and will degrade localisation."
)


@dataclass
class ReadinessFinding:
    signal: Signal
    available: bool
    note: str = ""


@dataclass
class ReadinessReport:
    """The Phase 0 answer: can this plant run RippleTwin, and on what?"""

    capability: Capability
    findings: List[ReadinessFinding]
    n_stations: int
    n_with_state: int
    coverage: float
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "signal": f.signal.name,
                    "required": f.signal.required,
                    "available": f.available,
                    "enables": f.signal.enables,
                    "purdue": f.signal.purdue_level,
                    "interface": f.signal.interface,
                    "if_missing": f.signal.if_missing,
                    "note": f.note,
                }
                for f in self.findings
            ]
        )

    def summary(self) -> str:
        lines = [
            f"CAPABILITY: {self.capability.value}",
            f"Stations: {self.n_stations}, with usable state data: "
            f"{self.n_with_state} ({self.coverage * 100:.0f}% coverage)",
        ]
        if self.blockers:
            lines.append("\nBLOCKERS (the twin cannot run):")
            lines += [f"  - {b}" for b in self.blockers]
        if self.warnings:
            lines.append("\nDEGRADED (it runs, with less):")
            lines += [f"  - {w}" for w in self.warnings]
        if self.notes:
            lines.append("\nNOTES:")
            lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def assess_readiness(
    available_signals: Iterable[str],
    n_stations: int,
    n_stations_with_state: int,
    clock_sync_s: Optional[float] = None,
) -> ReadinessReport:
    """Phase 0: decide whether a plant can run RippleTwin, and at what capability.

    ``available_signals`` are contract keys the plant can actually supply.
    ``n_stations_with_state`` is how many stations emit usable state data --
    which is the number that decides whether the mechanism has anything to work
    with, because the twin infers a blind station from the instrumented ones
    **on both sides of it**.
    """
    have: Set[str] = {str(s) for s in available_signals}
    findings = [ReadinessFinding(s, s.key in have) for s in DATA_CONTRACT]

    coverage = (n_stations_with_state / n_stations) if n_stations else 0.0
    blockers: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []

    for s in DATA_CONTRACT:
        if s.key in have:
            continue
        (blockers if s.required else warnings).append(
            f"{s.name} -- {s.if_missing}"
        )

    # Structural checks that matter more than the tag list.
    flow_structurally_viable = n_stations_with_state >= 2
    if not flow_structurally_viable:
        blockers.append(
            "FLOW PATH: fewer than two stations emit state data. The mechanism "
            "needs instrumented stations either side of a blind one; with fewer "
            "than two there is no boundary to find. The quality path is "
            "unaffected -- it runs off gate results and build sequence, and does "
            "not depend on how many stations are instrumented."
        )
    if 0 < coverage < 0.25:
        warnings.append(
            f"Only {coverage * 100:.0f}% of stations emit state data. Below "
            f"roughly 25% our measured exact-localisation rate falls into the "
            f"low tens of percent -- useful for narrowing a zone, not for "
            f"dispatching a technician to a named station."
        )
    if coverage >= 0.99:
        notes.append(
            "Every station is instrumented. Shadow-sensing has nothing to add "
            "here and our evaluation shows it adds nothing -- a conventional "
            "twin, or plain SPC, will serve. Use RippleTwin on the lines that "
            "have gaps."
        )

    if clock_sync_s is not None and clock_sync_s > CLOCK_SYNC_REQUIREMENT_S:
        warnings.append(
            f"Clock skew between stations is {clock_sync_s:.1f}s, above the "
            f"~{CLOCK_SYNC_REQUIREMENT_S:.0f}s the method assumes. {CLOCK_SYNC_NOTE}"
        )
    notes.append(CLOCK_SYNC_NOTE)

    flow_ok = (
        "station_state" in have
        and "build_sequence" in have
        and "shift_calendar" in have
        and "line_topology" in have
        and flow_structurally_viable
    )
    quality_ok = "inspection_results" in have and "vehicle_identity" in have

    if flow_ok and quality_ok:
        capability = Capability.FULL
    elif flow_ok:
        capability = Capability.FLOW_ONLY
    elif quality_ok:
        capability = Capability.QUALITY_ONLY
    else:
        capability = Capability.NOT_VIABLE

    return ReadinessReport(
        capability=capability,
        findings=findings,
        n_stations=n_stations,
        n_with_state=n_stations_with_state,
        coverage=coverage,
        blockers=blockers,
        warnings=warnings,
        notes=notes,
    )


def contract_frame() -> pd.DataFrame:
    """The full input contract as a table, for a Phase 0 conversation."""
    return pd.DataFrame(
        [
            {
                "Signal": s.name,
                "Required": "yes" if s.required else "no",
                "Purdue level": s.purdue_level,
                "Typical source": s.typical_source,
                "Interface": s.interface,
                "Used for": s.purpose,
                "If missing": s.if_missing,
            }
            for s in DATA_CONTRACT
        ]
    )
