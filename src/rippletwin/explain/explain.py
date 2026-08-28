"""Evidence-grounded explanations.

Every sentence produced here is assembled from numbers the model actually
computed. There is no language model anywhere in this path, and that is a
deliberate design decision rather than a limitation: an explanation that can
drift away from the evidence is worse than no explanation, because it invites a
supervisor to trust a recommendation for a reason that was never true.

The structure answers the six questions a floor supervisor actually asks, in the
order they ask them:

    WHAT changed, WHERE, WHY it matters, WHAT happens next,
    HOW confident, and WHAT to do.

Each claim carries its own provenance tag, so the interface can show whether a
number was measured, inferred, or predicted. Keeping that distinction visible
throughout is what separates this from a dashboard that presents an estimate
with the same authority as a reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..factory.topology import LineTopology
from ..twin.propagate import RippleForecast
from ..twin.shadow import ShadowResult

#: Provenance tags. These appear in the UI next to every number.
OBSERVED = "OBSERVED"    # read directly from a sensor
INFERRED = "INFERRED"    # estimated for a station with no sensor
PREDICTED = "PREDICTED"  # a forward projection that has not happened yet


@dataclass
class EvidenceItem:
    """One piece of supporting evidence behind an alert."""

    provenance: str
    station_id: str
    channel: str
    value: float
    units: str
    text: str


@dataclass
class Explanation:
    """A complete, auditable account of one alert."""

    station_id: str
    station_tier: str
    is_inferred: bool
    headline: str
    what_changed: str
    why_it_matters: str
    what_happens_next: str
    confidence_text: str
    confidence: float
    evidence: List[EvidenceItem] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    #: The second-best candidate and its posterior mass, when one exists --
    #: the model's own uncertainty made visible, not just its top pick. See
    #: README.md/docs/METHOD.md on preferring "likely contributor" /
    #: "candidate root cause" language over causal certainty.
    alternative_station_id: Optional[str] = None
    alternative_probability: Optional[float] = None

    def as_dict(self) -> dict:
        d = {
            "station_id": self.station_id,
            "station_tier": self.station_tier,
            "is_inferred": self.is_inferred,
            "headline": self.headline,
            "what_changed": self.what_changed,
            "why_it_matters": self.why_it_matters,
            "what_happens_next": self.what_happens_next,
            "confidence": self.confidence,
            "confidence_text": self.confidence_text,
            "evidence": [e.__dict__ for e in self.evidence],
            "caveats": list(self.caveats),
            "alternative_station_id": self.alternative_station_id,
            "alternative_probability": self.alternative_probability,
        }
        return d

    def as_text(self) -> str:
        lines = [
            self.headline,
            "",
            f"WHAT CHANGED : {self.what_changed}",
            f"WHY IT MATTERS: {self.why_it_matters}",
            f"WHAT'S NEXT  : {self.what_happens_next}",
            f"CONFIDENCE   : {self.confidence_text}",
            "",
            "EVIDENCE",
        ]
        for e in self.evidence:
            lines.append(f"  [{e.provenance:<9}] {e.text}")
        if self.alternative_station_id:
            lines.append("")
            lines.append(
                f"ALTERNATIVE HYPOTHESIS: {self.alternative_station_id} "
                f"({(self.alternative_probability or 0) * 100:.0f}% posterior)"
            )
        if self.caveats:
            lines.append("")
            lines.append("CAVEATS")
            for c in self.caveats:
                lines.append(f"  - {c}")
        return "\n".join(lines)


def _fmt_min(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "unknown"
    if x < 1:
        return "under a minute"
    return f"{x:.0f} min"


def explain_flow_alert(
    line: LineTopology,
    result: ShadowResult,
    forecast: Optional[RippleForecast],
    inferred_cycle_s: Optional[float],
    top_evidence_n: int = 4,
) -> Explanation:
    """Build the explanation for a flow (bottleneck) alert."""
    k = result.top_station
    stn = line.stations[k]
    is_inferred = stn.is_hidden

    d_block = result.evidence["d_blocked"]
    d_starve = result.evidence["d_starved"]
    z_proc = result.evidence["z_proc"]
    takt = line.takt_s

    # Pick the observed stations contributing most to the fit, either side.
    up = [i for i in line.nearest_observed_upstream(k, 6)]
    dn = [i for i in line.nearest_observed_downstream(k, 6)]

    ev: List[EvidenceItem] = []
    up_scored = sorted(
        [(i, d_block[i]) for i in up if np.isfinite(d_block[i])],
        key=lambda t: -t[1],
    )[:top_evidence_n // 2 + 1]
    dn_scored = sorted(
        [(i, d_starve[i]) for i in dn if np.isfinite(d_starve[i])],
        key=lambda t: -t[1],
    )[:top_evidence_n // 2 + 1]

    for i, val in up_scored:
        if val <= 0.02:
            continue
        ev.append(
            EvidenceItem(
                OBSERVED, line.stations[i].station_id, "blocked_time", val * 100, "% of takt",
                f"{line.stations[i].station_id} is blocked {val * 100:.0f}% of takt "
                f"above its normal level -- it cannot hand work forward.",
            )
        )
    for i, val in dn_scored:
        if val <= 0.02:
            continue
        ev.append(
            EvidenceItem(
                OBSERVED, line.stations[i].station_id, "starved_time", val * 100, "% of takt",
                f"{line.stations[i].station_id} is starved {val * 100:.0f}% of takt "
                f"above its normal level -- work is not arriving.",
            )
        )

    if not is_inferred and np.isfinite(z_proc[k]):
        ev.append(
            EvidenceItem(
                OBSERVED, stn.station_id, "cycle_time", float(z_proc[k]), "sigma",
                f"{stn.station_id}'s own cycle time is {z_proc[k]:.1f} sigma above "
                f"its expected value for this model mix and shift.",
            )
        )

    if inferred_cycle_s is not None:
        ev.append(
            EvidenceItem(
                INFERRED, stn.station_id, "cycle_time", float(inferred_cycle_s), "s",
                f"{stn.station_id}'s cycle time is estimated at {inferred_cycle_s:.0f}s "
                f"against a {takt:.0f}s takt, read from the departure rate of the "
                f"first instrumented station downstream.",
            )
        )

    if is_inferred:
        headline = (
            f"{stn.station_id} ({stn.zone}) -- inferred constraint, no sensor at this station"
        )
        what = (
            f"{stn.station_id} has no instrumentation. Its state is reconstructed from "
            f"the stations either side of it: work is backing up upstream and running "
            f"dry downstream, and the boundary between the two falls at {stn.station_id}."
        )
    else:
        headline = f"{stn.station_id} ({stn.zone}) -- measured constraint"
        what = (
            f"{stn.station_id}'s own cycle time has risen above its expected value, and "
            f"the surrounding flow pattern is consistent with it being the constraint."
        )

    if forecast is not None and forecast.is_binding:
        why = (
            f"At an estimated {forecast.constraint_cycle_s:.0f}s cycle against a "
            f"{forecast.takt_s:.0f}s takt, this station caps the line at "
            f"{forecast.sustained_rate_vph:.0f} vehicles/hour "
            f"({forecast.throughput_loss_pct * 100:.0f}% below target)."
        )
        nxt = (
            f"About {forecast.units_lost_at_horizon:.0f} vehicles lost over the next "
            f"{forecast.horizon_min:.0f} minutes if nothing changes. "
            f"Downstream starvation reaches {forecast.downstream_affected[0] if forecast.downstream_affected else 'the next station'} "
            f"in {_fmt_min(forecast.minutes_to_downstream_starve)}; upstream backs up to "
            f"{forecast.upstream_affected[0] if forecast.upstream_affected else 'the previous station'} "
            f"in {_fmt_min(forecast.minutes_to_upstream_block)}."
        )
        ev.append(
            EvidenceItem(
                PREDICTED, stn.station_id, "units_lost",
                float(forecast.units_lost_at_horizon), "vehicles",
                f"Projected loss of {forecast.units_lost_at_horizon:.0f} vehicles over "
                f"{forecast.horizon_min:.0f} minutes at the current constraint rate.",
            )
        )
    else:
        why = (
            "The flow pattern points at this station, but the estimated cycle time is "
            "still within takt, so the line is not yet losing output."
        )
        nxt = "No throughput loss projected while the constraint stays inside takt."

    conf = result.group_prob
    if result.confident:
        ctext = (
            f"{conf * 100:.0f}% of the posterior sits on "
            f"{line.stations[min(result.group)].station_id}"
            + (
                f"-{line.stations[max(result.group)].station_id}"
                if len(result.group) > 1
                else ""
            )
            + "."
        )
    else:
        ctext = (
            f"Only {conf * 100:.0f}% of the posterior is concentrated -- not enough to "
            f"name a single station with confidence."
        )

    caveats: List[str] = []
    if len(result.group) > 1:
        ids = ", ".join(line.stations[i].station_id for i in result.group)
        caveats.append(
            f"Adjacent candidates {ids} are not separable from the available sensors; "
            f"check them together."
        )
    if is_inferred:
        caveats.append(
            "This station's state is inferred, not measured. Confirm on the floor "
            "before acting on it."
        )
    if result.evidence.get("p_line_supply", 0) > 0.15:
        caveats.append(
            f"A line-wide supply shortfall also partly explains this pattern "
            f"({result.evidence['p_line_supply'] * 100:.0f}% posterior) -- check inbound "
            f"material before treating this as a station fault."
        )

    # Second-best candidate, station space only (excludes NULL/LINE_SUPPLY,
    # which are covered separately by the caveats above). Made visible so a
    # supervisor sees the model's own uncertainty, not just its top pick --
    # "likely contributor", never "caused".
    station_post = result.evidence.get("station_post")
    alt_id, alt_p = None, None
    if station_post is not None:
        ranked = sorted(
            ((i, p) for i, p in enumerate(station_post) if i != k),
            key=lambda t: -t[1],
        )
        if ranked and ranked[0][1] > 0.02:
            alt_id = line.stations[ranked[0][0]].station_id
            alt_p = float(ranked[0][1])

    return Explanation(
        station_id=stn.station_id,
        station_tier=stn.tier,
        is_inferred=is_inferred,
        headline=headline,
        what_changed=what,
        why_it_matters=why,
        what_happens_next=nxt,
        confidence_text=ctext,
        confidence=float(conf),
        evidence=ev,
        caveats=caveats,
        alternative_station_id=alt_id,
        alternative_probability=alt_p,
    )


def explain_quality_alert(
    line: LineTopology,
    station: int,
    m_hat: float,
    llr: float,
    rank: int,
    candidates: List[int],
    exposure: dict,
    dominant_types: List[str],
) -> Explanation:
    """Build the explanation for a quality-attribution alert."""
    stn = line.stations[station]
    is_inferred = stn.is_hidden
    gate = line.next_inspection_after(station + 1)
    gate_id = line.stations[gate].inspection_id if gate is not None else "end of line"

    ev = [
        EvidenceItem(
            INFERRED, stn.station_id, "defect_multiplier", float(m_hat), "x",
            f"Defects matching {stn.station_id}'s failure-mode signature "
            f"({', '.join(dominant_types)}) are running {m_hat:.1f}x above its "
            f"normal rate.",
        ),
        EvidenceItem(
            OBSERVED, gate_id, "gate_results", float(exposure.get("vehicles_in_flight", 0)),
            "vehicles",
            f"{exposure.get('vehicles_in_flight', 0)} vehicles have passed "
            f"{stn.station_id} and not yet reached {gate_id}.",
        ),
        EvidenceItem(
            PREDICTED, stn.station_id, "units_at_risk",
            float(exposure.get("expected_extra_defective_units", 0.0)), "vehicles",
            f"About {exposure.get('expected_extra_defective_units', 0.0):.1f} of those "
            f"in-flight vehicles are expected to carry a defect that would not have "
            f"occurred at the normal rate.",
        ),
    ]

    headline = (
        f"{stn.station_id} ({stn.zone}) -- inferred quality drift"
        + (", no sensor at this station" if is_inferred else "")
    )
    what = (
        f"Cycle times across the line are normal. What changed is the *mix* of defect "
        f"types arriving at {gate_id}: {', '.join(dominant_types)} are over-represented, "
        f"and those are {stn.station_id}'s characteristic failure modes."
    )
    why = (
        f"A defect made here is not caught until {gate_id}, so every vehicle in between "
        f"is accumulating value on top of a fault that will have to be reworked."
    )
    nxt = (
        f"At {m_hat:.1f}x the normal rate, roughly "
        f"{exposure.get('expected_total_defective_units', 0.0):.1f} of the "
        f"{exposure.get('vehicles_in_flight', 0)} in-flight vehicles are expected to fail "
        f"at {gate_id} unless the source is corrected."
    )
    conf = 1.0 / max(1, rank)
    ctext = (
        f"{stn.station_id} ranks #{rank} of {len(candidates)} candidate stations on "
        f"failure-mode evidence (log-likelihood ratio {llr:.1f})."
    )

    caveats = [
        "Defect attribution pools several hundred vehicles to gain statistical power, "
        "so it reacts more slowly than a bottleneck alert.",
    ]
    if rank > 1:
        caveats.append(
            "A higher-ranked candidate exists; treat this as one of a shortlist rather "
            "than a single conclusion."
        )
    if is_inferred:
        caveats.append(
            "This station has no sensor. The attribution rests on failure-mode "
            "propensity and vehicle genealogy, not on a measurement here."
        )
    if stn.zone == "PAINT":
        caveats.append(
            "Paint defects are humidity-sensitive. Rule out an ambient excursion before "
            "treating this as an equipment fault."
        )

    return Explanation(
        station_id=stn.station_id,
        station_tier=stn.tier,
        is_inferred=is_inferred,
        headline=headline,
        what_changed=what,
        why_it_matters=why,
        what_happens_next=nxt,
        confidence_text=ctext,
        confidence=float(conf),
        evidence=ev,
        caveats=caveats,
    )
