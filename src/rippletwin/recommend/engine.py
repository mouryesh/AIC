"""Recommendation and abstention.

RippleTwin never touches production equipment. It produces a recommendation for
a person, together with the evidence behind it and an honest statement of how
sure it is. The action space is deliberately small and entirely reversible --
inspect, check, re-sequence, escalate. Nothing here writes to a PLC, changes
line control logic, or stops the line automatically.

The abstention rule matters more than the recommendation rule. A twin that
always has an answer is a twin that will eventually be confidently wrong at a
station nobody can verify, and floor trust does not survive that twice. So when
the posterior is spread across several stations, when a line-wide supply problem
explains the pattern just as well, or when the estimated impact is below what is
worth interrupting a shift for, the correct output is to say so and escalate
rather than to name a station.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..factory.topology import LineTopology
from ..twin.propagate import RippleForecast
from ..twin.shadow import ShadowResult

# Action types. All are advisory, reversible, and performed by a person.
ACTION_INSPECT = "INSPECT_STATION"
ACTION_QUALITY_HOLD = "TARGETED_QUALITY_CHECK"
ACTION_CHECK_SUPPLY = "CHECK_INBOUND_MATERIAL"
ACTION_MONITOR = "MONITOR"
ACTION_ESCALATE = "ESCALATE_AMBIGUOUS"

PRIORITY_HIGH = "HIGH"
PRIORITY_MEDIUM = "MEDIUM"
PRIORITY_LOW = "LOW"


@dataclass
class Recommendation:
    """A proposed action, always awaiting human approval."""

    action: str
    priority: str
    target_stations: List[str]
    title: str
    detail: str
    rationale: str
    #: Estimated benefit if acted on now, in vehicles. None when not quantifiable.
    units_at_stake: Optional[float]
    confidence: float
    abstained: bool = False
    requires_approval: bool = True
    alternatives: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "priority": self.priority,
            "target_stations": list(self.target_stations),
            "title": self.title,
            "detail": self.detail,
            "rationale": self.rationale,
            "units_at_stake": self.units_at_stake,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "requires_approval": self.requires_approval,
            "alternatives": list(self.alternatives),
        }


@dataclass
class RecommendConfig:
    """Thresholds governing when the twin will and will not name a station."""

    #: Posterior mass on the candidate group required to name stations at all.
    min_confidence: float = 0.45
    #: Posterior mass required before the recommendation is treated as high priority.
    high_confidence: float = 0.75
    #: Projected vehicles lost per hour below which this is not worth interrupting for.
    min_units_at_stake: float = 1.5
    #: Posterior mass on LINE_SUPPLY above which supply is checked first.
    supply_precedence: float = 0.25
    #: Candidate group wider than this is reported as a zone, not a station.
    max_named_group: int = 3
    #: Stations at or before this index count as "head of line", where a slow
    #: station and an inbound shortfall look nearly identical from downstream.
    head_of_line_index: int = 1
    #: Upstream blocking, as a fraction of takt above normal, required to
    #: conclude a head-of-line station is genuinely the constraint.
    min_upstream_blocking: float = 0.04


def recommend_flow(
    line: LineTopology,
    result: ShadowResult,
    forecast: Optional[RippleForecast],
    cfg: RecommendConfig | None = None,
) -> Recommendation:
    """Turn an inferred constraint into an advisory action, or abstain."""
    cfg = cfg or RecommendConfig()
    group_ids = [line.stations[i].station_id for i in result.group]
    top = line.stations[result.top_station]
    p_supply = float(result.evidence.get("p_line_supply", 0.0))
    units = forecast.units_lost_at_horizon if forecast else None

    # --- head-of-line ambiguity -------------------------------------------
    # Near the head of the line, "station k is slow" and "nothing is arriving"
    # produce almost the same downstream picture, because there are too few
    # upstream stations to tell them apart. One observation separates them:
    #
    #   upstream station BLOCKED  -> it has work it cannot hand over  -> k is slow
    #   upstream station STARVED  -> it has no work at all            -> supply
    #
    # Without this check a material delay was attributed to S02 with 85%
    # confidence while S01 sat starved at 179% of takt -- the one thing that
    # cannot happen if S02 is the constraint.
    d_blocked = result.evidence.get("d_blocked")
    upstream = line.nearest_observed_upstream(result.top_station, k=3)
    if d_blocked is not None and result.top_station <= cfg.head_of_line_index:
        up_block = [
            float(d_blocked[i]) for i in upstream if np.isfinite(d_blocked[i])
        ]
        if not up_block or max(up_block) < cfg.min_upstream_blocking:
            return Recommendation(
                action=ACTION_CHECK_SUPPLY,
                priority=PRIORITY_MEDIUM,
                target_stations=group_ids,
                title="Check inbound material before investigating a station",
                detail=(
                    f"The evidence points at {top.station_id}, near the head of the "
                    f"line — but the stations upstream of it are not backing up. If "
                    f"{top.station_id} were genuinely slow they would be blocked, "
                    f"holding work they cannot hand over. They are not, which points "
                    f"to material not arriving rather than a station running slow."
                ),
                rationale=(
                    f"Upstream blocking is {max(up_block) * 100:.0f}% of takt above "
                    f"normal, below the {cfg.min_upstream_blocking * 100:.0f}% needed "
                    f"to separate a head-of-line constraint from an inbound shortfall."
                    if up_block
                    else "No instrumented station upstream to test against."
                ),
                units_at_stake=units,
                confidence=float(result.group_prob),
                abstained=True,
                alternatives=[
                    f"If inbound material is flowing normally, then investigate "
                    f"{top.station_id}."
                ],
            )

    # --- a line-wide supply shortfall is cheaper to rule out than a station teardown
    if p_supply >= cfg.supply_precedence:
        return Recommendation(
            action=ACTION_CHECK_SUPPLY,
            priority=PRIORITY_MEDIUM,
            target_stations=[],
            title="Check inbound material before investigating a station",
            detail=(
                "The starvation pattern is close to uniform across the line, which is "
                "what an upstream supply interruption looks like rather than a single "
                "slow station."
            ),
            rationale=(
                f"Line-supply hypothesis holds {p_supply * 100:.0f}% of the posterior, "
                f"against {result.group_prob * 100:.0f}% for the station group "
                f"{', '.join(group_ids)}."
            ),
            units_at_stake=units,
            confidence=p_supply,
            alternatives=[f"If material flow is normal, investigate {', '.join(group_ids)}."],
        )

    # --- not confident enough to name a station
    if not result.confident or result.group_prob < cfg.min_confidence:
        zone = top.zone
        return Recommendation(
            action=ACTION_ESCALATE,
            priority=PRIORITY_LOW,
            target_stations=group_ids,
            title=f"Ambiguous constraint in {zone} -- escalate, do not dispatch",
            detail=(
                f"Evidence points into {zone} but does not separate "
                f"{', '.join(group_ids)}. Sending a technician to one of them is as "
                f"likely to be wrong as right."
            ),
            rationale=(
                f"Posterior mass on the candidate group is {result.group_prob * 100:.0f}%, "
                f"below the {cfg.min_confidence * 100:.0f}% needed to name a station."
            ),
            units_at_stake=units,
            confidence=float(result.group_prob),
            abstained=True,
            alternatives=[
                "Review the zone with the shift lead.",
                "If this persists, it is a candidate location for added instrumentation.",
            ],
        )

    # --- confident, but is it worth acting on?
    if forecast is not None and not forecast.is_binding:
        return Recommendation(
            action=ACTION_MONITOR,
            priority=PRIORITY_LOW,
            target_stations=group_ids,
            title=f"Watch {top.station_id} -- deviating but still inside takt",
            detail=(
                f"{top.station_id} is drifting, but its estimated cycle time is still "
                f"within takt, so the line is not losing output yet."
            ),
            rationale=(
                f"Estimated cycle {forecast.constraint_cycle_s:.0f}s against "
                f"{forecast.takt_s:.0f}s takt."
            ),
            units_at_stake=0.0,
            confidence=float(result.group_prob),
        )

    if units is not None and units < cfg.min_units_at_stake:
        return Recommendation(
            action=ACTION_MONITOR,
            priority=PRIORITY_LOW,
            target_stations=group_ids,
            title=f"Watch {top.station_id} -- impact below action threshold",
            detail=(
                f"Projected loss is {units:.1f} vehicles over "
                f"{forecast.horizon_min:.0f} minutes, which does not justify pulling "
                f"a technician off other work."
            ),
            rationale="Impact below the configured action threshold.",
            units_at_stake=units,
            confidence=float(result.group_prob),
        )

    named = group_ids[: cfg.max_named_group]
    priority = (
        PRIORITY_HIGH
        if result.group_prob >= cfg.high_confidence and (units or 0) >= 4
        else PRIORITY_MEDIUM
    )
    inferred_note = (
        " This station has no sensor, so confirm the condition physically before "
        "committing to a repair."
        if top.is_hidden
        else ""
    )
    return Recommendation(
        action=ACTION_INSPECT,
        priority=priority,
        target_stations=named,
        title=f"Inspect {top.station_id} -- constraining the line now",
        detail=(
            f"Send a technician to {top.station_id} ({top.zone}). Check for tooling "
            f"wear, fixture binding and part-presentation issues that would extend the "
            f"work cycle.{inferred_note}"
        ),
        rationale=(
            f"Flow evidence localises the constraint to {top.station_id} with "
            f"{result.group_prob * 100:.0f}% posterior mass"
            + (
                f"; estimated cycle {forecast.constraint_cycle_s:.0f}s against "
                f"{forecast.takt_s:.0f}s takt, costing about "
                f"{units:.0f} vehicles/hour."
                if forecast
                else "."
            )
        ),
        units_at_stake=units,
        confidence=float(result.group_prob),
        # The fallback must exclude the station already being dispatched to --
        # the candidate group is ordered by line position, not by rank, so the
        # top candidate is not necessarily first in it.
        alternatives=[
            (
                f"If {top.station_id} is clear, check "
                f"{', '.join(g for g in group_ids if g != top.station_id)}."
            )
            if len(group_ids) > 1
            else "If the station is clear, re-check inbound material and part presentation."
        ],
    )


# --------------------------------------------------------------- taxonomy
#
# RippleTwin's actual decision is the Recommendation above -- a specific
# action, a priority, and an abstained flag. The five-way ALLOW / WATCH /
# INVESTIGATE / ESCALATE / ABSTAIN vocabulary below is a coarser read-out
# over that same decision, not a second decision layer: nothing here changes
# what the twin recommends, it only names where that recommendation sits on
# a shorter ladder for a UI or report that wants it.

TAXONOMY_ALLOW = "ALLOW"
TAXONOMY_WATCH = "WATCH"
TAXONOMY_INVESTIGATE = "INVESTIGATE"
TAXONOMY_ESCALATE = "ESCALATE"
TAXONOMY_ABSTAIN = "ABSTAIN"


def taxonomy_label(rec: "Recommendation") -> tuple:
    """Map a ``Recommendation`` onto (taxonomy label, reason).

    ``ESCALATE`` is reserved for the case the twin explicitly named
    (``ACTION_ESCALATE`` -- posterior spread across indistinguishable
    candidates); any other abstention (e.g. checking supply before blaming a
    station) is ``ABSTAIN`` -- the twin had a reason not to name a station,
    stated in ``rec.rationale``.
    """
    if rec.abstained:
        if rec.action == ACTION_ESCALATE:
            return TAXONOMY_ESCALATE, rec.rationale
        return TAXONOMY_ABSTAIN, rec.rationale
    if rec.action == ACTION_MONITOR:
        return TAXONOMY_WATCH, rec.rationale
    if rec.action in (ACTION_INSPECT, ACTION_QUALITY_HOLD, ACTION_CHECK_SUPPLY):
        return TAXONOMY_INVESTIGATE, rec.rationale
    return TAXONOMY_ALLOW, "No action recommended."


#: Maps ``twin.predict`` state names onto the same taxonomy, for the case
#: where no station is confident enough yet to produce a ``Recommendation``
#: at all (state ``NORMAL``/``DEGRADING``/etc. can exist before any alert
#: does). Kept as a plain dict rather than importing ``twin.predict`` here,
#: so this module does not have to know that module's states are spelled
#: exactly this way -- the caller passes the state string it already has.
PREDICT_STATE_TAXONOMY = {
    "NORMAL": TAXONOMY_ALLOW,
    "RECOVERING": TAXONOMY_ALLOW,
    "DEGRADING": TAXONOMY_WATCH,
    "WATCH": TAXONOMY_WATCH,
    "PREDICTED_CONSTRAINT": TAXONOMY_INVESTIGATE,
    "ACTIVE_BOTTLENECK": TAXONOMY_INVESTIGATE,
}


def recommend_quality(
    line: LineTopology,
    station: int,
    m_hat: float,
    rank: int,
    exposure: dict,
    shortlist: List[int],
    cfg: RecommendConfig | None = None,
) -> Recommendation:
    """Turn an inferred quality drift into an advisory action, or abstain."""
    cfg = cfg or RecommendConfig()
    stn = line.stations[station]
    gate = line.next_inspection_after(station + 1)
    gate_id = line.stations[gate].inspection_id if gate is not None else "end of line"
    at_risk = float(exposure.get("expected_extra_defective_units", 0.0))
    ids = [line.stations[i].station_id for i in shortlist]

    if rank > 3 or m_hat < 1.8:
        return Recommendation(
            action=ACTION_ESCALATE,
            priority=PRIORITY_LOW,
            target_stations=ids,
            title=f"Weak quality signal in {stn.zone} -- monitor, do not hold",
            detail=(
                "Failure-mode evidence is not concentrated enough to justify a targeted "
                "hold on any single station."
            ),
            rationale=f"{stn.station_id} ranks #{rank} with an estimated {m_hat:.1f}x rate.",
            units_at_stake=at_risk,
            confidence=1.0 / max(1, rank),
            abstained=True,
        )

    priority = PRIORITY_HIGH if at_risk >= 3 and rank == 1 else PRIORITY_MEDIUM
    inferred_note = (
        " This station has no sensor; the attribution rests on failure-mode signature "
        "and vehicle genealogy."
        if stn.is_hidden
        else ""
    )
    return Recommendation(
        action=ACTION_QUALITY_HOLD,
        priority=priority,
        target_stations=[stn.station_id],
        title=f"Targeted check at {stn.station_id} and audit in-flight units",
        detail=(
            f"Check {stn.station_id} for the condition producing "
            f"{', '.join(list((stn.defect_profile or {}).keys())[:2])}, and pull the "
            f"{exposure.get('vehicles_in_flight', 0)} vehicles already past it for a "
            f"targeted audit at {gate_id} rather than waiting for the normal gate "
            f"sample.{inferred_note}"
        ),
        rationale=(
            f"Defects matching {stn.station_id}'s signature are running {m_hat:.1f}x "
            f"above its normal rate; about {at_risk:.1f} in-flight vehicles are "
            f"expected to carry a defect that would not otherwise exist."
        ),
        units_at_stake=at_risk,
        confidence=1.0 / max(1, rank),
        alternatives=[f"Shortlist if {stn.station_id} is clear: {', '.join(ids[1:3])}."]
        if len(ids) > 1
        else [],
    )
