"""Turning a recommendation into something a plant can actually act on.

Why this module exists
----------------------
Practitioner accounts of failed predictive-maintenance deployments converge on
one point, and it is not accuracy:

    "The real challenge is usually not detecting the problem. It is shortening
    the time between the first warning sign and the moment somebody actually
    takes action."

The named failure modes are organisational, not algorithmic:

* *"a dashboard shows an alert but nobody owns the next action"* -- no owner
* alerts that never become work orders, so the CMMS only records history
* alert fatigue teaching the floor to *"wait until it becomes obvious"*
* and, decisively: *"the supervisor does not want to lose output. Nobody knows
  whether to stop production now or wait."*

A detector that stops at "S08 risk 0.81" walks into every one of those. So this
module converts an alert into a dispatchable job with three things attached that
the alert alone does not have:

1. **An owner** -- a specific role, not "the plant"
2. **A deadline** derived from the forecast rather than from a severity label
3. **The cost of waiting** -- an explicit answer to "now, or at end of shift?"

That third one is the interesting one. RippleTwin already forecasts throughput
loss in vehicles per hour, so the choice between acting now and acting at the
next break is arithmetic, not instinct. Giving a supervisor that number is
worth more than another decimal place of detection accuracy, because it is the
decision they are actually stuck on.

Nothing here writes to a control system. A work order is a request to a person.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..factory.topology import LineTopology
from ..twin.propagate import RippleForecast
from .engine import (
    ACTION_CHECK_SUPPLY,
    ACTION_ESCALATE,
    ACTION_INSPECT,
    ACTION_MONITOR,
    ACTION_QUALITY_HOLD,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    Recommendation,
)

# Who actually does the work. Mapping an action to a role is what turns an
# alert into somebody's job.
OWNER_BY_ACTION = {
    ACTION_INSPECT: "Maintenance technician (line-side)",
    ACTION_QUALITY_HOLD: "Quality technician",
    ACTION_CHECK_SUPPLY: "Material handler / logistics lead",
    ACTION_MONITOR: "Shift supervisor (watch only)",
    ACTION_ESCALATE: "Shift supervisor + process engineer",
}

# How long the job may sit before it stops being worth doing promptly.
# Deliberately coarse: a false sense of precision here is worse than none.
RESPONSE_MINUTES = {
    PRIORITY_HIGH: 15,
    PRIORITY_MEDIUM: 60,
    PRIORITY_LOW: 240,
}


@dataclass
class WaitingCost:
    """What it costs to defer this to the next natural break."""

    units_lost_per_hour: float
    minutes_until_next_break: float
    units_lost_if_deferred: float
    #: True when deferring is defensible on the numbers.
    defer_is_reasonable: bool
    rationale: str = ""


@dataclass
class WorkOrder:
    """A dispatchable job. Advisory, owned, and time-bounded."""

    work_order_id: str
    created_at: str
    #: Role, not a person: rosters change, roles do not.
    owner_role: str
    action: str
    priority: str
    target_stations: List[str]
    title: str
    instruction: str
    #: When this should be actioned by, in wall-clock terms.
    respond_by: str
    respond_within_min: int
    #: Answers "act now, or at the end of the shift?"
    waiting_cost: Optional[dict] = None
    #: What the technician should report back. This is the feedback signal.
    verification: str = ""
    escalate_to: str = ""
    escalate_after_min: int = 0
    requires_approval: bool = True
    #: Provenance, so a work order can always be traced to the alert.
    source_alert: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def as_cmms_payload(self) -> dict:
        """A shape a CMMS can accept without bespoke mapping.

        Deliberately generic: the field names below are the intersection of what
        common maintenance systems expect, so an integration is a field mapping
        rather than a project.
        """
        return {
            "externalId": self.work_order_id,
            "type": "INSPECTION" if self.action == ACTION_INSPECT else "TASK",
            "priority": self.priority,
            "assetIds": self.target_stations,
            "assignedRole": self.owner_role,
            "summary": self.title,
            "description": self.instruction,
            "dueAt": self.respond_by,
            "createdAt": self.created_at,
            "requiresAcknowledgement": True,
            "source": "RippleTwin",
            "sourceReference": self.source_alert.get("entry_id"),
            "verificationPrompt": self.verification,
        }


def _verification_prompt(action: str, stations: List[str]) -> str:
    """What we ask the technician to report back.

    This is not paperwork. It is the only way per-station precision ever gets
    measured, and it is what the ledger feeds back into the twin.
    """
    where = ", ".join(stations) if stations else "the named area"
    if action == ACTION_INSPECT:
        return (
            f"At {where}: was a condition found (tooling wear, fixture binding, "
            f"part presentation)? Record found / not found, and if found "
            f"elsewhere, which station."
        )
    if action == ACTION_QUALITY_HOLD:
        return (
            f"At {where}: did the targeted audit find the predicted defect type? "
            f"Record how many of the held units were affected."
        )
    if action == ACTION_CHECK_SUPPLY:
        return (
            "Was inbound material flowing normally? If yes, this was not a "
            "supply issue and the station shortlist should be investigated."
        )
    return "Record what was observed, including 'nothing found'."


def waiting_cost(
    forecast: Optional[RippleForecast],
    minutes_until_next_break: float = 120.0,
) -> Optional[WaitingCost]:
    """Answer the question the supervisor is actually stuck on.

    Practitioner accounts put it plainly: *"the supervisor does not want to lose
    output. Nobody knows whether to stop production now or wait."* That is a
    quantitative question and the forecast already contains the answer.

    Note what is being compared. Acting is not free either -- but a line-side
    inspection is typically done without stopping the line, whereas the loss
    accrues continuously while the constraint holds. So the number below is the
    cost of *deferring*, and it is deliberately framed as an input to the
    supervisor's judgement rather than as a verdict.
    """
    if forecast is None or not forecast.is_binding:
        return None
    per_hour = float(forecast.units_lost_at_horizon) * (
        60.0 / max(forecast.horizon_min, 1e-6)
    )
    deferred = per_hour * (minutes_until_next_break / 60.0)
    reasonable = deferred < 2.0
    return WaitingCost(
        units_lost_per_hour=per_hour,
        minutes_until_next_break=minutes_until_next_break,
        units_lost_if_deferred=deferred,
        defer_is_reasonable=reasonable,
        rationale=(
            f"Deferring to the next break in {minutes_until_next_break:.0f} min "
            f"costs about {deferred:.1f} vehicles at the current constraint rate."
            + (
                " That is small enough to be a reasonable call."
                if reasonable
                else " That is the cost of waiting, against a line-side check "
                "that does not normally require stopping the line."
            )
        ),
    )


def to_work_order(
    line: LineTopology,
    rec: Recommendation,
    forecast: Optional[RippleForecast] = None,
    now: Optional[datetime] = None,
    minutes_until_next_break: float = 120.0,
    sequence: int = 1,
    source_alert: Optional[dict] = None,
) -> Optional[WorkOrder]:
    """Convert a recommendation into an owned, time-bounded job.

    Returns ``None`` for advisory-only outputs that should not create a job.
    A monitor-only recommendation deliberately does **not** raise a work order:
    manufacturing a task out of "keep an eye on it" is how alert fatigue starts.
    """
    if rec.action == ACTION_MONITOR:
        return None

    now = now or datetime.utcnow()
    within = RESPONSE_MINUTES.get(rec.priority, 60)
    wc = waiting_cost(forecast, minutes_until_next_break)

    # An abstention still needs an owner -- it is a request to investigate a
    # zone, not a dispatch to a station -- but it is never urgent.
    if rec.abstained:
        within = max(within, RESPONSE_MINUTES[PRIORITY_LOW])

    wo_id = f"RT-{now.strftime('%Y%m%d')}-{sequence:04d}"
    respond_by = now + timedelta(minutes=within)

    instruction = rec.detail
    if rec.alternatives:
        instruction += f"\n\nIf that is clear: {rec.alternatives[0]}"
    if wc is not None:
        instruction += f"\n\nCost of waiting: {wc.rationale}"

    return WorkOrder(
        work_order_id=wo_id,
        created_at=now.isoformat(timespec="seconds") + "Z",
        owner_role=OWNER_BY_ACTION.get(rec.action, "Shift supervisor"),
        action=rec.action,
        priority=rec.priority,
        target_stations=list(rec.target_stations),
        title=rec.title,
        instruction=instruction,
        respond_by=respond_by.isoformat(timespec="seconds") + "Z",
        respond_within_min=within,
        waiting_cost=asdict(wc) if wc else None,
        verification=_verification_prompt(rec.action, list(rec.target_stations)),
        escalate_to="Process engineer" if rec.priority == PRIORITY_HIGH
        else "Shift supervisor",
        escalate_after_min=within * 2,
        requires_approval=True,
        source_alert=source_alert or {},
    )
