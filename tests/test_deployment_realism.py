"""Tests for the parts that decide whether this could run in a real plant.

These cover the two failure modes practitioner accounts name most often: the
data layer, and alerts that nobody owns.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from rippletwin.factory.topology import build_line
from rippletwin.integrate.contract import (
    DATA_CONTRACT,
    Capability,
    assess_readiness,
    contract_frame,
)
from rippletwin.recommend.dispatch import (
    OWNER_BY_ACTION,
    to_work_order,
    waiting_cost,
)
from rippletwin.recommend.engine import (
    ACTION_INSPECT,
    ACTION_MONITOR,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    Recommendation,
)
from rippletwin.twin.propagate import forecast_ripple

CONFIG = "configs/line_42.yaml"


@pytest.fixture(scope="module")
def line():
    return build_line(CONFIG, seed=7)


# ------------------------------------------------------------- data contract


def test_contract_is_complete_and_well_formed():
    keys = [s.key for s in DATA_CONTRACT]
    assert len(keys) == len(set(keys)), "duplicate signal keys"
    for s in DATA_CONTRACT:
        assert s.purpose and s.if_missing, (
            f"{s.key} must say what it is for and what happens without it"
        )
        assert s.interface, f"{s.key} must name a real interface"
        assert 0 <= s.purdue_level <= 5
    assert len(contract_frame()) == len(DATA_CONTRACT)


def test_a_well_instrumented_plant_gets_full_capability():
    have = [s.key for s in DATA_CONTRACT]
    r = assess_readiness(have, n_stations=42, n_stations_with_state=32)
    assert r.capability is Capability.FULL
    assert not r.blockers


def test_missing_quality_data_degrades_rather_than_blocks():
    have = [s.key for s in DATA_CONTRACT if s.key != "inspection_results"]
    r = assess_readiness(have, n_stations=42, n_stations_with_state=32)
    assert r.capability is Capability.FLOW_ONLY
    assert not r.blockers
    assert any("quality" in w.lower() or "defect" in w.lower() for w in r.warnings)


def test_missing_topology_is_a_blocker():
    have = [s.key for s in DATA_CONTRACT if s.key != "line_topology"]
    r = assess_readiness(have, n_stations=42, n_stations_with_state=32)
    assert r.capability is not Capability.FULL
    assert r.blockers


def test_too_few_instrumented_stations_kills_the_flow_path_only():
    """The flow mechanism needs sensors either side of a blind station.

    The quality path does not: it runs off gate results and build sequence, so
    it is coverage-independent — which our evaluation confirms, since its
    numbers are identical at every coverage level. So one instrumented station
    is fatal to flow localisation and irrelevant to defect attribution, and the
    assessment must say exactly that rather than declaring the plant hopeless.
    """
    have = [s.key for s in DATA_CONTRACT]
    r = assess_readiness(have, n_stations=42, n_stations_with_state=1)
    assert r.capability is Capability.QUALITY_ONLY
    assert any("FLOW PATH" in b for b in r.blockers)


def test_no_usable_data_at_all_is_not_viable():
    r = assess_readiness([], n_stations=42, n_stations_with_state=0)
    assert r.capability is Capability.NOT_VIABLE
    assert r.blockers


def test_full_coverage_is_reported_as_not_needing_us():
    """Honesty check: at 100% coverage we say a conventional twin will do."""
    have = [s.key for s in DATA_CONTRACT]
    r = assess_readiness(have, n_stations=42, n_stations_with_state=42)
    assert any("nothing to add" in n or "conventional" in n for n in r.notes)


def test_clock_skew_is_flagged():
    have = [s.key for s in DATA_CONTRACT]
    r = assess_readiness(have, n_stations=42, n_stations_with_state=32,
                         clock_sync_s=15.0)
    assert any("skew" in w.lower() for w in r.warnings)


def test_readiness_report_renders():
    have = [s.key for s in DATA_CONTRACT if s.key != "process_channels"]
    r = assess_readiness(have, n_stations=42, n_stations_with_state=32)
    assert "CAPABILITY" in r.summary()
    assert len(r.to_frame()) == len(DATA_CONTRACT)


# --------------------------------------------------------------- work orders


def _rec(action=ACTION_INSPECT, priority=PRIORITY_HIGH, abstained=False):
    return Recommendation(
        action=action, priority=priority, target_stations=["S08"],
        title="Inspect S08", detail="Send a technician to S08.",
        rationale="flow evidence", units_at_stake=12.0, confidence=0.9,
        abstained=abstained, alternatives=["If S08 is clear, check S09."],
    )


def test_every_actionable_recommendation_gets_an_owner(line):
    """The failure mode this exists to prevent: an alert nobody owns."""
    for action in OWNER_BY_ACTION:
        if action == ACTION_MONITOR:
            continue
        wo = to_work_order(line, _rec(action=action),
                           forecast_ripple(line, 8, 76.0), sequence=1)
        assert wo is not None
        assert wo.owner_role
        assert wo.respond_by
        assert wo.verification, "a work order must ask for a result back"


def test_monitor_does_not_create_a_work_order(line):
    """Manufacturing a task out of 'keep an eye on it' is how fatigue starts."""
    wo = to_work_order(line, _rec(action=ACTION_MONITOR, priority=PRIORITY_LOW),
                       None, sequence=1)
    assert wo is None


def test_priority_drives_the_deadline(line):
    fc = forecast_ripple(line, 8, 76.0)
    hi = to_work_order(line, _rec(priority=PRIORITY_HIGH), fc, sequence=1)
    lo = to_work_order(line, _rec(priority=PRIORITY_LOW), fc, sequence=2)
    assert hi.respond_within_min < lo.respond_within_min


def test_abstention_is_never_urgent(line):
    fc = forecast_ripple(line, 8, 76.0)
    a = to_work_order(line, _rec(priority=PRIORITY_HIGH, abstained=True), fc,
                      sequence=1)
    assert a.respond_within_min >= 240


def test_waiting_cost_answers_now_or_later(line):
    """The question practitioners say supervisors are actually stuck on."""
    fc = forecast_ripple(line, 8, 78.0, horizon_min=60.0)
    wc = waiting_cost(fc, minutes_until_next_break=120.0)
    assert wc is not None
    assert wc.units_lost_per_hour > 0
    assert wc.units_lost_if_deferred == pytest.approx(
        wc.units_lost_per_hour * 2.0, rel=1e-6
    )
    assert wc.rationale


def test_no_waiting_cost_when_the_constraint_does_not_bind(line):
    fc = forecast_ripple(line, 8, 50.0)  # inside takt
    assert waiting_cost(fc) is None


def test_cmms_payload_is_mappable(line):
    wo = to_work_order(line, _rec(), forecast_ripple(line, 8, 76.0), sequence=7)
    p = wo.as_cmms_payload()
    for k in ("externalId", "priority", "assetIds", "assignedRole", "summary",
              "dueAt", "requiresAcknowledgement", "source"):
        assert k in p
    assert p["source"] == "RippleTwin"
    assert p["assetIds"] == ["S08"]


def test_work_order_never_writes_to_control_systems(line):
    """Advisory only — nothing in the payload touches a control system."""
    wo = to_work_order(line, _rec(), forecast_ripple(line, 8, 76.0), sequence=1)
    assert wo.requires_approval
    payload = str(wo.as_cmms_payload()).lower()
    assert not any(f in payload for f in ("setpoint", "stop_line", "override"))
