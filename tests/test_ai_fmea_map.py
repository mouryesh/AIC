"""Tests for the one place a language model is allowed near the quality path.

The module reads a plant's control plan and proposes a station-to-defect-type
map. It is the highest-leverage AI integration in the project because it
removes a configuration barrier that would otherwise stop the quality path
before a pilot starts -- and it is safe because the output is an offline draft
a process engineer signs off, not a live input.

These tests pin the two properties that make that claim true: the model can
never name an asset that does not exist, and nothing is silently dropped.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rippletwin.ai import fmea_map
from rippletwin.ai.fmea_map import (
    MapProposalSet,
    _propose_llm,
    _score,
    _token_specificity,
    propose_defect_map,
)

STATIONS = [f"S{i:02d}" for i in range(1, 9)]
DEFECTS = [
    "weld_gap", "panel_misalign", "sealer_void", "torque_low",
    "paint_defect", "trim_gap", "electrical_fault", "leak",
]


@pytest.fixture
def control_plan():
    """A control plan written the way a plant writes one."""
    return pd.DataFrame(
        [
            ("S01", "Spot weld penetration insufficient at flange", 7),
            ("S02", "Sealer bead skips / voids in coverage", 5),
            ("S03", "Fastener not tightened to specified torque angle", 8),
            ("S04", "Orange peel and dirt inclusion in clearcoat", 4),
            ("S05", "Interior garnish clip not fully engaged, rattle", 3),
            ("S06", "Harness connector pin not latched, continuity fail", 7),
            ("S99", "Operation that is not on this line", 5),
            ("S07", "Operator ergonomics review pending", 2),
        ],
        # Deliberately awkward headers: these are not valid Python identifiers.
        columns=["Op No.", "Potential Failure Mode", "Sev"],
    )


def _top(res: MapProposalSet, station: str) -> str:
    prof = res.profiles()[station]
    return max(prof, key=prof.get)


def test_each_failure_mode_maps_to_the_right_defect_code(control_plan):
    res = propose_defect_map(control_plan, STATIONS, DEFECTS)
    assert _top(res, "S01") == "weld_gap"
    assert _top(res, "S02") == "sealer_void"
    assert _top(res, "S03") == "torque_low"
    assert _top(res, "S04") == "paint_defect"
    assert _top(res, "S05") == "trim_gap"
    assert _top(res, "S06") == "electrical_fault"


def test_awkward_column_headers_are_found(control_plan):
    """Regression: itertuples renames "Op No." to a positional _0, so getattr
    returned "" for every row and the mapper silently produced nothing."""
    res = propose_defect_map(control_plan, STATIONS, DEFECTS)
    assert len(res.proposals) > 0


def test_a_station_not_on_the_line_is_never_invented(control_plan):
    res = propose_defect_map(control_plan, STATIONS, DEFECTS)
    assert all(p.station_id in STATIONS for p in res.proposals)
    assert any("S99" in u for u in res.unmapped)


def test_an_unmappable_failure_mode_is_surfaced_not_dropped(control_plan):
    """An unmapped failure mode is a station the quality path stays blind to."""
    res = propose_defect_map(control_plan, STATIONS, DEFECTS)
    assert any("ergonomics" in u for u in res.unmapped)


def test_every_proposal_carries_its_source_text(control_plan):
    res = propose_defect_map(control_plan, STATIONS, DEFECTS)
    assert all(p.evidence.strip() for p in res.proposals)


def test_mechanical_pin_is_not_read_as_an_electrical_one():
    """"Panel locating pin worn" scored 1.00 for electrical_fault on the bare
    synonym "pin" before specificity weighting and evidence mass were added."""
    text = "Panel locating pin worn, dimension out of datum"
    spec = _token_specificity(fmea_map.DEFECT_SYNONYMS)
    mis = _score(text, "panel_misalign", fmea_map.DEFECT_SYNONYMS, spec)
    ele = _score(text, "electrical_fault", fmea_map.DEFECT_SYNONYMS, spec)
    assert mis > ele


def test_profiles_are_normalised(control_plan):
    res = propose_defect_map(control_plan, STATIONS, DEFECTS)
    for prof in res.profiles().values():
        assert abs(sum(prof.values()) - 1.0) < 1e-9


def test_severity_weights_the_proposal():
    """A higher-severity failure mode should carry more of a station's profile."""
    plan = pd.DataFrame(
        [("S01", "Spot weld penetration insufficient", 9),
         ("S01", "Sealer bead void", 2)],
        columns=["Operation", "Failure Mode", "Severity"],
    )
    res = propose_defect_map(plan, STATIONS, DEFECTS)
    prof = res.profiles()["S01"]
    assert prof["weld_gap"] > prof["sealer_void"]


def test_output_is_labelled_a_draft(control_plan):
    y = propose_defect_map(control_plan, STATIONS, DEFECTS).to_yaml()
    assert "DRAFT, NOT A CONFIGURATION" in y
    assert "engineer must confirm" in y
    assert "defect_profiles:" in y


def test_empty_result_is_still_a_usable_frame():
    empty = MapProposalSet()
    f = empty.to_frame()
    assert list(f.columns) == MapProposalSet.COLUMNS
    assert len(f) == 0


def test_free_text_without_a_model_fails_loudly_not_silently():
    with pytest.raises(ValueError, match="free-text"):
        propose_defect_map("some prose FMEA", STATIONS, DEFECTS)


def test_missing_columns_are_reported():
    plan = pd.DataFrame({"something": [1], "irrelevant": [2]})
    with pytest.raises(ValueError, match="failure-mode column"):
        propose_defect_map(plan, STATIONS, DEFECTS)


# ------------------------------------------------------------ the guardrail


def test_model_output_outside_the_vocabulary_is_rejected(monkeypatch):
    """The property the whole design rests on: a model cannot name an asset
    that does not exist, because anything outside the supplied lists is
    discarded rather than trusted."""
    monkeypatch.setattr(fmea_map, "_llm_json", lambda prompt: [
        {"station_id": "S01", "defect_type": "weld_gap", "weight": 0.9,
         "evidence": "real"},
        {"station_id": "S_DOES_NOT_EXIST", "defect_type": "weld_gap",
         "weight": 0.9, "evidence": "invented station"},
        {"station_id": "S02", "defect_type": "made_up_defect_code",
         "weight": 0.9, "evidence": "invented code"},
    ])
    res = _propose_llm("irrelevant", STATIONS, DEFECTS)
    assert len(res.proposals) == 1
    assert res.proposals[0].station_id == "S01"
    assert len(res.unmapped) == 2
    assert all("rejected" in u for u in res.unmapped)


def test_malformed_model_output_does_not_raise(monkeypatch):
    monkeypatch.setattr(fmea_map, "_llm_json", lambda prompt: [
        {"no_station_key": True},
        {"station_id": "S01", "defect_type": "weld_gap", "weight": "not a number"},
    ])
    res = _propose_llm("irrelevant", STATIONS, DEFECTS)
    assert res.proposals == []


def test_a_model_failure_falls_back_rather_than_stopping_a_pilot(
    monkeypatch, control_plan
):
    """A network failure must degrade to the offline matcher, not fail."""
    monkeypatch.setattr(fmea_map, "llm_available", lambda: True)
    monkeypatch.setattr(fmea_map, "_llm_json", lambda prompt: None)
    res = propose_defect_map(control_plan, STATIONS, DEFECTS, use_llm=True)
    assert res.backend == "deterministic"
    assert len(res.proposals) > 0


def test_deterministic_backend_needs_no_credentials(monkeypatch, control_plan):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = propose_defect_map(control_plan, STATIONS, DEFECTS)
    assert res.backend == "deterministic"


def test_deterministic_output_is_reproducible(control_plan):
    a = propose_defect_map(control_plan, STATIONS, DEFECTS).to_yaml()
    b = propose_defect_map(control_plan, STATIONS, DEFECTS).to_yaml()
    assert a == b
