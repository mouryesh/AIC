"""Tests for the grounded ask-the-twin copilot.

The property that matters most here is not "does it answer nicely" -- it is
"can it ever say a number that was not already computed". That is what the
guardrail exists to prevent, and it is what most of this file checks.
"""

from __future__ import annotations

import json

import pytest

from rippletwin.copilot import ask as ask_mod
from rippletwin.copilot.ask import EvidencePack, TwinCopilot, _verify_grounded


@pytest.fixture(autouse=True)
def no_incidental_llm_calls(monkeypatch):
    """Force the offline backend by default, regardless of the machine running
    the tests.

    Without this, a developer's local ``.streamlit/secrets.toml`` (which is
    git-ignored, so CI and a fresh clone never have it) would silently route
    every test through a real network call to a real LLM -- exactly the kind
    of non-determinism this project's own testing philosophy rejects
    elsewhere. Tests that specifically want to exercise the LLM path
    re-enable it explicitly and mock the network call.
    """
    monkeypatch.setattr(ask_mod, "_groq_available", lambda: False)
    monkeypatch.setattr(ask_mod, "_anthropic_available", lambda: False)


def _pack(**overrides) -> EvidencePack:
    defaults = dict(
        station_id="S02",
        is_inferred=True,
        tier="MANUAL",
        confidence_pct=88.0,
        headline="S02 (BODY) -- inferred constraint, no sensor at this station",
        what_changed="S02 has no instrumentation. Its state is reconstructed from neighbours.",
        why_it_matters="At an estimated 62s cycle against a 60s takt, this caps the line at 45 vehicles/hour.",
        what_happens_next="About 30 vehicles lost over the next 60 minutes if nothing changes.",
        caveats=["This station's state is inferred, not measured. Confirm on the floor before acting."],
        evidence_lines=["[OBSERVED] S03 is starved 40% of takt above its normal level."],
        recommendation_title="Targeted audit at S02",
        recommendation_detail="Send a technician to inspect S02 within 20 minutes.",
        recommendation_rationale="Posterior mass concentrated on S02 at 88%.",
        abstained=False,
        units_at_risk_per_hr=12.5,
    )
    defaults.update(overrides)
    return EvidencePack(**defaults)


def test_template_answers_are_grounded():
    pack = _pack()
    copilot = TwinCopilot()
    for q in [
        "why do you think it's this station",
        "how confident are you",
        "what if you're wrong",
        "what should I do",
        "is this urgent",
        "is it sensored",
        "tell me about this alert",
    ]:
        answer = copilot.ask(q, pack)
        assert answer.text.strip()
        assert answer.grounded, f"ungrounded numbers {answer.flagged_numbers} for question {q!r}"
        assert not answer.used_llm  # no API key in the test environment


def test_guardrail_rejects_fabricated_numbers():
    pack = _pack()
    fabricated = "This station has failed 17 times this month and will cost $4200 to fix."
    grounded, flagged = _verify_grounded(fabricated, pack)
    assert not grounded
    assert "17" in flagged
    assert "4200" in flagged


def test_guardrail_accepts_numbers_present_in_the_pack():
    pack = _pack()
    answer = "The twin is 88.0% confident and expects roughly 30 vehicles at risk."
    grounded, flagged = _verify_grounded(answer, pack)
    assert grounded, flagged


def test_confidence_question_reflects_inferred_flag():
    pack = _pack(is_inferred=True)
    copilot = TwinCopilot()
    answer = copilot.ask("how confident are you", pack)
    assert "no sensor" in answer.text.lower() or "inference" in answer.text.lower()


def test_abstained_alert_is_reflected_in_answer():
    pack = _pack(abstained=True, confidence_pct=40.0)
    copilot = TwinCopilot()
    answer = copilot.ask("how confident are you", pack)
    assert "abstain" in answer.text.lower()


def test_from_explanation_only_carries_computed_numbers(monkeypatch):
    from rippletwin.explain.explain import Explanation, EvidenceItem

    exp = Explanation(
        station_id="S07",
        station_tier="BASIC",
        is_inferred=False,
        headline="S07 -- measured constraint",
        what_changed="S07's own cycle time rose.",
        why_it_matters="Caps the line at 50 vehicles/hour.",
        what_happens_next="No projected loss.",
        confidence_text="92% posterior",
        confidence=0.92,
        evidence=[EvidenceItem("OBSERVED", "S07", "cycle_time", 3.1, "sigma", "S07 cycle time is 3.1 sigma high.")],
        caveats=[],
    )
    pack = EvidencePack.from_explanation(exp)
    assert pack.station_id == "S07"
    assert "92.0" in pack.allowed_numbers() or "92" in pack.allowed_numbers()

    copilot = TwinCopilot()
    answer = copilot.ask("why is this happening", pack)
    assert answer.grounded


def test_groq_backend_used_when_key_present_and_answer_is_grounded(monkeypatch):
    """Mocks the HTTP call -- never hits the real network in tests."""
    monkeypatch.setattr(ask_mod, "_groq_available", lambda: True)
    monkeypatch.setattr(ask_mod, "_secret", lambda name: "test-key-not-real" if name == "GROQ_API_KEY" else None)

    pack = _pack()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            content = (
                f"The twin is {pack.confidence_pct}% confident and expects "
                f"about {pack.units_at_risk_per_hr:.0f} vehicles at risk."
            )
            return json.dumps(
                {"choices": [{"message": {"content": content}}]}
            ).encode("utf-8")

    monkeypatch.setattr(
        ask_mod.urllib.request, "urlopen", lambda req, timeout=15: FakeResponse()
    )

    assert ask_mod.active_backend_name().startswith("Groq")
    answer = TwinCopilot().ask("how confident are you", pack)
    assert answer.used_llm
    assert answer.grounded
    assert "88" in answer.text


def test_groq_fabricated_number_is_caught_and_falls_back_to_template(monkeypatch):
    monkeypatch.setattr(ask_mod, "_groq_available", lambda: True)
    monkeypatch.setattr(ask_mod, "_secret", lambda name: "test-key-not-real" if name == "GROQ_API_KEY" else None)
    pack = _pack()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            content = "This station has failed 17 times and costs $4200 to fix."
            return json.dumps(
                {"choices": [{"message": {"content": content}}]}
            ).encode("utf-8")

    monkeypatch.setattr(
        ask_mod.urllib.request, "urlopen", lambda req, timeout=15: FakeResponse()
    )

    answer = TwinCopilot().ask("how confident are you", pack)
    assert not answer.used_llm  # rejected, fell back to template
    assert answer.grounded  # the template that replaced it is grounded
    assert "17" in answer.flagged_numbers
    assert "4200" in answer.flagged_numbers
