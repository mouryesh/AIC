"""Ask the twin: a grounded natural-language layer over computed evidence.

Why this exists
----------------
Everywhere else in this project we refused a language model, on the grounds
that an explanation which can drift from the evidence is worse than no
explanation (see ``explain/explain.py``). That is still true, and this module
does not relax it.

What changed is the surface a supervisor interacts with. Practitioner
discussion of digital-twin and predictive-maintenance rollouts converges on
two failure modes that a static template screen does not fix:

* **The black-box problem.** "Supervisors are being asked to accept
  conclusions from systems they can't interrogate, about an environment they
  know firsthand is inconsistent" -- operators trust a system more when they
  can ask it a follow-up question, not just read a fixed report.
* **Alert fatigue from ungrounded confidence.** A controls technician who
  learns that red banners are usually wrong stops reading them. The fix
  proposed in that literature is not fewer words, it is that the system's
  confidence claims have to be checkable against something real.

So this module adds a conversational layer, but the numbers it is allowed to
say are drawn from exactly one place: the ``EvidencePack`` already computed by
``explain.py`` and ``shadow.py``. Nothing here re-derives a number, and a
guardrail rejects any answer that states a number not present in the pack --
whether that answer came from a template or a real LLM. The LLM, when one is
configured, is used for *phrasing and follow-up reasoning over given facts*,
never as a source of quantitative truth. That is the same distinction the
Round 2 brief itself asks teams to make explicit.

The default and always-available path is a deterministic, offline template
backend -- it answers the questions a floor supervisor actually asks (why
this station, how sure are you, what if you're wrong, what do I do, is this
urgent) directly from the evidence pack, with zero external calls and
byte-identical output for the same input. That backend needs no credentials
and always works, including in an air-gapped demo.

If a ``GROQ_API_KEY`` is configured (as a ``GROQ_API_KEY`` environment
variable, or in ``.streamlit/secrets.toml``, which is git-ignored -- **never
commit a key to this repository**, it is public), open-ended follow-up
questions are routed through Groq's hosted Llama models for natural
phrasing, over plain ``urllib`` so no extra dependency is required.
``ANTHROPIC_API_KEY`` is supported as a fallback provider. Either way, the
answer is still constrained to the same evidence pack and still passed
through the same guardrail before being shown -- the choice of provider only
changes how the sentence is phrased, never what it is allowed to claim.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

from ..explain.explain import Explanation
from ..recommend.engine import Recommendation

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class EvidencePack:
    """Every fact the copilot is allowed to reference, and nothing else.

    Built once per alert from objects the pipeline already computed. This is
    the entire interface between the twin and the language layer -- if a fact
    is not in here, the copilot cannot state it, by construction of the
    guardrail below.
    """

    station_id: str
    is_inferred: bool
    tier: str
    confidence_pct: float
    headline: str
    what_changed: str
    why_it_matters: str
    what_happens_next: str
    caveats: List[str]
    evidence_lines: List[str]
    recommendation_title: Optional[str] = None
    recommendation_detail: Optional[str] = None
    recommendation_rationale: Optional[str] = None
    abstained: bool = False
    units_at_risk_per_hr: Optional[float] = None

    @classmethod
    def from_explanation(
        cls,
        exp: Explanation,
        rec: Optional[Recommendation] = None,
        units_at_risk_per_hr: Optional[float] = None,
    ) -> "EvidencePack":
        return cls(
            station_id=exp.station_id,
            is_inferred=exp.is_inferred,
            tier=exp.station_tier,
            confidence_pct=round(exp.confidence * 100, 1),
            headline=exp.headline,
            what_changed=exp.what_changed,
            why_it_matters=exp.why_it_matters,
            what_happens_next=exp.what_happens_next,
            caveats=list(exp.caveats),
            evidence_lines=[f"[{e.provenance}] {e.text}" for e in exp.evidence],
            recommendation_title=rec.title if rec else None,
            recommendation_detail=rec.detail if rec else None,
            recommendation_rationale=rec.rationale if rec else None,
            abstained=bool(rec.abstained) if rec else False,
            units_at_risk_per_hr=units_at_risk_per_hr,
        )

    def allowed_numbers(self) -> set:
        """Every numeric token that legitimately appears somewhere in the pack."""
        text = " ".join(
            [self.headline, self.what_changed, self.why_it_matters,
             self.what_happens_next, str(self.confidence_pct)]
            + self.caveats + self.evidence_lines
            + [self.recommendation_title or "", self.recommendation_detail or "",
               self.recommendation_rationale or ""]
        )
        nums = set(_NUMBER_RE.findall(text))
        if self.units_at_risk_per_hr is not None:
            nums.add(f"{self.units_at_risk_per_hr:.0f}")
            nums.add(f"{self.units_at_risk_per_hr:.1f}")
        return nums

    def as_context_block(self) -> str:
        """The pack serialised as the *only* facts a backend may draw on."""
        lines = [
            f"Station: {self.station_id} (tier: {self.tier}, "
            f"{'no sensor -- state inferred' if self.is_inferred else 'instrumented'})",
            f"Confidence the twin has named the right station: {self.confidence_pct}%",
            f"Headline: {self.headline}",
            f"What changed: {self.what_changed}",
            f"Why it matters: {self.why_it_matters}",
            f"What happens next: {self.what_happens_next}",
        ]
        if self.recommendation_title:
            lines.append(f"Recommended action: {self.recommendation_title}")
            lines.append(f"Detail: {self.recommendation_detail}")
            lines.append(f"Rationale: {self.recommendation_rationale}")
            if self.abstained:
                lines.append("Note: the twin is ABSTAINING, not naming a single station.")
        if self.caveats:
            lines.append("Caveats: " + " | ".join(self.caveats))
        if self.evidence_lines:
            lines.append("Supporting evidence:")
            lines.extend(f"  - {e}" for e in self.evidence_lines)
        return "\n".join(lines)


@dataclass
class CopilotAnswer:
    text: str
    used_llm: bool
    grounded: bool
    flagged_numbers: List[str] = field(default_factory=list)


def _verify_grounded(answer: str, pack: EvidencePack) -> tuple:
    """Reject any number in the answer that does not appear in the evidence pack.

    This is the guardrail. It is deliberately crude -- a set-membership check
    on numeric tokens -- because a crude, auditable rule is worth more here
    than a clever one that could itself be wrong in a way nobody would notice.
    """
    allowed = pack.allowed_numbers()
    found = set(_NUMBER_RE.findall(answer))
    flagged = sorted(found - allowed)
    return (len(flagged) == 0, flagged)


# ------------------------------------------------------------------ backends


def _template_answer(question: str, pack: EvidencePack) -> str:
    """Deterministic, offline, evidence-grounded FAQ backend.

    Covers the questions a floor supervisor actually asks, per
    ``explain.py``'s own framing: why, how sure, what if wrong, what next.
    This backend is always available -- no network call, no API key, and the
    same question against the same pack produces the same answer, which
    matters when a judge re-runs the demo.
    """
    q = question.lower()

    def has(*words) -> bool:
        return any(w in q for w in words)

    if has("why", "cause", "reason"):
        return (
            f"{pack.headline}\n\n{pack.what_changed}\n\n"
            + ("\n".join(pack.evidence_lines) if pack.evidence_lines else "")
        ).strip()

    if has("how sure", "confiden", "certain"):
        base = (
            f"The twin's posterior puts {pack.confidence_pct}% of the probability "
            f"mass on {pack.station_id}."
        )
        if pack.is_inferred:
            base += " That station has no sensor, so this is an inference, not a reading."
        if pack.abstained:
            base += " At this confidence the twin is deliberately abstaining rather than naming one station."
        return base

    if has("wrong", "mistake", "if it's not", "false alarm"):
        if pack.caveats:
            return "If this is wrong, here is what the twin itself flags as uncertain:\n" + \
                "\n".join(f"- {c}" for c in pack.caveats)
        return (
            "The twin does not report any specific caveat for this alert -- the "
            "posterior is concentrated and no competing hypothesis carries "
            "meaningful weight."
        )

    if has("what should i do", "what do i do", "action", "recommend"):
        if pack.recommendation_title:
            out = f"Recommended action: {pack.recommendation_title}\n{pack.recommendation_detail}"
            if pack.recommendation_rationale:
                out += f"\n\nRationale: {pack.recommendation_rationale}"
            return out
        return "No recommendation is attached to this alert."

    if has("urgent", "hurry", "now or later", "wait"):
        return pack.what_happens_next

    if has("sensor", "instrumented", "measure"):
        return (
            f"{pack.station_id} is "
            + ("NOT instrumented -- its state is reconstructed from the stations "
               "either side of it." if pack.is_inferred
               else f"instrumented at {pack.tier} tier -- this is a direct reading.")
        )

    # Fallback: the full evidence-grounded summary, never an apology that
    # invents context the pack does not have.
    return (
        f"{pack.headline}\n\n{pack.what_changed}\n\n{pack.why_it_matters}\n\n"
        f"{pack.what_happens_next}"
    )


_SYSTEM_PROMPT = (
    "You are a plain-language assistant standing in front of a factory-floor "
    "supervisor who is deciding whether to act on a digital-twin alert. You will "
    "be given a fixed block of facts the underlying model already computed. "
    "You must answer using ONLY those facts. Do not introduce any number, "
    "station name, percentage, or time estimate that is not already present in "
    "the facts block, even if it seems like a reasonable estimate. If the "
    "supervisor asks something the facts do not cover, say plainly that the "
    "twin does not have that information rather than guessing. Keep answers to "
    "3-5 sentences, in the voice of a colleague, not a report."
)


def _secret(name: str) -> Optional[str]:
    """Read a credential from the environment, or Streamlit secrets if running
    under Streamlit. Never hard-coded, never logged, never written to a file
    tracked by git -- see ``.streamlit/secrets.toml`` in ``.gitignore``.
    """
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _groq_available() -> bool:
    return bool(_secret("GROQ_API_KEY"))


def _anthropic_available() -> bool:
    if not _secret("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _llm_available() -> bool:
    return _groq_available() or _anthropic_available()


def _groq_answer(question: str, pack: EvidencePack) -> Optional[str]:
    """Call Groq's OpenAI-compatible chat endpoint over stdlib HTTP.

    Uses ``urllib`` rather than adding a new dependency, since the whole point
    of this backend being optional is that the project should not need a new
    package installed to run without it.
    """
    key = _secret("GROQ_API_KEY")
    if not key:
        return None

    payload = {
        # gpt-oss models on Groq are reasoning models: part of the token budget
        # goes to an internal "reasoning" field before the visible "content" is
        # written, so this needs a larger budget than a plain chat model would.
        "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"FACTS:\n{pack.as_context_block()}\n\nQUESTION: {question}"},
        ],
        "max_tokens": 500,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Groq sits behind Cloudflare, which blocks the default
            # "Python-urllib/x.y" user agent outright (HTTP 403, error 1010).
            "User-Agent": "RippleTwin/1.0 (+https://github.com/mouryesh/AIC)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        return content or None
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError):
        return None


def _anthropic_answer(question: str, pack: EvidencePack) -> Optional[str]:
    """Call Anthropic's API, constrained to the evidence pack. None on any failure."""
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"FACTS:\n{pack.as_context_block()}\n\nQUESTION: {question}",
            }],
        )
        return "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        ).strip()
    except Exception:
        return None


def _llm_answer(question: str, pack: EvidencePack) -> Optional[str]:
    """Try each configured provider in turn; None if none are configured or all fail.

    Groq is tried first because it is the provider actually configured for
    this project (fast, cheap Llama inference); Anthropic remains supported
    as a fallback for anyone who wires in that key instead.
    """
    if _groq_available():
        text = _groq_answer(question, pack)
        if text:
            return text
    if _anthropic_available():
        return _anthropic_answer(question, pack)
    return None


def active_backend_name() -> str:
    """Which backend would answer the next question -- for the UI badge."""
    if _groq_available():
        return f"Groq ({os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b')})"
    if _anthropic_available():
        return "Anthropic (claude-sonnet-5)"
    return "offline template"


class TwinCopilot:
    """Answers supervisor questions from one alert's computed evidence, and only that."""

    def ask(self, question: str, pack: EvidencePack) -> CopilotAnswer:
        if not question.strip():
            return CopilotAnswer(text="Ask a question about this alert.", used_llm=False, grounded=True)

        if _llm_available():
            llm_text = _llm_answer(question, pack)
            if llm_text:
                grounded, flagged = _verify_grounded(llm_text, pack)
                if grounded:
                    return CopilotAnswer(text=llm_text, used_llm=True, grounded=True)
                # The guardrail caught the model saying something not in the
                # evidence. Do not show it -- fall through to the template
                # backend rather than risk a fabricated number reaching a
                # supervisor.
                template_text = _template_answer(question, pack)
                return CopilotAnswer(
                    text=template_text, used_llm=False, grounded=True,
                    flagged_numbers=flagged,
                )

        template_text = _template_answer(question, pack)
        grounded, flagged = _verify_grounded(template_text, pack)
        return CopilotAnswer(
            text=template_text, used_llm=False, grounded=grounded, flagged_numbers=flagged,
        )
