"""Human-in-the-loop decision ledger.

RippleTwin proposes; a person decides. This module records that decision.

The ledger is append-only and hash-chained. Each entry stores the hash of the
previous one, so an entry cannot be altered or removed after the fact without
breaking every hash that follows it. That matters for two reasons a plant will
care about: a supervisor who overrode the system needs to be able to prove what
the system actually recommended at the time, and any later claim that the model
"would have caught it" has to be checkable against what it really said.

The ledger is also the training signal. Outcomes recorded here -- confirmed,
not-found, false-alarm -- are what recalibrate the twin's precision per station,
which is the feedback loop the Round 1 concept promised.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DECISION_APPROVED = "APPROVED"
DECISION_REJECTED = "REJECTED"
DECISION_DEFERRED = "DEFERRED"
#: A supervisor accepted the alert but redirected the action -- e.g. "check
#: the neighbour instead" or "audit in-flight units, don't inspect now".
#: Recorded with the same record_decision() call as the other three; the
#: note field is where the actual modification is stated, matching how
#: every other decision type already uses it.
DECISION_MODIFIED = "MODIFIED"
#: A supervisor judged the alert plausible but not theirs to resolve alone
#: -- routed to a shift lead / process engineer rather than approved or
#: rejected outright. Distinct from the twin's own ACTION_ESCALATE (an
#: abstention the *model* raises); this is a *human* escalation of a named
#: alert the twin was confident enough to make.
DECISION_ESCALATED = "ESCALATED"

OUTCOME_CONFIRMED = "CONFIRMED"        # the condition was found where predicted
OUTCOME_NOT_FOUND = "NOT_FOUND"        # nothing wrong at the named station
OUTCOME_FOUND_ELSEWHERE = "FOUND_ELSEWHERE"  # real problem, wrong station
OUTCOME_PENDING = "PENDING"

_GENESIS = "0" * 64


@dataclass
class LedgerEntry:
    """One alert, its recommendation, the human decision, and what happened."""

    entry_id: int
    timestamp: str
    run_id: str
    window: int
    alert_type: str                     # FLOW | QUALITY
    station_id: str
    station_tier: str
    is_inferred: bool
    confidence: float
    recommendation: dict
    explanation: dict
    decision: str = DECISION_DEFERRED
    decided_by: str = ""
    decision_note: str = ""
    outcome: str = OUTCOME_PENDING
    outcome_note: str = ""
    actual_station_id: str = ""
    prev_hash: str = _GENESIS
    entry_hash: str = ""

    def compute_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "entry_hash"}
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()


class DecisionLedger:
    """Append-only, hash-chained record of alerts and human decisions."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else None
        self.entries: List[LedgerEntry] = []
        if self.path and self.path.exists():
            self.load()

    # ------------------------------------------------------------------ writing

    def record_alert(
        self,
        run_id: str,
        window: int,
        alert_type: str,
        station_id: str,
        station_tier: str,
        is_inferred: bool,
        confidence: float,
        recommendation: dict,
        explanation: dict,
    ) -> LedgerEntry:
        prev = self.entries[-1].entry_hash if self.entries else _GENESIS
        e = LedgerEntry(
            entry_id=len(self.entries) + 1,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            run_id=run_id,
            window=window,
            alert_type=alert_type,
            station_id=station_id,
            station_tier=station_tier,
            is_inferred=is_inferred,
            confidence=float(confidence),
            recommendation=recommendation,
            explanation=explanation,
            prev_hash=prev,
        )
        e.entry_hash = e.compute_hash()
        self.entries.append(e)
        return e

    def record_decision(
        self, entry_id: int, decision: str, decided_by: str, note: str = ""
    ) -> LedgerEntry:
        """Append the human decision.

        Amending an existing entry would break the chain, so the decision is
        written as a *new* entry that supersedes the original. The original alert
        stays exactly as it was issued -- which is the point of the ledger.
        """
        src = self._get(entry_id)
        prev = self.entries[-1].entry_hash
        e = LedgerEntry(
            entry_id=len(self.entries) + 1,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            run_id=src.run_id,
            window=src.window,
            alert_type=src.alert_type + "_DECISION",
            station_id=src.station_id,
            station_tier=src.station_tier,
            is_inferred=src.is_inferred,
            confidence=src.confidence,
            recommendation={"supersedes_entry_id": entry_id},
            explanation={},
            decision=decision,
            decided_by=decided_by,
            decision_note=note,
            prev_hash=prev,
        )
        e.entry_hash = e.compute_hash()
        self.entries.append(e)
        return e

    def record_outcome(
        self,
        entry_id: int,
        outcome: str,
        note: str = "",
        actual_station_id: str = "",
    ) -> LedgerEntry:
        """Append what was actually found. This is the feedback signal."""
        src = self._get(entry_id)
        prev = self.entries[-1].entry_hash
        e = LedgerEntry(
            entry_id=len(self.entries) + 1,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            run_id=src.run_id,
            window=src.window,
            alert_type=src.alert_type + "_OUTCOME",
            station_id=src.station_id,
            station_tier=src.station_tier,
            is_inferred=src.is_inferred,
            confidence=src.confidence,
            recommendation={"supersedes_entry_id": entry_id},
            explanation={},
            outcome=outcome,
            outcome_note=note,
            actual_station_id=actual_station_id,
            prev_hash=prev,
        )
        e.entry_hash = e.compute_hash()
        self.entries.append(e)
        return e

    def _get(self, entry_id: int) -> LedgerEntry:
        for e in self.entries:
            if e.entry_id == entry_id:
                return e
        raise KeyError(f"no ledger entry {entry_id}")

    # ------------------------------------------------------------------ reading

    def verify(self) -> dict:
        """Re-derive the whole chain and report the first break, if any."""
        prev = _GENESIS
        for e in self.entries:
            if e.prev_hash != prev:
                return {"valid": False, "broken_at": e.entry_id, "reason": "prev_hash mismatch"}
            if e.compute_hash() != e.entry_hash:
                return {"valid": False, "broken_at": e.entry_id, "reason": "content altered"}
            prev = e.entry_hash
        return {"valid": True, "n_entries": len(self.entries)}

    def to_frame(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self.entries])

    def save(self, path: Optional[str | Path] = None) -> Path:
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("no path given for ledger")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            for e in self.entries:
                fh.write(json.dumps(asdict(e), default=str) + "\n")
        self.path = p
        return p

    def load(self, path: Optional[str | Path] = None) -> None:
        p = Path(path) if path else self.path
        self.entries = []
        if p is None or not p.exists():
            return
        for line in p.read_text().splitlines():
            if line.strip():
                self.entries.append(LedgerEntry(**json.loads(line)))


# ------------------------------------------------------------------- feedback


def precision_by_station(ledger: DecisionLedger) -> pd.DataFrame:
    """Per-station realised precision from recorded outcomes.

    This is the loop closing. A station where technicians repeatedly find
    nothing gets a lower trust weight; one where the call keeps being confirmed
    earns a higher one. It is deliberately a simple, auditable count rather than
    an opaque online update, because a supervisor has to be able to see why the
    system now trusts a station more than it did last month.
    """
    df = ledger.to_frame()
    if df.empty or "outcome" not in df.columns:
        return pd.DataFrame(columns=["station_id", "n_outcomes", "n_confirmed", "precision"])
    out = df[df["alert_type"].astype(str).str.endswith("_OUTCOME")]
    out = out[out["outcome"] != OUTCOME_PENDING]
    if out.empty:
        return pd.DataFrame(columns=["station_id", "n_outcomes", "n_confirmed", "precision"])
    g = (
        out.assign(confirmed=(out["outcome"] == OUTCOME_CONFIRMED).astype(int))
        .groupby("station_id")
        .agg(n_outcomes=("confirmed", "size"), n_confirmed=("confirmed", "sum"))
        .reset_index()
    )
    # Laplace smoothing so one unlucky call does not zero a station's trust.
    g["precision"] = (g["n_confirmed"] + 1.0) / (g["n_outcomes"] + 2.0)
    return g
