"""Turning a plant's control plan into the defect map the quality path needs.

The deployment problem this solves
----------------------------------
RippleTwin's quality path attributes a defect to the station that caused it by
asking which station the failing units passed through that the passing ones did
not. That works far better when it knows which stations can *physically*
produce which defect: a sealer station cannot cause a torque fault, and using
that fact narrows attribution from "somewhere in body shop" to a handful of
candidates.

In our simulator that knowledge is ``Station.defect_profile``, written by hand
in a YAML file. In a real plant it exists too -- it is the **process FMEA** and
the **control plan**, documents every automotive plant maintains because
customers and auditors require them. But it exists as a spreadsheet with columns
like ``Operation``, ``Potential Failure Mode``, ``Severity``, in prose written
by process engineers over a decade.

Nobody is going to retype that into our YAML format to trial an unproven tool.
The Phase 0 readiness report says so out loud: without this map the quality path
degrades from station-level to zone-level attribution. So this module reads the
plant's own document and proposes the map, as a **draft for an engineer to sign
off**.

Why a language model belongs here specifically
----------------------------------------------
This is the shape of task an LLM is genuinely good at and where being wrong is
cheap: fuzzy free text mapped onto a controlled vocabulary, run once, offline,
before anything is deployed, with a human approving the output. A mistake costs
review time. It cannot reach the localisation, the detection threshold, or a
work order.

Contrast that with the physics: an LLM in the propagation model would be a
non-deterministic component inside a live decision, which is exactly what the
industrial-agent literature reports as unready.

Two backends, and the default needs no credentials
--------------------------------------------------
**Deterministic (always available).** Failure-mode text is matched against the
plant's defect-code vocabulary by token overlap, using a synonym table of
ordinary automotive failure language. No network, no key, byte-identical output
for the same input. This is the default and it handles a normally-structured
control plan.

**LLM (optional).** For a control plan whose failure modes are written as prose,
or whose columns are not recognisable, ``GROQ_API_KEY`` or ``ANTHROPIC_API_KEY``
routes the text through a model. The guardrail is the important part: the model
is given the station list and the defect vocabulary, and **any station or defect
code it returns that is not in those lists is discarded**. It cannot invent an
asset. It can only match text it was given to codes it was given.

Every proposal carries the source row it came from, so an engineer reviewing the
draft can check it against the document rather than trust it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

# Ordinary failure language as it appears in an automotive control plan, mapped
# to the kind of defect code a plant's quality system uses. This is domain
# vocabulary, not a model: it is what makes the offline path actually work.
DEFECT_SYNONYMS: Dict[str, List[str]] = {
    "weld_gap": [
        "weld", "welding", "spot weld", "gap", "penetration", "nugget",
        "expulsion", "spatter", "joint", "fusion",
    ],
    "panel_misalign": [
        "align", "alignment", "misalign", "fit", "flush", "gap and flush",
        "locate", "locating", "position", "dimension", "datum", "clearance",
    ],
    "sealer_void": [
        "sealer", "sealant", "seal", "void", "bead", "adhesive", "gap in bead",
        "coverage", "skip", "porosity",
    ],
    "torque_low": [
        "torque", "tighten", "tightening", "fastener", "bolt", "screw", "nut",
        "clamp load", "angle", "cross thread", "loose",
    ],
    "paint_defect": [
        "paint", "coating", "orange peel", "run", "sag", "dirt", "crater",
        "colour", "color", "finish", "gloss", "primer", "clearcoat",
    ],
    "trim_gap": [
        "trim", "clip", "moulding", "molding", "garnish", "panel gap",
        "rattle", "squeak", "interior",
    ],
    "electrical_fault": [
        # "pin" alone is deliberately absent: a locating pin is mechanical.
        # Ambiguous bare nouns belong in a phrase, not on their own.
        "electrical", "connector", "harness", "connector pin", "continuity",
        "short circuit", "open circuit", "wiring", "terminal", "unlatched",
    ],
    "leak": [
        "leak", "water", "ingress", "seal failure", "pressure", "drip",
    ],
}

#: Column names a control plan actually uses, in the order we try them.
STATION_COLUMNS = [
    "operation", "op", "op_no", "station", "station_id", "process",
    "workstation", "equipment", "op_code", "operation_number",
]
FAILURE_COLUMNS = [
    "potential_failure_mode", "failure_mode", "failure", "defect",
    "defect_mode", "potential_defect", "characteristic", "failure_description",
    "potential_failure", "mode",
]
SEVERITY_COLUMNS = ["severity", "sev", "rpn", "risk", "occurrence", "occ"]


@dataclass
class DefectMapProposal:
    """One proposed station-to-defect link, with where it came from."""

    station_id: str
    defect_type: str
    weight: float
    #: The text in the plant's own document that produced this.
    evidence: str
    #: 0..1. Token-overlap strength for the deterministic path.
    confidence: float
    method: str = "deterministic"

    def as_row(self) -> dict:
        return {
            "station_id": self.station_id,
            "defect_type": self.defect_type,
            "weight": round(self.weight, 4),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "evidence": self.evidence,
        }


@dataclass
class MapProposalSet:
    """A reviewable draft, plus what could not be mapped."""

    proposals: List[DefectMapProposal] = field(default_factory=list)
    #: Rows we could not confidently map. Surfaced, never silently dropped --
    #: an unmapped failure mode is a station the quality path stays blind to.
    unmapped: List[str] = field(default_factory=list)
    backend: str = "deterministic"

    #: Column order of ``to_frame``, kept stable so an empty result is still
    #: a usable frame rather than a KeyError at the call site.
    COLUMNS = ["station_id", "defect_type", "weight", "confidence",
               "method", "evidence"]

    def to_frame(self) -> pd.DataFrame:
        rows = [p.as_row() for p in self.proposals]
        return pd.DataFrame(rows, columns=self.COLUMNS)

    def profiles(self) -> Dict[str, Dict[str, float]]:
        """Normalised ``{station_id: {defect_type: weight}}``, summing to 1."""
        out: Dict[str, Dict[str, float]] = {}
        for p in self.proposals:
            out.setdefault(p.station_id, {})
            out[p.station_id][p.defect_type] = (
                out[p.station_id].get(p.defect_type, 0.0) + p.weight
            )
        for stn, prof in out.items():
            total = sum(prof.values())
            if total > 0:
                out[stn] = {k: v / total for k, v in prof.items()}
        return out

    def to_yaml(self) -> str:
        """A draft an engineer edits and signs off. Not a live input."""
        lines = [
            "# PROPOSED station-to-defect-type map -- A DRAFT, NOT A CONFIGURATION.",
            "#",
            f"# Generated from a control plan by RippleTwin ({self.backend} backend).",
            "# Every line below is a proposal. A process engineer must confirm it",
            "# before it is used: a wrong link does not break the flow path, but it",
            "# will send a quality investigation to the wrong station.",
            "#",
            f"# {len(self.proposals)} proposals across "
            f"{len(self.profiles())} stations.",
        ]
        if self.unmapped:
            lines.append(
                f"# {len(self.unmapped)} failure mode(s) could NOT be mapped -- "
                "listed at the end."
            )
        lines.append("")
        lines.append("defect_profiles:")
        for stn, prof in sorted(self.profiles().items()):
            lines.append(f"  {stn}:")
            for dtype, w in sorted(prof.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {dtype}: {w:.4f}")
        if self.unmapped:
            lines.append("")
            lines.append("# UNMAPPED -- review these by hand:")
            for u in self.unmapped:
                lines.append(f"#   - {u}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- tokenising


def _tokens(text: str) -> List[str]:
    """Lowercase word tokens, with a crude plural/suffix strip."""
    words = re.findall(r"[a-z]+", str(text).lower())
    out = []
    for w in words:
        if len(w) > 4 and w.endswith("ing"):
            w = w[:-3]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Match a column by normalised name, tolerating a plant's own spelling."""
    norm = {re.sub(r"[^a-z0-9]", "_", str(c).lower()).strip("_"): c
            for c in df.columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    # Fall back to substring containment, longest candidate first.
    for cand in sorted(candidates, key=len, reverse=True):
        for k, orig in norm.items():
            if cand in k:
                return orig
    return None


def _token_specificity(synonyms: Dict[str, List[str]]) -> Dict[str, float]:
    """How much a token narrows the field, as inverse document frequency.

    A token appearing in one defect code's vocabulary is strong evidence; one
    appearing in several is weak. Computed over the synonym table itself, so a
    plant substituting its own vocabulary gets the weighting recomputed for
    free rather than inheriting ours.
    """
    import math
    n_types = max(len(synonyms), 1)
    seen: Dict[str, set] = {}
    for dtype, phrases in synonyms.items():
        for ph in phrases + [dtype.replace("_", " ")]:
            for t in _tokens(ph):
                seen.setdefault(t, set()).add(dtype)
    return {t: math.log(1.0 + n_types / len(d)) for t, d in seen.items()}


def _score(failure_text: str, defect_type: str,
           synonyms: Dict[str, List[str]],
           specificity: Optional[Dict[str, float]] = None) -> float:
    """How strongly a failure-mode description implies a defect code.

    Two things temper a raw token overlap, both learned from real control-plan
    text:

    * **Specificity.** Matching a token that only this defect code uses counts
      for more than one shared across codes.
    * **Evidence mass.** Matching a single common noun out of a nine-word
      failure description is weak, however well it covers the synonym phrase.
      Without this, "Panel locating pin worn" scored 1.00 for
      ``electrical_fault`` on the bare synonym "pin" -- a mechanical locating
      pin read as a connector pin.

    Lexical ambiguity of that kind is not fully solvable by token matching, and
    it is the reason this module emits a draft for an engineer to sign off and
    offers an LLM backend for prose control plans.
    """
    toks = set(_tokens(failure_text))
    if not toks:
        return 0.0
    spec = specificity or _token_specificity(synonyms)
    best = 0.0
    for phrase in synonyms.get(defect_type, []) + [defect_type.replace("_", " ")]:
        ptoks = set(_tokens(phrase))
        if not ptoks:
            continue
        matched = toks & ptoks
        if not matched:
            continue
        # Weighted coverage of the synonym phrase.
        num = sum(spec.get(t, 1.0) for t in matched)
        den = sum(spec.get(t, 1.0) for t in ptoks) or 1.0
        coverage = num / den
        # Evidence mass: one matched word is worth less than three.
        mass = 0.55 + 0.45 * min(1.0, len(matched) / 3.0)
        best = max(best, coverage * mass)
    return best


# ------------------------------------------------------------ deterministic


def _propose_deterministic(
    plan: pd.DataFrame,
    station_ids: Sequence[str],
    defect_types: Sequence[str],
    synonyms: Dict[str, List[str]],
    min_confidence: float,
) -> MapProposalSet:
    out = MapProposalSet(backend="deterministic")
    scol = _find_column(plan, STATION_COLUMNS)
    fcol = _find_column(plan, FAILURE_COLUMNS)
    if scol is None or fcol is None:
        raise ValueError(
            "could not find an operation column and a failure-mode column. "
            f"columns present: {list(plan.columns)}"
        )
    sevcol = _find_column(plan, SEVERITY_COLUMNS)
    known = {str(s).strip().lower(): str(s) for s in station_ids}
    spec = _token_specificity(synonyms)

    # Iterate the columns directly rather than via itertuples: a control plan
    # has headers like "Op No." and "Potential Failure Mode", and itertuples
    # renames anything that is not a valid Python identifier to a positional
    # _0/_1. getattr then silently returned "" for every row and the mapper
    # produced nothing at all, with no error.
    stations_col = plan[scol].astype(str)
    failures_col = plan[fcol].astype(str)
    sev_col = plan[sevcol] if sevcol else None

    for i in range(len(plan)):
        raw_station = stations_col.iloc[i].strip()
        failure = failures_col.iloc[i].strip()
        if not raw_station or not failure or raw_station.lower() == "nan":
            continue
        stn = known.get(raw_station.lower())
        if stn is None:
            # A control plan may name an operation the export never mentions.
            # Do not guess -- an invented station is worse than a gap.
            out.unmapped.append(f"{raw_station}: {failure} (station not on the line)")
            continue

        sev = 1.0
        if sev_col is not None:
            try:
                sev = max(float(sev_col.iloc[i]), 1.0)
            except (TypeError, ValueError):
                sev = 1.0

        scores = {d: _score(failure, d, synonyms, spec) for d in defect_types}
        best = max(scores.values()) if scores else 0.0
        if best < min_confidence:
            out.unmapped.append(f"{raw_station}: {failure}")
            continue
        for dtype, sc in scores.items():
            if sc >= min_confidence:
                out.proposals.append(DefectMapProposal(
                    station_id=stn, defect_type=dtype,
                    weight=sc * sev, evidence=f"{raw_station} | {failure}",
                    confidence=sc, method="deterministic",
                ))
    return out


# --------------------------------------------------------------------- LLM


def _secret(name: str) -> Optional[str]:
    """Read a credential from the environment. Never hard-coded, never logged."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def llm_available() -> bool:
    return bool(_secret("GROQ_API_KEY") or _secret("ANTHROPIC_API_KEY"))


def _llm_json(prompt: str) -> Optional[list]:
    """Ask a model for JSON. Returns None on any failure -- never raises.

    A failure here must degrade to the deterministic path, not stop a pilot.
    """
    key = _secret("GROQ_API_KEY")
    if key:
        import urllib.request
        body = json.dumps({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            return json.loads(txt).get("mappings", [])
        except Exception:
            return None

    if _secret("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            c = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
            m = c.messages.create(
                model="claude-sonnet-5", max_tokens=4096, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = m.content[0].text
            start, end = txt.find("{"), txt.rfind("}")
            return json.loads(txt[start:end + 1]).get("mappings", [])
        except Exception:
            return None
    return None


def _propose_llm(
    plan_text: str,
    station_ids: Sequence[str],
    defect_types: Sequence[str],
) -> Optional[MapProposalSet]:
    prompt = (
        "You are reading an automotive process FMEA / control plan.\n\n"
        "Map each failure mode to the station that can physically cause it and "
        "to one or more defect codes.\n\n"
        f"STATIONS (use these EXACT ids, no others):\n{list(station_ids)}\n\n"
        f"DEFECT CODES (use these EXACT codes, no others):\n{list(defect_types)}\n\n"
        "Return JSON: {\"mappings\": [{\"station_id\":..., \"defect_type\":..., "
        "\"weight\": 0.0-1.0, \"evidence\": \"the source text\"}]}\n"
        "If a failure mode does not clearly belong to a listed station and code, "
        "omit it. Do not invent identifiers.\n\n"
        f"CONTROL PLAN:\n{plan_text[:12000]}"
    )
    raw = _llm_json(prompt)
    if raw is None:
        return None

    # The guardrail. Anything outside the supplied vocabulary is discarded, so
    # the model cannot name an asset that does not exist.
    valid_s, valid_d = set(map(str, station_ids)), set(map(str, defect_types))
    out = MapProposalSet(backend="llm")
    for m in raw:
        try:
            s, d = str(m["station_id"]), str(m["defect_type"])
            w = float(m.get("weight", 0.5))
        except (KeyError, TypeError, ValueError):
            continue
        if s not in valid_s or d not in valid_d:
            out.unmapped.append(
                f"rejected: station={s!r} defect={d!r} not in the supplied lists"
            )
            continue
        out.proposals.append(DefectMapProposal(
            station_id=s, defect_type=d, weight=max(0.0, min(w, 1.0)),
            evidence=str(m.get("evidence", ""))[:200],
            confidence=max(0.0, min(w, 1.0)), method="llm",
        ))
    return out


# ------------------------------------------------------------------- public


def propose_defect_map(
    control_plan,
    station_ids: Sequence[str],
    defect_types: Sequence[str],
    synonyms: Optional[Dict[str, List[str]]] = None,
    min_confidence: float = 0.40,
    use_llm: bool = False,
) -> MapProposalSet:
    """Propose a station-to-defect-type map from a plant's control plan.

    ``control_plan`` is a DataFrame (a control-plan export) or a string of free
    text. ``use_llm`` opts into the model path; without a key, or if the call
    fails, it falls back to the deterministic matcher rather than failing a
    pilot.

    The result is a **draft**. ``MapProposalSet.to_yaml()`` renders it for an
    engineer to review, and nothing consumes it until they do.
    """
    synonyms = synonyms or DEFECT_SYNONYMS

    if use_llm and llm_available():
        text = (control_plan.to_csv(index=False)
                if isinstance(control_plan, pd.DataFrame) else str(control_plan))
        got = _propose_llm(text, station_ids, defect_types)
        if got is not None and got.proposals:
            return got
        # fall through to deterministic

    if not isinstance(control_plan, pd.DataFrame):
        raise ValueError(
            "free-text control plans need the LLM backend: set GROQ_API_KEY or "
            "ANTHROPIC_API_KEY and pass use_llm=True, or supply a table with an "
            "operation column and a failure-mode column."
        )
    return _propose_deterministic(
        control_plan, station_ids, defect_types, synonyms, min_confidence
    )


# ------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    """``python -m rippletwin.ai.fmea_map --plan control_plan.csv ...``"""
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="python -m rippletwin.ai.fmea_map",
        description=(
            "Propose a station-to-defect-type map from a plant's control plan. "
            "The output is a DRAFT for a process engineer to sign off."
        ),
    )
    ap.add_argument("--plan", required=True,
                    help="control plan / process FMEA (.csv or .xlsx)")
    ap.add_argument("--stations", required=True,
                    help="comma-separated station ids, or a file with one per line")
    ap.add_argument("--defects", required=True,
                    help="comma-separated defect codes, or a file with one per line")
    ap.add_argument("--out", help="write the draft YAML here")
    ap.add_argument("--min-confidence", type=float, default=0.40)
    ap.add_argument("--use-llm", action="store_true",
                    help="route prose control plans through a model "
                         "(needs GROQ_API_KEY or ANTHROPIC_API_KEY)")
    a = ap.parse_args(argv)

    def _list(v):
        if os.path.exists(v):
            with open(v) as fh:
                return [ln.strip() for ln in fh if ln.strip()]
        return [x.strip() for x in v.split(",") if x.strip()]

    plan = (pd.read_excel(a.plan) if a.plan.endswith((".xlsx", ".xls"))
            else pd.read_csv(a.plan))
    res = propose_defect_map(
        plan, _list(a.stations), _list(a.defects),
        min_confidence=a.min_confidence, use_llm=a.use_llm,
    )
    print(f"backend        : {res.backend}")
    print(f"proposals      : {len(res.proposals)} "
          f"across {len(res.profiles())} stations")
    print(f"unmapped       : {len(res.unmapped)}")
    if res.unmapped:
        print("\nNOT MAPPED -- review by hand:")
        for u in res.unmapped[:20]:
            print(f"  - {u}")
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(res.to_yaml())
        print(f"\ndraft written to {a.out}")
        print("Review it with a process engineer before using it.")
    else:
        print()
        print(res.to_yaml())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
