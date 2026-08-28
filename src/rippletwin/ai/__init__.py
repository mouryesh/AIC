"""Where a language model is allowed to help, and where it is not.

The rule this package follows
-----------------------------
RippleTwin's localisation is physics, not learning, and that is deliberate: the
baselines score a structural 0% on hidden stations because they can only rank
what they measure, and swapping conservation of material for a learned
correlation would trade away the one advantage that survives a sensor gap.

A 2026 PRISMA review of foundation-model agents in industrial automation
(arXiv:2605.02592, 88 papers from 2,341 screened) found 75.0% of systems at
TRL 4-6 and only 9.1% at TRL 7-9, with the most-reported limitations being
"lack of generalization, hallucination and output instability, data scarcity,
and inference latency". That is a good reason to keep a language model **at the
edges of a decision rather than inside it**.

So the test for putting an LLM somewhere is: *if it is wrong, what breaks?*

* ``fmea_map`` -- reads a control plan and proposes a station-to-defect-type
  map. Wrong answer costs an engineer review time, offline, before anything
  runs. The output is a draft a human signs off, never a live input.
* ``copilot`` (in ``rippletwin.copilot``) -- phrases an answer over numbers the
  twin already computed, with a guardrail rejecting any figure not present in
  the evidence pack.

Both are text-to-structure tasks over a controlled vocabulary. Neither can
reach the localisation, the threshold, or a work order's numbers.
"""

from .fmea_map import (
    DEFECT_SYNONYMS,
    DefectMapProposal,
    MapProposalSet,
    propose_defect_map,
)

__all__ = [
    "propose_defect_map",
    "DefectMapProposal",
    "MapProposalSet",
    "DEFECT_SYNONYMS",
]
