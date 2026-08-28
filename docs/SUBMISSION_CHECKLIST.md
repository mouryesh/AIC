# Round 2 submission checklist

## What the brief asks for

The Round 2 brief, under **"What Round 2 Asks You to Deliver"**, names exactly
three deliverables:

| # | Deliverable (verbatim from the brief) | Status |
|---|---|---|
| 1 | **Detailed Business Proposal** — problem framing, solution design, target users, business case and impact, a phased roadmap, and key risks with mitigations | ✅ `assets/RippleTwin_Round2_Proposal.pptx` (17 slides) + `docs/BUSINESS_CASE.md` |
| 2 | **Working Prototype** — a functional demonstration of the core mechanism; "a working proof-of-concept on illustrative or sample data is expected and encouraged" | ✅ `demo/run_demo.py`, `app/dashboard.py`, full source in `src/` |
| 3 | **Public GitHub repository**, including a prototype demo video and a README | ⚠️ repo complete; **push and video recording are manual** — see below |

Every element the brief names for the proposal is covered:

| Brief requirement | Where |
|---|---|
| Problem framing | Deck slides 3–4; README "The problem" |
| Solution design | Deck slides 5–7; `docs/METHOD.md` |
| Target users | Deck slide 11; dashboard's three views |
| Business case and impact | Deck slides 12–13; `docs/BUSINESS_CASE.md` |
| Phased roadmap | Deck slide 14; BUSINESS_CASE §6 |
| Key risks with mitigations | Deck slide 15; README "Limitations" |

## Track-4 reference parameters

| Brief parameter | What we built |
|---|---|
| "roughly 30–50 stations across body construction, paint, and final assembly" | 42 stations: 14 body, 10 paint, 18 final |
| "a majority of stations well-instrumented, a meaningful minority reliant on manual checks" | 76% instrumented (40% rich, 36% basic); 24% emit no telemetry |
| "production can only be paused ... during a small number of scheduled maintenance windows" | Central to the sensor-economics argument; integration is read-only |
| Mixed-model line | 3 variants (sedan / SUV / EV) with different per-zone work content |

## Solutioning areas the brief lists

| Area | Covered |
|---|---|
| Modelling approach — what to represent vs infer | ✅ explicit OBSERVED / INFERRED / PREDICTED split |
| Predictive techniques + how you'd validate before trusting output | ✅ 4 baselines at matched FPR **including the published Turning Point Method**, coverage sweep, 204 tests |
| Handling data gaps at sensor-poor stations | ✅ the core mechanism |
| Distinct views for supervisor / plant manager / leadership | ✅ three dashboard views |
| Integration around legacy PLCs without disrupting production | ✅ read-only; no PLC writes anywhere |
| Scalability & ROI | ✅ phased roadmap; editable ROI model |

## Technical

- [x] Data generator works — `python -m rippletwin.data.generate`
- [x] Preprocessing works — windowing + baseline scoring
- [x] Shadow-sensing implemented — two mechanisms, not a diagram
- [x] Baselines work — B0 SPC, B1 anomaly, B2 observed-only twin, **B3 Turning
      Point Method (Li, Chang & Ni 2009)** — all at a matched false-alarm rate
- [x] Prior art cited, not claimed — `docs/REFERENCES.md`
- [x] Sensor-placement guidance implemented and directionally validated
- [x] Input data contract specified against real interfaces (OPC UA / PackML / MES)
- [x] Phase 0 readiness assessment runnable — returns NOT_VIABLE when it should
- [x] Alerts become owned, time-bounded work orders with a CMMS payload
- [x] Evaluation works — 110 held-out episodes, 4 coverage levels
- [x] Results reproducible — fixed seeds; disjoint fit / calibration / test data
- [x] Demo works — deterministic, `python demo/run_demo.py`
- [x] Dashboard works — verified rendering in a browser, not just HTTP 200
- [x] Explainability works — provenance-tagged, no LLM in the path
- [x] Human-in-the-loop works — hash-chained ledger, tamper test passes
- [x] Tests pass — 79/79

## Repository hygiene

- [x] No secrets or credentials
- [x] No fabricated metrics — every table written by code that ran
- [x] No broken imports
- [x] No empty placeholder files
- [x] Results regenerate from source
- [x] Data directory intentionally empty (data is a function of code + seed)

## Honesty audit

- [x] Every result labelled a simulated prototype result on synthetic data
- [x] ILLUSTRATIVE ASSUMPTION vs PROTOTYPE RESULT vs OFFICIAL kept distinct
- [x] Limitations written before being asked for them
- [x] Failed designs documented rather than buried (`METHOD.md` §3)
- [x] The under-powered metric reported as under-powered, not quietly dropped
- [x] No claim of zero sensors, production readiness, or validated real-world ROI
- [x] Where a baseline beats us, we say so — SPC at 100% coverage, and the
      Turning Point Method's *detection* rate at 50% coverage
- [x] The mechanism is credited to its authors, not presented as ours
- [x] A hallucinated citation was caught and excluded (see REFERENCES.md)

## Manual steps remaining

1. **Create and push the GitHub repository.** The repo is committed locally.
   Commands are in the handover notes; we have no GitHub credentials in this
   environment and did not attempt to use any.
2. **Record the demo video** (2–3 min) following `docs/DEMO_VIDEO.md`, and add
   the link to the README.
3. **Fill in team details** on deck slide 2 (names, college, stream, graduation
   year) — the template requires them and we left them for the team to enter.
4. **Confirm the submission channel and deadline** from the competition portal.
