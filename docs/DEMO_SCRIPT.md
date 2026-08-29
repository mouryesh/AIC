# 3-minute demo script

This is the narration for the guided walkthrough. It matches the **🎬 Guided
demo** button in the dashboard sidebar (`streamlit run app/dashboard.py`),
which locks the scenario to **S1 — hidden bottleneck** and steps through
DETECT → EXPLAIN → FORECAST → ACT → PROVE with Back/Next controls, so a judge
never has to configure anything before seeing the product. Run it yourself
before presenting: click the button once, then Next four times.

Every number named below is read live off the running app, not scripted in
advance — the twin is deterministic for this scenario/seed, so the same
numbers should appear on your screen.

---

## 0:00–0:20 — Problem

*(Live Line, before clicking "Start guided demo")*

"Traditional digital twins can only see what their sensors measure. This
line has 42 stations. Ten of them — drawn as diamonds — send no telemetry at
all: no cycle time, no state, nothing. A conventional twin is blind at
exactly the stations most likely to hide a developing problem."

Point at the KPI strip: **LINE STATUS: ATTENTION**, **HIDDEN CONSTRAINT: S02**,
**SENSOR STATUS: NO SENSOR**.

## 0:20–0:50 — Detection

*(Click "▶ Start / restart guided demo" — Step 1/5 DETECT)*

"RippleTwin names S02 as the constraint, with no sensor on it. Look at the
line map: S02 is drawn as a diamond, outlined in red — the twin's strongest
visual emphasis, because that's the station it thinks is the problem. Above
it: upstream blocked, downstream starved, and S02 sits exactly on the
boundary between the two."

Read the callout: *"UPSTREAM (S01) BLOCKED → ◆ S02 NO SENSOR → DOWNSTREAM
(S03) STARVED."*

## 0:50–1:20 — Explanation

*(Next → Step 2/5 EXPLAIN. Scroll to "Why RippleTwin thinks S02 is the constraint")*

"This isn't a black box. Upstream evidence: S01 is blocked well above its
normal level — it can't hand work forward. Downstream evidence: S05, S07,
S09 are all starved above normal — work isn't arriving. The confidence
shown is 87%, stated as a probability, never as certainty — and the card
tells you the alternative hypothesis too: 13% posterior on S01 instead."

Point out the caveat: *"This station's state is inferred, not measured.
Confirm on the floor before acting on it."* — that line is not decoration;
every inferred value carries it.

## 1:20–1:50 — Forecast

*(Next → Step 3/5 FORECAST. Scroll to "What happens if we do nothing?")*

"This is where detection becomes a business consequence. If nothing
changes: upstream backs up to S01 in about 5 minutes, downstream starvation
reaches S03 in about 5 minutes, and over the next 60 minutes the line loses
roughly 10-12 vehicles — running at 48-50 vehicles/hour against a 60/hour
target. That number is PREDICTED, not measured — a forward projection from
the estimated cycle time, checkable by a plant engineer on paper."

## 1:50–2:10 — Action

*(Next → Step 4/5 ACT. Scroll to "Recommended action")*

"The recommendation: inspect S02, priority HIGH, owned by a line-side
maintenance technician, respond within 15 minutes, with an explicit
verification prompt for what they should report back. RippleTwin recommends
— a human decides. Click Approve."

Click **✅ Approve**. Point at the ledger counter incrementing and the
confirmation message naming the entry number. "That write is hash-chained —
System → Audit Log shows it, and any attempt to edit it after the fact would
break every hash that follows."

## 2:10–2:40 — Proof

*(Next → Step 5/5 PROVE, lands on System → Evidence)*

"Does this actually work? At 75% sensor coverage, RippleTwin localises the
exact faulty station 69% of the time when every baseline — including the
published Turning Point Method — scores exactly 0%, because naming an
un-instrumented station is outside what they can express at all. At 100%
coverage, with nothing hidden, RippleTwin and an observed-only twin are
identical — 92.8% vs 92.8%. The advantage appears exactly and only where
instrumentation is missing."

Scroll to **"When RippleTwin does NOT know"**: "And when two blind stations
sit next to each other, it says so and abstains rather than guessing — a
trustworthy industrial AI system should not always produce an answer."

## 2:40–3:00 — Business case

*(Click Business in the sidebar, or Exit guided demo and navigate manually)*

"Estimated annual value on this line: roughly $2M, built from transparent,
editable assumptions — every number here is labelled illustrative, not a
measured result. And the differentiator most people miss: it's not about
instrumenting every station. Plant → 'Where should we install the next
sensor?' ranks the blind stations by information value, so a retrofit budget
buys the most localisation improvement per dollar instead of guessing."

---

## If you have another 60 seconds

- **Incidents** — the same alert, restructured as What happened / Where /
  Why / What happens next / What should we do / What did the human decide /
  Audit trail. Try scenario **S7** (two simultaneous, unrelated faults) to
  show more than one incident at once.
- **Ask RippleTwin** (on Live Line, below the recommendation) — click one of
  the suggested questions. The answer is guardrailed against the same
  evidence pack already on screen; it cannot state a number the twin didn't
  compute.
- **S3 — normal variation** — switch scenarios in the sidebar to show the
  twin staying quiet on a shift change and mix change, which is the harder
  and more important half of "it works."
