# Judge Q&A

Answers are written to be checkable. Where a number appears, the file that
produced it is named. Where we do not know something, we say so.

---

## Business

**1. Why would a manufacturer pay for this?**
Because their twin covers the stations they instrumented, and the stations they
did not instrument are exactly where a developing problem hides longest. The
brief's own framing: lines "mix legacy and modern equipment, so sensor coverage
is often inconsistent". RippleTwin makes the uninstrumented part of the line
addressable without a retrofit programme.

**2. Where exactly does ROI come from?**
Three drivers, in descending confidence: (a) naming *which* station to send a
technician to instead of "somewhere in body shop"; (b) catching a defect source
before more in-flight units accumulate value on top of it; (c) recovered
throughput. Full arithmetic in `docs/BUSINESS_CASE.md`, with every input
editable in the dashboard's Leadership view.

**3. Your ROI number looks too good. What breaks it?**
It does, and we say so in the business case. The model is dominated by
contribution margin per recovered vehicle. **On a line that is not
capacity-constrained, a recovered vehicle is worth almost nothing** and the
throughput driver collapses. We would qualify that during Phase 0 rather than
sell throughput value to an unconstrained plant.

**4. Why not just install more sensors?**
Often you should, and we do not argue otherwise. But the brief notes retrofits
are limited to "a small number of scheduled maintenance windows per year" — the
constraint is calendar as much as cost. The stronger answer: **our coverage
experiment is a sensor-placement tool.** It shows which blind stations inference
already covers and which genuinely need instrumenting, so the retrofit budget
goes where it buys something. We are not competing with that budget; we are
telling you how to spend it.

**5. Who buys it?**
Plant Manager / Manufacturing Engineering, sponsored by Head of Manufacturing
Digital. The decision is not "should we have a twin" — they have one — it is
"do we retrofit, or extract more from what we already record".

**6. What is the payback period?**
Under the illustrative assumptions, around one month. That figure is
**ILLUSTRATIVE** and rests on the margin assumption in Q3. Treat the mechanism
as demonstrated and the money as unvalidated.

---

## Technical

**7. How exactly does shadow-sensing work?**
Material is conserved through a serial line. If station *k* slows, every station
downstream starves and every station upstream blocks; the boundary sits at *k*.
We score every station — instrumented or not — as a hypothesis about where that
boundary is, and return a posterior. No sensor at *k* is needed; sensors on both
sides of it are. Full derivation in `docs/METHOD.md`.

**8. How do you know the inferred hidden state is correct?**
We check it. The simulator knows the true processing time of blind stations; the
model never sees it. **Median error of the inferred cycle time at a station with
no sensor: ~3% at 75% coverage** (`results/tables/cycle_time_inference.csv`).
That is the sharpest falsifiable claim in the project — a number that is
unmeasurable in the model's input, estimated and then graded.

**9. Why is this different from anomaly detection?**
Anomaly detection tells you a station looks unusual. It cannot tell you whether
that station is a *cause* or a *victim* — and under a bottleneck, the victims
look far more anomalous than the cause. Our B1 baseline is exactly that, given
the same features, and it locates the source in 0% of hidden-source cases. The
flow model knows direction of causation a priori because material moves one way.

**10. Why is this a digital twin and not an ML dashboard?**
Because the structure is the model, not a feature. The line topology, buffer
capacities and failure-mode propensities are declared, not learned; the
propagation matrix is derived from buffer capacity; the forward forecast is the
flow arithmetic, not a regression. A dashboard shows you the stations it can
read. This one *simulates the line* well enough to reconstruct the stations it
cannot, and to project what happens next.

**11. How do you handle changing station relationships?**
Topology lives in a YAML file, not in code. Re-sequencing means editing it and
refitting the nominal baseline. Honest limitation: the propagation pattern
assumes a **serial** line — parallel sub-lines and rework loops require the
pattern matrix rebuilt from the real process graph. The likelihood generalises;
the specific +1/−1 pattern does not.

**12. What happens when a sensor fails?**
That station becomes hidden, which is the case the system is built for. It
degrades along the coverage curve rather than failing. The coverage sweep is
literally a simulation of progressive sensor loss.

**13. How do you handle new vehicle variants?**
Expectations are formed per window from that window's actual model mix, so a
mix change is not a deviation. A genuinely new variant needs a nominal
re-fit — until then its work content is unknown and the twin would flag it.
Scenario S5 tests exactly this: a mix change plus a supply delay, and the
correct answer is no station alert.

**14. How do you distinguish correlation from causation?**
This is the core of it. We do not learn "S07 blocks when S09 starves". We assert
a priori, from conservation of material, that a constraint blocks upstream and
starves downstream, and we test which station's position best explains the
observed asymmetry. Two competing hypotheses guard it: `NULL`, and
`LINE_SUPPLY` for a uniform starvation with no boundary — so an inbound material
delay is not blamed on a station.

**15. Isn't your simulator built to make your method work?**
Fair challenge. The physics is three standard serial-line recursions, written
before the estimator and not tuned to it. More to the point, the simulator
**broke our first two designs**: a symmetric pressure channel mislocalised by
one station, and a MAD z-score on starvation produced a 58% false-alarm rate.
Both are documented in `docs/METHOD.md`. A simulator built to flatter the method
would not have done that.

---

## Trust

**16. What happens when the model is wrong?**
It abstains. When posterior mass is spread across adjacent candidates it reports
the group and escalates instead of naming one
(`recommend_flow` → `ACTION_ESCALATE`). Every alert carries caveats naming what
could not be separated.

**17. What happens when it misses a defect?**
It will. The quality path returns a *shortlist*, not a verdict, and it pools
several hundred vehicles for power, so it is slower than the flow path. Existing
inspection gates are unaffected — RippleTwin adds a prior on where to look, it
does not replace a gate.

**18. Why would a supervisor trust it?**
Three reasons, in order of what actually matters on a floor: it is right about
things they can verify; it says when it does not know; and it is measured in
shadow mode for 8–12 weeks before anyone is asked to act. The ledger records
what was recommended and what was found, so trust is a measured quantity rather
than a claim.

**19. How is the recommendation explained?**
From the model's own evidence, with no language model anywhere in that path.
Each item is tagged OBSERVED / INFERRED / PREDICTED. We refused an LLM narrator
here specifically because an explanation that can drift from the evidence is
worse than none — it invites trust for a reason that was never true.

**20. What stops alert fatigue?**
The threshold is set from a **stated per-window false-alarm target** on held-out
nominal data, not hand-picked. Plus a persistence requirement and an impact
floor: a deviation that stays inside takt is reported as "watch", not an alert.

---

## Implementation

**21. How does it integrate with legacy equipment?**
Read-only. It consumes PLC station timestamps, MES gate results and the build
sequence — data these plants already record. It writes to no control system.

**22. Does it require new hardware?**
No new hardware to start. It requires that *some* stations are instrumented and
that blind stations have instrumented neighbours — that is the mechanism, and we
do not claim zero sensors.

**23. How long does deployment take?**
Phase 0 data readiness 2–4 weeks; shadow mode 8–12 weeks before anyone acts.
Deliberately slow at the start, because the brief warns that false alarms "erode
floor-level trust quickly" and that trust is not recoverable twice.

**24. How does it scale?**
New line = new topology YAML + baseline refit. The estimator, calibration
procedure and UI are unchanged. What does not transfer automatically is the
nominal baseline (equipment-specific) and the serial-line assumption.

---

## Prototype

**25. What exactly have you demonstrated?**
On synthetic data with known ground truth: that a station's state can be
reconstructed without a sensor at it, well enough to name the station and
estimate its cycle time to ~3%; that this beats three calibrated baselines
precisely where instrumentation is missing; and that the system stays quiet on
clean lines and abstains when ambiguous.

**26. What is simulated, and what is real code?**
The factory is simulated — the line, the disturbances, the defects. Everything
downstream is real, running code: feature construction, the estimator,
calibration, baselines, evaluation, explanation, recommendation, ledger,
dashboard. 45 tests pass. No result in this repo is hard-coded.

**27. What are the baselines, and are they fair?**
B0 SPC per station, B1 Isolation Forest on the same features, B2 our own flow
twin restricted to instrumented stations. **All four calibrated to the same
false-alarm rate on the same held-out nominal data, and put through an identical
detection rule.** B2 is the important one: same physics, same likelihood, same
tuning — the only difference is whether blind stations may be named. That
isolates shadow-sensing itself.

**28. What happens as sensor coverage decreases?**
Localisation of hidden sources degrades gracefully — see
`results/figures/coverage_curve.png`. The revealing point is the other end:
**at 100% coverage RippleTwin and B2 are identical to three decimal places**,
because with nothing hidden there is nothing to infer. The advantage appears
exactly and only where instrumentation is missing, which is what should happen
if the mechanism is real rather than an artefact of tuning.

**29. Your detection rates are not that high. Why?**
Because we report them per-window and per-episode, at a 1%-false-alarm operating
point, averaged across disturbance magnitudes down to 1.14×. Weak faults are
genuinely hard and we do not hide them behind an average — see
`results/tables/detection_by_magnitude.csv`.

---

## Competition

**30. What is genuinely innovative here?**
Inverting the framing. The industry answer to "we cannot see that station" is
"add a sensor". Ours is: the stations you already measure are physically coupled
to the one you cannot, and that coupling is a measurement instrument. Using
blocking/starving asymmetry as a *localisation* signal for an unmeasured station
is not standard practice in commercial twins.

**31. Why can't an existing industrial platform add this tomorrow?**
They could. **The algorithm is not the moat** and we do not pretend otherwise —
it is a few hundred lines of physics, documented publicly in this repo. What is
defensible: the outcome ledger accrues plant-specific precision data from day
one, and being the system that tells a plant *where instrumentation is worth
buying* is a position inside the capital planning cycle, not the operations
budget.

**32. Why should Accenture select this team?**
Because the work is falsifiable and we did the falsifying. Two designs were
killed by our own measurements and are documented rather than buried; the
comparison is calibrated to a matched false-alarm rate rather than staged; and
the metric that would have flattered us most — lead time — we report as
under-powered, and replaced with the finding that **most of these faults never
reach the production board at all**.

**33. What would you do next with more time?**
Validate the flow signature against one real line's PLC logs — the single thing
that would move this from plausible to proven. Then generalise the propagation
model beyond serial topology, and run a shadow-mode pilot long enough to measure
per-station precision on real outcomes.
