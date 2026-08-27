# Demo video — storyboard and narration

**Target length:** 2:50 (the Round 1 brief capped video at 3 minutes; Round 2's
"What Round 2 Asks You to Deliver" names a prototype demo video in the GitHub
repository without restating a limit. We treat 3:00 as the ceiling.)

**Rule for the whole video:** the screen shows the running prototype, not the
deck. Every number spoken is visible on screen at that moment.

**Recording setup**

```bash
streamlit run app/dashboard.py       # screen A — the product
python demo/run_demo.py --scenario S1  # screen B — the evidence, in a terminal
```

---

## 0:00–0:15 — The blind spot

**Screen:** Dashboard, Floor supervisor view, line map. 42 stations. Ten drawn
as hollow diamonds.

**Narration:**
> This is a 42-station vehicle assembly line. Thirty-two stations have sensors.
> Ten do not — they run on manual checklists, and every digital twin built on
> this line is blind to them. That is not an edge case. It is what a real plant
> looks like.

**On screen:** cursor hovers a diamond → tooltip reads *"NO SENSOR — state inferred"*.

---

## 0:15–0:35 — What breaks

**Screen:** Cut to the S1 scenario selector. Line running normally, no alerts.

**Narration:**
> A station with no sensor starts to slow down. Nothing measures it. By the time
> the production board shows the shortfall, the line has been losing vehicles
> for hours — and nobody can say which station to send a technician to.

**On screen:** "Active alerts: 0" while, in the ground-truth strip, the
disturbance has already begun.

---

## 0:35–1:00 — The mechanism, in one picture

**Screen:** The blocking / starvation profile chart.

**Narration:**
> Here is what makes this solvable. When a station slows, material stops
> arriving downstream and stops leaving upstream. Every station downstream
> starves. Every station upstream blocks. The boundary between the two sits
> exactly at the station causing it.
>
> We never need a sensor *at* that station. We need sensors on both sides of it.

**On screen:** red bars above the axis, blue below, the sign flip landing inside
a grey "no sensor" band. This is the single most important frame in the video —
hold it.

> This is conservation of material through a serial line. It is not a learned
> correlation, and that is the difference: a correlation model sees S07 blocking
> and S09 starving and cannot tell you which caused which. The physics can.

---

## 1:00–1:25 — Localisation

**Screen:** Posterior bar chart, then the KPI row.

**Narration:**
> RippleTwin scores every station as a hypothesis, including the ones it cannot
> measure, and returns a probability distribution — not a verdict.
>
> It names S02. S02 has no sensor.
>
> And it estimates S02's cycle time at 76 seconds against a 60-second takt. The
> simulator's ground truth is 77.3. That is a 1.2% error on a number that, in
> the data the model was given, is unmeasurable.

**On screen:** KPI cards — Station S02 / INFERRED — no sensor / Cycle 76s.

---

## 1:25–1:45 — Consequence

**Screen:** "What happens next" panel.

**Narration:**
> Then it propagates that forward through the same flow physics: 21% below
> target, about 13 vehicles lost in the next hour, downstream starvation
> reaching S03 within five minutes. No regression model — arithmetic a plant
> engineer can check on paper.

---

## 1:45–2:10 — Explanation and honesty

**Screen:** Evidence list with the OBSERVED / INFERRED / PREDICTED tags, then the
caveats.

**Narration:**
> Every line of that explanation is tagged with how it was obtained — measured,
> inferred, or predicted. No language model is involved anywhere in this path,
> so the explanation cannot drift away from the evidence.
>
> And it states its own limits: S01 and S02 cannot be fully separated from the
> available sensors, so it says so, and tells the supervisor to check both.

---

## 2:10–2:30 — Human decision

**Screen:** Recommendation card → click **Approve** → ledger entry appears.

**Narration:**
> It recommends; a person decides. RippleTwin never writes to a PLC and never
> stops the line. The supervisor approves, and the decision goes into a
> hash-chained ledger — so what the system recommended, and what was actually
> found, both stay provable afterwards. That ledger is also the feedback signal:
> per-station precision comes from real outcomes.

---

## 2:30–2:50 — The proof, and the limits

**Screen:** `results/figures/coverage_curve.png`.

**Narration:**
> We tested this against three baselines on forty held-out episodes, with every
> method calibrated to the same false-alarm rate — otherwise the comparison
> measures nothing.
>
> When the faulty station has no sensor, the conventional observed-only twin
> identifies it zero percent of the time. It structurally cannot. RippleTwin
> does — and at full sensor coverage the two are identical, which is exactly
> what should happen when there is nothing left to infer.
>
> This is synthetic data. Nothing here is a claim about a real plant yet. What
> it is, is a mechanism that is falsifiable — and it survived.

**Final card:**
```
RippleTwin — see the bottleneck before it arrives,
even at the stations you can't instrument.
github.com/<user>/RippleTwin
Simulated prototype results on synthetic data.
```

---

## Shot list

| # | Source | Length |
|---|---|---|
| 1 | Dashboard line map, hover a diamond | 15s |
| 2 | Normal state, no alerts | 20s |
| 3 | **Pressure profile chart** (hero shot) | 25s |
| 4 | Posterior chart + KPI cards | 25s |
| 5 | Ripple forecast panel | 20s |
| 6 | Evidence list + caveats | 25s |
| 7 | Approve → ledger | 20s |
| 8 | Coverage curve figure | 20s |

## Things to avoid

- Do not read the deck aloud over a static slide.
- Do not hide the abstention behaviour — showing the system decline to name a
  station is more persuasive than another confident answer.
- Do not crop the "simulated data" labelling out of the figures.
- Do not speed up the terminal output to look faster than it is.
