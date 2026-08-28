# Demo video — storyboard and narration

**Target length:** ~4:00. The original Round 1/2 cut (2:50, reproduced at
the end of this document) still stands as a shorter alternative; this
version adds the predictive/robustness beats added in the Round 2 upgrade
(early warning, sensor failure, defect prediction, cross-plant
generalization, ROI/sensor placement).

**Rule for the whole video, unchanged:** the screen shows the running
prototype, not the deck. Every number spoken is visible on screen at that
moment. Nothing is scripted for effect — every scenario shown is generated
by the actual system at record time, not pre-computed and pasted in.

**This document is a script, not a recording.** Recording, editing and
publishing the video are manual steps outside what code can do — the same
convention `docs/HANDOVER.md` uses for every step that requires a person.

**Recording setup**

```bash
streamlit run app/dashboard.py                              # screen A — the product
python demo/run_demo.py --scenario S1                        # screen B — deterministic evidence
python demo/run_streaming_demo.py --scenario S6_EARLY_WARNING # screen C — the streaming replay
```

---

## 0:00–0:15 — The blind spot

**Screen:** Dashboard, Floor supervisor view, line map. 42 stations. Ten drawn
as hollow diamonds.

**Narration:**
> This is a 42-station vehicle assembly line. Real assembly lines cannot
> instrument every station — ten here run on manual checklists, and every
> conventional digital twin is blind to them.

**On screen:** cursor hovers a diamond → tooltip reads *"NO SENSOR — state inferred"*.

---

## 0:15–0:40 — Normal production, then a subtle change

**Screen:** Switch to scenario **S6 — gradual ramp (early-warning demo)**.
Line running, "Predicted bottleneck risk" card reads NORMAL.

**Narration:**
> Watch this scenario instead of a sudden failure: a station degrades
> gradually. At first, nothing about the line looks wrong — no alert would
> fire on a conventional threshold, and the line is still hitting its
> numbers.

---

## 0:40–1:10 — The mechanism, in one picture

**Screen:** The blocking/starvation profile chart.

**Narration:**
> Here is what makes any of this solvable. When a station slows, material
> stops arriving downstream and stops leaving upstream. The boundary
> between the two sits exactly at the station causing it — we never need a
> sensor *at* that station, only on both sides of it. This is conservation
> of material, not a learned correlation.

**On screen:** hold the red/blue chart, the sign flip landing inside a grey
"no sensor" band — the hero shot from the original build, unchanged.

---

## 1:10–1:45 — Prediction appears, before the line is actually constrained

**Screen:** The "Predicted bottleneck risk" card and the risk-timeline chart
with the watch/detect threshold lines, as the scenario advances.

**Narration:**
> Now watch the risk state. It moves from NORMAL to DEGRADING to WATCH —
> using a second, looser evidence threshold, calibrated the same way the
> confident-detection threshold is — well before the line is actually
> losing output. This is the early-warning layer: a graded risk ladder, not
> a single alarm.

**On screen (read the live numbers, do not narrate placeholders):**
```
Station: <name at record time>
State: PREDICTED_CONSTRAINT
Risk: <value>          Confidence: <value>
Time-to-impact: <value> min
```

---

## 1:45–2:05 — Human reviews the recommendation

**Screen:** Recommendation card, then the five decision buttons
(Approve / Reject / Modify / Escalate).

**Narration:**
> The system recommends; it never decides. A supervisor can approve it,
> reject it, redirect the action, or escalate it to a shift lead — all four
> outcomes are recorded, hash-chained, and become the training signal for
> next time.

---

## 2:05–2:30 — The bottleneck actually occurs; the prediction is checked

**Screen:** Let the scenario continue to `ACTIVE_BOTTLENECK`; reveal ground
truth.

**Narration:**
> The constraint does bind, at the station the twin already named minutes
> earlier. Reveal ground truth: the twin was right, and it was right before
> it cost the line anything. That lead time is measured, not asserted — see
> `evaluation/early_warning.py`'s lead-time table, successes and misses
> both reported.

---

## 2:30–2:55 — A sensor fails mid-shift

**Screen:** Switch scenario context to demonstrate a dynamic sensor fault
(dropout/stale) at a neighbouring station; show the confidence metric
dropping.

**Narration:**
> Now a sensor itself fails — not the process, the instrumentation. The
> system does not go quiet and it does not fake confidence: it keeps
> operating on what evidence remains, and its own confidence measurably
> drops rather than staying artificially high. A stale sensor — one stuck
> reporting an old value — is caught from its own signature: a station that
> is actually running never reports the identical reading forever.

---

## 2:55–3:15 — Defect prediction, before inspection

**Screen:** (If wired into the demo build) a defect-risk readout for an
in-flight vehicle at a RICH-tier station.

**Narration:**
> The same idea applies to quality. Instead of waiting for an inspection
> gate to catch a defect, RippleTwin scores defect risk on the vehicle
> currently at the station — using torque, vibration and temperature
> deviation, the same station-level evidence a technician would check by
> hand. It is honest about where it can't: a station with literally no
> process telemetry has no coverage here, and says so.

---

## 3:15–3:35 — Cross-plant: the same engine, a different line

**Screen:** Terminal — `python -m rippletwin.evaluation.topology_experiment`
output, or the summary table.

**Narration:**
> This line is a serial chain. Real plants have parallel stations and
> rework loops. The same inference engine — not a new one — runs against a
> line with a genuine parallel branch and one with a rework spur, with no
> per-plant code.

---

## 3:35–3:50 — Sensor placement and ROI

**Screen:** Leadership tab — sensor-placement ranking and the ROI
calculator.

**Narration:**
> And it tells you where the next sensor should go, and what it's worth —
> the same physics model, run in reverse, needing no production data. Every
> number on this page is an editable, illustrative assumption, not a claim
> about a real plant's economics.

---

## 3:50–4:00 — Close

**Final card:**
```
RippleTwin — see the bottleneck before it arrives,
even at the stations you can't instrument.
github.com/<user>/RippleTwin
Simulated prototype results on synthetic data.
```

---

## Shot list (full ~4:00 cut)

| # | Source | Length |
|---|---|---|
| 1 | Dashboard line map, hover a diamond | 15s |
| 2 | S6 scenario, normal start | 25s |
| 3 | **Pressure profile chart** (hero shot) | 30s |
| 4 | Predicted-risk card + timeline, state climbing | 35s |
| 5 | Recommendation + decision buttons | 20s |
| 6 | Ground truth reveal, prediction confirmed | 25s |
| 7 | Sensor-fault scenario, confidence dropping | 25s |
| 8 | Defect-risk readout | 20s |
| 9 | Topology experiment output | 20s |
| 10 | Sensor placement + ROI (Leadership tab) | 15s |
| 11 | Final card | 10s |

## Things to avoid

- Do not read the deck aloud over a static slide.
- Do not hide the abstention behaviour — showing the system decline to name
  a station is more persuasive than another confident answer.
- Do not crop the "simulated data" labelling out of the figures.
- Do not speed up the terminal output to look faster than it is.
- Do not manually edit a number between what the running system printed and
  what appears on the final cut.

---

## Original build's shorter cut (2:50) — still valid

The original storyboard (hidden-bottleneck localisation only, no predictive
layer) remains a valid, shorter alternative if time is constrained:

0:00 blind spot → 0:15 what breaks → 0:35 mechanism (hero shot) → 1:00
localisation → 1:25 consequence → 1:45 explanation and honesty → 2:10 human
decision → 2:30 the proof and the limits → 2:50 close. Full narration for
this cut is preserved in git history (see the commit immediately before the
Round 2 predictive upgrade) and follows the same recording rules above.
