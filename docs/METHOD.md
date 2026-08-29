# How shadow-sensing works

This document is the technical argument. It is written to be attacked.

---

## 0. Relation to prior work — read this first

**The blockage/starvation boundary is not our idea.** It is the **Turning Point
Method**, published by Li, Chang and Ni in 2009, and it is one of the
best-established data-driven bottleneck detection methods in manufacturing
science. The 2023 systematic review of the field states the rule directly:

> "The Turning Point Method proposed in (Li et al., 2009) identifies the
> bottleneck station as the station that experiences a change (turning point),
> i.e. a scenario where the blockage is higher than starvation should change to
> a scenario where starvation is higher than the blockage."

That is exactly the physics this project rests on, and we would rather cite it
than be caught claiming it. The other dominant family is the **Active Period
Method** (Roser, Nakano and Tanaka, 2001–2002), which names the process holding
the longest uninterrupted active period.

### So what is actually ours?

Both published methods **scan the stations they can measure**. That is the
crack this project lives in:

> When the turning point falls inside a run of stations that have no telemetry,
> the scan cannot stop there. It can only name the nearest station it can see.
> The true source is not merely estimated poorly — it is **outside the method's
> output space**.

Measured on our line: with a hidden fault at S02, the Turning Point Method
detects the disturbance reliably and names **S03** — the first instrumented
station on the starved side of the gap. Exact-station accuracy: **0%**.
RippleTwin resolves into the gap and names S02.

Concretely, our contributions over the published methods are:

| | Published TPM / APM | RippleTwin |
|---|---|---|
| Output space | instrumented stations only | **every station, instrumented or not** |
| Decision rule | deterministic scan for a sign change | posterior over candidate positions, with uncertainty |
| Distance | station adjacency | **buffer-capacity distance** — a 14-slot inter-zone buffer decouples far more than one slot |
| Channels | blockage vs starvation compared | modelled separately, with their own amplitudes and length scales, because they are physically asymmetric |
| Competing explanations | none | explicit `NULL` and `LINE_SUPPLY` hypotheses, so it can stay silent or blame supply |
| Unidentifiable cases | always returns a station | **abstains** when two blind stations are structurally indistinguishable |
| Hidden station state | — | **estimates the unmeasured station's cycle time** (~3% median error) |
| Sensor layout | assumed | **recommends where the next sensor buys the most** |

The Active Period Method's own documented limitation is "a very high data
requirement" — it needs to know precisely when every process is active. That is
the assumption we are relaxing.

### Two things the same review says, which we rely on

* **"None of the existing literature provides real-world validation of
  methods."** Our synthetic-only validation is therefore the field norm, not a
  peculiar weakness of this project — though it remains the first thing a pilot
  must fix.
* The review's own future-research section calls for exactly this shape of
  system: a digital twin that "can automatically analyze the throughput
  bottlenecks from the real-time data sets, predict the expected dynamics,
  examine the different scenarios ... and prescribe actions", and which
  "continuously evolve[s] using the real-time data of the production system and
  shop floor engineers' feedback."

A related modern line of work uses graph attention networks over line topology
(e.g. BSTAN) to model station interactions. We deliberately did not go there:
the propagation we need can be **derived** from buffer capacity, and a derived
model is inspectable, needs no training data, and cannot drift. See §4.

---

## 1. The claim

> A vehicle assembly line does not need a sensor at every station to build a
> digital twin that can locate a developing problem — including a problem at a
> station with no instrumentation at all.

The claim is *not* that sensors are unnecessary. It is that a partially
instrumented line already contains more information about its blind spots than
a conventional twin extracts, because the stations are physically coupled.

---

## 2. Why the coupling is exploitable

A serial line with finite buffers obeys three recursions. For vehicle `v` at
station `i` with outbound buffer capacity `B_i`:

```
start_i(v)     = max( departure_{i-1}(v), departure_i(v-1) )
end_i(v)       = start_i(v) + proc_i(v)
departure_i(v) = max( end_i(v), start_{i+1}(v - B_i) )
```

Two quantities fall out, and both are recorded by any PLC that timestamps a
station:

```
starved_i(v) = max(0, departure_{i-1}(v) - departure_i(v-1))
    the station was free and idle, waiting for work to arrive

blocked_i(v) = departure_i(v) - end_i(v)
    the station had finished, but could not release: the buffer ahead was full
```

Now suppose station `k` slows down. Material is conserved, so:

- fewer parts leave `k` → every station **downstream starves**
- fewer parts can enter `k` → the buffer behind fills → every station
  **upstream blocks**

The boundary between "blocked" and "starved" sits at `k`. **We never need a
sensor at `k`. We need sensors on both sides of it.**

This is the entire mechanism, and it is worth being precise about what kind of
claim it is. It is a *structural* argument, derived from conservation of
material through a serial line. It is not a learned correlation. A
correlation-based detector observes that S07 blocking co-occurs with S09
starving and has no basis for saying which caused which, or that the cause is
the station between them. The flow model knows the direction of causation a
priori, because material only moves one way down the line.

### Verified, not assumed

`tests/test_physics.py::test_slowdown_blocks_upstream_and_starves_downstream`
injects a slowdown at four different stations and asserts the signature directly
against simulator ground truth. If that test fails, the approach is unsound and
everything downstream of it is meaningless.

---

## 3. Three things that were wrong in the first version

All were found by measurement, and all three changed the design.

### 3.1 Blocking and starving are not symmetric

The first implementation modelled a single signed channel,
`pressure = blocked − starved`, and fitted one amplitude to it.

That is physically wrong, and it mislocalised faults by one station.

- **Starvation** downstream appears within a few vehicles: the buffer runs dry.
- **Blocking** upstream appears only once the intervening buffer *fills*, which
  takes `buffer_capacity / rate_deficit` vehicles.

Measured on this line during one disturbance: ~18 s of downstream starvation
against ~1.2 s of upstream blocking in the same window. Forcing one amplitude to
explain both drags the fitted boundary toward the side carrying more signal.

The fix is two channels, each with its own non-negative amplitude and its own
propagation length scale. This also buys graceful degradation: early in an
event, before any buffer has filled, the blocking channel contributes nothing
and localisation runs on the starvation boundary alone — which is where warning
lead time comes from.

### 3.2 A z-score was the wrong transform for starvation

Fitted on a nominal run and evaluated on *held-out* nominal data, the original
`z_starved` had mean **+3.07** and 99th percentile **27.5** — on data with
nothing wrong with it.

`starved_s` is zero-inflated with a heavy tail. Its MAD under nominal flow is
near zero, so a MAD-scaled z-score turns an ordinary 15-second starvation into a
40-sigma event. That single miscalibration produced a **58% false-alarm rate**.

The fix is to work in **fraction of takt** instead. Bounded, stable, and
directly meaningful: "S24 is losing 30% of takt to starvation" is a sentence a
plant engineer can act on.

`tests/test_shadow_sensing.py::test_deviation_channels_are_calibrated_on_heldout_nominal`
now pins this down.

---

### 3.3 Supply starvation is not uniform

`LINE_SUPPLY` was first modelled as *uniform* starvation across the line. Real
supply starvation decays from the head as buffers absorb it, so a decaying
station hypothesis always fitted it better — and an inbound material delay was
attributed to station S02 with 85% posterior mass, while S01 sat **starved at
179% of takt**.

That is the one thing that cannot happen if S02 is the constraint: if S02 were
slow, S01 would be *blocked*, holding work it could not hand over.

Two changes followed. The head-shortfall hypothesis now uses the same decay
profile, so the comparison is fair; and the recommendation layer applies the
discriminator explicitly:

| Upstream station | Interpretation |
|---|---|
| **blocked** | it holds work it cannot pass on → the station ahead is slow |
| **starved** | it has no work at all → material is not arriving |

Near the head of the line there are too few upstream stations for the likelihood
to settle this on its own, so the rule is applied directly and the system routes
to "check inbound material" rather than naming a station.

---

## 4. Estimation

For each candidate station `k`, the expected profile at observing station `i` is

```
B[k, i] = exp(-D(i,k) / lambda_block)    for i < k,  else 0
S[k, i] = exp(-D(i,k) / lambda_starve)   for i > k,  else 0
```

where `D(i,k)` is the **cumulative buffer capacity** between `i` and `k` — the
physically meaningful distance for propagation. Two stations three apart with a
14-slot inter-zone buffer between them are far more decoupled than two stations
three apart inside body shop.

One non-negative amplitude per channel is fitted by least squares, and the
hypothesis is scored by Gaussian log-likelihood on **both** channels at **all**
observed stations. Predicting zero blocking downstream is itself information: a
station that is starved where the hypothesis says it should be blocked counts
against that hypothesis.

Two further hypotheses compete:

| Hypothesis | Profile | Why it is needed |
|---|---|---|
| `NULL` | no deviation | the line is merely noisy |
| `LINE_SUPPLY` | starvation decaying from the head, no blocking anywhere, no boundary | an inbound material delay starves the line from the front and must not be blamed on a station (see 3.3) |

### Observed vs inferred candidates

For an *observed* candidate we also have `z_proc` — whether that station's own
tool cycle got slower. Blocking and starving do not change a station's
processing time, so `z_proc` cleanly separates "this station is the constraint"
from "this station is a victim of the constraint".

It is applied as a two-sided likelihood ratio, so it cuts both ways: an
instrumented station running at normal speed is evidence *against* it being the
constraint. Hidden stations get no such term — we have no direct evidence about
them either way. **That asymmetry is what lets the twin prefer a blind station
over an instrumented neighbour that is demonstrably running fine.**

---

## 5. Calibration: the part that makes the false-alarm rate honest

Stations on a coupled line are **not independent observations**. Measured
cross-station correlation on nominal data is ≈ 0.38–0.41, giving an effective
sample size of roughly **2.2 of 32 stations**. Treating them as independent
inflates the evidence by ~14×, and the detector fires constantly.

Two things are therefore calibrated on a **second, independent** disturbance-free
run (never on the run used to fit the baseline, and never on evaluation data):

- **`tau`** — the correlation correction, `n_eff / n_observed`, applied to the
  flow log-likelihood.
- **`detect_llr`** — the detection threshold, read off the empirical null
  distribution of the statistic at a stated per-window false-alarm target.

Choosing the threshold this way makes the false-alarm rate a **design
parameter with a target**, rather than an accident of a hand-picked constant.

---

## 6. The second path: quality

The flow model detects a pure quality drift in **0% of windows**. That is the
correct result for that mechanism: a station that keeps perfect takt while
producing bad work leaves no timing signature at all.

So the twin carries a second, independent path.

Three pieces of structure make attribution possible without a sensor:

1. **Vehicle genealogy** — every vehicle passes every station in a known order.
   Alignment is in *vehicle-index space*, so a defect found at the end of the
   line is counted against the pool that was passing through the source station
   when it was made. No interpolation is involved, and hidden stations are on
   identical footing to instrumented ones.
2. **Failure-mode propensity** — process FMEA already says which station can
   physically produce which defect. A sealer station cannot cause a torque fault.
3. **Detection lag** — the gap between where a defect is made and where it
   surfaces.

The discriminative signal is **not the defect count, it is the shape of the
failure-mode histogram**. Pooled counts are modelled as a sum of per-station
Poisson contributions,

```
E[O_t] = N * sum_k lambda_k * p_k(t)
```

and for each candidate a single multiplier `m_k >= 1` is fitted on that
station's contribution alone, scored by likelihood ratio.

> An earlier version soft-assigned each defect across candidates in proportion
> to a prior and tested each station's total. That diluted an eleven-fold drift
> across every station sharing the failure mode, and ranked the true source
> **11th of 42**. The mixture formulation moved it to **1st**.

**Honest limitation:** this path pools several hundred vehicles for statistical
power, so it reacts more slowly than the flow path, and it returns a *shortlist*
rather than a verdict.

---

## 7. What would falsify this

| Claim | How it is checked | Where |
|---|---|---|
| The blocking/starving signature is real | asserted against simulator ground truth at 4 stations | `test_physics.py` |
| Deviation channels are calibrated | held-out nominal must score ≈ 0 | `test_shadow_sensing.py` |
| A blind station's cycle time is estimable | compared against ground truth the model cannot see | `test_shadow_sensing.py`, `cycle_time_inference.csv` |
| The advantage comes from shadow-sensing, not tuning | B2 is the *same model* minus hidden candidates | `baselines.py` |
| It is not just firing more often | every method calibrated to a matched false-alarm rate | `experiments.py` |
| It stays quiet when nothing is wrong | clean episodes and a supply-delay scenario | `test_shadow_sensing.py` |
| A supply delay is not blamed on a station | upstream blocked vs starved discriminator | `test_pipeline_and_hitl.py` |
| Cold-start transients are excluded | windows begin only once the line has filled | `test_pipeline_and_hitl.py` |

The sharpest single check is the inferred cycle time. In the view given to the
model, that station's cycle time is **unmeasurable**. The simulator knows it. So
the estimate is *checked*, not asserted — and at 100% coverage RippleTwin and
B2 produce **identical** results, which is exactly what should happen when
there is nothing hidden to infer.

---

## 7½. A design we tried and did not promote: flow+quality LLR fusion for ambiguous blind pairs

Section 3 documents three things that were *wrong* and got fixed. This one is
different: the mechanism is not wrong, it is just — on the evidence gathered
— not shown to help, and it was not promoted into the default pipeline.

**The idea.** Section 8 below (and `docs/RESULTS.md` §7) documents that
adjacent blind-station pairs (S32/S33, S37/S38, and several others) are
~85–97% cosine-confusable to the flow model — it cannot always tell which of
the pair is the real constraint. `twin/genealogy.py::quality_state` computes
an entirely independent LLR, from defect-type propensity rather than timing,
for the same candidate stations. `twin/evidence_fusion.py` (new, additive)
combines the two log-likelihoods for stations inside a flagged-ambiguous
group, in the same "independent evidence channels add in log-space" spirit
`shadow.py`'s own `z_proc` term already uses. It is wired into
`twin/pipeline.py::infer()` behind `ctx.enable_evidence_fusion` (default
`False`) and never overwrites the audited `top_station` field — it only adds
opt-in `fused_top_station`/`fused_llr_margin` columns.

**The gated comparison.** Per the implementation runbook's decision gate: ran
the full flagship protocol twice, identical `ExperimentConfig` (8 tune + 24
test episodes, seeds 1000–1007/5000–5023, coverages 1.00/0.75/0.50/0.25,
`target_window_fpr=0.01`), differing only in `fusion_enabled`.

- **Every non-`COMBINED` row of `by_fault_kind.csv` was byte-identical**
  between the two runs (`non_combined_base.equals(non_combined_fused)` is
  `True`) — the gating is clean; fusion genuinely touches nothing outside its
  intended scope.
- **The `COMBINED` rows were also byte-identical**, for every method
  including RippleTwin, at every coverage level. This is not, by itself,
  evidence that fusion "matched" flow-only performance — `evaluate_localization`
  (via `models/baselines.py::build_methods`) reads only `shadow`'s original
  `top_station`/`detected` columns, exactly as designed so fusion could never
  destabilise the audited number. It **structurally cannot see**
  `fused_top_station` at all. Any two runs that differ only in
  `fusion_enabled` will produce an identical `by_fault_kind.csv`, regardless
  of what the fusion logic concludes internally.
- To find out whether `fused_top_station` would actually have changed
  anything, the two `COMBINED`-fault-kind held-out test episodes (seeds 5008
  and 5021 — this is the entire `COMBINED` slice of the standard 24-episode
  test split) were inspected directly, comparing `fused_top_station` against
  `top_station` window-by-window during the fault, at coverage 0.75 and 0.50
  where the true station (S10 for seed 5008, S32 for seed 5021) sits inside a
  flagged-ambiguous group. Seed 5008 never reached a detected window in this
  slice at either coverage, so it contributes no comparison. For seed 5021,
  fusion **did** fire — `fused_llr_margin` was a real, non-`NaN` number on
  every detected window, confirming overlapping quality evidence existed —
  but `fused_top_station` equalled `top_station` on **every one of the 9
  (coverage 0.75) and 10 (coverage 0.50) detected windows**. Localisation
  accuracy against ground truth was identical either way (0.556 at 0.75
  coverage, 0.700 at 0.50 coverage, exact-station match rate). Zero of 19
  fusion evaluations changed the pick.

**Reading this honestly.** This is not a "worse" result, and it is not
compelling enough to call a "narrow win" either — it is a **null result on a
badly underpowered sample**. The standard evaluation protocol contains
exactly two `COMBINED`-fault-kind held-out episodes, one of which never
produced a comparison at all. Two data points, on which fusion agreed with
the unfused posterior one hundred percent of the time, is not evidence that
quality-path evidence never helps disambiguate a flagged group — it is
evidence that this particular narrow mechanism, on this particular thin
slice of this synthetic line, has not yet been shown to move the needle.
Forcing either a "promote it" or "it's worse" conclusion out of two episodes
would overstate what was actually measured.

**Decision: not promoted.** `twin/evidence_fusion.py` and the
`enable_evidence_fusion`/`quality_baseline` plumbing stay in the repository,
fully tested (`tests/test_evidence_fusion.py`, including the regression
guard proving `infer()`'s default output is byte-identical whether or not
the attributes are present), opt-in, and off by default. `top_station`
remains the sole audited field every existing table and dashboard reads. A
real test of this idea needs either a larger synthetic `COMBINED` episode
set restricted to ambiguous-group source stations, or real production data
where quality alerts and flow disturbances co-occur more than twice in 24
episodes — neither of which this evaluation pass had the budget to build.

---

## 8. Known limitations

- **Serial-line assumption.** The propagation model assumes one directed path.
  Parallel sub-lines, rework loops and merge/split points need the pattern
  matrix rebuilt from the real graph. The likelihood generalises; the specific
  `+1 upstream / −1 downstream` pattern does not.
- **Adjacent blind stations are not separable.** Two hidden stations next to
  each other with no sensor between them cannot be told apart by flow evidence.
  The system reports the contiguous group and its combined mass, and abstains
  from naming one.
- **The constraint must bind.** A station that slows but stays inside takt
  produces no starvation, so there is nothing to localise. This is reported as
  "deviating but inside takt" rather than dressed up as a detection.
- **Quality attribution is a shortlist.** Median rank of the true source is
  mid-single-digits out of 42, not 1.
- **Everything is synthetic.** The physics is faithful and the disturbances are
  plausible, but no claim here has been validated against a real production
  line. That is the first thing a pilot would have to establish.
