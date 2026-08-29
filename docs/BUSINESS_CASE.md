# Business case

## Reading rules

Four categories of statement appear in this document. They are never mixed.

| Tag | Meaning |
|---|---|
| **OFFICIAL** | Taken from the Accenture Innovation Challenge Round 2 brief |
| **PROTOTYPE RESULT** | Measured by our code on synthetic data, reproducible from this repo |
| **ILLUSTRATIVE ASSUMPTION** | A number we chose, stated so it can be replaced with a real one |
| **EXPECTED PILOT IMPACT** | What we would expect to see, requiring real-world validation |

No number in this document is a measurement from a real production line. We have
never had access to one.

---

## 1. Who buys this, and why

**Primary buyer:** Plant Manager / Manufacturing Engineering, with the Head of
Manufacturing Digital as economic sponsor.

The purchase decision is *not* "should we have a digital twin". Most large
automotive plants already have, or are being sold, some form of one. The
decision is: **"our twin only covers the instrumented stations — do we spend on
retrofitting the rest, or on extracting more from what we already have?"**

RippleTwin is the second option, and it is deliberately positioned as a
complement to instrumentation rather than a replacement for it.

---

## 2. Why the problem is expensive

**OFFICIAL** — the Round 2 brief states the conditions directly:

- assembly lines "mix legacy and modern equipment, so sensor coverage is often
  inconsistent — some stations are richly instrumented, others rely entirely on
  manual checklists"
- "a defect introduced early in the line may not surface until a much later
  inspection point, by which time many downstream units may carry the same
  undetected issue"
- "most plants only allow retrofits during scheduled, infrequent maintenance
  windows"

That last point is the one that decides the business case. A retrofit programme
is not merely a cost — it is a cost that can only be spent a few times a year.
Inference deploys against data the plant already records.

---

## 3. Value drivers

| Driver | Mechanism | Evidence class |
|---|---|---|
| Earlier constraint detection | names the station before the shortfall is visible on the board | PROTOTYPE RESULT |
| Reduced dispatch waste | names *which* station, not "somewhere in body shop" | PROTOTYPE RESULT |
| Fewer defect escapes | attributes defect-type excess to a source before the next gate | PROTOTYPE RESULT (shortlist) |
| Lower instrumentation requirement | blind stations become inferable rather than dark | PROTOTYPE RESULT (coverage sweep) |
| Faster twin deployment | no retrofit window needed to start | ILLUSTRATIVE ASSUMPTION |

---

## 4. The model

All inputs below are **ILLUSTRATIVE ASSUMPTIONS**. They are exposed as editable
fields in the Leadership view of the dashboard precisely so that a reviewer can
replace them and watch the answer change.

### Inputs

| Input | Value | Basis |
|---|---|---|
| Contribution margin per vehicle | $2,200 | placeholder, plant-specific |
| Average rework cost per defect | $420 | placeholder, plant-specific |
| Hidden-station disturbances per line per year | 60 | roughly one per week |
| Earlier reaction per event | 25 min | placeholder — lead-time testing was inconclusive (see below) |
| Recovery factor | 55% | finding a constraint sooner shortens it, it does not remove it |
| Line nominal output | 50 veh/h | **PROTOTYPE RESULT** from the simulated line |
| Fully-installed station retrofit | $18,000 | placeholder |
| RippleTwin year-1 deployment | $150,000 | placeholder |
| Annual opex | 20% of deployment | placeholder |

### Arithmetic

```
vehicles recovered   = events x minutes_saved x (rate/60) x recovery_factor
                     = 60 x 25 x (50/60) x 0.55            = 687 vehicles/yr
throughput value     = 687 x $2,200                        = $1,512,000
defects avoided      = 60 events x 6 units                 = 360
quality value        = 360 x $420                          = $151,200
                                                             -----------
annual gross value                                           $1,663,200
less opex (20% of $150,000)                                     -$30,000
                                                             -----------
net annual value                                             $1,633,200
payback on $150,000                                          ~1.1 months
```

### Where this is weakest — read this before believing it

The result is dominated by one assumption: **contribution margin per recovered
vehicle**. If a plant is not capacity-constrained, a recovered vehicle is worth
close to nothing, because the line would have made up the shortfall anyway. In
that case throughput value collapses and the case rests on the quality and
dispatch-efficiency drivers alone.

So the honest statement is conditional:

> RippleTwin's throughput value is real **only on a capacity-constrained line**.
> On a line running below demand, the case must be made on defect escapes and
> maintenance dispatch efficiency, and it is a much smaller case.

We would rather state that than present a single impressive number.

The 25-minute figure is an **ILLUSTRATIVE ASSUMPTION**, not a measured result —
we tried to measure it and are saying plainly that we couldn't. The early-warning
lead-time experiment is real (`docs/RESULTS.md`, `results/tables/early_warning_summary.csv`),
but the sample of faults that ever produced a visible board-level shortfall was
too small (single digits) to responsibly quote a reliable figure, so `docs/RESULTS.md`
does not, and neither should this document. Note, too, that where the metric
*could* be computed, it trended the wrong way: this prototype's own measured
mean and median lead time were both negative — detection arrived after the
reference point, not before it — which is exactly why 25 minutes is presented
here as a planning assumption for a future deployment, not a claim about the
current system. Replace it with your own estimate in the dashboard's Leadership
view.

---

## 5. Sensor economics

The comparison is not only cost, it is calendar.

| | Retrofit every blind station (PLC-integrated) | RippleTwin |
|---|---|---|
| Capital | 10 stations x $18,000 = **$180,000** | **$150,000** year 1 |
| When it can be installed | scheduled maintenance windows only | against existing data |
| Ongoing | sensor calibration and failure | model monitoring and drift |
| Scope | those 10 stations | every station on the line |
| Failure mode | sensor dies, station goes dark again | inference degrades, and reports that it has |

**We do not claim zero sensors.** RippleTwin needs instrumented stations on both
sides of a blind one; that is the mechanism. The claim is narrower and testable:
the stations already instrumented carry more information about their neighbours
than a conventional twin extracts, and the coverage sweep in
`results/tables/` shows exactly where that stops being true.

The most defensible framing to a plant: **the coverage experiment is a sensor
placement tool.** It tells you which blind stations inference already covers well
enough, and which ones genuinely need a sensor — so the retrofit budget goes to
the stations where it actually buys something.

That reframes the sale. RippleTwin is not competing with the instrumentation
budget; it is telling you how to spend it.

### 5a. The $18,000 figure is the expensive way to retrofit a station — say so

**ILLUSTRATIVE ASSUMPTION, correction.** $18,000/station assumes a
PLC-integrated retrofit: wiring into the control cabinet, new I/O points,
commissioning. That is a real number for that scope, but it is not the only
way to get blocked/starved-equivalent telemetry out of a legacy station, and
presenting it as the only option overstates RippleTwin's cost advantage in a
way that would not survive a plant engineer's first question.

**Non-invasive current-clamp monitoring** — a clamp-on CT sensor around the
motor's power lead, no PLC program touched, no cabinet opened — is a
well-established alternative for exactly this use case, and runs
**roughly $200–$2,000 per machine installed**, with data flowing in minutes
rather than weeks because there is no control-system integration to
commission.[^clamp]

| Option | Cost / station | Integration | What it recovers |
|---|---|---|---|
| PLC-integrated retrofit | **$18,000** | control-cabinet wiring, new I/O, commissioning | full RICH-tier telemetry: cycle time, blocked/starved, torque/vibration/temperature |
| Non-invasive clamp-on retrofit | **$200–$2,000** | clip a CT sensor on the power lead; no PLC change | a running/idle signal close to BASIC-tier blocked/starved — motor current drop approximates station idle, current spike approximates blocked/jammed |
| RippleTwin | **$150,000 year 1** (whole line) | reads data already collected | infers the same station from its instrumented neighbours; no new hardware |

This changes the honest version of the pitch. It is **not** "sensors cost
$180,000, we cost $150,000." It is:

> A plant can very likely get approximate BASIC-tier signal onto most of its
> blind stations for **$2,000–$20,000 total** across 10 stations using
> non-invasive clamps — cheaper than we originally assumed, and without
> waiting for a maintenance window. RippleTwin's case does not rest on sensors
> being expensive. It rests on two things the clamp-on route does not solve:
> **(1)** clamp-on current draw is a coarser, noisier proxy for
> blocked/starved than a PLC-derived signal — it still needs calibration and
> does not recover torque/vibration/temperature at all, and **(2)** even a
> fully clamp-on-instrumented line still has to interpret that data; RippleTwin
> is the layer that turns per-station signal, however it is collected, into a
> located constraint, a forecast and a recommendation. A plant instrumenting
> cheaply still benefits from not having to build that layer itself.

**What we have not done:** validated that clamp-on current draw actually
produces a usable blocked/starved-equivalent signal on *this* line's motors —
that is a real engineering question (load type, duty cycle, sensor placement
all affect it) and belongs in a Phase 0 assessment, not asserted here.

[^clamp]: Cost range compiled from vendor and industry sourcing on
    non-invasive current-clamp / CT-based machine monitoring (2026), e.g.
    Guidewheel's comparison of PLC monitoring vs. clamp-on current sensors
    (<https://www.guidewheel.com/blog/plc-monitoring-vs-current-sensors>) and
    retrofit-IIoT cost guides citing $200–$2,000 per asset for clamp-based
    retrofits. Treat as an **ILLUSTRATIVE ASSUMPTION** sourced from public
    vendor material, not a quote for this line.

---

## 6. Deployment

| Phase | Duration | Scope | Exit criterion |
|---|---|---|---|
| 0 — Data readiness | 2–4 weeks | confirm PLC timestamps, gate results, build sequence are recoverable | required tables exist and are joinable |
| 1 — Shadow mode | 8–12 weeks | alerts logged, nobody acts on them | measured precision per station on the ledger |
| 2 — Live advisory | 12 weeks | supervisors see and act on alerts | adoption, and precision holding on real outcomes |
| 3 — Plant rollout | 2 quarters | remaining lines | per-line configuration under one week |
| 4 — Multi-plant | 12 months+ | other sites | baselines transfer across equipment vintages |

Phase 1 exists because of the brief's own warning: **"false alarms about defects
that don't materialise can erode floor-level trust in the system quickly."**
Shadow mode accumulates predictions and outcomes in the ledger at zero
operational risk, so precision is *measured* before anyone is asked to act.

**Integration:** read-only. RippleTwin consumes PLC timestamps, MES gate results
and the build sequence. It writes to no control system. The brief notes that
"modifying live production systems (PLCs, line control logic) carries real
operational risk" — so the prototype does not do it, and neither should a pilot.

---

## 7. Scalability

| Level | What changes | What does not |
|---|---|---|
| Another line | topology YAML, refit baseline | estimator, calibration procedure, UI |
| Another plant | station count, buffer sizes, variants | the flow physics |
| Another vintage | sensor tiers, noise scales | the propagation model |

The one genuine limitation: the propagation matrix assumes a **serial line**.
Parallel sub-lines, rework loops and merge/split points need it rebuilt from the
real process graph. The likelihood formulation generalises; the specific
"+1 upstream / −1 downstream" pattern does not.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| No real-world validation | **High** | shadow-mode pilot before any operational claim |
| Alert fatigue kills adoption | High | threshold set from a stated false-alarm target; abstention when ambiguous |
| Adjacent blind stations inseparable | Medium | report the group, never guess one; flag as a sensor-placement candidate |
| Baseline drift after retooling | Medium | monitor deviation channels; refit on a new nominal window |
| Plant lacks joinable data | Medium | Phase 0 gate before commercial commitment |
| Value depends on capacity constraint | **High** | qualify the line during Phase 0; do not sell throughput value to an unconstrained plant |
| An incumbent platform copies it | Medium | see below |

## 9. On the moat

Being honest: **the algorithm is not the moat.** It is a few hundred lines of
physics and a likelihood ratio, and it is documented publicly in this repo. An
incumbent could implement it.

What is defensible is narrower:

1. **The insight itself is the differentiator, and it is not the default.** The
   industry framing is "add sensors to see more". This inverts it: use the
   coupling between stations you already measure. Most twin roadmaps do not
   start there.
2. **The outcome ledger compounds.** Per-station precision measured against what
   technicians actually found is plant-specific data that accrues from day one
   and does not transfer to a competitor.
3. **Sensor-placement guidance is a durable wedge.** Being the system that tells
   a plant where instrumentation is actually worth buying is a position inside
   the capital planning cycle, not just the operations budget.
