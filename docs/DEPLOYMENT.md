# Deploying this in a real plant

Written to answer the question a manufacturing consultant will actually ask:
*"what do you need from my plant, where does your software sit, and who acts on
the output?"*

---

## 0. The failure mode we are designing against

Reviews of failed industrial digital-twin projects converge on one cause, and it
is not the model. It is the data layer — *"beautiful replicas that don't support
decision-making"*, built on fragmented and partially integrated data, with heavy
investment in visualisation and little in the plumbing underneath.

The predictive-maintenance literature adds a second, organisational one:

> "The real challenge is usually not detecting the problem. It is shortening the
> time between the first warning sign and the moment somebody actually takes
> action."

with the named symptoms being *"a dashboard shows an alert but nobody owns the
next action"*, alerts that never become work orders, and the decisive one:
*"the supervisor does not want to lose output. Nobody knows whether to stop
production now or wait."*

Both shaped what we built. RippleTwin has **no 3D model and deliberately modest
visualisation**; the whole product is the inference and what happens after it.

---

## 1. What we need from the plant

Every input is listed in `rippletwin.integrate.contract`, with its Purdue level,
the interface that exposes it, and **what we lose if it is missing**. Run it:

```bash
PYTHONPATH=src python -m rippletwin.integrate.assess --contract
```

| Signal | Required | From | Interface |
|---|---|---|---|
| Station state (running / blocked / starved / fault) | **yes** | PLC or existing OEE system | OPC UA (PackML), MQTT Sparkplug, historian |
| VIN / serial read per station | **yes** | scanner, MES traceability | MES API |
| Build sequence with model variant | **yes** | MES / scheduling | database view |
| Shift pattern and planned downtime | **yes** | MES / ERP calendar | database view |
| Station order and buffer capacities | **yes** | layout + controls drawings | one-time YAML |
| Gate results with defect codes | no | MES quality module | MES API |
| Station-to-defect-type map (process FMEA) | no | control plan | one-time YAML |
| Per-vehicle cycle time | no | PLC timestamps | OPC UA |
| Torque / vibration / temperature | no | tightening controllers | OPC UA |
| Buffer occupancy | no | conveyor counters | SCADA |

**Nothing outside that list is read, and nothing is written back.**

### Why the required list is short

The critical signal — blocked and starved state — is not exotic. It is a
standard equipment state in OEE systems, conventionally derived at the PLC as:

```
STARVED = motor running AND infeed empty
BLOCKED = motor running AND outfeed full
```

from the infeed/outfeed photocells most conveyor-linked stations already carry
for interlock purposes. PackML (ISA-TR88.00.02) standardises these states across
modern Siemens, Allen-Bradley, Beckhoff and B&R PLCs.

**The most useful thing we learned researching this:** because starved and
blocked are *line* losses rather than station losses, plants normally exclude
them from a station's own OEE and file them as a nuisance category — penalising
a machine for being starved is a reliable way to start an argument with its
operators. So this data is frequently collected and rarely used.

> RippleTwin's primary input is, in a real sense, waste data the plant is
> already paying to store.

That is the cheapest possible starting position for an industrial analytics
project, and it is why Phase 0 is weeks rather than quarters.

---

## 2. Where the software sits

```
Level 4   Enterprise IT / ERP
          ─────────────────────────────────────────────
Level 3.5 Industrial DMZ          <- historian replica, if that is the pattern
          ─────────────────────────────────────────────
Level 3   MES / operations        <-  RIPPLETWIN RUNS HERE
                                      reads: PLC state, MES traceability,
                                             gate results, build sequence
                                      writes: nothing into OT
          ─────────────────────────────────────────────
Level 2   SCADA / HMI / line control
Level 1   PLCs
Level 0   Sensors and actuators
```

RippleTwin is a **Level 3 read-only consumer**. It can equally read a Level 3.5
DMZ historian replica where a site's segmentation requires it. Data crosses the
IT/OT boundary in one direction only.

This is not caution for its own sake. The Round 2 brief itself notes that
"modifying live production systems (PLCs, line control logic) carries real
operational risk", and a system that cannot write cannot cause an incident. It
also removes the single largest obstacle to a pilot: no OT change request, no
validation of control logic, no maintenance window needed to start.

### Timing

Station events must share a time base to within **about a second** — roughly a
sixtieth of takt — because the method compares event ordering across stations.
NTP against a plant time server is sufficient; free-running PLC clocks are not.

This is a real deployment risk rather than a theoretical one: industry guidance
on OEE architecture is notably silent on clock synchronisation, so it is worth
checking rather than assuming. Where events are timestamped on arrival at a
historian instead of at the PLC, queueing delay appears as cycle-time variation
and will degrade localisation. `assess_readiness(..., clock_sync_s=...)` flags it.

### Volume

One row per vehicle per instrumented station. At 60-second takt and 42 stations
that is roughly 60 × 42 ≈ 2,500 rows/hour, about 60k rows/day. This is a
laptop-scale problem, not a big-data one, and it fits inside an existing
historian without new infrastructure.

---

## 3. Phase 0: find out before committing

```bash
PYTHONPATH=src python -m rippletwin.integrate.assess \
  --signals station_state,vehicle_identity,build_sequence,shift_calendar,line_topology \
  --stations 42 --instrumented 30 --clock-skew-s 0.5
```

Returns one of four verdicts:

| Verdict | Meaning |
|---|---|
| `FULL` | both the flow and quality paths run |
| `FLOW_ONLY` | bottleneck localisation runs; no defect attribution |
| `QUALITY_ONLY` | defect attribution runs; too few instrumented stations for flow |
| `NOT_VIABLE` | say so now, before anyone spends money |

Two verdicts are worth dwelling on because they are the honest ones:

* **`NOT_VIABLE`.** The most valuable output this tool can produce. Most failed
  twin projects should have received this answer and did not.
* **100% coverage.** The assessment explicitly reports that shadow-sensing has
  nothing to add on a fully instrumented line, because our own evaluation shows
  it adds nothing there. Use RippleTwin on the lines that have gaps.

---

## 4. What happens to an alert

An alert that stops at "S08, risk 0.81" walks straight into the failure mode
above. So the twin emits a **work order**, not a notification:

| Field | Why it is there |
|---|---|
| Owner **role** | "Maintenance technician (line-side)" — not "the plant". Roles outlive rosters. |
| Respond-by time | derived from forecast impact, not from a severity label |
| **Cost of waiting** | "~12.5 vehicles/hour while this holds; deferring to the break costs ~25" |
| Verification prompt | what to report back — the only way precision is ever measured |
| Escalation | who it goes to, and when, if nobody acts |
| CMMS payload | generic field names, so integration is a mapping not a project |

The cost-of-waiting number is the one we would defend hardest. It answers the
exact question practitioners say supervisors are stuck on, and RippleTwin
already has it: the ripple forecast is in vehicles per hour, so *act now vs act
at the break* is arithmetic.

Deliberately, a **monitor-only recommendation raises no work order at all**.
Manufacturing a task out of "keep an eye on it" is how alert fatigue starts.

---

## 5. Rollout

| Phase | Duration | What happens | Exit criterion |
|---|---|---|---|
| 0 — Readiness | 2–4 weeks | run the assessment; confirm tags, traceability, clock sync | a `FULL` or `FLOW_ONLY` verdict, and topology captured |
| 1 — Shadow mode | 8–12 weeks | alerts logged to the ledger, **nobody acts** | measured precision per station on real outcomes |
| 2 — Live advisory | 12 weeks | work orders reach technicians | technicians act, and precision holds |
| 3 — Plant rollout | 2 quarters | remaining lines | per-line configuration under one week |
| 4 — Multi-plant | 12 months+ | other sites and vintages | baselines transfer across equipment vintages |

Phase 1 exists because of the brief's own warning that false alarms "erode
floor-level trust in the system quickly" — and because trust is not recoverable
twice. It also matches standard practice for commissioning OEE systems, where
the recommendation is to run a shift alongside the manual log to build
credibility before anyone is asked to believe the number.

**The verification prompt on each work order is what makes shadow mode work.**
Without a recorded "found / not found", Phase 1 produces alerts and no evidence,
and Phase 2 becomes a matter of faith.

---

## 6. What would stop this in a real plant

Stated plainly, because these are the things that actually kill deployments:

1. **Traceability gaps.** If a VIN cannot be linked to station events, the
   quality path cannot run. Automotive plants generally have this for recall
   reasons, but "generally" is not "always", and it is a Phase 0 check.
2. **Manual stations with no photocells at all.** If a station has no infeed or
   outfeed sensing *and* neither does its neighbour, there is no boundary to
   find. The placement tool identifies exactly these.
3. **Clock skew**, as above.
4. **Non-serial topology.** Parallel sub-lines, rework loops and merge/split
   points need the propagation matrix rebuilt from the real process graph. The
   likelihood generalises; the specific pattern does not.
5. **Nobody owns the work order.** The most likely failure, and the least
   technical. If Phase 2 begins without an agreed owner role and response time,
   it will produce a well-calibrated alert stream that nobody reads.
