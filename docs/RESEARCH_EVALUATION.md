# Research evaluation — twelve papers against RippleTwin's actual situation

Deep technical due diligence on twelve candidate papers, evaluated against
RippleTwin's actual architecture, design constraints, and validated results —
not against their abstracts. This document is the output of a research-review
pass only; it recommends, it does not implement. Where an item is recommended
for implementation, the detailed procedure lives in
[docs/IMPLEMENTATION_RUNBOOK.md](IMPLEMENTATION_RUNBOOK.md).

Grounding: `README.md`, `docs/METHOD.md`, `docs/REFERENCES.md`,
`docs/JUDGE_QA.md`, `docs/LIMITATIONS.md`, and the actual code in
`twin/shadow.py`, `twin/genealogy.py`, `twin/placement.py`,
`models/baselines.py`, and `evaluation/experiments.py` (not just their
docstrings). The measured test count at the time of this review was **204**
(`pytest --collect-only`).

---

## 1. Executive summary

Of the 12 papers: **2 IMPLEMENT**, **4 CITE ONLY**, **1 FUTURE WORK**,
**4 REJECT**, **1 INSUFFICIENT EVIDENCE** (full text unavailable in the
research index — flagged rather than guessed from the abstract). Of 3
hybrids identified: **1 IMPLEMENT**, **1 CITE ONLY**, **1 FUTURE WORK**.

The single strongest finding is **Paper 11** (Ground-Truth-Aware Stress
Testing, arXiv:2608.14917). It is a domain-agnostic *evaluation
methodology* — not a smart-building-specific model — and RippleTwin's
existing architecture (a deterministic seeded simulator,
`factory/sensor_health.py`'s fault injection as a corrupted-sensing layer,
and `evaluation/experiments.py`'s "one physics run per episode, coverage
levels are *views* over that run" design) already satisfies every
structural precondition the methodology needs. It lets RippleTwin report,
for the first time, the difference between *"sensor dropout changed which
station we named"* and *"sensor dropout changed the actual consequence of
the recommendation"* — directly strengthening the sensor-fault-robustness
story `factory/sensor_health.py` already sets up but doesn't yet measure at
the decision/outcome level. It is also low-risk: fully additive, writes to
a new results directory, touches no calibrated number.

The runner-up, and the most scientifically interesting single idea in the
whole exercise, is **Hybrid 2**: fusing the flow-path posterior
(`twin/shadow.py`) with the independent quality-path posterior
(`twin/genealogy.py::quality_state`) specifically for `COMBINED`-fault-kind
episodes where `twin/placement.py::ambiguity()` has already flagged two
blind stations as near-indistinguishable. It is the only candidate in the
whole set that could move the needle on the project's own documented
weakest point (adjacent-blind-station ambiguity) — but honestly, only for
the narrow subset of incidents that produce *both* a flow disturbance and a
quality signature. It does not fix the general 25%-coverage collapse.

---

## 2. Per-paper evaluations

### 1. arXiv:2306.16120 — Data-driven diagnostic analysis of dynamic bottlenecks in serial manufacturing systems

**What it actually proposes.** Two simple, deterministic statistics
computed *on top of* whatever bottleneck-detection method a plant already
runs: relative bottleneck frequency (`rbf`, the fraction of time-steps a
station is *the* named bottleneck over an observation window — sums to 1
across stations) and relative bottleneck severity (`rbs`, a momentary
snapshot metric — the current leader always scores 1, and every other
station's score is the ratio of its own "active period" to the leader's,
letting you see a *successor* bottleneck rising before it takes over). Both
metrics are explicitly method-agnostic — the paper states "there are no
restrictions in the choice of methods as long as they clearly identify a
single station as a bottleneck for each time point." Validated on a
7-station serial DES with static and shifting single/dual bottlenecks; no
training, no ML.

**Where in RippleTwin this would touch.** `twin/placement.py` and a new
small reporting layer. Notably, `placement.py::suspicion_from_shadow`
*already computes something functionally equivalent to `rbf`* — "how often
each station has been the twin's leading suspect" — just as an internal
weighting helper for `recommend_sensors`, not exposed as a standalone
shift-diagnosis report.

**Genuine scientific fit.** Strong for the frequency half, because
RippleTwin's own posterior (a single named `top_station` per window, with
persistence) is exactly the kind of pluggable "any method that names one
station per time point" the paper requires — no adaptation needed. Weaker,
but adaptable, for `rbs`: the paper's severity ratio is defined over each
station's measured *active period*, which RippleTwin doesn't track for
non-leading candidates. But `shadow.py`'s posterior already carries a
runner-up-vs-leader mass ratio in spirit (`group_prob`, competing
hypothesis scores) — substituting posterior-mass ratio for active-period
ratio preserves the metric's actual purpose (catch a shifting bottleneck
early) without requiring anything RippleTwin doesn't already compute.

**What would genuinely change if adopted.** A new formula — not a new
estimator. `rbf` mostly formalizes what `suspicion_from_shadow` already
computes internally; the new piece is a momentary shift-severity number
derived from the existing posterior's runner-up mass, giving supervisors an
early "your #2 suspect is catching up" signal that doesn't exist today.

**Impact on existing validated results.** None. This is a pure post-hoc
aggregation over already-computed `ShadowSensor` output frames — it reads
`top_station`/`llr`/`group_prob`, writes new tables, and touches no
existing calibration, threshold, or estimator.

**Verdict: IMPLEMENT** — small, safe, genuinely useful, and the `rbf` half
is honestly closer to "expose what already exists" than "build something
new." Full execution plan: [Plan A](IMPLEMENTATION_RUNBOOK.md#plan-a).

**Outcome (built).** Shipped as `evaluation/bottleneck_diagnosis.py`,
merged to `main`. `ShadowSensor` needed one additive column
(`runner_up_station`/`runner_up_prob`) that didn't already exist — the
only place this touched a core file. Sanity-checked against a real
`S1_HIDDEN_BOTTLENECK` run: the top-ranked station matched the injected
fault exactly. 5 new tests, all passing.

---

### 2. arXiv:2607.24819 — Dynamic Multi-Criteria Bottleneck Severity Index (DMBSI) for semiconductor wafer manufacturing

**What it actually proposes.** A composite bottleneck-severity score
combining five sub-scores (processing load, queue congestion, process-time
variability, cycle-time sensitivity, rework burden) computed directly from
standard MES log fields, with the five weights calibrated by a genetic
algorithm against 22 real production lots' *observed* cycle-time
contributions (5-fold CV, r=0.80 vs. r=0.74 for an expert-heuristic
baseline). Validated on a real 200mm Seagate wafer fab with reentrant
(revisit-the-same-step) process flow.

**Where in RippleTwin this would touch.** Conceptually `twin/predict.py`'s
severity/early-warning layer — but only in the fully-instrumented regime,
because every one of the five sub-scores is computed from that station's
*own* MES timestamps.

**Genuine scientific fit.** This is the sharpest topology mismatch in the
whole list. DMBSI's entire mechanism assumes every station being scored
reports its own log — there is no notion of, and no mechanism for, a
station with zero telemetry. It is validated on reentrant, batch-lot,
MES-timestamp process flow (a lot revisits the same step type at multiple
stages), not a single-pass discrete-unit serial line. Its rework sub-score
(S₅) models loop-back-through-the-same-step rework, which doesn't map onto
RippleTwin's spur-and-remerge rework model
(`configs/plant_c_rework.yaml`). Worst of all for direct adoption: it needs
historical ground-truth cycle-time-attribution data to GA-fit its weights —
a supervised calibration step RippleTwin's core estimator deliberately
avoids.

**What would genuinely change if adopted.** Nothing usable for the
hidden-station problem. At best, if RippleTwin's line were ever fully
instrumented (the exact regime where the project's own docs already say
"shadow-sensing has nothing to add"), a DMBSI-style composite could rank
observed-station bottlenecks slightly better than a single metric — a
marginal win in exactly the case that isn't RippleTwin's differentiator.

**Impact on existing validated results.** None (not adopted).

**Verdict: CITE ONLY.** Zero code risk, real positioning value: a leading
multi-criteria bottleneck-severity framework from a real $100B/yr-capex
industry, validated on real fab data, *still* assumes every station reports
its own log. That's a strong, independently-sourced data point for
`docs/METHOD.md`'s argument that the hidden-station problem is the one
thing conventional bottleneck analytics — even sophisticated,
industrially-validated ones — don't address.

---

### 3. arXiv:2507.09742 — Causality-informed Anomaly Detection in Partially Observable Sensor Networks (Causal DQ)

**What it actually proposes.** A deep Q-network for *sequentially choosing
where to place sensors* under partial observability, where a causal graph
(learned via PC-algorithm-style conditional-independence tests on
observational data, with no interventions) is folded into the RL agent's
state, reward, and exploration via a "causal entropy" regularizer — proven
to converge faster and with tighter error bounds than a non-causal DQN.
Trained on synthetic Erdős–Rényi random-DAG environments (p=10/50/100
variables) plus one real-case application; it is fundamentally a trained,
opaque neural policy.

**Where in RippleTwin this would touch.** `twin/placement.py` (sensor
recommendation) or the causal-vs-correlation framing in `docs/METHOD.md`.

**Genuine scientific fit.** The paper's headline argument — "a
correlation-only detector can't tell you which variable caused an anomaly
vs. which is a downstream victim, and knowing the causal graph means one
sensor can stand in for several effects" — is *exactly* RippleTwin's own
founding argument (README: "A correlation model sees S07 blocking and S09
starving and has no basis for saying which caused which"). But the
mechanism doesn't transfer: Causal DQ *discovers* causal direction
generically via conditional-independence tests over an arbitrary
multivariate graph, then *learns* a placement policy via RL. RippleTwin
already knows the causal direction analytically, a priori, from
conservation of material — it has no discovery problem to solve, and
reintroducing a trained RL policy would reinstate exactly the
training-data/black-box dependency the project rejected for the GNN/BSTAN
alternative in `METHOD.md §0`.

**What would genuinely change if adopted.** Nothing in the code. A new
citation — independent, recent (2025), rigorous (regret-bound-backed)
academic validation of "causality beats correlation for
partial-observability sensor problems," from a different application
domain (general IIoT), strengthening the case that RippleTwin's B1
(Isolation Forest, correlation-only, 0% on hidden sources) fails for a
*principled*, not incidental, reason.

**Impact on existing validated results.** None.

**Verdict: CITE ONLY.**

---

### 4. arXiv:2205.02827 / PMC9790886 — Identifying Cause-and-Effect Relationships of Manufacturing Errors using Sequence-to-Sequence Learning (VW car-body production)

**What it actually proposes.** A two-module pipeline (VMAS), trained on
real Volkswagen Commercial Vehicles PDA (production data acquisition) logs:
(1) an MLE-threshold peak-detection labels each per-action duration as
"source error" (logged) or "knock-on error" (unlogged, delayed downstream,
~71.68% of ~40k sequences contain one or the other); (2) LSTM/GRU/
Transformer seq2seq models are trained (50 epochs, 80/20 split) to forecast
the next 1–7 actions' durations, with the Transformer winning at longer
horizons. Requires a large historical labeled corpus per action and
produces an opaque forecaster.

**Where in RippleTwin this would touch.** `twin/genealogy.py` (RCA) or
`twin/propagate.py` (ripple forecast) — the phenomenon it targets (an
upstream disturbance cascading into unlogged downstream delays) is
structurally the *same physical signature* the blocked/starved boundary
formalizes.

**Genuine scientific fit.** The *setting* transfers (fully-automated,
sequential-station car-body assembly — genuinely the right topology). The
*method* does not: VMAS's PDA data logs every station's own action
durations — there is no hidden-station scenario anywhere in the paper; it
is the fully-observed regime RippleTwin's B2 baseline already represents.
And its causal link between source and knock-on errors is not asserted a
priori from physics — it's *learned* purely from temporal co-occurrence in
a training corpus, which is precisely the "learned correlation" RippleTwin's
flow model is built to avoid (a correlation-based detector "has no basis
for saying which caused which," per README). Its own source-vs-knock-on
distinction is exactly the physical fact RippleTwin *derives* (block
upstream, starve downstream) rather than learns.

**What would genuinely change if adopted.** Nothing beneficial.
`twin/propagate.py` already computes the downstream-delay cascade
deterministically and auditably from flow arithmetic ("a plant engineer can
check it on paper"); a seq2seq forecaster trained on RippleTwin's synthetic
telemetry would be a strictly worse, opaque, RMSE-scored replacement for
something already computed exactly, with none of VMAS's actual value (real
production-data insight) surviving the swap.

**Impact on existing validated results.** None (not adopted). If it were
force-fit into `twin/propagate.py`'s role, it would invalidate every
downstream ripple-forecast number in `results/`.

**Verdict: REJECT.** Argued precisely: (a) requires full instrumentation at
every station, defeating the entire premise of the comparison against
RippleTwin's differentiator; (b) requires a large historical training
corpus and produces a black-box forecaster, directly contradicting the
deterministic/checkable design constraint — the identical objection
`METHOD.md §0` already raises against BSTAN/GNN applies verbatim; (c) the
phenomenon it targets is already solved, more cheaply and more auditably,
by `twin/propagate.py`'s existing flow arithmetic.

---

### 5. arXiv:2604.26593 — PiGGO: Physics-Guided Learnable Graph Kalman Filters for Virtual Sensing

**What it actually proposes.** A learned Graph Neural ODE (with
physics-guided inductive biases from structural mechanics — mass/stiffness
connectivity) serving as the continuous-time state-transition model inside
an Extended Kalman Filter, for estimating unmeasured node states
(displacement/velocity/acceleration) in a *continuous nonlinear vibrating
structure* (a sensor-array truss, a bridge model) from sparse
accelerometer/strain readings, with claimed generalization across
topologically similar structures. Requires offline training on labeled
response trajectories with *known initial state* — an explicit,
acknowledged limitation of the method.

**Where in RippleTwin this would touch.** `twin/shadow.py`'s propagation
matrices — this is the paper with the most surface-level resemblance to
RippleTwin's actual problem (infer unmeasured graph-node state from sparse
sensors + topology).

**Genuine scientific fit.** The surface resemblance ("virtual sensing at
unmeasured graph nodes via a topology-aware model") does not survive
contact with the domain: PiGGO's substrate is continuous nonlinear
vibrational dynamics (an ODE with mass/stiffness/nonlinear restoring
forces), while RippleTwin's is discrete-event conservation-of-material with
`max()`-governed blocked/starved/processing states — there is no
continuous, differentiable displacement field to learn. PiGGO's
"generalization across topologies" claim is about physically similar
mechanical structures of different span/size sharing the *same*
nonlinearity form learned from training data — not about generalizing
across categorically different manufacturing-line shapes the way
RippleTwin's *derived* (untrained) buffer-distance propagation already does
across `line_42`, `plant_b_parallel`, and `plant_c_rework` with zero
retraining. Adopting PiGGO's mechanism would require training data and
known-initial-state episodes, reinstating exactly what the derived-not-
learned choice was made to avoid.

**What would genuinely change if adopted.** Nothing directly
transplantable. The closest existing analog — wrapping a derived model
inside a Bayesian filter for uncertainty-aware online estimation — is
already `shadow.py`'s Gaussian likelihood + posterior (with
`ShadowConfig.ewma_alpha` as a lightweight temporal-smoothing stand-in for
a Kalman update). Reframing that as "a Kalman filter" per PiGGO's
terminology would be relabeling, not a new capability.

**Impact on existing validated results.** None (not adopted).

**Verdict: REJECT.** Different physical regime (continuous mechanical
dynamics vs. discrete-event queueing/conservation-of-material) that doesn't
transfer its learned dynamics; its core mechanism (a trained GNODE, with a
known-initial-state training requirement the authors themselves flag as
limiting) reintroduces the training-data/black-box dependency the project
has already argued against for the same class of alternative, and offers no
capability the derived buffer-distance model doesn't already provide for a
topology it generalizes across *without* training.

---

### 6. arXiv:2402.00043 — Interactive Root Cause Analysis in Manufacturing with Causal Bayesian Networks and Knowledge Graphs

**Access caveat, stated plainly.** Full-text retrieval for this paper
returned mostly the reference list rather than the method sections
(introduction, system overview, evaluation). What follows is based on the
abstract plus what could be inferred from the (partial) available text, and
should be treated as lower-confidence than the other entries.

**What it actually proposes (abstract-level).** An interactive RCA tool for
EV manufacturing that combines a large-scale, expert-authored Knowledge
Graph of the manufacturing process with a Causal Bayesian Network learned
from sensor data — the KG constrains/guides the CBN learning so it doesn't
miss known cause-effect relationships or learn spurious ones — plus an
interactive UI where a process expert adds/removes KG edges, closing the
loop between expert and learned model, at BMW's electric-vehicle production
line.

**Where in RippleTwin this would touch.** `twin/genealogy.py::candidate_prior`
(the FMEA-derived structural prior for defect attribution) and `hitl/`
(the decision ledger).

**Genuine scientific fit (low confidence).** The high-level pattern —
expert structural knowledge bounds the hypothesis space, data narrows/
calibrates it, a human interactively corrects the result — is one
RippleTwin already implements independently: `candidate_prior` *is* the
KG-equivalent (FMEA-derived), `quality_state`'s Poisson-mixture LLR *is*
the data-narrows-the-hypothesis step, and `hitl/` *is* the
interactive-correction step (plus `ai/fmea_map.py`'s explicit "draft, not
configuration" review discipline). The BMW paper's stated problem — a CBN
with "a vast amount of potential cause-effect relationships" at
whole-manufacturing-process scale — is a genuinely larger-scale problem
than RippleTwin's 42-station, station-level defect attribution; adopting
general-purpose CBN structure learning for RippleTwin's much smaller
candidate space would likely be unneeded complexity relative to the
lighter-weight Poisson mixture already in place.

**What would genuinely change if adopted.** Nothing structurally required,
given the access caveat above.

**Impact on existing validated results.** None (not adopted).

**Verdict: CITE ONLY**, explicitly caveated as provisional (full text
unavailable) — independent, published (BMW EV manufacturing) validation
that the "structural prior + data-driven refinement + human correction"
pattern is a recognized, sound approach in this exact industry, worth a
line in `docs/REFERENCES.md`'s "adjacent work" section.

---

### 7. arXiv:2006.03610 — Root Cause Analysis in Lithium-Ion Battery Production with FMEA-Based Large-Scale Bayesian Network

**What it actually proposes.** A method to build a large-scale Bayesian
Network *directly from an FMEA*, requiring **no production data**: each
FMEA-identified failure becomes a binary random variable; Leaky Noisy-OR
gates reduce conditional-probability-table elicitation from
exponential-in-parents to one linearly-surveyable "trigger probability" per
cause-effect edge; a genetic-algorithm-based consistency-repair procedure
finds the closest-to-expert-given but internally-consistent probability set
when independently-elicited multi-expert ratings contradict each other (a
real, frequently-occurring problem the authors report); inference runs via
likelihood-weighting simulation. Validated at BMW's lithium-ion battery
pilot line, explicitly for the low-production-data ramp-up regime, with the
paper stating production data could later *refine* the network but is not
required to build it.

**Where in RippleTwin this would touch.**
`twin/genealogy.py::candidate_prior` / `QualityBaseline` — the flat
station→defect-type propensity table used today.

**Genuine scientific fit.** Strong on philosophy: no training data, fully
expert-elicited, every trigger probability traces to an explicit FMEA
rating — squarely compatible with the derivable/explainable design
constraint. Partial on scope: the battery paper models ~600 process
*characteristics* and up to 2,100 cross-stage cause-effect relationships
within a single multi-stage cell-production process — a genuinely
finer-grained, multi-hop causal graph than RippleTwin's current
station-level, single-hop defect-type propensity. Building an equivalent
network for the 42-station line would need new expert elicitation
(inter-defect trigger probabilities), beyond what `ai/fmea_map.py`
currently extracts (station↔defect-type pairs only, not multi-hop chains).
Crucially, the paper's BN machinery does **not** address the actually-hard
problem `genealogy.py` solves today — attributing a defect to a station
with *zero telemetry* via vehicle-genealogy interpolation and
detection-lag back-projection. The battery-cell BN assumes clean per-lot
fault records exist; it has no notion of an unmeasured station.

**What would genuinely change if adopted.** `candidate_prior()` could be
generalized from a flat propensity table into a proper multi-hop Bayesian
network with Noisy-OR trigger probabilities (elicited alongside
`ai/fmea_map.py`'s existing FMEA ingestion), letting the quality path
reason about cascading failure chains (e.g., "sealer drift → downstream
fixture misalignment → paint defect") instead of only single-station
propensity. This is a genuinely new capability — not currently present.

**Impact on existing validated results.** High, if adopted as a
replacement. `quality_state()`'s Poisson-mixture-LLR-over-station scoring
would need re-deriving to run over a BN posterior instead of a flat
per-station multiplier grid; every number in
`results/tables/quality_attribution.csv` would need full re-generation and
re-validation against the existing protocol.

**Verdict: FUTURE WORK.** Scientifically sound and philosophically
compatible with the project's constraints — genuinely worth doing — but it
requires new FMEA elicitation work beyond current scope (multi-hop trigger
probabilities, not just station→defect-type weights) and a substantial
rearchitecture of `genealogy.py`'s scoring math with full re-validation.
That is real, honest, non-trivial scope: an explicit roadmap item, not a
bolt-on.

---

### 8. PMID:36780862 — MPGE and RootRank: multi-level fault propagation root cause characterization

**Access caveat, stated plainly.** The paper's full text could not be
retrieved from the research index at all (two separate targeted attempts
returned "no full-text passages available"). Only the title and a
truncated abstract fragment are available: "existing root cause diagnosis
models only consider pairwise direct causality and ignore the multi-level
fault propagation, which may lead to..." — the sentence is cut off before
the mechanism, validation domain, or data requirements are stated.

**What it actually proposes.** Cannot be established with confidence. I
will not infer the mechanism, the process type it was validated on
(continuous vs. discrete-event), or its data requirements from a truncated
abstract and present that as verified.

**Verdict: INSUFFICIENT EVIDENCE — not adoptable pending further access.**
This is explicitly *not* a science-based rejection — it is an access
limitation. Recommendation: leave this paper out of `docs/REFERENCES.md`
entirely until someone can obtain the full text (it is a Neural
Networks / Elsevier journal article, likely behind a paywall not covered by
the research index) rather than citing a paper whose actual claims haven't
been verified.

---

### 9. arXiv:1711.08264 — Efficient constrained sensor placement for observability of linear systems

**What it actually proposes.** Formal graph-theoretic sensor-placement
theory for linear time-invariant systems, no training data, fully
deterministic: (i) the *structural observability index* (minimum steps to
fully recover state from outputs) — placing the minimum sensors to bound
this index at ≤2 is polynomial (O(d³log d)); at ≥3 it's proven NP-complete
in general, but polynomial for systems whose structure is a directed tree
with a self-loop at every state; (ii) *cardinality-constrained* placement
(choose r sensors to maximize the number of structurally observable states)
is proven NP-hard, but the objective function Ξ(S) is proven **monotone
submodular**, which is exactly the property that gives a simple greedy
algorithm a formal (1−1/e)-approximation guarantee
(Nemhauser-Wolsey-Fisher). Demonstrated on the IEEE 118-bus power grid
(~400 states).

**Where in RippleTwin this would touch.**
`twin/placement.py::recommend_sensors` — the same problem, the same shape
of algorithm (start empty, greedily add the candidate that most increases
coverage), and `placement.py`'s own docstring already carries a
hand-derived algebraic argument (`(aD - cb)² ≥ 0`) for why its
`separability` metric is non-decreasing when a sensor is added — a
*single-step monotonicity* claim, which is weaker than, but suggestively
similar in flavor to, the paper's formal submodularity proof.

**Genuine scientific fit.** Very strong at the formal/conceptual level.
RippleTwin's line is literally a graph-structured linear/near-linear
dynamical system (conservation of material propagating through buffered
stations), and `placement.py`'s greedy algorithm is already structurally
identical to the paper's Algorithm 4. The open, genuinely checkable
question the review surfaces: is `separability(S)` actually *submodular*
(diminishing marginal returns — the gain from adding sensor `s` to a small
set is ≥ the gain from adding it to a larger superset), not merely
non-decreasing under one addition? `placement.py`'s existing test
(`test_instrumenting_a_station_reduces_ambiguity`) verifies the weaker
property empirically for single additions; it does not verify
submodularity, which is the actual property needed to claim the (1−1/e)
guarantee. This is a real, well-posed, small mathematical question — not
yet answered in the codebase.

**What would genuinely change if adopted.** Nothing in behavior —
`placement.py` already runs a greedy algorithm that does the right thing
empirically. What would change is the *justification*: from "we checked
this holds in one test" to "this is a formally (1−1/e)-optimal greedy
algorithm, following the general submodular sensor-placement result of
Dey, Balachandran & Chatterjee (2017), provided `separability` is
submodular" — an upgrade to `docs/METHOD.md`'s and `placement.py`'s
docstring rigor, contingent on someone actually verifying (on paper, or
with a small property-based test) that the submodularity condition holds
for RippleTwin's specific `separability` function.

**Impact on existing validated results.** None. Purely a
documentation/justification exercise.

**Verdict: CITE ONLY** for the current codebase. This paper is also the
anchor of Hybrid 1, below — it is exactly the kind of "paper A formally
justifies existing module B" the hybrid step was designed to surface.

---

### 10. arXiv:2208.00584 — A sensitivity-based approach to optimal sensor selection for process networks

**What it actually proposes.** A method for minimum-sensor-set
observability and placement in nonlinear process networks: construct a
local sensitivity matrix S(k,0) = ∂y(k)/∂x(0) (via the chain rule through a
*differentiable* nonlinear state-transition function f), test observability
via its rank (SVD), then successively orthogonalize columns to rank
candidate sensors by how much *new*, linearly-independent information each
adds (avoiding near-redundant sensor pairs) — computationally far cheaper
than the mixed-integer-programming alternatives it replaces. Validated on a
4-CSTR chemical reactor network and a wastewater treatment plant —
continuous-state, continuous-time (discretized) nonlinear processes.

**Where in RippleTwin this would touch.** `twin/placement.py` — same
problem class as Paper 9, via a different mathematical device.

**Genuine scientific fit.** This is the paper the review most needed to
check carefully for topology transfer, and it fails the check: the method
fundamentally requires a differentiable state-transition function
`f(x(k), u(k))` with a well-defined Jacobian `∂h_i/∂x_j`. RippleTwin's
factory is a discrete-event simulator whose transitions are governed by
`max()` and buffer-capacity comparisons (the three recursions in
`docs/METHOD.md §2`) — there is no smooth, differentiable state to
linearize, and "the initial state x(0)" (a continuous vector to be
recovered) has no analog in a problem where the unknown is *which
categorical station* is the constraint. Its validation domain (CSTRs,
wastewater treatment — continuous-flow process control) is categorically
different from discrete-unit assembly-line manufacturing. Forcing a
linearization onto RippleTwin's dynamics to compute this sensitivity matrix
would be a strictly worse approximation of the exact, closed-form
buffer-capacity-distance propagation model already in use.

**What would genuinely change if adopted.** Nothing implementable — there
is no differentiable simulator to differentiate. Its underlying *principle*
(avoid picking sensors that are near-linearly-dependent on ones you already
have) is already present in `placement.py`'s cosine-similarity/
`confusable_with` mechanism, so even the transferable insight offers
nothing beyond what's already implemented.

**Impact on existing validated results.** None (not adopted).

**Verdict: REJECT.** Topology/domain mismatch is fatal and specific: the
method requires a continuous, differentiable nonlinear state-transition
function, which a discrete-event serial line with `max()`-governed
blocked/starved/processing transitions does not have; and the
redundancy-avoidance principle it embodies is already implemented via a
mechanism (cosine similarity over derived propagation vectors) that
doesn't need a differentiable model at all.

---

### 11. arXiv:2608.14917 — Ground-Truth-Aware Stress Testing of a Closed-Loop Digital Twin Under Sensor Drift and Missing Data

**What it actually proposes.** A domain-agnostic *evaluation methodology*,
not a smart-building-specific model: maintain a latent ground-truth state
generator, separate from a corrupted-sensing layer (drift, noise,
missingness); compare an oracle policy (reads latent state directly)
against practical sensor-driven policies under paired Monte Carlo (common
random numbers, so all policies see identical exogenous conditions within
a replicate); report **two separate metrics** — Decision Mismatch Rate (how
often the practical policy's action differs from the oracle's) and
outcome-level gaps (the actual physical/business consequence) — and show
empirically that these diverge: at 8× nominal sensor drift, decision
mismatch rose to 7.0% while the CO2-exceedance outcome gap stayed within
noise. Fully synthetic, no training, general-purpose evaluation pattern by
construction.

**Where in RippleTwin this would touch.** `evaluation/` — a new module
alongside (not replacing) `evaluation/experiments.py` and
`factory/sensor_health.py`.

**Genuine scientific fit.** This is the strongest architectural match in
the whole set, and it is worth being explicit about *why*: RippleTwin
already has every structural precondition the methodology needs, built for
unrelated reasons. It already has a latent ground-truth simulator
(`factory/simulator.py`, full observability). It already has a corrupted-
sensing layer applied *on top of* that ground truth
(`factory/sensor_health.py::apply_sensor_faults` — DROPOUT/INTERMITTENT/
NOISY/STALE — structurally identical in role to the paper's drift/
missingness sensor layer). It already has an oracle-equivalent
(coverage=1.00, or `B2_observed_only_twin` as a weaker oracle). And
critically, it already runs the paired/common-random-numbers design the
paper argues for: `evaluation/experiments.py::_views` docstring states
plainly, "Coverage levels are *views* over that run, so differences
between levels are differences in observability and nothing else" — one
physics run per episode, multiple corrupted views, exactly the paper's
paired evaluation pattern. What RippleTwin's *current* evaluation does not
report is the paper's central distinction: it conflates "did RippleTwin
name the right station" (a decision-level fact) with "did that naming
matter for the recommended action" (an outcome-level fact) into a single
localization-accuracy number.

**What would genuinely change if adopted.** A new metric pair — decision
mismatch vs. outcome gap, computed and reported *separately* — not a new
formula for anything existing computes today.

**Impact on existing validated results.** None. Purely additive: reuses
the existing simulator, existing fault injection, existing shadow/
recommend pipeline, existing episode/seed set, and writes to a new results
table. No existing calibration, threshold, or estimator is touched.

**Verdict: IMPLEMENT.** Full execution plan:
[Plan B](IMPLEMENTATION_RUNBOOK.md#plan-b) — the top recommendation of the
whole review.

**Outcome (built).** Shipped as `evaluation/stress_test.py`, merged to
`main`. Full run against the standard 24 held-out test episodes:
**oracle self-check was exactly 0 mismatch on every episode** (the
mandatory pass gate — see `docs/STRESS_TEST.md`). Reported honestly:
coverage loss dominates decision mismatch (0.35→0.59 going 75%→50%
coverage) far more than the specific dynamic sensor faults tested moved
it. 3 new tests, all passing.

---

### 12. arXiv:1903.03783 — Performance evaluation of a production line operated under an echelon buffer policy

**What it actually proposes.** An analytical decomposition method (nested
two-machine subsystems, each solved as an exact 2-D discrete-time Markov
chain, iterated to convergence via flow-balance relationships) for
evaluating throughput/WIP of a serial line operated under an "echelon
buffer" (EB) policy — where each machine may store its output in **any**
downstream buffer, not just its own immediate one (a shared/flexible
buffer-allocation discipline, connected to CONWIP as a special case) —
compared against the traditional "installation buffer" (IB) policy where
each machine only uses its own dedicated buffer. Machines are modeled as
Bernoulli/geometric-processing-time servers in discrete time slots.

**Where in RippleTwin this would touch.** `factory/topology.py`'s buffer
model / `twin/shadow.py::buffer_distance_matrix` — both currently assume
each station has its own **dedicated** out-buffer, i.e., an IB-policy line.

**Genuine scientific fit.** Two independent mismatches, both fatal to
direct use. First, a **policy** mismatch: RippleTwin's line matches the
paper's own description of the IB policy (a standard synchronized conveyor
line), not the EB policy the paper actually analyzes — the paper itself
notes EB is more common in facilities where "material handling is
performed by humans" (forklifts, trolleys) than in synchronized
conveyor-fed automotive assembly. Second, a **stochastic-model** mismatch:
the paper's machines are Bernoulli/geometric discrete-time servers, a
standard queueing-theory simplification; RippleTwin's simulator computes
the exact three-recursion flow-conservation equations (`start_i`, `end_i`,
`departure_i` from `docs/METHOD.md §2`) directly, in continuous time, with
no such discretization. Even setting the policy mismatch aside, the
paper's contribution — an *approximate* analytical decomposition for
throughput/WIP that avoids running a full simulator — solves a problem
RippleTwin doesn't have: it already computes these quantities *exactly*
via full discrete-event simulation.

**What would genuinely change if adopted.** Nothing beneficial. Adopting an
approximate decomposition method in place of exact simulation output would
be a strict downgrade in fidelity, for a buffer-allocation discipline
RippleTwin's target line doesn't use.

**Impact on existing validated results.** None (not adopted).

**Verdict: REJECT.** Argued precisely: (1) topology/policy mismatch —
echelon (shared, flexible) buffer allocation vs. RippleTwin's dedicated
per-station buffers, and the paper's own author notes EB fits
manually-material-handled facilities better than synchronized conveyor
lines; (2) stochastic-model mismatch — Bernoulli/geometric discrete-time
servers vs. RippleTwin's continuous flow-conservation recursions; (3)
RippleTwin already computes exactly, via full simulation, what this paper
only approximates.

---

## 3. Hybrids

### Hybrid 1 — Papers 3 (Causal DQ) + 9 (structural observability): formal justification for two existing design choices, zero code change

**What changes.** Nothing in code. `docs/METHOD.md §0` and
`docs/JUDGE_QA.md`'s Q0c already argue, in prose, that the flow-model's a
priori causal direction beats a learned/correlational alternative, and that
`placement.py`'s greedy sensor recommendation is the right shape of
algorithm. Citing Paper 3 (an independent, 2025, mathematically-grounded
demonstration that causality-aware placement strictly dominates
correlation-only placement under partial observability, with proven
tighter error bounds) alongside Paper 9 (an independent, mathematically-
grounded proof that greedy is a provably-good approximation for monotone
submodular sensor-placement objectives, exactly the shape of
`placement.py`'s objective) would let `docs/METHOD.md` upgrade both
arguments from "we believe this and tested it empirically" to "two
independent lines of published theory, from different sub-fields, back
this design," without touching a single line of `src/`.

**Where.** `docs/METHOD.md §0` and `docs/REFERENCES.md`.

**Fit against constraints.** Trivially passes all four (no training data,
deterministic, offline, doesn't break tests) — it's pure citation.

**Effort/risk.** Effort: near-zero (a paragraph and two citations). Risk:
none.

**Verdict: CITE ONLY.**

---

### Hybrid 2 — Flow-path + quality-path LLR fusion for ambiguous blind pairs (addresses the coverage-collapse limitation, narrowly)

**What changes.** For episodes where a flow disturbance *and* a quality
alert both fire in overlapping windows (`fault_kind == "COMBINED"`, a
category `evaluation/experiments.py` already tracks), and where
`twin/placement.py::ambiguity()` has flagged the leading candidate's group
as highly confusable (the documented ~97% case for adjacent blind pairs
like S32/S33 and S37/S38), add a new evidence-combination step:
`combined_score(station) = flow_llr(station) + quality_llr(station)`,
restricted to the ambiguous group, using `genealogy.py::quality_state`'s
per-station LLR for the same candidates over the same window range. This is
Bayesian evidence fusion — additive log-likelihoods from two structurally
independent physical channels (timing vs. defect-type propensity) — which
is philosophically the same move `shadow.py` already makes when it folds
`z_proc` into the flow posterior as "a proper likelihood ratio... [that]
cuts both ways."

**Which papers contribute what.** No single paper among the 12 proposes
exactly this fusion. It is *informed by* the general pattern Paper 7
(FMEA-BN) demonstrates — that Bayesian evidence combination across
multiple independently-observed failure signals, over a structural
(expert-derived) network, is a sound, training-data-free way to sharpen a
posterior — applied to RippleTwin's own two existing, currently-unfused
evidence channels. This is the review's own synthesis, not a direct
transplant of either paper's mechanism.

**Where.** New module `twin/evidence_fusion.py`; additive touch to
`twin/pipeline.py::infer()`.

**Fit against constraints.** No training data (pure LLR arithmetic over
existing evidence). Deterministic and auditable (sum of two
already-explainable LLRs). Offline. Does **not** break existing tests if
implemented as an additive column (see execution plan).

**Does it address the two documented weak points?** Directly relevant to
(a) the adjacent-blind-station ambiguity, but honestly **narrow**: it only
fires when a flow disturbance *and* a quality signature co-occur, and
README's own finding #4 notes most hidden-station faults produce *no*
quality signature at all (the flow model detects a pure quality drift in
0% of windows, and the reverse — most slowdowns produce no defects — is
implied by the same asymmetry). So this would measurably help the
`COMBINED`-fault-kind subset specifically, not the general 25%-coverage
collapse. It does **not** meaningfully touch (b), the quality-path
"shortlist not verdict" weakness (that's Hybrid 3, below).

**Effort/risk.** Medium risk (touches the inference pipeline additively;
real window-alignment complexity — flow windows are 20-vehicle/stride-5,
quality pools 200 vehicles per test, a genuinely different granularity).

**Verdict: IMPLEMENT**, with the caveat stated honestly in the write-up:
narrow scope, gated promotion only. Full execution plan:
[Plan C](IMPLEMENTATION_RUNBOOK.md#plan-c).

**Outcome (built, not promoted).** Shipped as `twin/evidence_fusion.py`,
merged to `main`, opt-in and off by default — `top_station` is never
overwritten (verified directly in `twin/pipeline.py`). The gated
before/after comparison (same seeds, same protocol, `fusion_enabled`
toggled) came back clean but inconclusive: fusion fired on real evidence
in both `COMBINED`-fault-kind held-out episodes but never changed the
station pick on any of 19 detected windows — an honestly underpowered
null result on a 2-episode sample, not a win. Documented as a negative
result in `docs/METHOD.md` rather than promoted. 4 new tests, all passing,
including a regression guard proving every existing caller of `infer()`
is byte-identical whether or not the fusion attributes are present.

---

### Hybrid 3 — Papers 7 (FMEA→Bayesian Network) + 6 (interactive expert correction): a richer, still-auditable quality-attribution path

**What changes.** Paper 7's Noisy-OR FMEA→BN construction (deterministic,
no training data, GA-based consistency repair across multiple experts)
would generalize `genealogy.py::candidate_prior` from a flat
station→defect-type propensity table into a multi-hop cause-effect network
capable of representing cascades (e.g., "sealer drift → downstream fixture
misalignment → paint defect") — a genuinely richer causal structure than
the current single-hop Poisson mixture. Paper 6's interactive-correction UI
pattern (a process expert adds/removes edges, closing the loop between
expert and learned model) would extend RippleTwin's existing `hitl/` ledger
and `ai/fmea_map.py`'s "draft, not configuration, reviewer signs off"
discipline to let an engineer directly correct individual BN edges before
they're trusted, keeping the richer structure auditable rather than
becoming an unaccountable black box.

**Which papers contribute what.** Paper 7 → the construction algorithm
(deterministic BN-from-FMEA, no training data). Paper 6 → the interactive
human-correction pattern (abstract-level only, per the access caveat in its
own entry).

**Where.** `twin/genealogy.py::candidate_prior`/`QualityBaseline` (new
construction), `ai/fmea_map.py` (new elicitation surface for inter-defect
trigger probabilities), `hitl/` (new edge-correction UI hook).

**Fit against constraints.** No training data (Paper 7's construction is
elicitation-only). Deterministic, checkable — every trigger probability
traces to an explicit expert rating, same as the existing FMEA-derived
prior. Directly compatible with the "draft, not configuration" philosophy
already in place for `ai/fmea_map.py`.

**Does it address the two documented weak points?** This is the hybrid
that most directly targets (b), the quality-attribution "shortlist not
verdict" limitation — and there's a concrete precedent for optimism:
`docs/METHOD.md §6` documents that switching from soft-assignment to the
current Poisson-mixture formulation moved the true source's rank from
**11th to 1st** of 42 — i.e., structural refinements to this exact scoring
mechanism have already produced large, measured rank improvements once
before. It does not address (a), the flow-side coverage collapse, at all.

**Effort/risk.** Large. Requires new FMEA elicitation work beyond
`ai/fmea_map.py`'s current scope (inter-defect trigger probabilities, not
just station↔defect-type pairs), a from-scratch re-derivation of
`quality_state()`'s scoring math to run over a BN posterior, a new UI
surface in `hitl/`, and full re-validation of
`results/tables/quality_attribution.csv` against the existing protocol.
Not a bolt-on — a real, multi-week project.

**Verdict: FUTURE WORK** — genuinely the most impactful idea for the
quality path's known weakest point, honestly not safe or feasible as a
quick addition. Written into `docs/LIMITATIONS.md`'s roadmap as an
explicit, named next step, citing both papers.

---

## 4. Summary table

| Item | Verdict | One-line reason |
|---|---|---|
| 1. Bottleneck diagnosis metrics (2306.16120) | **IMPLEMENT** | `rbf` mostly formalizes existing `suspicion_from_shadow`; new `rbs`-style shift-severity is small, safe, additive |
| 2. DMBSI (2607.24819) | CITE ONLY | Assumes full instrumentation and reentrant/batch MES flow; strengthens positioning, not code |
| 3. Causal DQ (2507.09742) | CITE ONLY | Solves a discovery problem RippleTwin already solves analytically; trained RL policy contradicts determinism |
| 4. VMAS seq2seq (2205.02827) | REJECT | Requires full instrumentation everywhere + a black-box trained forecaster for a phenomenon already solved deterministically |
| 5. PiGGO (2604.26593) | REJECT | Continuous nonlinear structural dynamics, trained GNODE with known-initial-state requirement — wrong physical regime |
| 6. Interactive CBN+KG RCA (2402.00043) | CITE ONLY (low confidence — full text unavailable) | Pattern RippleTwin already independently implements at smaller scale |
| 7. FMEA-based BN, battery (2006.03610) | FUTURE WORK | No training data, philosophically compatible, but needs new elicitation + full rearchitecture of quality-path scoring |
| 8. MPGE/RootRank (36780862) | INSUFFICIENT EVIDENCE | Full text unavailable from the research index; abstract truncated before mechanism is stated |
| 9. Structural observability (1711.08264) | CITE ONLY | Formally justifies `placement.py`'s existing greedy algorithm; zero code change |
| 10. Sensitivity-based sensor selection (2208.00584) | REJECT | Requires a differentiable continuous state-transition model; RippleTwin's dynamics are discrete-event |
| 11. Ground-truth-aware stress test (2608.14917) | **IMPLEMENT** | Near-perfect architectural fit; strengthens sensor-fault-robustness story with near-zero risk |
| 12. Echelon buffer policy (1903.03783) | REJECT | Wrong buffer-allocation policy and stochastic model; RippleTwin already computes exactly what this approximates |
| Hybrid 1: Papers 3+9 | CITE ONLY | Two independent published theories back two existing design choices; zero code |
| Hybrid 2: Flow+quality LLR fusion | **IMPLEMENT** | Only lead that touches the documented adjacent-blind-station weakness — narrowly, for `COMBINED` fault kind |
| Hybrid 3: Papers 7+6 | FUTURE WORK | Most impactful idea for the quality path's "shortlist not verdict" weakness; genuinely large-scope |

Detailed, step-by-step implementation procedures for the three IMPLEMENT
items live in [docs/IMPLEMENTATION_RUNBOOK.md](IMPLEMENTATION_RUNBOOK.md).
