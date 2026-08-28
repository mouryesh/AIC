# Limitations

We would rather state these than have them found. This document collects
every scoping decision and known gap from both Round 1/2's original build
and the predictive/robustness upgrade on top of it — the README's
"Limitations" section is the short version; this is the full one, with the
reasoning behind each item.

## Carried over from the original build

1. **Everything is synthetic.** The physics is faithful and the
   disturbances are plausible, but nothing in this repository is validated
   against a real production line. That is the first thing a pilot must
   establish (see `docs/DEPLOYMENT.md`'s Phase 0 assessment).
2. **Adjacent blind stations are not separable.** Two hidden stations side
   by side cannot be told apart by flow evidence alone. The system says so
   and abstains (`twin/placement.py::ambiguity`) rather than guessing.
3. **The constraint must bind.** A station that slows but stays inside takt
   creates no starvation, so there is nothing to localise yet — reported as
   `WATCH`/`DEGRADING` (Phase 2), not dressed up as a confident detection.
4. **Quality attribution is a shortlist**, not a verdict, and reacts more
   slowly than a flow alert because defects are rare (see
   `twin/genealogy.py`'s `pool_vehicles` discussion).
5. **The ROI model rests on contribution margin.** On a line that is not
   capacity-constrained, the throughput driver largely collapses.

## New in the predictive/robustness upgrade

6. **Topology generalization has real scoping limits.**
   `factory/graph_simulator.py`'s `GraphLineSimulator` processes vehicles
   **vehicle-major, in release order**, treating global release order as
   every edge's FIFO discipline — including across merges. At a real merge
   point fed by branches of meaningfully different latency, a vehicle
   released later can genuinely arrive first; this simulator does not
   reproduce that reordering. Adequate for the two short, similar-latency
   demonstration topologies (`configs/plant_b_parallel.yaml`,
   `configs/plant_c_rework.yaml`); not claimed exact for branches of very
   different length. Windowing (`features/windows.py`) is also still
   vehicle-id-ordered, not per-branch arrival-rank-ordered, so precision
   degrades somewhat near a merge point — a known, documented approximation
   rather than a silent one.
7. **Rework is modeled as an acyclic spur-and-remerge**, not a true upstream
   cycle. `LineTopology.topological_order()` raises on a genuine cycle by
   design (see its docstring) — a real rework loop that sends a vehicle
   backward through earlier stations is out of scope. The spur model
   (detour through an extra station, then rejoin ahead of the next one) is
   the common real-world shape for most automotive rework anyway, but it is
   a modeling choice, not an oversight.
8. **The graph propagation forecast reports only the dominant branch.**
   `twin/propagate.py::forecast_ripple` walks the first (lowest-index)
   successor/predecessor at a split/merge rather than enumerating every
   branch. Exhaustive multi-branch forecast text was judged not worth the
   complexity for a demonstration-scale topology.
9. **Sensor coverage below ~12% is not reachable on line_42** — 4 inspection
   gates plus station 0 are always instrumented by design (a plant that
   cannot read its own end-of-line test has no coverage problem worth
   modeling), which alone is 5/42 ≈ 11.9% of the line. The brief's "10%"
   level is reported as this actual floor instead of a rounding choice; see
   `evaluation/coverage_matrix.py`.
10. **`DEGRADING` is a deliberately loose bottom rung and flickers on
    ordinary noise.** It exists to give the earliest possible warning at a
    stated, calibrated false-alarm cost (a second threshold,
    `ShadowConfig.watch_llr`, read off the same null distribution as
    `detect_llr` at a looser target rate) — but that looseness means it
    fires and clears constantly under normal variation. Treat `WATCH` and
    above as the actionable early-warning threshold; `DEGRADING` is
    informational. The streaming demo (`demo/run_streaming_demo.py`)
    filters its event-timeline summary to `WATCH`-or-above for exactly this
    reason, while still printing every window's true state, unfiltered.
11. **Predictive defect risk has a real, structural coverage gap.** Unlike
    the flow path — which infers a hidden station's state from conservation
    of material at its *neighbours* — there is no equivalent physical
    channel for defect risk. A MANUAL station emits no telemetry for any
    vehicle at all, so there is nothing to score there, for anyone, ever
    (`twin/defect_risk.py::coverage_gap_report`). Post-hoc attribution via
    `twin/genealogy.py` still works once a defect is found downstream; a
    mid-flight prediction at the station itself does not.
12. **Defect-prediction precision is genuinely weak at any operating point
    that also catches a useful fraction of real defects** — defects are
    rare (on the order of 1e-3 per vehicle-visit at nominal), so the class
    imbalance is severe. See `evaluation/defect_prediction.py` and its
    results table rather than a summary claim.
13. **The distribution-shift experiment covers process noise, micro-stop
    rate, and fault magnitude — not takt or topology.**
    `evaluation/distribution_shift.py::perturb_line` deliberately keeps
    tiers/buffers/takt fixed (a fitted `TwinContext` depends on them
    structurally); a takt shift without re-baselining is a different,
    harder experiment not attempted here.
14. **The feedback loop's effect on headline accuracy is not established at
    the sample sizes available.** `evaluation/feedback_experiment.py`
    reports the mechanism working (a validated-confirmed station's
    posterior measurably rises) but explicitly does not claim the resulting
    prior moves aggregate localisation accuracy — see that module's
    `honest_verdict` field, which can (and does, at small n) read "NO
    MEASURABLE CHANGE".
15. **Calibration numbers are illustrative, not statistically powered.**
    `evaluation/calibration.py`'s reliability tables mark bins below a
    minimum sample count as `reliable=False` and exclude them from the
    expected-calibration-error figure — read `n` per bin before trusting a
    gap.
16. **The streaming demo is a paced replay, not a live connection.** Every
    value is computed causally from a fixed seed before pacing begins;
    `time.sleep` between windows is cosmetic. Labeled as such on every
    screen of its own output (`demo/run_streaming_demo.py`).
17. **Surge/performance numbers are machine-relative, not an SLA.**
    `evaluation/surge.py` measures latency and peak memory on whatever
    machine ran it, for one run; useful for spotting a latency cliff, not
    for capacity planning.

## What did **not** change

The physics-informed hidden-state estimator (non-negative least squares
over calibrated propagation matrices), the probabilistic localisation
posterior, defect genealogy, abstention, the hash-chained audit trail, the
four baseline comparisons, and the synthetic-data honesty conventions are
all unchanged from the pre-upgrade build — extended (topology, sensor
dynamics, feedback) rather than replaced, and verified byte-identical on
every overlapping episode against the pre-upgrade evaluation output at
every phase (see commit messages for the specific verification each time).
