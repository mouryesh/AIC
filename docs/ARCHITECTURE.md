# Architecture

This document reflects the pipeline after the Round 2 predictive/robustness
upgrade (Phases 2-9 on top of the original shadow-sensing build). Every box
has an implementation; nothing here is aspirational. File paths are given so
each box can be checked directly.

```
                    telemetry from INSTRUMENTED stations only
                    inspection gate results · build sequence · ambient
                    (+ optional dynamic sensor faults: DROPOUT / INTERMITTENT
                     / NOISY / STALE -- factory/sensor_health.py)
                                      |
                    +-----------------v------------------+
                    |  vehicle-indexed windowing         |  align in genealogy
                    |  (20 vehicles, stride 5)            |  space, not wall clock
                    |  features/windows.py                |
                    +-----------------+------------------+
                                      |
                    +-----------------v------------------+
                    |  nominal baseline, frozen           |  mix- and shift-aware
                    |  deviations in FRACTION OF TAKT      |  expectations
                    |  features/baseline.py                |
                    +--------+---------------+-----------+
                             |               |
              flow path      |               |    quality path
        +--------------------v--+       +----v----------------------+
        | blocked / starved      |      | defect-type histogram vs  |
        | asymmetry vs candidate |      | per-station failure modes |
        | propagation profiles   |      | (Poisson mixture LLR)     |
        | twin/shadow.py          |      | twin/genealogy.py          |
        | (serial fast path, or   |      +----------+----------------+
        |  graph-generalized --   |                 |
        |  twin/topology.py's     |                 |
        |  is_graph dispatch)     |                 |
        +-----------+------------+                 |
                    |                              |
                    +--------------+---------------+
                                   |
                    +--------------v-------------------+
                    |  posterior over ALL stations      |  <-- includes stations
                    |  + NULL + LINE_SUPPLY             |      with no sensor;
                    |  (+ optional per-station prior     |      optionally weighted
                    |   from twin/feedback.py)           |      by validated history
                    +--------------+-------------------+
                                   |
              +--------------------+--------------------+
              |                    |                     |
   +----------v---------+ +--------v----------+ +--------v----------------+
   | EARLY WARNING       | | DEFECT RISK        | | ripple forecast        |
   | NORMAL/DEGRADING/   | | per (vehicle,      | | (flow physics,         |
   | WATCH/PREDICTED_    | | station), before   | | graph-generalized)     |
   | CONSTRAINT/ACTIVE_  | | any gate looks      | | twin/propagate.py       |
   | BOTTLENECK/         | | twin/defect_risk.py |                          |
   | RECOVERING           | |                     |                          |
   | twin/predict.py       | |                     |                          |
   +----------+---------+ +--------+----------+ +--------+----------------+
              |                    |                     |
              +--------------------+---------------------+
                                   |
                    +--------------v-------------------+
                    |  explanation (OBS/INF/PRED,       |  what-if counterfactuals
                    |  alternative hypothesis)          |  (twin/whatif.py) --
                    |  explain/explain.py                |  simulation-based
                    |                                    |  projections only
                    |  recommendation / ALLOW-WATCH-     |
                    |  INVESTIGATE-ESCALATE-ABSTAIN      |
                    |  recommend/engine.py                |
                    +--------------+-------------------+
                                   |
                    +--------------v-------------------+
                    |  HUMAN approves / rejects /       |
                    |  modifies / escalates             |
                    |  outcome -> hash-chained ledger    |  -> per-station precision
                    |  hitl/ledger.py                     |     -> twin/feedback.py
                    |                                    |        priors (closes the
                    |                                    |        loop, above)
                    +----------------------------------+
```

## What generalizes, and how

The core design constraint carried through every phase of the upgrade:
**every generalization reduces to the pre-upgrade computation exactly when
the generalizing feature is not used**, verified by a full flagship
evaluation re-run at every phase (byte-identical results on every
overlapping episode).

| Generalization | Dispatch point | Falls back to (unchanged) |
|---|---|---|
| Non-serial topology (parallel/merge/rework) | `LineTopology.is_graph` in `twin/shadow.py`, `twin/propagate.py` | The original `i<k`/`i>k` index comparison and cumulative-buffer prefix sum, byte-for-byte |
| Dynamic sensor faults | `factory/sensor_health.py`, applied as telemetry post-processing before the pipeline ever sees it | The pipeline is unaware faults exist; a station absent from a window is already handled as `NaN` |
| Per-station feedback prior | `ShadowConfig.station_prior_weight` (default `None`) in `twin/shadow.py::_posterior` | The original uniform `(1 - null_prior) / (n + 1)` prior |
| Critical-vs-random coverage demotion | `strategy=` on `factory/topology.py::apply_coverage` (default `"random"`) | The original random `rng.choice` demotion |

## New modules by pipeline stage

| Stage | Module | Role |
|---|---|---|
| Sensor dynamics | `factory/sensor_health.py` | DROPOUT/INTERMITTENT/NOISY/STALE telemetry faults; stale-window detection |
| Topology | `factory/graph_simulator.py`, `factory/topology.py`'s `Edge`/graph methods | Non-serial demonstration topologies + the reachability the inference engine needs |
| Early prediction | `twin/predict.py` | The 6-state risk ladder, derived from `ShadowSensor`'s own trajectory |
| Defect prediction | `twin/defect_risk.py` | Per-vehicle, pre-inspection defect risk from RICH-tier process channels |
| Feedback | `twin/feedback.py` | Ledger outcomes → bounded per-station prior |
| What-if | `twin/whatif.py` | Counterfactual re-runs of the existing forecast/placement math |
| Evaluation | `evaluation/early_warning.py`, `defect_prediction.py`, `coverage_matrix.py`, `distribution_shift.py`, `calibration.py`, `topology_experiment.py`, `feedback_experiment.py`, `surge.py`, `run_round2.py` | One dedicated experiment module per new capability, plus one entry point that regenerates all of them |
| Demo | `demo/run_streaming_demo.py` | Paced replay of a deterministic run, explicitly labeled as such |

## What did not change

`twin/genealogy.py` (defect attribution), `hitl/ledger.py`'s core
hash-chain mechanics, the four baselines (`models/baselines.py`), the
evaluation harness backing the flagship README numbers
(`evaluation/experiments.py`), and the physics itself
(`factory/simulator.py::LineSimulator`) are all untouched. New capability
was built on top of them or alongside them, not through them — see
`docs/LIMITATIONS.md` for the specific scoping decisions that kept it that
way.
