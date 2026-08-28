# Round 2 Scorecard

Every row maps a requirement from the Round 2 master brief's final
acceptance criteria (§47 A-R) to what implements it and where the evidence
lives. Nothing here is claimed satisfied without a file to point at — most
rows point at a test, an experiment table, or both.

> **Reminder that applies to every row below:** everything is a
> **simulated prototype result on synthetic data**. See
> `docs/LIMITATIONS.md` for the specific scoping decision behind each
> capability, and the original build's `README.md`/`docs/METHOD.md` for
> what was already true before this upgrade.

| # | Requirement | Implementation | Evidence |
|---|---|---|---|
| A | Early prediction — an emerging bottleneck predicted before it becomes active | `twin/predict.py`'s 6-state ladder (NORMAL→DEGRADING→WATCH→PREDICTED_CONSTRAINT→ACTIVE_BOTTLENECK, +RECOVERING), driven by a second calibrated "watch" threshold below the confident-detection threshold | `tests/test_early_warning.py::test_predictor_warns_before_the_constraint_binds_on_s6`; `evaluation/early_warning.py` |
| B | Lead time — the system reports prediction lead time | `evaluation/metrics.py::true_bottleneck_onset` (a tighter, model-internal reference than the pre-existing board-moment proxy) + `evaluate_early_warning` | `results/tables/early_warning_summary.csv` (mean/median/min lead time, miss rate, false-alarm rate — successes and failures both reported) |
| C | Defect prediction — a potential defect predicted before downstream inspection | `twin/defect_risk.py`: per-(vehicle, station) risk from RICH-tier process-channel z-scores, before any gate looks | `tests/test_defect_risk.py`; `evaluation/defect_prediction.py` |
| D | Missing sensors — the system continues operating when sensors are unavailable | Pre-existing MANUAL-tier stations (unchanged) + Phase 4's dynamic DROPOUT/INTERMITTENT | `tests/test_sensor_dynamics.py::test_pipeline_survives_a_dropout_overlapping_a_real_disturbance` |
| E | Sensor failure — the system handles dynamically failing sensors | `factory/sensor_health.py`: DROPOUT/INTERMITTENT/NOISY/STALE, plus `flag_stale_windows` detection from collapsed variance | `tests/test_sensor_dynamics.py` (14 tests, incl. empirically measured confidence degradation and 0-false-positive stale detection) |
| F | Confidence — confidence changes with evidence quality | Mechanically true of the existing NNLS/likelihood math (less/noisier evidence → weaker likelihood → lower `group_prob`); measured directly for Phase 4's dynamic faults | `tests/test_sensor_dynamics.py::test_confidence_degrades_when_the_nearest_evidence_drops_out` (group_prob 0.86→0.36 under DROPOUT, without becoming confidently wrong) |
| G | Abstention — the system refuses to overclaim when evidence is insufficient | Pre-existing `Recommendation.abstained` + `ACTION_ESCALATE` (unchanged), read out onto the brief's exact vocabulary by `recommend/engine.py::taxonomy_label` (ALLOW/WATCH/INVESTIGATE/ESCALATE/ABSTAIN) | `tests/test_sensor_dynamics.py::test_taxonomy_escalate_vs_abstain`, `test_taxonomy_watch_and_investigate` |
| H | Topology — the system works beyond a purely serial line | `factory/topology.py`'s `Edge`/graph methods, dispatched via `LineTopology.is_graph` in `twin/shadow.py`/`twin/propagate.py`; `factory/graph_simulator.py` for demonstration data | `tests/test_topology_graph.py` (13 tests); 100%/100% detection+within-1 accuracy on both Plant B (parallel) and Plant C (rework) in smoke tests, `evaluation/topology_experiment.py`'s `topology_summary.csv` |
| I | Cross-plant — the same engine works on multiple configurations | `configs/plant_b_parallel.yaml`, `configs/plant_c_rework.yaml` run through the identical unmodified `fit_context`/`infer` pipeline as the flagship line — no per-plant code | `evaluation/topology_experiment.py`; `tests/test_scenarios_suite.py::test_scenario_K_different_plant_configuration` |
| J | Robustness — evaluated under distribution shift | `evaluation/distribution_shift.py::perturb_line` (higher process noise, higher micro-stop rate, harsher fault magnitude), context fit on the unperturbed distribution and evaluated with **no re-fitting** | `results/tables/distribution_shift_summary.csv`; `tests/test_cross_plant_robustness.py` |
| K | Human-in-loop — human decisions are captured | Pre-existing `hitl/ledger.py` hash-chained ledger (unchanged), decision vocabulary completed with `DECISION_MODIFIED`/`DECISION_ESCALATED`; wired to two new dashboard buttons | `tests/test_feedback_loop.py::test_decision_vocabulary_has_all_four_types`; dashboard Floor supervisor view |
| L | Feedback — actual outcomes can be fed back into the system | `twin/feedback.py::priors_from_precision` + `apply_feedback`, consuming the pre-existing `hitl/ledger.py::precision_by_station`; a new opt-in `ShadowConfig.station_prior_weight` (default `None`, exact pre-upgrade behavior otherwise) | `evaluation/feedback_experiment.py` — demonstrates a concrete before/after posterior shift (0.891→0.935 in the shipped example) and **honestly reports** whether it moves headline accuracy at the available sample size (often "NO MEASURABLE CHANGE" — not hidden) |
| M | Explainability — predictions expose evidence | Pre-existing `explain/explain.py` (OBSERVED/INFERRED/PREDICTED tags, unchanged) extended with an always-surfaced second-best hypothesis + its probability | `tests/test_feedback_loop.py::test_explain_flow_alert_surfaces_an_alternative_hypothesis` |
| N | Streaming — the dashboard can show a live production simulation | `demo/run_streaming_demo.py`: paced, window-by-window, causally-computed replay, explicitly labeled as a replay (not a live connection) on every line of its own output | `tests/test_scenarios_suite.py::test_streaming_demo_script_runs` |
| O | ROI — business impact can be explored interactively | Pre-existing interactive Leadership-tab calculator extended with explicit production-value/hr, downtime-cost/hr, maintenance cost, ROI %, break-even/payback — closing the gap against the brief's full input/output list | `app/dashboard.py` Leadership view |
| P | Validation — results are measured quantitatively | Every capability above has a dedicated `evaluation/*.py` module writing CSV/JSON tables under `results/tables/`, none hand-typed | `evaluation/run_round2.py` (single entry point regenerating all of them) |
| Q | Reproducibility — experiments can be rerun from a clean environment | `python -m rippletwin.evaluation.run_round2` (all new experiments, one command, stated seeds) alongside the pre-existing `python -m rippletwin.evaluation.experiments` (flagship) | Both commands verified from a clean run during this upgrade; see commit history for the byte-identical-results check performed at every phase |
| R | Documentation — README and this scorecard accurately describe the implementation | This document, `docs/ARCHITECTURE.md`, `docs/LIMITATIONS.md`, `docs/SIGNALS.md`, README's restructured top section | — |

## Additional brief requirements not in the A-R list, tracked separately

| Requirement (brief section) | Implementation | Evidence |
|---|---|---|
| Multi-causal / intermittent bottlenecks (§31 Scenario E) | `factory/scenarios.py::scenario_multiple_abnormalities` — two independent, simultaneous, unrelated disturbances | `tests/test_scenarios_suite.py::test_scenario_E_multiple_simultaneous_abnormalities` |
| Contradictory sensor evidence (§31 Scenario F) | A real fault plus a STALE+NOISY dual fault at neighbouring stations during the same window | `tests/test_scenarios_suite.py::test_scenario_F_contradictory_sensor_evidence` |
| Production surge / high load (§31-32 Scenario H) | `factory/scenarios.py::scenario_production_surge` + `evaluation/surge.py` (latency, throughput, peak memory) | `results/tables/surge_test.json` |
| Sensor placement under budget (§17) | Pre-existing `twin/placement.py::recommend_sensors` (unchanged) + new `twin/whatif.py::whatif_add_sensor` for a single named candidate | `app/dashboard.py` Leadership "What if we added a sensor" expander |
| Calibration — do probabilities mean what they claim (§26) | `evaluation/calibration.py`: Brier score + reliability table for both new probabilistic heads (bottleneck risk, defect risk), with unreliable bins explicitly excluded from the summary statistic | `results/tables/calibration_summary.csv`, `calibration_*_reliability.csv` |
| Multi-modal signal documentation (§18) | `docs/SIGNALS.md` — every existing channel mapped to role, failure mode detected, and consumer; no new sensor types added | `docs/SIGNALS.md` |
| Causality vs. correlation language (§23) | Audit pass over `explain/explain.py`, `app/dashboard.py`, `ai/fmea_map.py`; two instances of "caused" softened to "most likely responsible for" in the statistical (non-physics-certain) attribution path | Commit message for Phases 8-9 |
| Scenario suite must not crash (§31, all twelve letters A-L) | `factory/scenarios.py::SCENARIO_SUITE_MAP` + composition of existing pieces | `tests/test_scenarios_suite.py` — all twelve run the real pipeline end to end |

## What is explicitly *not* claimed

- **Not** validated on real production data — every number is synthetic,
  stated on every experiment module's own docstring and in every generated
  table's manifest.
- **Not** a claim that the feedback loop improves accuracy — it is
  reported as measured, and at the sample sizes used here that is often "no
  measurable change."
- **Not** a claim that non-serial topology support is production-grade —
  `factory/graph_simulator.py` is a small, separate demonstrator with
  stated scoping limits (`docs/LIMITATIONS.md` items 6-8), not a
  generalization of the flagship `LineSimulator`.
- **Not** production-ready. TRL remains what the pre-upgrade `README.md`
  states it was: validated in a laboratory (simulated) environment, ready
  for a Phase 0 pilot assessment, not for live deployment.
