# Implementation runbook — Plans A, B, C

This is a complete, step-by-step implementation procedure for the three
IMPLEMENT items from [docs/RESEARCH_EVALUATION.md](RESEARCH_EVALUATION.md).
It is written for an implementing agent working with no other context than
this file and the repository itself.

## Ground rules (read first, apply to all three plans)

1. **Additive only, until a decision gate says otherwise.** Every new
   capability lands as a new file, a new column, or a new results
   directory. Nothing in `twin/shadow.py`'s calibration (`tau`,
   `detect_llr`, `watch_llr`), nothing in `models/baselines.py`'s
   `apply_detection_rule`/`calibrate_threshold`, and no existing column on
   an existing DataFrame is renamed, removed, or silently redefined.
2. **The existing test suite is the tripwire.** Run `pytest -q` after
   *every* plan (not just at the end). If the count drops below 204 passing
   or any existing test starts failing, stop and fix before writing more
   code — do not proceed to the next plan with a broken tree.
3. **Verify before you build.** Several steps below say "confirm X by
   reading Y first." Do not assume the internals described in this runbook
   are exactly right down to the column name — they were read from the
   source during the review, but confirm the exact signature/columns in
   the actual file before writing code against them, and adjust this
   runbook's snippets accordingly if reality differs slightly.
4. **Never touch `results/tables/` in place.** New evaluation output goes
   into a new directory (`results_stress_test/`, `results_fusion_eval/`)
   with `manifest.json` or an equivalent header stating what produced it,
   exactly like the existing tables do (`"result_type": "SIMULATED
   PROTOTYPE RESULT on synthetic data"`).
5. **Commit per plan.** Three commits (or more, if a plan needs fixups),
   not one giant commit — it keeps the history reviewable and lets any one
   plan be reverted independently if its decision gate fails.
6. Work from the repository root: `/home/mouryesh/mouryesh/AIC/RippleTwin`.
   Python invocations that import `rippletwin` need `PYTHONPATH=src`
   (see README's own command examples).

---

## Plan A

**Goal:** expose a per-station "bottleneck frequency" report and a
per-window "shift severity" diagnostic, computed entirely from output
`ShadowSensor` already produces. Zero change to the estimator.

### A0 — Verify assumptions

1. Open `src/rippletwin/twin/shadow.py` and find `ShadowSensor.run(...)`.
   Confirm the exact columns of the DataFrame it returns. The review
   confirmed `window`, `top_station`, `llr`, `detected`, `t_mid_s`,
   `v_start`, `v_end` exist (the last three via `evaluation/experiments.py`'s
   comment "`shadow` already carries `t_mid_s` / `v_start` / `v_end`").
   Confirm whether a runner-up candidate and its posterior mass
   (`group_prob` or similar) are already present per window.
   - **If a runner-up station + its mass already exist as columns:** skip
     straight to A2, use those columns.
   - **If not:** you need one small additive change to `ShadowSensor.run()`
     (or wherever the per-window posterior is finalized) to also emit
     `runner_up_station: int` and `runner_up_prob: float` (the second-
     highest posterior mass among non-NULL, non-LINE_SUPPLY station
     hypotheses, and its station index). Locate the posterior computation
     (search for where `group_prob` or the final per-window row dict is
     assembled) and add these two fields there — additive columns only,
     nothing existing renamed.
2. Open `src/rippletwin/twin/placement.py::suspicion_from_shadow` and
   confirm its exact logic (counts `top_station` values where
   `detected == True`, normalizes with a `0.25 + 0.75 * fraction` floor).
   Plan A's `bottleneck_frequency` should mirror this counting logic but
   expose it as a first-class, standalone report (not folded into a
   0.25-floor placement weight) — i.e., a plain fraction of leading
   windows per station, summing to 1 across stations, matching the
   paper's actual `rbf` definition.

### A1 — New module

Create `src/rippletwin/evaluation/bottleneck_diagnosis.py`:

```python
"""Bottleneck frequency and shift-severity diagnostics.

Two deterministic, post-hoc statistics computed over ShadowSensor's own
output — no new estimator, no training data. Following West, Schwenken &
Deuse (2023), "Data-driven approach for diagnostic analysis of dynamic
bottlenecks in serial manufacturing systems" (arXiv:2306.16120):

- ``bottleneck_frequency`` (their rbf): the fraction of windows in which
  each station was the leading (detected) suspect. RippleTwin already
  computes something functionally equivalent internally, in
  ``twin.placement.suspicion_from_shadow``; this exposes the plain,
  un-weighted fraction as a first-class report, matching the paper's own
  definition (sums to 1 across stations).
- ``bottleneck_shift_severity`` (their rbs): a per-window ratio of the
  runner-up candidate's posterior mass to the leader's. The paper's own
  metric uses each station's measured "active period"; RippleTwin has no
  such per-non-leader measurement, so the substitution here is posterior
  mass, which serves the same purpose (an early signal that a second
  candidate is about to overtake the current leader) using evidence
  RippleTwin already computes.
"""

from __future__ import annotations

import pandas as pd

from ..factory.topology import LineTopology


def bottleneck_frequency(shadow: pd.DataFrame, line: LineTopology) -> pd.DataFrame:
    """Fraction of detected windows each station led, over the given shadow frame.

    ``shadow`` is the frame produced by ``ShadowSensor.run`` (or
    ``twin.pipeline.infer``'s second return value) for one episode or shift.
    Returns one row per station with columns ``station``, ``station_id``,
    ``zone``, ``rbf``, ``n_windows_leading``. ``rbf`` sums to 1 across
    stations when at least one window was detected; if none were, every
    row is 0.
    """
    if shadow.empty or "detected" not in shadow.columns:
        return pd.DataFrame(
            columns=["station", "station_id", "zone", "rbf", "n_windows_leading"]
        )
    leading = shadow[shadow["detected"]]
    counts = leading["top_station"].value_counts()
    total = float(counts.sum()) or 1.0
    rows = []
    for s in line.stations:
        n = int(counts.get(s.index, 0))
        rows.append(
            {
                "station": s.index,
                "station_id": s.station_id,
                "zone": s.zone,
                "rbf": n / total,
                "n_windows_leading": n,
            }
        )
    return pd.DataFrame(rows)


def bottleneck_shift_severity(shadow: pd.DataFrame) -> pd.DataFrame:
    """Per-window ratio of the runner-up's posterior mass to the leader's.

    Requires ``shadow`` to carry ``runner_up_station`` and
    ``runner_up_prob`` alongside ``top_station`` and its own posterior
    mass column (confirm the exact leader-mass column name in
    ``ShadowSensor.run`` during step A0 — this sketch assumes
    ``group_prob`` is the leader's mass; adjust if the real column differs).
    Returns ``window``, ``leader_station``, ``runner_up_station``,
    ``severity_ratio`` — 1.0 exactly at the leader's own row by
    construction, mirroring the paper's ``rbs_BN == 1`` convention.
    """
    required = {"window", "top_station", "runner_up_station", "runner_up_prob", "group_prob"}
    missing = required - set(shadow.columns)
    if missing:
        raise KeyError(
            f"bottleneck_shift_severity: shadow frame is missing columns {missing}. "
            "See runbook step A0 — ShadowSensor.run() may need the additive "
            "runner_up_station/runner_up_prob columns added first."
        )
    out = shadow[["window", "top_station", "runner_up_station", "runner_up_prob", "group_prob"]].copy()
    out = out.rename(columns={"top_station": "leader_station"})
    leader_mass = out["group_prob"].replace(0.0, pd.NA)
    out["severity_ratio"] = (out["runner_up_prob"] / leader_mass).fillna(0.0).clip(0.0, 1.0)
    return out[["window", "leader_station", "runner_up_station", "severity_ratio"]]
```

### A2 — Tests

Create `tests/test_bottleneck_diagnosis.py`:

```python
"""Tests for evaluation.bottleneck_diagnosis (Plan A, RESEARCH_EVALUATION.md #1)."""

import pandas as pd
import pytest

from rippletwin.evaluation.bottleneck_diagnosis import (
    bottleneck_frequency,
    bottleneck_shift_severity,
)
from rippletwin.factory.topology import build_line


LINE = build_line("configs/line_42.yaml", seed=7)


def test_bottleneck_frequency_sums_to_one_when_detections_exist():
    shadow = pd.DataFrame(
        {
            "window": [0, 1, 2, 3],
            "top_station": [2, 2, 5, 2],
            "detected": [True, True, True, False],
        }
    )
    rbf = bottleneck_frequency(shadow, LINE)
    assert rbf["rbf"].sum() == pytest.approx(1.0)
    assert rbf.loc[rbf["station"] == 2, "rbf"].iloc[0] == pytest.approx(2 / 3)
    assert rbf.loc[rbf["station"] == 5, "rbf"].iloc[0] == pytest.approx(1 / 3)


def test_bottleneck_frequency_empty_shadow_returns_empty_frame():
    empty = pd.DataFrame(columns=["window", "top_station", "detected"])
    rbf = bottleneck_frequency(empty, LINE)
    assert rbf.empty or rbf["rbf"].sum() == 0


def test_shift_severity_is_one_at_leaders_own_row_by_construction():
    shadow = pd.DataFrame(
        {
            "window": [0],
            "top_station": [3],
            "runner_up_station": [4],
            "runner_up_prob": [0.6],
            "group_prob": [0.6],
        }
    )
    sev = bottleneck_shift_severity(shadow)
    assert sev["severity_ratio"].iloc[0] == pytest.approx(1.0)


def test_shift_severity_rises_before_a_bottleneck_shift():
    # Synthetic two-station alternating-bottleneck sequence: station 3 leads
    # early, station 4's runner-up mass climbs, then station 4 takes over.
    shadow = pd.DataFrame(
        {
            "window": [0, 1, 2, 3],
            "top_station": [3, 3, 3, 4],
            "runner_up_station": [4, 4, 4, 3],
            "runner_up_prob": [0.10, 0.30, 0.55, 0.65],
            "group_prob": [0.80, 0.70, 0.60, 0.65],
        }
    )
    sev = bottleneck_shift_severity(shadow)
    ratios = sev["severity_ratio"].tolist()
    assert ratios[0] < ratios[1] < ratios[2]  # rising before the shift
    assert ratios[3] == pytest.approx(1.0)     # new leader's own row


def test_shift_severity_raises_on_missing_columns():
    incomplete = pd.DataFrame({"window": [0], "top_station": [1]})
    with pytest.raises(KeyError):
        bottleneck_shift_severity(incomplete)
```

### A3 — Validate

```bash
cd /home/mouryesh/mouryesh/AIC/RippleTwin
PYTHONPATH=src pytest tests/test_bottleneck_diagnosis.py -v
PYTHONPATH=src pytest -q   # full suite — must stay at 204 passing, 0 failing
```

Then a manual sanity check on the flagship demo (not a new test — a human-
readable confirmation):

```bash
PYTHONPATH=src python -c "
import sys; sys.path.insert(0, 'src')
from rippletwin.evaluation.bottleneck_diagnosis import bottleneck_frequency
from rippletwin.factory.topology import build_line
from rippletwin.factory import scenarios as SC
from rippletwin.twin.pipeline import build_windows, fit_context, infer, simulate
from rippletwin.evaluation.views import full_observability, telemetry_view
from rippletwin.features.windows import WindowSpec

line = build_line('configs/line_42.yaml', seed=7)
sim_line = full_observability(line)
spec = WindowSpec.for_line(line, width=20, stride=5)
nom = simulate(sim_line, SC.nominal_run(2600), seed=1)
cal = simulate(sim_line, SC.nominal_run(2200), seed=2)
nom_v = telemetry_view(nom, line, sim_line)
cal_v = telemetry_view(cal, line, sim_line)
ctx = fit_context(line, nom_v, calibration_run=cal_v, spec=spec, target_window_fpr=0.01)
scen = SC.get_scenario('S1_HIDDEN_BOTTLENECK')(line, seed=20260301)
res = simulate(sim_line, scen, seed=20260301 + 77)
res_v = telemetry_view(res, line, sim_line)
scored, shadow, sensor = infer(ctx, res_v)
print(bottleneck_frequency(shadow, line).sort_values('rbf', ascending=False).head(5))
"
```

Confirm the top row's `station_id` matches S1's known injected station
(check `factory/scenarios.py::S1_HIDDEN_BOTTLENECK` or the demo's own
printed ground truth for the exact station name). If the scenario-lookup
call above doesn't match the real API, adapt it to however
`demo/run_demo.py --scenario S1` actually builds its scenario — that file
is the reference implementation for "run one scenario end to end."

### A4 — Commit

```bash
git add src/rippletwin/evaluation/bottleneck_diagnosis.py tests/test_bottleneck_diagnosis.py
# If shadow.py needed the additive runner_up columns:
git add src/rippletwin/twin/shadow.py
git commit -m "$(cat <<'EOF'
Add bottleneck frequency/shift-severity diagnostics (Plan A)

Post-hoc, additive reporting over ShadowSensor's existing output.
Formalizes what twin.placement.suspicion_from_shadow already computes
internally (rbf) and adds a new momentary shift-severity signal (rbs),
following West, Schwenken & Deuse (2023), arXiv:2306.16120. No estimator,
calibration, or existing table is touched — see docs/RESEARCH_EVALUATION.md
item 1 and docs/IMPLEMENTATION_RUNBOOK.md Plan A.
EOF
)"
```

---

## Plan B

**Goal:** a decision-vs-outcome stress-test harness, following the
paired-Monte-Carlo, oracle-vs-practical-policy methodology of
arXiv:2608.14917. Fully additive: new module, new results directory,
reuses the existing simulator, `factory/sensor_health.py` fault injection,
and `evaluation/experiments.py`'s existing episode/seed/coverage machinery
read-only.

### B0 — Verify assumptions

1. Re-read `src/rippletwin/evaluation/experiments.py::run_experiment` in
   full (already summarized in the review). Confirm the exact call
   signatures you'll reuse: `build_line`, `apply_coverage`,
   `full_observability`, `WindowSpec.for_line`, `simulate`,
   `telemetry_view`, `fit_context`, `build_windows`, `infer`. Confirm
   `infer(ctx, res_v)` returns exactly `(scored, shadow, sensor)`.
2. Read `src/rippletwin/factory/sensor_health.py::SensorFault` and
   `apply_sensor_faults` in full (already summarized in the review).
   Confirm the exact point in the pipeline where sensor faults get applied
   relative to `telemetry_view` — i.e., do faults apply to
   `res_v.telemetry` *before* `build_windows`/`ctx.baseline.score`, or does
   the harness need to call `apply_sensor_faults` on the already-projected
   telemetry and then re-run `build_windows` on the corrupted frame? Trace
   one existing caller (search the test suite, e.g.
   `tests/test_sensor_dynamics.py`, for a real end-to-end example of
   `apply_sensor_faults` feeding into `infer`) and copy that wiring
   exactly — do not guess it.
3. Read `src/rippletwin/recommend/dispatch.py` briefly to see what a
   "recommended action" actually looks like (work order, escalate,
   monitor, abstain). Decide whether Decision Mismatch will compare full
   dispatch payloads or the simpler `(top_station, detected)` pair as an
   action proxy. **Use the simpler proxy for the first version** —
   `(top_station, detected)` disagreement is the core of what the paper
   measures (did the *decision* change), and it avoids depending on
   `dispatch.py`'s more complex payload shape. Note this scoping choice in
   the module docstring.

### B1 — New module

Create `src/rippletwin/evaluation/stress_test.py`:

```python
"""Decision-vs-outcome stress testing, oracle vs. sensor-corrupted views.

Following Saad Saoud (2026), "Ground-Truth-Aware Stress Testing of a
Closed-Loop Digital Twin Under Sensor Drift and Missing Data"
(arXiv:2608.14917): separate the question "did sensor corruption change
the decision" from "did it change the consequence." RippleTwin's existing
architecture already supplies every precondition this needs —
evaluation.experiments.run_experiment's own docstring states coverage
levels are VIEWS over one physics run, which is exactly the paired /
common-random-numbers design the paper argues for. This module adds
nothing to the estimator; it is a read-only consumer of the existing
simulator, twin.pipeline.infer, and factory.sensor_health's fault
injection.

Decision proxy: (top_station, detected) — see runbook step B0.3 for why
the simpler proxy was chosen over recommend.dispatch's full payload for
this first version.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..factory import scenarios as SC
from ..factory.sensor_health import SensorFault, apply_sensor_faults
from ..factory.topology import LineTopology, apply_coverage, build_line
from ..features.windows import WindowSpec
from ..twin.pipeline import build_windows, fit_context, infer, simulate
from .experiments import ExperimentConfig
from .views import full_observability, telemetry_view


@dataclass
class StressCondition:
    """One point in the stress grid: a coverage level plus an optional
    sensor-fault severity applied on top of it."""

    coverage: float
    fault_kind: str | None = None       # None, or one of sensor_health's DROPOUT/INTERMITTENT/NOISY/STALE
    fault_fraction_of_run: float = 0.0  # what fraction of the episode's duration the fault covers
    label: str = ""


def _oracle_condition(coverage: float) -> StressCondition:
    return StressCondition(coverage=coverage, fault_kind=None, label=f"oracle_cov{coverage:.2f}")


def decision_mismatch(oracle_shadow: pd.DataFrame, stress_shadow: pd.DataFrame) -> pd.DataFrame:
    """Per-window comparison of (top_station, detected) between two paired runs.

    Both frames must come from the SAME episode/seed (paired by construction
    — see run_stress_test). Returns one row per window with a boolean
    ``decision_mismatch`` column.
    """
    o = oracle_shadow[["window", "top_station", "detected"]].rename(
        columns={"top_station": "oracle_station", "detected": "oracle_detected"}
    )
    s = stress_shadow[["window", "top_station", "detected"]].rename(
        columns={"top_station": "stress_station", "detected": "stress_detected"}
    )
    merged = o.merge(s, on="window", how="inner")
    merged["decision_mismatch"] = (
        (merged["oracle_station"] != merged["stress_station"])
        | (merged["oracle_detected"] != merged["stress_detected"])
    )
    return merged


def run_stress_test(
    cfg: ExperimentConfig | None = None,
    stress_grid: Sequence[StressCondition] | None = None,
    out_dir: str | Path = "results_stress_test",
    verbose: bool = True,
) -> dict:
    """Run the oracle-vs-stress paired comparison over the existing episode set.

    Reuses cfg's episode/seed generation from ExperimentConfig so results
    are directly cross-referenceable against the flagship coverage-sweep
    tables. Writes to out_dir, never to the existing results/ directory.
    """
    cfg = cfg or ExperimentConfig()
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    configured = build_line(cfg.line_config, seed=cfg.line_seed)
    sim_line = full_observability(configured)
    spec = WindowSpec.for_line(configured, width=cfg.window_width, stride=cfg.window_stride)

    oracle_line = full_observability(configured)

    stress_grid = stress_grid or [
        StressCondition(coverage=0.75, fault_kind=None, label="cov75_clean"),
        StressCondition(coverage=0.50, fault_kind=None, label="cov50_clean"),
        StressCondition(coverage=0.75, fault_kind="DROPOUT", fault_fraction_of_run=0.30, label="cov75_dropout30pct"),
        StressCondition(coverage=0.50, fault_kind="NOISY", fault_fraction_of_run=0.30, label="cov50_noisy30pct"),
        StressCondition(coverage=0.75, fault_kind="STALE", fault_fraction_of_run=0.30, label="cov75_stale30pct"),
    ]

    nominal_full = simulate(sim_line, SC.nominal_run(cfg.nominal_vehicles), seed=1)
    calib_full = simulate(sim_line, SC.nominal_run(cfg.calibration_vehicles), seed=2)

    oracle_nom_v = telemetry_view(nominal_full, oracle_line, sim_line)
    oracle_cal_v = telemetry_view(calib_full, oracle_line, sim_line)
    oracle_ctx = fit_context(
        oracle_line, oracle_nom_v, calibration_run=oracle_cal_v, spec=spec,
        target_window_fpr=cfg.target_window_fpr,
    )

    coverage_ctx: Dict[float, object] = {}
    coverage_views: Dict[float, LineTopology] = {}
    for cond in stress_grid:
        if cond.coverage not in coverage_views:
            v = apply_coverage(configured, cond.coverage, seed=int(round(cond.coverage * 1000)))
            coverage_views[cond.coverage] = v
            nom_v = telemetry_view(nominal_full, v, sim_line)
            cal_v = telemetry_view(calib_full, v, sim_line)
            coverage_ctx[cond.coverage] = fit_context(
                v, nom_v, calibration_run=cal_v, spec=spec,
                target_window_fpr=cfg.target_window_fpr,
            )

    def episode_seeds(split: str) -> List[int]:
        if split == "tune":
            return [cfg.tune_seed_base + i for i in range(cfg.n_tune_episodes)]
        return [cfg.test_seed_base + i for i in range(cfg.n_test_episodes)]

    rows: List[dict] = []
    self_check_rows: List[dict] = []

    for split in ("test",):  # stress testing runs on the held-out split only
        for seed in episode_seeds(split):
            scen = SC.random_episode(configured, seed=seed, n_vehicles=cfg.episode_vehicles)
            res_full = simulate(sim_line, scen, seed=seed + 77)

            # --- oracle pass: full observability, no sensor faults -------
            oracle_res_v = telemetry_view(res_full, oracle_line, sim_line)
            oracle_scored = oracle_ctx.baseline.score(build_windows(oracle_res_v, oracle_line, spec), oracle_line)
            _, oracle_shadow, _ = infer(oracle_ctx, oracle_res_v)

            # Mandatory self-check: oracle vs itself must be zero mismatch.
            self_mismatch = decision_mismatch(oracle_shadow, oracle_shadow)
            self_check_rows.append(
                {"seed": seed, "self_mismatch_rate": float(self_mismatch["decision_mismatch"].mean())}
            )

            for cond in stress_grid:
                v = coverage_views[cond.coverage]
                ctx = coverage_ctx[cond.coverage]
                res_v = telemetry_view(res_full, v, sim_line)

                if cond.fault_kind is not None:
                    # Apply an existing sensor_health fault to one observed
                    # station, covering fault_fraction_of_run of the episode's
                    # duration. See B0.2: confirm this is the right point in
                    # the pipeline to inject before build_windows is called.
                    if v.observed_indices:
                        target_station = int(v.observed_indices[len(v.observed_indices) // 2])
                        t_span = float(res_v.telemetry["t_start_s"].max() - res_v.telemetry["t_start_s"].min()) \
                            if "t_start_s" in res_v.telemetry.columns else float(res_v.telemetry["t_depart_s"].max())
                        fault = SensorFault(
                            station=target_station,
                            kind=cond.fault_kind,
                            t_start_s=0.0,
                            t_end_s=t_span * cond.fault_fraction_of_run,
                        )
                        res_v.telemetry = apply_sensor_faults(res_v.telemetry, [fault], seed=seed)

                scored = ctx.baseline.score(build_windows(res_v, v, spec), v)
                _, stress_shadow, _ = infer(ctx, res_v)

                dm = decision_mismatch(oracle_shadow, stress_shadow)
                dmr = float(dm["decision_mismatch"].mean()) if len(dm) else np.nan

                # Outcome gap: median station-distance between oracle's and
                # stress condition's named station, on windows where the
                # oracle detected something. A cheap, already-available proxy
                # for "did the consequence change" — see runbook B0.3.
                paired = oracle_shadow[["window", "top_station", "detected"]].merge(
                    stress_shadow[["window", "top_station"]], on="window", suffixes=("_oracle", "_stress")
                )
                detected_rows = paired[paired["detected"]]
                if len(detected_rows):
                    outcome_gap = float(
                        (detected_rows["top_station_oracle"] - detected_rows["top_station_stress"]).abs().median()
                    )
                else:
                    outcome_gap = np.nan

                rows.append(
                    {
                        "seed": seed,
                        "condition": cond.label,
                        "coverage": cond.coverage,
                        "fault_kind": cond.fault_kind,
                        "decision_mismatch_rate": dmr,
                        "outcome_gap_station_distance": outcome_gap,
                        "n_windows": len(dm),
                    }
                )
            if verbose:
                print(f"[stress_test] seed={seed} done ({time.time() - t_start:.0f}s)")

    result_df = pd.DataFrame(rows)
    self_check_df = pd.DataFrame(self_check_rows)
    result_df.to_csv(out_dir / "tables" / "decision_outcome_gap.csv", index=False)
    self_check_df.to_csv(out_dir / "tables" / "oracle_self_check.csv", index=False)

    summary = (
        result_df.groupby(["condition", "coverage", "fault_kind"])
        .agg(
            n=("seed", "size"),
            mean_dmr=("decision_mismatch_rate", "mean"),
            median_outcome_gap=("outcome_gap_station_distance", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "tables" / "decision_outcome_gap_summary.csv", index=False)

    manifest = {
        "generated_by": "rippletwin.evaluation.stress_test.run_stress_test",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "methodology_reference": "arXiv:2608.14917 (Saad Saoud, 2026)",
        "config": asdict(cfg),
        "stress_grid": [cond.__dict__ for cond in stress_grid],
        "oracle_self_check_max_mismatch": float(self_check_df["self_mismatch_rate"].max()) if len(self_check_df) else None,
        "runtime_s": round(time.time() - t_start, 1),
    }
    (out_dir / "tables" / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    if verbose:
        print(f"\n[stress_test] done in {time.time() - t_start:.0f}s")
        print(f"[stress_test] oracle self-check max mismatch: {manifest['oracle_self_check_max_mismatch']}")

    return {"detail": result_df, "summary": summary, "self_check": self_check_df, "manifest": manifest}


if __name__ == "__main__":
    run_stress_test()
```

**Note for the implementing agent:** the `t_start_s`/`t_depart_s` column
guess and the `SensorFault` wiring point in the block above are best-effort
sketches based on what the review read, *not* verified against a running
test. Step B0 explicitly asks you to confirm these against
`tests/test_sensor_dynamics.py`'s real usage before trusting this code —
adjust the fault-injection block to match the real, working pattern that
file demonstrates.

### B2 — Tests

Create `tests/test_stress_test.py`:

```python
"""Tests for evaluation.stress_test (Plan B, RESEARCH_EVALUATION.md #11)."""

import pytest

from rippletwin.evaluation.experiments import ExperimentConfig
from rippletwin.evaluation.stress_test import StressCondition, run_stress_test


def _tiny_cfg() -> ExperimentConfig:
    # Small enough to run in a test suite: 2 test episodes, small vehicle counts.
    return ExperimentConfig(
        n_tune_episodes=0,
        n_test_episodes=2,
        episode_vehicles=300,
        nominal_vehicles=600,
        calibration_vehicles=600,
    )


def test_oracle_self_check_is_always_zero(tmp_path):
    cfg = _tiny_cfg()
    grid = [StressCondition(coverage=0.75, fault_kind=None, label="cov75_clean")]
    result = run_stress_test(cfg, stress_grid=grid, out_dir=tmp_path, verbose=False)
    assert result["self_check"]["self_mismatch_rate"].max() == pytest.approx(0.0)


def test_decision_mismatch_rate_nondecreasing_with_fault_severity(tmp_path):
    cfg = _tiny_cfg()
    grid = [
        StressCondition(coverage=0.75, fault_kind=None, fault_fraction_of_run=0.0, label="clean"),
        StressCondition(coverage=0.75, fault_kind="DROPOUT", fault_fraction_of_run=0.10, label="dropout10"),
        StressCondition(coverage=0.75, fault_kind="DROPOUT", fault_fraction_of_run=0.50, label="dropout50"),
    ]
    result = run_stress_test(cfg, stress_grid=grid, out_dir=tmp_path, verbose=False)
    means = result["summary"].set_index("condition")["mean_dmr"]
    assert means["clean"] <= means["dropout10"] + 1e-9
    assert means["dropout10"] <= means["dropout50"] + 1e-9


def test_outcome_gap_metric_has_no_nan_crash(tmp_path):
    cfg = _tiny_cfg()
    grid = [StressCondition(coverage=0.50, fault_kind="NOISY", fault_fraction_of_run=0.20, label="cov50_noisy")]
    result = run_stress_test(cfg, stress_grid=grid, out_dir=tmp_path, verbose=False)
    assert not result["detail"].empty
    assert "outcome_gap_station_distance" in result["detail"].columns
```

If B0's investigation reveals the fault-injection wiring needs to differ
from the sketch in B1, update these tests' `fault_kind`/`fault_fraction_of_run`
usage to match the corrected implementation — the *properties* under test
(self-check is zero, mismatch rises with severity, no crash) should not
change.

### B3 — Validate

```bash
cd /home/mouryesh/mouryesh/AIC/RippleTwin
PYTHONPATH=src pytest tests/test_stress_test.py -v
PYTHONPATH=src pytest -q   # full suite — must stay green
```

Then run the full-scale version once, against the same episode/seed set as
the flagship run, so the numbers are honestly cross-referenceable:

```bash
PYTHONPATH=src python -m rippletwin.evaluation.stress_test
```

This will take a while (it runs `n_test_episodes` full episodes × the
stress grid — expect a similar order of magnitude to
`evaluation/experiments.py`'s own ~25-minute flagship run; let it run to
completion). Confirm on completion:
1. `results_stress_test/tables/oracle_self_check.csv` — every
   `self_mismatch_rate` is exactly 0. If not, the harness itself has a bug
   — do not trust any other number in this run until this is fixed.
2. `results_stress_test/tables/decision_outcome_gap_summary.csv` —
   sanity-check that `mean_dmr` rises with fault severity/coverage loss,
   as expected.

### B4 — Write up and commit

Add a short new section to `docs/RESULTS.md` (or a new
`docs/STRESS_TEST.md`, linked from `docs/RESULTS.md`) presenting the
`decision_outcome_gap_summary.csv` numbers, labeled `SIMULATED PROTOTYPE
RESULT` per the existing convention, and stating plainly that the oracle
self-check passed. Do this only after B3's self-check confirms 0.

```bash
git add src/rippletwin/evaluation/stress_test.py tests/test_stress_test.py
git add docs/RESULTS.md   # or docs/STRESS_TEST.md, whichever you created
git commit -m "$(cat <<'EOF'
Add decision-vs-outcome stress test harness (Plan B)

Oracle-vs-sensor-corrupted paired comparison, following Saad Saoud
(2026)'s ground-truth-aware stress-testing methodology (arXiv:2608.14917).
Purely additive: reuses the existing simulator, factory.sensor_health
fault injection, and evaluation.experiments' episode/seed/coverage
machinery read-only; writes to results_stress_test/, never to results/.
See docs/RESEARCH_EVALUATION.md item 11 and
docs/IMPLEMENTATION_RUNBOOK.md Plan B.
EOF
)"
```

(Do not include `results_stress_test/` itself in the commit unless the
project's convention is to check in generated tables — check whether
`results/` is tracked in git or gitignored first: `git check-ignore -v
results/tables/manifest.json`. Match whatever the existing convention is
for `results_stress_test/`.)

---

## Plan C

**Goal:** fuse the flow-path LLR and the quality-path LLR for the narrow
case of `COMBINED`-fault-kind episodes where `placement.py::ambiguity()`
has flagged a candidate group as highly confusable. This is the
highest-risk of the three plans — it touches `twin/pipeline.py` (additively)
and requires a gated, paired before/after comparison before any promotion.
**Do this only after Plans A and B are committed and the suite is green.**

### C0 — Verify assumptions (do not skip — this is where the real risk is)

1. Re-read `src/rippletwin/twin/genealogy.py::quality_state` in full
   (already summarized in the review). Confirm exactly how its `window`
   values relate to time — it pools `pool_vehicles=200` vehicles ending at
   each `window_bounds` row's `v_end`, which is a **much coarser**
   granularity than the flow path's 20-vehicle/stride-5 windows. Before
   writing any fusion code, write down explicitly: given a flow-path
   window index `w_flow` with time range `[t_lo, t_hi]` (from
   `_window_times`/`window_bounds_from` in `experiments.py`/`genealogy.py`),
   which quality-path window(s) overlap that same time range? This mapping
   is the actual hard part of this plan — get it right on paper before
   writing code.
2. Re-read `src/rippletwin/twin/placement.py::ambiguity` in full (already
   read during the review). Confirm the exact meaning of its returned
   `ambiguity` column (cosine similarity to best-matching rival) and
   `confusable_with` column, and how to get "the set of stations in one
   ambiguous group" from its output (it returns one row per station with
   its single best-matching partner — you'll need to derive connected
   groups, e.g., stations mutually confusable above a threshold, likely
   via a simple threshold + graph-connected-components pass, or by reusing
   whatever grouping logic already exists elsewhere in the codebase for
   the documented "S32/S33 and S37/S38" adjacent-blind-pair finding —
   search for where that finding is generated, e.g., in
   `evaluation/`, before writing new grouping logic from scratch).
3. Confirm `twin/pipeline.py::infer`'s exact signature and where in its
   body the `shadow` frame is finalized before being returned — this is
   the addition point for the new, opt-in fusion columns.

### C1 — New module

Create `src/rippletwin/twin/evidence_fusion.py`:

```python
"""Flow-path + quality-path evidence fusion for ambiguous blind-station groups.

Scope, deliberately narrow (see docs/RESEARCH_EVALUATION.md Hybrid 2):
fires ONLY when (a) the flow posterior's leading candidate belongs to a
group placement.ambiguity() has flagged as highly confusable, AND (b) an
independent quality-path signal (twin.genealogy.quality_state) exists for
at least one candidate in that group over an overlapping time window.
Everywhere else, this module has zero effect.

Mechanism: additive log-likelihood combination,
    combined_score(station) = flow_llr(station) + quality_llr(station)
restricted to stations in the ambiguous group — the same "independent
evidence channels combine additively in log-space" principle
twin.shadow's own z_proc likelihood-ratio term already uses. No training
data, fully deterministic, fully auditable (both terms are already
individually explainable).

This module NEVER overwrites twin.shadow's `top_station`. It produces new,
opt-in columns (`fused_top_station`, `fused_llr_margin`) that a caller may
choose to read. Promotion to replace `top_station` as the default is a
separate, explicitly gated decision — see runbook step C4 and
docs/RESEARCH_EVALUATION.md Hybrid 2's decision gate.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

AMBIGUITY_THRESHOLD = 0.90  # cosine similarity above which two stations are "flagged confusable"


def ambiguous_groups(ambiguity_df: pd.DataFrame, threshold: float = AMBIGUITY_THRESHOLD) -> List[List[int]]:
    """Connected groups of mutually-confusable hidden stations.

    ``ambiguity_df`` is the output of twin.placement.ambiguity(). Two
    stations are linked if either names the other as its best-matching
    rival (``confusable_with``) at ``ambiguity >= threshold``. Returns a
    list of station-index groups (singletons excluded — only groups of
    size >= 2 are ambiguous by definition).
    """
    flagged = ambiguity_df[ambiguity_df["ambiguity"] >= threshold]
    if flagged.empty:
        return []
    id_to_index = {row.station_id: row.station for row in ambiguity_df.itertuples()}
    edges = set()
    for row in flagged.itertuples():
        partner_idx = id_to_index.get(row.confusable_with)
        if partner_idx is not None:
            edges.add(frozenset({row.station, partner_idx}))
    # Union-find over the edge set to get connected components.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for edge in edges:
        a, b = tuple(edge)
        union(a, b)

    groups: dict = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


def fuse_ambiguous_group(
    flow_shadow_row: pd.Series,
    quality_state_df: pd.DataFrame,
    ambiguity_group: List[int],
    window_lo_s: float,
    window_hi_s: float,
    window_bounds: pd.DataFrame,
) -> Optional[dict]:
    """Re-rank one flow-flagged ambiguous group using overlapping quality LLR.

    ``window_bounds`` (from twin.genealogy.window_bounds_from) maps
    quality-path window indices to (t_lo, t_hi) so overlap with the flow
    window's [window_lo_s, window_hi_s] can be found — see runbook step
    C0.1 for why this alignment is the crux of the whole plan.

    Returns None if no quality evidence overlaps the group/window (the
    fusion condition doesn't hold — caller should fall back to the
    unfused top_station). Otherwise returns
    {"fused_top_station": int, "fused_llr_margin": float}.
    """
    overlapping = window_bounds[
        (window_bounds["t_hi"] >= window_lo_s) & (window_bounds["t_lo"] <= window_hi_s)
    ]
    if overlapping.empty:
        return None
    q = quality_state_df[
        quality_state_df["window"].isin(overlapping["window"])
        & quality_state_df["station"].isin(ambiguity_group)
    ]
    if q.empty:
        return None

    quality_llr_by_station = q.groupby("station")["llr"].sum().to_dict()
    if not any(quality_llr_by_station.get(s, 0.0) > 0.0 for s in ambiguity_group):
        return None  # quality path has no signal on ANY group member — nothing to fuse

    flow_llr = float(flow_shadow_row.get("llr", 0.0))
    scores = {}
    for s in ambiguity_group:
        # Flow LLR is only meaningfully attributable to the row's own leading
        # candidate; other group members get the flow LLR at 0 contribution
        # (they were tied/near-tied by construction of an ambiguous group,
        # which is exactly why we don't have a per-station flow LLR here —
        # confirm during C0 whether ShadowSensor exposes a per-candidate LLR
        # vector rather than only the winning one, and use it if so, which
        # would make this fusion sharper).
        own_flow = flow_llr if s == int(flow_shadow_row["top_station"]) else 0.0
        scores[s] = own_flow + quality_llr_by_station.get(s, 0.0)

    best_station = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
    return {"fused_top_station": best_station, "fused_llr_margin": margin}
```

**Note for the implementing agent:** the comment inside `fuse_ambiguous_group`
about "flow LLR is only meaningfully attributable to the row's own leading
candidate" flags a real, unresolved design question — confirm during C0
whether `ShadowSensor` internally has a full per-candidate LLR vector (not
just the winner) that could be exposed and used here for a sharper fusion.
If it does, prefer exposing and using it over the single-candidate
approximation above; if it doesn't, the approximation above is a
reasonable, honestly-documented first version, but note the limitation in
the module docstring rather than silently shipping it as if it were exact.

### C2 — Additive touch to `twin/pipeline.py`

In `src/rippletwin/twin/pipeline.py::infer`, after the existing `shadow`
frame is finalized (do not change anything before this point), add an
**optional** post-step:

```python
# --- optional evidence fusion for flagged-ambiguous groups (Plan C / Hybrid 2) ---
# Opt-in only: never overwrites top_station. See twin/evidence_fusion.py.
if getattr(ctx, "enable_evidence_fusion", False):
    from . import evidence_fusion as EF
    from . import genealogy as GN
    from .placement import ambiguity as compute_ambiguity

    amb_df = compute_ambiguity(line, line.observed_indices)
    groups = EF.ambiguous_groups(amb_df)
    if groups and not shadow.empty:
        wb = GN.window_bounds_from(scored)
        # quality_state_df and window_bounds must already be available in
        # this scope from the existing quality-path computation, or computed
        # fresh here — confirm during C0.3 exactly what's already in scope
        # inside infer() so this doesn't duplicate work unnecessarily.
        fused_stations, fused_margins = [], []
        for _, row in shadow.iterrows():
            group = next((g for g in groups if int(row["top_station"]) in g), None)
            if group is None:
                fused_stations.append(row["top_station"])
                fused_margins.append(np.nan)
                continue
            result = EF.fuse_ambiguous_group(
                row, quality_state_df, group, row.get("t_lo_s", row.get("t_mid_s", 0.0)),
                row.get("t_hi_s", row.get("t_mid_s", 0.0)), wb,
            )
            if result is None:
                fused_stations.append(row["top_station"])
                fused_margins.append(np.nan)
            else:
                fused_stations.append(result["fused_top_station"])
                fused_margins.append(result["fused_llr_margin"])
        shadow = shadow.assign(fused_top_station=fused_stations, fused_llr_margin=fused_margins)
```

This is deliberately gated behind `ctx.enable_evidence_fusion` (default
`False`/absent) so that **every existing caller of `infer()` — including
all 204 existing tests and `evaluation/experiments.py`'s flagship run — is
byte-identical to before**, since the attribute won't exist on their
context objects and the `getattr(..., False)` short-circuits immediately.
Confirm this default-off behavior explicitly in a test (C3).

### C3 — Tests

Create `tests/test_evidence_fusion.py`:

```python
"""Tests for twin.evidence_fusion (Plan C / Hybrid 2, RESEARCH_EVALUATION.md)."""

import pandas as pd
import pytest

from rippletwin.twin.evidence_fusion import ambiguous_groups, fuse_ambiguous_group
from rippletwin.factory.topology import build_line


LINE = build_line("configs/line_42.yaml", seed=7)


def test_ambiguous_groups_links_mutually_confusable_stations():
    amb = pd.DataFrame(
        [
            {"station": 32, "station_id": "S33", "ambiguity": 0.97, "confusable_with": "S34"},
            {"station": 33, "station_id": "S34", "ambiguity": 0.97, "confusable_with": "S33"},
            {"station": 10, "station_id": "S11", "ambiguity": 0.10, "confusable_with": "S12"},
        ]
    )
    groups = ambiguous_groups(amb, threshold=0.90)
    assert [32, 33] in groups
    assert not any(10 in g for g in groups)


def test_fuse_returns_none_when_no_quality_signal_overlaps():
    row = pd.Series({"top_station": 5, "llr": 12.0, "t_mid_s": 100.0})
    empty_quality = pd.DataFrame(columns=["window", "station", "llr"])
    wb = pd.DataFrame({"window": [0], "t_lo": [90.0], "t_hi": [110.0]})
    result = fuse_ambiguous_group(row, empty_quality, [5, 6], 90.0, 110.0, wb)
    assert result is None


def test_fuse_picks_the_station_quality_evidence_favors():
    row = pd.Series({"top_station": 5, "llr": 3.0, "t_mid_s": 100.0})
    quality = pd.DataFrame(
        [
            {"window": 0, "station": 5, "llr": 0.5},
            {"window": 0, "station": 6, "llr": 9.0},  # much stronger quality evidence for station 6
        ]
    )
    wb = pd.DataFrame({"window": [0], "t_lo": [90.0], "t_hi": [110.0]})
    result = fuse_ambiguous_group(row, quality, [5, 6], 90.0, 110.0, wb)
    assert result is not None
    assert result["fused_top_station"] == 6


def test_infer_default_behavior_is_unchanged_when_fusion_disabled():
    # This is the critical regression guard: confirm twin.pipeline.infer's
    # shadow output is byte-identical with and without the fusion attribute
    # present-but-False on the context object, on a real scenario. Wire this
    # up against however tests/test_shadow_sensing.py already builds a
    # TwinContext + runs infer() for a real scenario, and assert
    # shadow.equals(shadow_with_fusion_attr_false).
    pytest.skip(
        "Wire against the existing TwinContext-building pattern in "
        "tests/test_shadow_sensing.py or tests/test_pipeline_and_hitl.py "
        "before this plan is considered complete — see runbook step C3."
    )
```

The last test is intentionally left as a `pytest.skip` with an explicit
instruction rather than a guessed implementation, because it needs to
match the real `TwinContext` construction pattern used elsewhere in the
suite — **do not skip actually writing it**; un-skip and complete it using
the real pattern from `tests/test_shadow_sensing.py` or
`tests/test_pipeline_and_hitl.py` before this plan is considered done. It
is the single most important test in this whole plan, because it's the
regression guard proving the additive touch to `pipeline.py::infer` truly
has zero effect on every existing caller.

### C4 — Validate: full suite, then the gated before/after comparison

```bash
cd /home/mouryesh/mouryesh/AIC/RippleTwin
PYTHONPATH=src pytest tests/test_evidence_fusion.py -v
PYTHONPATH=src pytest -q   # full suite — must stay at 204+ passing, 0 failing, 0 changed
```

Then the actual decision-gate comparison. Run the flagship protocol twice,
identical `ExperimentConfig` (same 8 tune + 24 test episodes, same seeds
1000–1007/5000–5023, same 4 coverage levels, same `target_window_fpr=0.01`),
differing only in `enable_evidence_fusion`:

```bash
# Baseline (fusion disabled — should reproduce today's results/ numbers exactly):
PYTHONPATH=src python -c "
import sys; sys.path.insert(0, 'src')
from rippletwin.evaluation.experiments import run_experiment, ExperimentConfig
run_experiment(ExperimentConfig(), out_dir='results_fusion_eval_baseline')
"

# Fusion-enabled — requires wiring enable_evidence_fusion=True into the
# TwinContext that fit_context() builds inside run_experiment; add a
# fusion_enabled: bool = False field to ExperimentConfig, thread it through
# to fit_context/TwinContext, and confirm fit_context's signature accepts it
# (or add the plumbing there) before running this:
PYTHONPATH=src python -c "
import sys; sys.path.insert(0, 'src')
from rippletwin.evaluation.experiments import run_experiment, ExperimentConfig
run_experiment(ExperimentConfig(fusion_enabled=True), out_dir='results_fusion_eval_fused')
"
```

Then compare the two runs' `flow_faults_hidden_source.csv` and a
`COMBINED`-fault-kind-only slice of `by_fault_kind.csv`, station-by-station:

```bash
PYTHONPATH=src python -c "
import pandas as pd
base = pd.read_csv('results_fusion_eval_baseline/tables/by_fault_kind.csv')
fused = pd.read_csv('results_fusion_eval_fused/tables/by_fault_kind.csv')
combined_base = base[base['fault_kind'] == 'COMBINED']
combined_fused = fused[fused['fault_kind'] == 'COMBINED']
print('--- baseline COMBINED ---'); print(combined_base)
print('--- fused COMBINED ---'); print(combined_fused)
non_combined_base = base[base['fault_kind'] != 'COMBINED'].reset_index(drop=True)
non_combined_fused = fused[fused['fault_kind'] != 'COMBINED'].reset_index(drop=True)
print('non-COMBINED rows identical:', non_combined_base.equals(non_combined_fused))
"
```

### C5 — Decision gate (apply exactly, do not shortcut)

- If `non_combined rows identical` above is **not True**: the fusion logic
  leaked into cases it shouldn't touch. Stop, fix the gating in C2 (the
  `getattr`/group-membership check), and re-run C4 from the top. Do not
  proceed to writing anything into `docs/`.
- If it **is True**, compare `combined_fused` against `combined_base`
  station-by-station on the exact-localization metric:
  - **Matches or improves** on the `COMBINED`+ambiguous-group slice, with
    everything else unchanged: `fused_top_station` may be documented as
    the new default recommendation *for that slice only* — write this up
    honestly and narrowly in `docs/METHOD.md` (a new short subsection,
    not a rewrite of the existing localization story), citing the exact
    numbers from both runs. Do **not** change `top_station`'s definition
    or any existing `results/tables/*.csv` — the promotion is: future
    dashboard/recommendation code *may* prefer `fused_top_station` when
    present, `top_station` remains the audited, always-present field.
  - **Worse or merely different:** do not promote anything. Write it up
    in `docs/METHOD.md`'s "designs we killed" style — an honest negative
    result, exactly like the three failed designs already documented
    there — with the actual numbers from `results_fusion_eval_baseline`
    vs. `results_fusion_eval_fused`. The code stays in the repo as a
    documented, opt-in, non-default module; it is never wired into the
    default pipeline.

### C6 — Commit

Regardless of which branch of C5 you land on, commit the code +
tests + honest write-up together:

```bash
git add src/rippletwin/twin/evidence_fusion.py tests/test_evidence_fusion.py
git add src/rippletwin/twin/pipeline.py
git add src/rippletwin/evaluation/experiments.py   # only if fusion_enabled field was added
git add docs/METHOD.md
git commit -m "$(cat <<'EOF'
Add flow+quality LLR fusion for ambiguous blind-station groups (Plan C / Hybrid 2)

Opt-in, additive evidence fusion for COMBINED-fault-kind episodes where
placement.ambiguity() flags a highly confusable candidate group -
never wired into the default pipeline; enable_evidence_fusion defaults
False and every existing caller of twin.pipeline.infer is unaffected
(see tests/test_evidence_fusion.py's regression guard). Decision-gate
comparison against the unfused baseline, same seeds/protocol, documented
in docs/METHOD.md. See docs/RESEARCH_EVALUATION.md Hybrid 2 and
docs/IMPLEMENTATION_RUNBOOK.md Plan C.
EOF
)"
```

---

## Final step — push

Only after all three plans are committed (or after however many of them
you completed — do not hold the earlier plans' commits hostage to a later
plan's decision gate) and `pytest -q` is green:

```bash
cd /home/mouryesh/mouryesh/AIC/RippleTwin
git log --oneline main..HEAD          # sanity check: your commits, nothing unexpected
git push -u origin feature/plan-a-b-diagnostics
```

This pushes the feature branch only — it does not touch `main`. Open a PR
from `feature/plan-a-b-diagnostics` into `main` for review once pushed;
do not merge it yourself.

If, at any point, a plan's decision gate fails or a step can't be
completed as written (an API doesn't match what this runbook assumed),
**stop, commit whatever is safely working so far, and report back with
specifics** rather than guessing past the mismatch — this runbook was
written from a research review, not from having run this exact code, and
several steps say so explicitly (A0, B0, C0). Verification steps exist
precisely so a real discrepancy gets caught and reported, not papered
over.
