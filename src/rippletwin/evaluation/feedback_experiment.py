"""Does closing the feedback loop actually help? (Round 2 brief §21)

Two separate questions, answered separately:

1. **Mechanism check**: given a validated ledger history (some stations
   repeatedly confirmed, one repeatedly a false accusation), does
   ``twin.feedback.priors_from_precision`` -> ``apply_feedback`` visibly
   move the posterior on a held-out example in the expected direction? This
   is a small, concrete before/after demonstration, not a statistical claim.

2. **Does it move the headline number**: applying the resulting priors to
   the same held-out localisation evaluation the rest of this repository
   uses, does aggregate accuracy improve, stay flat, or get worse? Reported
   honestly either way -- if it does not help at the sample sizes available
   here, that is exactly what should be reported (see README.md's
   "Round 1 -> Round 2" table: "including the parts of it we failed").

Every number here is a SIMULATED PROTOTYPE RESULT on synthetic data.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..factory import scenarios as SC
from ..factory.topology import build_line
from ..hitl.ledger import (
    DECISION_APPROVED,
    DECISION_REJECTED,
    OUTCOME_CONFIRMED,
    OUTCOME_NOT_FOUND,
    DecisionLedger,
)
from ..twin.feedback import apply_feedback, priors_from_precision
from ..twin.pipeline import fit_context, infer, simulate
from . import metrics as M


def _synthetic_validated_ledger(line, good_station_id: str, bad_station_id: str, n_each: int = 6) -> DecisionLedger:
    """A ledger as if `n_each` alerts at each station were reviewed and their
    outcomes verified on the floor: `good_station_id` confirmed every time,
    `bad_station_id` never found where predicted."""
    ledger = DecisionLedger()
    for i in range(n_each):
        e = ledger.record_alert(
            "synthetic", i, "FLOW", good_station_id, "MANUAL", True, 0.8, {}, {}
        )
        ledger.record_decision(e.entry_id, DECISION_APPROVED, "supervisor")
        ledger.record_outcome(e.entry_id, OUTCOME_CONFIRMED, "Confirmed on the floor.")
    for i in range(n_each):
        e = ledger.record_alert(
            "synthetic", 100 + i, "FLOW", bad_station_id, "MANUAL", True, 0.6, {}, {}
        )
        ledger.record_decision(e.entry_id, DECISION_APPROVED, "supervisor")
        ledger.record_outcome(e.entry_id, OUTCOME_NOT_FOUND, "Nothing found at the named station.")
    return ledger


@dataclass
class FeedbackExperimentConfig:
    line_config: str = "configs/line_42.yaml"
    line_seed: int = 7
    nominal_vehicles: int = 2200
    calibration_vehicles: int = 1800
    n_episodes: int = 14
    episode_vehicles: int = 1200
    episode_seed_base: int = 9200


def run_feedback_experiment(
    cfg: FeedbackExperimentConfig | None = None,
    out_dir: str | Path = "results",
    verbose: bool = True,
) -> dict:
    cfg = cfg or FeedbackExperimentConfig()
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    line = build_line(cfg.line_config, seed=cfg.line_seed)
    nominal = simulate(line, SC.nominal_run(cfg.nominal_vehicles), seed=1)
    calib = simulate(line, SC.nominal_run(cfg.calibration_vehicles), seed=2)
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)

    hidden = line.hidden_indices
    good_idx, bad_idx = hidden[0], hidden[1] if len(hidden) > 1 else hidden[0]
    good_id, bad_id = line.stations[good_idx].station_id, line.stations[bad_idx].station_id
    ledger = _synthetic_validated_ledger(line, good_id, bad_id)
    weights = priors_from_precision(line, ledger)
    fed_cfg = apply_feedback(ctx.shadow_cfg, weights)

    # --- 1. mechanism check: a held-out episode where `good_idx` is truly
    # the constraint. Does the confirmed-history prior increase its
    # posterior relative to the unmodified prior, all else equal?
    scen = SC.scenario_hidden_bottleneck(line)
    # Force the disturbance onto good_idx specifically, matching the
    # scenario's own disturbance shape.
    from ..factory.simulator import Disturbance, EVENT_SLOWDOWN
    d = Disturbance(station=good_idx, kind=EVENT_SLOWDOWN, t_start_s=36_000, t_end_s=86_000,
                     magnitude=1.32, ramp_s=2_400, label="feedback-mechanism-check")
    res = simulate(line, type(scen)(scenario_id="FEEDBACK_CHECK", title="", question="",
                                     n_vehicles=1800, disturbances=[d]), seed=20260301)
    scored, shadow_before, sensor_before = infer(ctx, res)
    ctx_after = type(ctx)(line=ctx.line, baseline=ctx.baseline, spec=ctx.spec,
                           shadow_cfg=fed_cfg, calibration=ctx.calibration)
    _, shadow_after, sensor_after = infer(ctx_after, res)

    mid = len(shadow_before) // 2
    before_p = float(shadow_before.iloc[mid]["top_prob"]) if shadow_before.iloc[mid]["top_station"] == good_idx else None
    # Compare posterior on good_idx specifically at a matched window.
    w_check = shadow_before.iloc[mid]["window"]
    before_good = float(sensor_before.last_results[mid].posterior.get(good_idx, np.nan))
    after_good = float(sensor_after.last_results[mid].posterior.get(good_idx, np.nan))

    mechanism = {
        "good_station_id": good_id, "bad_station_id": bad_id,
        "window_checked": int(w_check),
        "posterior_on_good_station_before_feedback": before_good,
        "posterior_on_good_station_after_feedback": after_good,
        "moved_in_expected_direction": bool(after_good >= before_good),
        "weights": weights,
    }
    if verbose:
        print(f"mechanism check: P({good_id}) before={before_good:.3f} after={after_good:.3f}")

    # --- 2. headline check: aggregate localisation accuracy, with and
    # without the feedback-adjusted prior, over held-out episodes.
    rows = []
    for regime, use_cfg in (("baseline", ctx.shadow_cfg), ("with_feedback", fed_cfg)):
        regime_ctx = type(ctx)(line=ctx.line, baseline=ctx.baseline, spec=ctx.spec,
                                shadow_cfg=use_cfg, calibration=ctx.calibration)
        for i in range(cfg.n_episodes):
            seed = cfg.episode_seed_base + i
            scen_i = SC.random_episode(line, seed=seed, n_vehicles=cfg.episode_vehicles)
            res_i = simulate(line, scen_i, seed=seed + 77)
            scored_i, shadow_i, _ = infer(regime_ctx, res_i)
            truth = M.episode_truth(res_i, line)
            wt = shadow_i[["window"]].drop_duplicates()
            row = {"regime": regime, "seed": seed, "has_fault": truth.has_fault}
            if truth.has_fault:
                m = M.evaluate_localization(shadow_i, wt, truth)
                row.update({f"loc_{k}": v for k, v in m.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "feedback_headline_raw.csv", index=False)
    numeric = [c for c in df.columns if c not in ("regime", "seed", "has_fault")
               and pd.api.types.is_numeric_dtype(df[c])]
    headline = df.groupby("regime")[numeric].mean().reset_index()
    headline.to_csv(out_dir / "tables" / "feedback_headline_summary.csv", index=False)

    baseline_top1 = float(headline.loc[headline["regime"] == "baseline", "loc_top1"].iloc[0]) \
        if "loc_top1" in headline.columns else float("nan")
    feedback_top1 = float(headline.loc[headline["regime"] == "with_feedback", "loc_top1"].iloc[0]) \
        if "loc_top1" in headline.columns else float("nan")
    headline_delta = feedback_top1 - baseline_top1 if np.isfinite(baseline_top1) and np.isfinite(feedback_top1) else float("nan")

    honest_verdict = (
        "IMPROVED" if headline_delta > 0.02 else
        "NO MEASURABLE CHANGE" if abs(headline_delta) <= 0.02 else
        "WORSE"
    ) if np.isfinite(headline_delta) else "INSUFFICIENT DATA"

    manifest = {
        "generated_by": "rippletwin.evaluation.feedback_experiment.run_feedback_experiment",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "config": asdict(cfg),
        "mechanism_check": mechanism,
        "headline_top1_baseline": baseline_top1,
        "headline_top1_with_feedback": feedback_top1,
        "headline_delta": headline_delta,
        "honest_verdict": honest_verdict,
        "caveat": (
            f"n={cfg.n_episodes} episodes per regime -- a headline delta this size is "
            f"not a statistically powered claim either way. Reported as measured."
        ),
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "tables" / "feedback_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    if verbose:
        print(f"headline: baseline top1={baseline_top1:.3f} with_feedback top1={feedback_top1:.3f} "
              f"delta={headline_delta:+.3f} -> {honest_verdict}")
        print(f"done in {time.time() - t0:.0f}s")
    return {"mechanism": mechanism, "headline": headline, "manifest": manifest}
