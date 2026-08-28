"""Early-warning experiment: does the predictive layer buy real lead time?

Reuses the same episode-generation and pipeline-fitting machinery as
``evaluation.experiments`` (one physics run, coverage as a view, tune/test
split by seed). Scores ``twin.predict.run_predictor`` on the dedicated
gradual-ramp scenario (``S6_EARLY_WARNING``) plus a sample of the same
random-episode corpus used elsewhere, so the reported false-alarm rate is not
measured only on the one scenario built to make the predictor look good.

Every number here is a **simulated prototype result** on synthetic data, same
as ``evaluation.experiments``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..factory import scenarios as SC
from ..factory.topology import LineTopology, apply_coverage, build_line
from ..features.windows import WindowSpec
from ..twin import predict as PR
from ..twin.pipeline import build_windows, fit_context, infer, simulate
from . import metrics as M
from .views import full_observability, telemetry_view


@dataclass
class EarlyWarningConfig:
    line_config: str = "configs/line_42.yaml"
    line_seed: int = 7
    #: ``None`` uses the line as configured (~0.76 coverage); a float applies
    #: ``apply_coverage`` first.
    coverage: Optional[float] = None
    nominal_vehicles: int = 2600
    calibration_vehicles: int = 2200
    n_random_episodes: int = 16
    episode_vehicles: int = 1200
    target_window_fpr: float = 0.01
    watch_target_fpr: float = 0.05
    episode_seed_base: int = 9000
    window_width: int = 20
    window_stride: int = 5


def _run_one_episode(
    scen, sim_line: LineTopology, view: LineTopology, ctx, seed: int
) -> dict:
    res_full = simulate(sim_line, scen, seed=seed)
    res_v = telemetry_view(res_full, view, sim_line)
    scored, shadow, sensor = infer(ctx, res_v)
    pred = PR.run_predictor(shadow, view, res_v.telemetry, scored, ctx.shadow_cfg)

    truth = M.episode_truth(res_v, view, view_line=view)
    onset = M.true_bottleneck_onset(view, truth) if truth.has_fault else None
    ew = M.evaluate_early_warning(pred, truth, onset)
    return {
        "scenario": scen.scenario_id,
        "seed": seed,
        "has_fault": truth.has_fault,
        "source_hidden": truth.source_is_hidden if truth.has_fault else None,
        "true_onset_s": onset,
        **ew,
    }


def run_early_warning_experiment(
    cfg: EarlyWarningConfig | None = None,
    out_dir: str | Path = "results",
    verbose: bool = True,
) -> dict:
    cfg = cfg or EarlyWarningConfig()
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    configured = build_line(cfg.line_config, seed=cfg.line_seed)
    sim_line = full_observability(configured)
    view = (
        configured
        if cfg.coverage is None
        else apply_coverage(configured, cfg.coverage, seed=11)
    )
    spec = WindowSpec.for_line(configured, width=cfg.window_width, stride=cfg.window_stride)

    nominal_full = simulate(sim_line, SC.nominal_run(cfg.nominal_vehicles), seed=1)
    calib_full = simulate(sim_line, SC.nominal_run(cfg.calibration_vehicles), seed=2)
    nom_v = telemetry_view(nominal_full, view, sim_line)
    cal_v = telemetry_view(calib_full, view, sim_line)

    ctx = fit_context(
        view, nom_v, calibration_run=cal_v, spec=spec,
        target_window_fpr=cfg.target_window_fpr, watch_target_fpr=cfg.watch_target_fpr,
    )
    if verbose:
        c = ctx.calibration
        print(
            f"early-warning calibration: detect_llr={c['detect_llr']:.2f} "
            f"watch_llr={c['watch_llr']:.2f} (target fpr "
            f"{cfg.target_window_fpr}/{cfg.watch_target_fpr})"
        )

    rows: List[dict] = []

    # The dedicated demonstration scenario, run at several seeds so it is not
    # a single anecdote.
    for i in range(6):
        scen = SC.scenario_gradual_bottleneck(configured, seed=606 + i)
        rows.append(_run_one_episode(scen, sim_line, view, ctx, seed=6000 + i))

    # A sample of the general random-episode corpus -- includes clean episodes
    # (false-alarm exposure) and other fault kinds/magnitudes.
    for i in range(cfg.n_random_episodes):
        seed = cfg.episode_seed_base + i
        scen = SC.random_episode(configured, seed=seed, n_vehicles=cfg.episode_vehicles)
        rows.append(_run_one_episode(scen, sim_line, view, ctx, seed=seed + 77))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "early_warning_raw.csv", index=False)

    faulted = df[df["has_fault"] == True]  # noqa: E712
    clean = df[df["has_fault"] == False]  # noqa: E712
    with_lead = faulted[faulted["missed"] == 0.0]["lead_time_min"].dropna()

    summary = {
        "n_episodes": int(len(df)),
        "n_faulted": int(len(faulted)),
        "n_clean": int(len(clean)),
        "n_missed": int(faulted["missed"].sum()) if len(faulted) else 0,
        "miss_rate": float(faulted["missed"].mean()) if len(faulted) else float("nan"),
        "n_with_lead_time": int(len(with_lead)),
        "mean_lead_time_min": float(with_lead.mean()) if len(with_lead) else float("nan"),
        "median_lead_time_min": float(with_lead.median()) if len(with_lead) else float("nan"),
        "min_lead_time_min": float(with_lead.min()) if len(with_lead) else float("nan"),
        "false_alarm_rate": float(df["false_alarm"].mean()) if len(df) else float("nan"),
        "false_alarm_rate_on_clean_episodes": (
            float(clean["false_alarm"].mean()) if len(clean) else float("nan")
        ),
    }
    summary_row = pd.DataFrame([summary])
    summary_row.to_csv(out_dir / "tables" / "early_warning_summary.csv", index=False)

    manifest = {
        "generated_by": "rippletwin.evaluation.early_warning.run_early_warning_experiment",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "config": asdict(cfg),
        "calibration": ctx.calibration,
        "summary": summary,
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "tables" / "early_warning_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    if verbose:
        print(
            f"early warning: {summary['n_faulted']} faulted episodes, "
            f"{summary['n_missed']} missed, median lead time "
            f"{summary['median_lead_time_min']:.1f} min, false-alarm rate "
            f"{summary['false_alarm_rate']:.3f} "
            f"(done in {time.time() - t0:.0f}s)"
        )

    return {"raw": df, "summary": summary, "manifest": manifest}
