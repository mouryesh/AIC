"""The experiment harness: baseline comparison and the sensor-coverage sweep.

Protocol
--------
* One physics run per episode, always at full observability. Coverage levels are
  *views* over that run, so differences between levels are differences in
  observability and nothing else.
* The nominal baseline and the detector calibration are fitted per coverage
  level (they must be -- they depend on which stations are observed), on
  disturbance-free runs, using seeds disjoint from the evaluation episodes.
* Episodes are split into a tuning set and a held-out set by seed. Any threshold
  or length scale that was chosen by looking at data was chosen on the tuning
  set; reported numbers come from the held-out set.

Every number this module produces is a **simulated prototype result** on
synthetic data. It is not a measurement of any real production line.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..factory import scenarios as SC
from ..factory.topology import LineTopology, apply_coverage, build_line
from ..features.windows import WindowSpec
from ..models import baselines as BL
from ..twin import genealogy as GN
from ..twin.pipeline import build_windows, fit_context, infer, simulate
from ..twin.shadow import infer_hidden_cycle_time
from . import metrics as M
from .views import full_observability, telemetry_view

DEFAULT_COVERAGES = (1.00, 0.75, 0.50, 0.25)


@dataclass
class ExperimentConfig:
    line_config: str = "configs/line_42.yaml"
    line_seed: int = 7
    n_tune_episodes: int = 8
    n_test_episodes: int = 24
    episode_vehicles: int = 1200
    nominal_vehicles: int = 2600
    calibration_vehicles: int = 2200
    coverages: Sequence[float] = DEFAULT_COVERAGES
    target_window_fpr: float = 0.01
    window_width: int = 20
    window_stride: int = 5
    #: Seeds for tuning episodes start here; test seeds start well past them.
    tune_seed_base: int = 1000
    test_seed_base: int = 5000
    #: Opt-in flow+quality evidence fusion for ambiguous blind-station groups
    #: (Plan C / Hybrid 2, see twin/evidence_fusion.py). Defaults False, so
    #: every existing call to run_experiment() reproduces prior results
    #: exactly. When True, each coverage view's already-fitted QualityBaseline
    #: (qbaselines[c], computed below regardless) is attached to that view's
    #: TwinContext so twin.pipeline.infer's opt-in fusion step can run.
    fusion_enabled: bool = False


def _window_times(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby("window")
        .agg(t_mid_s=("t_depart_s_min", "min"), v_start=("v_start", "first"),
             v_end=("v_end", "first"))
        .reset_index()
    )


def _views(base_line: LineTopology, coverages: Sequence[float]) -> Dict[float, LineTopology]:
    """Build one observability view per coverage level.

    The 0.76 view is the line as configured; other levels are produced by
    demoting instrumented stations to MANUAL. Inspection gates are never
    demoted -- a plant that cannot read its own end-of-line test has a different
    problem.
    """
    out: Dict[float, LineTopology] = {}
    for c in coverages:
        out[c] = apply_coverage(base_line, c, seed=int(round(c * 1000)))
    return out


def run_experiment(
    cfg: ExperimentConfig | None = None,
    out_dir: str | Path = "results",
    verbose: bool = True,
) -> dict:
    """Run the full evaluation and write results to ``out_dir``."""
    cfg = cfg or ExperimentConfig()
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    spec = WindowSpec.for_line(
        build_line(cfg.line_config, seed=cfg.line_seed),
        width=cfg.window_width, stride=cfg.window_stride,
    )

    configured = build_line(cfg.line_config, seed=cfg.line_seed)
    sim_line = full_observability(configured)  # physics substrate

    # Coverage 1.0 means "instrumented everywhere", which is the idealised twin
    # the literature usually assumes. The configured line sits at ~0.76.
    coverages = list(cfg.coverages)
    views = _views(configured, coverages)
    views[1.00] = full_observability(configured)

    if verbose:
        print(f"line: {configured.n_stations} stations, configured coverage "
              f"{configured.coverage:.2f}")
        for c in coverages:
            v = views[c]
            print(f"  view {c:.2f}: observed={len(v.observed_indices)} "
                  f"hidden={len(v.hidden_indices)}")

    # ---- shared disturbance-free runs (physics at full observability) --------
    nominal_full = simulate(sim_line, SC.nominal_run(cfg.nominal_vehicles), seed=1)
    calib_full = simulate(sim_line, SC.nominal_run(cfg.calibration_vehicles), seed=2)

    # The line's own nominal output rate. Lead time is measured against a
    # shortfall relative to THIS, not against theoretical takt -- see
    # metrics.production_board_moment for why that distinction decides whether
    # the metric means anything.
    nominal_rate_vph = float(nominal_full.meta["throughput_vph"])
    if verbose:
        print(f"  nominal output rate: {nominal_rate_vph:.1f} veh/h "
              f"({nominal_rate_vph / (3600 / configured.takt_s) * 100:.0f}% of takt rate)")

    contexts: Dict[float, object] = {}
    qbaselines: Dict[float, GN.QualityBaseline] = {}
    nominal_scored: Dict[float, pd.DataFrame] = {}
    prior = GN.candidate_prior(configured)

    for c in coverages:
        v = views[c]
        nom_v = telemetry_view(nominal_full, v, sim_line)
        cal_v = telemetry_view(calib_full, v, sim_line)
        ctx = fit_context(v, nom_v, calibration_run=cal_v, spec=spec,
                          target_window_fpr=cfg.target_window_fpr)
        contexts[c] = ctx
        nominal_scored[c] = ctx.baseline.score(build_windows(nom_v, v, spec), v)
        nom_att = GN.attribute_defects(v, GN.explode_defects(nom_v.inspections), prior)
        qbaselines[c] = GN.QualityBaseline.fit(
            v, nom_att, n_vehicles=len(nom_v.vehicles)
        )
        if cfg.fusion_enabled:
            # Reuses the QualityBaseline already fit above -- no duplicate
            # work. See TwinContext.enable_evidence_fusion/quality_baseline
            # in twin/pipeline.py and evidence_fusion.py (Plan C).
            ctx.enable_evidence_fusion = True
            ctx.quality_baseline = qbaselines[c]
        if verbose:
            cal = ctx.calibration
            print(f"  view {c:.2f}: tau={cal['tau']:.4f} "
                  f"detect_llr={cal['detect_llr']:.1f} "
                  f"corr={cal['mean_pairwise_corr']:.3f}")

    # ---- episodes ------------------------------------------------------------
    def episode_seeds(split: str) -> List[int]:
        if split == "tune":
            return [cfg.tune_seed_base + i for i in range(cfg.n_tune_episodes)]
        return [cfg.test_seed_base + i for i in range(cfg.n_test_episodes)]

    # ---- per-method thresholds, calibrated on the held-out nominal run -------
    # Each method is placed at the same target false-alarm rate on its own score
    # scale. Comparing detection rates without this step measures nothing: a
    # method with a permissive threshold always "detects" more.
    FPR_GRID = (0.005, 0.01, 0.02, 0.05, 0.10)
    thresholds: Dict[float, Dict[str, float]] = {}
    thresholds_by_fpr: Dict[float, Dict[str, Dict[float, float]]] = {}

    for c in coverages:
        v = views[c]
        ctx = contexts[c]
        cal_v = telemetry_view(calib_full, v, sim_line)
        cal_scored = ctx.baseline.score(build_windows(cal_v, v, spec), v)
        _, cal_shadow, _ = infer(ctx, cal_v)
        cal_raw = BL.build_methods(
            v, cal_scored, cal_shadow, nominal_scored[c], ctx.shadow_cfg,
            ctx.baseline.sigma_blocked, ctx.baseline.sigma_starved,
        )
        thresholds[c] = {
            name: BL.calibrate_threshold(fr, cfg.target_window_fpr)
            for name, fr in cal_raw.items()
        }
        thresholds_by_fpr[c] = {
            name: {t: BL.calibrate_threshold(fr, t) for t in FPR_GRID}
            for name, fr in cal_raw.items()
        }
        if verbose:
            print(f"  view {c:.2f} thresholds @fpr={cfg.target_window_fpr}: "
                  + ", ".join(f"{k.split('_')[0]}={t:.2f}"
                              for k, t in thresholds[c].items()))

    loc_rows: List[dict] = []
    fa_rows: List[dict] = []
    cyc_rows: List[dict] = []
    qual_rows: List[dict] = []
    roc_rows: List[dict] = []

    for split in ("tune", "test"):
        for seed in episode_seeds(split):
            scen = SC.random_episode(
                configured, seed=seed, n_vehicles=cfg.episode_vehicles
            )
            res_full = simulate(sim_line, scen, seed=seed + 77)

            for c in coverages:
                v = views[c]
                ctx = contexts[c]
                res_v = telemetry_view(res_full, v, sim_line)
                scored, shadow, sensor = infer(ctx, res_v)
                wt = _window_times(scored)
                truth = M.episode_truth(
                    res_v, configured, view_line=v,
                    reference_rate_vph=nominal_rate_vph,
                )

                raw = BL.build_methods(
                    v, scored, shadow, nominal_scored[c], ctx.shadow_cfg,
                    ctx.baseline.sigma_blocked, ctx.baseline.sigma_starved,
                )

                for name, frame in raw.items():
                    thr = thresholds[c].get(name)
                    if thr is None:
                        continue
                    # Identical detection rule and matched false-alarm rate for
                    # every method, so detection rates are comparable at all.
                    fr = BL.apply_detection_rule(frame, thr, persistence=2)
                    base = {
                        "split": split, "seed": seed, "coverage": c, "method": name,
                        "fault_kind": truth.kind, "source_hidden": truth.source_is_hidden,
                        "magnitude": truth.magnitude, "threshold": thr,
                        "true_station": truth.station,
                    }
                    if truth.has_fault:
                        m = M.evaluate_localization(fr, wt, truth)
                        if m:
                            loc_rows.append({**base, **m})
                    f = M.evaluate_false_alarms(fr, truth, wt)
                    if f:
                        fa_rows.append({**base, **f})

                    # Operating curve: sweep the threshold across a range of
                    # target false-alarm rates so the comparison is not hostage
                    # to one arbitrary operating point.
                    if split == "test" and truth.has_fault:
                        for tf in (0.005, 0.01, 0.02, 0.05, 0.10):
                            t2 = thresholds_by_fpr[c][name].get(tf)
                            if t2 is None:
                                continue
                            fr2 = BL.apply_detection_rule(frame, t2, persistence=2)
                            m2 = M.evaluate_localization(fr2, wt, truth)
                            f2 = M.evaluate_false_alarms(fr2, truth, wt)
                            if m2:
                                roc_rows.append({
                                    **base, "target_fpr": tf,
                                    **m2,
                                    "false_alarm_rate": (f2 or {}).get("false_alarm_rate", np.nan),
                                })

                # --- inferred cycle time, RippleTwin only ---------------------
                if truth.has_fault and truth.kind in ("SLOWDOWN", "COMBINED"):
                    # `shadow` already carries t_mid_s / v_start / v_end, so no
                    # merge is needed (merging would collide and suffix them).
                    t0 = truth.t_start_s + truth.ramp_s
                    det = shadow[
                        shadow["detected"]
                        & (shadow["t_mid_s"] >= t0)
                        & (shadow["t_mid_s"] <= truth.t_end_s)
                        & (shadow["top_station"] == truth.station)
                    ]
                    if len(det):
                        r = det.iloc[len(det) // 2]
                        vs, ve = int(r["v_start"]), int(r["v_end"])
                        est = infer_hidden_cycle_time(
                            v, res_v.telemetry, truth.station, vs, ve
                        )
                        ce = M.cycle_time_error(
                            est, res_v.passes, truth.station, vs, ve
                        )
                        if ce:
                            cyc_rows.append(
                                {"split": split, "seed": seed, "coverage": c,
                                 "source_hidden": truth.source_is_hidden, **ce}
                            )

                # --- quality attribution path ---------------------------------
                wb = GN.window_bounds_from(scored)
                dfn = GN.explode_defects(res_v.inspections)
                qs = GN.quality_state(v, dfn, wb, qbaselines[c], pool_vehicles=200)
                if not qs.empty:
                    qa = GN.quality_alerts(qs)
                    row = {"split": split, "seed": seed, "coverage": c,
                           "fault_kind": truth.kind,
                           "source_hidden": truth.source_is_hidden}
                    if truth.has_fault and truth.kind in ("QUALITY_DRIFT", "COMBINED"):
                        wsel = wb[(wb["t_lo"] > truth.t_start_s + truth.ramp_s)
                                  & (wb["t_hi"] < truth.t_end_s)]["window"]
                        during = qa[qa["window"].isin(wsel)]
                        if len(during):
                            rank = during.groupby("station")["llr"].mean().sort_values(
                                ascending=False
                            )
                            order = list(rank.index)
                            pos = order.index(truth.station) + 1 if truth.station in order else np.nan
                            row.update({
                                "q_rank": pos,
                                "q_top1": float(pos == 1) if np.isfinite(pos) else np.nan,
                                "q_top3": float(pos <= 3) if np.isfinite(pos) else np.nan,
                                "q_top5": float(pos <= 5) if np.isfinite(pos) else np.nan,
                                "q_alert_rate": float(during["quality_alert"].mean()),
                                "q_m_hat": float(
                                    during[during["station"] == truth.station]["m_hat"].mean()
                                ),
                                "q_true_multiplier": truth.magnitude,
                            })
                            qual_rows.append(row)
                    elif not truth.has_fault:
                        row["q_false_alarm_rate"] = float(qa["quality_alert"].mean())
                        qual_rows.append(row)

            if verbose:
                print(f"  [{split}] episode seed={seed} done "
                      f"({time.time() - t_start:.0f}s)")

    # ---- assemble ------------------------------------------------------------
    loc = pd.DataFrame(loc_rows)
    fa = pd.DataFrame(fa_rows)
    cyc = pd.DataFrame(cyc_rows)
    qual = pd.DataFrame(qual_rows)
    roc = pd.DataFrame(roc_rows)

    loc.to_csv(out_dir / "tables" / "localization_raw.csv", index=False)
    fa.to_csv(out_dir / "tables" / "false_alarms_raw.csv", index=False)
    cyc.to_csv(out_dir / "tables" / "cycle_time_raw.csv", index=False)
    qual.to_csv(out_dir / "tables" / "quality_raw.csv", index=False)
    roc.to_csv(out_dir / "tables" / "operating_curve_raw.csv", index=False)

    if len(roc):
        roc_sum = M.summarise(
            roc.to_dict("records"), by=["coverage", "method", "target_fpr"]
        )
        roc_sum.to_csv(out_dir / "tables" / "operating_curve.csv", index=False)
        # Detection stratified by how strong the disturbance actually was --
        # an aggregate number hides that weak drifts are genuinely hard.
        r2 = roc[roc["target_fpr"] == cfg.target_window_fpr].copy()
        if len(r2):
            r2["mag_band"] = pd.cut(
                r2["magnitude"], [0, 1.20, 1.30, 1.50, 100],
                labels=["<=1.20x", "1.20-1.30x", "1.30-1.50x", "quality drift"],
            )
            mag_sum = M.summarise(
                r2.to_dict("records"), by=["coverage", "method", "mag_band"]
            )
            mag_sum.to_csv(out_dir / "tables" / "detection_by_magnitude.csv", index=False)
        else:
            mag_sum = pd.DataFrame()
    else:
        roc_sum = pd.DataFrame()
        mag_sum = pd.DataFrame()

    test_loc = loc[loc["split"] == "test"]
    test_fa = fa[fa["split"] == "test"]

    by_method = M.summarise(
        test_loc.to_dict("records"), by=["coverage", "method"]
    ) if len(test_loc) else pd.DataFrame()
    hidden_only = (
        M.summarise(
            test_loc[test_loc["source_hidden"]].to_dict("records"),
            by=["coverage", "method"],
        )
        if len(test_loc)
        else pd.DataFrame()
    )
    fa_summary = (
        M.summarise(test_fa.to_dict("records"), by=["coverage", "method"])
        if len(test_fa)
        else pd.DataFrame()
    )

    # The flow model is, by design, blind to a pure quality drift and to a burst
    # of micro-stops: neither changes the line's timing signature in the way a
    # constraint does. Averaging those into one detection number would hide both
    # what the flow path does well and where the quality path is required.
    FLOW_KINDS = ("SLOWDOWN", "COMBINED")
    flow_only = (
        M.summarise(
            test_loc[test_loc["fault_kind"].isin(FLOW_KINDS)].to_dict("records"),
            by=["coverage", "method"],
        )
        if len(test_loc)
        else pd.DataFrame()
    )
    flow_hidden = (
        M.summarise(
            test_loc[
                test_loc["fault_kind"].isin(FLOW_KINDS) & test_loc["source_hidden"]
            ].to_dict("records"),
            by=["coverage", "method"],
        )
        if len(test_loc)
        else pd.DataFrame()
    )
    by_kind = (
        M.summarise(test_loc.to_dict("records"), by=["coverage", "method", "fault_kind"])
        if len(test_loc)
        else pd.DataFrame()
    )

    for name, df in [
        ("baseline_comparison", by_method),
        ("hidden_source_only", hidden_only),
        ("flow_faults_only", flow_only),
        ("flow_faults_hidden_source", flow_hidden),
        ("by_fault_kind", by_kind),
        ("false_alarms", fa_summary),
    ]:
        if len(df):
            df.to_csv(out_dir / "tables" / f"{name}.csv", index=False)

    if len(cyc):
        cyc_test = cyc[cyc["split"] == "test"]
        cyc_sum = (
            cyc_test.groupby(["coverage", "source_hidden"])
            .agg(n=("abs_error_pct", "size"),
                 mae_pct=("abs_error_pct", "mean"),
                 median_pct=("abs_error_pct", "median"),
                 mae_s=("abs_error_s", "mean"))
            .reset_index()
        )
        cyc_sum.to_csv(out_dir / "tables" / "cycle_time_inference.csv", index=False)
    else:
        cyc_sum = pd.DataFrame()

    if len(qual):
        q_test = qual[qual["split"] == "test"].copy()
        # A short run may contain no clean episodes (or no quality-drift ones),
        # so the corresponding columns can be absent entirely.
        for col in ["q_top1", "q_top3", "q_top5", "q_rank", "q_false_alarm_rate",
                    "q_m_hat", "q_true_multiplier"]:
            if col not in q_test.columns:
                q_test[col] = np.nan
        q_sum = (
            q_test.groupby("coverage")
            .agg(n=("seed", "size"),
                 q_top1=("q_top1", "mean"),
                 q_top3=("q_top3", "mean"),
                 q_top5=("q_top5", "mean"),
                 q_median_rank=("q_rank", "median"),
                 q_m_hat=("q_m_hat", "mean"),
                 q_true_multiplier=("q_true_multiplier", "mean"),
                 q_false_alarm_rate=("q_false_alarm_rate", "mean"))
            .reset_index()
        )
        q_sum.to_csv(out_dir / "tables" / "quality_attribution.csv", index=False)
    else:
        q_sum = pd.DataFrame()

    manifest = {
        "generated_by": "rippletwin.evaluation.experiments.run_experiment",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "config": asdict(cfg),
        "line": configured.summary(),
        "calibration": {str(c): contexts[c].calibration for c in coverages},
        "nominal_rate_vph": nominal_rate_vph,
        "n_localization_rows": int(len(loc)),
        "n_episodes_test": int(cfg.n_test_episodes),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (out_dir / "tables" / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    if verbose:
        print(f"\ndone in {time.time() - t_start:.0f}s")

    return {
        "operating_curve": roc_sum,
        "detection_by_magnitude": mag_sum,
        "localization": loc,
        "false_alarms": fa,
        "cycle_time": cyc,
        "quality": qual,
        "baseline_comparison": by_method,
        "hidden_source_only": hidden_only,
        "flow_faults_only": flow_only,
        "flow_faults_hidden_source": flow_hidden,
        "by_fault_kind": by_kind,
        "fa_summary": fa_summary,
        "cycle_summary": cyc_sum,
        "quality_summary": q_sum,
        "manifest": manifest,
    }
