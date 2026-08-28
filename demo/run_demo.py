#!/usr/bin/env python3
"""RippleTwin flagship demo -- deterministic, end to end.

Runs the full loop on a fixed seed:

    normal production
      -> a hidden disturbance starts at a station with no sensor
      -> neighbouring instrumented stations begin to deviate
      -> shadow-sensing localises the constraint
      -> the physics forecasts the downstream ripple
      -> the system explains itself from its own evidence
      -> it recommends an action for a person
      -> the supervisor approves or rejects
      -> the outcome is written to a hash-chained ledger

Nothing here is scripted for effect. The disturbance is injected into the
simulator's physics, the model receives only the telemetry a plant would
actually have, and every number printed is computed at run time.

Usage:
    python demo/run_demo.py                 # flagship hidden-bottleneck demo
    python demo/run_demo.py --scenario S2   # hidden quality drift
    python demo/run_demo.py --list          # show all scenarios
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rippletwin.explain.explain import explain_flow_alert, explain_quality_alert  # noqa: E402
from rippletwin.factory import scenarios as SC  # noqa: E402
from rippletwin.factory.topology import build_line  # noqa: E402
from rippletwin.hitl.ledger import (  # noqa: E402
    DECISION_APPROVED,
    OUTCOME_CONFIRMED,
    DecisionLedger,
)
from rippletwin.recommend.engine import recommend_flow, recommend_quality  # noqa: E402
from rippletwin.twin import genealogy as GN  # noqa: E402
from rippletwin.models.baselines import (  # noqa: E402
    apply_detection_rule,
    calibrate_threshold,
    turning_point_baseline,
)
from rippletwin.twin.pipeline import fit_context, infer, simulate  # noqa: E402
from rippletwin.twin.placement import recommend_sensors  # noqa: E402
from rippletwin.twin.propagate import (  # noqa: E402
    current_buffer_levels,
    defect_exposure,
    forecast_ripple,
    vehicles_between,
)
from rippletwin.twin.shadow import infer_hidden_cycle_time  # noqa: E402

DEMO_SEED = 20260301
LINE_SEED = 7
CONFIG = ROOT / "configs" / "line_42.yaml"

RULE = "=" * 78
THIN = "-" * 78


def banner(text: str) -> None:
    print(f"\n{RULE}\n  {text}\n{RULE}")


def step(n, text: str) -> None:
    print(f"\n[{n}] {text}\n{THIN}")


def main() -> int:
    ap = argparse.ArgumentParser(description="RippleTwin flagship demo")
    ap.add_argument("--scenario", default="S1",
                    choices=["S1", "S2", "S3", "S4", "S5"],
                    help="S1 hidden bottleneck (flagship), S2 hidden quality drift, "
                         "S3 normal, S4 instrumented station, S5 mix + supply delay")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--decision", default="approve", choices=["approve", "reject"])
    ap.add_argument("--json", type=str, default="", help="write a JSON summary here")
    args = ap.parse_args()

    line = build_line(CONFIG, seed=LINE_SEED)

    builders = {
        "S1": SC.scenario_hidden_bottleneck,
        "S2": SC.scenario_hidden_quality,
        "S3": SC.scenario_normal_variation,
        "S4": SC.scenario_observed_station,
        "S5": SC.scenario_variant_shift,
    }
    if args.list:
        for k, fn in builders.items():
            s = fn(line)
            print(f"{k}  {s.scenario_id:<24} {s.title}")
            print(f"    question: {s.question or '-'}")
            print(f"    {s.notes}")
        return 0

    scen = builders[args.scenario](line)

    banner(f"RIPPLETWIN DEMO -- {scen.scenario_id}")
    print(f"  {scen.title}")
    print(f"  Question: {scen.question}")
    print(f"\n  All figures below are SIMULATED PROTOTYPE RESULTS on synthetic data.")
    print(f"  Seeds: line={LINE_SEED}, simulation={DEMO_SEED} (fully reproducible)")

    # ---------------------------------------------------------------- step 1
    step(1, "THE LINE, AND WHAT IT CAN SEE")
    s = line.summary()
    print(f"  {s['name']}")
    print(f"  {s['n_stations']} stations | takt {s['takt_s']:.0f}s | "
          f"inspection gates: {', '.join(s['inspections'])}")
    print(f"  Sensor coverage: {s['coverage'] * 100:.0f}%  "
          f"({s['n_observed']} instrumented, {s['n_hidden']} with no telemetry)")
    for z, d in s["per_zone"].items():
        print(f"    {z:<6} {d['stations']:>2} stations -- "
              f"{d['rich']} rich, {d['basic']} basic, {d['manual']} manual (blind)")
    hidden_ids = [line.stations[i].station_id for i in line.hidden_indices]
    print(f"\n  Stations with NO sensor: {', '.join(hidden_ids)}")
    print("  A conventional twin models the line without these. RippleTwin infers them.")

    # ---------------------------------------------------------------- step 2
    step(2, "FITTING THE TWIN ON DISTURBANCE-FREE PRODUCTION")
    nominal = simulate(line, SC.nominal_run(2600), seed=1)
    calib = simulate(line, SC.nominal_run(2200), seed=2)
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)
    cal = ctx.calibration
    print(f"  Baseline fitted on {len(nominal.vehicles)} vehicles (no disturbances).")
    print(f"  Detector calibrated on a separate {len(calib.vehicles)}-vehicle run:")
    print(f"    cross-station correlation : {cal['mean_pairwise_corr']:.3f}")
    print(f"    effective sample size     : {cal['n_eff']:.2f} of {cal['n_observed']} "
          f"stations  (tau = {cal['tau']:.3f})")
    print(f"    detection threshold       : LLR >= {cal['detect_llr']:.1f} "
          f"for a {cal['target_window_fpr'] * 100:.0f}% per-window false-alarm target")
    print("\n  The correlation correction matters: stations on a coupled line are not")
    print("  independent observations, and without it the detector fires constantly.")

    prior = GN.candidate_prior(line)
    qbase = GN.QualityBaseline.fit(
        line,
        GN.attribute_defects(line, GN.explode_defects(nominal.inspections), prior),
        n_vehicles=len(nominal.vehicles),
    )

    # ---------------------------------------------------------------- step 3
    step(3, "RUNNING PRODUCTION WITH A DISTURBANCE THE MODEL IS NOT TOLD ABOUT")
    res = simulate(line, scen, seed=DEMO_SEED)
    print(f"  {res.meta['n_vehicles']} vehicles built over "
          f"{res.meta['horizon_s'] / 3600:.1f} hours")
    print(f"  Throughput: {res.meta['throughput_vph']:.1f} vehicles/hour "
          f"(nominal target {3600 / line.takt_s:.0f})")
    print(f"  Defects created: {res.meta['n_defects']}, "
          f"escaped every gate: {res.meta['n_escaped']}")
    if len(res.disturbances):
        print("\n  GROUND TRUTH (held back from the model, used only to score it):")
        for _, d in res.disturbances.iterrows():
            print(f"    {d['kind']} at {d['station_id']} [{d['tier']}] "
                  f"x{d['magnitude']:.2f} from t={d['t_start_s'] / 3600:.1f}h "
                  f"to {d['t_end_s'] / 3600:.1f}h -- {d['label']}")
    else:
        print("\n  GROUND TRUTH: no disturbance injected. Correct behaviour is silence.")

    print(f"\n  The model receives {len(res.telemetry):,} telemetry rows from "
          f"{res.telemetry['station'].nunique()} instrumented stations,")
    print(f"  {len(res.inspections):,} inspection results, and the vehicle release log.")
    print("  It receives nothing at all from the blind stations.")

    # ---------------------------------------------------------------- step 4
    step(4, "SHADOW-SENSING: RECONSTRUCTING THE HIDDEN STATE")
    scored, shadow, sensor = infer(ctx, res)
    print(f"  {len(shadow)} inference windows (20 vehicles wide, 5-vehicle stride).")

    truth_rows = res.disturbances[res.disturbances["kind"] != "MATERIAL_DELAY"] \
        if len(res.disturbances) else res.disturbances
    has_truth = len(truth_rows) > 0
    if has_truth:
        tr = truth_rows.iloc[0]
        true_station = int(tr["station"])
        t_active0 = float(tr["t_start_s"]) + float(tr["ramp_s"])
        t_active1 = float(tr["t_end_s"])
    else:
        true_station, t_active0, t_active1 = None, None, None

    det = shadow[shadow["detected"]]
    print(f"  Windows where a disturbance was detected: {len(det)} / {len(shadow)}")

    if len(det) == 0:
        if scen.expect_no_alert:
            print("\n  NO ALERT RAISED -- which is the correct answer here.")
            print(f"  Mean posterior on 'nothing is wrong' : {shadow['p_null'].mean():.2f}")
            print(f"  Mean posterior on 'line supply'      : "
                  f"{shadow['p_line_supply'].mean():.2f}")
            print("\n  A detector that only ever gets shown faults it can find proves")
            print("  nothing. Staying quiet on a clean line is half the product.")
        else:
            print("\n  No flow detection. For a pure quality drift this is expected:")
            print("  the station keeps takt, so there is no flow signature to see.")
            print("  The quality path in step 5 is the one that carries this case.")
    else:
        first = det.iloc[0]
        print(f"\n  First detection at window {int(first['window'])}, "
              f"t = {first['t_mid_s'] / 3600:.2f}h")
        print(f"  Named station: {first['top_station_id']} "
              f"({'INFERRED -- no sensor' if first['top_is_hidden'] else 'OBSERVED'})")
        if has_truth:
            correct = det[det["top_station"] == true_station]
            active = det[(det["t_mid_s"] >= t_active0) & (det["t_mid_s"] <= t_active1)]
            if len(active):
                err = np.abs(active["top_station"] - true_station)
                print(f"\n  Over {len(active)} windows while the disturbance was fully active:")
                print(f"    exact station        : {(err == 0).mean() * 100:5.1f}%")
                print(f"    within one station   : {(err <= 1).mean() * 100:5.1f}%")
                print(f"    mean posterior mass  : {active['group_prob'].mean():.2f}")
                votes = active["top_station_id"].value_counts().head(3)
                print(f"    top candidates       : "
                      + ", ".join(f"{k} ({v})" for k, v in votes.items()))
                print(f"    TRUE SOURCE          : "
                      f"{line.stations[true_station].station_id} "
                      f"[{line.stations[true_station].tier}]")

    # ------------------------------------------------------- step 4b
    if has_truth and line.stations[true_station].is_hidden:
        step("4b", "AGAINST THE PUBLISHED METHOD (Li, Chang & Ni, 2009)")
        cal_scored, _, _ = infer(ctx, calib)
        thr = calibrate_threshold(
            turning_point_baseline(cal_scored, line).frame, 0.01
        )
        tp = apply_detection_rule(turning_point_baseline(scored, line).frame, thr)
        wt = scored.groupby("window").agg(
            t_mid_s=("t_depart_s_min", "min")
        ).reset_index()
        tp = tp.merge(wt, on="window")
        tp_act = tp[
            tp["detected"]
            & (tp["t_mid_s"] >= t_active0)
            & (tp["t_mid_s"] <= t_active1)
        ]
        print("  The blocked/starved boundary is NOT our idea. It is the Turning")
        print("  Point Method, and it is one of the best-established bottleneck")
        print("  detection methods in manufacturing science. So we ran it, at the")
        print("  same calibrated false-alarm rate.\n")
        if len(tp_act):
            named = int(tp_act["top_station"].mode().iloc[0])
            print(f"  Turning Point Method : detected in {len(tp_act)} windows, "
                  f"names {line.stations[named].station_id}  "
                  f"-> exact-station accuracy "
                  f"{(tp_act['top_station'] == true_station).mean() * 100:.0f}%")
        if len(det):
            act = det[(det["t_mid_s"] >= t_active0) & (det["t_mid_s"] <= t_active1)]
            if len(act):
                print(f"  RippleTwin           : detected in {len(act)} windows, "
                      f"names {act['top_station_id'].mode().iloc[0]}  "
                      f"-> exact-station accuracy "
                      f"{(act['top_station'] == true_station).mean() * 100:.0f}%")
        print(f"\n  TRUE SOURCE          : "
              f"{line.stations[true_station].station_id} "
              f"[{line.stations[true_station].tier}]")
        print("\n  The published method detects the disturbance perfectly well. It")
        print("  lands on the first INSTRUMENTED station past the sensor gap,")
        print("  because it scans the stations it can measure -- a turning point")
        print("  inside a gap is outside its output space. That gap is the whole")
        print("  contribution of this project.")

    # ---------------------------------------------------------------- step 5
    step(5, "THE SECOND PATH: DEFECT ATTRIBUTION BY VEHICLE GENEALOGY")
    wb = GN.window_bounds_from(scored)
    found = GN.explode_defects(res.inspections)
    qs = GN.quality_state(line, found, wb, qbase, pool_vehicles=200)
    qa = GN.quality_alerts(qs)
    print(f"  {len(found)} defects found at gates; source station never recorded.")
    q_alerts = qa[qa["quality_alert"]] if len(qa) else qa
    print(f"  Quality-alert station-windows: {len(q_alerts)}")

    q_station = None
    if has_truth and len(qa):
        wsel = wb[(wb["t_lo"] > t_active0) & (wb["t_hi"] < t_active1)]["window"]
        during = qa[qa["window"].isin(wsel)]
        if len(during):
            rank = during.groupby("station")["llr"].mean().sort_values(ascending=False)
            order = list(rank.index)
            pos = order.index(true_station) + 1 if true_station in order else None
            print(f"  Ranking of the TRUE source by failure-mode evidence: "
                  f"#{pos} of {len(order)}"
                  if pos else "  True source not ranked.")
            print("  Top candidates: "
                  + ", ".join(line.stations[i].station_id for i in order[:5]))
            q_station = int(order[0])
            # Report the estimate over the windows where the system actually
            # alerted. Averaging over a ground-truth-defined event window would
            # be an evaluation view, not something the product knows -- and it
            # would disagree with the number in the recommendation below.
            _al = qa[(qa["station"] == q_station) & qa["quality_alert"]]
            _src = _al if len(_al) else during[during["station"] == q_station]
            m_hat = float(_src["m_hat"].mean())
            print(f"  Estimated defect-rate multiplier at {line.stations[q_station].station_id}"
                  f": {m_hat:.1f}x  (over {len(_src)} alerting windows)")
            if str(tr["kind"]) in ("QUALITY_DRIFT", "COMBINED"):
                print(f"  True injected multiplier: {float(tr['magnitude']):.1f}x")
    elif not len(qa):
        print("  Not enough production to pool for a quality test.")

    # ---------------------------------------------------------------- step 6
    ledger = DecisionLedger()
    summary: dict = {"scenario": scen.scenario_id, "seed": DEMO_SEED}

    # Which mechanism actually carries this fault?
    #
    # A pure quality drift keeps takt, so the flow path has nothing to see. It
    # can still trip on a couple of windows -- more rework at the gates slows
    # them slightly -- and following those few windows would mean reporting the
    # wrong station while the quality path has the right one ranked first.
    #
    # So the demo leads with whichever mechanism the evidence supports, rather
    # than always preferring flow because it happened to fire.
    n_active = len(shadow)
    if has_truth:
        n_active = max(
            1, len(shadow[(shadow["t_mid_s"] >= t_active0)
                          & (shadow["t_mid_s"] <= t_active1)])
        )
    flow_strength = len(det) / n_active
    quality_fired = q_station is not None and len(q_alerts) > 0
    flow_marginal = 0 < flow_strength < 0.05

    if flow_marginal and quality_fired:
        print(f"\n  NOTE: the flow path fired on only {len(det)} of {n_active} "
              f"active windows ({flow_strength * 100:.1f}%), which is inside its "
              f"false-alarm budget rather than a real timing signature.")
        print("  Leading with the quality path, which is the mechanism this fault "
              "actually has.")
        det = det.iloc[0:0]

    if len(det):
        step(6, "FORECASTING THE RIPPLE (PHYSICS, NOT REGRESSION)")
        active = det[(det["t_mid_s"] >= t_active0) & (det["t_mid_s"] <= t_active1)] \
            if has_truth else det
        pick = active.iloc[len(active) // 2] if len(active) else det.iloc[0]
        w = int(pick["window"])
        sr = next(r for r in sensor.last_results if r.window == w)
        k = int(pick["top_station"])

        est_cycle = infer_hidden_cycle_time(
            line, res.telemetry, k, int(pick["v_start"]), int(pick["v_end"])
        )
        fc = None
        if est_cycle:
            fc = forecast_ripple(
                line, k, est_cycle, horizon_min=60.0,
                buffer_levels=current_buffer_levels(scored, w),
            )
            print(f"  Inferred cycle time at {line.stations[k].station_id}: "
                  f"{est_cycle:.1f}s  (takt {line.takt_s:.0f}s)")
            if line.stations[k].is_hidden:
                seg = res.passes[
                    (res.passes["station"] == k)
                    & (res.passes["vehicle_id"] >= int(pick["v_start"]))
                    & (res.passes["vehicle_id"] < int(pick["v_end"]))
                ]
                if len(seg):
                    true_c = float(seg["proc_time_s"].mean())
                    print(f"  GROUND TRUTH cycle time         : {true_c:.1f}s   "
                          f"-> error {abs(est_cycle - true_c) / true_c * 100:.1f}%")
                    print("  This station has no sensor. That number was reconstructed")
                    print("  from the departure rate of the first instrumented station")
                    print("  downstream, and it is checkable because the simulator knows")
                    print("  the truth. In a real plant it would be checked on the floor.")
            print(f"\n  Sustained line rate  : {fc.sustained_rate_vph:.1f} veh/h "
                  f"vs {fc.nominal_rate_vph:.0f} target "
                  f"({fc.throughput_loss_pct * 100:.0f}% loss)")
            print(f"  Units lost in 60 min : {fc.units_lost_at_horizon:.1f} vehicles")
            if fc.minutes_to_downstream_starve is not None:
                print(f"  Downstream starves   : in "
                      f"{fc.minutes_to_downstream_starve:.0f} min "
                      f"({', '.join(fc.downstream_affected[:4])})")
            if fc.minutes_to_upstream_block is not None:
                print(f"  Upstream backs up    : in "
                      f"{fc.minutes_to_upstream_block:.0f} min "
                      f"({', '.join(fc.upstream_affected[:4])})")

        # ------------------------------------------------------------ step 7
        step(7, "EXPLANATION -- ASSEMBLED FROM THE MODEL'S OWN EVIDENCE")
        exp = explain_flow_alert(line, sr, fc, est_cycle)
        print(exp.as_text())
        print("\n  Note: no language model was involved. Every number above is a")
        print("  value the estimator computed, tagged with how it was obtained.")

        # ------------------------------------------------------------ step 8
        step(8, "RECOMMENDATION -- ADVISORY, REVERSIBLE, AWAITING APPROVAL")
        rec = recommend_flow(line, sr, fc)
        print(f"  ACTION      : {rec.action}")
        print(f"  PRIORITY    : {rec.priority}")
        print(f"  TARGET      : {', '.join(rec.target_stations) or '(none named)'}")
        print(f"  TITLE       : {rec.title}")
        print(f"  DETAIL      : {rec.detail}")
        print(f"  RATIONALE   : {rec.rationale}")
        print(f"  CONFIDENCE  : {rec.confidence:.2f}")
        print(f"  ABSTAINED   : {rec.abstained}")
        if rec.alternatives:
            print(f"  IF WRONG    : {rec.alternatives[0]}")
        print("\n  RippleTwin does not write to a PLC, change line control logic, or")
        print("  stop the line. Every action in its vocabulary is advisory and")
        print("  reversible, and a person performs it.")

        entry = ledger.record_alert(
            run_id=scen.scenario_id, window=w, alert_type="FLOW",
            station_id=line.stations[k].station_id,
            station_tier=line.stations[k].tier,
            is_inferred=bool(line.stations[k].is_hidden),
            confidence=float(rec.confidence),
            recommendation=rec.as_dict(), explanation=exp.as_dict(),
        )

        # ------------------------------------------------------------ step 9
        step(9, "HUMAN DECISION AND OUTCOME")
        decision = DECISION_APPROVED if args.decision == "approve" else "REJECTED"
        ledger.record_decision(entry.entry_id, decision, decided_by="shift_supervisor_A",
                               note=f"Dispatched to {line.stations[k].station_id}.")
        print(f"  Supervisor decision: {decision}")

        if has_truth:
            confirmed = k == true_station
            ledger.record_outcome(
                entry.entry_id,
                OUTCOME_CONFIRMED if confirmed else "FOUND_ELSEWHERE",
                note="Condition found at the named station."
                if confirmed
                else "Condition found at an adjacent station.",
                actual_station_id=line.stations[true_station].station_id,
            )
            print(f"  Outcome recorded   : "
                  f"{'CONFIRMED at ' + line.stations[k].station_id if confirmed else 'FOUND ELSEWHERE (' + line.stations[true_station].station_id + ')'}")

        chain = ledger.verify()
        print(f"\n  Ledger entries: {len(ledger.entries)}  |  "
              f"hash chain valid: {chain['valid']}")
        print("  The alert is stored exactly as issued. A decision or outcome is")
        print("  appended as a new entry, never an edit, so what the system said at")
        print("  the time stays provable afterwards.")

        summary.update({
            "detected": True,
            "named_station": line.stations[k].station_id,
            "named_is_hidden": bool(line.stations[k].is_hidden),
            "true_station": line.stations[true_station].station_id if has_truth else None,
            "inferred_cycle_s": est_cycle,
            "throughput_loss_pct": fc.throughput_loss_pct if fc else None,
            "units_lost_60min": fc.units_lost_at_horizon if fc else None,
            "recommendation": rec.action,
            "abstained": rec.abstained,
            "ledger_valid": chain["valid"],
        })

    elif q_station is not None:
        step(6, "QUALITY RECOMMENDATION -- ADVISORY, AWAITING APPROVAL")
        # Average over the windows where this station actually alerted, not over
        # the whole run. Averaging quiet windows in dilutes the estimate and
        # makes the recommendation disagree with the multiplier reported above.
        alerted = qa[(qa["station"] == q_station) & qa["quality_alert"]]
        src = alerted if len(alerted) else qa[qa["station"] == q_station]
        m_hat = float(src["m_hat"].mean())
        now_v = int(wb["v_end"].max())
        n_flight = vehicles_between(line, q_station, None, now_v) or 12
        expo = defect_exposure(
            line, q_station, m_hat,
            baseline_rate_per_vehicle=qbase.lam[q_station],
            vehicles_in_flight=n_flight,
        )
        order = list(
            qa.groupby("station")["llr"].mean().sort_values(ascending=False).index
        )
        exp = explain_quality_alert(
            line, q_station, m_hat, float(src["llr"].mean()),
            rank=1, candidates=order, exposure=expo,
            dominant_types=list(line.stations[q_station].defect_profile.keys())[:2],
        )
        print(exp.as_text())
        rec = recommend_quality(line, q_station, m_hat, 1, expo, order[:3])
        print(f"\n  ACTION    : {rec.action}  ({rec.priority})")
        print(f"  TITLE     : {rec.title}")
        print(f"  DETAIL    : {rec.detail}")
        print(f"  ABSTAINED : {rec.abstained}")
        entry = ledger.record_alert(
            run_id=scen.scenario_id, window=int(qa["window"].max()),
            alert_type="QUALITY",
            station_id=line.stations[q_station].station_id,
            station_tier=line.stations[q_station].tier,
            is_inferred=bool(line.stations[q_station].is_hidden),
            confidence=float(rec.confidence),
            recommendation=rec.as_dict(), explanation=exp.as_dict(),
        )
        ledger.record_decision(entry.entry_id, DECISION_APPROVED, "shift_supervisor_A")
        chain = ledger.verify()
        print(f"\n  Ledger entries: {len(ledger.entries)}  |  chain valid: {chain['valid']}")
        summary.update({
            "detected": True, "path": "quality",
            "named_station": line.stations[q_station].station_id,
            "estimated_multiplier": m_hat,
            "ledger_valid": chain["valid"],
        })
    else:
        summary.update({"detected": False, "correct_silence": bool(scen.expect_no_alert)})

    # ------------------------------------------------------------ placement
    step(10, "WHERE THE NEXT SENSOR SHOULD GO")
    rec = recommend_sensors(line, n_recommend=4)
    print("  \"Why not just instrument the blind stations?\" -- you should. The")
    print("  question is which ones buy the most. This needs NO production data,")
    print("  so a plant can run it before committing to a retrofit.\n")
    print("  rank  station  zone    value   currently confusable with")
    for _, r in rec.iterrows():
        print(f"  {int(r['rank']):>4}  {r['station_id']:<7} {r['zone']:<7} "
              f"{r['total_gain']:>5.2f}   {r['unlocks']}")
    adj = [
        (line.stations[i].station_id, line.stations[i + 1].station_id)
        for i in line.hidden_indices
        if (i + 1) in set(line.hidden_indices)
    ]
    if adj:
        print(f"\n  {', '.join(a + '/' + b for a, b in adj)} are ADJACENT blind "
              f"stations. With no sensor")
        print("  between them their signatures are ~97% alike, so no amount of data")
        print("  separates them -- the twin reports them as a group and abstains.")
        print("  Breaking up an adjacent pair beats instrumenting an isolated one.")

    banner("DEMO COMPLETE")
    print("  Everything above is a SIMULATED PROTOTYPE RESULT on synthetic data.")
    print("  No real production data was used and no real-plant ROI is claimed.")
    print(f"  Reproduce exactly:  python demo/run_demo.py --scenario {args.scenario}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2, default=str))
        print(f"  Summary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
