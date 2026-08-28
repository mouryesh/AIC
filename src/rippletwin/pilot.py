"""One command a plant runs on its own data.

    python -m rippletwin.pilot --export path/to/mapping.yaml

Why this module is the difference between a demo and a pilot
------------------------------------------------------------
Everything else in this repository can be evaluated only by us, on our
simulator, on our terms. That is the position almost every industrial analytics
prototype is in when it is presented, and it is why so many of them never get
past the presentation: a plant cannot check the claim without first funding an
integration.

This inverts that. A plant exports data it already has, fills in a mapping file
with its own column names, and runs one command. It gets back:

1. **A data verdict** -- is this export even usable, and what is wrong with it
2. **A topology proposal** -- inferred from the data, with the assumptions listed
3. **A Phase 0 capability verdict** -- FULL / FLOW_ONLY / QUALITY_ONLY / NOT_VIABLE
4. **Findings** -- which blind stations the twin flags, over the scored period
5. **Work orders** -- owned, time-bounded, with the cost of waiting

No credentials, no network, no OT change, no port opened. A file drop is
deliberately the first integration step because it is the one a plant can
authorise without a project.

The honest limitation, stated up front
--------------------------------------
The twin needs a **known-normal period** to learn what normal looks like, and
an export does not label which of its hours were normal. By default this splits
the export chronologically and assumes the earliest slice is representative.
That assumption is stated in the report every time, and ``--baseline-vehicles``
lets a plant name a period it knows was clean. Getting this wrong does not
produce a wrong answer quietly -- it produces a baseline with a fault already
inside it, which *suppresses* the signal rather than inventing one. Erring
toward missing a fault rather than crying wolf is the right direction here,
because false alarms are the documented way these systems lose the floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import List, Optional

import numpy as np
import pandas as pd

from .factory.topology import LineTopology
from .features.windows import WindowSpec
from .ingest.csv_adapter import TEMPLATE_YAML, PlantExportSpec, load_plant_export
from .ingest.plant_data import PlantData
from .ingest.topology_infer import infer_line
from .integrate.contract import assess_readiness
from .recommend.dispatch import to_work_order
from .recommend.engine import recommend_flow
from .twin.pipeline import fit_context, infer
from .twin.shadow import infer_hidden_cycle_time
from .twin.propagate import forecast_ripple

BAR = "=" * 74


def _rule(title: str) -> str:
    return f"\n{BAR}\n{title}\n{BAR}"


def run_pilot(
    export_yaml: str,
    baseline_frac: float = 0.4,
    calib_frac: float = 0.2,
    baseline_vehicles: Optional[str] = None,
    out_dir: Optional[str] = None,
    max_findings: int = 20,
) -> dict:
    """Run the whole Phase 0 assessment on a plant export. Returns a summary."""
    spec = PlantExportSpec.from_yaml(export_yaml)
    report: List[str] = []
    summary: dict = {"export": export_yaml, "plant_name": spec.plant_name}

    report.append(_rule("RIPPLETWIN PILOT ASSESSMENT"))
    report.append(f"plant   : {spec.plant_name or '(unnamed)'}")
    report.append(f"export  : {spec.root}")

    # ---------------------------------------------------------- 1. load ----
    data = load_plant_export(spec)
    report.append(
        f"loaded  : {len(data.telemetry):,} telemetry rows, "
        f"{data.telemetry['station'].nunique()} instrumented stations, "
        f"{data.n_vehicles:,} units"
    )

    # ------------------------------------------------------ 2. validate ----
    val = data.validate(
        n_stations=spec.n_stations,
        takt_s=spec.takt_s,
        clock_sync_s=spec.clock_sync_s,
    )
    report.append(_rule("1. DATA QUALITY"))
    report.append(val.summary())
    summary["data_ok"] = val.ok
    summary["data_issues"] = [asdict(i) for i in val.issues]
    summary["data_stats"] = val.stats

    if not val.ok:
        report.append(
            "\nSTOPPING: the export cannot be used as-is. Every blocker above is "
            "a data problem, not a modelling one -- fix them and re-run."
        )
        return _finish(report, summary, out_dir)

    # ------------------------------------------------------ 3. topology ----
    line, assumptions = infer_line(
        data, n_stations=spec.n_stations, takt_s=spec.takt_s,
        name=spec.plant_name or "plant-line",
        station_order=spec.station_order,
    )
    if spec.station_order:
        report.append("")
    else:
        assumptions.insert(0, (
            "No full station list was supplied. Add 'line.stations' to the "
            "mapping file -- an ordered list of EVERY station, instrumented or "
            "not. It is the highest-value thing you can give us: without it the "
            "blind stations are placed by guesswork, and in our own testing that "
            "guess moved a localisation twelve stations away from the truth."
        ))
    report.append(_rule("2. INFERRED LINE TOPOLOGY"))
    report.append(
        f"  stations         : {line.n_stations} "
        f"({len(line.observed_indices)} instrumented, "
        f"{len(line.hidden_indices)} blind)"
    )
    report.append(f"  coverage         : {line.coverage:.0%}")
    report.append(f"  takt             : {line.takt_s:.1f}s")
    report.append("\n  ASSUMPTIONS AN ENGINEER MUST CONFIRM:")
    for a in assumptions:
        report.append(f"    - {a}")
    summary["topology"] = {
        "n_stations": line.n_stations,
        "n_observed": len(line.observed_indices),
        "coverage": round(line.coverage, 3),
        "takt_s": line.takt_s,
        "assumptions": assumptions,
    }

    # ----------------------------------------------------- 4. readiness ----
    readiness = assess_readiness(
        data.contract_signals(),
        n_stations=line.n_stations,
        n_stations_with_state=len(line.observed_indices),
        clock_sync_s=spec.clock_sync_s,
    )
    report.append(_rule("3. PHASE 0 CAPABILITY VERDICT"))
    report.append(readiness.summary())
    summary["capability"] = str(readiness.capability)

    if readiness.blockers:
        report.append(
            "\nSTOPPING: capability blockers above must be resolved before the "
            "twin can run."
        )
        return _finish(report, summary, out_dir)

    # ------------------------------------------------- 5. fit and score ----
    n_veh = data.n_vehicles
    if baseline_vehicles:
        a, b = (int(x) for x in baseline_vehicles.split(":"))
        base_lo, base_hi = a, b
        cal_lo, cal_hi = b, min(n_veh, b + int((b - a) * 0.5))
        named = True
    else:
        base_hi = int(n_veh * baseline_frac)
        cal_hi = int(n_veh * (baseline_frac + calib_frac))
        base_lo, cal_lo, named = 0, base_hi, False

    report.append(_rule("4. FITTING"))
    report.append(
        f"  baseline period  : units {base_lo:,}-{base_hi:,}"
        f"{'  (named by the plant)' if named else '  (assumed representative)'}"
    )
    report.append(f"  calibration      : units {cal_lo:,}-{cal_hi:,}")
    report.append(f"  scored           : units {cal_hi:,}-{n_veh:,}")
    if not named:
        report.append(
            "\n  NOTE: no known-good period was named, so the earliest slice of "
            "the export\n  is assumed normal. If a fault was already present "
            "then, it is absorbed into\n  the baseline and will NOT be flagged. "
            "Re-run with --baseline-vehicles A:B\n  naming a period the plant "
            "knows was clean."
        )

    nominal = _slice(data, base_lo, base_hi)
    calib = _slice(data, cal_lo, cal_hi)
    scored_data = _slice(data, cal_hi, n_veh)

    spec_w = WindowSpec.for_line(line)
    try:
        ctx = fit_context(line, nominal, calibration_run=calib, spec=spec_w)
        scored, shadow, sensor = infer(ctx, scored_data)
    except ValueError as exc:
        report.append(
            f"\nSTOPPING: {exc}\n  The export is too short for this line. A window "
            f"is {spec_w.width} units with a\n  warm-up of {spec_w.warmup}; supply "
            f"at least a few thousand units."
        )
        return _finish(report, summary, out_dir)

    report.append(f"  windows scored   : {len(shadow)}")
    report.append(f"  detection thresh : LLR {ctx.shadow_cfg.detect_llr:.1f} "
                  f"(calibrated to 1% per-window false alarms)")

    # -------------------------------------------------------- 6. results ---
    fired = shadow[shadow["detected"]].copy()
    hidden = fired[fired["top_is_hidden"]]
    report.append(_rule("5. FINDINGS"))
    report.append(f"  windows flagged        : {len(fired)} of {len(shadow)} "
                  f"({len(fired) / max(1, len(shadow)):.1%})")
    report.append(f"  naming a BLIND station : {len(hidden)}")
    summary["n_windows"] = len(shadow)
    summary["n_flagged"] = int(len(fired))
    summary["n_blind_station_findings"] = int(len(hidden))

    if fired.empty:
        report.append(
            "\n  No constraint exceeded the detection threshold over the scored "
            "period.\n  On a healthy line that is the correct output."
        )
    else:
        top = (
            fired.groupby(["top_station", "top_station_id", "top_is_hidden"])
            .agg(windows=("window", "count"), mean_llr=("llr", "mean"),
                 mean_prob=("top_prob", "mean"))
            .reset_index()
            .sort_values("windows", ascending=False)
            .head(max_findings)
        )
        report.append("\n  station        blind   windows   mean LLR   mean p")
        report.append("  " + "-" * 54)
        for r in top.itertuples():
            report.append(
                f"  {str(r.top_station_id):<14} {'YES' if r.top_is_hidden else ' - ':<7}"
                f"{r.windows:>7}   {r.mean_llr:>8.1f}   {r.mean_prob:>6.2f}"
            )
        summary["findings"] = top.to_dict("records")

    # ---------------------------------------------------- 7. work orders ---
    report.append(_rule("6. WORK ORDERS"))
    orders = _work_orders(line, sensor, shadow, scored, scored_data.telemetry)
    if not orders:
        report.append("  None raised. Monitor-only advice does not create a job.")
    for i, wo in enumerate(orders[:5], 1):
        report.append(f"\n  [{i}] {wo.work_order_id}  {wo.title}")
        report.append(f"      owner      : {wo.owner_role}")
        report.append(f"      respond by : {wo.respond_within_min} min")
        if wo.waiting_cost:
            report.append(f"      waiting    : {wo.waiting_cost['rationale']}")
        report.append(f"      verify     : {wo.verification}")
    summary["n_work_orders"] = len(orders)
    summary["work_orders"] = [w.as_cmms_payload() for w in orders[:20]]

    report.append(_rule("NEXT STEP"))
    report.append(
        "  Phase 1 is shadow mode: run this weekly, log every finding, and have\n"
        "  a technician record found / not found against each one. After 8-12\n"
        "  weeks you have per-station precision measured on YOUR line, which is\n"
        "  the only number that should decide whether this goes live."
    )
    return _finish(report, summary, out_dir)


def _slice(data: PlantData, lo: int, hi: int) -> PlantData:
    """Restrict an export to a vehicle-index range, preserving absent inputs."""
    tel = data.telemetry
    tel = tel[(tel["vehicle_id"] >= lo) & (tel["vehicle_id"] < hi)].copy()
    veh = data.vehicles
    veh = veh[(veh["vehicle_id"] >= lo) & (veh["vehicle_id"] < hi)].copy()
    # Windowing indexes vehicles from zero, so rebase.
    tel["vehicle_id"] -= lo
    veh["vehicle_id"] -= lo
    insp = None
    if data.has_quality_path:
        insp = data.inspections
        insp = insp[(insp["vehicle_id"] >= lo) & (insp["vehicle_id"] < hi)].copy()
        insp["vehicle_id"] -= lo
    return PlantData(
        telemetry=tel, vehicles=veh, inspections=insp,
        environment=data.environment if data.has_environment else None,
        meta=data.meta,
    )


def _work_orders(line: LineTopology, sensor, shadow: pd.DataFrame,
                 scored: pd.DataFrame, telemetry: pd.DataFrame) -> list:
    """Raise a work order for each distinct confident finding."""
    out = []
    seen = set()
    # ``last_results`` is a list in window order, not a dict -- index it rather
    # than probing for an attribute that does not exist. An earlier version used
    # a guarded ``sensor.results.get(...)``, which silently produced zero work
    # orders on every run: the guard turned a missing attribute into "no
    # findings" instead of an error.
    by_window = {r.window: r for r in getattr(sensor, "last_results", [])}
    # A detection that localises to a station but falls below the confidence
    # floor still deserves a job -- the recommender turns it into a zone-level
    # investigation rather than a station dispatch, and downgrades its urgency.
    fired = shadow[shadow["detected"]]
    for row in fired.itertuples():
        key = int(row.top_station)
        if key in seen:
            continue
        seen.add(key)
        res = by_window.get(int(row.window))
        if res is None:
            continue
        # forecast_ripple takes the CONSTRAINT'S CYCLE TIME IN SECONDS, not a
        # rate. Passing units/hour here made every forecast non-binding (46 is
        # less than a 60s takt, so the line looked healthy) and silently
        # suppressed every work order.
        #
        # The right input is the twin's own estimate of the blind station's
        # cycle time -- the headline capability, reused for the forecast rather
        # than recomputed. Where it cannot be estimated, fall back to the cycle
        # time implied by the window's achieved rate.
        cyc = infer_hidden_cycle_time(
            line, telemetry, int(row.top_station),
            int(row.v_start), int(row.v_end),
        )
        if cyc is None:
            rate = _observed_rate(scored, int(row.window), line)
            cyc = 3600.0 / max(rate, 1e-6)
        fc = forecast_ripple(line, int(row.top_station), float(cyc))
        rec = recommend_flow(line, res, fc)
        wo = to_work_order(line, rec, fc, sequence=len(out) + 1,
                           source_alert={"entry_id": f"win-{int(row.window)}"})
        if wo is not None:
            out.append(wo)
    return out


def _observed_rate(scored: pd.DataFrame, window: int, line: LineTopology) -> float:
    """Units/hour the line actually achieved over one window.

    Measured per station, then taken across stations. Every instrumented
    station sees all of the window's units, so its own first-to-last departure
    span *is* the window's duration. Spanning from the first station's earliest
    departure to the last station's latest instead measures the time a unit
    takes to traverse the whole line, which on a 42-station line is about forty
    times too long -- it reported 5.7 units/hour against a takt of 60, so every
    forecast came back non-binding and no work order was ever raised.
    """
    g = scored[scored["window"] == window]
    if g.empty:
        return 3600.0 / line.takt_s
    n = float(g["v_end"].iloc[0] - g["v_start"].iloc[0])
    # n departures span n-1 intervals.
    intervals = max(n - 1.0, 1.0)
    span = float((g["t_depart_s_max"] - g["t_depart_s_min"]).median())
    return (intervals / span * 3600.0) if span > 0 else 3600.0 / line.takt_s


def _finish(report: List[str], summary: dict, out_dir: Optional[str]) -> dict:
    text = "\n".join(report)
    print(text)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "pilot_report.txt"), "w") as fh:
            fh.write(text + "\n")
        with open(os.path.join(out_dir, "pilot_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"\nwritten: {out_dir}/pilot_report.txt, {out_dir}/pilot_summary.json")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m rippletwin.pilot",
        description="Run RippleTwin's Phase 0 assessment on a plant's own export.",
    )
    p.add_argument("--export", help="path to the plant's mapping YAML")
    p.add_argument("--emit-template", metavar="PATH",
                   help="write a blank mapping file and exit")
    p.add_argument("--baseline-vehicles", metavar="A:B",
                   help="unit range known to be normal, e.g. 0:5000")
    p.add_argument("--baseline-frac", type=float, default=0.4)
    p.add_argument("--calib-frac", type=float, default=0.2)
    p.add_argument("--out", metavar="DIR", help="write report and JSON here")
    args = p.parse_args(argv)

    if args.emit_template:
        with open(args.emit_template, "w") as fh:
            fh.write(TEMPLATE_YAML)
        print(f"wrote {args.emit_template}")
        print("Fill in your column names, then run:")
        print(f"  python -m rippletwin.pilot --export {args.emit_template}")
        return 0

    if not args.export:
        p.error("one of --export or --emit-template is required")

    run_pilot(
        args.export,
        baseline_frac=args.baseline_frac,
        calib_frac=args.calib_frac,
        baseline_vehicles=args.baseline_vehicles,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
