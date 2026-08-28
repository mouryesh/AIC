#!/usr/bin/env python3
"""RippleTwin streaming demo -- a paced, window-by-window replay.

This is a REPLAY of a deterministic, already-computed simulation, paced for
a human to watch -- not a live connection to a factory. That distinction is
stated up front, on every line of output, because pretending otherwise
would be exactly the kind of thing docs/METHOD.md and README.md argue this
project should never do.

Every number printed was computed causally: `twin.predict.run_predictor`'s
state at window w depends only on windows <= w, exactly as it would in a
real deployment consuming telemetry as it arrives. The pacing is cosmetic;
the computation is not re-ordered to make the demo look better, and nothing
here is scripted for effect (matching demo/run_demo.py's own stated rule).

Usage:
    python demo/run_streaming_demo.py                    # ~90s replay of S6
    python demo/run_streaming_demo.py --scenario S1_HIDDEN_BOTTLENECK
    python demo/run_streaming_demo.py --duration 45       # faster replay
    python demo/run_streaming_demo.py --no-pace           # dump immediately (CI/testing)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rippletwin.factory import scenarios as SC  # noqa: E402
from rippletwin.factory.topology import build_line  # noqa: E402
from rippletwin.twin import predict as PR  # noqa: E402
from rippletwin.twin.pipeline import fit_context, infer, simulate  # noqa: E402
from rippletwin.recommend.engine import taxonomy_label, recommend_flow  # noqa: E402
from rippletwin.twin.propagate import current_buffer_levels, forecast_ripple  # noqa: E402
from rippletwin.twin.shadow import infer_hidden_cycle_time  # noqa: E402

CONFIG = ROOT / "configs" / "line_42.yaml"
LINE_SEED = 7
DEMO_SEED = 20260301

SCENARIOS = {
    "S1_HIDDEN_BOTTLENECK": SC.scenario_hidden_bottleneck,
    "S6_EARLY_WARNING": SC.scenario_gradual_bottleneck,
    "S7_MULTIPLE_ABNORMALITIES": SC.scenario_multiple_abnormalities,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="S6_EARLY_WARNING", choices=sorted(SCENARIOS))
    ap.add_argument("--duration", type=float, default=90.0, help="target wall-clock seconds for the replay")
    ap.add_argument("--no-pace", action="store_true", help="print immediately, no sleep (for CI/testing)")
    args = ap.parse_args()

    print("=" * 78)
    print("RIPPLETWIN -- STREAMING DEMO (paced replay of a deterministic simulation)")
    print("This is NOT a live factory connection. Every value below was computed")
    print("causally, window by window, from a fixed seed -- the pacing is cosmetic.")
    print("=" * 78)

    line = build_line(str(CONFIG), seed=LINE_SEED)
    nominal = simulate(line, SC.nominal_run(2600), seed=1)
    calib = simulate(line, SC.nominal_run(2200), seed=2)
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)

    scen = SCENARIOS[args.scenario](line)
    res = simulate(line, scen, seed=DEMO_SEED)
    scored, shadow, sensor = infer(ctx, res)
    pred = PR.run_predictor(shadow, line, res.telemetry, scored, ctx.shadow_cfg)

    n = len(pred)
    pace_s = max(0.0, args.duration / max(n, 1)) if not args.no_pace else 0.0
    print(f"\nScenario: {scen.scenario_id} -- {scen.title}")
    print(f"{n} windows, pacing at {pace_s:.2f}s/window (target {args.duration:.0f}s total)\n")

    prev_state = PR.STATE_NORMAL
    events = []
    t_wall0 = time.time()

    for _, prow in pred.iterrows():
        w = int(prow["window"])
        state = prow["state"]
        station_id = prow["station_id"]
        risk = float(prow["risk"])
        conf = float(prow["confidence"])
        tti = prow["time_to_impact_min"]
        t_sim_h = float(prow["t_mid_s"]) / 3600.0

        transitioned = state != prev_state
        # DEGRADING is deliberately loose (§ the state ladder's lowest bar)
        # and flickers on ordinary noise -- honestly reported per-window
        # below, but the event *timeline* only highlights crossings into or
        # out of WATCH-or-above, so the narrative summary is not drowned in
        # DEGRADING<->NORMAL noise. See docs/LIMITATIONS.md.
        notable = transitioned and (
            PR.state_rank(state) >= PR.state_rank(PR.STATE_WATCH)
            or PR.state_rank(prev_state) >= PR.state_rank(PR.STATE_WATCH)
        )
        marker = " <-- STATE CHANGE" if transitioned else ""
        tti_txt = f"{tti:.0f} min" if tti == tti else "n/a"  # NaN check without importing numpy here

        print(
            f"[t={t_sim_h:5.2f}h  w={w:4d}] state={state:20s} "
            f"station={station_id or '--':5s} risk={risk:.2f} conf={conf * 100:4.0f}% "
            f"time-to-impact={tti_txt}{marker}"
        )
        if notable:
            events.append((t_sim_h, prev_state, state, station_id))
        prev_state = state

        if pace_s > 0:
            time.sleep(pace_s)

    wall_s = time.time() - t_wall0
    print(f"\nReplay finished in {wall_s:.1f}s wall-clock ({n} windows).\n")

    print("Event timeline (state transitions):")
    if events:
        for t_sim_h, frm, to, sid in events:
            print(f"  t={t_sim_h:5.2f}h  {frm} -> {to}  (leading candidate: {sid or '--'})")
    else:
        print("  (no state transitions -- risk trajectory stayed flat)")

    det = shadow[shadow["detected"]]
    if len(det):
        idx = len(det) // 2
        w = int(det.iloc[idx]["window"])
        sr = next(r for r in sensor.last_results if r.window == w)
        k = sr.top_station
        est_cycle = infer_hidden_cycle_time(line, res.telemetry, k, sr.v_start, sr.v_end)
        fc = forecast_ripple(
            line, k, est_cycle or line.takt_s, horizon_min=60.0,
            buffer_levels=current_buffer_levels(scored, w),
        ) if est_cycle else None
        rec = recommend_flow(line, sr, fc)
        label, reason = taxonomy_label(rec)
        print(f"\nFinal recommendation (mid-episode, window {w}): [{label}] {rec.title}")
        print(f"  {reason}")
    else:
        print("\nNo confident station-level alert was raised this run.")

    print("\n" + "=" * 78)
    print("Every figure above is a SIMULATED PROTOTYPE RESULT on synthetic data.")
    print("=" * 78)


if __name__ == "__main__":
    main()
