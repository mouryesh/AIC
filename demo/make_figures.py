#!/usr/bin/env python3
"""Generate the mechanism figure used in the deck and the video.

Runs the flagship scenario and plots the blocking / starvation profile at the
moment RippleTwin localises the fault. The station marked as the true source is
one the model received no telemetry from.

    python demo/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rippletwin.evaluation.figures import build_all, pressure_profile_figure  # noqa: E402
from rippletwin.factory import scenarios as SC  # noqa: E402
from rippletwin.factory.topology import build_line  # noqa: E402
from rippletwin.twin.pipeline import fit_context, infer, simulate  # noqa: E402

DEMO_SEED = 20260301


def main() -> int:
    line = build_line(ROOT / "configs" / "line_42.yaml", seed=7)
    nominal = simulate(line, SC.nominal_run(2600), seed=1)
    calib = simulate(line, SC.nominal_run(2200), seed=2)
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)

    scen = SC.scenario_hidden_bottleneck(line)
    res = simulate(line, scen, seed=DEMO_SEED)
    scored, shadow, _ = infer(ctx, res)

    truth = res.disturbances.iloc[0]
    true_station = int(truth["station"])
    t0 = float(truth["t_start_s"]) + float(truth["ramp_s"])
    det = shadow[
        shadow["detected"]
        & (shadow["t_mid_s"] >= t0)
        & (shadow["t_mid_s"] <= float(truth["t_end_s"]))
    ]
    if det.empty:
        print("no detection in the flagship scenario; cannot draw the figure")
        return 1
    window = int(det.iloc[len(det) // 2]["window"])

    out = ROOT / "results" / "figures" / "pressure_profile.png"
    pressure_profile_figure(line, scored, window, true_station, out)
    print(f"wrote {out}")
    print(f"  true source {line.stations[true_station].station_id} "
          f"({line.stations[true_station].tier}) at window {window}")

    for p in build_all(ROOT / "results"):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
