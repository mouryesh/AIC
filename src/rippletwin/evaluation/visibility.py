"""How many hidden-station faults ever become visible at line level at all?

This exists because the lead-time metric turned out to be badly under-powered:
it is only defined for episodes where the line's output eventually falls
materially below its own normal rate *and stays there*, and across 40 held-out
episodes that happened in only a handful.

The natural reading of that is not "the metric is broken". It is the finding
itself. A single station slowing down on a 42-station line with decoupling
buffers frequently never shows up as an aggregate throughput shortfall -- the
buffers absorb it, and the loss is real but diffuse. Those faults are invisible
to any monitoring built on line-level output, indefinitely, not merely for a
while.

So the honest headline is not "we warn N minutes earlier". It is:

    a large share of hidden-station disturbances never surface on the
    production board at all, and for those, lead time is not shorter --
    it is undefined, because the alternative never arrives.

This module measures that share, and how often RippleTwin named the station
anyway. It re-simulates the evaluation episodes rather than trusting a cached
column, so the number is derived from the same physics as everything else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..factory import scenarios as SC
from ..factory.topology import apply_coverage, build_line
from ..twin.pipeline import simulate
from .metrics import production_board_moment
from .views import full_observability

FLOW_KINDS = ("SLOWDOWN", "COMBINED")


def measure(
    line_config: str = "configs/line_42.yaml",
    line_seed: int = 7,
    test_seed_base: int = 5000,
    n_test_episodes: int = 40,
    episode_vehicles: int = 1200,
    nominal_vehicles: int = 2600,
    localization_csv: str | Path = "results/tables/localization_raw.csv",
    out_csv: str | Path = "results/tables/board_visibility.csv",
    coverage: float = 0.75,
) -> pd.DataFrame:
    """Measure how often a hidden-station fault ever reaches the production board."""
    configured = build_line(line_config, seed=line_seed)
    sim_line = full_observability(configured)
    view = apply_coverage(configured, coverage, seed=int(round(coverage * 1000)))

    nominal = simulate(sim_line, SC.nominal_run(nominal_vehicles), seed=1)
    ref_rate = float(nominal.meta["throughput_vph"])

    rows = []
    for i in range(n_test_episodes):
        seed = test_seed_base + i
        scen = SC.random_episode(configured, seed=seed, n_vehicles=episode_vehicles)
        real = [d for d in scen.disturbances if d.kind != "MATERIAL_DELAY"]
        if not real:
            continue
        d = real[0]
        if d.kind not in FLOW_KINDS:
            continue

        res = simulate(sim_line, scen, seed=seed + 77)
        board = production_board_moment(
            res.passes, configured, after_t_s=d.t_start_s, reference_rate_vph=ref_rate
        )
        rows.append(
            {
                "seed": seed,
                "station": d.station,
                "station_id": configured.stations[d.station].station_id,
                "kind": d.kind,
                "magnitude": d.magnitude,
                "source_hidden": bool(view.stations[d.station].is_hidden),
                "board_moment_s": board,
                "ever_visible_on_board": board is not None,
                "throughput_vph": res.meta["throughput_vph"],
                "nominal_rate_vph": ref_rate,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    loc_path = Path(localization_csv)
    if loc_path.exists():
        loc = pd.read_csv(loc_path)
        rt = loc[
            (loc["split"] == "test")
            & (loc["method"] == "RippleTwin")
            & (np.isclose(loc["coverage"], coverage))
        ][["seed", "top1_episode", "within1_episode", "detected_episode"]]
        df = df.merge(rt, on="seed", how="left")

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Headline numbers, split by whether the source station had a sensor."""
    if df.empty:
        return {}
    out = {}
    for label, sub in [
        ("all_flow_faults", df),
        ("hidden_source", df[df["source_hidden"]]),
        ("observed_source", df[~df["source_hidden"]]),
    ]:
        if not len(sub):
            continue
        invisible = sub[~sub["ever_visible_on_board"]]
        rec = {
            "n_episodes": int(len(sub)),
            "n_never_visible_on_board": int(len(invisible)),
            "pct_never_visible": round(100.0 * len(invisible) / len(sub), 1),
        }
        if "within1_episode" in sub.columns and len(invisible):
            rec["rippletwin_located_when_invisible_pct"] = round(
                100.0 * invisible["within1_episode"].fillna(0).mean(), 1
            )
            rec["rippletwin_detected_when_invisible_pct"] = round(
                100.0 * invisible["detected_episode"].fillna(0).mean(), 1
            )
        out[label] = rec
    return out
