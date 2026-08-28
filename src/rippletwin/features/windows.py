"""Windowing and feature construction from observed telemetry.

Alignment choice
----------------
Windows are indexed by **vehicle sequence**, not wall-clock time. This matters.
When station k slows down while processing vehicle v, it is *that same vehicle*
that then starves the stations downstream and backs up the stations upstream.
Vehicle index is therefore the natural coordinate in which a disturbance and its
ripple are simultaneous; wall-clock time smears them by the line's transit delay.

Everything in this module is computed from ``telemetry`` (observed stations
only), ``vehicles`` and ``environment``. Ground-truth tables are never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..factory.topology import LineTopology

#: Channels aggregated per station per window.
FLOW_CHANNELS = ["proc_time_s", "blocked_s", "starved_s"]
PROCESS_CHANNELS = ["torque_nm", "vibration_mm_s", "station_temp_c"]


@dataclass
class WindowSpec:
    """Vehicle-sequence windowing."""

    width: int = 20
    stride: int = 5

    #: Vehicles to skip at the start of a run. Until the line has filled,
    #: every station is starved simply because production has not reached it
    #: yet, and the deviation channels are meaningless. A real deployment has
    #: the same blind spot after a cold start.
    warmup: int = 0

    def windows(self, n_vehicles: int) -> List[tuple]:
        out = []
        v = int(self.warmup)
        while v + self.width <= n_vehicles:
            out.append((v, v + self.width))
            v += self.stride
        return out

    @classmethod
    def for_line(cls, line, width: int = 20, stride: int = 5) -> "WindowSpec":
        """Windowing with a warm-up long enough for the line to fill."""
        fill = line.n_stations + sum(
            min(s.out_buffer, 100) for s in line.stations[:-1]
        )
        return cls(width=width, stride=stride, warmup=int(fill))


def aggregate_windows(
    telemetry: pd.DataFrame,
    vehicles: pd.DataFrame,
    line: LineTopology,
    spec: WindowSpec,
) -> pd.DataFrame:
    """Aggregate observed telemetry into (window, station) rows.

    Returns a long frame with one row per observed station per window, plus the
    window's variant mix and shift, which are needed to form fair expectations.
    """
    n_vehicles = int(vehicles["vehicle_id"].max()) + 1
    wins = spec.windows(n_vehicles)
    if not wins:
        raise ValueError("no windows: n_vehicles smaller than window width")

    win_id = np.full(n_vehicles, -1, dtype=object)
    # Build an explicit (window, vehicle) index so overlapping strides work.
    pairs = []
    for w, (a, b) in enumerate(wins):
        pairs.append(pd.DataFrame({"window": w, "vehicle_id": np.arange(a, b)}))
    idx = pd.concat(pairs, ignore_index=True)

    tel = telemetry.merge(idx, on="vehicle_id", how="inner")

    agg_map: Dict[str, list] = {
        "proc_time_s": ["mean", "std", "max"],
        "blocked_s": ["mean", "max"],
        "starved_s": ["mean", "max"],
        "t_depart_s": ["min", "max"],
    }
    # Buffer occupancy is optional in the data contract -- a plant with no
    # conveyor counters is explicitly supported -- so aggregate it only when it
    # is actually there. It used to be unconditional, which meant such a plant
    # crashed here rather than degrading.
    for c in ("buffer_level", "buffer_capacity"):
        if c in tel.columns:
            agg_map[c] = ["mean"] if c == "buffer_level" else ["max"]
    for c in PROCESS_CHANNELS:
        if c in tel.columns:
            agg_map[c] = ["mean", "std"]

    g = tel.groupby(["window", "station"], observed=True).agg(agg_map)
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()

    # Window-level context: variant mix and dominant shift.
    vmix = (
        tel.drop_duplicates(["window", "vehicle_id"])
        .groupby("window")["variant"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    for v in line.variants:
        if v not in vmix.columns:
            vmix[v] = 0.0
    vmix = vmix[[v for v in line.variants]].add_prefix("mix_").reset_index()

    vsh = vehicles.merge(idx, on="vehicle_id", how="inner")
    shift_mode = (
        vsh.groupby("window")["shift"].agg(lambda s: s.value_counts().idxmax()).reset_index()
    )
    wbounds = pd.DataFrame(
        {"window": np.arange(len(wins)),
         "v_start": [a for a, _ in wins],
         "v_end": [b for _, b in wins]}
    )

    out = (
        g.merge(vmix, on="window", how="left")
        .merge(shift_mode, on="window", how="left")
        .merge(wbounds, on="window", how="left")
    )
    out["station_id"] = out["station"].map({s.index: s.station_id for s in line.stations})
    out["zone"] = out["station"].map({s.index: s.zone for s in line.stations})
    out["tier"] = out["station"].map({s.index: s.tier for s in line.stations})
    # Buffer fill ratio, where derivable. Absent counters give NaN, not an error.
    for c in ("buffer_level_mean", "buffer_capacity_max"):
        if c not in out.columns:
            out[c] = np.nan
    out["buffer_fill"] = out["buffer_level_mean"] / out["buffer_capacity_max"].replace(0, np.nan)
    return out


def attach_environment(windows: pd.DataFrame, environment: pd.DataFrame) -> pd.DataFrame:
    """Join mean ambient conditions over each window's time span."""
    env = environment.sort_values("t_s")
    t_mid = (windows["t_depart_s_min"] + windows["t_depart_s_max"]) / 2.0
    pos = np.searchsorted(env["t_s"].to_numpy(), t_mid.to_numpy(), side="left")
    pos = np.clip(pos, 0, len(env) - 1)
    out = windows.copy()
    out["ambient_temp_c"] = env["ambient_temp_c"].to_numpy()[pos]
    out["humidity_pct"] = env["humidity_pct"].to_numpy()[pos]
    return out
