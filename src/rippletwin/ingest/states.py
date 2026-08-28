"""Turning a PLC state log into the per-vehicle durations the twin consumes.

Why this module is the crux of deployability
--------------------------------------------
Our data contract asks for "station state (running / blocked / starved / fault)"
and the deployment note explains that this is a standard OEE state, derived at
the PLC as::

    STARVED = motor running AND infeed empty
    BLOCKED = motor running AND outfeed full

All true. But it quietly skips over the shape of the data. A plant does **not**
have a table of "vehicle 412 was blocked at S07 for 18.3 seconds". A historian
holds a *state-change log*::

    timestamp             station   state
    2026-03-02 06:00:00   S07       RUNNING
    2026-03-02 06:00:45   S07       BLOCKED
    2026-03-02 06:01:02   S07       RUNNING

and, separately, a *traceability log* of VIN reads::

    timestamp             station   vin
    2026-03-02 06:01:02   S07       WVW-0000412

The twin needs those two joined into per-vehicle durations. That join is this
module. Without it, "we need blocked and starved state" is a sentence in a
slide deck; with it, it is an integration a controls engineer can actually
schedule.

How the join works
------------------
For one station, the VIN reads give departure instants ``d_0 < d_1 < ...``.
Vehicle ``v`` occupied that station over ``(d_{v-1}, d_v]`` -- it arrived when
its predecessor left and it left at its own read. Any BLOCKED or STARVED
interval in the state log is then apportioned to whichever occupancy window it
overlaps, splitting an interval that straddles a boundary.

Two modelling choices worth stating, because a manufacturing engineer will ask:

* **Starvation before the first vehicle is discarded.** There is no occupancy
  window to attribute it to, and counting it against vehicle 0 would make every
  run start with a phantom disturbance.
* **A state interval open at the end of the log is truncated** at the last
  observed timestamp rather than extended, because we do not know when it
  ended and guessing long would manufacture a bottleneck.

Both are conservative in the same direction: they under-report rather than
over-report dwell, which biases us toward missing a fault rather than
inventing one. Given that false alarms are the documented way these systems
lose the floor's trust, that is the direction to err in.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

#: State labels we understand, and the duration column each contributes to.
#: The left-hand side is matched case-insensitively after stripping, so a plant
#: using PackML "Blocked", a historian using "BLK", and a spreadsheet using
#: "blocked" all land in the same place.
STATE_ALIASES: Dict[str, str] = {
    "blocked": "blocked_s",
    "blocking": "blocked_s",
    "blk": "blocked_s",
    "outfeed_full": "blocked_s",
    "starved": "starved_s",
    "starving": "starved_s",
    "strv": "starved_s",
    "infeed_empty": "starved_s",
    "waiting_for_part": "starved_s",
}

#: States that mean the station was working. Not summed, but recognised so an
#: unknown label can be reported rather than silently ignored.
RUNNING_STATES = {"running", "run", "execute", "producing", "active", "auto"}

#: States that are neither running nor flow-blocked: real station downtime.
FAULT_STATES = {"fault", "faulted", "stopped", "aborted", "held", "down", "estop"}


def normalise_state(label: str) -> str:
    """Map a plant's state label onto ``blocked_s``/``starved_s``/``other``."""
    key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    return STATE_ALIASES.get(key, "other")


def close_state_intervals(
    states: pd.DataFrame,
    t_col: str = "t_s",
    station_col: str = "station",
    state_col: str = "state",
    horizon_s: Optional[float] = None,
) -> pd.DataFrame:
    """Turn a state-*change* log into closed intervals.

    A historian records the instant a station entered a state; the interval ends
    when the next change at that station arrives. The final open interval is
    truncated at ``horizon_s`` (default: the last timestamp anywhere in the log)
    rather than extended, so an unterminated BLOCKED at the end of an export
    cannot manufacture an arbitrarily large dwell.
    """
    if states.empty:
        return pd.DataFrame(columns=[station_col, "state_kind", "t0", "t1"])

    s = states[[station_col, state_col, t_col]].copy()
    s = s.sort_values([station_col, t_col], kind="mergesort")
    end = float(horizon_s) if horizon_s is not None else float(s[t_col].max())
    s["t0"] = s[t_col].astype(float)
    s["t1"] = s.groupby(station_col, observed=True)["t0"].shift(-1)
    s["t1"] = s["t1"].fillna(end)
    # A trailing change at the horizon yields a zero-length interval, which is
    # harmless; a negative one means the log is out of order.
    s["t1"] = np.maximum(s["t1"].to_numpy(), s["t0"].to_numpy())
    s["state_kind"] = s[state_col].map(normalise_state)
    return s[[station_col, "state_kind", "t0", "t1"]].reset_index(drop=True)


def unknown_state_labels(states: pd.DataFrame, state_col: str = "state") -> List[str]:
    """Labels we did not recognise as blocked, starved, running or fault.

    Surfaced by the validator rather than silently dropped: an unmapped label
    that actually meant "starved" would quietly delete the signal we depend on.
    """
    out = set()
    for lab in states[state_col].astype(str).unique():
        key = lab.strip().lower().replace(" ", "_").replace("-", "_")
        if key in STATE_ALIASES or key in RUNNING_STATES or key in FAULT_STATES:
            continue
        out.add(lab)
    return sorted(out)


def attribute_states_to_vehicles(
    states: pd.DataFrame,
    scans: pd.DataFrame,
    station_col: str = "station",
) -> pd.DataFrame:
    """Join a closed state-interval log to VIN reads, per vehicle per station.

    ``states`` must be the output of :func:`close_state_intervals`.
    ``scans`` needs ``vehicle_id``, ``station`` and ``t_s`` -- the instant the
    VIN was read as the unit left that station.

    Returns telemetry rows with ``t_start_s``, ``t_depart_s``, ``blocked_s`` and
    ``starved_s``. Processing time is left to ``PlantData.from_frames``, which
    derives it as occupancy minus blocked.
    """
    if scans.empty:
        return pd.DataFrame(
            columns=["vehicle_id", "station", "t_start_s", "t_depart_s",
                     "blocked_s", "starved_s"]
        )

    rows = []
    flow = states[states["state_kind"].isin(["blocked_s", "starved_s"])]

    for stn, sc in scans.groupby(station_col, observed=True):
        sc = sc.sort_values("t_s", kind="mergesort")
        depart = sc["t_s"].to_numpy(dtype=float)
        vids = sc["vehicle_id"].to_numpy()
        # Vehicle v held the station from its predecessor's departure to its
        # own. The first vehicle has no predecessor, so its window opens at its
        # own arrival, which we do not observe -- give it a zero-length window
        # so nothing is attributed to it rather than attributing everything.
        start = np.concatenate([[depart[0]], depart[:-1]])

        st = flow[flow[station_col] == stn]
        blocked = np.zeros(len(depart))
        starved = np.zeros(len(depart))

        if not st.empty:
            t0 = st["t0"].to_numpy(dtype=float)
            t1 = st["t1"].to_numpy(dtype=float)
            kind = st["state_kind"].to_numpy()
            # Overlap of each state interval with each occupancy window. The
            # windows are contiguous and sorted, so we can bound the search
            # instead of forming the full cross product.
            lo = np.searchsorted(depart, t0, side="left")
            hi = np.searchsorted(start, t1, side="right")
            for j in range(len(t0)):
                a, b = lo[j], min(hi[j], len(depart))
                if b <= a:
                    continue
                ov = np.minimum(t1[j], depart[a:b]) - np.maximum(t0[j], start[a:b])
                ov = np.clip(ov, 0.0, None)
                if kind[j] == "blocked_s":
                    blocked[a:b] += ov
                else:
                    starved[a:b] += ov

        rows.append(pd.DataFrame({
            "vehicle_id": vids,
            "station": stn,
            "t_start_s": start,
            "t_depart_s": depart,
            "blocked_s": blocked,
            "starved_s": starved,
        }))

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["vehicle_id", "station"]).reset_index(drop=True)


def discover_topology(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Infer station order from the data, so nobody hand-writes 42 stations.

    A serial line reveals its own order: for any vehicle, stations it passed
    earlier have earlier departure times. Ranking stations by their median
    departure position across vehicles recovers the sequence without a layout
    drawing, which matters because "send us your line topology" is exactly the
    kind of request that adds two weeks to a pilot.

    Returns a frame of ``station`` in inferred process order with the median
    inter-station delay, which is a first estimate of takt.

    This is a *proposal for an engineer to confirm*, not an authority. It cannot
    detect parallel sub-lines or rework loops, and the report says so.
    """
    if telemetry.empty:
        return pd.DataFrame(columns=["order", "station", "median_t_depart_s"])

    med = (
        telemetry.groupby("station", observed=True)["t_depart_s"]
        .median()
        .sort_values()
    )
    out = pd.DataFrame({
        "order": np.arange(len(med)),
        "station": med.index.to_numpy(),
        "median_t_depart_s": med.to_numpy(),
    })
    return out
