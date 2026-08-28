"""Dynamic sensor health: telemetry-level faults layered on top of the
static observability view.

``topology.py::apply_coverage`` decides, once, for an entire run, whether a
station is instrumented at all -- a station is MANUAL or it is not, for the
whole horizon. Real sensors do not fail that cleanly: a PLC timestamp source
can drop out for twenty minutes and come back, a noisy photocell can add
jitter without ever going fully dark, and a stuck reading can silently
repeat the same value long after the line has moved on.

This module applies those failure modes as a *post-processing* step on
already-projected telemetry -- the same point ``SimResult.as_plant_data()``
and ``evaluation.views.telemetry_view`` already draw the ground-truth/
observed boundary at. Ground truth (``passes``, ``disturbances``,
``defects``) is never touched; only what a plant could see is altered,
exactly the way a real sensor failure corrupts a data stream, never the
factory floor itself.

Four modes
----------
DROPOUT
    Rows removed entirely for the interval, at an already-instrumented
    station. Mechanically identical to that station being MANUAL, but only
    for a bounded window -- and because the estimator already treats a
    station absent from a window's data as unmeasured evidence (``NaN``,
    not zero: see ``twin/shadow.py``'s ``mask = np.isfinite(...)``), dropout
    works with the existing inference math completely unchanged. The only
    new work is deciding which rows to remove.

INTERMITTENT
    DROPOUT applied as a series of short on/off bursts inside a longer
    interval, rather than one continuous gap.

NOISY
    Rows kept, but corrupted with extra noise on the flow channels.

STALE
    Rows kept, but frozen at the last good reading from before the fault
    began -- zero apparent variance. Deliberately the hardest failure mode
    to catch, because a stale sensor still *looks* confidently instrumented;
    ``flag_stale_windows`` exists specifically to catch it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

DROPOUT = "DROPOUT"
INTERMITTENT = "INTERMITTENT"
NOISY = "NOISY"
STALE = "STALE"

QUALITY_OBSERVED = "OBSERVED"
QUALITY_DROPPED = "DROPPED"     # row removed entirely -- absent, not a tag on a row
QUALITY_NOISY = "NOISY"
QUALITY_STALE = "STALE"
QUALITY_SUSPECT = "SUSPECT"     # detected as probably stale/unreliable after the fact

_FLOW_COLS = ("proc_time_s", "cycle_time_s", "blocked_s", "starved_s")


@dataclass
class SensorFault:
    """One dynamic fault: a station, a kind, and a time interval."""

    station: int
    kind: str
    t_start_s: float
    t_end_s: float
    #: NOISY: extra Gaussian noise std, as a fraction of the channel's own
    #: nominal scale (interpreted per-column at apply time).
    noise_frac: float = 0.6
    #: INTERMITTENT: burst on/off durations, seconds.
    burst_on_s: float = 300.0
    burst_off_s: float = 300.0


def _in_interval(t: np.ndarray, f: SensorFault) -> np.ndarray:
    return (t >= f.t_start_s) & (t < f.t_end_s)


def _intermittent_mask(t: np.ndarray, f: SensorFault) -> np.ndarray:
    """True where an intermittent dropout is currently 'off' (data missing)."""
    in_window = _in_interval(t, f)
    period = max(f.burst_on_s + f.burst_off_s, 1.0)
    phase = np.mod(t - f.t_start_s, period)
    off = phase >= f.burst_on_s
    return in_window & off


def apply_sensor_faults(
    telemetry: pd.DataFrame,
    faults: Sequence[SensorFault],
    seed: int = 0,
) -> pd.DataFrame:
    """Apply a list of dynamic sensor faults to already-projected telemetry.

    Returns a new frame, sorted by (station, t_start_s) as the input is
    assumed to be, with a ``data_quality`` column tagging every surviving
    row. DROPOUT/INTERMITTENT rows are removed outright, so they carry no
    tag -- their absence *is* the signal, consumed by the existing
    NaN-tolerant windowing/likelihood machinery with no further change.
    """
    if not faults:
        out = telemetry.copy()
        out["data_quality"] = QUALITY_OBSERVED
        return out

    rng = np.random.default_rng(seed)
    out = telemetry.copy()
    out["data_quality"] = QUALITY_OBSERVED
    t = out["t_start_s"].to_numpy(dtype=float) if "t_start_s" in out.columns else (
        out["t_depart_s"].to_numpy(dtype=float)
    )
    station = out["station"].to_numpy(dtype=int)

    drop_mask = np.zeros(len(out), dtype=bool)

    for f in faults:
        sel = station == f.station
        if not sel.any():
            continue

        if f.kind == DROPOUT:
            drop_mask |= sel & _in_interval(t, f)

        elif f.kind == INTERMITTENT:
            drop_mask |= sel & _intermittent_mask(t, f)

        elif f.kind == NOISY:
            m = sel & _in_interval(t, f)
            if not m.any():
                continue
            out.loc[m, "data_quality"] = QUALITY_NOISY
            for c in _FLOW_COLS:
                if c not in out.columns:
                    continue
                vals = out.loc[m, c].to_numpy(dtype=float)
                scale = np.nanstd(vals) or (np.nanmean(np.abs(vals)) * 0.2 + 1e-6)
                noise = rng.normal(0.0, f.noise_frac * max(scale, 1e-6), size=vals.shape)
                out.loc[m, c] = np.maximum(0.0, vals + noise)

        elif f.kind == STALE:
            m = sel & _in_interval(t, f)
            if not m.any():
                continue
            out.loc[m, "data_quality"] = QUALITY_STALE
            idx = out.index[m]
            before = out[sel & (t < f.t_start_s)]
            for c in _FLOW_COLS:
                if c not in out.columns:
                    continue
                if len(before):
                    frozen = float(before[c].iloc[-1])
                else:
                    frozen = float(out.loc[idx, c].iloc[0]) if len(idx) else 0.0
                out.loc[idx, c] = frozen

        else:
            raise ValueError(f"unknown sensor fault kind: {f.kind!r}")

    out = out.loc[~drop_mask].reset_index(drop=True)
    return out


def flag_stale_windows(
    windows: pd.DataFrame,
    std_col: str = "proc_time_s_std",
    count_col: str = "proc_time_s_count",
    std_floor: float = 0.05,
    min_count: int = 6,
) -> pd.DataFrame:
    """Flag (window, station) rows whose processing-time variance has
    collapsed to near zero -- the signature a frozen/stale sensor leaves,
    since a genuinely operating station never reports the identical cycle
    time on every vehicle.

    This is the detection half of STALE handling: the twin cannot know a
    fault schedule exists (that would defeat the point), so it has to infer
    "this reading is probably not trustworthy" from the data's own shape.
    Flagged rows are treated as SUSPECT and should be excluded before
    scoring -- the same NaN-tolerant path DROPOUT already exercises, rather
    than inventing a new partial-trust weighting inside the likelihood
    (see module docstring: same failure class, same fix, no new machinery).

    ``min_count`` guards against flagging a window with too few vehicles to
    estimate variance from at all -- absence of evidence, not evidence of
    staleness.
    """
    out = windows.copy()
    out["stale_suspect"] = False
    if std_col not in out.columns:
        return out
    cnt = out[count_col] if count_col in out.columns else pd.Series(min_count, index=out.index)
    suspect = (out[std_col].fillna(0.0) < std_floor) & (cnt >= min_count)
    out.loc[suspect, "stale_suspect"] = True
    return out


def mask_suspect_rows(windows: pd.DataFrame) -> pd.DataFrame:
    """Drop rows flagged by ``flag_stale_windows`` -- SUSPECT is handled
    exactly like DROPOUT/no-data-this-window: absent, not down-weighted."""
    if "stale_suspect" not in windows.columns:
        return windows
    return windows.loc[~windows["stale_suspect"]].reset_index(drop=True)


def summarize_data_quality(telemetry_with_quality: pd.DataFrame) -> pd.DataFrame:
    """Per-station fraction of rows in each data-quality state, for display.

    Dropped rows never appear in ``telemetry_with_quality`` at all (they
    were removed by ``apply_sensor_faults``), so this reports OBSERVED /
    NOISY / STALE only among the rows that survived; a station's dropout
    fraction has to be read off row *counts* against a station with no
    faults, which the dashboard does separately.
    """
    if telemetry_with_quality.empty or "data_quality" not in telemetry_with_quality.columns:
        return pd.DataFrame(columns=["station", "quality", "fraction"])
    g = (
        telemetry_with_quality.groupby(["station", "data_quality"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals = g.groupby("station")["n"].transform("sum")
    g["fraction"] = g["n"] / totals
    return g
