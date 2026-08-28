"""Early bottleneck prediction: a graded risk state read off the shadow
sensor's own window-to-window trajectory, before a constraint binds.

Nothing here is a new detector. ``ShadowSensor`` already produces, per window,
a log-likelihood ratio against NULL (``llr``), fitted flow amplitudes
(``amp_starve``/``amp_block``) and a posterior mass on the leading candidate
(``group_prob``). This module asks a different question of that same
sequence: not "is this window anomalous" but "is this getting worse, and
when would it start to cost output" -- by fitting a short trend across recent
windows and comparing it to a *second*, looser threshold calibrated the same
way ``detect_llr`` already is (``ShadowSensor.calibrate``, empirical null
distribution, stated target false-alarm rate).

State ladder
------------
``NORMAL`` < ``DEGRADING`` < ``WATCH`` < ``PREDICTED_CONSTRAINT`` <
``ACTIVE_BOTTLENECK``, with ``RECOVERING`` re-entered from any elevated state
once the trend reverses. The boundary between ``PREDICTED_CONSTRAINT`` and
``ACTIVE_BOTTLENECK`` is exactly the boundary ``RippleForecast.is_binding``
already draws (estimated constraint cycle time crosses takt) -- this module
does not invent a second notion of "bottleneck".

Time-to-impact is not a hand-tuned countdown: it is the same crossing,
reached by linearly extrapolating the constraint's own estimated cycle time
(the same estimate ``infer_hidden_cycle_time``/observed processing time
already produce) across the recent trend, forward to the point it would
cross takt. When the trend is flat or improving, no crossing exists and no
time-to-impact is reported -- which is the correct answer, not a missing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..factory.topology import LineTopology
from .shadow import ShadowConfig, infer_hidden_cycle_time

STATE_NORMAL = "NORMAL"
STATE_DEGRADING = "DEGRADING"
STATE_WATCH = "WATCH"
STATE_PREDICTED_CONSTRAINT = "PREDICTED_CONSTRAINT"
STATE_ACTIVE_BOTTLENECK = "ACTIVE_BOTTLENECK"
STATE_RECOVERING = "RECOVERING"

_ELEVATED = (STATE_WATCH, STATE_PREDICTED_CONSTRAINT, STATE_ACTIVE_BOTTLENECK)
_STATE_RANK = {
    STATE_NORMAL: 0,
    STATE_RECOVERING: 0,
    STATE_DEGRADING: 1,
    STATE_WATCH: 2,
    STATE_PREDICTED_CONSTRAINT: 3,
    STATE_ACTIVE_BOTTLENECK: 4,
}


@dataclass
class PredictConfig:
    """Tunables for the trend/state layer. Slope thresholds are resolved from
    the calibrated noise scale of the LLR statistic unless overridden."""

    #: Windows of trailing history used to fit the trend.
    trend_window: int = 5
    #: Minimum points before a trend is trusted at all.
    min_trend_points: int = 3
    #: Windows of trailing cycle-time-estimate history used to extrapolate
    #: time-to-impact.
    cyc_trend_window: int = 4
    #: Explicit override for the "rising" LLR slope (per window). ``None``
    #: derives it from ``cfg.llr_noise_std`` at run time.
    rising_slope: Optional[float] = None
    #: Explicit override for the "declining" LLR slope. ``None`` derives it.
    declining_slope: Optional[float] = None
    #: Horizon beyond which a projected time-to-impact is not reported --
    #: an extrapolation that far out is noise, not a forecast.
    max_time_to_impact_min: float = 180.0

    def resolve_slopes(self, cfg: ShadowConfig) -> Tuple[float, float]:
        rising = self.rising_slope
        if rising is None:
            rising = max(0.02, 0.5 * cfg.llr_noise_std / max(1, self.trend_window))
        declining = self.declining_slope
        if declining is None:
            declining = -rising
        return float(rising), float(declining)


@dataclass
class PredictionResult:
    window: int
    t_mid_s: float
    station: Optional[int]
    station_id: Optional[str]
    state: str
    #: 0..1 read-out of how close the LLR sits to the detection threshold.
    risk: float
    llr: float
    llr_trend: float
    is_hidden: bool
    #: Minutes until the extrapolated constraint cycle time would cross takt.
    #: ``None`` when there is no rising trend to extrapolate, or the
    #: constraint is already binding (see ``ACTIVE_BOTTLENECK``).
    time_to_impact_min: Optional[float]
    confidence: float
    evidence: dict = field(default_factory=dict)


def _linear_slope(y: List[float]) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    yv = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(yv)):
        return 0.0
    try:
        return float(np.polyfit(x, yv, 1)[0])
    except np.linalg.LinAlgError:
        return 0.0


def _estimate_constraint_cycle_s(
    line: LineTopology,
    telemetry: pd.DataFrame,
    scored: pd.DataFrame,
    station: int,
    window: int,
    v_start: int,
    v_end: int,
) -> Optional[float]:
    """Best available estimate of a candidate's processing time this window.

    Hidden station: read off the departure rate of the nearest observed
    downstream station (``infer_hidden_cycle_time``). Observed station: the
    window's own mean processing time, already computed by ``baseline.score``.
    """
    if line.stations[station].is_hidden:
        return infer_hidden_cycle_time(line, telemetry, station, v_start, v_end)
    row = scored[(scored["window"] == window) & (scored["station"] == station)]
    if row.empty:
        return None
    val = float(row["proc_time_s_mean"].iloc[0])
    return val if np.isfinite(val) else None


def run_predictor(
    shadow_df: pd.DataFrame,
    line: LineTopology,
    telemetry: pd.DataFrame,
    scored: pd.DataFrame,
    cfg: ShadowConfig,
    pcfg: Optional[PredictConfig] = None,
) -> pd.DataFrame:
    """Derive a per-window predicted-bottleneck state from a completed
    ``ShadowSensor.run()`` output.

    Causal by construction: window ``w``'s state depends only on windows
    ``<= w`` in ``shadow_df`` (already itself causal/EWMA-smoothed) plus
    ``telemetry``/``scored`` rows within that window's own vehicle range.
    Nothing here looks ahead.
    """
    pcfg = pcfg or PredictConfig()
    rising_slope, declining_slope = pcfg.resolve_slopes(cfg)

    df = shadow_df.sort_values("window").reset_index(drop=True)
    dt_s = float(np.median(np.diff(df["t_mid_s"].to_numpy()))) if len(df) > 1 else 0.0

    llr_hist: List[float] = []
    cyc_hist_by_station: Dict[int, List[Tuple[int, float]]] = {}
    prev_state = STATE_NORMAL
    rows: List[PredictionResult] = []

    for _, r in df.iterrows():
        w = int(r["window"])
        llr = float(r["llr"]) if np.isfinite(r["llr"]) else 0.0
        llr_hist.append(llr)
        recent_llr = llr_hist[-pcfg.trend_window :]
        llr_trend = (
            _linear_slope(recent_llr) if len(recent_llr) >= pcfg.min_trend_points else 0.0
        )

        confident = bool(r["confident"])
        station = int(r["top_station"]) if pd.notna(r["top_station"]) else None
        station_id = str(r["top_station_id"]) if pd.notna(r.get("top_station_id")) else None
        is_hidden = bool(r["top_is_hidden"]) if pd.notna(r.get("top_is_hidden")) else False

        cyc_est = None
        is_binding = None
        if station is not None:
            cyc_est = _estimate_constraint_cycle_s(
                line, telemetry, scored, station, w, int(r["v_start"]), int(r["v_end"])
            )
            if cyc_est is not None:
                cyc_hist_by_station.setdefault(station, []).append((w, cyc_est))
                is_binding = cyc_est > line.takt_s

        # --- state ----------------------------------------------------
        if confident and llr >= cfg.detect_llr:
            state = STATE_ACTIVE_BOTTLENECK if is_binding else STATE_PREDICTED_CONSTRAINT
        elif llr >= cfg.watch_llr:
            state = STATE_WATCH
        elif llr_trend > rising_slope:
            state = STATE_DEGRADING
        elif prev_state in _ELEVATED and llr_trend < declining_slope:
            state = STATE_RECOVERING
        elif prev_state == STATE_RECOVERING and llr_trend <= 0:
            state = STATE_RECOVERING
        else:
            state = STATE_NORMAL

        # --- time-to-impact --------------------------------------------------
        time_to_impact = None
        if (
            state in (STATE_DEGRADING, STATE_WATCH, STATE_PREDICTED_CONSTRAINT)
            and station is not None
            and dt_s > 0
        ):
            hist = cyc_hist_by_station.get(station, [])[-pcfg.cyc_trend_window :]
            if len(hist) >= 3:
                ws = np.array([h[0] for h in hist], dtype=float)
                cs = np.array([h[1] for h in hist], dtype=float)
                slope = _linear_slope(list(cs))
                if slope > 1e-9 and cs[-1] < line.takt_s:
                    windows_to_cross = (line.takt_s - cs[-1]) / slope
                    if windows_to_cross > 0:
                        minutes = windows_to_cross * dt_s / 60.0
                        if minutes <= pcfg.max_time_to_impact_min:
                            time_to_impact = float(minutes)

        risk = float(np.clip(llr / max(cfg.detect_llr, 1e-6), 0.0, 1.5) / 1.5)

        rows.append(
            PredictionResult(
                window=w,
                t_mid_s=float(r["t_mid_s"]),
                station=station,
                station_id=station_id,
                state=state,
                risk=risk,
                llr=llr,
                llr_trend=float(llr_trend),
                is_hidden=is_hidden,
                time_to_impact_min=time_to_impact,
                confidence=float(r["group_prob"]) if pd.notna(r.get("group_prob")) else 0.0,
                evidence={
                    "amp_starve": float(r.get("amp_starve", np.nan)),
                    "amp_block": float(r.get("amp_block", np.nan)),
                    "constraint_cycle_s": cyc_est,
                    "takt_s": float(line.takt_s),
                    "watch_llr": cfg.watch_llr,
                    "detect_llr": cfg.detect_llr,
                },
            )
        )
        prev_state = state

    out = pd.DataFrame([vars(r) for r in rows])
    if len(out):
        ev = out.pop("evidence")
        for col in ("amp_starve", "amp_block", "constraint_cycle_s", "takt_s"):
            out[col] = [e.get(col) for e in ev]
    return out


def state_rank(state: str) -> int:
    """Ordinal rank so states can be compared/aggregated ("did it ever reach
    at least WATCH")."""
    return _STATE_RANK.get(state, 0)
