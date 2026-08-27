"""Learned nominal behaviour and physically-scaled deviation channels.

A deviation only means something relative to what the line *should* be doing
right now. "Should" depends on the variant mix in the window and on the shift --
an SUV-heavy window on night shift is legitimately slower than a sedan-heavy
window on day shift, and a twin that flags that is a twin nobody will trust.

Units
-----
Blocked and starved time are expressed as a **fraction of takt**, not as a
z-score. Two reasons, one statistical and one practical:

* Statistically, ``starved_s`` is zero-inflated with a heavy tail. Its MAD under
  nominal flow is near zero, so a MAD-based z-score turns an ordinary 15-second
  starvation into a 40-sigma event. Takt fractions are bounded and stable.
* Practically, "S24 is losing 30% of takt to starvation" is a sentence a plant
  engineer can act on. "S24 has z = 47" is not.

Processing time keeps a z-score, because there it is well behaved and because
the quantity of interest is genuinely "is this station slower than it should be
for this mix", which is naturally a standardised comparison.

The baseline is fitted on a disturbance-free reference run and then frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..factory.topology import LineTopology

_MAD_TO_SIGMA = 1.4826


def _robust_scale(x: np.ndarray, floor: float) -> float:
    """MAD-based scale estimate with a floor to avoid divide-by-almost-zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return max(floor, 1e-6)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * _MAD_TO_SIGMA
    return float(max(mad, floor))


@dataclass
class NominalBaseline:
    """Frozen expectation of nominal behaviour per station."""

    takt_s: float = 60.0
    #: Median processing time per (station, variant), for mix-aware expectations.
    proc_by_variant: Dict[int, Dict[str, float]] = field(default_factory=dict)
    shift_factor: Dict[int, Dict[str, float]] = field(default_factory=dict)
    #: Residual centring for processing time, per station (removes fit bias).
    proc_resid_center: Dict[int, float] = field(default_factory=dict)
    proc_scale: Dict[int, float] = field(default_factory=dict)
    #: Nominal blocked / starved as a fraction of takt, per station.
    blocked_frac_center: Dict[int, float] = field(default_factory=dict)
    starved_frac_center: Dict[int, float] = field(default_factory=dict)
    #: Robust noise scale of those fractions, per station.
    blocked_frac_scale: Dict[int, float] = field(default_factory=dict)
    starved_frac_scale: Dict[int, float] = field(default_factory=dict)
    process_center: Dict[str, Dict[int, float]] = field(default_factory=dict)
    process_scale: Dict[str, Dict[int, float]] = field(default_factory=dict)
    stations: List[int] = field(default_factory=list)
    #: Pooled noise scales used by the shadow-sensing likelihood.
    sigma_blocked: float = 0.03
    sigma_starved: float = 0.05

    # ------------------------------------------------------------------ fitting

    @classmethod
    def fit(
        cls,
        nominal_windows: pd.DataFrame,
        nominal_telemetry: pd.DataFrame,
        line: LineTopology,
    ) -> "NominalBaseline":
        """Fit on a reference run that contains no injected disturbances."""
        obj = cls(takt_s=float(line.takt_s))
        obj.stations = sorted(nominal_windows["station"].unique().tolist())

        # --- per (station, variant) nominal processing time
        pv = (
            nominal_telemetry.groupby(["station", "variant"])["proc_time_s"]
            .median()
            .unstack()
        )
        for st in pv.index:
            vals = {
                str(v): float(pv.loc[st, v])
                for v in pv.columns
                if np.isfinite(pv.loc[st, v])
            }
            if vals:
                obj.proc_by_variant[int(st)] = vals

        # --- shift multiplier per station
        tel = nominal_telemetry
        if "shift" not in tel.columns:
            tel = tel.copy()
            tel["shift"] = None
        for st, grp in tel.groupby("station"):
            base = grp["proc_time_s"].median()
            obj.shift_factor[int(st)] = {}
            for sh, g2 in grp.groupby("shift", dropna=False):
                if sh is None or (isinstance(sh, float) and np.isnan(sh)):
                    continue
                obj.shift_factor[int(st)][str(sh)] = float(
                    g2["proc_time_s"].median() / base
                )

        # --- processing-time residual centring and scale.
        # Window means sit slightly above the per-vehicle median because the
        # processing-time distribution is right-skewed and carries micro-stops.
        # Without this correction every observed station carries a small positive
        # bias, which would systematically favour observed stations as suspects.
        raw = obj._expected_proc_frame(nominal_windows, line)
        resid = nominal_windows["proc_time_s_mean"].to_numpy() - raw
        st_arr = nominal_windows["station"].to_numpy()
        for st in obj.stations:
            m = st_arr == st
            r = resid[m]
            r = r[np.isfinite(r)]
            if r.size:
                obj.proc_resid_center[int(st)] = float(np.median(r))
                obj.proc_scale[int(st)] = _robust_scale(r, 0.30)

        # --- blocked / starved as takt fractions
        bf = nominal_windows["blocked_s_mean"].to_numpy() / obj.takt_s
        sf = nominal_windows["starved_s_mean"].to_numpy() / obj.takt_s
        for st in obj.stations:
            m = st_arr == st
            b, s = bf[m], sf[m]
            obj.blocked_frac_center[int(st)] = float(np.median(b[np.isfinite(b)]))
            obj.starved_frac_center[int(st)] = float(np.median(s[np.isfinite(s)]))
            # Floor at 1.5% of takt: below that the difference is not operationally
            # meaningful and treating it as signal only manufactures false alarms.
            obj.blocked_frac_scale[int(st)] = _robust_scale(b, 0.015)
            obj.starved_frac_scale[int(st)] = _robust_scale(s, 0.015)

        obj.sigma_blocked = float(
            np.median(list(obj.blocked_frac_scale.values())) or 0.03
        )
        obj.sigma_starved = float(
            np.median(list(obj.starved_frac_scale.values())) or 0.05
        )

        # --- process channels (RICH stations only)
        for ch in ["torque_nm_mean", "vibration_mm_s_mean", "station_temp_c_mean"]:
            if ch not in nominal_windows.columns:
                continue
            obj.process_center[ch] = {}
            obj.process_scale[ch] = {}
            for st, grp in nominal_windows.groupby("station"):
                vals = grp[ch].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size < 8:
                    continue
                obj.process_center[ch][int(st)] = float(np.median(vals))
                obj.process_scale[ch][int(st)] = _robust_scale(vals, 0.05)

        return obj

    # ------------------------------------------------------------------ scoring

    def expected_proc(
        self, station: int, mix: Dict[str, float], shift: Optional[str]
    ) -> float:
        """Mix- and shift-adjusted expected processing time for a window."""
        pv = self.proc_by_variant.get(int(station))
        if not pv:
            return float("nan")
        total = sum(mix.values()) or 1.0
        fallback = float(np.mean(list(pv.values())))
        exp = sum(pv.get(v, fallback) * (w / total) for v, w in mix.items())
        if shift is not None:
            exp *= self.shift_factor.get(int(station), {}).get(str(shift), 1.0)
        return float(exp)

    def _expected_proc_frame(
        self, windows: pd.DataFrame, line: LineTopology
    ) -> np.ndarray:
        """Vectorised expected processing time for every row of a window frame."""
        mix_cols = [c for c in windows.columns if c.startswith("mix_")]
        variant_names = [c[len("mix_"):] for c in mix_cols]
        st_arr = windows["station"].to_numpy(dtype=int)
        sh_arr = (
            windows["shift"].to_numpy()
            if "shift" in windows.columns
            else np.array([None] * len(windows))
        )
        mix_mat = (
            windows[mix_cols].to_numpy(dtype=float) if mix_cols else None
        )
        out = np.full(len(windows), np.nan)
        for r in range(len(windows)):
            mix = (
                {variant_names[j]: mix_mat[r, j] for j in range(len(variant_names))}
                if mix_mat is not None
                else {}
            )
            out[r] = self.expected_proc(int(st_arr[r]), mix, sh_arr[r])
        return out

    def score(self, windows: pd.DataFrame, line: LineTopology) -> pd.DataFrame:
        """Attach deviation channels to a window frame.

        Produces, per (window, station):
          ``z_proc``     - processing time vs mix/shift-adjusted expectation, in sigma
          ``d_blocked``  - blocked time above nominal, as a fraction of takt
          ``d_starved``  - starved time above nominal, as a fraction of takt
          ``pressure``   - d_blocked - d_starved, for display and interpretation
        """
        out = windows.copy()

        exp_proc = self._expected_proc_frame(out, line)
        center = out["station"].map(self.proc_resid_center).astype(float).to_numpy()
        scale = out["station"].map(self.proc_scale).astype(float).to_numpy()
        out["expected_proc_s"] = exp_proc + np.nan_to_num(center)
        out["z_proc"] = (out["proc_time_s_mean"].to_numpy() - out["expected_proc_s"]) / scale

        bfrac = out["blocked_s_mean"].to_numpy() / self.takt_s
        sfrac = out["starved_s_mean"].to_numpy() / self.takt_s
        out["blocked_frac"] = bfrac
        out["starved_frac"] = sfrac
        out["d_blocked"] = bfrac - out["station"].map(self.blocked_frac_center).astype(float)
        out["d_starved"] = sfrac - out["station"].map(self.starved_frac_center).astype(float)

        for ch, name in [
            ("torque_nm_mean", "z_torque"),
            ("vibration_mm_s_mean", "z_vibration"),
            ("station_temp_c_mean", "z_temp"),
        ]:
            if ch in out.columns and ch in self.process_center:
                c = out["station"].map(self.process_center[ch]).astype(float)
                s = out["station"].map(self.process_scale[ch]).astype(float)
                out[name] = (out[ch] - c) / s
            else:
                out[name] = np.nan

        # Directional flow signal, in takt fractions. Positive means the station
        # cannot release (something ahead is slow); negative means it has nothing
        # to work on (something behind is slow).
        out["pressure"] = out["d_blocked"] - out["d_starved"]

        for col in ["z_proc", "d_blocked", "d_starved", "pressure"]:
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        return out
