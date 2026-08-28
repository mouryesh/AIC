"""Predictive defect risk: which vehicle is becoming more likely to carry a
defect, before any inspection gate has looked at it.

``twin.genealogy`` answers a retrospective question: a defect was *found* at
a gate, and which unmeasured station probably made it? This module answers a
different, forward-looking one: given the process evidence available right
now at a station, how likely is the vehicle currently there to be carrying
(or about to acquire) a defect -- before it ever reaches a gate.

Evidence and its source
------------------------
* ``z_proc`` -- processing-time deviation from the mix/shift-adjusted
  expectation, reusing ``features.baseline.NominalBaseline`` exactly as the
  flow path does (a rushed or strained cycle is measurably more defect-prone
  in the simulator's own physics -- see ``factory/simulator.py``'s
  ``p_def *= 1.0 + 0.9 * max(0, h - 1)``, so this is not an invented
  correlation).
* ``z_torque`` / ``z_vibration`` / ``z_temp`` -- RICH-tier process channels,
  scored against a *raw per-vehicle-visit* baseline (``RawProcessBaseline``,
  fit on the same disturbance-free reference run everything else uses).
  These are only available where the station carries process channels at
  all, which is exactly where confidence should be highest.
* The station's own structural defect-type profile
  (``Station.defect_profile`` -- the same process-FMEA-derived prior
  ``genealogy.py`` uses), so a torque anomaly at a sealer station is never
  reported as a torque-fault risk.

Confidence is capped by what evidence is actually available at that
station's tier: RICH stations get the full combination, BASIC stations get
timing evidence only, and MANUAL stations -- no automatic telemetry of any
kind -- have no process evidence to score at all, and the correct answer is
to abstain rather than invent a number.

The combination is a small, transparent logistic on the raw evidence
(weights and a bias/scale set by ``fit_scale`` against held-out nominal
data, not asserted), not a black box -- consistent with the rest of the
project's "physics/statistics first" design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..factory.topology import LineTopology

_MAD_TO_SIGMA = 1.4826
_RAW_CHANNELS = ("torque_nm", "vibration_mm_s", "station_temp_c")
_Z_NAMES = {"torque_nm": "z_torque", "vibration_mm_s": "z_vibration", "station_temp_c": "z_temp"}


@dataclass
class RawProcessBaseline:
    """Per-station, per-vehicle-visit expectation of the RICH process
    channels, fit directly on raw telemetry rather than window means.

    ``features.baseline.NominalBaseline`` already fits ``process_center`` /
    ``process_scale`` for these channels, but on *window-aggregated* means --
    correct for the flow path's per-window scoring, understated as a
    per-vehicle noise scale (a window mean over ~20 vehicles has much lower
    variance than a single reading). Scoring a single in-flight vehicle needs
    the per-vehicle scale, hence a separate small fit here rather than
    reusing the window-level one.
    """

    center: Dict[str, Dict[int, float]] = field(default_factory=dict)
    scale: Dict[str, Dict[int, float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, nominal_telemetry: pd.DataFrame, line: LineTopology) -> "RawProcessBaseline":
        obj = cls()
        for ch in _RAW_CHANNELS:
            if ch not in nominal_telemetry.columns:
                continue
            obj.center[ch] = {}
            obj.scale[ch] = {}
            for st, grp in nominal_telemetry.groupby("station"):
                vals = grp[ch].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size < 20:
                    continue
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med)) * _MAD_TO_SIGMA)
                obj.center[ch][int(st)] = med
                obj.scale[ch][int(st)] = max(mad, 0.05)
        return obj

    def score(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        out = telemetry.copy()
        for ch in _RAW_CHANNELS:
            zc = _Z_NAMES[ch]
            if ch not in out.columns or ch not in self.center:
                out[zc] = np.nan
                continue
            c = out["station"].map(self.center[ch]).astype(float)
            s = out["station"].map(self.scale[ch]).astype(float)
            out[zc] = (out[ch] - c) / s
        return out


@dataclass
class DefectRiskConfig:
    #: Weight on processing-time deviation (available at RICH and BASIC).
    w_z_proc: float = 0.30
    #: Weight on the mean of available RICH process-channel z-scores.
    w_z_process: float = 0.85
    #: Logistic bias/scale. Defaults are placeholders; ``fit_scale`` sets
    #: them from a held-out nominal run so ``risk`` reads near 0 when
    #: nothing is wrong, the same discipline ``ShadowSensor.calibrate`` uses.
    bias: float = -3.2
    scale: float = 1.0
    #: Confidence ceiling by tier -- BASIC has no process channels to
    #: corroborate a timing deviation, so it can never reach RICH's ceiling.
    confidence_rich: float = 0.85
    confidence_basic: float = 0.45

    def fit_scale(self, nominal_combined: np.ndarray) -> None:
        """Set bias/scale so the combined evidence reads near zero risk on
        held-out nominal data, and so one unit of evidence beyond the
        nominal spread moves risk by a fixed, checkable amount."""
        v = nominal_combined[np.isfinite(nominal_combined)]
        if v.size < 20:
            return
        mu, sd = float(np.mean(v)), float(np.std(v))
        sd = max(sd, 1e-6)
        # risk(mu) ~= 0.05, risk(mu + 4*sd) ~= 0.5 -- a combined-evidence
        # excursion of four nominal standard deviations is treated as the
        # point past which "probably a real drift" becomes the better bet.
        self.scale = float(2.94 / (4.0 * sd))
        self.bias = float(-self.scale * (mu + 4.0 * sd))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _combined_evidence(row: pd.Series, cfg: DefectRiskConfig) -> Tuple[float, dict]:
    zp = row.get("z_proc", np.nan)
    zs = [row.get(c) for c in ("z_torque", "z_vibration", "z_temp")]
    zs = [z for z in zs if z is not None and np.isfinite(z)]
    proc_term = max(0.0, float(zp)) if np.isfinite(zp) else 0.0
    process_term = max(0.0, float(np.mean(zs))) if zs else 0.0
    combined = cfg.w_z_proc * proc_term + cfg.w_z_process * process_term
    return combined, {"z_proc": zp, "z_torque": row.get("z_torque"),
                       "z_vibration": row.get("z_vibration"), "z_temp": row.get("z_temp"),
                       "n_process_channels": len(zs)}


@dataclass
class DefectRiskResult:
    vehicle_id: int
    station: int
    station_id: str
    tier: str
    is_hidden: bool
    risk: float
    predicted_defect_type: Optional[str]
    likely_origin_station_id: str
    expected_detection_gate: Optional[str]
    confidence: float
    abstained: bool
    evidence: List[str] = field(default_factory=list)


def score_vehicles(
    line: LineTopology,
    telemetry: pd.DataFrame,
    proc_baseline,  # features.baseline.NominalBaseline
    raw_baseline: RawProcessBaseline,
    cfg: Optional[DefectRiskConfig] = None,
) -> pd.DataFrame:
    """Score every (vehicle, station) row in ``telemetry`` for defect risk.

    Only rows for stations that emit telemetry at all can be scored --
    MANUAL stations have no process evidence, so the caller should treat
    those vehicle-visits as an explicit abstention (see
    ``predict_defect_risk``), not silently skip them.
    """
    cfg = cfg or DefectRiskConfig()
    df = raw_baseline.score(telemetry)

    # z_proc from the existing, already-fit NominalBaseline. Shift is
    # ignored here (telemetry alone does not carry it) -- a documented
    # simplification; mix (variant) is the dominant driver.
    def _z_proc(r) -> float:
        pv = proc_baseline.proc_by_variant.get(int(r["station"]))
        if not pv:
            return np.nan
        exp = pv.get(str(r["variant"]))
        if exp is None:
            return np.nan
        exp += proc_baseline.proc_resid_center.get(int(r["station"]), 0.0)
        scale = proc_baseline.proc_scale.get(int(r["station"]), np.nan)
        if not np.isfinite(scale) or scale <= 0:
            return np.nan
        return (float(r["proc_time_s"]) - exp) / scale

    df["z_proc"] = df.apply(_z_proc, axis=1)

    combined = np.full(len(df), np.nan)
    for i, (_, r) in enumerate(df.iterrows()):
        c, _ = _combined_evidence(r, cfg)
        combined[i] = c
    df["_combined"] = combined
    df["risk"] = _sigmoid(cfg.scale * combined + cfg.bias)
    return df


def coverage_gap_report(line: LineTopology) -> pd.DataFrame:
    """Stations with zero predictive defect-risk coverage.

    Unlike the flow path -- which infers a hidden station's state from
    conservation of material at its *neighbours* -- there is no equivalent
    physical channel for defect risk. A MANUAL station emits no telemetry
    for any vehicle at all (``factory/simulator.py`` never writes a
    telemetry row for it), so there is nothing to score there, for anyone,
    ever. ``twin.genealogy`` can still attribute an *already-found* defect
    back to such a station after the fact (a defect surfaces downstream and
    the failure-mode signature is structural evidence); a mid-flight risk
    prediction at the station itself is a genuine, different gap.

    Reported explicitly, the same way ``twin.placement`` reports where a new
    sensor buys the most for the flow path -- an unscored gap is exactly the
    place worth pointing a plant's next instrumentation budget.
    """
    rows = [
        {
            "station_id": s.station_id,
            "zone": s.zone,
            "reason": (
                "no process telemetry at all -- predictive defect risk "
                "cannot be scored here (post-hoc attribution via "
                "twin.genealogy still works once a defect is found downstream)"
            ),
        }
        for s in line.stations
        if s.is_hidden
    ]
    return pd.DataFrame(rows, columns=["station_id", "zone", "reason"])


def predict_defect_risk(
    line: LineTopology,
    scored_vehicles: pd.DataFrame,
    cfg: Optional[DefectRiskConfig] = None,
) -> List[DefectRiskResult]:
    """Turn scored (vehicle, station) rows into the vehicle-facing prediction
    shape: predicted defect type, probability, likely origin, expected
    detection point, confidence, evidence -- or an explicit abstention.
    """
    cfg = cfg or DefectRiskConfig()
    out: List[DefectRiskResult] = []
    for _, r in scored_vehicles.iterrows():
        station = int(r["station"])
        stn = line.stations[station]
        gate = line.next_inspection_after(station + 1)
        gate_id = line.stations[gate].inspection_id if gate is not None else None

        if stn.is_hidden:
            # Defensive only: telemetry never contains a MANUAL-station row
            # (see factory/simulator.py), so this branch is not reachable
            # through the normal score_vehicles -> predict_defect_risk path.
            # The actual coverage gap is reported by coverage_gap_report().
            out.append(
                DefectRiskResult(
                    vehicle_id=int(r["vehicle_id"]), station=station,
                    station_id=stn.station_id, tier=stn.tier, is_hidden=True,
                    risk=float("nan"), predicted_defect_type=None,
                    likely_origin_station_id=stn.station_id,
                    expected_detection_gate=gate_id, confidence=0.0, abstained=True,
                    evidence=["No process telemetry at this station -- no basis to "
                              "score defect risk. This is a genuine gap, not a "
                              "confident zero."],
                )
            )
            continue

        combined, parts = _combined_evidence(r, cfg)
        risk = float(_sigmoid(cfg.scale * combined + cfg.bias))
        n_rich = parts["n_process_channels"]
        ceiling = cfg.confidence_rich if n_rich > 0 else cfg.confidence_basic
        # Confidence scales with how far the evidence sits from the nominal
        # band, capped by what the station's tier can support -- a BASIC
        # station can never report RICH-level confidence, whatever the risk.
        confidence = float(ceiling * min(1.0, 0.5 + 0.5 * risk))

        profile = stn.defect_profile or {}
        pred_type = max(profile, key=profile.get) if profile else None

        ev: List[str] = []
        if np.isfinite(parts["z_proc"]):
            ev.append(f"cycle time deviation z={parts['z_proc']:.1f}")
        if np.isfinite(parts.get("z_torque", np.nan)):
            ev.append(f"torque deviation z={parts['z_torque']:.1f}")
        if np.isfinite(parts.get("z_vibration", np.nan)):
            ev.append(f"vibration deviation z={parts['z_vibration']:.1f}")
        if np.isfinite(parts.get("z_temp", np.nan)):
            ev.append(f"temperature deviation z={parts['z_temp']:.1f}")
        if not ev:
            ev.append("no usable process evidence this visit")

        out.append(
            DefectRiskResult(
                vehicle_id=int(r["vehicle_id"]), station=station,
                station_id=stn.station_id, tier=stn.tier, is_hidden=False,
                risk=risk, predicted_defect_type=pred_type,
                likely_origin_station_id=stn.station_id,
                expected_detection_gate=gate_id, confidence=confidence,
                abstained=False, evidence=ev,
            )
        )
    return out
