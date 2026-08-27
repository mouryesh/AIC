"""Discrete-event simulator for a mixed-model serial assembly line.

Why this exists
---------------
RippleTwin claims it can infer the state of a station that has no sensor. That
claim is only testable if we have ground truth for hidden stations -- which no
real plant will hand out. So we build a line whose *physics* is explicit, run
disturbances through it, and then deliberately throw away the telemetry from a
subset of stations before the model ever sees it.

The physics
-----------
A serial line with finite buffers is governed by three recursions. For vehicle
``v`` at station ``i`` with outbound buffer capacity ``B_i``:

    start_i(v)     = max( departure_{i-1}(v), departure_i(v-1) )
    end_i(v)       = start_i(v) + proc_i(v)
    departure_i(v) = max( end_i(v), start_{i+1}(v - B_i) )

From these fall out the two signals that make shadow-sensing possible:

    starved_i(v) = max(0, departure_{i-1}(v) - departure_i(v-1))
        -- the station was free and idle, waiting for a part to arrive.

    blocked_i(v) = departure_i(v) - end_i(v)
        -- the station had finished, but could not release: the buffer ahead was full.

If station ``k`` slows down, material-flow conservation forces a *directional,
asymmetric* signature: every station downstream of k starves (parts stop
arriving) and every station upstream of k eventually blocks (parts stop leaving).
The sign flip in that profile sits exactly at k. That is not a learned
correlation -- it is a consequence of conservation of material through a serial
line, and it is what lets us localise a fault at a station we cannot measure.

Note the timing asymmetry, which the simulator reproduces faithfully:
starvation downstream appears within a few vehicles, while upstream blocking
only appears once the intervening buffer saturates. Warning lead time therefore
comes mostly from the downstream side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .topology import LineTopology, TIER_RICH

# --------------------------------------------------------------------- events

#: A station physically degrades: processing time inflates. Produces a bottleneck.
EVENT_SLOWDOWN = "SLOWDOWN"
#: A station starts injecting defects with little or no change in cycle time.
#: This is the genuinely hard case: nothing about the *timing* of the line changes much.
EVENT_QUALITY_DRIFT = "QUALITY_DRIFT"
#: Tooling wear: both slower and more defect-prone.
EVENT_COMBINED = "COMBINED"
#: Intermittent short stoppages rather than sustained slowdown.
EVENT_MICROSTOP_BURST = "MICROSTOP_BURST"
#: Upstream supply interruption -- starves the whole line from station 0.
EVENT_MATERIAL_DELAY = "MATERIAL_DELAY"

from .topology import _ZONE_DEFECT_TYPES as DEFECT_TYPES


@dataclass
class Disturbance:
    """A ground-truth disturbance injected into the line.

    This object is written to the ground-truth table and used for evaluation
    only. It is never exposed to the model or the dashboard's inference path.
    """

    station: int
    kind: str
    t_start_s: float
    t_end_s: float
    #: Peak multiplier. For SLOWDOWN/COMBINED this scales processing time
    #: (1.30 == 30% slower). For QUALITY_DRIFT/COMBINED it scales defect rate.
    magnitude: float
    #: Seconds over which the disturbance ramps from nominal to full magnitude.
    ramp_s: float = 900.0
    label: str = ""

    def intensity(self, t: float) -> float:
        """Fraction of full magnitude active at time ``t`` (0.0 .. 1.0)."""
        if t < self.t_start_s or t > self.t_end_s:
            return 0.0
        if self.ramp_s <= 0:
            return 1.0
        return float(min(1.0, (t - self.t_start_s) / self.ramp_s))


@dataclass
class SimResult:
    """Everything a run produces, split by what the plant can and cannot see."""

    #: Per vehicle x station ground truth. Includes hidden stations. EVAL ONLY.
    passes: pd.DataFrame
    #: What the plant actually records. Observed stations only. MODEL INPUT.
    telemetry: pd.DataFrame
    #: Inspection gate results. MODEL INPUT.
    inspections: pd.DataFrame
    #: Vehicle release log (variant, shift). MODEL INPUT.
    vehicles: pd.DataFrame
    #: Ambient conditions. MODEL INPUT.
    environment: pd.DataFrame
    #: Injected disturbances. EVAL ONLY.
    disturbances: pd.DataFrame
    #: Defects with their true source station. EVAL ONLY (source is never given to the model).
    defects: pd.DataFrame
    meta: dict = field(default_factory=dict)


class LineSimulator:
    """Simulates production over a horizon, with optional injected disturbances."""

    def __init__(
        self,
        line: LineTopology,
        seed: int = 42,
        start: datetime | None = None,
    ) -> None:
        self.line = line
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.start = start or datetime(2026, 3, 2, 6, 0, 0)  # a Monday, shift A

    # ------------------------------------------------------------------ helpers

    def _shift_for(self, t_abs: datetime) -> dict:
        hour = t_abs.hour
        for sh in self.line.shifts:
            s0 = sh["start_hour"]
            s1 = (s0 + sh["hours"]) % 24
            if s0 < s1:
                if s0 <= hour < s1:
                    return sh
            else:  # wraps midnight
                if hour >= s0 or hour < s1:
                    return sh
        return self.line.shifts[0]

    def _environment(self, n_steps: int, step_s: float) -> pd.DataFrame:
        """Ambient temperature and humidity on a daily cycle, with noise.

        Humidity matters: it is a genuine confounder for paint defects. A model
        that blames a paint station for a humidity-driven defect spike is wrong,
        and we want that failure mode to be *possible* in this dataset.
        """
        env_cfg = self.line.environment
        t = np.arange(n_steps) * step_s
        day_phase = 2 * np.pi * (t / 86400.0)

        tc = env_cfg["ambient_temp_c"]
        temp = (
            tc["mean"]
            + tc["daily_amplitude"] * np.sin(day_phase - np.pi / 2)
            + self.rng.normal(0, tc["noise"], n_steps)
        )
        hc = env_cfg["humidity_pct"]
        hum = (
            hc["mean"]
            + hc["daily_amplitude"] * np.sin(day_phase - np.pi / 2 + 0.6)
            + self.rng.normal(0, hc["noise"], n_steps)
        )
        # A slow multi-hour humidity excursion, so the confounder is not purely diurnal.
        n_exc = max(1, n_steps // 4000)
        for _ in range(n_exc):
            c = self.rng.integers(0, n_steps)
            w = self.rng.integers(n_steps // 60 or 1, n_steps // 20 or 2)
            lo, hi = max(0, c - w), min(n_steps, c + w)
            bump = 18.0 * np.exp(-0.5 * ((np.arange(lo, hi) - c) / (w / 2.2)) ** 2)
            hum[lo:hi] += bump

        return pd.DataFrame(
            {
                "t_s": t,
                "timestamp": [self.start + timedelta(seconds=float(x)) for x in t],
                "ambient_temp_c": temp,
                "humidity_pct": np.clip(hum, 5, 99),
            }
        )

    # ------------------------------------------------------------------- run

    def run(
        self,
        n_vehicles: int,
        disturbances: Sequence[Disturbance] = (),
        run_id: str = "run",
    ) -> SimResult:
        line = self.line
        n = line.n_stations
        rng = self.rng
        takt = line.takt_s

        # ---- vehicle schedule -------------------------------------------------
        variants = list(line.variants.keys())
        shares = np.array([line.variants[v]["share"] for v in variants], float)
        shares = shares / shares.sum()
        # Build the model mix in shuffled blocks: real mixed-model lines run a
        # levelled sequence, not an i.i.d. draw, and the block structure creates
        # genuine slow-moving load variation the model must not mistake for a fault.
        block = 20
        seq: List[str] = []
        while len(seq) < n_vehicles:
            counts = np.round(shares * block).astype(int)
            counts[0] += block - counts.sum()
            blk = [variants[i] for i, c in enumerate(counts) for _ in range(max(0, c))]
            rng.shuffle(blk)
            seq.extend(blk)
        veh_variant = np.array(seq[:n_vehicles])

        release = np.arange(n_vehicles, dtype=float) * takt
        release += rng.normal(0, takt * 0.012, n_vehicles)  # release jitter
        release = np.maximum.accumulate(release)

        # Material delays push the whole release schedule back.
        for d in disturbances:
            if d.kind == EVENT_MATERIAL_DELAY:
                mask = release >= d.t_start_s
                dur = min(d.t_end_s - d.t_start_s, (d.t_end_s - d.t_start_s))
                release[mask] += dur * d.magnitude
                release = np.maximum.accumulate(release)

        horizon_s = float(release[-1] + n * takt * 2)
        env_step = 60.0
        env = self._environment(int(horizon_s / env_step) + 2, env_step)
        env_hum = env["humidity_pct"].to_numpy()
        env_temp = env["ambient_temp_c"].to_numpy()

        def env_at(t: float) -> tuple:
            k = int(min(len(env_hum) - 1, max(0, t // env_step)))
            return env_temp[k], env_hum[k]

        # ---- pre-compute per-station constants --------------------------------
        base_cycle = np.array([s.base_cycle_s for s in line.stations])
        manual_content = np.array([s.manual_content for s in line.stations])
        noise_cv = np.array([s.process_noise_cv for s in line.stations])
        ms_rate = np.array([s.microstop_rate for s in line.stations])
        ms_lo = np.array([s.microstop_range_s[0] for s in line.stations])
        ms_hi = np.array([s.microstop_range_s[1] for s in line.stations])
        out_buf = np.array([s.out_buffer for s in line.stations])
        base_defect = np.array([s.base_defect_rate for s in line.stations])
        zone_of = [s.zone for s in line.stations]

        # Slow, benign per-station drift over the horizon (tool wear, seasonal).
        # Present in *every* station so the model cannot treat any drift as a fault.
        drift_amp = rng.normal(0, 0.012, n)
        drift_phase = rng.uniform(0, 2 * np.pi, n)

        variant_wf = {
            v: np.array([line.variants[v]["work_factor"][z] for z in zone_of])
            for v in variants
        }
        variant_sus = {v: line.variants[v]["defect_susceptibility"] for v in variants}

        # Disturbance lookup by station.
        dist_by_station: Dict[int, List[Disturbance]] = {}
        for d in disturbances:
            if d.kind == EVENT_MATERIAL_DELAY:
                continue
            dist_by_station.setdefault(d.station, []).append(d)

        # ---- state arrays -----------------------------------------------------
        start = np.zeros((n_vehicles, n))
        end = np.zeros((n_vehicles, n))
        dep = np.zeros((n_vehicles, n))
        proc = np.zeros((n_vehicles, n))
        blocked = np.zeros((n_vehicles, n))
        starved = np.zeros((n_vehicles, n))
        health = np.ones((n_vehicles, n))          # ground-truth slowdown multiplier
        microstop = np.zeros((n_vehicles, n))
        defect_mult = np.ones((n_vehicles, n))     # ground-truth defect multiplier

        # RICH process channels
        torque = np.full((n_vehicles, n), np.nan)
        vibration = np.full((n_vehicles, n), np.nan)
        station_temp = np.full((n_vehicles, n), np.nan)

        defect_rows: List[dict] = []
        # carried[v] -> list of open defect dicts riding on that vehicle
        carried: Dict[int, List[dict]] = {}
        inspect_rows: List[dict] = []

        shift_cache: Dict[int, dict] = {}

        for v in range(n_vehicles):
            var = veh_variant[v]
            wf = variant_wf[var]
            sus = variant_sus[var]

            # Shift is set by the time the vehicle is released.
            hr_key = int(release[v] // 3600)
            if hr_key not in shift_cache:
                shift_cache[hr_key] = self._shift_for(
                    self.start + timedelta(seconds=float(release[v]))
                )
            sh = shift_cache[hr_key]
            op_base = sh["operator_factor"]
            op_cv = sh["operator_cv"]

            prev_dep = release[v]

            for i in range(n):
                # ---- processing time -----------------------------------------
                t_ref = prev_dep  # time the vehicle arrives at this station

                h = 1.0
                dmult = 1.0
                for d in dist_by_station.get(i, ()):
                    inten = d.intensity(t_ref)
                    if inten <= 0:
                        continue
                    if d.kind in (EVENT_SLOWDOWN, EVENT_COMBINED):
                        h *= 1.0 + (d.magnitude - 1.0) * inten
                    if d.kind in (EVENT_QUALITY_DRIFT, EVENT_COMBINED):
                        dmult *= 1.0 + (d.magnitude - 1.0) * inten
                    if d.kind == EVENT_MICROSTOP_BURST:
                        # handled below via elevated stoppage probability
                        pass
                health[v, i] = h
                defect_mult[v, i] = dmult

                # operator variation applies only to the manual share of the work
                op_draw = rng.normal(op_base, op_cv)
                op_mult = 1.0 + manual_content[i] * (op_draw - 1.0)

                drift = 1.0 + drift_amp[i] * np.sin(
                    2 * np.pi * t_ref / (86400.0 * 3.0) + drift_phase[i]
                )

                lognoise = float(np.exp(rng.normal(0, noise_cv[i]) - 0.5 * noise_cv[i] ** 2))

                p = base_cycle[i] * wf[i] * op_mult * h * drift * lognoise

                # ---- micro-stops ---------------------------------------------
                rate = ms_rate[i]
                for d in dist_by_station.get(i, ()):
                    if d.kind == EVENT_MICROSTOP_BURST:
                        rate *= 1.0 + (d.magnitude - 1.0) * d.intensity(t_ref)
                if rng.random() < rate:
                    ms = float(rng.uniform(ms_lo[i], ms_hi[i]))
                    microstop[v, i] = ms
                    p += ms
                proc[v, i] = p

                # ---- flow recursion -------------------------------------------
                prev_station_dep_same_veh = prev_dep
                prev_veh_dep_same_station = dep[v - 1, i] if v > 0 else -np.inf

                st = max(prev_station_dep_same_veh, prev_veh_dep_same_station)
                start[v, i] = st
                starved[v, i] = max(
                    0.0,
                    prev_station_dep_same_veh - prev_veh_dep_same_station
                    if v > 0
                    else 0.0,
                )
                end[v, i] = st + p

                # departure is gated by room in the outbound buffer
                b = out_buf[i]
                if i < n - 1 and v - b >= 0:
                    dep[v, i] = max(end[v, i], start[v - b, i + 1])
                else:
                    dep[v, i] = end[v, i]
                blocked[v, i] = dep[v, i] - end[v, i]

                # ---- process channels ------------------------------------------
                # Computed for every station regardless of tier. Which of them a
                # given experiment is allowed to *see* is decided later by
                # ``telemetry_view``, so that the sensor-coverage sweep can vary
                # observability over one fixed physics run instead of re-rolling
                # the dice for every coverage level.
                if True:
                    amb_t, _ = env_at(t_ref)
                    # torque rises with mechanical degradation; vibration more so
                    torque[v, i] = 118.0 * (1.0 + 0.55 * (h - 1.0)) + rng.normal(0, 2.4)
                    vibration[v, i] = 1.85 * (1.0 + 2.10 * (h - 1.0)) + rng.normal(0, 0.16)
                    station_temp[v, i] = (
                        amb_t + 12.0 + 9.0 * (h - 1.0) + rng.normal(0, 0.7)
                    )

                # ---- defect injection ------------------------------------------
                z = zone_of[i]
                p_def = base_defect[i] * sus * dmult
                if z == "PAINT":
                    _, hum = env_at(t_ref)
                    if hum > 65.0:
                        p_def += self.line.environment["paint_humidity_sensitivity"] * (
                            hum - 65.0
                        )
                # A slowed station is also somewhat more defect-prone even when the
                # disturbance is nominally a pure SLOWDOWN (rushed / strained process).
                p_def *= 1.0 + 0.9 * max(0.0, h - 1.0)

                if rng.random() < p_def:
                    # Draw the failure mode from this station's own propensity
                    # profile, not uniformly across the zone.
                    prof = line.stations[i].defect_profile
                    if prof:
                        types = list(prof.keys())
                        dtype = str(rng.choice(types, p=np.array([prof[t] for t in types])))
                    else:
                        dtype = DEFECT_TYPES[z][int(rng.integers(0, len(DEFECT_TYPES[z])))]
                    sev = float(np.clip(rng.gamma(2.2, 0.20), 0.05, 1.0))
                    rec = {
                        "vehicle_id": v,
                        "source_station": i,
                        "source_station_id": line.stations[i].station_id,
                        "source_zone": z,
                        "defect_type": dtype,
                        "severity": sev,
                        "t_injected_s": float(start[v, i]),
                        "detected_at": None,
                        "detected_station_id": None,
                        "escaped": True,
                    }
                    defect_rows.append(rec)
                    carried.setdefault(v, []).append(rec)

                # ---- inspection --------------------------------------------------
                stn = line.stations[i]
                if stn.is_inspection:
                    open_defects = [
                        d for d in carried.get(v, ()) if d["detected_at"] is None
                    ]
                    found = []
                    for d in open_defects:
                        if d["source_zone"] not in stn.inspection_covers:
                            continue
                        # detection probability scales with severity
                        p_detect = stn.inspection_detect_prob * (0.45 + 0.55 * d["severity"])
                        if rng.random() < p_detect:
                            d["detected_at"] = i
                            d["detected_station_id"] = stn.station_id
                            d["escaped"] = False
                            found.append(d)
                    inspect_rows.append(
                        {
                            "vehicle_id": v,
                            "station": i,
                            "station_id": stn.station_id,
                            "gate_id": stn.inspection_id,
                            "t_s": float(end[v, i]),
                            "result": "FAIL" if found else "PASS",
                            "n_defects_found": len(found),
                            "defect_types": "|".join(sorted(d["defect_type"] for d in found)),
                            "max_severity": max((d["severity"] for d in found), default=0.0),
                            "variant": var,
                        }
                    )
                    # A failed vehicle is reworked -- rework consumes extra time.
                    if found:
                        rework = float(rng.uniform(45, 220) * max(d["severity"] for d in found))
                        dep[v, i] += rework
                        blocked[v, i] = dep[v, i] - end[v, i]

                prev_dep = dep[v, i]

        # ---- assemble frames ---------------------------------------------------
        veh_idx = np.repeat(np.arange(n_vehicles), n)
        stn_idx = np.tile(np.arange(n), n_vehicles)

        passes = pd.DataFrame(
            {
                "vehicle_id": veh_idx,
                "station": stn_idx,
                "station_id": [line.stations[i].station_id for i in stn_idx],
                "zone": [zone_of[i] for i in stn_idx],
                "variant": np.repeat(veh_variant, n),
                "t_start_s": start.ravel(),
                "t_end_s": end.ravel(),
                "t_depart_s": dep.ravel(),
                "proc_time_s": proc.ravel(),
                "blocked_s": blocked.ravel(),
                "starved_s": starved.ravel(),
                "microstop_s": microstop.ravel(),
                "true_health": health.ravel(),
                "true_defect_mult": defect_mult.ravel(),
                "tier": [line.stations[i].tier for i in stn_idx],
            }
        )
        # cycle time as a plant would compute it: departure-to-departure occupancy
        passes["cycle_time_s"] = passes["t_depart_s"] - passes["t_start_s"]

        # ---- telemetry: ONLY observed stations ---------------------------------
        obs = set(line.observed_indices)
        tel = passes[passes["station"].isin(obs)].copy()
        tel = tel[
            [
                "vehicle_id",
                "station",
                "station_id",
                "zone",
                "variant",
                "t_start_s",
                "t_depart_s",
                "cycle_time_s",
                "proc_time_s",
                "blocked_s",
                "starved_s",
            ]
        ].reset_index(drop=True)

        rich = set(line.rich_indices)
        tq = torque.ravel()
        vb = vibration.ravel()
        stmp = station_temp.ravel()
        sel = passes["station"].isin(obs).to_numpy()
        tel["torque_nm"] = tq[sel]
        tel["vibration_mm_s"] = vb[sel]
        tel["station_temp_c"] = stmp[sel]
        tel["has_process_channels"] = tel["station"].isin(rich)

        # Buffer occupancy is only derivable when BOTH endpoints of the arc are
        # observed -- you need departures out of i and starts into i+1.
        buf_rows = []
        for i in sorted(obs):
            if i + 1 in obs:
                dep_i = dep[:, i]
                start_next = start[:, i + 1]
                # occupancy seen by vehicle v = how many earlier vehicles have left i
                # but not yet started i+1, at the moment v departs i
                occ = np.searchsorted(np.sort(start_next), dep_i, side="left")
                level = np.arange(len(dep_i)) - occ + 1
                buf_rows.append(
                    pd.DataFrame(
                        {
                            "vehicle_id": np.arange(n_vehicles),
                            "station": i,
                            "buffer_level": np.clip(level, 0, out_buf[i]),
                            "buffer_capacity": out_buf[i],
                        }
                    )
                )
        if buf_rows:
            buf = pd.concat(buf_rows, ignore_index=True)
            tel = tel.merge(buf, on=["vehicle_id", "station"], how="left")
        else:
            tel["buffer_level"] = np.nan
            tel["buffer_capacity"] = np.nan

        tel["timestamp"] = [
            self.start + timedelta(seconds=float(x)) for x in tel["t_depart_s"]
        ]

        vehicles = pd.DataFrame(
            {
                "vehicle_id": np.arange(n_vehicles),
                "variant": veh_variant,
                "release_t_s": release,
                "release_timestamp": [
                    self.start + timedelta(seconds=float(x)) for x in release
                ],
                "shift": [
                    self._shift_for(self.start + timedelta(seconds=float(x)))["id"]
                    for x in release
                ],
            }
        )

        defects = pd.DataFrame(defect_rows) if defect_rows else pd.DataFrame(
            columns=[
                "vehicle_id", "source_station", "source_station_id", "source_zone",
                "defect_type", "severity", "t_injected_s", "detected_at",
                "detected_station_id", "escaped",
            ]
        )
        inspections = pd.DataFrame(inspect_rows)
        if not inspections.empty:
            inspections["timestamp"] = [
                self.start + timedelta(seconds=float(x)) for x in inspections["t_s"]
            ]

        dist_rows = [
            {
                "station": d.station,
                "station_id": line.stations[d.station].station_id
                if d.kind != EVENT_MATERIAL_DELAY
                else "LINE",
                "kind": d.kind,
                "t_start_s": d.t_start_s,
                "t_end_s": d.t_end_s,
                "magnitude": d.magnitude,
                "ramp_s": d.ramp_s,
                "label": d.label,
                "tier": line.stations[d.station].tier
                if d.kind != EVENT_MATERIAL_DELAY
                else "LINE",
            }
            for d in disturbances
        ]
        dist_df = pd.DataFrame(dist_rows) if dist_rows else pd.DataFrame(
            columns=["station", "station_id", "kind", "t_start_s", "t_end_s",
                     "magnitude", "ramp_s", "label", "tier"]
        )

        meta = {
            "run_id": run_id,
            "seed": self.seed,
            "n_vehicles": int(n_vehicles),
            "n_stations": int(n),
            "start": self.start.isoformat(),
            "horizon_s": float(dep[-1, -1]),
            "coverage": float(line.coverage),
            "n_observed": len(obs),
            "n_hidden": len(line.hidden_indices),
            "throughput_vph": float(n_vehicles / (dep[-1, -1] / 3600.0)),
            "n_defects": int(len(defects)),
            "n_escaped": int(defects["escaped"].sum()) if len(defects) else 0,
        }

        return SimResult(
            passes=passes,
            telemetry=tel,
            inspections=inspections,
            vehicles=vehicles,
            environment=env,
            disturbances=dist_df,
            defects=defects,
            meta=meta,
        )
