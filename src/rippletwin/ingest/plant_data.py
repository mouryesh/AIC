"""The data a plant actually hands over, and what we do when it is incomplete.

Why this module exists
----------------------
Before this, every entry point into the twin was typed to ``SimResult`` -- the
simulator's output, which carries ground-truth tables alongside observable
telemetry. That had two consequences, one embarrassing and one dangerous.

The embarrassing one: **a real plant could not run this software.** A plant has
a historian export, not a ``SimResult``. The method could be perfect and it
would still have been undeployable.

The dangerous one: the separation between "what the plant can see" and "what
only the evaluator can see" was a *convention* -- a comment saying EVAL ONLY and
the discipline not to touch it. Conventions leak. ``PlantData`` makes it
structural: there is no field on this object that holds the answer, so the
inference path cannot read one even by mistake.

Degrading instead of crashing
-----------------------------
The data contract in ``rippletwin.integrate.contract`` lists five required
signals and five optional ones. The windowing code, written against the
simulator, quietly assumed several of the *optional* ones were always present:
buffer occupancy, per-vehicle cycle time, ambient conditions. A plant with no
conveyor counters -- which the contract explicitly says is fine -- would have
crashed.

So this module derives what it can and marks the rest absent:

* ``proc_time_s`` missing -> derived as
  ``(t_depart - t_start) - blocked - starved``, because occupancy decomposes as
  ``starved + processing + blocked``: the station waits for the part, works on
  it, then may hold a finished unit it cannot hand on.
* ``buffer_level`` / ``buffer_capacity`` missing -> filled with NaN, which the
  aggregator now tolerates. Buffer fill becomes unavailable; localisation does
  not depend on it.
* ``environment`` missing -> ambient columns are simply not attached.
* ``inspections`` missing -> the flow path runs, the quality path does not.
  This is the ``FLOW_ONLY`` capability, and it is the common case.

Units
-----
Plants export milliseconds about as often as seconds, and a silent factor of
1000 would not raise an error anywhere -- it would just produce confident
nonsense. ``validate()`` checks the observed cycle time against the configured
takt and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: Columns the flow path cannot run without.
REQUIRED_TELEMETRY = ["vehicle_id", "station", "t_depart_s", "blocked_s", "starved_s"]

#: Columns we use when present and derive or skip when absent.
OPTIONAL_TELEMETRY = [
    "t_start_s",
    "proc_time_s",
    "buffer_level",
    "buffer_capacity",
    "torque_nm",
    "vibration_mm_s",
    "station_temp_c",
    "variant",
    "station_id",
    "timestamp",
]

REQUIRED_VEHICLES = ["vehicle_id"]

#: Severity ordering for validation findings.
BLOCKER = "BLOCKER"
WARNING = "WARNING"
NOTE = "NOTE"


@dataclass
class ValidationIssue:
    """One finding about a plant's data, with what it costs us."""

    severity: str
    code: str
    message: str
    #: What the twin loses if this is not fixed.
    consequence: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        tail = f" -> {self.consequence}" if self.consequence else ""
        return f"[{self.severity}] {self.code}: {self.message}{tail}"


@dataclass
class ValidationReport:
    """The answer to 'is this export usable?', before anyone fits anything."""

    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def blockers(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == BLOCKER]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        """True when the twin can run at all on this data."""
        return not self.blockers

    def add(self, severity: str, code: str, message: str, consequence: str = "") -> None:
        self.issues.append(ValidationIssue(severity, code, message, consequence))

    def summary(self) -> str:
        lines = [
            "DATA VALIDATION",
            f"  verdict          : {'USABLE' if self.ok else 'NOT USABLE'}",
        ]
        for k, v in self.stats.items():
            lines.append(f"  {k:<17}: {v}")
        if not self.issues:
            lines.append("  no issues found")
        for i in self.issues:
            lines.append(f"  {i}")
        return "\n".join(lines)


@dataclass
class PlantData:
    """Observable production data. Contains no ground truth, by construction.

    This is what the twin consumes. A simulated run produces one via
    ``SimResult.as_plant_data()``; a real line produces one via
    ``rippletwin.ingest.csv_adapter``. The inference path cannot tell the
    difference, which is the entire point.
    """

    #: One row per vehicle per *instrumented* station.
    telemetry: pd.DataFrame
    #: Build sequence: vehicle_id, variant, shift.
    vehicles: pd.DataFrame
    #: Gate results. ``None`` when the plant has no per-VIN defect coding.
    inspections: Optional[pd.DataFrame] = None
    #: Ambient conditions. ``None`` when unavailable.
    environment: Optional[pd.DataFrame] = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- properties

    @property
    def has_quality_path(self) -> bool:
        return self.inspections is not None and not self.inspections.empty

    @property
    def has_environment(self) -> bool:
        return self.environment is not None and not self.environment.empty

    @property
    def observed_stations(self) -> List[int]:
        return sorted(self.telemetry["station"].unique().tolist())

    @property
    def n_vehicles(self) -> int:
        return int(self.vehicles["vehicle_id"].max()) + 1

    def contract_signals(self) -> List[str]:
        """Which contract signals this export actually carries.

        Feeds ``assess_readiness`` so the Phase 0 verdict is computed from the
        data in hand rather than from what somebody said they had.
        """
        have = ["station_state", "vehicle_identity", "line_topology"]
        if "variant" in self.vehicles.columns:
            have.append("build_sequence")
        if "shift" in self.vehicles.columns:
            have.append("shift_calendar")
        if self.has_quality_path:
            have.append("inspection_results")
        tel = self.telemetry.columns
        if "proc_time_s" in tel and self.telemetry["proc_time_s"].notna().any():
            have.append("station_cycle_time")
        if any(c in tel for c in ("torque_nm", "vibration_mm_s", "station_temp_c")):
            have.append("process_channels")
        if "buffer_level" in tel and self.telemetry["buffer_level"].notna().any():
            have.append("buffer_level")
        return have

    # ------------------------------------------------------------ derivation

    @classmethod
    def from_frames(
        cls,
        telemetry: pd.DataFrame,
        vehicles: pd.DataFrame,
        inspections: Optional[pd.DataFrame] = None,
        environment: Optional[pd.DataFrame] = None,
        meta: Optional[dict] = None,
    ) -> "PlantData":
        """Normalise an export: fill derivable columns, leave the rest absent.

        Deliberately does not raise on missing optional data. Call
        ``validate()`` to find out what was missing and what it cost.
        """
        tel = telemetry.copy()

        # Processing time is optional in the contract but used everywhere. A
        # station's processing time is how long it held the unit minus the time
        # it spent blocked -- holding a finished unit it could not pass on.
        if "proc_time_s" not in tel.columns:
            if "t_start_s" in tel.columns:
                # ``t_start_s`` is the start of *occupancy* -- the instant the
                # unit became this station's responsibility, i.e. when its
                # predecessor departed. That is what a VIN-scan join can
                # actually observe, and it is not the same as the instant work
                # began: between the two the station may sit starved, waiting
                # for the part to arrive.
                #
                #     occupancy = starved + processing + blocked
                #
                # so processing is occupancy minus BOTH flow losses. Subtracting
                # only ``blocked`` inflates processing time by exactly the
                # starvation, which is not a small error -- it is the channel
                # the localisation reads. Verified against simulator ground
                # truth: the residual equalled ``starved_s`` to 0.0 with
                # correlation 1.000 before this term was added.
                tel["proc_time_s"] = (
                    tel["t_depart_s"]
                    - tel["t_start_s"]
                    - tel["blocked_s"].fillna(0.0)
                    - tel["starved_s"].fillna(0.0)
                ).clip(lower=0.0)
                tel.attrs["proc_time_derived"] = True
            else:
                tel["proc_time_s"] = np.nan

        # Buffer occupancy is explicitly optional. NaN propagates to an
        # unavailable buffer_fill rather than an exception.
        for col in ("buffer_level", "buffer_capacity"):
            if col not in tel.columns:
                tel[col] = np.nan

        if "variant" not in tel.columns and "variant" in vehicles.columns:
            tel = tel.merge(vehicles[["vehicle_id", "variant"]], on="vehicle_id",
                            how="left")
        if "variant" not in tel.columns:
            tel["variant"] = "UNKNOWN"

        veh = vehicles.copy()
        if "shift" not in veh.columns:
            veh["shift"] = "UNKNOWN"
        if "variant" not in veh.columns:
            veh["variant"] = "UNKNOWN"

        return cls(
            telemetry=tel,
            vehicles=veh,
            inspections=inspections,
            environment=environment,
            meta=meta or {},
        )

    # ------------------------------------------------------------ validation

    def validate(
        self,
        n_stations: Optional[int] = None,
        takt_s: Optional[float] = None,
        clock_sync_s: Optional[float] = None,
    ) -> ValidationReport:
        """Check an export for the things that actually break pilots."""
        r = ValidationReport()
        tel = self.telemetry

        missing = [c for c in REQUIRED_TELEMETRY if c not in tel.columns]
        if missing:
            r.add(BLOCKER, "MISSING_COLUMNS",
                  f"telemetry lacks {', '.join(missing)}",
                  "the flow path cannot run")
            return r  # nothing further is meaningful

        if tel.empty:
            r.add(BLOCKER, "EMPTY", "telemetry has no rows", "nothing to infer from")
            return r

        n_obs = tel["station"].nunique()
        r.stats["rows"] = len(tel)
        r.stats["vehicles"] = int(tel["vehicle_id"].nunique())
        r.stats["stations_observed"] = int(n_obs)

        if n_stations:
            cov = n_obs / float(n_stations)
            r.stats["coverage"] = round(cov, 3)
            if n_obs < 2:
                r.add(BLOCKER, "COVERAGE",
                      f"only {n_obs} instrumented station(s)",
                      "shadow-sensing needs sensors either side of a blind station")
            elif cov >= 0.999:
                r.add(NOTE, "FULL_COVERAGE",
                      "every station is instrumented",
                      "shadow-sensing adds nothing here; a conventional twin suffices")
            elif cov < 0.30:
                r.add(WARNING, "LOW_COVERAGE",
                      f"coverage is {cov:.0%}",
                      "our evaluation shows exact localisation falls sharply below ~50%")

        # Duplicate (vehicle, station) rows silently double-count a station's
        # blocked time, which shifts the turning point.
        dup = tel.duplicated(["vehicle_id", "station"]).sum()
        if dup:
            r.add(WARNING, "DUPLICATE_ROWS",
                  f"{dup} duplicate (vehicle, station) rows",
                  "double-counted dwell time biases localisation")

        # Units. A silent seconds/milliseconds mix-up produces confident
        # nonsense rather than an error, so check it explicitly.
        med_proc = float(np.nanmedian(tel["proc_time_s"])) if \
            tel["proc_time_s"].notna().any() else float("nan")
        if takt_s and np.isfinite(med_proc) and med_proc > 0:
            ratio = med_proc / takt_s
            r.stats["median_proc_s"] = round(med_proc, 2)
            if ratio > 50:
                r.add(BLOCKER, "UNITS",
                      f"median processing time {med_proc:.0f} vs takt {takt_s:.0f}s "
                      f"({ratio:.0f}x)",
                      "durations look like milliseconds; rescale before running")
            elif ratio > 3 or ratio < 0.02:
                r.add(WARNING, "UNITS_SUSPECT",
                      f"median processing time is {ratio:.2f}x takt",
                      "check the unit of duration columns")

        # Negative durations mean the export is malformed, not that the line
        # ran backwards.
        for col in ("blocked_s", "starved_s", "proc_time_s"):
            neg = int((tel[col] < 0).sum())
            if neg:
                r.add(WARNING, "NEGATIVE_DURATION",
                      f"{neg} negative values in {col}",
                      "usually a timestamp ordering or clock problem")

        if tel["proc_time_s"].isna().all():
            r.add(WARNING, "NO_CYCLE_TIME",
                  "no per-vehicle cycle time and no start timestamps to derive it",
                  "blocked/starved localisation still runs; "
                  "hidden cycle-time estimation does not")

        if tel["buffer_level"].isna().all():
            r.add(NOTE, "NO_BUFFER_LEVEL",
                  "no buffer occupancy",
                  "optional; buffer fill will be unavailable")

        if not self.has_quality_path:
            r.add(NOTE, "NO_INSPECTIONS",
                  "no gate results with defect codes",
                  "FLOW_ONLY: defect attribution is unavailable")

        if not self.has_environment:
            r.add(NOTE, "NO_ENVIRONMENT",
                  "no ambient conditions",
                  "optional; drops an environmental covariate")

        # Clock synchronisation. The method compares event ordering across
        # stations, so a skew comparable to takt destroys the signal.
        if clock_sync_s is not None:
            r.stats["clock_sync_s"] = clock_sync_s
            limit = (takt_s / 60.0) if takt_s else 1.0
            if clock_sync_s > limit * 10:
                r.add(BLOCKER, "CLOCK_SKEW",
                      f"clock skew {clock_sync_s:.1f}s across stations",
                      "event ordering is unreliable; NTP-sync the PLCs first")
            elif clock_sync_s > limit:
                r.add(WARNING, "CLOCK_SKEW",
                      f"clock skew {clock_sync_s:.1f}s exceeds the ~{limit:.1f}s target",
                      "localisation will degrade")

        # Sequence gaps. Windowing is by vehicle index, so a hole in the build
        # sequence silently changes what a window spans.
        vids = np.sort(self.vehicles["vehicle_id"].unique())
        if len(vids) > 1:
            gaps = int(np.sum(np.diff(vids) != 1))
            if gaps:
                r.add(WARNING, "SEQUENCE_GAPS",
                      f"{gaps} gap(s) in the vehicle sequence",
                      "windows will span more wall-clock time than expected")

        return r
