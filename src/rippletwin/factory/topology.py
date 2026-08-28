"""Line topology: stations, zones, buffers and the process graph.

The topology is the *structural* half of the digital twin. It is not learned from
data -- it encodes what the plant physically is: an ordered set of stations,
connected by finite buffers, grouped into zones, with inspection gates at known
points and a known instrumentation tier per station.

Every downstream component (shadow-sensing, propagation, explanation) reasons
over this structure rather than over an unordered bag of sensor channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

# Instrumentation tiers, ordered from most to least observable.
TIER_RICH = "RICH"
TIER_BASIC = "BASIC"
TIER_MANUAL = "MANUAL"

#: Tiers that emit automatic timing telemetry. MANUAL stations emit nothing.
OBSERVED_TIERS = (TIER_RICH, TIER_BASIC)

#: Failure modes physically producible in each zone. Kept here rather than in the
#: simulator because it is structural process knowledge the twin is allowed to
#: use, not a property of any particular production run.
_ZONE_DEFECT_TYPES = {
    "BODY": ["weld_gap", "panel_misalign", "sealer_void"],
    "PAINT": ["orange_peel", "runs_sags", "color_mismatch"],
    "FINAL": ["torque_out_of_spec", "harness_misroute", "trim_gap", "water_leak"],
}


@dataclass
class Station:
    """A single workstation on the line."""

    index: int
    station_id: str
    zone: str
    base_cycle_s: float
    #: Fraction of the station's work that is manual (drives operator variation).
    manual_content: float
    tier: str
    #: Buffer capacity on the *outbound* side of this station (slots to the next station).
    out_buffer: int
    #: Per-vehicle probability of a micro-stoppage.
    microstop_rate: float
    microstop_range_s: tuple
    process_noise_cv: float
    #: Baseline probability this station injects a defect into a nominal vehicle.
    base_defect_rate: float
    #: Which defect types this station can physically produce, and how often.
    #: Real plants know this from process FMEA -- a sealer station cannot cause a
    #: torque fault. This is structural process knowledge, not a learned pattern,
    #: and it is what narrows defect attribution from "somewhere in body shop"
    #: to a specific handful of candidate stations.
    defect_profile: dict = field(default_factory=dict)
    #: Set for stations that perform a quality inspection.
    inspection_id: Optional[str] = None
    inspection_covers: tuple = ()
    inspection_detect_prob: float = 0.0

    @property
    def is_observed(self) -> bool:
        """True when the station emits automatic cycle/blocked/starved telemetry."""
        return self.tier in OBSERVED_TIERS

    @property
    def is_hidden(self) -> bool:
        """True for MANUAL stations -- the ones shadow-sensing must reconstruct."""
        return self.tier == TIER_MANUAL

    @property
    def has_process_channels(self) -> bool:
        """True when torque/vibration/temperature channels are available."""
        return self.tier == TIER_RICH

    @property
    def is_inspection(self) -> bool:
        return self.inspection_id is not None


@dataclass
class LineTopology:
    """An ordered serial line with finite buffers between consecutive stations."""

    name: str
    takt_s: float
    stations: List[Station]
    zones: Dict[str, dict]
    variants: Dict[str, dict]
    shifts: List[dict]
    environment: Dict[str, dict]

    # ------------------------------------------------------------------ basics

    def __len__(self) -> int:
        return len(self.stations)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def station_ids(self) -> List[str]:
        return [s.station_id for s in self.stations]

    def __getitem__(self, index: int) -> Station:
        return self.stations[index]

    def by_id(self, station_id: str) -> Station:
        for s in self.stations:
            if s.station_id == station_id:
                return s
        raise KeyError(f"unknown station: {station_id}")

    # ----------------------------------------------------------- observability

    @property
    def observed_indices(self) -> List[int]:
        return [s.index for s in self.stations if s.is_observed]

    @property
    def hidden_indices(self) -> List[int]:
        return [s.index for s in self.stations if s.is_hidden]

    @property
    def rich_indices(self) -> List[int]:
        return [s.index for s in self.stations if s.has_process_channels]

    @property
    def inspection_indices(self) -> List[int]:
        return [s.index for s in self.stations if s.is_inspection]

    @property
    def coverage(self) -> float:
        """Fraction of stations emitting automatic telemetry."""
        return len(self.observed_indices) / self.n_stations

    def nearest_observed_upstream(self, index: int, k: int = 3) -> List[int]:
        """The k closest observed stations strictly upstream of ``index``."""
        out = [i for i in range(index - 1, -1, -1) if self.stations[i].is_observed]
        return out[:k]

    def nearest_observed_downstream(self, index: int, k: int = 3) -> List[int]:
        """The k closest observed stations strictly downstream of ``index``."""
        out = [i for i in range(index + 1, self.n_stations) if self.stations[i].is_observed]
        return out[:k]

    def next_inspection_after(self, index: int) -> Optional[int]:
        """Index of the first inspection gate at or after ``index``."""
        for i in self.inspection_indices:
            if i >= index:
                return i
        return None

    # ------------------------------------------------------------------ graph

    def adjacency(self) -> np.ndarray:
        """Directed material-flow adjacency (i -> i+1)."""
        n = self.n_stations
        adj = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            adj[i, i + 1] = 1.0
        return adj

    def buffer_between(self, i: int, j: int) -> int:
        """Buffer capacity on the arc i -> j (j must equal i + 1)."""
        if j != i + 1:
            raise ValueError("buffers are only defined between consecutive stations")
        return self.stations[i].out_buffer

    def zone_of(self, index: int) -> str:
        return self.stations[index].zone

    def zone_bounds(self, zone: str) -> tuple:
        idx = [s.index for s in self.stations if s.zone == zone]
        return min(idx), max(idx)

    def summary(self) -> dict:
        """Compact description used in reports and the dashboard header."""
        per_zone = {}
        for zid in self.zones:
            idx = [s for s in self.stations if s.zone == zid]
            per_zone[zid] = {
                "stations": len(idx),
                "rich": sum(s.tier == TIER_RICH for s in idx),
                "basic": sum(s.tier == TIER_BASIC for s in idx),
                "manual": sum(s.tier == TIER_MANUAL for s in idx),
            }
        return {
            "name": self.name,
            "n_stations": self.n_stations,
            "takt_s": self.takt_s,
            "coverage": self.coverage,
            "n_observed": len(self.observed_indices),
            "n_hidden": len(self.hidden_indices),
            "inspections": [self.stations[i].inspection_id for i in self.inspection_indices],
            "per_zone": per_zone,
        }


# --------------------------------------------------------------------- builder


def _assign_tiers(
    stations_meta: Sequence[dict],
    sensing_cfg: dict,
    rng: np.random.Generator,
) -> List[str]:
    """Assign instrumentation tiers with a zone bias toward older equipment.

    MANUAL stations are the hidden ones. They are deliberately *not* spread
    uniformly: real plants have clusters of un-instrumented legacy equipment, and
    a hidden station sitting between two other hidden stations is a much harder
    inference problem than an isolated one. We keep that difficulty.
    """
    n = len(stations_meta)
    n_manual = int(round(sensing_cfg["manual_fraction"] * n))
    n_rich = int(round(sensing_cfg["rich_fraction"] * n))
    n_basic = n - n_manual - n_rich

    bias = sensing_cfg.get("manual_bias_zones", {})
    weights = np.array([bias.get(m["zone"], 1.0) for m in stations_meta], dtype=float)

    # Inspection gates are always instrumented -- a plant knows its own test results.
    inspection_mask = np.array([m["is_inspection"] for m in stations_meta])
    weights[inspection_mask] = 0.0
    # Never hide the very first station: it has no upstream evidence to infer from.
    weights[0] = 0.0

    weights = weights / weights.sum()
    manual_idx = rng.choice(n, size=n_manual, replace=False, p=weights)

    tiers = [None] * n
    for i in manual_idx:
        tiers[i] = TIER_MANUAL

    remaining = [i for i in range(n) if tiers[i] is None]
    # RICH goes preferentially to inspection gates and newer zones (PAINT/FINAL).
    rich_pref = np.array(
        [2.5 if stations_meta[i]["is_inspection"] else 1.0 for i in remaining], dtype=float
    )
    rich_pref /= rich_pref.sum()
    rich_idx = rng.choice(remaining, size=min(n_rich, len(remaining)), replace=False, p=rich_pref)
    for i in rich_idx:
        tiers[i] = TIER_RICH
    for i in range(n):
        if tiers[i] is None:
            tiers[i] = TIER_BASIC

    assert sum(t == TIER_MANUAL for t in tiers) == n_manual
    assert n_basic >= 0
    return tiers


def build_line(config_path: str | Path, seed: int = 7) -> LineTopology:
    """Construct the line topology from a YAML config.

    ``seed`` controls station-to-station heterogeneity and the placement of
    instrumentation tiers. It is separate from the simulation seed so the *same
    physical line* can be run under many different production scenarios.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    rng = np.random.default_rng(seed)

    line_cfg = cfg["line"]
    zones_cfg = {z["id"]: z for z in cfg["zones"]}
    inspections = {int(i["station"]): i for i in cfg["inspections"]}
    inter_zone = cfg["inter_zone_buffer"]

    total = sum(z["station_count"] for z in cfg["zones"])
    if total != line_cfg["n_stations"]:
        raise ValueError(
            f"zone station_count sums to {total}, expected {line_cfg['n_stations']}"
        )

    # ---- pass 1: positional metadata
    meta: List[dict] = []
    idx = 0
    zone_order = [z["id"] for z in cfg["zones"]]
    for z in cfg["zones"]:
        for k in range(z["station_count"]):
            meta.append(
                {
                    "index": idx,
                    "zone": z["id"],
                    "pos_in_zone": k,
                    "last_in_zone": k == z["station_count"] - 1,
                    "is_inspection": idx in inspections,
                }
            )
            idx += 1

    tiers = _assign_tiers(meta, cfg["sensing"], rng)

    # ---- pass 2: build stations
    stations: List[Station] = []
    for m in meta:
        z = zones_cfg[m["zone"]]
        i = m["index"]

        # Station-to-station heterogeneity in nominal cycle time.
        spread = z["cycle_spread"]
        base_cycle = z["base_cycle_seconds"] * float(rng.normal(1.0, spread / 2.0))
        base_cycle = float(np.clip(base_cycle, z["base_cycle_seconds"] * (1 - spread),
                                   z["base_cycle_seconds"] * (1 + spread)))

        # Outbound buffer: large at zone boundaries, small inside a zone.
        if m["last_in_zone"] and m["zone"] != zone_order[-1]:
            nxt = zone_order[zone_order.index(m["zone"]) + 1]
            out_buffer = int(inter_zone[f"{m['zone']}_{nxt}"])
        elif m["index"] == len(meta) - 1:
            out_buffer = 10**6  # end of line: never blocked
        else:
            out_buffer = int(z["internal_buffer"])

        # Manual work content drives how much operator/shift variation applies.
        # Un-instrumented stations are manual-heavy by construction.
        if tiers[i] == TIER_MANUAL:
            manual_content = float(rng.uniform(0.65, 0.95))
        elif tiers[i] == TIER_BASIC:
            manual_content = float(rng.uniform(0.25, 0.60))
        else:
            manual_content = float(rng.uniform(0.05, 0.30))

        # Process-FMEA style defect propensity: a station produces one dominant
        # failure mode plus a lighter secondary one, not the whole zone's
        # catalogue uniformly. This is knowledge a plant already has, and it is
        # what makes defect attribution to a specific station tractable.
        zone_types = _ZONE_DEFECT_TYPES[m["zone"]]
        order = rng.permutation(len(zone_types))
        wts = np.zeros(len(zone_types))
        wts[order[0]] = rng.uniform(0.60, 0.80)
        if len(zone_types) > 1:
            wts[order[1]] = rng.uniform(0.15, 0.30)
        leftover = max(0.0, 1.0 - wts.sum())
        if len(zone_types) > 2:
            wts[order[2:]] = leftover / max(1, len(zone_types) - 2)
        wts = wts / wts.sum()
        defect_profile = {zone_types[j]: float(wts[j]) for j in range(len(zone_types))}

        insp = inspections.get(i)
        stations.append(
            Station(
                index=i,
                station_id=f"S{i + 1:02d}",
                zone=m["zone"],
                base_cycle_s=base_cycle,
                manual_content=manual_content,
                tier=tiers[i],
                out_buffer=out_buffer,
                microstop_rate=float(z["microstop_rate"]),
                microstop_range_s=tuple(z["microstop_seconds"]),
                process_noise_cv=float(z["process_noise_cv"]),
                # Manual-heavy stations carry a slightly higher nominal defect rate.
                base_defect_rate=float(0.0016 + 0.0042 * manual_content),
                defect_profile=defect_profile,
                inspection_id=insp["id"] if insp else None,
                inspection_covers=tuple(insp["covers"]) if insp else (),
                inspection_detect_prob=float(insp["detect_prob"]) if insp else 0.0,
            )
        )

    return LineTopology(
        name=line_cfg["name"],
        takt_s=float(line_cfg["takt_seconds"]),
        stations=stations,
        zones=zones_cfg,
        variants={v["id"]: v for v in cfg["variants"]},
        shifts=cfg["shifts"],
        environment=cfg["environment"],
    )


def _critical_order(line: "LineTopology", candidates: List[int]) -> List[int]:
    """Rank currently-observed ``candidates`` by how much separability across
    the rest of the line would be lost if each, alone, went dark.

    Local import of ``twin.placement`` -- that module already imports from
    ``twin.shadow``, which imports this module, so importing it at module
    load time here would be circular. Deferring to call time (the same
    pattern ``factory/simulator.py::SimResult.as_plant_data`` already uses
    for its own cross-package import) avoids that without restructuring
    either module.
    """
    from ..twin.placement import ambiguity

    observed = set(line.observed_indices)
    base = ambiguity(line, observed).set_index("station")["separability"]
    losses: Dict[int, float] = {}
    for c in candidates:
        without = ambiguity(line, observed - {c}).set_index("station")["separability"]
        loss = float((base - without).clip(lower=0.0).sum())
        losses[c] = loss
    return sorted(candidates, key=lambda i: -losses[i])


def apply_coverage(
    line: LineTopology, target_coverage: float, seed: int = 11,
    strategy: str = "random",
) -> LineTopology:
    """Return a copy of ``line`` degraded to a target observed-station fraction.

    Used by the sensor-coverage experiment. Stations are *demoted* to MANUAL --
    RICH stations lose their process channels first (becoming BASIC), then BASIC
    stations go dark entirely. Inspection gates and station 0 are never demoted,
    because a plant that cannot read its own end-of-line test has no data problem
    worth solving.

    ``strategy``:

    * ``"random"`` (default) -- uniformly random among demotable observed
      stations, as the existing coverage sweep has always done.
    * ``"critical"`` -- hides the stations that currently carry the most
      value-of-information first (``_critical_order``, the same
      separability calculation ``twin.placement`` uses to *recommend* a
      sensor, run in reverse to ask what removing one costs). This is the
      "losing your most useful sensors" scenario, as distinct from losing
      random ones -- see ``docs/RESULTS.md``'s coverage matrix.
    """
    import copy

    new = copy.deepcopy(line)
    rng = np.random.default_rng(seed)

    protected = set(new.inspection_indices) | {0}
    demotable = [s.index for s in new.stations if s.index not in protected]

    n_target_observed = int(round(target_coverage * new.n_stations))
    n_currently_observed = len(new.observed_indices)
    n_to_hide = n_currently_observed - n_target_observed

    if n_to_hide <= 0:
        return new

    candidates = [i for i in demotable if new.stations[i].is_observed]
    if n_to_hide > len(candidates):
        raise ValueError(
            f"cannot reach coverage {target_coverage:.2f}: only {len(candidates)} "
            f"demotable observed stations (inspection gates are protected)"
        )
    if strategy == "critical":
        hide = _critical_order(new, candidates)[:n_to_hide]
    elif strategy == "random":
        hide = rng.choice(candidates, size=n_to_hide, replace=False)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")
    for i in hide:
        new.stations[i].tier = TIER_MANUAL
    return new
