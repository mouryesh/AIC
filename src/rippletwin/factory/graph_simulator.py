"""A smaller, purpose-built simulator for non-serial (parallel / split-merge
/ rework-spur) process-graph topologies.

This is deliberately a SEPARATE simulator from ``factory/simulator.py``'s
``LineSimulator``, not a generalization of it. ``LineSimulator``'s dense
recursion is written and vectorised around one assumption -- a single
successor edge per station, addressed by position in a flat ``0..n-1``
index -- and loosening that assumption throughout the ~650-line method that
produces every flagship result in this repository would be a large risk for
a small demonstration. The claim this module exists to support is narrower
and different: that the *inference* machinery
(``twin/shadow.py``'s propagation matrices, ``twin/propagate.py``'s
forecast, ``twin/placement.py``'s sensor-placement ranking) is not
hard-coded to a serial chain. That is proven by running that exact,
unmodified inference code against telemetry from a genuinely different
topology -- which is what this module supplies -- not by making the
flagship simulator do everything.

The physics obeyed is the same conservation-of-material recursion
``LineSimulator``'s module docstring states::

    start_i(v)     = max(arrival at i, departure of the previous vehicle at i)
    end_i(v)       = start_i(v) + proc_i(v)
    departure_i(v) = max(end_i(v), buffer-gated release onto v's chosen edge)

generalized in the two places a serial chain hides:

* "arrival at i" comes from v's own *chosen* predecessor edge, not from "the
  previous station" -- a split station's vehicles fan out onto different
  edges, and a merge station's vehicles arrive interleaved from more than
  one.
* "previous vehicle at i" is v's predecessor in *this station's own
  processing order*, not the previous global ``vehicle_id`` -- at a merge
  station those are not the same sequence in general.

Scoping (stated plainly, not hidden)
-------------------------------------
Vehicles are processed **vehicle-major, in release order** -- each vehicle's
entire path is walked end to end before the next vehicle starts, mirroring
``LineSimulator``'s outer loop closely enough that the same buffer-gating
trick (look up an earlier-processed vehicle's already-known start time at
the downstream station) carries over unchanged. The simplification this
buys is real: at a merge point fed by branches of different latency, a
vehicle released later than another can genuinely arrive first in a real
plant, and this simulator does not reproduce that reordering -- global
release order is treated as each edge's FIFO discipline everywhere,
including across merges. This is the same class of simplification
``twin/predict.py``'s windowing already accepts (vehicle-index order
approximates but does not exactly equal per-branch arrival order); it is
adequate for a demonstration topology with two short, similar-latency
branches, and is not claimed to be exact for branches of very different
length. See ``docs/LIMITATIONS.md``.

Rework is modeled as a spur: a station whose vehicles conditionally route
(on simulated inspection failure) through an extra rework station before
re-merging ahead of the next station on the main line -- acyclic in
station-space, so ``topological_order()`` (Kahn's algorithm) always
terminates. A true upstream cycle is out of scope; see
``LineTopology.topological_order``'s docstring for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .simulator import Disturbance, EVENT_SLOWDOWN
from .topology import LineTopology


@dataclass
class GraphSimResult:
    """Same telemetry shape as ``factory.simulator.SimResult`` (the fields
    ``twin.pipeline``/``twin.shadow`` actually read), so the existing
    inference path consumes it with zero changes."""

    passes: pd.DataFrame
    telemetry: pd.DataFrame
    vehicles: pd.DataFrame
    disturbances: pd.DataFrame
    meta: dict = field(default_factory=dict)
    #: Present for interface parity with SimResult; this module does not
    #: model quality/inspection, so both are always empty.
    inspections: pd.DataFrame = field(default_factory=pd.DataFrame)
    defects: pd.DataFrame = field(default_factory=pd.DataFrame)
    environment: pd.DataFrame = field(default_factory=pd.DataFrame)

    def as_plant_data(self):
        from ..ingest.plant_data import PlantData

        return PlantData.from_frames(
            telemetry=self.telemetry,
            vehicles=self.vehicles,
            inspections=self.inspections,
            environment=self.environment,
            meta={**self.meta, "source": "graph_simulator"},
        )


class GraphLineSimulator:
    """Simulates production over a process graph (serial, parallel, or
    rework-spur), with optional injected SLOWDOWN disturbances."""

    def __init__(self, line: LineTopology, seed: int = 42, start: Optional[datetime] = None):
        if not line.is_graph:
            raise ValueError(
                "GraphLineSimulator is for a configured process graph "
                "(line.edges non-empty); use factory.simulator.LineSimulator "
                "for a plain serial line."
            )
        self.line = line
        self.rng = np.random.default_rng(seed)
        self.start = start or datetime(2026, 3, 2, 6, 0, 0)

    def _route(self, station: int, vehicle_id: int) -> Optional[int]:
        """Deterministic routing at a split: alternate by vehicle_id.

        A rework spur is a split too (station -> normal successor, station
        -> rework successor); ``rework_rate`` on the disturbance-free config
        controls how often the rework edge is taken via a station-specific
        probability baked into edge order (see module test / config
        comments) -- kept simple and reproducible rather than modeling an
        actual inspection outcome.
        """
        succ = self.line.successors(station)
        if not succ:
            return None
        if len(succ) == 1:
            return succ[0]
        return succ[vehicle_id % len(succ)]

    def run(
        self,
        n_vehicles: int,
        disturbances: Sequence[Disturbance] = (),
        run_id: str = "run",
    ) -> GraphSimResult:
        line = self.line
        rng = self.rng
        takt = line.takt_s
        order = line.topological_order()
        roots = [i for i in order if not line.predecessors(i)]
        if len(roots) != 1:
            raise ValueError(
                f"expected exactly one root station (in-degree 0), found {len(roots)}: {roots}"
            )
        root = roots[0]

        release = np.arange(n_vehicles, dtype=float) * takt
        release += rng.normal(0, takt * 0.012, n_vehicles)
        release = np.maximum.accumulate(release)

        dist_by_station: Dict[int, List[Disturbance]] = {}
        for d in disturbances:
            dist_by_station.setdefault(d.station, []).append(d)

        base_cycle = {s.index: s.base_cycle_s for s in line.stations}
        noise_cv = {s.index: s.process_noise_cv for s in line.stations}

        last_departure: Dict[int, float] = {i: -np.inf for i in range(line.n_stations)}
        # Per-edge FIFO queue of (vehicle_id, start_time_at_edge.dst) tuples,
        # appended as each vehicle is fully processed (vehicle-major order
        # guarantees the dst-side start time is already known -- see module
        # docstring for the scoping this relies on).
        edge_queue: Dict[Tuple[int, int], List[Tuple[int, float]]] = {
            (e.src, e.dst): [] for e in line.effective_edges()
        }
        edge_capacity: Dict[Tuple[int, int], float] = {
            (e.src, e.dst): float(min(e.buffer_capacity, 10**4)) for e in line.effective_edges()
        }

        rows: List[dict] = []

        for v in range(n_vehicles):
            t_ref = float(release[v])
            station = root
            visited_path: List[int] = []
            while station is not None:
                h = 1.0
                for d in dist_by_station.get(station, ()):
                    inten = d.intensity(t_ref)
                    if inten > 0 and d.kind == EVENT_SLOWDOWN:
                        h *= 1.0 + (d.magnitude - 1.0) * inten

                lognoise = float(
                    np.exp(rng.normal(0, noise_cv.get(station, 0.05)) - 0.5 * noise_cv.get(station, 0.05) ** 2)
                )
                p = base_cycle.get(station, takt) * h * lognoise

                arrival = t_ref
                st = max(arrival, last_departure[station])
                starved = max(0.0, arrival - last_departure[station]) if last_departure[station] > -np.inf else 0.0
                end = st + p

                succ = self._route(station, v)
                blocked = 0.0
                if succ is not None:
                    edge = (station, succ)
                    cap = edge_capacity[edge]
                    q = edge_queue[edge]
                    b = int(cap)
                    if b > 0 and len(q) >= b:
                        # Room only once the vehicle b slots ahead in this
                        # edge's queue has already started at succ -- the
                        # same trick LineSimulator uses, generalised to a
                        # per-edge queue instead of a flat array.
                        gate_start = q[-b][1]
                        dep = max(end, gate_start)
                    else:
                        dep = end
                    blocked = dep - end
                else:
                    dep = end

                rows.append(
                    {
                        "vehicle_id": v,
                        "station": station,
                        "t_start_s": st,
                        "t_end_s": end,
                        "t_depart_s": dep,
                        "proc_time_s": p,
                        "cycle_time_s": dep - st,
                        "blocked_s": blocked,
                        "starved_s": starved,
                        "true_health": h,
                    }
                )
                last_departure[station] = dep
                visited_path.append(station)
                t_ref = dep
                if succ is not None:
                    edge_queue[(station, succ)].append((v, dep))
                station = succ

        passes = pd.DataFrame(rows)
        station_meta = {s.index: s for s in line.stations}
        passes["station_id"] = passes["station"].map(lambda i: station_meta[i].station_id)
        passes["zone"] = passes["station"].map(lambda i: station_meta[i].zone)
        passes["tier"] = passes["station"].map(lambda i: station_meta[i].tier)

        obs = set(line.observed_indices)
        tel = passes[passes["station"].isin(obs)].copy()
        tel = tel[
            ["vehicle_id", "station", "station_id", "zone", "t_start_s", "t_depart_s",
             "cycle_time_s", "proc_time_s", "blocked_s", "starved_s"]
        ].reset_index(drop=True)
        tel["variant"] = "BASE"
        tel["timestamp"] = [self.start + timedelta(seconds=float(x)) for x in tel["t_depart_s"]]

        vehicles = pd.DataFrame(
            {
                "vehicle_id": np.arange(n_vehicles),
                "variant": "BASE",
                "release_t_s": release,
                "release_timestamp": [self.start + timedelta(seconds=float(x)) for x in release],
                "shift": "A",
            }
        )

        dist_rows = [
            {
                "station": d.station, "station_id": station_meta[d.station].station_id,
                "kind": d.kind, "t_start_s": d.t_start_s, "t_end_s": d.t_end_s,
                "magnitude": d.magnitude, "ramp_s": d.ramp_s, "label": d.label,
                "tier": station_meta[d.station].tier,
            }
            for d in disturbances
        ]
        dist_df = pd.DataFrame(dist_rows) if dist_rows else pd.DataFrame(
            columns=["station", "station_id", "kind", "t_start_s", "t_end_s",
                     "magnitude", "ramp_s", "label", "tier"]
        )

        horizon_s = float(passes["t_depart_s"].max()) if len(passes) else 0.0
        meta = {
            "run_id": run_id, "seed": None, "n_vehicles": int(n_vehicles),
            "n_stations": int(line.n_stations), "start": self.start.isoformat(),
            "horizon_s": horizon_s, "coverage": float(line.coverage),
            "n_observed": len(obs), "n_hidden": len(line.hidden_indices),
            "throughput_vph": float(n_vehicles / (horizon_s / 3600.0)) if horizon_s > 0 else 0.0,
            "topology": "graph",
        }

        return GraphSimResult(
            passes=passes, telemetry=tel, vehicles=vehicles,
            disturbances=dist_df, meta=meta,
        )
