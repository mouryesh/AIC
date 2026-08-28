"""Topology-generalization experiment (Round 2 brief §12-13): the same
inference engine, completely unmodified, run against three topologies --

* Plant A -- the flagship serial line (``configs/line_42.yaml``).
* Plant B -- serial + one parallel branch (``configs/plant_b_parallel.yaml``).
* Plant C -- serial + a rework spur (``configs/plant_c_rework.yaml``).

``twin.pipeline.fit_context``/``infer`` is called identically for all three;
the only thing that differs between them is which simulator produces their
telemetry (``LineSimulator`` for the serial line, ``GraphLineSimulator`` for
the other two) and the topology itself. This is the concrete evidence for
"the topology should be configuration-driven... do not create separate
algorithms for each plant."

Every number here is a SIMULATED PROTOTYPE RESULT on synthetic data, and
Plant B/C are small demonstrator topologies, not additional flagship lines
-- see ``factory/graph_simulator.py``'s module docstring for the scoping
this accepts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..factory.graph_simulator import GraphLineSimulator
from ..factory.simulator import Disturbance, EVENT_SLOWDOWN, LineSimulator
from ..factory.topology import LineTopology, build_line
from ..twin.pipeline import fit_context, infer


@dataclass
class PlantSpec:
    name: str
    config: str
    seed: int = 7
    is_graph: bool = False


PLANTS: List[PlantSpec] = [
    PlantSpec("Plant A (serial, flagship)", "configs/line_42.yaml", 7, False),
    PlantSpec("Plant B (parallel branch)", "configs/plant_b_parallel.yaml", 7, True),
    PlantSpec("Plant C (rework spur)", "configs/plant_c_rework.yaml", 7, True),
]


def _simulate(line: LineTopology, is_graph: bool, n_vehicles: int, disturbances, seed: int, run_id: str):
    if is_graph:
        return GraphLineSimulator(line, seed=seed).run(n_vehicles, disturbances, run_id=run_id)
    return LineSimulator(line, seed=seed).run(n_vehicles, disturbances, run_id=run_id)


def _pick_station(line: LineTopology, rng: np.random.Generator) -> int:
    hidden_ok = [
        i for i in line.hidden_indices
        if line.nearest_observed_upstream(i, 1) and line.nearest_observed_downstream(i, 1)
    ]
    pool = hidden_ok if hidden_ok and rng.random() < 0.6 else line.observed_indices
    pool = [i for i in pool if not line.stations[i].is_inspection] or list(pool)
    return int(rng.choice(pool))


def run_topology_experiment(
    out_dir: str | Path = "results",
    n_episodes: int = 8,
    n_vehicles: int = 1200,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in PLANTS:
        t_plant = time.time()
        line = build_line(spec.config, seed=spec.seed)
        nom_n = int(n_vehicles * 1.3)
        nominal = _simulate(line, spec.is_graph, nom_n, [], seed=1, run_id="nominal")
        calib = _simulate(line, spec.is_graph, int(n_vehicles * 1.1), [], seed=2, run_id="calib")
        ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.02)

        for ep in range(n_episodes):
            seed = 7000 + ep
            rng = np.random.default_rng(seed)
            k = _pick_station(line, rng)
            mag = float(rng.uniform(1.25, 1.55))
            t0 = float(rng.uniform(0.2, 0.4)) * n_vehicles * line.takt_s
            dur = float(rng.uniform(0.25, 0.4)) * n_vehicles * line.takt_s
            d = Disturbance(
                station=k, kind=EVENT_SLOWDOWN, t_start_s=t0, t_end_s=t0 + dur,
                magnitude=mag, ramp_s=1200.0, label="topology-experiment",
            )
            res = _simulate(line, spec.is_graph, n_vehicles, [d], seed=seed + 77, run_id=f"ep{seed}")

            t_infer = time.time()
            scored, shadow, sensor = infer(ctx, res)
            infer_latency_s = time.time() - t_infer

            during = shadow[
                (shadow["t_mid_s"] >= d.t_start_s + d.ramp_s) & (shadow["t_mid_s"] <= d.t_end_s)
            ]
            det = during[during["detected"]]
            quiet = shadow[(shadow["t_mid_s"] < d.t_start_s) | (shadow["t_mid_s"] > d.t_end_s)]

            rows.append({
                "plant": spec.name, "seed": seed, "true_station": k,
                "true_station_id": line.stations[k].station_id,
                "source_hidden": bool(line.stations[k].is_hidden),
                "n_windows_during": int(len(during)),
                "detection_rate": float(det.shape[0] / len(during)) if len(during) else np.nan,
                "top1": float((det["top_station"] == k).mean()) if len(det) else np.nan,
                "within1": float((np.abs(det["top_station"].to_numpy() - k) <= 1).mean()) if len(det) else np.nan,
                "mean_confidence": float(det["group_prob"].mean()) if len(det) else np.nan,
                "false_alarm_rate": float(quiet["detected"].mean()) if len(quiet) else np.nan,
                "infer_latency_s": infer_latency_s,
            })
            if verbose:
                print(
                    f"  {spec.name}: ep seed={seed} station={line.stations[k].station_id} "
                    f"hidden={line.stations[k].is_hidden} top1={rows[-1]['top1']}"
                )
        if verbose:
            print(f"{spec.name}: done in {time.time() - t_plant:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "topology_raw.csv", index=False)

    summary = (
        df.groupby("plant")
        .agg(
            n_episodes=("seed", "size"),
            detection_rate=("detection_rate", "mean"),
            top1=("top1", "mean"),
            within1=("within1", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            false_alarm_rate=("false_alarm_rate", "mean"),
            mean_infer_latency_s=("infer_latency_s", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "tables" / "topology_summary.csv", index=False)

    manifest = {
        "generated_by": "rippletwin.evaluation.topology_experiment.run_topology_experiment",
        "result_type": "SIMULATED PROTOTYPE RESULT on synthetic data",
        "plants": [asdict(p) for p in PLANTS],
        "n_episodes_per_plant": n_episodes,
        "summary": summary.to_dict("records"),
    }
    (out_dir / "tables" / "topology_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    if verbose:
        print(summary.to_string(index=False))
    return {"raw": df, "summary": summary, "manifest": manifest}
