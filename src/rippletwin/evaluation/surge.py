"""Surge / performance telemetry (Round 2 brief §32-33): throughput,
inference latency, and peak memory under a high-volume run with no injected
fault (``factory.scenarios.scenario_production_surge``, S9).

The question is graceful degradation, not raw speed: does latency scale
sanely with volume, and does the false-alarm rate stay controlled under
load the way it does at normal volume. Every number here is measured on
this machine, for this run -- a relative, illustrative reading, not a
production SLA.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

from ..factory import scenarios as SC
from ..factory.topology import build_line
from ..twin.pipeline import fit_context, infer, simulate


@dataclass
class SurgeConfig:
    line_config: str = "configs/line_42.yaml"
    line_seed: int = 7
    nominal_vehicles: int = 2200
    calibration_vehicles: int = 1800
    surge_vehicles: int = 6000


def run_surge_test(
    cfg: SurgeConfig | None = None,
    out_dir: str | Path = "results",
    verbose: bool = True,
) -> dict:
    cfg = cfg or SurgeConfig()
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    line = build_line(cfg.line_config, seed=cfg.line_seed)
    nominal = simulate(line, SC.nominal_run(cfg.nominal_vehicles), seed=1)
    calib = simulate(line, SC.nominal_run(cfg.calibration_vehicles), seed=2)

    t0 = time.time()
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)
    fit_latency_s = time.time() - t0

    scen = SC.scenario_production_surge(line, seed=909)
    scen.n_vehicles = cfg.surge_vehicles

    tracemalloc.start()
    t1 = time.time()
    res = simulate(line, scen, seed=1234)
    sim_latency_s = time.time() - t1

    t2 = time.time()
    scored, shadow, sensor = infer(ctx, res)
    infer_latency_s = time.time() - t2
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_windows = int(len(shadow))
    result = {
        "n_vehicles": cfg.surge_vehicles,
        "n_windows": n_windows,
        "fit_context_latency_s": round(fit_latency_s, 3),
        "simulate_latency_s": round(sim_latency_s, 3),
        "infer_latency_s": round(infer_latency_s, 3),
        "infer_latency_per_window_ms": round(
            infer_latency_s / n_windows * 1000, 3
        ) if n_windows else float("nan"),
        "peak_memory_mb": round(peak / (1024 * 1024), 1),
        "throughput_vph": round(float(res.meta["throughput_vph"]), 1),
        "false_alarm_rate": round(float(shadow["detected"].mean()), 4) if n_windows else float("nan"),
        "config": asdict(cfg),
        "note": "Measured on this machine for this run -- illustrative, not an SLA.",
    }
    (out_dir / "tables" / "surge_test.json").write_text(json.dumps(result, indent=2, default=str))
    if verbose:
        print(json.dumps(result, indent=2, default=str))
    return result
