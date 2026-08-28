"""Named production scenarios used for training, evaluation and the demo.

Each scenario is a deterministic function of a seed, so any result quoted
anywhere in this project can be regenerated exactly.

The set deliberately includes cases RippleTwin should *not* fire on. A detector
that only ever gets shown faults it can find is not evidence of anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .simulator import (
    Disturbance,
    EVENT_COMBINED,
    EVENT_MATERIAL_DELAY,
    EVENT_MICROSTOP_BURST,
    EVENT_QUALITY_DRIFT,
    EVENT_SLOWDOWN,
)
from .topology import LineTopology


@dataclass
class Scenario:
    """A named run: a vehicle count plus the disturbances injected into it."""

    scenario_id: str
    title: str
    question: str
    n_vehicles: int
    disturbances: List[Disturbance]
    #: True when the correct behaviour is to raise no station-level alert.
    expect_no_alert: bool = False
    notes: str = ""


def _pick(
    line: LineTopology,
    hidden: bool,
    zone: Optional[str],
    rng: np.random.Generator,
    require_observed_neighbours: bool = True,
) -> int:
    """Choose a station to disturb, subject to observability and zone."""
    pool = line.hidden_indices if hidden else line.observed_indices
    if zone:
        pool = [i for i in pool if line.stations[i].zone == zone]
    pool = [i for i in pool if not line.stations[i].is_inspection]
    if require_observed_neighbours:
        pool = [
            i
            for i in pool
            if line.nearest_observed_upstream(i, 1) and line.nearest_observed_downstream(i, 1)
        ]
    if not pool:
        raise ValueError(f"no candidate station (hidden={hidden}, zone={zone})")
    return int(rng.choice(pool))


# --------------------------------------------------------------- the five cases


def scenario_hidden_bottleneck(line: LineTopology, seed: int = 101) -> Scenario:
    """S1 -- a station with no sensor slows down and throttles the line."""
    rng = np.random.default_rng(seed)
    k = _pick(line, hidden=True, zone="BODY", rng=rng)
    return Scenario(
        scenario_id="S1_HIDDEN_BOTTLENECK",
        title="Hidden bottleneck at an un-instrumented station",
        question="Can the twin locate a constraint it cannot measure?",
        n_vehicles=1800,
        disturbances=[
            Disturbance(
                station=k,
                kind=EVENT_SLOWDOWN,
                t_start_s=36_000,
                t_end_s=86_000,
                magnitude=1.32,
                ramp_s=2_400,
                label="tool wear on a manual station",
            )
        ],
        notes=f"true source = {line.stations[k].station_id} (MANUAL, no telemetry)",
    )


def scenario_hidden_quality(line: LineTopology, seed: int = 202) -> Scenario:
    """S2 -- a hidden station starts injecting defects, with almost no timing change.

    This is the case the flow model alone cannot solve, and it is why the twin
    carries a second, genealogy-based attribution path.
    """
    rng = np.random.default_rng(seed)
    k = _pick(line, hidden=True, zone="FINAL", rng=rng)
    return Scenario(
        scenario_id="S2_HIDDEN_QUALITY",
        title="Hidden quality drift that surfaces only at a later gate",
        question="Can the twin attribute defects back to an unmeasured source station?",
        n_vehicles=1800,
        disturbances=[
            Disturbance(
                station=k,
                kind=EVENT_QUALITY_DRIFT,
                t_start_s=30_000,
                t_end_s=88_000,
                magnitude=11.0,
                ramp_s=3_000,
                label="fixture drift injecting defects without slowing the cycle",
            )
        ],
        notes=f"true source = {line.stations[k].station_id} (MANUAL); defects surface downstream",
    )


def scenario_normal_variation(line: LineTopology, seed: int = 303) -> Scenario:
    """S3 -- nothing is wrong. Any station-level alert here is a false alarm."""
    return Scenario(
        scenario_id="S3_NORMAL",
        title="Normal variation only",
        question="Does the twin stay quiet when the line is merely noisy?",
        n_vehicles=1800,
        disturbances=[],
        expect_no_alert=True,
        notes="mix changes, shift changes and micro-stops only",
    )


def scenario_observed_station(line: LineTopology, seed: int = 404) -> Scenario:
    """S4 -- a well-instrumented station degrades; direct monitoring should suffice."""
    rng = np.random.default_rng(seed)
    k = _pick(line, hidden=False, zone="PAINT", rng=rng)
    return Scenario(
        scenario_id="S4_OBSERVED_STATION",
        title="Degradation at a fully instrumented station",
        question="Does the twin still agree with the sensor when a sensor exists?",
        n_vehicles=1800,
        disturbances=[
            Disturbance(
                station=k,
                kind=EVENT_COMBINED,
                t_start_s=34_000,
                t_end_s=84_000,
                magnitude=1.28,
                ramp_s=2_400,
                label="bearing wear on an instrumented station",
            )
        ],
        notes=f"true source = {line.stations[k].station_id} ({line.stations[k].tier})",
    )


def scenario_variant_shift(line: LineTopology, seed: int = 505) -> Scenario:
    """S5 -- an EV-heavy production block plus a material delay. Neither is a fault.

    Both events change the line's timing substantially. Neither should be
    attributed to a station. The EV block loads final assembly; the material
    delay starves the whole line uniformly from the head.
    """
    return Scenario(
        scenario_id="S5_VARIANT_AND_SUPPLY",
        title="Variant-mix change and an upstream material delay",
        question="Does the twin avoid blaming a station for a mix or supply effect?",
        n_vehicles=1800,
        disturbances=[
            Disturbance(
                station=0,
                kind=EVENT_MATERIAL_DELAY,
                t_start_s=48_000,
                t_end_s=56_000,
                magnitude=0.35,
                ramp_s=0.0,
                label="inbound material delay",
            )
        ],
        expect_no_alert=True,
        notes="correct behaviour is LINE_SUPPLY or NULL, not a station",
    )


def scenario_gradual_bottleneck(line: LineTopology, seed: int = 606) -> Scenario:
    """S6 -- a slow, gentle ramp designed to still look operational at T-10.

    Every other scenario ramps a disturbance in over minutes
    (``ramp_s`` of 2,400-3,000s, i.e. 40-50 minutes) and is scored only once
    fully ramped in (``evaluate_localization``'s ``settle_fraction``). This
    scenario exists for a different question: not "can the twin localise a
    disturbance once it has arrived" but "does the twin's *risk trajectory*
    rise perceptibly while the disturbance is still ramping in, before the
    constraint has bound". The magnitude and ramp length are chosen so the
    line does not become genuinely output-limited (``forecast.is_binding``)
    until well after the ramp is under way -- there is a real window in which
    the correct twin behaviour is DEGRADING/WATCH, not yet
    PREDICTED_CONSTRAINT or silence.

    ``t_start_s`` is placed so a 1,800s (30 minute) ramp lands the T-30/../T0
    narrative from the master brief inside the scored region of an
    1,800-vehicle run.
    """
    rng = np.random.default_rng(seed)
    k = _pick(line, hidden=True, zone="BODY", rng=rng)
    t_start = 40_000.0
    ramp = 1_800.0  # 30 minutes: T-30 -> T0
    return Scenario(
        scenario_id="S6_EARLY_WARNING",
        title="Gradual degradation at a hidden station -- still operational at T-10",
        question="Does the twin's risk trajectory rise before the line is actually constrained?",
        n_vehicles=2200,
        disturbances=[
            Disturbance(
                station=k,
                kind=EVENT_SLOWDOWN,
                t_start_s=t_start,
                t_end_s=t_start + ramp + 40_000.0,
                magnitude=1.22,
                ramp_s=ramp,
                label="slow tool-wear ramp on a manual station",
            )
        ],
        notes=(
            f"true source = {line.stations[k].station_id} (MANUAL, no telemetry); "
            f"ramp={ramp:.0f}s, so T-30..T0 of the master-brief narrative maps to "
            f"[{t_start:.0f}, {t_start + ramp:.0f}]s simulation time"
        ),
    )


FLAGSHIP_SCENARIOS: Dict[str, Callable[..., Scenario]] = {
    "S1_HIDDEN_BOTTLENECK": scenario_hidden_bottleneck,
    "S2_HIDDEN_QUALITY": scenario_hidden_quality,
    "S3_NORMAL": scenario_normal_variation,
    "S4_OBSERVED_STATION": scenario_observed_station,
    "S5_VARIANT_AND_SUPPLY": scenario_variant_shift,
    "S6_EARLY_WARNING": scenario_gradual_bottleneck,
}


# ------------------------------------------------------------------ bulk corpora


def nominal_run(n_vehicles: int = 2600) -> Scenario:
    """A disturbance-free reference run. Used to fit the nominal baseline."""
    return Scenario(
        scenario_id="NOMINAL",
        title="Disturbance-free reference production",
        question="What does this line look like when nothing is wrong?",
        n_vehicles=n_vehicles,
        disturbances=[],
        expect_no_alert=True,
        notes="baseline fitting only; never used for evaluation",
    )


def random_episode(
    line: LineTopology,
    seed: int,
    n_vehicles: int = 1400,
    p_fault: float = 0.72,
) -> Scenario:
    """One randomly-parameterised episode for building train/test corpora.

    A fraction of episodes are deliberately fault-free so that false-alarm rate
    is measurable on the same corpus that measures detection.
    """
    rng = np.random.default_rng(seed)
    if rng.random() > p_fault:
        return Scenario(
            scenario_id=f"EP{seed}_CLEAN",
            title="Clean episode",
            question="",
            n_vehicles=n_vehicles,
            disturbances=[],
            expect_no_alert=True,
        )

    hidden = bool(rng.random() < 0.55)
    zone = str(rng.choice(["BODY", "PAINT", "FINAL"]))
    try:
        k = _pick(line, hidden=hidden, zone=zone, rng=rng)
    except ValueError:
        k = _pick(line, hidden=not hidden, zone=zone, rng=rng)

    kind = str(
        rng.choice(
            [EVENT_SLOWDOWN, EVENT_QUALITY_DRIFT, EVENT_COMBINED, EVENT_MICROSTOP_BURST],
            p=[0.38, 0.28, 0.22, 0.12],
        )
    )
    if kind in (EVENT_SLOWDOWN, EVENT_COMBINED):
        mag = float(rng.uniform(1.14, 1.45))
    elif kind == EVENT_MICROSTOP_BURST:
        mag = float(rng.uniform(4.0, 12.0))
    else:
        mag = float(rng.uniform(6.0, 16.0))

    t0 = float(rng.uniform(0.22, 0.55)) * n_vehicles * line.takt_s
    dur = float(rng.uniform(0.20, 0.38)) * n_vehicles * line.takt_s

    dists = [
        Disturbance(
            station=k,
            kind=kind,
            t_start_s=t0,
            t_end_s=t0 + dur,
            magnitude=mag,
            ramp_s=float(rng.uniform(900, 3600)),
            label=f"{kind.lower()} at {line.stations[k].station_id}",
        )
    ]
    # Occasionally add an unrelated supply delay as a distractor.
    if rng.random() < 0.18:
        ts = float(rng.uniform(0.6, 0.85)) * n_vehicles * line.takt_s
        dists.append(
            Disturbance(
                station=0,
                kind=EVENT_MATERIAL_DELAY,
                t_start_s=ts,
                t_end_s=ts + 5_000,
                magnitude=float(rng.uniform(0.2, 0.4)),
                ramp_s=0.0,
                label="inbound material delay (distractor)",
            )
        )
    return Scenario(
        scenario_id=f"EP{seed}_{kind}",
        title=f"{kind} at {line.stations[k].station_id}",
        question="",
        n_vehicles=n_vehicles,
        disturbances=dists,
    )
