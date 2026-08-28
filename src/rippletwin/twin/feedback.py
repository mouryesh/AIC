"""Closing the human-feedback loop (Round 2 brief §21): turning recorded
ledger outcomes into a per-station prior adjustment for future inference.

``hitl/ledger.py::precision_by_station`` already computes, from recorded
outcomes, how often a station named by an alert was actually confirmed.
This module is the other half: consuming that output to adjust
``ShadowConfig.station_prior_weight`` so a station with a track record of
being confirmed correct starts future windows with a modestly higher prior,
and one with a track record of false accusations starts lower.

This is deliberately not a black-box online learner. The update is a
transparent function of a stated, auditable quantity (Laplace-smoothed
precision, already computed by the ledger), bounded so a handful of
outcomes cannot swing a station's prior by more than a stated factor. A
supervisor has to be able to see why the system now trusts a station
differently than it did last month -- the same reasoning
``hitl/ledger.py``'s own docstring gives for keeping ``precision_by_station``
a simple count rather than an opaque update.

Whether this measurably improves headline localisation accuracy is an
empirical question, checked in
``evaluation/feedback_experiment.py`` rather than assumed.
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np

from ..factory.topology import LineTopology
from ..hitl.ledger import DecisionLedger, precision_by_station
from .shadow import ShadowConfig


def priors_from_precision(
    line: LineTopology,
    ledger: DecisionLedger,
    min_outcomes: int = 3,
    max_weight: float = 2.0,
    min_weight: float = 0.5,
) -> Dict[int, float]:
    """Per-station prior weight from recorded ledger outcomes.

    A station needs at least ``min_outcomes`` recorded outcomes before its
    weight moves away from 1.0 (uniform, i.e. no feedback yet) -- one lucky
    or unlucky call should not move a station's prior at all, the same
    discipline ``ShadowSensor.calibrate`` applies elsewhere (a threshold set
    from too little data is not trusted).

    Precision of 0.5 (the Laplace prior's own midpoint, i.e. "no
    information") maps to weight 1.0; the mapping is linear out to
    ``min_weight``/``max_weight`` at precision 0/1.
    """
    df = precision_by_station(ledger)
    weights: Dict[int, float] = {}
    if df.empty:
        return weights
    id_to_idx = {s.station_id: s.index for s in line.stations}
    for _, r in df.iterrows():
        idx = id_to_idx.get(r["station_id"])
        if idx is None or r["n_outcomes"] < min_outcomes:
            continue
        precision = float(r["precision"])
        if precision >= 0.5:
            weight = 1.0 + (max_weight - 1.0) * (precision - 0.5) / 0.5
        else:
            weight = 1.0 - (1.0 - min_weight) * (0.5 - precision) / 0.5
        weights[int(idx)] = float(np.clip(weight, min_weight, max_weight))
    return weights


def apply_feedback(cfg: ShadowConfig, weights: Dict[int, float]) -> ShadowConfig:
    """A COPY of ``cfg`` with ``station_prior_weight`` set.

    Never mutates the caller's config -- a rejected or reverted feedback
    update is just discarding the copy, not undoing an in-place change.
    """
    new_cfg = copy.deepcopy(cfg)
    new_cfg.station_prior_weight = dict(weights)
    return new_cfg
