"""Bottleneck frequency and shift-severity diagnostics.

Two deterministic, post-hoc statistics computed over ShadowSensor's own
output -- no new estimator, no training data. Following West, Schwenken &
Deuse (2023), "Data-driven approach for diagnostic analysis of dynamic
bottlenecks in serial manufacturing systems" (arXiv:2306.16120):

- ``bottleneck_frequency`` (their rbf): the fraction of windows in which
  each station was the leading (detected) suspect. RippleTwin already
  computes something functionally equivalent internally, in
  ``twin.placement.suspicion_from_shadow``; this exposes the plain,
  un-weighted fraction as a first-class report, matching the paper's own
  definition (sums to 1 across stations).
- ``bottleneck_shift_severity`` (their rbs): a per-window ratio of the
  runner-up candidate's posterior mass to the leader's. The paper's own
  metric uses each station's measured "active period"; RippleTwin has no
  such per-non-leader measurement, so the substitution here is posterior
  mass, which serves the same purpose (an early signal that a second
  candidate is about to overtake the current leader) using evidence
  RippleTwin already computes. ``runner_up_station``/``runner_up_prob``
  are additive columns ``ShadowSensor.run`` now emits (see twin/shadow.py);
  the leader's own posterior mass is ``group_prob``, which already existed.
"""

from __future__ import annotations

import pandas as pd

from ..factory.topology import LineTopology


def bottleneck_frequency(shadow: pd.DataFrame, line: LineTopology) -> pd.DataFrame:
    """Fraction of detected windows each station led, over the given shadow frame.

    ``shadow`` is the frame produced by ``ShadowSensor.run`` (or
    ``twin.pipeline.infer``'s second return value) for one episode or shift.
    Returns one row per station with columns ``station``, ``station_id``,
    ``zone``, ``rbf``, ``n_windows_leading``. ``rbf`` sums to 1 across
    stations when at least one window was detected; if none were, every
    row is 0.
    """
    if shadow.empty or "detected" not in shadow.columns:
        return pd.DataFrame(
            columns=["station", "station_id", "zone", "rbf", "n_windows_leading"]
        )
    leading = shadow[shadow["detected"]]
    counts = leading["top_station"].value_counts()
    total = float(counts.sum()) or 1.0
    rows = []
    for s in line.stations:
        n = int(counts.get(s.index, 0))
        rows.append(
            {
                "station": s.index,
                "station_id": s.station_id,
                "zone": s.zone,
                "rbf": n / total,
                "n_windows_leading": n,
            }
        )
    return pd.DataFrame(rows)


def bottleneck_shift_severity(shadow: pd.DataFrame) -> pd.DataFrame:
    """Per-window ratio of the runner-up's posterior mass to the leader's.

    Requires ``shadow`` to carry ``runner_up_station`` and
    ``runner_up_prob`` alongside ``top_station`` and ``group_prob``
    (the leader's own posterior mass, already emitted by
    ``ShadowSensor.run``). Returns ``window``, ``leader_station``,
    ``runner_up_station``, ``severity_ratio`` -- 1.0 exactly at the
    leader's own row by construction, mirroring the paper's
    ``rbs_BN == 1`` convention.
    """
    required = {"window", "top_station", "runner_up_station", "runner_up_prob", "group_prob"}
    missing = required - set(shadow.columns)
    if missing:
        raise KeyError(
            f"bottleneck_shift_severity: shadow frame is missing columns {missing}. "
            "See runbook step A0 -- ShadowSensor.run() may need the additive "
            "runner_up_station/runner_up_prob columns added first."
        )
    out = shadow[["window", "top_station", "runner_up_station", "runner_up_prob", "group_prob"]].copy()
    out = out.rename(columns={"top_station": "leader_station"})
    leader_mass = out["group_prob"].replace(0.0, pd.NA)
    out["severity_ratio"] = (out["runner_up_prob"] / leader_mass).fillna(0.0).clip(0.0, 1.0)
    return out[["window", "leader_station", "runner_up_station", "severity_ratio"]]
