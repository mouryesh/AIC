"""Observability views over a single fixed physics run.

The sensor-coverage experiment asks: how much predictive capability survives as
direct instrumentation is removed? Answering that honestly requires holding the
*physics* fixed and varying only what we are allowed to see. If each coverage
level re-ran the simulator, differences between levels would partly be
differences between random draws, and the experiment would prove nothing.

So we simulate once, at full observability, and then construct restricted views
of the resulting telemetry. Ground truth -- including the true state of every
station the view has blinded -- stays identical across levels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..factory.simulator import SimResult
from ..factory.topology import LineTopology, TIER_RICH


def telemetry_view(
    res: SimResult, view_line: LineTopology, source_line: LineTopology
) -> SimResult:
    """Restrict a run's telemetry to what ``view_line`` can observe.

    ``res`` must come from a simulation run at full observability. Stations that
    are MANUAL in ``view_line`` emit nothing at all; stations that are not RICH
    lose their torque / vibration / temperature channels. Buffer occupancy is
    dropped wherever it is no longer derivable, which requires both endpoints of
    the arc to be observed.
    """
    obs = set(view_line.observed_indices)
    rich = set(view_line.rich_indices)

    tel = res.telemetry[res.telemetry["station"].isin(obs)].copy()

    for c in ["torque_nm", "vibration_mm_s", "station_temp_c"]:
        if c in tel.columns:
            tel.loc[~tel["station"].isin(rich), c] = np.nan
    tel["has_process_channels"] = tel["station"].isin(rich)

    # Buffer level survives only where station i and i+1 are both observed.
    if "buffer_level" in tel.columns:
        derivable = {i for i in obs if (i + 1) in obs}
        mask = ~tel["station"].isin(derivable)
        tel.loc[mask, "buffer_level"] = np.nan
        tel.loc[mask, "buffer_capacity"] = np.nan

    return SimResult(
        passes=res.passes,
        telemetry=tel.reset_index(drop=True),
        inspections=res.inspections,
        vehicles=res.vehicles,
        environment=res.environment,
        disturbances=res.disturbances,
        defects=res.defects,
        meta={**res.meta, "view_coverage": view_line.coverage},
    )


def full_observability(line: LineTopology) -> LineTopology:
    """A copy of ``line`` with every station instrumented to RICH.

    Used as the simulation substrate so that a single run can be viewed at any
    coverage level afterwards.
    """
    import copy

    full = copy.deepcopy(line)
    for s in full.stations:
        s.tier = TIER_RICH
    return full
