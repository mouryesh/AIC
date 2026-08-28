"""Tests for twin.whatif -- counterfactual projections."""

from __future__ import annotations

from rippletwin.factory.topology import build_line
from rippletwin.twin.whatif import (
    whatif_add_sensor,
    whatif_cycle_time_improvement,
    whatif_repair,
)

CONFIG = "configs/line_42.yaml"


def _line():
    return build_line(CONFIG, seed=7)


def test_whatif_repair_reduces_or_eliminates_projected_loss():
    line = _line()
    k = line.hidden_indices[0]
    degraded_cycle = line.stations[k].base_cycle_s * 1.3
    result = whatif_repair(line, k, degraded_cycle)
    assert result.before is not None and result.after is not None
    assert result.after.units_lost_at_horizon <= result.before.units_lost_at_horizon
    assert not result.after.is_binding  # repaired = back to nominal, no longer binding
    assert "not a measurement" in result.disclaimer.lower() or "projection" in result.disclaimer.lower()


def test_whatif_cycle_time_improvement_scales_with_pct():
    line = _line()
    k = line.hidden_indices[0]
    degraded_cycle = line.stations[k].base_cycle_s * 1.4
    small = whatif_cycle_time_improvement(line, k, degraded_cycle, pct_improvement=0.1)
    large = whatif_cycle_time_improvement(line, k, degraded_cycle, pct_improvement=0.3)
    assert large.after.units_lost_at_horizon <= small.after.units_lost_at_horizon


def test_whatif_add_sensor_reduces_ambiguity_at_that_station():
    line = _line()
    k = line.hidden_indices[0]
    result = whatif_add_sensor(line, k)
    assert result.station_id == line.stations[k].station_id
    # Instrumenting a station removes it from the inference problem, so its
    # own separability after adding it should not be worse than before.
    assert result.separability_after >= result.separability_before
