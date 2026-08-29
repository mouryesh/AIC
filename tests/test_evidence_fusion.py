"""Tests for twin.evidence_fusion (Plan C / Hybrid 2, RESEARCH_EVALUATION.md)."""

import pandas as pd
import pytest

from rippletwin.factory import scenarios as SC
from rippletwin.factory.topology import build_line
from rippletwin.twin.evidence_fusion import ambiguous_groups, fuse_ambiguous_group
from rippletwin.twin.pipeline import fit_context, infer, simulate


LINE = build_line("configs/line_42.yaml", seed=7)


def test_ambiguous_groups_links_mutually_confusable_stations():
    amb = pd.DataFrame(
        [
            {"station": 32, "station_id": "S33", "ambiguity": 0.97, "confusable_with": "S34"},
            {"station": 33, "station_id": "S34", "ambiguity": 0.97, "confusable_with": "S33"},
            {"station": 10, "station_id": "S11", "ambiguity": 0.10, "confusable_with": "S12"},
        ]
    )
    groups = ambiguous_groups(amb, threshold=0.90)
    assert [32, 33] in groups
    assert not any(10 in g for g in groups)


def test_fuse_returns_none_when_no_quality_signal_overlaps():
    row = pd.Series({"top_station": 5, "llr": 12.0, "t_mid_s": 100.0})
    empty_quality = pd.DataFrame(columns=["window", "station", "llr"])
    wb = pd.DataFrame({"window": [0], "t_lo": [90.0], "t_hi": [110.0]})
    result = fuse_ambiguous_group(row, empty_quality, [5, 6], 90.0, 110.0, wb)
    assert result is None


def test_fuse_picks_the_station_quality_evidence_favors():
    row = pd.Series({"top_station": 5, "llr": 3.0, "t_mid_s": 100.0})
    quality = pd.DataFrame(
        [
            {"window": 0, "station": 5, "llr": 0.5},
            {"window": 0, "station": 6, "llr": 9.0},  # much stronger quality evidence for station 6
        ]
    )
    wb = pd.DataFrame({"window": [0], "t_lo": [90.0], "t_hi": [110.0]})
    result = fuse_ambiguous_group(row, quality, [5, 6], 90.0, 110.0, wb)
    assert result is not None
    assert result["fused_top_station"] == 6


def test_infer_default_behavior_is_unchanged_when_fusion_disabled():
    """Critical regression guard: twin.pipeline.infer's shadow output must be
    byte-identical with and without the fusion attributes present-but-off on
    the context object, on a real scenario. This proves the additive touch
    to pipeline.py::infer has zero effect on every existing caller -- none of
    which sets enable_evidence_fusion/quality_baseline at all, which is
    exactly the "attribute absent" case getattr(..., False) covers, and this
    test additionally covers the "attribute present but off" case explicitly.
    """
    nominal = simulate(LINE, SC.nominal_run(1800), seed=1)
    calib = simulate(LINE, SC.nominal_run(1500), seed=2)
    ctx = fit_context(LINE, nominal, calibration_run=calib, target_window_fpr=0.01)

    scen = SC.scenario_hidden_bottleneck(LINE)
    res = simulate(LINE, scen, seed=20260301)

    _, shadow_default, _ = infer(ctx, res)

    # Explicitly present but off/absent -- must produce identical output.
    ctx.enable_evidence_fusion = False
    ctx.quality_baseline = None
    _, shadow_explicit_off, _ = infer(ctx, res)

    assert shadow_default.equals(shadow_explicit_off)

    # enable_evidence_fusion=True with no quality_baseline set must ALSO have
    # zero effect -- both are required for fusion to run.
    ctx.enable_evidence_fusion = True
    ctx.quality_baseline = None
    _, shadow_half_enabled, _ = infer(ctx, res)
    assert shadow_default.equals(shadow_half_enabled)
