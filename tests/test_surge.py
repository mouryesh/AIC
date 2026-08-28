"""Tests for the surge/performance experiment (evaluation.surge)."""

from __future__ import annotations

from rippletwin.evaluation.surge import SurgeConfig, run_surge_test


def test_surge_test_runs_and_reports_expected_fields(tmp_path):
    cfg = SurgeConfig(surge_vehicles=800, nominal_vehicles=1200, calibration_vehicles=1000)
    result = run_surge_test(cfg, out_dir=tmp_path, verbose=False)
    for key in (
        "n_windows", "fit_context_latency_s", "simulate_latency_s",
        "infer_latency_s", "infer_latency_per_window_ms", "peak_memory_mb",
        "throughput_vph", "false_alarm_rate",
    ):
        assert key in result
    assert result["n_windows"] > 0
    assert result["infer_latency_s"] >= 0
    assert (tmp_path / "tables" / "surge_test.json").exists()
