"""Tests for evaluation.report.build_round2_appendix -- must not crash on a
missing or partial results directory, and must produce a real file when the
tables exist."""

from __future__ import annotations

from rippletwin.evaluation.report import build_round2_appendix


def test_missing_tables_directory_does_not_crash(tmp_path):
    out = build_round2_appendix(results_dir=tmp_path, out=tmp_path / "RESULTS_ROUND2.md")
    assert out.exists()
    text = out.read_text()
    assert "SIMULATED PROTOTYPE RESULT" in text.upper() or "simulated prototype result" in text


def test_appendix_reads_real_generated_tables(tmp_path):
    from rippletwin.evaluation.early_warning import EarlyWarningConfig, run_early_warning_experiment

    run_early_warning_experiment(EarlyWarningConfig(n_random_episodes=2), out_dir=tmp_path, verbose=False)
    out = build_round2_appendix(results_dir=tmp_path, out=tmp_path / "RESULTS_ROUND2.md")
    text = out.read_text()
    assert "Early bottleneck prediction" in text
    assert "Lead time" in text
