"""Smoke test for the single reproducible entry point
(evaluation.run_round2) that regenerates every Round 2 experiment table.

Each constituent experiment already has its own dedicated test file; this
only checks the CLI wiring itself runs end to end with --quick.
"""

from __future__ import annotations

import sys

from rippletwin.evaluation import run_round2


def test_run_round2_quick_mode_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["run_round2", "--quick", "--out-dir", str(tmp_path)],
    )
    run_round2.main()  # must not raise
    assert (tmp_path / "tables" / "early_warning_summary.csv").exists()
    assert (tmp_path / "tables" / "topology_summary.csv").exists()
    assert (tmp_path / "tables" / "surge_test.json").exists()
