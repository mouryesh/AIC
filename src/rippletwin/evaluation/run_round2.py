"""Single reproducible entry point for every experiment added in the Round 2
upgrade (Phases 2-9). Regenerates every table `evaluation.report`'s Round 2
appendix reads -- run this before `python -m rippletwin.evaluation.report`.

    python -m rippletwin.evaluation.run_round2                 # everything
    python -m rippletwin.evaluation.run_round2 --seed 42        # a different seed
    python -m rippletwin.evaluation.run_round2 --quick          # smaller episode counts, for a fast check

Every module invoked here already states its own seeds and config in its
own manifest JSON under results/tables/; this script's job is only to run
them all with one command and report how long it took, per the brief's
"regenerate from a clean environment with one command" requirement.
"""

from __future__ import annotations

import argparse
import time

from . import calibration as CAL
from . import coverage_matrix as CM
from . import defect_prediction as DP
from . import distribution_shift as DS
from . import early_warning as EW
from . import feedback_experiment as FB
from . import surge as SURGE
from . import topology_experiment as TOPO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--quick", action="store_true", help="smaller episode counts, for a fast sanity check")
    ap.add_argument("--seed", type=int, default=None, help="override the base episode seed for every experiment")
    args = ap.parse_args()

    t0 = time.time()
    n = 4 if args.quick else None  # None -> each module's own default

    print("=== early warning ===")
    ew_cfg = EW.EarlyWarningConfig(n_random_episodes=n or 16)
    EW.run_early_warning_experiment(ew_cfg, out_dir=args.out_dir)

    print("\n=== defect prediction ===")
    dp_cfg = DP.DefectPredictionConfig(n_episodes=n or 14)
    DP.run_defect_prediction_experiment(dp_cfg, out_dir=args.out_dir)

    print("\n=== sensor coverage matrix ===")
    cm_cfg = CM.CoverageMatrixConfig(n_episodes=n or 10)
    CM.run_coverage_matrix(cm_cfg, out_dir=args.out_dir)

    print("\n=== distribution shift ===")
    ds_cfg = DS.DistributionShiftConfig(n_episodes=n or 12)
    DS.run_distribution_shift_experiment(ds_cfg, out_dir=args.out_dir)

    print("\n=== calibration ===")
    cal_cfg = CAL.CalibrationConfig(n_episodes=n or 14)
    CAL.run_calibration_experiment(cal_cfg, out_dir=args.out_dir)

    print("\n=== topology generalization (Plant A/B/C) ===")
    TOPO.run_topology_experiment(out_dir=args.out_dir, n_episodes=n or 8)

    print("\n=== feedback loop ===")
    fb_cfg = FB.FeedbackExperimentConfig(n_episodes=n or 14)
    FB.run_feedback_experiment(fb_cfg, out_dir=args.out_dir)

    print("\n=== surge / performance ===")
    surge_cfg = SURGE.SurgeConfig(surge_vehicles=2000 if args.quick else 6000)
    SURGE.run_surge_test(surge_cfg, out_dir=args.out_dir)

    print(f"\nall Round 2 experiments done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
