"""Phase 0 CLI: can this plant run RippleTwin, and on what?

    python -m rippletwin.integrate.assess --demo
    python -m rippletwin.integrate.assess --signals station_state,build_sequence,... \
        --stations 42 --instrumented 30

Intended to be run in the first meeting, before anyone commits to anything.
The most useful answer it can give is "not yet, and here is what is missing" --
which is the answer most digital-twin projects should have received and did not.
"""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd

from .contract import DATA_CONTRACT, assess_readiness, contract_frame


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RippleTwin data-readiness assessment")
    ap.add_argument("--signals", default="",
                    help="comma-separated contract keys the plant can supply")
    ap.add_argument("--stations", type=int, default=42)
    ap.add_argument("--instrumented", type=int, default=32,
                    help="stations emitting usable running/blocked/starved state")
    ap.add_argument("--clock-skew-s", type=float, default=None)
    ap.add_argument("--contract", action="store_true",
                    help="print the full input contract and exit")
    ap.add_argument("--demo", action="store_true",
                    help="assess the synthetic reference line")
    args = ap.parse_args(argv)

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)

    if args.contract:
        print("RIPPLETWIN INPUT CONTRACT — everything the twin reads\n")
        print(contract_frame().to_string(index=False))
        print("\nNothing outside this list is consumed. Nothing is written back.")
        return 0

    if args.demo:
        signals = [s.key for s in DATA_CONTRACT if s.key != "process_channels"]
        stations, instrumented = 42, 32
        print("Assessing the synthetic reference line "
              "(no torque/vibration channels).\n")
    else:
        signals = [s.strip() for s in args.signals.split(",") if s.strip()]
        stations, instrumented = args.stations, args.instrumented

    known = {s.key for s in DATA_CONTRACT}
    unknown = [s for s in signals if s not in known]
    if unknown:
        print(f"WARNING: unrecognised signal keys ignored: {', '.join(unknown)}")
        print(f"         valid keys: {', '.join(sorted(known))}\n")

    report = assess_readiness(
        signals, n_stations=stations, n_stations_with_state=instrumented,
        clock_sync_s=args.clock_skew_s,
    )
    print("=" * 78)
    print("  RIPPLETWIN DATA READINESS ASSESSMENT")
    print("=" * 78)
    print(report.summary())
    print("\n" + "-" * 78)
    print("SIGNAL BY SIGNAL")
    print("-" * 78)
    f = report.to_frame()
    print(f[["signal", "required", "available", "enables"]].to_string(index=False))
    print("\nRun with --contract for the full contract including interfaces and "
          "what each missing signal costs you.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
