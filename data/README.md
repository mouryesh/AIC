# Data

**This directory is empty by design, and stays that way in version control.**

RippleTwin uses no proprietary or real production data. Every dataset it
consumes is generated on demand by the simulator in
`src/rippletwin/factory/`, from a fixed seed.

That means there is nothing to download and nothing to keep in sync: the data is
a deterministic function of the code and the seed, so a clone reproduces byte-
identical inputs.

## Generating a dataset

```bash
python -m rippletwin.data.generate --scenario S1_HIDDEN_BOTTLENECK --out data/raw
```

## What gets written

| File | Contents | Who may read it |
|---|---|---|
| `telemetry.csv` | observed stations only: cycle, blocked, starved, buffer, process channels | **model input** |
| `inspections.csv` | gate results: pass/fail, defect types found | **model input** |
| `vehicles.csv` | release log: variant, shift | **model input** |
| `environment.csv` | ambient temperature and humidity | **model input** |
| `passes.csv` | every station including blind ones, with true processing times | **evaluation only** |
| `disturbances.csv` | the injected faults, with true source station and magnitude | **evaluation only** |
| `defects.csv` | every defect with its true source station | **evaluation only** |

The split is not a convention — it is enforced in code. `telemetry.csv` is built
by filtering to `line.observed_indices`, and the ground-truth columns
(`true_health`, `true_defect_mult`) never appear in it. Two tests pin this down:
`test_hidden_stations_emit_no_telemetry` and
`test_ground_truth_is_separate_from_telemetry`.
