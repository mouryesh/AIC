# Signals

Every signal below already exists in the simulator and is consumed by at
least one model in this repository. None were added for visual complexity —
each has a stated role, a stated failure mode it helps identify, and a
stated consumer. No new sensor types are introduced in the Round 2 upgrade;
the existing signal set already covers every role the brief asks for.

| Signal | What it represents | What failure it helps identify | Which model consumes it |
|---|---|---|---|
| `proc_time_s` (cycle time) | How long a station actually held a vehicle | A station running slower than its own mix/shift-adjusted expectation — the direct evidence a station is the constraint | `features/baseline.py::NominalBaseline` (z_proc), `twin/shadow.py`'s direct-evidence likelihood term, `twin/defect_risk.py` (a rushed cycle correlates with defect probability in the simulator's own physics) |
| `blocked_s` | Time a station finished a vehicle but could not release it (buffer ahead full) | A slow station **downstream** of this one — the upstream signature of a constraint | `twin/shadow.py`'s propagation-matrix fit (`B`), `twin/propagate.py`'s upstream forecast |
| `starved_s` | Time a station sat idle waiting for a part | A slow station **upstream** of this one — the downstream signature of a constraint, and the one that arrives first (see `factory/simulator.py`'s module docstring on timing asymmetry) | `twin/shadow.py`'s propagation-matrix fit (`S`), `twin/propagate.py`'s downstream forecast, the primary source of early-warning lead time in `twin/predict.py` |
| `buffer_level` / `buffer_capacity` | Current occupancy of the buffer immediately after a station | How much slack remains before a developing constraint actually starves/blocks its neighbours | `twin/propagate.py::forecast_ripple`'s minutes-to-impact estimate, `twin/predict.py`'s buffer-fill trend |
| `torque_nm` | Fastening/joining torque at a RICH station | Mechanical degradation (tooling wear, fixture binding) — rises with the same `health` multiplier that slows a station down | `features/baseline.py` (z_torque), `twin/defect_risk.py`'s predictive risk score |
| `vibration_mm_s` | Mechanical vibration at a RICH station | Bearing/tooling wear, more sensitively than torque (`simulator.py`'s vibration channel scales ~4x faster with degradation than torque) | `twin/defect_risk.py`, RICH-station confidence ceiling logic |
| `station_temp_c` | Local process temperature at a RICH station | Thermal drift correlated with degradation, net of the ambient/diurnal cycle | `twin/defect_risk.py` |
| `ambient_temp_c` / `humidity_pct` | Plant-floor environment | A genuine confounder for paint-zone defects — humidity above ~65% raises defect probability independent of any station fault; a model that ignores this blames a station for weather | `features/windows.py::attach_environment`, quality attribution's humidity-aware discussion in `docs/METHOD.md` |
| Inspection gate result (`PASS`/`FAIL`, defect type) | The plant's own quality record: a defect was found, and its type | The plant's ground truth for **where a defect was caught**, never where it was made | `twin/genealogy.py`'s attribution path (post-hoc: trace an already-found defect to its likely source) |
| PLC-derived `data_quality` tag (OBSERVED/NOISY/STALE, `factory/sensor_health.py`) | Whether a reading is trustworthy right now | Sensor degradation itself — a stuck or corrupted channel, distinct from the process it is supposed to measure | `factory/sensor_health.py::flag_stale_windows`, surfaced to the dashboard as OBSERVED/INFERRED/SUSPECT |

## Physics/statistics first, ML second

Every signal above feeds a **derived, checkable quantity** (a z-score, a
takt fraction, a fitted amplitude) before it ever reaches a decision. None
of them are fed raw into a black-box classifier. Where the combination
genuinely needs weighting several channels against each other —
`twin/defect_risk.py`'s risk score — the combination is a small, calibrated
logistic on the derived channels, not an opaque model over raw sensor
values. See `README.md`'s "Where AI is used, and where it deliberately is
not" for the fuller argument.
