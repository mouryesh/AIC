# User guide — running the pilot tool on your own data

This is a walkthrough for a plant engineer or manufacturing IT person, not
a developer. Every command below is copied from, or directly verified
against, `src/rippletwin/pilot.py`'s actual `argparse` setup — nothing here
is aspirational. If a command in this guide stops matching the code, the
code is right and this file is stale; file an issue.

Related reading, in case you land here first: [docs/DEPLOYMENT.md](DEPLOYMENT.md)
covers what RippleTwin needs from your plant and where it sits on your
network; this guide covers the actual commands.

---

## 0. What this tool is and isn't

`rippletwin.pilot` is a **Phase 0 readiness assessment**, not a
deployment. You give it an export of data your plant already has, and it
tells you, in order:

1. Whether the export is even usable, and what's wrong if it isn't.
2. What line topology it inferred, and which assumptions an engineer has
   to confirm.
3. A capability verdict: `FULL`, `FLOW_ONLY`, `QUALITY_ONLY`, or `NOT_VIABLE`.
4. Findings — which stations (including blind ones) the twin flagged,
   over the period you gave it.
5. Work orders — who should look, by when, and what it's costing to wait.

**No credentials, no network access, no OT connection, no port opened.**
It reads files you export and writes a text report and a JSON summary. It
never writes to a PLC or control system, and it never sends anything
anywhere.

---

## 1. Try it without a plant first

Before touching real data, run it against a synthetic export the repo
generates for you — this proves the tool works on your machine and shows
you what the output actually looks like:

```bash
python demo/make_plant_export.py
python -m rippletwin.pilot --export demo/plant_export/mapping.yaml
```

`demo/make_plant_export.py` writes what a real historian export actually
looks like — awkward column names (`Equipment`, `SerialNo`, `EventTime`),
ISO timestamp strings, no pre-joined dwell times — and injects a slowdown
at a station with **no telemetry at all**, which then never appears in the
export. Naming that station is the whole test. The ground truth is
written to `demo/plant_export/ANSWER.txt`, which the pilot tool never
reads, so you can check the result rather than take it on trust.

---

## 2. Get a blank mapping file

```bash
python -m rippletwin.pilot --emit-template mapping.yaml
```

This writes a blank mapping YAML and exits — nothing else runs. Open
`mapping.yaml`. You'll fill in the **left-hand side** of each entry with
**your own column and file names**; the right-hand side is fixed, because
that's what the rest of the tool expects internally. You are not
renaming your files or your data — only telling RippleTwin what to call
them.

The template has four sections:

- **`line:`** — `takt_s` (planned seconds per unit), `n_stations` (total,
  instrumented or not), `clock_sync_s` (worst-case clock skew between
  your station time sources — see §5 below, this one matters more than
  it looks).
- **`files:`** — either **Shape A** (`states` + `scans`: your PLC/OEE
  state-change log and your MES VIN-scan log, the common case — the tool
  does the interval join for you) or **Shape B** (`telemetry`: one file,
  if a reporting layer already produced per-unit dwell times). Use one
  shape, delete the other. `vehicles` (build sequence) is required;
  `inspections` (gate results) is optional and unlocks defect
  attribution; `environment` is optional.
- **`columns:`** — for each file, map your actual column headers
  (left) to what RippleTwin needs (right). Nothing is renamed on disk;
  this is a translation table.
- **`line.stations`** (add this yourself — it isn't pre-filled) — an
  **ordered list of every station, instrumented or not**. This is the
  single highest-value thing you can supply. Without it, blind stations
  are placed by guesswork — and in our own testing, that guess moved a
  localisation **twelve stations** away from the truth. With it, it's a
  day with a controls engineer reading off an equipment list, not a data
  project.

---

## 3. Run the assessment

```bash
python -m rippletwin.pilot --export mapping.yaml
```

That's the whole command for a default run. What actually happens, in
order (this is `pilot.py::run_pilot`, not marketing copy):

1. **Load** — reads your files per the mapping.
2. **Validate** — checks the export is usable at all (row counts, station
   coverage, whether the state log's span matches the VIN reads' span per
   station — a state log truncated to a fraction of its rows returns a
   verdict of `NOT USABLE` rather than silently under-reporting).
3. **Infer topology** — builds the line from your data, prints every
   assumption it had to make so an engineer can check them.
4. **Phase 0 capability verdict** — `FULL` / `FLOW_ONLY` / `QUALITY_ONLY`
   / `NOT_VIABLE`, with named blockers if it's not viable. If your line
   turns out to be fully instrumented, it will tell you that too — and
   that shadow-sensing has nothing to add on a line with nothing hidden.
5. **Fit and score** — see §4 below for the one assumption that actually
   matters here.
6. **Findings** — how many windows fired, and specifically how many
   named a **blind** station (no sensor at all).
7. **Work orders** — owner, respond-by time, and the cost of waiting, for
   each distinct finding. A monitor-only recommendation raises no work
   order on purpose — inventing a task out of "keep an eye on it" is how
   alert fatigue starts.

To save the report and a machine-readable JSON summary instead of just
printing to the terminal:

```bash
python -m rippletwin.pilot --export mapping.yaml --out results/
```

This writes `results/pilot_report.txt` and `results/pilot_summary.json`.

---

## 4. The one assumption you should not skip

The twin needs a **known-normal period** to learn what normal looks
like. Your export doesn't label which hours were normal, so by default
the tool assumes the **earliest slice is representative** and says so in
the report every time. Getting this wrong doesn't produce a wrong answer
loudly — it produces a baseline with a fault already absorbed into it,
which *suppresses* the signal rather than inventing a false one. (That's
the deliberate failure direction: missing a fault is recoverable, crying
wolf is not — false alarms are the documented way these systems lose the
floor.)

If you know a clean period, name it:

```bash
python -m rippletwin.pilot --export mapping.yaml --baseline-vehicles 0:5000
```

`A:B` is a **unit-index range**, not a date — the report tells you the
unit range it actually scored, so you can cross-check it against your own
production log for that window. Two more flags exist for finer control
over the split between baseline and calibration periods, with sane
defaults if you don't set them:

```bash
--baseline-frac 0.4    # default: first 40% of the export, if --baseline-vehicles isn't given
--calib-frac 0.2        # default: next 20%, used to calibrate the detection threshold
```

---

## 5. Reading the report

A few things worth knowing before you read your first real one:

- **`ASSUMPTIONS AN ENGINEER MUST CONFIRM`** in the topology section is
  not boilerplate — read every line. If you didn't supply `line.stations`,
  the first assumption listed is exactly that gap, stated as a warning,
  not buried.
- **`naming a BLIND station`** in the findings section is the number that
  matters most. A conventional twin can only ever report on instrumented
  stations; this is the count of times RippleTwin resolved into a gap a
  conventional twin structurally cannot see into.
- **Clock sync matters more than it looks.** `clock_sync_s` in your
  mapping file should reflect your actual worst-case skew between time
  sources, not a guess of zero. Events timestamped on arrival at a
  historian rather than at the PLC will show queueing delay as cycle-time
  variation, and the readiness assessment flags this explicitly rather
  than silently mis-scoring it.
- **`STOPPING:`** anywhere in the report means the tool refused to
  produce a result rather than produce a confident wrong one. Every stop
  condition names the specific blocker and what to do about it — that's
  deliberate; a wrong answer that looks like a right one is the failure
  mode this whole design is built to avoid (see the bug table in the
  main [README](../README.md#run-it-on-a-real-plants-data) for five real
  examples this caught during testing).

---

## 6. What happens next

The report's own last section says it plainly: **Phase 1 is shadow
mode.** Run the assessment weekly (or however your data export cadence
allows), log every finding, and have a technician record found/not-found
against each one. After 8–12 weeks you have per-station precision
measured on *your* line — the only number that should decide whether this
goes live, not anything in this repository.
