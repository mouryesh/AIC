"""Charts generated from the experiment tables.

Every figure here reads a CSV that ``run_experiment`` actually wrote. Nothing is
drawn from a hard-coded number, so a figure cannot drift away from the result it
claims to show: if the experiment has not been run, these functions fail rather
than produce a plausible-looking picture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = {
    "RippleTwin": "#7500c0",
    "B3_TurningPoint": "#d62728",
    "B2_observed_only_twin": "#1f77b4",
    "B1_IsolationForest": "#ff7f0e",
    "B0_SPC_observed": "#7f7f7f",
}
LABEL = {
    "RippleTwin": "RippleTwin (shadow-sensing)",
    "B3_TurningPoint": "B3 Turning Point Method (Li et al. 2009)",
    "B2_observed_only_twin": "B2 observed-only twin",
    "B1_IsolationForest": "B1 anomaly detection",
    "B0_SPC_observed": "B0 SPC on sensors",
}
FOOTER = "Simulated prototype result on synthetic data — not a real production line."


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


METRIC_LABEL = {
    "within1": "source located within one station (%)",
    "top1": "exact source station identified (%)",
}


def coverage_curve(tables: Path, out: Path, metric: str = "within1") -> Optional[Path]:
    """Localisation accuracy against sensor coverage, per method.

    This is the figure that carries the business case: it shows how much
    capability survives as instrumentation is removed.

    ``metric`` matters for honesty as much as for content. ``within1`` gives an
    observed-only twin credit for naming a station adjacent to the true blind
    one, which is genuinely useful to a technician; ``top1`` does not. Pairing a
    ``within1`` chart with a ``top1`` table on the same slide reads as a
    contradiction, so callers should keep the two consistent.
    """
    src = tables / "hidden_source_only.csv"
    fallback = tables / "baseline_comparison.csv"
    path = src if src.exists() else fallback
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if metric not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for m, g in df.groupby("method"):
        g = g.sort_values("coverage")
        ax.plot(g["coverage"] * 100, g[metric] * 100, marker="o", linewidth=2,
                color=PALETTE.get(m, "#333"), label=LABEL.get(m, m))
    _style(ax,
           "Localisation accuracy vs sensor coverage",
           "stations with a sensor (%)",
           METRIC_LABEL.get(metric, metric))
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01, FOOTER, fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def operating_curve(tables: Path, out: Path, coverage: float = 0.75) -> Optional[Path]:
    """Detection rate against false-alarm rate — the fair comparison."""
    path = tables / "operating_curve.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[np.isclose(df["coverage"], coverage)]
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for m, g in df.groupby("method"):
        g = g.sort_values("false_alarm_rate")
        ax.plot(g["false_alarm_rate"] * 100, g["detection_rate"] * 100,
                marker="o", linewidth=2, color=PALETTE.get(m, "#333"),
                label=LABEL.get(m, m))
    _style(ax,
           f"Detection vs false alarms at {coverage * 100:.0f}% sensor coverage",
           "false-alarm rate per window (%)",
           "detection rate while a fault is active (%)")
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01,
             "Every method calibrated on the same held-out nominal data. " + FOOTER,
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def cycle_time_accuracy(tables: Path, out: Path) -> Optional[Path]:
    """Error of the inferred cycle time at stations with no sensor."""
    path = tables / "cycle_time_raw.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["split"] == "test"] if "split" in df.columns else df
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    groups, labels, colors = [], [], []
    for hidden, lbl, col in [(True, "no sensor (inferred)", "#7500c0"),
                             (False, "instrumented (measured)", "#1f77b4")]:
        g = df[df["source_hidden"] == hidden]["abs_error_pct"].dropna()
        if len(g):
            groups.append(g.to_numpy())
            labels.append(f"{lbl}\nn={len(g)}")
            colors.append(col)
    if not groups:
        return None
    bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    for med in bp["medians"]:
        med.set_color("#111")
        med.set_linewidth(1.6)
    _style(ax, "Error of the inferred station cycle time", "", "absolute error (%)")
    fig.text(0.01, 0.01,
             "Ground truth known only to the simulator. " + FOOTER,
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def false_alarm_bars(tables: Path, out: Path) -> Optional[Path]:
    """False-alarm rate per method per coverage, at the shared operating point."""
    path = tables / "false_alarms.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    covs = sorted(df["coverage"].unique())
    methods = [m for m in LABEL if m in set(df["method"])]
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    x = np.arange(len(covs))
    for j, m in enumerate(methods):
        g = df[df["method"] == m].set_index("coverage").reindex(covs)
        ax.bar(x + j * width, g["false_alarm_rate"].to_numpy() * 100, width,
               color=PALETTE.get(m, "#333"), label=LABEL.get(m, m))
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"{c * 100:.0f}%" for c in covs])
    _style(ax, "False-alarm rate at the shared operating point",
           "sensor coverage", "false alarms per window (%)")
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01, FOOTER, fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def pressure_profile_figure(line, scored, window, true_station, out: Path) -> Path:
    """The blocking/starvation profile — the mechanism, drawn from real data."""
    g = scored[scored["window"] == window].sort_values("station")
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=160)

    ax.bar(g["station"], g["d_blocked"] * 100, color="#d63a3a",
           label="blocked above normal (work cannot leave)")
    ax.bar(g["station"], -g["d_starved"] * 100, color="#4c9be8",
           label="starved above normal (work is not arriving)")

    for i in line.hidden_indices:
        ax.axvspan(i - 0.5, i + 0.5, color="#9aa0a6", alpha=0.20, zorder=0)
    if true_station is not None:
        ax.axvline(true_station, color="#111", linestyle=":", linewidth=1.6)
        ax.annotate(f"true source\n{line.stations[true_station].station_id} (no sensor)",
                    xy=(true_station, ax.get_ylim()[1] * 0.72),
                    xytext=(true_station + 3, ax.get_ylim()[1] * 0.72),
                    fontsize=8, arrowprops=dict(arrowstyle="->", lw=1))
    ax.axhline(0, color="#333", linewidth=0.8)
    _style(ax,
           "The boundary between blocking and starvation locates the constraint",
           "station index  (grey bands = no sensor)", "% of takt")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.text(0.01, 0.01,
             "Conservation of material through a serial line, not a learned "
             "correlation. " + FOOTER, fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_all(results_dir: str | Path = "results") -> list:
    """Generate every figure that has data behind it."""
    results = Path(results_dir)
    tables, figures = results / "tables", results / "figures"
    made = []
    for fn, name in [
        (lambda: coverage_curve(tables, figures / "coverage_curve.png"), "coverage_curve"),
        (lambda: coverage_curve(tables, figures / "coverage_curve_top1.png",
                                metric="top1"), "coverage_curve_top1"),
        (lambda: operating_curve(tables, figures / "operating_curve.png"), "operating_curve"),
        (lambda: cycle_time_accuracy(tables, figures / "cycle_time_accuracy.png"), "cycle_time"),
        (lambda: false_alarm_bars(tables, figures / "false_alarms.png"), "false_alarms"),
    ]:
        p = fn()
        if p:
            made.append(p)
    return made
