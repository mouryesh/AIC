"""RippleTwin dashboard -- three stakeholder views over one twin.

Run with:
    streamlit run app/dashboard.py

The three views are not three products. They read the same inferred line state
and differ only in what they aggregate and how far ahead they look:

    Floor supervisor -- this shift, this station, what do I do in the next hour
    Plant manager    -- this week, which stations keep costing me, where to invest
    Leadership       -- the rollout case, and what instrumentation it avoids

Throughout, every value is tagged OBSERVED, INFERRED or PREDICTED. A supervisor
who cannot tell a measurement from an estimate will eventually trust the wrong
one, and the first time an inferred value is wrong the whole system loses the
floor. Keeping the distinction visible is a safety property, not decoration.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from rippletwin.explain.explain import explain_flow_alert  # noqa: E402
from rippletwin.factory import scenarios as SC  # noqa: E402
from rippletwin.factory.topology import build_line  # noqa: E402
from rippletwin.hitl.ledger import (  # noqa: E402
    DECISION_APPROVED,
    DECISION_REJECTED,
    OUTCOME_CONFIRMED,
    OUTCOME_NOT_FOUND,
    DecisionLedger,
)
from rippletwin.recommend.engine import recommend_flow  # noqa: E402
from rippletwin.twin import genealogy as GN  # noqa: E402
from rippletwin.twin.pipeline import fit_context, infer, simulate  # noqa: E402
from rippletwin.twin.placement import ambiguity, recommend_sensors  # noqa: E402
from rippletwin.twin.propagate import current_buffer_levels, forecast_ripple  # noqa: E402
from rippletwin.twin.shadow import infer_hidden_cycle_time  # noqa: E402

CONFIG = ROOT / "configs" / "line_42.yaml"
LINE_SEED = 7
DEMO_SEED = 20260301

TIER_COLOR = {"RICH": "#1f9d55", "BASIC": "#4c9be8", "MANUAL": "#9aa0a6"}
RISK_SCALE = [[0.0, "#1f9d55"], [0.5, "#f2b705"], [1.0, "#d63a3a"]]

st.set_page_config(page_title="RippleTwin", layout="wide",
                   initial_sidebar_state="expanded")


# ----------------------------------------------------------------- data layer


@st.cache_resource(show_spinner="Fitting the twin on disturbance-free production...")
def load_twin():
    line = build_line(CONFIG, seed=LINE_SEED)
    nominal = simulate(line, SC.nominal_run(2600), seed=1)
    calib = simulate(line, SC.nominal_run(2200), seed=2)
    ctx = fit_context(line, nominal, calibration_run=calib, target_window_fpr=0.01)
    prior = GN.candidate_prior(line)
    qbase = GN.QualityBaseline.fit(
        line,
        GN.attribute_defects(line, GN.explode_defects(nominal.inspections), prior),
        n_vehicles=len(nominal.vehicles),
    )
    return line, ctx, qbase, nominal


@st.cache_data(show_spinner="Running production...")
def run_scenario(scenario_key: str, _line, _ctx, _qbase):
    builders = {
        "S1_HIDDEN_BOTTLENECK": SC.scenario_hidden_bottleneck,
        "S2_HIDDEN_QUALITY": SC.scenario_hidden_quality,
        "S3_NORMAL": SC.scenario_normal_variation,
        "S4_OBSERVED_STATION": SC.scenario_observed_station,
        "S5_VARIANT_AND_SUPPLY": SC.scenario_variant_shift,
    }
    scen = builders[scenario_key](_line)
    res = simulate(_line, scen, seed=DEMO_SEED)
    scored, shadow, sensor = infer(_ctx, res)
    wb = GN.window_bounds_from(scored)
    qs = GN.quality_state(_line, GN.explode_defects(res.inspections), wb, _qbase,
                          pool_vehicles=200)
    qa = GN.quality_alerts(qs) if len(qs) else qs
    return scen, res, scored, shadow, sensor, qa


# ------------------------------------------------------------------- visuals


def line_map(line, scored, shadow_row, window, risk_by_station):
    """Station topology coloured by inferred risk, shaped by instrumentation."""
    n = line.n_stations
    xs, ys, texts, colors, symbols, sizes, borders = [], [], [], [], [], [], []
    per_row = 14
    for s in line.stations:
        row = s.index // per_row
        col = s.index % per_row
        xs.append(col)
        ys.append(-row)
        r = float(risk_by_station.get(s.index, 0.0))
        colors.append(r)
        # Hidden stations are drawn as hollow diamonds: their state is inferred,
        # and the shape says so before any number is read.
        symbols.append("diamond" if s.is_hidden else "circle")
        sizes.append(30 if s.is_hidden else 26)
        borders.append("#111111" if s.is_hidden else "#ffffff")
        texts.append(
            f"<b>{s.station_id}</b> ({s.zone})<br>"
            f"{'NO SENSOR - state inferred' if s.is_hidden else s.tier + ' - measured'}<br>"
            f"risk {r:.2f}"
            + (f"<br>gate: {s.inspection_id}" if s.is_inspection else "")
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=sizes, color=colors, colorscale=RISK_SCALE, cmin=0, cmax=1,
                    symbol=symbols, line=dict(width=2, color=borders),
                    colorbar=dict(title="risk", thickness=12)),
        text=[s.station_id for s in line.stations],
        textposition="middle center",
        textfont=dict(size=8, color="white"),
        hovertext=texts, hoverinfo="text", showlegend=False,
    ))
    # Flow arrows within each row
    for i in range(n - 1):
        if i // per_row == (i + 1) // per_row:
            fig.add_annotation(
                x=(i + 1) % per_row, y=-(i // per_row),
                ax=i % per_row, ay=-(i // per_row),
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=0.7,
                arrowwidth=1, arrowcolor="rgba(120,120,120,0.45)",
            )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def pressure_profile(line, scored, window, true_station=None):
    """The blocking / starvation profile the localisation actually reads."""
    g = scored[scored["window"] == window].sort_values("station")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=g["station"], y=g["d_blocked"] * 100, name="blocked (above normal)",
        marker_color="#d63a3a",
        hovertemplate="%{x}: blocked +%{y:.0f}% of takt<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=g["station"], y=-g["d_starved"] * 100, name="starved (above normal)",
        marker_color="#4c9be8",
        hovertemplate="%{x}: starved +%{customdata:.0f}% of takt<extra></extra>",
        customdata=g["d_starved"] * 100,
    ))
    for i in line.hidden_indices:
        fig.add_vrect(x0=i - 0.45, x1=i + 0.45, fillcolor="#9aa0a6",
                      opacity=0.18, line_width=0)
    if true_station is not None:
        fig.add_vline(x=true_station, line_color="#111111", line_dash="dot",
                      annotation_text="true source", annotation_position="top")
    fig.update_layout(
        barmode="relative", height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="station index (grey bands = no sensor)",
        yaxis_title="% of takt",
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def posterior_chart(line, sr):
    post = sr.evidence["station_post"]
    ids = [s.station_id for s in line.stations]
    colors = ["#9aa0a6" if s.is_hidden else "#4c9be8" for s in line.stations]
    top = sr.top_station
    colors[top] = "#d63a3a"
    fig = go.Figure(go.Bar(x=ids, y=post, marker_color=colors))
    fig.update_layout(
        height=230, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="P(this station is the constraint)",
        xaxis=dict(tickangle=-90, tickfont=dict(size=8)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def risk_timeline(shadow, truth_row=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=shadow["t_mid_s"] / 3600, y=shadow["llr"], mode="lines",
        name="evidence (log-likelihood ratio)", line=dict(color="#4c9be8", width=2),
    ))
    det = shadow[shadow["detected"]]
    if len(det):
        fig.add_trace(go.Scatter(
            x=det["t_mid_s"] / 3600, y=det["llr"], mode="markers",
            name="alert raised", marker=dict(color="#d63a3a", size=5),
        ))
    if truth_row is not None:
        fig.add_vrect(
            x0=truth_row["t_start_s"] / 3600, x1=truth_row["t_end_s"] / 3600,
            fillcolor="#f2b705", opacity=0.15, line_width=0,
            annotation_text="disturbance active (ground truth)",
            annotation_position="top left",
        )
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="hours into the run", yaxis_title="evidence",
        legend=dict(orientation="h", y=1.15),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --------------------------------------------------------------------- app

line, ctx, qbase, nominal = load_twin()

st.sidebar.title("RippleTwin")
st.sidebar.caption("Digital twin for a mixed-model vehicle assembly line")
view = st.sidebar.radio(
    "View", ["Floor supervisor", "Plant manager", "Leadership"], index=0
)
scenario_key = st.sidebar.selectbox(
    "Scenario",
    ["S1_HIDDEN_BOTTLENECK", "S2_HIDDEN_QUALITY", "S3_NORMAL",
     "S4_OBSERVED_STATION", "S5_VARIANT_AND_SUPPLY"],
    format_func=lambda k: {
        "S1_HIDDEN_BOTTLENECK": "S1 - hidden bottleneck (no sensor)",
        "S2_HIDDEN_QUALITY": "S2 - hidden quality drift",
        "S3_NORMAL": "S3 - normal variation (should stay quiet)",
        "S4_OBSERVED_STATION": "S4 - fault at an instrumented station",
        "S5_VARIANT_AND_SUPPLY": "S5 - mix change + supply delay (not a fault)",
    }[k],
)

scen, res, scored, shadow, sensor, qa = run_scenario(scenario_key, line, ctx, qbase)

truth_rows = res.disturbances[res.disturbances["kind"] != "MATERIAL_DELAY"] \
    if len(res.disturbances) else res.disturbances
truth_row = truth_rows.iloc[0] if len(truth_rows) else None
true_station = int(truth_row["station"]) if truth_row is not None else None

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Line** {line.n_stations} stations, takt {line.takt_s:.0f}s  \n"
    f"**Coverage** {line.coverage * 100:.0f}% instrumented  \n"
    f"**Blind stations** {len(line.hidden_indices)}"
)
show_truth = st.sidebar.checkbox("Reveal ground truth (evaluation only)", value=False)
st.sidebar.markdown("---")
st.sidebar.warning(
    "All figures are SIMULATED PROTOTYPE RESULTS on synthetic data. "
    "No real production data is used and no real-plant ROI is claimed."
)

if "ledger" not in st.session_state:
    st.session_state.ledger = DecisionLedger()
ledger: DecisionLedger = st.session_state.ledger

det = shadow[shadow["detected"]]

# ============================================================ FLOOR SUPERVISOR
if view == "Floor supervisor":
    st.title("Floor supervisor")
    st.caption("This shift. What is happening now, and what to do about it.")

    if len(det) == 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Active alerts", "0")
        c2.metric("Line state", "Normal")
        c3.metric("Mean P(nothing wrong)", f"{shadow['p_null'].mean():.2f}")
        st.success(
            "**No station-level alert.** The twin is not silent because it sees "
            "nothing -- it is silent because the evidence does not beat the "
            "no-fault hypothesis at the calibrated threshold."
        )
        if scen.expect_no_alert:
            st.info(f"This scenario is designed to produce no alert. {scen.notes}")
        else:
            st.info(
                "A pure quality drift keeps takt, so there is no flow signature. "
                "See the quality section below."
            )
        w = int(shadow["window"].iloc[len(shadow) // 2])
        risk = {i: 0.0 for i in range(line.n_stations)}
    else:
        idx = len(det) // 2
        w = int(det.iloc[idx]["window"])
        sr = next(r for r in sensor.last_results if r.window == w)
        k = sr.top_station
        stn = line.stations[k]

        est_cycle = infer_hidden_cycle_time(
            line, res.telemetry, k, sr.v_start, sr.v_end
        )
        fc = forecast_ripple(
            line, k, est_cycle or line.takt_s, horizon_min=60.0,
            buffer_levels=current_buffer_levels(scored, w),
        ) if est_cycle else None
        exp = explain_flow_alert(line, sr, fc, est_cycle)
        rec = recommend_flow(line, sr, fc)
        risk = {i: float(sr.evidence["station_post"][i]) for i in range(line.n_stations)}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Station", stn.station_id,
                  "INFERRED - no sensor" if stn.is_hidden else f"OBSERVED - {stn.tier}")
        c2.metric("Confidence", f"{sr.group_prob * 100:.0f}%")
        c3.metric("Cycle time",
                  f"{est_cycle:.0f}s" if est_cycle else "n/a",
                  f"takt {line.takt_s:.0f}s")
        c4.metric("Units at risk / hr",
                  f"{fc.units_lost_at_horizon:.0f}" if fc else "n/a")

        if stn.is_hidden:
            st.warning(
                f"**{stn.station_id} has no sensor.** Everything shown for this "
                f"station is inferred from the instrumented stations either side of "
                f"it. Confirm on the floor before committing to a repair."
            )

        st.subheader(exp.headline)
        a, b = st.columns([3, 2])
        with a:
            st.markdown(f"**What changed** — {exp.what_changed}")
            st.markdown(f"**Why it matters** — {exp.why_it_matters}")
            st.markdown(f"**What happens next** — {exp.what_happens_next}")
            st.markdown(f"**Confidence** — {exp.confidence_text}")
        with b:
            st.markdown("**Evidence**")
            for e in exp.evidence:
                tag = {"OBSERVED": "🟦", "INFERRED": "🟨", "PREDICTED": "🟥"}[e.provenance]
                st.markdown(f"{tag} `{e.provenance}` {e.text}")
        if exp.caveats:
            for cav in exp.caveats:
                st.caption(f"⚠️ {cav}")

        st.markdown("---")
        st.subheader(f"Recommended action — {rec.priority}")
        if rec.abstained:
            st.error(f"**{rec.title}** (the twin is abstaining)")
        else:
            st.info(f"**{rec.title}**")
        st.write(rec.detail)
        st.caption(f"Rationale: {rec.rationale}")
        if rec.alternatives:
            st.caption(f"If wrong: {rec.alternatives[0]}")

        d1, d2, d3 = st.columns(3)
        if d1.button("✅ Approve", use_container_width=True):
            e = ledger.record_alert(
                scen.scenario_id, w, "FLOW", stn.station_id, stn.tier,
                stn.is_hidden, rec.confidence, rec.as_dict(), exp.as_dict())
            ledger.record_decision(e.entry_id, DECISION_APPROVED, "supervisor")
            ledger.record_outcome(e.entry_id, OUTCOME_CONFIRMED,
                                  "Condition confirmed on the floor.")
            st.success(f"Approved and logged (entry {e.entry_id}).")
        if d2.button("❌ Reject", use_container_width=True):
            e = ledger.record_alert(
                scen.scenario_id, w, "FLOW", stn.station_id, stn.tier,
                stn.is_hidden, rec.confidence, rec.as_dict(), exp.as_dict())
            ledger.record_decision(e.entry_id, DECISION_REJECTED, "supervisor",
                                   "Supervisor judged this a false alarm.")
            ledger.record_outcome(e.entry_id, OUTCOME_NOT_FOUND,
                                  "Nothing found at the named station.")
            st.warning(f"Rejected and logged (entry {e.entry_id}).")
        d3.metric("Ledger entries", len(ledger.entries))

        st.markdown("---")
        st.subheader("The evidence the localisation actually reads")
        st.plotly_chart(
            pressure_profile(line, scored, w, true_station if show_truth else None),
            use_container_width=True)
        st.caption(
            "Blocking above the axis, starvation below. The boundary between them "
            "is where the constraint sits — including at stations with no sensor "
            "(grey bands). This is conservation of material, not a learned correlation."
        )
        st.plotly_chart(posterior_chart(line, sr), use_container_width=True)

    st.markdown("---")
    st.subheader("Line map")
    st.plotly_chart(line_map(line, scored, None, w, risk), use_container_width=True)
    st.caption("Diamonds = no sensor (state inferred). Circles = instrumented (measured).")

    st.subheader("Evidence over the shift")
    st.plotly_chart(risk_timeline(shadow, truth_row if show_truth else None),
                    use_container_width=True)

    if len(qa):
        q_al = qa[qa["quality_alert"]]
        if len(q_al):
            st.markdown("---")
            st.subheader("Quality attribution (second shadow-sensing path)")
            rank = (qa.groupby(["station", "station_id", "tier", "is_hidden"])["llr"]
                    .mean().reset_index().sort_values("llr", ascending=False).head(5))
            rank["m_hat"] = rank["station"].map(
                qa.groupby("station")["m_hat"].mean())
            st.dataframe(
                rank.rename(columns={
                    "station_id": "Station", "tier": "Instrumentation",
                    "is_hidden": "No sensor", "llr": "Evidence (LLR)",
                    "m_hat": "Est. defect multiplier"})[
                    ["Station", "Instrumentation", "No sensor",
                     "Evidence (LLR)", "Est. defect multiplier"]],
                use_container_width=True, hide_index=True)
            st.caption(
                "Cycle times are normal here. What changed is the *mix* of defect "
                "types reaching the gates, matched against each station's known "
                "failure modes."
            )

# ============================================================== PLANT MANAGER
elif view == "Plant manager":
    st.title("Plant manager")
    st.caption("This week. Where the recurring losses are, and where to invest.")

    last = line.n_stations - 1
    end = res.passes[res.passes["station"] == last]
    hours = res.meta["horizon_s"] / 3600
    actual = len(end) / hours
    target = 3600 / line.takt_s
    n_defects = res.meta["n_defects"]
    n_escaped = res.meta["n_escaped"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Throughput", f"{actual:.1f} veh/h", f"{actual - target:+.1f} vs target")
    c2.metric("Performance ratio", f"{actual / target * 100:.0f}%")
    c3.metric("Defects found", f"{n_defects}")
    c4.metric("Escaped all gates", f"{n_escaped}",
              f"{n_escaped / max(n_defects,1) * 100:.0f}% escape rate")

    st.info(
        "Throughput and defect counts are OBSERVED. The station attribution below "
        "is INFERRED for stations with no sensor."
    )

    st.subheader("Where the twin spent its suspicion")
    if len(shadow):
        counts = (shadow[shadow["detected"]]["top_station"].value_counts()
                  if shadow["detected"].any() else pd.Series(dtype=int))
        rows = []
        for s in line.stations:
            rows.append({
                "Station": s.station_id, "Zone": s.zone,
                "Instrumentation": "none (inferred)" if s.is_hidden else s.tier,
                "Alert windows": int(counts.get(s.index, 0)),
            })
        df = pd.DataFrame(rows)
        top = df[df["Alert windows"] > 0].sort_values("Alert windows", ascending=False)
        if len(top):
            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            st.success("No station accumulated alert time this run.")

    st.subheader("Sensor coverage by zone")
    s = line.summary()
    zrows = []
    for z, d in s["per_zone"].items():
        zrows.append({
            "Zone": z, "Stations": d["stations"],
            "Rich": d["rich"], "Basic": d["basic"], "No sensor": d["manual"],
            "Coverage": f"{(d['rich'] + d['basic']) / d['stations'] * 100:.0f}%",
        })
    st.dataframe(pd.DataFrame(zrows), use_container_width=True, hide_index=True)
    st.caption(
        "Zones with the least instrumentation are where a conventional twin is "
        "blind, and where inference contributes the most."
    )

    st.subheader("Defects by discovery gate")
    if len(res.inspections):
        g = (res.inspections.groupby("gate_id")
             .agg(inspected=("result", "size"),
                  failed=("result", lambda x: (x == "FAIL").sum()))
             .reset_index())
        g["fail rate"] = (g["failed"] / g["inspected"] * 100).round(2).astype(str) + "%"
        st.dataframe(g, use_container_width=True, hide_index=True)
        st.caption(
            "A defect caught at the end-of-line test has already accumulated the "
            "full value-add of every station after the one that caused it."
        )

    st.subheader("Decision ledger")
    lf = ledger.to_frame()
    if len(lf):
        st.dataframe(
            lf[["entry_id", "timestamp", "alert_type", "station_id",
                "is_inferred", "decision", "outcome"]],
            use_container_width=True, hide_index=True)
        st.caption(f"Hash chain valid: {ledger.verify()['valid']}")
    else:
        st.caption("No decisions recorded yet. Approve or reject an alert in the "
                   "supervisor view.")

# ================================================================= LEADERSHIP
else:
    st.title("Leadership")
    st.caption("The rollout case, and the instrumentation it avoids.")

    st.warning(
        "**Every figure on this page is an ILLUSTRATIVE ASSUMPTION**, shown so the "
        "arithmetic is inspectable. None of it is measured from a real plant. "
        "The prototype evidence lives in `results/tables/`; this page is the "
        "business model built on top of transparent inputs you can change."
    )

    st.subheader("Assumptions (edit these)")
    a1, a2, a3 = st.columns(3)
    veh_margin = a1.number_input("Contribution margin per vehicle (USD)",
                                 500, 20000, 2200, step=100)
    rework_cost = a2.number_input("Average rework cost per defect (USD)",
                                  50, 5000, 420, step=10)
    prod_hours = a3.number_input("Productive hours per year", 1000, 8000, 3800, step=100)

    b1, b2, b3 = st.columns(3)
    sensor_cost = b1.number_input("Fully-installed cost per station retrofit (USD)",
                                  1000, 100000, 18000, step=1000)
    n_blind = b2.number_input("Blind stations on this line",
                              1, 40, len(line.hidden_indices))
    deploy_cost = b3.number_input("RippleTwin deployment, year 1 (USD)",
                                  10000, 1000000, 150000, step=10000)

    c1, c2 = st.columns(2)
    events_per_year = c1.number_input(
        "Hidden-station disturbances per line per year", 1, 500, 60)
    minutes_saved = c2.number_input(
        "Minutes of earlier reaction per event", 1, 240, 25)

    st.markdown("---")
    st.subheader("The arithmetic")
    takt_per_hour = 3600 / line.takt_s
    veh_per_min = takt_per_hour / 60
    # Only a fraction of takt is genuinely recovered: a constraint is not fully
    # eliminated the moment it is found.
    recovery = 0.55
    units_recovered = events_per_year * minutes_saved * veh_per_min * recovery
    throughput_value = units_recovered * veh_margin

    defects_avoided = events_per_year * 6.0
    quality_value = defects_avoided * rework_cost

    annual_value = throughput_value + quality_value
    opex = deploy_cost * 0.20
    net = annual_value - opex
    payback_months = (deploy_cost / max(net, 1e-9)) * 12

    m1, m2, m3 = st.columns(3)
    m1.metric("Throughput value / yr", f"${throughput_value:,.0f}",
              f"{units_recovered:.0f} vehicles")
    m2.metric("Quality value / yr", f"${quality_value:,.0f}",
              f"{defects_avoided:.0f} defects")
    m3.metric("Net value / yr", f"${net:,.0f}", f"payback {payback_months:.1f} months")

    st.caption(
        f"Recovery factor {recovery:.0%} — finding a constraint sooner does not "
        f"eliminate it, it shortens it. Opex assumed at 20% of deployment cost."
    )

    st.markdown("---")
    st.subheader("Where the next sensor should go")
    st.markdown(
        "The question is not *whether* to instrument the blind stations — it is "
        "**which ones buy the most**. This ranking comes from the propagation "
        "model and needs **no production data**, so it can be run before "
        "committing to a retrofit."
    )
    rec = recommend_sensors(line, n_recommend=5)
    amb = ambiguity(line, line.observed_indices)
    if len(rec):
        show = rec.rename(columns={
            "rank": "Priority", "station_id": "Station", "zone": "Zone",
            "total_gain": "Value of information",
            "own_ambiguity_before": "Currently confusable",
            "unlocks": "Also helps resolve",
        })
        show["Currently confusable"] = (
            show["Currently confusable"] * 100
        ).round(0).astype(int).astype(str) + "%"
        show["Value of information"] = show["Value of information"].round(2)
        st.dataframe(
            show[["Priority", "Station", "Zone", "Currently confusable",
                  "Value of information", "Also helps resolve"]],
            use_container_width=True, hide_index=True)

        blind_adj = [
            (line.stations[i].station_id, line.stations[i + 1].station_id)
            for i in line.hidden_indices
            if (i + 1) in set(line.hidden_indices)
        ]
        if blind_adj:
            pairs = ", ".join(f"{a}/{b}" for a, b in blind_adj)
            st.info(
                f"**{pairs}** are adjacent blind stations. With no sensor between "
                f"them they produce almost the same signature at every observing "
                f"station, so no amount of data will separate them — the twin "
                f"reports them as a group and abstains. Breaking up an adjacent "
                f"pair is worth more than instrumenting an isolated blind station."
            )
    st.caption(
        "Ranked on separability — the residual left when the closest rival "
        "hypothesis is fitted to a station's response, in units of measurement "
        "noise. Chosen over the more intuitive similarity score because that one "
        "is not monotone: it can imply a new sensor made things worse."
    )

    st.markdown("---")
    st.subheader("Sensor economics")
    retro = n_blind * sensor_cost
    e1, e2, e3 = st.columns(3)
    e1.metric("Instrument every blind station", f"${retro:,.0f}",
              f"{n_blind} stations x ${sensor_cost:,.0f}")
    e2.metric("RippleTwin, year 1", f"${deploy_cost:,.0f}")
    e3.metric("Difference", f"${retro - deploy_cost:,.0f}")
    st.markdown(
        "RippleTwin does **not** claim zero sensors. It claims that the stations "
        "already instrumented carry more information than a conventional twin "
        "extracts from them, and that inference can cover the remainder well "
        "enough to act on — for the ones it cannot, the coverage experiment in "
        "`results/tables/` shows exactly where that breaks down."
    )
    st.caption(
        "Retrofits are also constrained by scheduled maintenance windows, so the "
        "comparison is not only cost but calendar: inference deploys against data "
        "the plant already has."
    )

    st.markdown("---")
    st.subheader("Rollout")
    st.markdown(
        """
| Phase | Scope | What it proves | Main risk |
|---|---|---|---|
| 1 | One line, shadow mode | Alerts match what the floor finds | Precision too low to trust |
| 2 | One line, live advisory | Supervisors act on it | Adoption, alert fatigue |
| 3 | Whole plant | Same twin, different topologies | Per-line configuration effort |
| 4 | Multi-plant | Transfer across equipment vintages | Baselines do not transfer |
        """
    )
    st.caption(
        "Phase 1 runs in shadow mode deliberately: the ledger accumulates "
        "predictions and outcomes with no operational risk, and precision per "
        "station is measured before anyone is asked to act on it."
    )
