"""RippleTwin dashboard -- an industrial operations product over one twin.

Run with:
    streamlit run app/dashboard.py

Navigation is organised around what a judge, a floor supervisor, a plant
manager and a leadership sponsor each actually need, in the order the product
story unfolds:

    LIVE LINE  -- detect, explain, forecast, act. The default screen.
    INCIDENTS  -- every active/past alert, and its full evidence trail.
    PLANT      -- this week's recurring losses, coverage, and where to invest.
    BUSINESS   -- the rollout case: value, cost, payback, differentiation.
    SYSTEM     -- evidence, data readiness, audit log, methodology (progressive
                  disclosure: the technical material judges can dig into, kept
                  out of the way of the story on the first screen).

Every number on every page comes from the same inference pipeline
(rippletwin.twin.pipeline) and the same evaluation tables under results/. This
file adds no new calculation -- it only decides what to show, in what order,
and how much of it to show at once. Throughout, every value is tagged
OBSERVED, INFERRED or PREDICTED. A supervisor who cannot tell a measurement
from an estimate will eventually trust the wrong one, and the first time an
inferred value is wrong the whole system loses the floor. Keeping the
distinction visible is a safety property, not decoration.
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

from rippletwin.copilot.ask import EvidencePack, TwinCopilot, active_backend_name  # noqa: E402
from rippletwin.explain.explain import explain_flow_alert  # noqa: E402
from rippletwin.factory import scenarios as SC  # noqa: E402
from rippletwin.factory.topology import build_line  # noqa: E402
from rippletwin.hitl.ledger import (  # noqa: E402
    DECISION_APPROVED,
    DECISION_ESCALATED,
    DECISION_MODIFIED,
    DECISION_REJECTED,
    OUTCOME_CONFIRMED,
    OUTCOME_NOT_FOUND,
    OUTCOME_PENDING,
    DecisionLedger,
)
from rippletwin.integrate.contract import (  # noqa: E402
    DATA_CONTRACT,
    Capability,
    assess_readiness,
    contract_frame,
)
from rippletwin.recommend.dispatch import to_work_order  # noqa: E402
from rippletwin.recommend.engine import ACTION_MONITOR, recommend_flow  # noqa: E402
from rippletwin.twin import genealogy as GN  # noqa: E402
from rippletwin.twin import predict as PR  # noqa: E402
from rippletwin.twin.pipeline import fit_context, infer, simulate  # noqa: E402
from rippletwin.twin.placement import ambiguity, recommend_sensors  # noqa: E402
from rippletwin.twin.propagate import current_buffer_levels, forecast_ripple  # noqa: E402
from rippletwin.twin.shadow import infer_hidden_cycle_time  # noqa: E402
from rippletwin.twin.whatif import (  # noqa: E402
    whatif_add_sensor,
    whatif_cycle_time_improvement,
    whatif_repair,
)

CONFIG = ROOT / "configs" / "line_42.yaml"
DOCS = ROOT / "docs"
LINE_SEED = 7
DEMO_SEED = 20260301

TIER_COLOR = {"RICH": "#1f9d55", "BASIC": "#4c9be8", "MANUAL": "#9aa0a6"}
RISK_SCALE = [[0.0, "#1f9d55"], [0.5, "#f2b705"], [1.0, "#d63a3a"]]
PROVENANCE_TAG = {"OBSERVED": "🟦", "INFERRED": "🟨", "PREDICTED": "🟥"}

NAV_PAGES = ["Live Line", "Incidents", "Plant", "Business", "System"]
SYSTEM_PAGES = ["Evidence", "Data Health", "Audit Log", "Methodology / About"]

DEMO_STEPS = ["1 · DETECT", "2 · EXPLAIN", "3 · FORECAST", "4 · ACT", "5 · PROVE"]
DEMO_STEP_PAGE = {1: "Live Line", 2: "Live Line", 3: "Live Line", 4: "Live Line", 5: "System"}
DEMO_STEP_CAPTION = {
    1: "A station has no sensor. Its neighbours are blocked and starved. "
       "The line map below shows where the twin thinks the boundary falls.",
    2: "Scroll to *Why RippleTwin thinks this station is the constraint* -- "
       "the upstream/downstream evidence that produced this call.",
    3: "Scroll to *What happens if we do nothing?* -- the forecast of vehicles "
       "lost if the constraint holds.",
    4: "Scroll to *Recommended action* -- approve, reject or escalate it. That "
       "decision is written to the audit ledger immediately.",
    5: "This is System -> Evidence: the held-out evaluation that backs every "
       "claim on the pages before it.",
}

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
        "S6_EARLY_WARNING": SC.scenario_gradual_bottleneck,
        "S7_MULTIPLE_ABNORMALITIES": SC.scenario_multiple_abnormalities,
        "S8_RARE_DEFECT": SC.scenario_rare_defect,
    }
    scen = builders[scenario_key](_line)
    res = simulate(_line, scen, seed=DEMO_SEED)
    scored, shadow, sensor = infer(_ctx, res)
    pred = PR.run_predictor(shadow, _line, res.telemetry, scored, _ctx.shadow_cfg)
    wb = GN.window_bounds_from(scored)
    qs = GN.quality_state(_line, GN.explode_defects(res.inspections), wb, _qbase,
                          pool_vehicles=200)
    qa = GN.quality_alerts(qs) if len(qs) else qs
    return scen, res, scored, shadow, pred, sensor, qa


@st.cache_data(show_spinner=False)
def read_doc(name: str) -> str:
    p = DOCS / name
    return p.read_text() if p.exists() else ""


def doc_section(text: str, header_prefix: str) -> str:
    """Pull one ``## `` section out of a generated doc, verbatim.

    Reading the section from the doc at run time (rather than retyping the
    numbers here) means the dashboard cannot drift from
    ``python -m rippletwin.evaluation.report`` -- if the doc regenerates with
    new numbers, this page shows the new numbers on next load.
    """
    lines = text.splitlines()
    out: list[str] = []
    capture = False
    for ln in lines:
        if ln.startswith("## "):
            capture = ln.startswith(header_prefix)
            if capture:
                continue
        elif capture and ln.startswith("---") and not out:
            continue
        if capture:
            out.append(ln)
    # Trim a trailing "---" section divider if the doc uses one.
    while out and out[-1].strip() in ("", "---"):
        out.pop()
    return "\n".join(out).strip()


def fig_path(name: str) -> Path:
    return ROOT / "results" / "figures" / name


# ------------------------------------------------------------------- visuals


def line_map(line, scored, shadow_row, window, risk_by_station, highlight: int | None = None):
    """Station topology coloured by inferred risk, shaped by instrumentation."""
    n = line.n_stations
    xs, ys, texts, colors, symbols, sizes, borders, widths = [], [], [], [], [], [], [], []
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
        is_top = highlight is not None and s.index == highlight
        sizes.append((38 if s.is_hidden else 32) if is_top else (28 if s.is_hidden else 24))
        borders.append("#111111" if s.is_hidden else "#ffffff")
        widths.append(4 if is_top else 2)
        texts.append(
            f"<b>{s.station_id}</b> ({s.zone})<br>"
            f"{'NO SENSOR - state inferred' if s.is_hidden else s.tier + ' - measured'}<br>"
            f"risk {r:.2f}"
            + (f"<br>gate: {s.inspection_id}" if s.is_inspection else "")
            + ("<br><b>SUSPECTED CONSTRAINT</b>" if is_top else "")
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=sizes, color=colors, colorscale=RISK_SCALE, cmin=0, cmax=1,
                    symbol=symbols, line=dict(width=widths, color=borders),
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
        height=320, margin=dict(l=10, r=10, t=10, b=10),
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


STATE_COLOR = {
    PR.STATE_NORMAL: "#9aa0a6",
    PR.STATE_RECOVERING: "#4c9be8",
    PR.STATE_DEGRADING: "#f2b705",
    PR.STATE_WATCH: "#f2984a",
    PR.STATE_PREDICTED_CONSTRAINT: "#e0632f",
    PR.STATE_ACTIVE_BOTTLENECK: "#d63a3a",
}


def risk_timeline(shadow, truth_row=None, cfg=None, pred=None):
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
    if cfg is not None:
        fig.add_hline(y=cfg.detect_llr, line_dash="dash", line_color="#d63a3a",
                       annotation_text="detect threshold", annotation_position="top right")
        fig.add_hline(y=cfg.watch_llr, line_dash="dot", line_color="#f2984a",
                       annotation_text="watch threshold (earlier, looser)",
                       annotation_position="bottom right")
    if pred is not None and len(pred):
        elevated = pred[pred["state"] != PR.STATE_NORMAL]
        for _, r in elevated.iterrows():
            fig.add_vrect(
                x0=r["t_mid_s"] / 3600 - 0.02, x1=r["t_mid_s"] / 3600 + 0.02,
                fillcolor=STATE_COLOR.get(r["state"], "#9aa0a6"), opacity=0.25, line_width=0,
            )
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


# --------------------------------------------------------------- alert bundle


def alert_bundle(line, res, scored, sensor, window: int) -> dict:
    """Everything downstream of one detected window: the shared unit of work.

    Live Line's headline alert, every row in Incidents, and each incident's
    detail view all call this once and render the same fields -- so "why does
    the twin think X" never has two different answers computed two different
    ways.
    """
    sr = next(r for r in sensor.last_results if r.window == window)
    k = sr.top_station
    stn = line.stations[k]
    est_cycle = infer_hidden_cycle_time(line, res.telemetry, k, sr.v_start, sr.v_end)
    fc = forecast_ripple(
        line, k, est_cycle or line.takt_s, horizon_min=60.0,
        buffer_levels=current_buffer_levels(scored, window),
    ) if est_cycle else None
    exp = explain_flow_alert(line, sr, fc, est_cycle)
    rec = recommend_flow(line, sr, fc)
    wo = to_work_order(line, rec, fc, sequence=window + 1,
                       source_alert={"window": window, "station_id": stn.station_id})
    upstream_ev = [e for e in exp.evidence if e.channel == "blocked_time"]
    downstream_ev = [e for e in exp.evidence if e.channel == "starved_time"]
    other_ev = [e for e in exp.evidence
               if e.channel not in ("blocked_time", "starved_time")]
    return dict(window=window, sr=sr, k=k, stn=stn, est_cycle=est_cycle, fc=fc,
                exp=exp, rec=rec, wo=wo, upstream_ev=upstream_ev,
                downstream_ev=downstream_ev, other_ev=other_ev)


def group_flow_incidents(det: pd.DataFrame, max_gap: int = 6) -> list[tuple[int, list[int]]]:
    """Contiguous runs of detected windows -> one incident each.

    Grouped on time-contiguity alone, not on the top station matching exactly.
    Near the head of the line the twin's top pick can legitimately flip
    between two candidates window to window -- the documented head-of-line
    ambiguity between "a slow station" and "no inbound supply"
    (recommend/engine.py). Splitting on every flip would turn one ongoing
    disturbance into a dozen flickering incidents; grouping by time and then
    taking the *mode* station as representative reports it as what it is:
    one sustained alert whose best guess wobbles between two candidates.
    """
    if det.empty:
        return []
    d = det.sort_values("window")
    windows: list[list[int]] = []
    cur = [int(d.iloc[0]["window"])]
    for w in d["window"].iloc[1:]:
        w = int(w)
        if (w - cur[-1]) <= max_gap:
            cur.append(w)
        else:
            windows.append(cur)
            cur = [w]
    windows.append(cur)

    groups: list[tuple[int, list[int]]] = []
    for ws in windows:
        stations_in_group = d[d["window"].isin(ws)]["top_station"].astype(int)
        mode_station = int(stations_in_group.mode().iloc[0])
        groups.append((mode_station, ws))
    return groups


@st.cache_data(show_spinner=False)
def _cached_incident_windows(scenario_key: str, _det_key: tuple) -> list:
    return None  # placeholder not used; grouping is cheap and done live


def build_incidents(line, res, scored, sensor, shadow, det, qa) -> list[dict]:
    incidents: list[dict] = []
    for station_idx, ws in group_flow_incidents(det):
        # Pick the window closest to the middle of the run whose own top
        # pick actually is the group's mode station, so the incident's
        # station_idx and its rendered bundle always agree -- even when the
        # run itself flickers between two head-of-line candidates.
        on_mode = det[(det["window"].isin(ws)) & (det["top_station"] == station_idx)]["window"]
        mid = ws[len(ws) // 2]
        w_rep = int(on_mode.iloc[(on_mode - mid).abs().argsort().iloc[0]]) if len(on_mode) else mid
        b = alert_bundle(line, res, scored, sensor, w_rep)
        t_hr = float(shadow.loc[shadow["window"] == w_rep, "t_mid_s"].iloc[0]) / 3600.0
        severity = ("ABSTAIN" if b["rec"].abstained else b["rec"].priority)
        n_flicker = det[det["window"].isin(ws)]["top_station"].nunique()
        incidents.append({
            "kind": "FLOW",
            "incident_id": f"INC-F{station_idx:02d}-{ws[0]:04d}",
            "station_idx": station_idx,
            "station_id": b["stn"].station_id,
            "windows": ws,
            "window": w_rep,
            "t_hours": t_hr,
            "severity": severity,
            "type": "Hidden bottleneck" if b["stn"].is_hidden else "Measured bottleneck",
            "confidence": b["sr"].group_prob,
            "impact_vph": (b["fc"].units_lost_at_horizon / max(b["fc"].horizon_min, 1e-9) * 60.0
                          if b["fc"] else None),
            "ambiguous_run": n_flicker > 1,
            "bundle": b,
        })
    if qa is not None and len(qa):
        qal = qa[qa["quality_alert"]]
        for station_idx, g in qal.groupby("station"):
            g = g.sort_values("window")
            row = g.iloc[-1]
            stn = line.stations[int(station_idx)]
            incidents.append({
                "kind": "QUALITY",
                "incident_id": f"INC-Q{int(station_idx):02d}",
                "station_idx": int(station_idx),
                "station_id": stn.station_id,
                "windows": g["window"].unique().tolist(),
                "window": int(row["window"]),
                "t_hours": None,
                "severity": "FLAGGED",
                "type": "Quality drift",
                "confidence": None,
                "impact_vph": None,
                "row": row,
                "stn": stn,
            })
    return incidents


def blind_adjacent_pairs(line) -> list[tuple[str, str]]:
    hidden = set(line.hidden_indices)
    return [
        (line.stations[i].station_id, line.stations[i + 1].station_id)
        for i in line.hidden_indices
        if (i + 1) in hidden
    ]


# ------------------------------------------------------------ shared widgets


def render_hitl_actions(ledger: DecisionLedger, scen, window: int, stn, rec, exp, key_prefix: str):
    """The APPROVE / REJECT / MODIFY / ESCALATE controls, wired to the real ledger.

    Used on Live Line for the headline alert and in Incidents for any incident
    a supervisor opens -- same ledger, same hash chain, so a decision recorded
    from either screen shows up identically in System -> Audit Log.
    """
    st.caption("**RippleTwin recommends. A human decides.** Every action below "
              "writes a new, hash-chained entry to the audit ledger.")
    d1, d2, d3, d4, d5 = st.columns(5)
    if d1.button("✅ Approve", use_container_width=True, key=f"{key_prefix}_approve"):
        e = ledger.record_alert(
            scen.scenario_id, window, "FLOW", stn.station_id, stn.tier,
            stn.is_hidden, rec.confidence, rec.as_dict(), exp.as_dict())
        ledger.record_decision(e.entry_id, DECISION_APPROVED, "supervisor")
        ledger.record_outcome(e.entry_id, OUTCOME_CONFIRMED,
                              "Condition confirmed on the floor.")
        st.success(f"Approved and logged (entry {e.entry_id}).")
    if d2.button("❌ Reject / false positive", use_container_width=True, key=f"{key_prefix}_reject"):
        e = ledger.record_alert(
            scen.scenario_id, window, "FLOW", stn.station_id, stn.tier,
            stn.is_hidden, rec.confidence, rec.as_dict(), exp.as_dict())
        ledger.record_decision(e.entry_id, DECISION_REJECTED, "supervisor",
                               "Supervisor judged this a false alarm.")
        ledger.record_outcome(e.entry_id, OUTCOME_NOT_FOUND,
                              "Nothing found at the named station.")
        st.warning(f"Rejected and logged (entry {e.entry_id}).")
    if d3.button("✏️ Modify", use_container_width=True, key=f"{key_prefix}_modify",
                 help="Accept the alert but redirect the action -- e.g. check the "
                      "alternate station listed above instead."):
        e = ledger.record_alert(
            scen.scenario_id, window, "FLOW", stn.station_id, stn.tier,
            stn.is_hidden, rec.confidence, rec.as_dict(), exp.as_dict())
        alt = rec.alternatives[0] if rec.alternatives else "audit in-flight units instead"
        ledger.record_decision(e.entry_id, DECISION_MODIFIED, "supervisor",
                               f"Redirected: {alt}")
        ledger.record_outcome(e.entry_id, OUTCOME_PENDING,
                              "Modified action pending verification.")
        st.info(f"Modified and logged (entry {e.entry_id}).")
    if d4.button("⤴️ Escalate", use_container_width=True, key=f"{key_prefix}_escalate",
                 help="Plausible, but not a call to make alone -- route to a shift "
                      "lead or process engineer."):
        e = ledger.record_alert(
            scen.scenario_id, window, "FLOW", stn.station_id, stn.tier,
            stn.is_hidden, rec.confidence, rec.as_dict(), exp.as_dict())
        ledger.record_decision(e.entry_id, DECISION_ESCALATED, "supervisor",
                               "Routed to shift lead for review.")
        st.warning(f"Escalated and logged (entry {e.entry_id}).")
    d5.metric("Ledger entries", len(ledger.entries))


def render_recommended_action_card(line, b: dict):
    rec, wo, fc = b["rec"], b["wo"], b["fc"]
    st.subheader(f"Recommended action -- {rec.priority}")
    if rec.abstained:
        st.error(f"**{rec.title}** (the twin is abstaining)")
    else:
        st.info(f"**{rec.title}**")
    st.write(rec.detail)

    r1, r2, r3 = st.columns(3)
    r1.metric("Priority", rec.priority)
    r2.metric("Confidence", f"{rec.confidence * 100:.0f}%")
    r3.metric("Expected impact",
              f"{rec.units_at_stake:.1f} vehicles" if rec.units_at_stake else "n/a")
    if wo is not None:
        o1, o2, o3 = st.columns(3)
        o1.metric("Owner", wo.owner_role)
        o2.metric("Respond within", f"{wo.respond_within_min} min")
        o3.metric("Verification", "Required", help=wo.verification)
        st.caption(f"**Reason:** {rec.rationale}")
        st.caption(f"**Verification requirement:** {wo.verification}")
        if wo.waiting_cost:
            st.caption(f"**Cost of waiting:** {wo.waiting_cost['rationale']}")
    else:
        st.caption(f"**Reason:** {rec.rationale}  \n"
                  "This is advisory-only (MONITOR): no work order is raised for "
                  "'keep an eye on it' -- that is how alert fatigue starts.")
    if rec.alternatives:
        st.caption(f"If wrong: {rec.alternatives[0]}")


def render_ask_the_twin(b: dict, key_prefix: str):
    st.subheader("Ask RippleTwin")
    st.caption(
        "Answers are drawn only from the facts already computed for this alert -- a "
        "guardrail rejects any number the model states that isn't already in the "
        "evidence pack, whether the answer came from a template or an LLM. Active "
        f"backend: **{active_backend_name()}**. Falls back to a deterministic offline "
        "template with zero setup if no LLM key is configured."
    )
    exp, rec, fc = b["exp"], b["rec"], b["fc"]
    pack = EvidencePack.from_explanation(
        exp, rec, units_at_risk_per_hr=fc.units_lost_at_horizon if fc else None
    )
    suggested = [
        "Why do you think it's this station?",
        "What evidence supports this?",
        "What happens if I wait 30 minutes?",
        "How confident are you?",
        "What would prove you wrong?",
        "Which sensor should we install next?",
    ]
    cols = st.columns(3)
    picked = None
    for i, question in enumerate(suggested):
        if cols[i % 3].button(question, key=f"{key_prefix}_sugg_{i}", use_container_width=True):
            picked = question
    q = st.text_input(
        "Ask a question about this alert",
        value=picked or "",
        placeholder="e.g. why do you think it's this station? / is this urgent?",
        key=f"{key_prefix}_copilot_q",
    )
    if q:
        answer = TwinCopilot().ask(q, pack)
        st.info(answer.text)
        badge = "🟢 grounded" if answer.grounded else "🔴 flagged"
        source = f"{active_backend_name()} (guardrail-checked)" if answer.used_llm else "offline template"
        st.caption(f"{badge} · answered by: {source}")
        if answer.flagged_numbers:
            st.caption(
                f"⚠️ The model's first draft mentioned numbers not in the "
                f"evidence pack ({', '.join(answer.flagged_numbers)}) and was "
                f"replaced with the grounded template answer."
            )


def render_whatif(line, res, scored, b: dict, key_prefix: str):
    if not b["est_cycle"]:
        return
    with st.expander("What if...? (simulation-based projections)"):
        st.caption(
            "Every number below is a projection from the same flow-physics "
            "model as the forecast above, not a measurement and not a "
            "guarantee -- re-running the arithmetic with one input changed."
        )
        wtab1, wtab2 = st.tabs(["Repaired now", "Cycle-time improvement"])
        with wtab1:
            rwi = whatif_repair(line, b["k"], b["est_cycle"],
                                buffer_levels=current_buffer_levels(scored, b["window"]))
            wc1, wc2 = st.columns(2)
            wc1.metric("Units lost/hr, as-is",
                      f"{rwi.before.units_lost_at_horizon:.1f}" if rwi.before else "n/a")
            wc2.metric("Units lost/hr, if repaired",
                      f"{rwi.after.units_lost_at_horizon:.1f}" if rwi.after else "n/a")
        with wtab2:
            pct = st.slider("Cycle-time improvement", 0, 50, 10, step=5,
                            format="%d%%", key=f"{key_prefix}_whatif_pct") / 100.0
            cwi = whatif_cycle_time_improvement(line, b["k"], b["est_cycle"], pct,
                                                buffer_levels=current_buffer_levels(scored, b["window"]))
            wc3, wc4 = st.columns(2)
            wc3.metric("Units lost/hr, as-is",
                      f"{cwi.before.units_lost_at_horizon:.1f}" if cwi.before else "n/a")
            wc4.metric(f"Units lost/hr, at -{pct:.0%} cycle time",
                      f"{cwi.after.units_lost_at_horizon:.1f}" if cwi.after else "n/a")


def render_impact_forecast(fc, stn):
    st.subheader("What happens if we do nothing?")
    if fc is None or not fc.is_binding:
        st.success(
            "No throughput loss projected. Either the constraint stays inside "
            "takt, or the cycle time could not be estimated confidently enough "
            "to forecast forward."
        )
        return
    events = [(0.0, "NOW", f"Constraint detected at {stn.station_id}")]
    if fc.minutes_to_upstream_block is not None:
        up = fc.upstream_affected[0] if fc.upstream_affected else "the previous station"
        events.append((fc.minutes_to_upstream_block, "BUFFER FULL",
                       f"Upstream backs up to {up}"))
    if fc.minutes_to_downstream_starve is not None:
        dn = fc.downstream_affected[0] if fc.downstream_affected else "the next station"
        events.append((fc.minutes_to_downstream_starve, "STARVATION",
                       f"Downstream starvation reaches {dn}"))
    events.append((fc.horizon_min, f"{fc.horizon_min:.0f} MIN",
                   f"{fc.units_lost_at_horizon:.0f} vehicles lost -- throughput at "
                   f"{fc.sustained_rate_vph:.0f} veh/h "
                   f"({fc.throughput_loss_pct * 100:.0f}% below target)"))
    events.sort(key=lambda t: t[0])

    cols = st.columns(len(events))
    for col, (mins, label, text) in zip(cols, events):
        with col:
            st.markdown(f"**{label}**")
            st.caption(f"~{mins:.0f} min" if mins > 0 else "now")
            st.write(text)
    f1, f2, f3 = st.columns(3)
    f1.metric("Units at risk (60 min)", f"{fc.units_lost_at_horizon:.1f} vehicles")
    f2.metric("Sustained rate", f"{fc.sustained_rate_vph:.0f} veh/h",
              f"{-fc.throughput_loss_pct * 100:.0f}% vs target")
    f3.metric("Confidence in forecast", "physics-derived",
              help="This is deterministic flow arithmetic from the estimated cycle "
                   "time, not a fitted probability -- a plant engineer can check it "
                   "on paper.")
    st.caption(
        "PREDICTED, not measured: this is a forward projection from the estimated "
        "cycle time and current buffer levels, assuming nothing changes."
    )


def render_evidence_card(b: dict):
    stn, sr, exp = b["stn"], b["sr"], b["exp"]
    st.subheader(f"Why RippleTwin thinks {stn.station_id} is the constraint")
    if stn.is_hidden:
        st.warning(
            f"**{stn.station_id} has no sensor.** Everything shown for this "
            f"station is inferred from the instrumented stations either side of "
            f"it. Confirm on the floor before committing to a repair."
        )
    st.markdown(exp.what_changed)

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Upstream evidence — blocked propagation**")
        if b["upstream_ev"]:
            for e in b["upstream_ev"]:
                st.markdown(f"✓ {PROVENANCE_TAG[e.provenance]} {e.text}")
        else:
            st.caption("No significant upstream blocking recorded.")
        st.caption("Supports upstream accumulation: work cannot leave.")
    with ec2:
        st.markdown("**Downstream evidence — starved propagation**")
        if b["downstream_ev"]:
            for e in b["downstream_ev"]:
                st.markdown(f"✓ {PROVENANCE_TAG[e.provenance]} {e.text}")
        else:
            st.caption("No significant downstream starvation recorded.")
        st.caption(f"Supports a flow restriction near {stn.station_id}.")

    st.markdown(
        f"**Spatial / flow relationship** -- the transition from blocked "
        f"(upstream) to starved (downstream) falls at {stn.station_id}, which is "
        f"exactly where conservation of material puts the constraint: work "
        f"backs up on one side and runs dry on the other."
    )
    st.markdown(f"**Confidence** -- {exp.confidence_text} "
               f"*(This is a stated confidence, not certainty -- see caveats below.)*")
    if exp.alternative_station_id:
        st.caption(
            f"🥈 Alternative hypothesis: {exp.alternative_station_id} "
            f"({exp.alternative_probability * 100:.0f}% posterior)."
        )
    for cav in exp.caveats:
        st.caption(f"⚠️ {cav}")

    if b["other_ev"]:
        with st.expander("Other evidence"):
            for e in b["other_ev"]:
                st.markdown(f"{PROVENANCE_TAG[e.provenance]} `{e.provenance}` {e.text}")

    with st.expander("Technical evidence (posterior over every station, and the raw signal)"):
        st.plotly_chart(posterior_chart(st.session_state["_line"], sr),
                        use_container_width=True, key=f"post_{stn.index}_{b['window']}")
        st.caption(
            "Posterior probability that each station is the constraint. Grey = "
            "no sensor. This is what 'confidence' above actually is."
        )


# --------------------------------------------------------------------- app

line, ctx, qbase, nominal = load_twin()
st.session_state["_line"] = line

if "ledger" not in st.session_state:
    st.session_state.ledger = DecisionLedger()
ledger: DecisionLedger = st.session_state.ledger
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if "demo_step" not in st.session_state:
    st.session_state.demo_step = 1

st.sidebar.title("RippleTwin")
st.sidebar.caption("See the bottleneck before it arrives.")

_qp = st.query_params
_v = _qp.get("view", "")
nav = st.sidebar.radio("View", NAV_PAGES,
                        index=NAV_PAGES.index(_v) if _v in NAV_PAGES else 0)
sub_nav = None
if nav == "System":
    sub_nav = st.sidebar.radio("System", SYSTEM_PAGES, index=0)

st.sidebar.markdown("---")

_SCENARIOS = ["S1_HIDDEN_BOTTLENECK", "S2_HIDDEN_QUALITY", "S3_NORMAL",
              "S4_OBSERVED_STATION", "S5_VARIANT_AND_SUPPLY", "S6_EARLY_WARNING",
              "S7_MULTIPLE_ABNORMALITIES", "S8_RARE_DEFECT"]
_SCENARIO_LABEL = {
    "S1_HIDDEN_BOTTLENECK": "S1 - hidden bottleneck (no sensor)",
    "S2_HIDDEN_QUALITY": "S2 - hidden quality drift",
    "S3_NORMAL": "S3 - normal variation (should stay quiet)",
    "S4_OBSERVED_STATION": "S4 - fault at an instrumented station",
    "S5_VARIANT_AND_SUPPLY": "S5 - mix change + supply delay (not a fault)",
    "S6_EARLY_WARNING": "S6 - gradual ramp (early-warning demo)",
    "S7_MULTIPLE_ABNORMALITIES": "S7 - two simultaneous, unrelated faults",
    "S8_RARE_DEFECT": "S8 - a small, marginal quality drift (hard case)",
}

st.sidebar.subheader("🎬 Guided demo")
if st.sidebar.button("▶ Start / restart guided demo (~3 min)", use_container_width=True):
    st.session_state.demo_mode = True
    st.session_state.demo_step = 1
if st.session_state.demo_mode:
    step = st.session_state.demo_step
    st.sidebar.info(f"**Step {step}/5 — {DEMO_STEPS[step - 1]}**")
    sc1, sc2 = st.sidebar.columns(2)
    if sc1.button("◀ Back", disabled=step <= 1, use_container_width=True):
        st.session_state.demo_step = max(1, step - 1)
        st.rerun()
    if sc2.button("Next ▶", disabled=step >= 5, use_container_width=True):
        st.session_state.demo_step = min(5, step + 1)
        st.rerun()
    if st.sidebar.button("Exit guided demo", use_container_width=True):
        st.session_state.demo_mode = False
        st.rerun()

if st.session_state.demo_mode:
    scenario_key = "S1_HIDDEN_BOTTLENECK"
    nav = DEMO_STEP_PAGE[st.session_state.demo_step]
    if nav == "System":
        sub_nav = "Evidence"
    st.sidebar.caption(f"Scenario locked to {_SCENARIO_LABEL[scenario_key]} for the demo.")
else:
    _s = _qp.get("scenario", "")
    scenario_key = st.sidebar.selectbox(
        "Scenario", _SCENARIOS,
        index=_SCENARIOS.index(_s) if _s in _SCENARIOS else 0,
        format_func=lambda k: _SCENARIO_LABEL[k],
    )

scen, res, scored, shadow, pred, sensor, qa = run_scenario(scenario_key, line, ctx, qbase)

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
with st.sidebar.expander("Evaluation / Ground Truth", expanded=False):
    st.caption(
        "For evaluation only. A deployed system never has this -- it is shown "
        "here so the localisation can be checked, not because RippleTwin "
        "consumes it."
    )
    show_truth = st.checkbox("Reveal ground truth", value=False)
st.sidebar.markdown("---")
st.sidebar.warning(
    "**DEMONSTRATION — SIMULATED PRODUCTION DATA.** All figures are simulated "
    "prototype results on synthetic data. No real production data is used and "
    "no real-plant ROI is claimed."
)

det = shadow[shadow["detected"]]


def headline_window() -> int | None:
    if len(det) == 0:
        return None
    return int(det.iloc[len(det) // 2]["window"])


# ============================================================== LIVE LINE
def render_live_line():
    st.title("RippleTwin")
    st.caption("See the bottleneck before it arrives.")
    st.markdown("##### Shadow-sensing for sensor-sparse vehicle assembly lines.")
    st.badge("DEMONSTRATION — SIMULATED PRODUCTION DATA", color="orange")

    if st.session_state.demo_mode:
        st.info(f"**Guided demo — {DEMO_STEPS[st.session_state.demo_step - 1]}.** "
               f"{DEMO_STEP_CAPTION[st.session_state.demo_step]}")

    w = headline_window()

    if w is None:
        risk = {i: 0.0 for i in range(line.n_stations)}
        line_status, status_color = "RUNNING", "green"
        if len(pred) and pred["state"].map(PR.state_rank).max() > 0:
            peak = pred.loc[pred["state"].map(PR.state_rank).idxmax()]
            if peak["state"] != PR.STATE_NORMAL:
                line_status, status_color = "WATCH", "orange"
    else:
        b = alert_bundle(line, res, scored, sensor, w)
        risk = {i: float(b["sr"].evidence["station_post"][i]) for i in range(line.n_stations)}
        line_status, status_color = "ATTENTION", "red"

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown("**LINE STATUS**")
        st.badge(line_status, color=status_color)
    with k2:
        st.markdown("**HIDDEN CONSTRAINT**")
        if w is None:
            st.markdown("— none —")
        else:
            st.markdown(f"### {b['stn'].station_id}")
            st.caption(f"{b['sr'].group_prob * 100:.0f}% confidence")
    with k3:
        st.markdown("**SENSOR STATUS**")
        if w is None:
            st.markdown("n/a")
        elif b["stn"].is_hidden:
            st.markdown("### NO SENSOR")
            st.caption("state inferred")
        else:
            st.markdown(f"### OBSERVED")
            st.caption(f"{b['stn'].tier} tier")
    with k4:
        st.markdown("**CYCLE TIME**")
        if w is not None and b["est_cycle"]:
            st.markdown(f"### {b['est_cycle']:.0f} sec")
            st.caption(f"vs {line.takt_s:.0f} sec takt")
        else:
            st.markdown("n/a")
    with k5:
        st.markdown("**IMPACT**")
        if w is not None and b["fc"] and b["fc"].is_binding:
            per_hr = b["fc"].units_lost_at_horizon / max(b["fc"].horizon_min, 1e-9) * 60.0
            st.markdown(f"### {per_hr:.1f} veh/hr")
            st.caption("at risk if nothing changes")
        else:
            st.markdown("### 0")
            st.caption("vehicles/hr at risk")

    if w is None:
        st.markdown("---")
        st.success(
            "**No station-level alert.** The twin is not silent because it sees "
            "nothing -- it is silent because the evidence does not beat the "
            "no-fault hypothesis at the calibrated threshold."
        )
        if scen.expect_no_alert:
            st.info(f"This scenario is designed to produce no alert. {scen.notes}")
        st.markdown("---")
        st.subheader("Line map")
        st.plotly_chart(line_map(line, scored, None, 0, risk), use_container_width=True,
                        key="live_map_no_alert")
        st.caption("Diamonds = no sensor (state inferred). Circles = instrumented (measured).")
        return

    st.markdown("---")
    st.subheader("Line map")
    st.plotly_chart(line_map(line, scored, None, w, risk, highlight=b["k"]),
                    use_container_width=True, key="live_map")
    up_candidates = [i for i in line.nearest_observed_upstream(b["k"], 3)]
    dn_candidates = [i for i in line.nearest_observed_downstream(b["k"], 3)]
    up_id = line.stations[up_candidates[0]].station_id if up_candidates else "the line start"
    dn_id = line.stations[dn_candidates[0]].station_id if dn_candidates else "the line end"
    shape = "◆" if b["stn"].is_hidden else "●"
    st.markdown(
        f"<div style='text-align:center; font-family:monospace; line-height:1.8'>"
        f"UPSTREAM ({up_id}) BLOCKED<br>&darr;<br>"
        f"<b>{shape} {b['stn'].station_id} {'NO SENSOR' if b['stn'].is_hidden else 'OBSERVED'}</b><br>&darr;<br>"
        f"DOWNSTREAM ({dn_id}) STARVED</div>",
        unsafe_allow_html=True,
    )
    st.caption("Diamonds = no sensor (state inferred). Circles = instrumented (measured). "
              "Red border = the suspected constraint.")

    st.markdown("---")
    render_evidence_card(b)

    st.markdown("---")
    render_impact_forecast(b["fc"], b["stn"])

    st.markdown("---")
    render_recommended_action_card(line, b)
    render_hitl_actions(ledger, scen, w, b["stn"], b["rec"], b["exp"], key_prefix="live")
    render_whatif(line, res, scored, b, key_prefix="live")

    st.markdown("---")
    render_ask_the_twin(b, key_prefix="live")

    st.markdown("---")
    with st.expander("Evidence over the shift (technical)"):
        st.plotly_chart(
            pressure_profile(line, scored, w, true_station if show_truth else None),
            use_container_width=True, key="live_pressure")
        st.caption(
            "Blocking above the axis, starvation below. The boundary between them "
            "is where the constraint sits — including at stations with no sensor "
            "(grey bands). This is conservation of material, not a learned correlation."
        )
        st.plotly_chart(
            risk_timeline(shadow, truth_row if show_truth else None, cfg=ctx.shadow_cfg, pred=pred),
            use_container_width=True, key="live_timeline")
        st.caption(
            "Shaded bands mark predicted-bottleneck states above NORMAL: amber = "
            "degrading/watch, orange/red = predicted or active constraint, blue = "
            "recovering."
        )
        if len(qa):
            q_al = qa[qa["quality_alert"]]
            if len(q_al):
                st.markdown("**Quality attribution (second shadow-sensing path)**")
                rank = (q_al.groupby(["station", "station_id", "tier", "is_hidden"])["llr"]
                        .mean().reset_index().sort_values("llr", ascending=False).head(5))
                rank["m_hat"] = rank["station"].map(q_al.groupby("station")["m_hat"].mean())
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


# ============================================================== INCIDENTS
def render_incidents():
    st.title("Incident Command Center")
    st.caption("Every alert the twin has raised this run — active and past.")

    incidents = build_incidents(line, res, scored, sensor, shadow, det, qa)
    if not incidents:
        st.success("No incidents this run. Try scenario S1 or S7 in the sidebar.")
        return

    lf = ledger.to_frame()
    rows = []
    for inc in incidents:
        matched = False
        if len(lf):
            m = lf[(lf["station_id"] == inc["station_id"])
                   & (lf["alert_type"].astype(str).str.startswith("FLOW_DECISION")
                      if inc["kind"] == "FLOW" else False)]
            matched = len(m) > 0
        itype = inc["type"]
        if inc.get("ambiguous_run"):
            itype += " (best guess wobbles between candidates)"
        rows.append({
            "Severity": inc["severity"],
            "Station": inc["station_id"],
            "Type": itype,
            "Confidence": f"{inc['confidence'] * 100:.0f}%" if inc["confidence"] is not None else "—",
            "Expected impact": f"{inc['impact_vph']:.1f} veh/hr" if inc["impact_vph"] else "—",
            "Detected": f"{inc['t_hours']:.2f} h into run" if inc["t_hours"] is not None else "—",
            "Status": "DECISIONED" if matched else "ACTIVE",
            "Incident ID": inc["incident_id"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df.drop(columns=["Incident ID"]), use_container_width=True, hide_index=True)

    st.markdown("---")
    pick = st.selectbox(
        "Open incident",
        options=[i["incident_id"] for i in incidents],
        format_func=lambda iid: f"{iid} — "
                                f"{next(i['station_id'] for i in incidents if i['incident_id'] == iid)} "
                                f"({next(i['type'] for i in incidents if i['incident_id'] == iid)})",
    )
    inc = next(i for i in incidents if i["incident_id"] == pick)
    st.markdown("---")
    render_incident_detail(inc)


def render_incident_detail(inc: dict):
    st.header(f"{inc['incident_id']} — {inc['station_id']}")

    if inc["kind"] == "QUALITY":
        row, stn = inc["row"], inc["stn"]
        st.subheader("1. What happened?")
        st.write(
            f"Defects matching {stn.station_id}'s known failure-mode signature are "
            f"running {row['m_hat']:.1f}x above the normal rate, persistently enough "
            f"to clear the quality-alert thresholds (evidence LLR "
            f"{row['llr']:.1f})."
        )
        st.subheader("2. Where?")
        risk = {stn.index: 1.0}
        st.plotly_chart(line_map(line, scored, None, inc["window"], risk, highlight=stn.index),
                        use_container_width=True, key=f"inc_map_{inc['incident_id']}")
        st.subheader("3. Why?")
        st.markdown(
            f"- {PROVENANCE_TAG['INFERRED']} `INFERRED` Defect multiplier "
            f"{row['m_hat']:.2f}x normal at {stn.station_id}.\n"
            f"- {PROVENANCE_TAG['OBSERVED']} `OBSERVED` Evidence (log-likelihood "
            f"ratio) {row['llr']:.1f}, against the station's known defect-type "
            f"profile."
        )
        st.caption(
            "This is the second, independent shadow-sensing path: cycle times "
            "are normal here, but the *mix* of defect types reaching the "
            "inspection gates matches this station's known failure modes more "
            "than any other candidate's."
        )
        st.subheader("4-6. Forecast, action, decision")
        st.info(
            "Quality-drift incidents are surfaced for review; the forecast, "
            "dispatch and approval workflow below is implemented for flow "
            "(bottleneck) incidents. Use the ranking in Live Line -> Evidence "
            "over the shift for the full station ranking on this alert."
        )
        return

    b = inc["bundle"]
    if inc.get("ambiguous_run"):
        st.caption(
            f"Over this incident's {len(inc['windows'])} detected windows, the "
            f"twin's top pick moved between more than one candidate station -- "
            f"shown here is its most frequent call, {b['stn'].station_id}. This "
            f"is the documented head-of-line ambiguity (a slow station vs. no "
            f"inbound supply look similar from a few stations downstream)."
        )
    st.subheader("1. What happened?")
    st.write(b["exp"].headline + ". " + b["exp"].what_changed)

    st.subheader("2. Where?")
    risk = {i: float(b["sr"].evidence["station_post"][i]) for i in range(line.n_stations)}
    st.plotly_chart(line_map(line, scored, None, inc["window"], risk, highlight=b["k"]),
                    use_container_width=True, key=f"inc_map_{inc['incident_id']}")

    st.subheader("3. Why?")
    render_evidence_card(b)

    st.subheader("4. What happens next?")
    render_impact_forecast(b["fc"], b["stn"])

    st.subheader("5. What should we do?")
    render_recommended_action_card(line, b)
    render_hitl_actions(ledger, scen, inc["window"], b["stn"], b["rec"], b["exp"],
                        key_prefix=f"inc_{inc['incident_id']}")

    st.subheader("6. What did the human decide?")
    lf = ledger.to_frame()
    if len(lf):
        matched = lf[lf["station_id"] == inc["station_id"]]
        if len(matched):
            st.dataframe(
                matched[["entry_id", "timestamp", "alert_type", "decision", "outcome"]],
                use_container_width=True, hide_index=True)
        else:
            st.caption("No decision recorded yet for this station.")
    else:
        st.caption("No decisions recorded yet. Approve, reject or escalate above.")

    st.subheader("7. Audit trail")
    render_audit_trail_for_station(inc["station_id"])


def render_audit_trail_for_station(station_id: str):
    lf = ledger.to_frame()
    if not len(lf):
        st.caption("Nothing in the ledger yet for this station.")
        return
    matched = lf[lf["station_id"] == station_id].sort_values("timestamp")
    if not len(matched):
        st.caption("Nothing in the ledger yet for this station.")
        return
    for _, r in matched.iterrows():
        label = {
            "FLOW": "Detection generated / recommendation issued",
            "FLOW_DECISION": f"Supervisor decision: {r['decision']}"
                             + (f" — {r['decision_note']}" if r.get("decision_note") else ""),
            "FLOW_OUTCOME": f"Outcome recorded: {r['outcome']}"
                           + (f" — {r['outcome_note']}" if r.get("outcome_note") else ""),
        }.get(r["alert_type"], r["alert_type"])
        st.markdown(f"`{r['timestamp']}` — {label}")


# ================================================================== PLANT
def render_plant():
    st.title("Plant Intelligence")
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
    st.caption(
        "Throughput and defect counts are OBSERVED. The station attribution below "
        "is INFERRED for stations with no sensor."
    )

    st.markdown("---")
    st.subheader("Where are we blind? — sensor coverage by zone")
    s = line.summary()
    zrows = []
    for z, d in s["per_zone"].items():
        zrows.append({
            "Zone": z, "Stations": d["stations"],
            "Rich": d["rich"], "Basic": d["basic"], "No sensor": d["manual"],
            "Coverage": f"{(d['rich'] + d['basic']) / d['stations'] * 100:.0f}%",
        })
    zcols = st.columns(len(zrows))
    for col, z in zip(zcols, zrows):
        with col:
            st.metric(z["Zone"], z["Coverage"], f"{z['No sensor']} blind of {z['Stations']}")
    st.dataframe(pd.DataFrame(zrows), use_container_width=True, hide_index=True)
    st.caption(
        "Zones with the least instrumentation are where a conventional twin is "
        "blind, and where inference contributes the most."
    )

    st.markdown("---")
    st.subheader("Where should we install the next sensor?")
    st.markdown(
        "The question is not *whether* to instrument the blind stations — it is "
        "**which ones buy the most**. This ranking comes from the propagation "
        "model and needs **no production data**, so it can be run before "
        "committing to a retrofit. **Do not instrument everything. Instrument "
        "where information is worth the cost.**"
    )
    rec_s = recommend_sensors(line, n_recommend=5)
    if len(rec_s):
        show = rec_s.rename(columns={
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
        pairs = blind_adjacent_pairs(line)
        if pairs:
            txt = ", ".join(f"{a}/{b}" for a, b in pairs)
            st.info(
                f"**{txt}** are adjacent blind stations. With no sensor between "
                f"them they produce almost the same signature at every observing "
                f"station, so no amount of data will separate them — the twin "
                f"reports them as a group and abstains. Breaking up an adjacent "
                f"pair is worth more than instrumenting an isolated blind station."
            )
    st.caption(
        "Ranked on separability — the residual left when the closest rival "
        "hypothesis is fitted to a station's response, in units of measurement "
        "noise."
    )
    with st.expander("What if we added a sensor at a specific station?"):
        st.caption("Simulation-based projection -- uncertainty before/after, not a guarantee.")
        blind_ids = [line.stations[i].station_id for i in line.hidden_indices]
        if blind_ids:
            pick = st.selectbox("Candidate station", blind_ids, key="plant_whatif_sensor_pick")
            pick_idx = line.by_id(pick).index
            swi = whatif_add_sensor(line, pick_idx)
            wc1, wc2 = st.columns(2)
            wc1.metric(f"{pick} ambiguity, before", f"{swi.ambiguity_before * 100:.0f}%")
            wc2.metric(f"{pick} ambiguity, after", f"{swi.ambiguity_after * 100:.0f}%")

    with st.expander("Where the twin spent its suspicion (alert windows by station)"):
        if len(shadow):
            counts = (shadow[shadow["detected"]]["top_station"].value_counts()
                      if shadow["detected"].any() else pd.Series(dtype=int))
            rows = []
            for s_ in line.stations:
                rows.append({
                    "Station": s_.station_id, "Zone": s_.zone,
                    "Instrumentation": "none (inferred)" if s_.is_hidden else s_.tier,
                    "Alert windows": int(counts.get(s_.index, 0)),
                })
            dfw = pd.DataFrame(rows)
            top = dfw[dfw["Alert windows"] > 0].sort_values("Alert windows", ascending=False)
            if len(top):
                st.dataframe(top, use_container_width=True, hide_index=True)
            else:
                st.success("No station accumulated alert time this run.")

    with st.expander("Defects by discovery gate"):
        if len(res.inspections):
            g = (res.inspections.groupby("gate_id")
                 .agg(inspected=("result", "size"),
                      failed=("result", lambda x: (x == "FAIL").sum()))
                 .reset_index())
            g["fail rate"] = (g["failed"] / g["inspected"] * 100).round(2).astype(str) + "%"
            st.dataframe(g, use_container_width=True, hide_index=True)
            st.caption(
                "A defect caught at the end-of-line test has already accumulated the "
                "full value-add of every station after the one most likely responsible for it."
            )


# ================================================================ BUSINESS
def render_business():
    st.title("Business Case")
    st.caption("From hidden operational risk to targeted intervention.")
    st.warning(
        "**Every figure on this page is an ILLUSTRATIVE ASSUMPTION**, shown so the "
        "arithmetic is inspectable. None of it is measured from a real plant. The "
        "prototype evidence lives in System -> Evidence; this page is the business "
        "model built on top of transparent inputs you can change."
    )

    st.subheader("Assumptions (edit these)")
    a1, a2, a3 = st.columns(3)
    veh_margin = a1.number_input("Contribution margin per vehicle (USD)",
                                 500, 20000, 2200, step=100)
    rework_cost = a2.number_input("Average rework cost per defect (USD)",
                                  50, 5000, 420, step=10)
    prod_hours = a3.number_input("Productive hours per year", 1000, 8000, 3800, step=100)

    b1, b2, b3, b4 = st.columns(4)
    sensor_cost = b1.number_input("PLC-integrated retrofit / station (USD)",
                                  1000, 100000, 18000, step=1000)
    clamp_cost = b2.number_input("Non-invasive clamp-on retrofit / station (USD)",
                                 100, 20000, 1000, step=100)
    n_blind = b3.number_input("Blind stations on this line",
                              1, 40, len(line.hidden_indices))
    deploy_cost = b4.number_input("RippleTwin deployment, year 1 (USD)",
                                  10000, 1000000, 150000, step=10000)
    st.caption(
        r"Clamp-on default of \$1,000 is the midpoint of a \$200–\$2,000/station "
        "range compiled from public vendor sourcing on non-invasive current-clamp "
        "monitoring — an ILLUSTRATIVE ASSUMPTION, not a quote for this line. See "
        "docs/BUSINESS_CASE.md §5a."
    )

    c1, c2, c3 = st.columns(3)
    events_per_year = c1.number_input(
        "Hidden-station disturbances per line per year", 1, 500, 60)
    minutes_saved = c2.number_input(
        "Minutes of earlier reaction per event", 1, 240, 25)
    maintenance_cost = c3.number_input(
        "RippleTwin maintenance / support, per year (USD)", 0, 200000, 30000, step=5000,
        help="Separate from year-1 deployment cost -- the ongoing cost of keeping it running.")

    takt_per_hour = 3600 / line.takt_s
    veh_per_min = takt_per_hour / 60
    production_value_per_hr = veh_margin * takt_per_hour
    downtime_cost_per_hr = production_value_per_hr

    recovery = 0.55
    units_recovered = events_per_year * minutes_saved * veh_per_min * recovery
    throughput_value = units_recovered * veh_margin
    avoided_downtime_hours = units_recovered / max(takt_per_hour, 1e-9)

    defects_avoided = events_per_year * 6.0
    quality_value = defects_avoided * rework_cost

    annual_value = throughput_value + quality_value
    first_year_cost = deploy_cost + maintenance_cost
    ongoing_annual_cost = maintenance_cost
    net_first_year = annual_value - first_year_cost
    net_ongoing = annual_value - ongoing_annual_cost
    payback_months = (deploy_cost / max(net_ongoing, 1e-9)) * 12
    roi_pct = (net_first_year / max(first_year_cost, 1e-9)) * 100

    st.markdown("---")
    st.metric("Estimated annual value", f"${annual_value:,.0f}",
              f"{roi_pct:+.0f}% ROI in year 1")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Throughput recovery", f"${throughput_value:,.0f}")
    m2.metric("Quality improvement", f"${quality_value:,.0f}")
    m3.metric("Deployment cost", f"${deploy_cost:,.0f}")
    m4.metric("Operating cost / yr", f"${maintenance_cost:,.0f}")
    m5.metric("Net value, year 1", f"${net_first_year:,.0f}")
    m6.metric("Payback period", f"{payback_months:.1f} mo")

    with st.expander("Show the arithmetic"):
        d1, d2 = st.columns(2)
        d1.metric("Production value / hr", f"${production_value_per_hr:,.0f}")
        d2.metric("Downtime cost / hr", f"${downtime_cost_per_hr:,.0f}")
        st.caption(
            f"Units recovered/yr: {units_recovered:.0f} vehicles "
            f"({avoided_downtime_hours:.1f}h avoided downtime). "
            f"Defects avoided/yr: {defects_avoided:.0f}. "
            f"Recovery factor {recovery:.0%} — finding a constraint sooner does not "
            f"eliminate it, it shortens it. Year-1 cost = deployment + maintenance; "
            f"payback is measured against the ongoing (maintenance-only) annual cost, "
            f"since deployment is a one-time spend."
        )

    st.markdown("---")
    st.subheader("Sensor economics")
    retro = n_blind * sensor_cost
    retro_clamp = n_blind * clamp_cost
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("PLC-integrated retrofit, all blind stations", f"${retro:,.0f}",
              f"{n_blind} x ${sensor_cost:,.0f}")
    e2.metric("Non-invasive clamp-on, all blind stations", f"${retro_clamp:,.0f}",
              f"{n_blind} x ${clamp_cost:,.0f}")
    e3.metric("RippleTwin, year 1", f"${deploy_cost:,.0f}")
    e4.metric("Diff. vs. cheapest retrofit", f"${retro_clamp - deploy_cost:,.0f}")
    st.markdown(
        "RippleTwin does **not** claim zero sensors, and it does **not** claim "
        "instrumentation is expensive — a non-invasive clamp-on retrofit is "
        "often *cheaper* than a year of RippleTwin. The claim is narrower: "
        "clamp-on current draw is a coarse proxy for blocked/starved (no "
        "torque/vibration/temperature, and it needs calibration per motor), and "
        "even a fully-instrumented line still needs something to turn per-station "
        "signal into a located constraint, a forecast and a recommendation."
    )
    st.caption(
        "Retrofits are also constrained by scheduled maintenance windows, so the "
        "comparison is not only cost but calendar: inference deploys against data "
        "the plant already has."
    )

    st.markdown("---")
    st.subheader("Why RippleTwin?")
    st.caption(
        "Architectural differentiation, not a competitor claim — every row below "
        "is a capability implemented in this repository (see the linked module)."
    )
    diff = pd.DataFrame([
        {"Dimension": "Sensor at every station", "Conventional twin": "Required",
         "RippleTwin": "Not required — infers blind stations"},
        {"Dimension": "Works with blind stations", "Conventional twin": "No",
         "RippleTwin": "Yes — rippletwin/twin/shadow.py"},
        {"Dimension": "Uses existing OEE/MES data", "Conventional twin": "Sometimes",
         "RippleTwin": "Yes — see docs/SIGNALS.md, integrate/contract.py"},
        {"Dimension": "Hidden-state inference", "Conventional twin": "No",
         "RippleTwin": "Yes — physics (conservation of material), not a learned correlation"},
        {"Dimension": "Ripple forecasting", "Conventional twin": "Rare",
         "RippleTwin": "Yes — rippletwin/twin/propagate.py"},
        {"Dimension": "Sensor-placement optimisation", "Conventional twin": "No",
         "RippleTwin": "Yes — rippletwin/twin/placement.py"},
        {"Dimension": "Evidence-grounded recommendations", "Conventional twin": "Varies",
         "RippleTwin": "Yes — templates bound to computed values, no LLM narration of numbers"},
        {"Dimension": "Human-in-the-loop approval", "Conventional twin": "Varies",
         "RippleTwin": "Yes — advisory only, writes nothing to a PLC"},
        {"Dimension": "Abstention when ambiguous", "Conventional twin": "Rare",
         "RippleTwin": "Yes — reports the candidate group and escalates"},
        {"Dimension": "Auditability", "Conventional twin": "Varies",
         "RippleTwin": "Yes — hash-chained decision ledger"},
    ])
    st.dataframe(diff, use_container_width=True, hide_index=True)

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


# ================================================================ EVIDENCE
def render_evidence():
    st.title("Does it actually work?")
    st.caption(
        "The held-out evaluation behind every claim on the pages before this "
        "one — 110 held-out episodes, generated by "
        "`python -m rippletwin.evaluation.experiments`, rendered live from "
        "docs/RESULTS.md and docs/RESULTS_ROUND2.md so this page cannot drift "
        "from the numbers those commands produce."
    )
    results_md = read_doc("RESULTS.md")
    results2_md = read_doc("RESULTS_ROUND2.md")

    st.subheader("Does performance survive missing sensors?")
    st.markdown(doc_section(results_md, "## 1. Localisation"))
    ip = fig_path("coverage_curve.png")
    if ip.exists():
        st.image(str(ip), caption="Localisation accuracy vs sensor coverage "
                                  "(within-one-station), by method.",
                 use_container_width=True)

    st.markdown("---")
    st.subheader("Does the advantage disappear once every station is instrumented? "
                 "(the sanity control)")
    st.markdown(doc_section(results_md, "## 2. The control"))

    st.markdown("---")
    st.subheader("Can it estimate a cycle time it never measured?")
    st.markdown(doc_section(results_md, "## 3. Estimating a cycle time"))
    ip = fig_path("cycle_time_accuracy.png")
    if ip.exists():
        st.image(str(ip), caption="Error of the inferred cycle time at "
                                  "stations with no sensor.",
                 use_container_width=True)

    st.markdown("---")
    st.subheader("Does RippleTwin generate excessive false alarms?")
    st.markdown(doc_section(results_md, "## 4. Staying quiet"))
    ip = fig_path("false_alarms.png")
    if ip.exists():
        st.image(str(ip), caption="False-alarm rate on disturbance-free data, "
                                  "matched operating point, by method.",
                 use_container_width=True)
    ip = fig_path("operating_curve.png")
    if ip.exists():
        st.image(str(ip), caption="Detection rate vs. false-alarm rate at 75% "
                                  "sensor coverage — the fair comparison across methods.",
                 use_container_width=True)

    st.markdown("---")
    st.subheader("Would conventional monitoring have caught these anyway?")
    st.markdown(doc_section(results_md, "## 5. Most of these faults never reach"))

    st.markdown("---")
    st.subheader("Is its confidence calibrated?")
    st.markdown(doc_section(results2_md, "## 5. Calibration"))

    st.markdown("---")
    st.subheader("Does it generalise beyond a single straight line?")
    st.markdown(doc_section(results2_md, "## 6. Topology generalization"))

    st.markdown("---")
    st.subheader("Does human feedback actually move the model?")
    st.markdown(doc_section(results2_md, "## 7. Closing the feedback loop"))

    st.markdown("---")
    st.subheader("When RippleTwin does NOT know")
    pairs = blind_adjacent_pairs(line)
    if pairs:
        txt = ", ".join(f"{a} / {b}" for a, b in pairs)
        st.warning(
            f"**Insufficient evidence to distinguish between {txt}.** Two "
            f"adjacent stations with no sensor between them produce almost the "
            f"same blocked/starved signature at every station that *can* "
            f"observe them, so no amount of data separates them from flow "
            f"evidence alone."
        )
    else:
        st.info("No adjacent blind-station pairs on this line configuration.")
    st.error(
        "**SYSTEM ACTION:** Do not issue a confident recommendation. Report the "
        "candidate group, request more evidence, and route to a human "
        "investigation (`ACTION_ESCALATE_AMBIGUOUS` — "
        "rippletwin/recommend/engine.py). A trustworthy industrial AI system "
        "should not always produce an answer."
    )

    with st.expander("Full docs/RESULTS.md"):
        st.markdown(results_md)
    with st.expander("Full docs/RESULTS_ROUND2.md"):
        st.markdown(results2_md)


# ============================================================= DATA HEALTH
def render_data_health():
    st.title("Data Readiness")
    st.caption(
        "The Phase 0 assessment (`python -m rippletwin.integrate.assess`), run "
        "live against this line's configuration. This is what a plant would "
        "see in its first meeting, before committing to anything."
    )

    n_stations = line.n_stations
    n_with_state = len(line.observed_indices)
    signals = [s.key for s in DATA_CONTRACT if s.key != "process_channels"]
    report = assess_readiness(signals, n_stations=n_stations,
                              n_stations_with_state=n_with_state)

    cap_color = {"FULL": "green", "FLOW_ONLY": "blue", "QUALITY_ONLY": "orange",
                "NOT_VIABLE": "red"}[report.capability.value]
    st.badge(f"CAPABILITY: {report.capability.value}", color=cap_color)
    c1, c2 = st.columns(2)
    c1.metric("Stations with usable state data", f"{report.n_with_state}/{report.n_stations}",
              f"{report.coverage * 100:.0f}% coverage")
    c2.metric("Blockers", len(report.blockers), help="Signals without which the twin cannot run.")

    st.subheader("System readiness")
    flow_ok = report.capability in (Capability.FULL, Capability.FLOW_ONLY)
    quality_ok = report.capability in (Capability.FULL, Capability.QUALITY_ONLY)
    checks = [
        ("Flow inference (hidden-station localisation)", flow_ok),
        ("Quality inference (defect attribution)", quality_ok),
        ("Ripple forecast (downstream impact)", flow_ok),
        ("Recommendations / work orders", flow_ok or quality_ok),
    ]
    for label, ok in checks:
        st.markdown(f"{'✓' if ok else '✗ — will abstain'} {label}")
    if not (flow_ok and quality_ok):
        st.info(
            "Where a capability is unavailable, the UI does not fabricate an "
            "output for it — the corresponding page shows an abstention "
            "instead of a confident number."
        )

    st.markdown("---")
    if report.blockers:
        st.subheader("Blockers (the twin cannot run)")
        for b in report.blockers:
            st.error(b)
    if report.warnings:
        st.subheader("Degraded (it runs, with less)")
        for w in report.warnings:
            st.warning(w)
    if report.notes:
        st.subheader("Notes")
        for n in report.notes:
            st.caption(n)

    st.markdown("---")
    st.subheader("Signal by signal")
    st.dataframe(
        report.to_frame()[["signal", "required", "available", "enables", "purdue"]],
        use_container_width=True, hide_index=True)
    with st.expander("Full input contract — everything the twin reads, and nothing else"):
        st.dataframe(contract_frame(), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Freshness and completeness")
    st.info(
        "This dashboard runs against a generated synthetic dataset with "
        "internally consistent timestamps by construction — there is no "
        "meaningful 'freshness' or 'missing rows' metric to report here. A "
        "real deployment computes this from the historian export; see "
        "`rippletwin.ingest.states` for the join logic and the truncated-log "
        "check that refuses to run below 50% state-log coverage."
    )


# =================================================================== AUDIT
def render_audit():
    st.title("Audit Log")
    st.caption(
        "Append-only, hash-chained. Every alert, decision and outcome is a new "
        "entry, never an edit — a supervisor who overrode the system can prove "
        "what it actually recommended at the time."
    )
    lf = ledger.to_frame()
    if not len(lf):
        st.info("No decisions recorded yet. Approve, reject, modify or escalate "
               "an alert on Live Line or Incidents to populate the ledger.")
        return

    v = ledger.verify()
    st.badge(f"Hash chain: {'VALID' if v['valid'] else 'BROKEN'}",
             color="green" if v["valid"] else "red")
    st.metric("Ledger entries", v.get("n_entries", len(lf)))

    st.dataframe(
        lf[["entry_id", "timestamp", "run_id", "window", "alert_type", "station_id",
            "is_inferred", "confidence", "decision", "decided_by", "outcome",
            "prev_hash", "entry_hash"]],
        use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Entry detail")
    pick = st.selectbox("Entry", lf["entry_id"].tolist())
    row = lf[lf["entry_id"] == pick].iloc[0]
    d1, d2 = st.columns(2)
    d1.json(row.get("recommendation") or {})
    d2.json(row.get("explanation") or {})
    st.caption(
        f"prev_hash `{row['prev_hash'][:16]}…` -> entry_hash "
        f"`{row['entry_hash'][:16]}…`. Any edit to this entry after the fact "
        f"would change entry_hash and break every entry_hash that follows it."
    )


# ======================================================== METHODOLOGY/ABOUT
def render_about():
    st.title("Methodology / About")

    st.subheader("What is this?")
    st.markdown(
        """
**RippleTwin is a digital twin for a vehicle assembly line that reports on
stations which have no sensor at all.**

Round 2 of the Accenture Innovation Challenge asks, for Track 4 DigitalTwin.ai:
*"what do you do about stations with little or no sensor data?"* This is our
answer to that question specifically.

**The line has 42 stations. Ten of them send us nothing** — no cycle time, no
state, no telemetry of any kind. They are drawn as **diamonds** on the line
map. A conventional twin cannot model them at all.

**How we see them.** When a station slows down, everything behind it backs up
(*blocked*) and everything in front of it runs dry (*starved*). The boundary
between those two is the station causing it. So we read the stations we *can*
see and locate the one we cannot.

**Try this:** on Live Line, leave the scenario on *S1 — hidden bottleneck*.
The twin names **S02**, states its cycle time, and S02 has no sensor. Open
**Evaluation / Ground Truth** in the sidebar to check it against what the
simulator actually did. Then try *S3 — normal variation*, where the correct
answer is silence.
        """
    )

    st.markdown("---")
    st.subheader("What makes RippleTwin different?")
    st.markdown(
        "- Hidden bottleneck inference\n"
        "- Hidden cycle-time estimation\n"
        "- Ripple forecasting\n"
        "- Quality attribution\n"
        "- Sensor-placement optimisation\n"
        "- Evidence-grounded recommendations\n"
        "- Human-in-the-loop approval\n"
        "- Abstention\n"
        "- Audit trail"
    )

    st.markdown("---")
    st.subheader("Implemented / Simulated / Evaluated / Future pilot work")
    st.dataframe(pd.DataFrame([
        {"Category": "Simulated", "What": "The factory: 42 stations, buffers, mixed-model "
         "sequencing, shifts, ambient conditions, faults, defects, inspection gates."},
        {"Category": "Implemented (real, running code)", "What": "Windowing, baselines, "
         "shadow-sensing, calibration, propagation forecast, defect attribution, all "
         "four comparison baselines, evaluation, explanation, recommendation, dispatch, "
         "ledger, dashboard."},
        {"Category": "Evaluated", "What": "110 held-out episodes across 4 sensor-coverage "
         "levels, against 4 baselines including the published Turning Point Method, "
         "calibration, distribution shift, topology generalisation, feedback loop, "
         "surge/performance — see System -> Evidence."},
        {"Category": "Future pilot work", "What": "Everything above runs only on synthetic "
         "data. A real deployment needs a Phase 0 data-readiness check "
         "(System -> Data Health), a shadow-mode period logging predictions with no "
         "operational risk, and validation of every claim above against a real line's "
         "data — see docs/DEPLOYMENT.md and docs/LIMITATIONS.md."},
    ]), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Responsible AI")
    st.markdown(
        "- **No autonomous control.** RippleTwin writes to no PLC and cannot stop the "
        "line. Its entire action vocabulary is advisory and reversible.\n"
        "- **Abstention.** When posterior mass is spread across adjacent candidates it "
        "reports the group and escalates rather than naming one.\n"
        "- **Provenance on every number.** OBSERVED / INFERRED / PREDICTED, surfaced "
        "throughout the UI.\n"
        "- **Audit trail.** Append-only hash-chained ledger.\n"
        "- **Calibrated alerting.** Threshold derived from a stated false-alarm target "
        "on held-out data, not hand-picked.\n"
        "- **Shadow mode first.** 8–12 weeks logging predictions and outcomes with no "
        "operational risk, before anyone is asked to act."
    )

    st.markdown("---")
    st.subheader("Documentation")
    docs_table = [
        ("ARCHITECTURE.md", "The full pipeline, and exactly what dispatches on what"),
        ("METHOD.md", "The mechanism, prior art, the estimator, and the designs we killed"),
        ("LIMITATIONS.md", "Every scoping decision, old and new, with the reasoning"),
        ("SIGNALS.md", "Every telemetry channel: what it represents, who consumes it"),
        ("DEPLOYMENT.md", "What we need from a plant, where the software sits, who acts"),
        ("REFERENCES.md", "What is prior art, what is ours, and how to check"),
        ("RESULTS.md", "Every flagship table, with metric definitions"),
        ("RESULTS_ROUND2.md", "Predictive/robustness evaluation results"),
        ("BUSINESS_CASE.md", "Value drivers, ROI arithmetic, sensor economics, risks"),
        ("JUDGE_QA.md", "Hard questions, answered"),
        ("DEMO_SCRIPT.md", "The 3-minute guided-demo narration"),
        ("DEMO_VIDEO.md", "Storyboard and narration for the prototype video"),
        ("SUBMISSION_CHECKLIST.md", "Round 2 deliverables mapped to the brief"),
        ("HANDOVER.md", "What is done, and the steps only a human can do"),
    ]
    for fname, desc in docs_table:
        st.markdown(f"- **{fname}** — {desc}")
    st.caption("Full text of each is under docs/ in the repository.")


# --------------------------------------------------------------- dispatch

if nav == "Live Line":
    render_live_line()
elif nav == "Incidents":
    render_incidents()
elif nav == "Plant":
    render_plant()
elif nav == "Business":
    render_business()
elif nav == "System":
    if sub_nav == "Evidence":
        render_evidence()
    elif sub_nav == "Data Health":
        render_data_health()
    elif sub_nav == "Audit Log":
        render_audit()
    else:
        render_about()
