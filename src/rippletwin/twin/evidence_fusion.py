"""Flow-path + quality-path evidence fusion for ambiguous blind-station groups.

Scope, deliberately narrow (see docs/RESEARCH_EVALUATION.md Hybrid 2):
fires ONLY when (a) the flow posterior's leading candidate belongs to a
group placement.ambiguity() has flagged as highly confusable, AND (b) an
independent quality-path signal (twin.genealogy.quality_state) exists for
at least one candidate in that group over an overlapping time window.
Everywhere else, this module has zero effect.

Mechanism: additive log-likelihood combination,
    combined_score(station) = flow_llr(station) + quality_llr(station)
restricted to stations in the ambiguous group -- the same "independent
evidence channels combine additively in log-space" principle
twin.shadow's own z_proc likelihood-ratio term already uses. No training
data, fully deterministic, fully auditable (both terms are already
individually explainable).

This module NEVER overwrites twin.shadow's `top_station`. It produces new,
opt-in columns (`fused_top_station`, `fused_llr_margin`) that a caller may
choose to read. Promotion to replace `top_station` as the default is a
separate, explicitly gated decision -- see runbook step C4 and
docs/RESEARCH_EVALUATION.md Hybrid 2's decision gate.

Known limitation (runbook C0/C1 note): ``ShadowSensor`` does not currently
expose a per-candidate flow LLR vector, only the winning station's LLR (the
``llr`` field on each ``ShadowResult``/shadow row). So ``fuse_ambiguous_group``
below can only attribute the flow LLR to the row's own leading candidate;
other group members get 0 flow contribution, which is an approximation, not
an exact per-candidate score. This is honestly documented here rather than
silently shipped as if it were exact -- a sharper version exposing the full
per-candidate vector is a natural follow-up, not attempted in this plan.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

AMBIGUITY_THRESHOLD = 0.90  # cosine similarity above which two stations are "flagged confusable"


def ambiguous_groups(ambiguity_df: pd.DataFrame, threshold: float = AMBIGUITY_THRESHOLD) -> List[List[int]]:
    """Connected groups of mutually-confusable hidden stations.

    ``ambiguity_df`` is the output of twin.placement.ambiguity(). Two
    stations are linked if either names the other as its best-matching
    rival (``confusable_with``) at ``ambiguity >= threshold``. Returns a
    list of station-index groups (singletons excluded -- only groups of
    size >= 2 are ambiguous by definition).
    """
    flagged = ambiguity_df[ambiguity_df["ambiguity"] >= threshold]
    if flagged.empty:
        return []
    id_to_index = {row.station_id: row.station for row in ambiguity_df.itertuples()}
    edges = set()
    for row in flagged.itertuples():
        partner_idx = id_to_index.get(row.confusable_with)
        if partner_idx is not None:
            edges.add(frozenset({row.station, partner_idx}))
    # Union-find over the edge set to get connected components.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for edge in edges:
        a, b = tuple(edge)
        union(a, b)

    groups: dict = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


def fuse_ambiguous_group(
    flow_shadow_row: pd.Series,
    quality_state_df: pd.DataFrame,
    ambiguity_group: List[int],
    window_lo_s: float,
    window_hi_s: float,
    window_bounds: pd.DataFrame,
) -> Optional[dict]:
    """Re-rank one flow-flagged ambiguous group using overlapping quality LLR.

    ``window_bounds`` (from twin.genealogy.window_bounds_from) maps
    quality-path window indices to (t_lo, t_hi) so overlap with the flow
    window's [window_lo_s, window_hi_s] can be found -- see runbook step
    C0.1 for why this alignment is the crux of the whole plan.

    Returns None if no quality evidence overlaps the group/window (the
    fusion condition doesn't hold -- caller should fall back to the
    unfused top_station). Otherwise returns
    {"fused_top_station": int, "fused_llr_margin": float}.
    """
    overlapping = window_bounds[
        (window_bounds["t_hi"] >= window_lo_s) & (window_bounds["t_lo"] <= window_hi_s)
    ]
    if overlapping.empty:
        return None
    q = quality_state_df[
        quality_state_df["window"].isin(overlapping["window"])
        & quality_state_df["station"].isin(ambiguity_group)
    ]
    if q.empty:
        return None

    quality_llr_by_station = q.groupby("station")["llr"].sum().to_dict()
    if not any(quality_llr_by_station.get(s, 0.0) > 0.0 for s in ambiguity_group):
        return None  # quality path has no signal on ANY group member -- nothing to fuse

    flow_llr = float(flow_shadow_row.get("llr", 0.0))
    scores = {}
    for s in ambiguity_group:
        # Flow LLR is only meaningfully attributable to the row's own leading
        # candidate; other group members get the flow LLR at 0 contribution
        # (they were tied/near-tied by construction of an ambiguous group,
        # which is exactly why we don't have a per-station flow LLR here --
        # confirmed during C0 that ShadowSensor exposes only the winning
        # candidate's LLR, not a full per-candidate vector; see module
        # docstring's "Known limitation").
        own_flow = flow_llr if s == int(flow_shadow_row["top_station"]) else 0.0
        scores[s] = own_flow + quality_llr_by_station.get(s, 0.0)

    best_station = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
    return {"fused_top_station": best_station, "fused_llr_margin": margin}
