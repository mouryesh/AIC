"""Shadow-sensing: reconstructing the state of stations that have no sensor.

The mechanism
-------------
In a serial line with finite buffers, material is conserved. If station ``k``
slows, parts stop arriving downstream and stop leaving upstream:

    every station downstream of k  ->  starves
    every station upstream of k    ->  blocks

The boundary between the two sits at k. We never need a sensor *at* k -- we need
sensors on both sides of it.

This is a structural argument, not a correlation. A correlation-based detector
learns "when S07 blocks, S09 starves" and has no idea which of them caused it.
The flow model knows the direction of causation a priori, because material only
moves one way down the line.

Blocking and starving are not symmetric
---------------------------------------
An earlier version of this model treated ``blocked - starved`` as a single
signed "pressure" channel and fitted one amplitude to it. That is physically
wrong and it mislocalised faults by one station. The reason:

* Starvation downstream appears within a few vehicles and at full magnitude --
  the buffer simply runs dry.
* Blocking upstream appears only once the intervening buffer *fills*, which
  takes ``buffer_capacity / (rate deficit)`` vehicles, and its magnitude is
  capped by how far the backlog has propagated.

Measured on this line, a disturbance produced ~18 s of downstream starvation
against ~1.2 s of upstream blocking in the same window. Forcing one amplitude
to explain both drags the fitted boundary toward the side with more signal.

So the two channels are modelled separately, each with its own non-negative
amplitude and its own propagation length scale. This also means the model
degrades gracefully: early in an event, before any buffer has filled, the
blocking channel contributes nothing and localisation runs on the starvation
boundary alone -- which is precisely where the warning lead time comes from.

Observed vs inferred
--------------------
For an *observed* candidate we additionally have ``z_proc``: we can see directly
whether that station's own tool cycle got slower. Blocking and starving do not
change a station's processing time, so ``z_proc`` cleanly separates "this
station is the constraint" from "this station is a victim of the constraint".

The evidence is applied as a proper likelihood ratio, so it cuts both ways: an
observed station running at normal speed is evidence *against* that station
being the constraint. Hidden stations receive no such term, which is exactly
right -- we have no direct evidence about them either way, only structural
evidence. This asymmetry is what lets the twin prefer a hidden neighbour over an
instrumented station that is demonstrably running fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..factory.topology import LineTopology

NULL_HYPOTHESIS = "NULL"
LINE_SUPPLY_HYPOTHESIS = "LINE_SUPPLY"

_LOG_SQRT_2PI = 0.9189385332046727


@dataclass
class ShadowConfig:
    """Tunables for the hidden-state estimator.

    Propagation length scales are in **buffer slots**, because that is the
    physical quantity that governs how far a disturbance travels before being
    absorbed. Starvation reaches further than blocking within a given window,
    for the timing reason described in the module docstring.
    """

    lambda_starve: float = 16.0
    lambda_block: float = 9.0
    #: Assumed processing-time deviation, in sigma, of a station that is the constraint.
    fault_z_mean: float = 3.5
    fault_z_sd: float = 3.0
    #: Weight on the direct processing-time likelihood ratio.
    direct_weight: float = 1.0
    #: Consecutive windows a candidate must lead before an alert is raised.
    persistence: int = 2
    #: Posterior mass the candidate group needs before the twin names a station.
    confidence_floor: float = 0.45
    #: EWMA smoothing factor applied to per-window log-likelihood ratios.
    ewma_alpha: float = 0.55
    #: Prior probability that nothing is wrong.
    null_prior: float = 0.90
    #: Detection threshold on the calibrated log-likelihood ratio. Set by
    #: ``ShadowSensor.calibrate`` against a target false-alarm rate; the default
    #: is only a placeholder for un-calibrated use.
    detect_llr: float = 8.0
    #: Correlation-correction factor applied to the flow log-likelihood.
    #: Stations on a coupled line are not independent observations; treating
    #: them as such inflates evidence by roughly 1/tau. Set by ``calibrate``.
    tau: float = 1.0
    #: A looser, earlier pre-alarm threshold on the same LLR statistic, set by
    #: ``calibrate`` at a higher target false-alarm rate than ``detect_llr``.
    #: Used by ``twin.predict`` to raise a WATCH state before the line has
    #: enough evidence to name a station -- the same statistic, a different
    #: point on its own calibrated null distribution, not a separate detector.
    watch_llr: float = 4.0
    #: Standard deviation of the LLR statistic under the null (no-fault)
    #: distribution, from the same calibration run. Used by ``twin.predict``
    #: to size "is this trending up" as a multiple of the statistic's own
    #: noise rather than an arbitrary slope constant.
    llr_noise_std: float = 1.0
    #: Optional per-station-index prior weight (1.0 = uniform), set by
    #: ``twin.feedback.apply_feedback`` from validated human-in-the-loop
    #: outcomes (``hitl.ledger.precision_by_station``). ``None`` -- every
    #: caller that has not explicitly closed the feedback loop -- reproduces
    #: today's uniform prior exactly; see ``ShadowSensor._posterior``.
    station_prior_weight: Optional[Dict[int, float]] = None


# ------------------------------------------------------------------ propagation


def buffer_distance_matrix(line: LineTopology) -> np.ndarray:
    """Cumulative buffer capacity between every ordered pair of stations.

    ``D[i, k]`` is the total number of buffer slots lying between station i and
    station k -- the physically meaningful distance for disturbance propagation.
    Two stations three apart with a 14-slot inter-zone buffer between them are
    far more decoupled than two stations three apart inside body shop.

    Dispatches on ``line.is_graph``: a plain serial line (the default, and
    every previously-published result in this repository) runs the exact
    prefix-sum computation this function has always used. A configured
    process graph runs ``_graph_buffer_distance_matrix`` instead, which
    reduces to the identical prefix-sum result on a chain but generalizes to
    a DAG via shortest-path distance.
    """
    if line.is_graph:
        return _graph_buffer_distance_matrix(line)
    n = line.n_stations
    caps = np.array(
        [min(line.stations[i].out_buffer, 10**4) for i in range(n)], dtype=float
    )
    prefix = np.concatenate([[0.0], np.cumsum(caps[: n - 1])])
    return np.abs(prefix[:, None] - prefix[None, :])


def _graph_buffer_distance_matrix(line: LineTopology) -> np.ndarray:
    """Shortest-path distance using buffer capacity as edge weight, treating
    edges as undirected for the purpose of distance (a buffer decouples two
    stations regardless of which direction you measure across it in -- the
    same symmetry ``|prefix[i] - prefix[k]|`` already has on a chain).

    On a serial chain this is the unique-path sum, i.e. numerically identical
    to the prefix-sum fast path; it is only reached at all when
    ``line.is_graph`` is True, so the fast path is never displaced.
    """
    import heapq

    n = line.n_stations
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for e in line.effective_edges():
        w = float(min(e.buffer_capacity, 10**4))
        adj[e.src].append((e.dst, w))
        adj[e.dst].append((e.src, w))

    D = np.full((n, n), np.inf)
    for src in range(n):
        dist = np.full(n, np.inf)
        dist[src] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, src)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        D[src] = dist
    # Unreachable pairs (a disconnected graph, not expected from any config
    # this project ships) get a large-but-finite distance rather than inf,
    # so downstream exp(-D/lambda) decays to ~0 instead of producing NaN.
    D[~np.isfinite(D)] = 1e6
    return D


def propagation_matrices(
    line: LineTopology, cfg: ShadowConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Expected blocking and starvation profiles for each candidate constraint.

    Returns ``(B, S)`` where ``B[k, i]`` is the relative blocking induced at
    station ``i`` when ``k`` is the constraint (non-zero only upstream of k) and
    ``S[k, i]`` is the relative starvation (non-zero only downstream of k).

    Dispatches on ``line.is_graph`` exactly as ``buffer_distance_matrix``
    does. On a graph, "upstream of k" / "downstream of k" generalize from
    ``i < k`` / ``i > k`` to ``i in line.ancestors(k)`` / ``i in
    line.descendants(k)`` -- graph reachability rather than index comparison.
    """
    if line.is_graph:
        return _graph_propagation_matrices(line, cfg)
    n = line.n_stations
    D = buffer_distance_matrix(line)
    idx = np.arange(n)
    upstream = np.subtract.outer(idx, idx) > 0    # [k, i] True when i < k
    downstream = np.subtract.outer(idx, idx) < 0  # [k, i] True when i > k

    B = np.where(upstream, np.exp(-D / cfg.lambda_block), 0.0)
    S = np.where(downstream, np.exp(-D / cfg.lambda_starve), 0.0)
    return B, S


def _graph_propagation_matrices(
    line: LineTopology, cfg: ShadowConfig
) -> Tuple[np.ndarray, np.ndarray]:
    n = line.n_stations
    D = buffer_distance_matrix(line)
    B = np.zeros((n, n))
    S = np.zeros((n, n))
    for k in range(n):
        for i in line.ancestors(k):
            B[k, i] = np.exp(-D[k, i] / cfg.lambda_block)
        for i in line.descendants(k):
            S[k, i] = np.exp(-D[k, i] / cfg.lambda_starve)
    return B, S


def _norm_logpdf(x: np.ndarray | float, mu: float, sd: float) -> np.ndarray | float:
    z = (np.asarray(x, dtype=float) - mu) / sd
    return -0.5 * z**2 - np.log(sd) - _LOG_SQRT_2PI


@dataclass
class ShadowResult:
    """Per-window output of the hidden-state estimator."""

    window: int
    posterior: Dict[object, float]
    top_station: Optional[int]
    top_prob: float
    #: Contiguous group of stations that jointly explain the data, and its mass.
    group: List[int]
    group_prob: float
    #: Calibrated log-likelihood ratio of the best hypothesis against NULL.
    llr: float
    #: Fitted starvation amplitude, in takt fractions. The severity read-out.
    amp_starve: float
    #: Fitted blocking amplitude, in takt fractions.
    amp_block: float
    #: True when the top candidate has no sensor -- an inferred, not observed, state.
    top_is_hidden: bool
    detected: bool
    confident: bool
    v_start: int = 0
    v_end: int = 0
    t_mid_s: float = 0.0
    evidence: dict = field(default_factory=dict)
    #: Second-highest posterior mass among non-NULL, non-LINE_SUPPLY station
    #: hypotheses, and its station index. Additive (Plan A, see
    #: evaluation/bottleneck_diagnosis.py) -- unused by any existing caller.
    runner_up_station: Optional[int] = None
    runner_up_prob: float = 0.0


class ShadowSensor:
    """Estimates hidden station state from observed flow deviations."""

    def __init__(self, line: LineTopology, cfg: ShadowConfig | None = None) -> None:
        self.line = line
        self.cfg = cfg or ShadowConfig()
        self.B, self.S = propagation_matrices(line, self.cfg)
        # Starvation profile for a shortfall entering at the head of the line.
        # An inbound supply interruption starves station 0 hardest and decays
        # downstream as buffers absorb it -- exactly like a constraint sitting
        # just upstream of station 0.
        D0 = buffer_distance_matrix(line)[0]
        self.head_starve = np.exp(-D0 / self.cfg.lambda_starve)
        self.observed = np.array(line.observed_indices, dtype=int)
        self._obs_mask = np.zeros(line.n_stations, dtype=bool)
        self._obs_mask[self.observed] = True
        self._ewma: Optional[np.ndarray] = None
        self._lead_history: List[int] = []
        self.last_results: List[ShadowResult] = []

    # ------------------------------------------------------------------ scoring

    def _hypothesis_scores(
        self,
        d_blocked: np.ndarray,
        d_starved: np.ndarray,
        z_proc: np.ndarray,
        sigma_b: float,
        sigma_s: float,
    ) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        """Score every hypothesis for one window.

        Returns ``(station_scores, null_score, line_score, amp_block, amp_starve)``
        as un-normalised log-likelihoods (flow term already correlation-corrected,
        direct term added).
        """
        cfg = self.cfg
        n = self.line.n_stations
        mask = np.isfinite(d_blocked) & np.isfinite(d_starved)
        if mask.sum() < 4:
            return np.full(n, -np.inf), 0.0, -np.inf, np.zeros(n), np.zeros(n)

        db = np.where(mask, np.nan_to_num(d_blocked), 0.0)
        ds = np.where(mask, np.nan_to_num(d_starved), 0.0)
        m = mask.astype(float)

        B = self.B * m[None, :]
        S = self.S * m[None, :]

        # Non-negative least squares for one amplitude per channel. A constraint
        # slows the line; it never speeds it up, so negative amplitudes are
        # physically meaningless and are clipped to zero.
        bb = np.einsum("ki,ki->k", B, B)
        ss = np.einsum("ki,ki->k", S, S)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_b = np.where(bb > 1e-12, (B @ db) / bb, 0.0)
            a_s = np.where(ss > 1e-12, (S @ ds) / ss, 0.0)
        a_b = np.maximum(np.nan_to_num(a_b), 0.0)
        a_s = np.maximum(np.nan_to_num(a_s), 0.0)

        # Residuals on BOTH channels for ALL observed stations. Predicting zero
        # blocking downstream and zero starvation upstream is itself information:
        # a station that is starved when the hypothesis says it should be blocked
        # counts against that hypothesis.
        rb = db[None, :] - a_b[:, None] * B
        rs = ds[None, :] - a_s[:, None] * S
        ll_flow = -0.5 * (
            np.einsum("ki,ki->k", rb, rb) / sigma_b**2
            + np.einsum("ki,ki->k", rs, rs) / sigma_s**2
        )

        ll_null = float(
            -0.5 * (np.sum(db**2) / sigma_b**2 + np.sum(ds**2) / sigma_s**2)
        )

        # LINE_SUPPLY: an inbound shortfall starves the line from the head, with
        # no blocking anywhere and therefore no boundary to localise.
        #
        # This was originally modelled as *uniform* starvation, which is wrong.
        # Real supply starvation decays downstream as buffers absorb it, so a
        # decaying station hypothesis fitted the data better and a material
        # delay was attributed to station S02 with overwhelming confidence --
        # while station S01 sat starved at 179% of takt, which is the one thing
        # that cannot happen if S02 is the constraint (S01 would be *blocked*).
        #
        # Modelling the head shortfall with the same decay profile makes the
        # comparison fair, and the discriminator becomes the behaviour of the
        # first station: blocked means something ahead is slow, starved means
        # nothing is arriving.
        prof = self.head_starve * m
        denom = float(prof @ prof)
        a_line = max(0.0, float(prof @ ds / denom)) if denom > 1e-12 else 0.0
        r_line_s = ds - a_line * prof
        ll_line = float(
            -0.5 * (np.sum(db**2) / sigma_b**2 + np.sum(r_line_s**2) / sigma_s**2)
        )

        # Correlation correction. Stations on a coupled line are far from
        # independent observations; without this the evidence is inflated by
        # roughly 1/tau and the detector fires constantly.
        ll_flow = ll_flow * cfg.tau
        ll_null = ll_null * cfg.tau
        ll_line = ll_line * cfg.tau

        # Direct processing-time evidence, as a two-sided likelihood ratio.
        direct = np.zeros(n)
        zp = np.nan_to_num(z_proc, nan=0.0)
        obs = self._obs_mask
        direct[obs] = cfg.direct_weight * (
            _norm_logpdf(zp[obs], cfg.fault_z_mean, cfg.fault_z_sd)
            - _norm_logpdf(zp[obs], 0.0, 1.0)
        )

        return ll_flow + direct, ll_null, ll_line, a_b, a_s

    def _posterior(
        self, ll_station: np.ndarray, ll_null: float, ll_line: float
    ) -> Dict[object, float]:
        cfg = self.cfg
        n = self.line.n_stations
        log_prior_null = np.log(cfg.null_prior)
        log_prior_other = np.log((1.0 - cfg.null_prior) / (n + 1))

        # Per-station prior from validated human feedback (twin/feedback.py),
        # optional and off by default. When cfg.station_prior_weight is None
        # -- every existing caller, every existing test, every published
        # result -- log_prior_station reduces to exactly log_prior_other
        # repeated n times, i.e. the identical uniform prior this method has
        # always used. Only a caller that explicitly sets the weight map
        # (via twin.feedback.apply_feedback) sees different behaviour.
        if cfg.station_prior_weight:
            w = np.array(
                [cfg.station_prior_weight.get(i, 1.0) for i in range(n)], dtype=float
            )
            w = w / w.sum()
            log_prior_station = np.log((1.0 - cfg.null_prior) * n * w / (n + 1) + 1e-300)
        else:
            log_prior_station = np.full(n, log_prior_other)

        keys: List[object] = list(range(n)) + [NULL_HYPOTHESIS, LINE_SUPPLY_HYPOTHESIS]
        logp = np.concatenate(
            [
                ll_station + log_prior_station,
                [ll_null + log_prior_null],
                [ll_line + log_prior_other],
            ]
        )
        logp = np.where(np.isfinite(logp), logp, -np.inf)
        w = np.exp(logp - np.max(logp))
        w = w / w.sum()
        return {k: float(v) for k, v in zip(keys, w)}

    # ------------------------------------------------------------------ windows

    def estimate_window(
        self,
        window: int,
        d_blocked: np.ndarray,
        d_starved: np.ndarray,
        z_proc: np.ndarray,
        sigma_b: float,
        sigma_s: float,
        v_start: int = 0,
        v_end: int = 0,
        t_mid_s: float = 0.0,
        use_ewma: bool = True,
    ) -> ShadowResult:
        """Estimate hidden state for a single window."""
        cfg = self.cfg
        n = self.line.n_stations
        ll_st, ll_null, ll_line, a_b, a_s = self._hypothesis_scores(
            d_blocked, d_starved, z_proc, sigma_b, sigma_s
        )

        # Temporal smoothing on the log-likelihood ratio against NULL. A genuine
        # fault persists across windows; a noise excursion does not.
        rel = np.where(np.isfinite(ll_st), ll_st - ll_null, -np.inf)
        finite_rel = np.where(np.isfinite(rel), rel, 0.0)
        if use_ewma:
            if self._ewma is None:
                self._ewma = finite_rel.copy()
            else:
                self._ewma = cfg.ewma_alpha * finite_rel + (1 - cfg.ewma_alpha) * self._ewma
            rel_s = self._ewma
        else:
            rel_s = finite_rel

        post = self._posterior(rel_s, 0.0, ll_line - ll_null)
        station_post = np.array([post[k] for k in range(n)])
        top = int(np.argmax(station_post))
        top_p = float(station_post[top])
        llr = float(rel_s[top])

        # Runner-up: second-highest posterior mass among the n station
        # hypotheses (excluding NULL/LINE_SUPPLY), and its index. Additive
        # field for Plan A's shift-severity diagnostic.
        if n > 1:
            runner_up = int(np.argmax(np.where(np.arange(n) == top, -np.inf, station_post)))
            runner_up_p = float(station_post[runner_up])
        else:
            runner_up = None
            runner_up_p = 0.0

        # Adjacent stations -- especially adjacent hidden ones -- are often
        # genuinely indistinguishable. Rather than pretend otherwise, report the
        # contiguous group around the top candidate and its combined mass.
        group = [top]
        lo, hi = top - 1, top + 1
        while lo >= 0 and station_post[lo] >= 0.25 * top_p:
            group.append(lo)
            lo -= 1
        while hi < n and station_post[hi] >= 0.25 * top_p:
            group.append(hi)
            hi += 1
        group = sorted(group)
        group_p = float(station_post[group].sum())

        self._lead_history.append(top)
        recent = self._lead_history[-cfg.persistence:]
        persistent = len(recent) >= cfg.persistence and all(
            abs(r - top) <= 1 for r in recent
        )

        detected = bool(
            llr >= cfg.detect_llr
            and top_p > post[NULL_HYPOTHESIS]
            and top_p > post[LINE_SUPPLY_HYPOTHESIS]
            and persistent
        )
        confident = bool(detected and group_p >= cfg.confidence_floor)

        return ShadowResult(
            window=window,
            posterior=post,
            top_station=top,
            top_prob=top_p,
            group=group,
            group_prob=group_p,
            llr=llr,
            amp_starve=float(a_s[top]),
            amp_block=float(a_b[top]),
            top_is_hidden=not bool(self._obs_mask[top]),
            detected=detected,
            confident=confident,
            v_start=v_start,
            v_end=v_end,
            t_mid_s=t_mid_s,
            runner_up_station=runner_up,
            runner_up_prob=runner_up_p,
            evidence={
                "p_null": post[NULL_HYPOTHESIS],
                "p_line_supply": post[LINE_SUPPLY_HYPOTHESIS],
                "d_blocked": d_blocked.copy(),
                "d_starved": d_starved.copy(),
                "z_proc": z_proc.copy(),
                "station_post": station_post.copy(),
            },
        )

    def reset(self) -> None:
        """Clear temporal state. Call between independent runs."""
        self._ewma = None
        self._lead_history = []
        self.last_results = []

    def _unpack(self, grp: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.line.n_stations
        db = np.full(n, np.nan)
        ds = np.full(n, np.nan)
        zp = np.full(n, np.nan)
        st = grp["station"].to_numpy(dtype=int)
        db[st] = grp["d_blocked"].to_numpy(dtype=float)
        ds[st] = grp["d_starved"].to_numpy(dtype=float)
        zp[st] = grp["z_proc"].to_numpy(dtype=float)
        return db, ds, zp

    def run(
        self,
        scored_windows: pd.DataFrame,
        sigma_b: float,
        sigma_s: float,
        use_ewma: bool = True,
    ) -> pd.DataFrame:
        """Estimate hidden state for every window in a scored frame."""
        self.reset()
        rows = []
        for w, grp in scored_windows.groupby("window", sort=True):
            db, ds, zp = self._unpack(grp)
            t_mid = float((grp["t_depart_s_min"].min() + grp["t_depart_s_max"].max()) / 2.0)
            r = self.estimate_window(
                int(w), db, ds, zp, sigma_b, sigma_s,
                v_start=int(grp["v_start"].iloc[0]),
                v_end=int(grp["v_end"].iloc[0]),
                t_mid_s=t_mid,
                use_ewma=use_ewma,
            )
            self.last_results.append(r)
            rows.append(
                {
                    "window": r.window,
                    "v_start": r.v_start,
                    "v_end": r.v_end,
                    "t_mid_s": r.t_mid_s,
                    "top_station": r.top_station,
                    "top_station_id": self.line.stations[r.top_station].station_id,
                    "top_prob": r.top_prob,
                    "group_lo": min(r.group),
                    "group_hi": max(r.group),
                    "group_prob": r.group_prob,
                    "llr": r.llr,
                    "amp_starve": r.amp_starve,
                    "amp_block": r.amp_block,
                    "top_is_hidden": r.top_is_hidden,
                    "detected": r.detected,
                    "confident": r.confident,
                    "p_null": r.evidence["p_null"],
                    "p_line_supply": r.evidence["p_line_supply"],
                    "runner_up_station": r.runner_up_station,
                    "runner_up_prob": r.runner_up_prob,
                }
            )
        return pd.DataFrame(rows)

    # -------------------------------------------------------------- calibration

    def calibrate(
        self,
        nominal_scored: pd.DataFrame,
        sigma_b: float,
        sigma_s: float,
        target_window_fpr: float = 0.01,
        watch_target_fpr: float = 0.05,
    ) -> dict:
        """Set ``tau`` and ``detect_llr`` from disturbance-free reference data.

        Two things are calibrated here, and both are calibrated on data the
        evaluation never sees:

        ``tau``   -- the correlation correction. Stations on a coupled line move
                     together, so the naive independent-Gaussian likelihood
                     overstates the evidence. We estimate an effective sample
                     size from the observed cross-station correlation and scale
                     the flow log-likelihood by ``n_eff / n_observed``.

        ``detect_llr`` -- the detection threshold, read off the empirical null
                     distribution of the statistic at the requested per-window
                     false-alarm rate. Choosing it this way means the false-alarm
                     rate is a design parameter with a stated target rather than
                     an accident of a hand-picked constant.
        """
        piv = nominal_scored.pivot_table(index="window", columns="station", values="pressure")
        piv = piv.dropna(axis=1, how="any")
        # A station that never blocks or starves across the whole baseline
        # period has zero variance, and correlating it divides by zero. On the
        # simulator that essentially never happens; on a real line an isolated
        # or well-buffered station is entirely capable of it, so drop those
        # columns rather than let a NaN propagate into tau.
        if piv.shape[1]:
            sd = piv.std(axis=0)
            # A tolerance, not ``> 0``: a column can be constant to within
            # floating-point noise and still pass a strict test, then underflow
            # inside corrcoef's own standardisation.
            piv = piv.loc[:, sd > 1e-9 * np.maximum(1.0, piv.mean(axis=0).abs())]
        n_obs = piv.shape[1]
        if n_obs >= 3 and len(piv) >= 10:
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.corrcoef(piv.to_numpy().T)
            off = C[~np.eye(n_obs, dtype=bool)]
            rho = float(np.clip(np.nanmean(off), 0.0, 0.98))
            n_eff = n_obs / (1.0 + (n_obs - 1) * rho)
            self.cfg.tau = float(np.clip(n_eff / n_obs, 1e-3, 1.0))
        else:
            rho, n_eff = float("nan"), float("nan")

        # Rebuild with tau applied, then read the null distribution of the LLR.
        self.reset()
        llrs = []
        for w, grp in nominal_scored.groupby("window", sort=True):
            db, ds, zp = self._unpack(grp)
            r = self.estimate_window(int(w), db, ds, zp, sigma_b, sigma_s)
            llrs.append(r.llr)
        llrs = np.array(llrs, dtype=float)
        llrs = llrs[np.isfinite(llrs)]
        thr = float(np.quantile(llrs, 1.0 - target_window_fpr)) if llrs.size else 8.0
        self.cfg.detect_llr = max(thr, 1.0)

        # Watch threshold: the same statistic, read off the same null
        # distribution, at a looser (higher) target false-alarm rate than
        # detect_llr. Clamped strictly below detect_llr so WATCH can only ever
        # fire earlier than, never instead of, a confident detection -- the
        # quantiles already guarantee this in expectation, but the clamp holds
        # under the small-sample noise of an empirical quantile.
        watch_thr = (
            float(np.quantile(llrs, 1.0 - watch_target_fpr)) if llrs.size else 4.0
        )
        self.cfg.watch_llr = float(
            np.clip(min(watch_thr, self.cfg.detect_llr * 0.9), 0.25, self.cfg.detect_llr)
        )
        self.cfg.llr_noise_std = float(np.std(llrs)) if llrs.size >= 2 else 1.0
        self.reset()

        return {
            "n_observed": int(n_obs),
            "mean_pairwise_corr": rho,
            "n_eff": float(n_eff),
            "tau": self.cfg.tau,
            "detect_llr": self.cfg.detect_llr,
            "target_window_fpr": target_window_fpr,
            "watch_llr": self.cfg.watch_llr,
            "watch_target_fpr": watch_target_fpr,
            "llr_noise_std": self.cfg.llr_noise_std,
            "null_llr_median": float(np.median(llrs)) if llrs.size else float("nan"),
        }


# --------------------------------------------------- physical state read-out


def infer_hidden_cycle_time(
    line: LineTopology,
    telemetry: pd.DataFrame,
    station: int,
    v_start: int,
    v_end: int,
) -> Optional[float]:
    """Estimate the processing time of a station we cannot measure.

    When station ``k`` is the constraint, everything downstream of it is starved,
    so the departure rate at the first observed station downstream is set by k
    itself. The mean inter-departure interval there is therefore an estimate of
    k's processing time.

    This is the number a plant engineer will check first, and it is falsifiable:
    the simulator records the true processing time of hidden stations, so the
    error of this estimate is measured in the evaluation rather than asserted.

    The estimate is only valid while k is genuinely the constraint. When it is
    not, this returns a lower bound rather than an estimate, which is why the
    caller only surfaces it for a confidently-localised station.
    """
    downstream = line.nearest_observed_downstream(station, k=1)
    if not downstream:
        return None
    j = downstream[0]
    seg = telemetry[
        (telemetry["station"] == j)
        & (telemetry["vehicle_id"] >= v_start)
        & (telemetry["vehicle_id"] < v_end)
    ].sort_values("vehicle_id")
    if len(seg) < 4:
        return None
    gaps = np.diff(seg["t_depart_s"].to_numpy())
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if gaps.size < 3:
        return None
    # Trimmed mean: rework at inspection gates creates occasional long
    # departures that are not the constraint's doing.
    lo, hi = np.percentile(gaps, [5, 85])
    core = gaps[(gaps >= lo) & (gaps <= hi)]
    return float(np.mean(core)) if core.size else float(np.mean(gaps))
