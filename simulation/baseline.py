"""
Baseline deployment policies for comparison with Sky (CVaR–MILP).

* **SC (Single-Cloud):** all four logical ops use endpoints from one cloud provider;
  within that silo, each layer picks the candidate with maximum utility μ.
* **LO (Logical-Optimal):** each layer independently picks argmax μ (ignores cost,
  latency, cross-edge effects) — utility upper bound for the linear μ model.
* **DO (Deterministic-Optimal):** replace stochastic ξ with plug-in means of ρ via
  calibration sampling (not closed-form E[·]); midpoint segment/split execution from
  measured bounds, fixed LLM token path, and empirical mean network delay from trace
  samples per link; maximise Σ w_i μ_i subject to deterministic **mean** cost &
  latency ≤ each query’s SLO (linear program, no CVaR slacks).

Run from the ``simulation/`` directory (``python baseline.py``).
"""

from __future__ import annotations

from typing import NamedTuple

import random
import pulp as pl

from sim_env import config as cfg
from sim_env.cost import egress_cost_usd, llm_token_cost_usd, split_cost_usd, storage_cost_usd, video_service_cost_usd
from sim_env.execution_latency import llm_decode_duration_sec, segment_split_bounds_at
from sim_env.network import LinkCategory, classify_link, reset_link_counters, sample_link
from sim_env.utility import PhysicalNode, QueryProfile, physical_node_utility

import sky as sky_mod
import utils as wf_utils

OPS = sky_mod.OPS_ORDER
PROVIDERS_SINGLE_CLOUD = ("GCP", "AWS", "Aliyun")

# Empirical network mean: samples per directed edge (coefficient build).
_NET_MEAN_SAMPLES = 64


class BaselineResult(NamedTuple):
    name: str
    nodes: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode]
    total_utility: float
    pulp_status: str | None


def _mean_network_transfer_and_rtt(
    src: tuple[str, str],
    dst: tuple[str, str],
    xfer_gb: float,
    n: int = _NET_MEAN_SAMPLES,
) -> tuple[float, float]:
    """
    E[transfer_seconds] + E[RTT/2] for (src,dst) using trace means in ``sample_link`` pool.
    """
    cat = classify_link(src, dst)
    if cat == LinkCategory.NONE:
        return 0.0, 0.0

    reset_link_counters([(src, dst)])
    tsf = 0.0
    rtt_h = 0.0
    for _ in range(n):
        sm = sample_link(src, dst)
        tsf += wf_utils._transfer_seconds(xfer_gb, sm)
        rtt_h += 0.5 * (sm.rtt_ms / 1000.0)
    return tsf / float(n), rtt_h / float(n)


def _edge_mean_cost_latency(
    na: PhysicalNode,
    nb: PhysicalNode,
    xfer_gb: float,
) -> tuple[float, float]:
    ep_s, ep_d = sky_mod._endpoint(na), sky_mod._endpoint(nb)
    cents = egress_cost_usd(ep_s, ep_d, xfer_gb)
    t_tr, t_rtt = _mean_network_transfer_and_rtt(ep_s, ep_d, xfer_gb)
    return cents, t_tr + t_rtt


def _deterministic_local_cost_latency(
    i: int,
    node: PhysicalNode,
    s_in: list[float],
    rho: tuple[float, float, float, float],
    seg_minutes: float,
    token_seed: int,
) -> tuple[float, float]:
    rho_i = float(rho[i])
    stor = storage_cost_usd(
        node.provider, node.region, float(s_in[i]) * (1.0 + rho_i), days=1.0
    )
    op = OPS[i]
    dur_sec = max(seg_minutes * 60.0, 1e-6)
    bd = segment_split_bounds_at(dur_sec)

    if op == "segment":
        exe = video_service_cost_usd(node.provider, node.region, "segment", seg_minutes)
        t_exe = 0.5 * (bd.segment_min_sec + bd.segment_max_sec)
        return exe + stor, t_exe

    if op == "split":
        exe = split_cost_usd(node.provider, node.region, minutes=1.0)
        t_exe = 0.5 * (bd.split_min_sec + bd.split_max_sec)
        return exe + stor, t_exe

    cap_pair, q_pair = wf_utils._resolve_llm_tokens(
        seg_minutes,
        caption_tokens=None,
        query_tokens=None,
        rng_seed=token_seed,
    )
    cin, cout = cap_pair
    qin, qout = q_pair

    if op == "caption":
        mm = node.model or ""
        exe = llm_token_cost_usd(node.provider, node.region, mm, cin, cout)
        t_exe = llm_decode_duration_sec(mm, cout)
        return exe + stor, t_exe

    mm = node.model or ""
    exe = llm_token_cost_usd(node.provider, node.region, mm, qin, qout)
    t_exe = llm_decode_duration_sec(mm, qout)
    return exe + stor, t_exe


def _prepare_deterministic_coefficients(
    cands: tuple[tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...]],
    queries: list[QueryProfile],
    mean_rho: tuple[float, float, float, float],
    token_seed: int,
) -> tuple[
    list[list[list[float]]],
    list[list[list[list[float]]]],
    list[list[list[float]]],
    list[list[list[list[float]]]],
]:
    """Per query index q: same shape as ``sky_mod.prepare_coefficients``."""
    a_c: list[list[list[float]]] = []
    b_c: list[list[list[list[float]]]] = []
    a_l: list[list[list[float]]] = []
    b_l: list[list[list[list[float]]]] = []

    for q_idx, qprof in enumerate(queries):
        s_src = qprof.s_src_gb
        sn, xfer = wf_utils._propagate_sizes_gb(s_src, mean_rho)
        seg_min = sky_mod._segment_minutes(sn[0])
        seed = token_seed + q_idx * 17

        ac: list[list[float]] = []
        al: list[list[float]] = []
        for i in range(4):
            rc, rl = [], []
            for ki, node in enumerate(cands[i]):
                c_ij, lat_ij = _deterministic_local_cost_latency(
                    i, node, sn, mean_rho, seg_min, seed + ki
                )
                rc.append(c_ij)
                rl.append(lat_ij)
            ac.append(rc)
            al.append(rl)

        bc_e: list[list[list[float]]] = []
        bl_e: list[list[list[float]]] = []
        for ei in range(3):
            xg = xfer[ei]
            mat_c: list[list[float]] = []
            mat_l: list[list[float]] = []
            for ka, na in enumerate(cands[ei]):
                row_c = []
                row_l = []
                for kb, nb in enumerate(cands[ei + 1]):
                    ec, el = _edge_mean_cost_latency(na, nb, xg)
                    row_c.append(ec)
                    row_l.append(el)
                mat_c.append(row_c)
                mat_l.append(row_l)
            bc_e.append(mat_c)
            bl_e.append(mat_l)

        a_c.append(ac)
        b_c.append(bc_e)
        a_l.append(al)
        b_l.append(bl_e)

    return a_c, b_c, a_l, b_l


def logical_optimal_baseline(
    cands: tuple[tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...]],
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> BaselineResult:
    """LO: independent argmax μ per layer (within full candidate lists)."""
    picks: list[PhysicalNode] = []
    u_tot = 0.0
    for i in range(4):
        best_j = max(
            range(len(cands[i])),
            key=lambda j: weights[i] * physical_node_utility(cands[i][j]),
        )
        picks.append(cands[i][best_j])
        u_tot += weights[i] * physical_node_utility(cands[i][best_j])
    return BaselineResult(
        "Logical-Optimal",
        tuple(picks),  # type: ignore[arg-type]
        float(u_tot),
        None,
    )


def single_cloud_baseline(
    cands: tuple[tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...]],
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> BaselineResult:
    """
    SC: choose cloud provider P such that all four layers admit candidates on P,
    then per-layer argmax μ within U_i ∩ {P}.
    """
    best: BaselineResult | None = None
    for prov in PROVIDERS_SINGLE_CLOUD:
        filt: list[tuple[PhysicalNode, ...]] = []
        ok = True
        for i in range(4):
            layer = tuple(n for n in cands[i] if n.provider == prov)
            if not layer:
                ok = False
                break
            filt.append(layer)
        if not ok:
            continue
        ft = (
            tuple(filt[0]),
            tuple(filt[1]),
            tuple(filt[2]),
            tuple(filt[3]),
        )
        res = logical_optimal_baseline(ft, weights=weights)
        if best is None or res.total_utility > best.total_utility:
            best = BaselineResult("Single-Cloud", res.nodes, res.total_utility, None)

    if best is None:
        raise RuntimeError(
            "No single provider supports all four operations (check WORKFLOW_OPERATIONS)."
        )
    return best


def deterministic_optimal_baseline(
    queries: list[QueryProfile],
    cands: tuple[tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...]],
    *,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    token_seed: int = 0,
    time_limit_sec: int | None = None,
    rho_calibration_samples: int = 8192,
    rho_calibration_seed: int = 4242,
) -> BaselineResult:
    """
    DO: LP max Σ w_i μ_i x_i s.t. mean cost & mean latency ≤ SLO per query.

    Mean-field ``mean_rho`` is a plug-in average from ``rho_calibration_samples`` i.i.d.
    draws (not closed-form E[·] from hidden lognormal parameters).
    """
    calib_rng = random.Random(rho_calibration_seed)
    mean_rho = cfg.plugin_mean_data_conversion_ratios(
        n_calibration_samples=rho_calibration_samples,
        rng=calib_rng,
        operations=OPS,
    )
    a_cost, b_cost, a_lat, b_lat = _prepare_deterministic_coefficients(
        cands, queries, mean_rho, token_seed
    )

    dims = tuple(len(cands[i]) for i in range(4))
    prob = pl.LpProblem("baseline_do_lp", pl.LpMaximize)

    x = [
        [pl.LpVariable(f"do_x_{i}_{k}", cat=pl.LpBinary) for k in range(dims[i])]
        for i in range(4)
    ]
    y_edge: list[list[list[pl.LpVariable]]] = []
    for ei in range(3):
        mat = [
            [
                pl.LpVariable(f"do_y_{ei}_{ka}_{kb}", lowBound=0, upBound=1)
                for kb in range(dims[ei + 1])
            ]
            for ka in range(dims[ei])
        ]
        y_edge.append(mat)

    for i in range(4):
        prob += pl.lpSum(x[i][kk] for kk in range(dims[i])) == 1

    for ei in range(3):
        for ka in range(dims[ei]):
            for kb in range(dims[ei + 1]):
                yv = y_edge[ei][ka][kb]
                prob += yv <= x[ei][ka]
                prob += yv <= x[ei + 1][kb]
                prob += yv >= x[ei][ka] + x[ei + 1][kb] - 1

    util_terms: list = []
    for i in range(4):
        for k in range(dims[i]):
            util_terms.append(
                weights[i] * physical_node_utility(cands[i][k]) * x[i][k]
            )

    nq = len(queries)
    for q_idx in range(nq):
        qprof = queries[q_idx]
        lc_terms: list = []
        for i in range(4):
            for k in range(dims[i]):
                lc_terms.append(a_cost[q_idx][i][k] * x[i][k])
        for ei in range(3):
            for ka in range(dims[ei]):
                for kb in range(dims[ei + 1]):
                    lc_terms.append(b_cost[q_idx][ei][ka][kb] * y_edge[ei][ka][kb])
        lt_terms: list = []
        for i in range(4):
            for k in range(dims[i]):
                lt_terms.append(a_lat[q_idx][i][k] * x[i][k])
        for ei in range(3):
            for ka in range(dims[ei]):
                for kb in range(dims[ei + 1]):
                    lt_terms.append(b_lat[q_idx][ei][ka][kb] * y_edge[ei][ka][kb])

        prob += pl.lpSum(lc_terms) <= qprof.theta_cost + 1e-9
        prob += pl.lpSum(lt_terms) <= qprof.theta_latency_sec + 1e-9

    prob += pl.lpSum(util_terms)

    solver = pl.PULP_CBC_CMD(msg=0, timeLimit=time_limit_sec if time_limit_sec else None)
    prob.solve(solver)
    status_str = str(pl.LpStatus[prob.status])

    if status_str != "Optimal":
        placeholder = tuple(cands[i][0] for i in range(4))
        return BaselineResult(
            "Deterministic-Optimal",
            placeholder,
            float("nan"),
            status_str,
        )

    picks: list[int] = []
    for i in range(4):
        vals = [(k, float(pl.value(v) or 0.0)) for k, v in enumerate(x[i])]
        ks = sorted(vals, key=lambda t: t[1], reverse=True)[0][0]
        picks.append(int(ks))

    chosen = tuple(cands[i][picks[i]] for i in range(4))
    util_obj = pl.value(prob.objective)
    util_val = float(util_obj if util_obj is not None else 0.0)

    return BaselineResult(
        "Deterministic-Optimal",
        chosen,
        util_val,
        status_str,
    )


def run_all_baselines(
    queries: list[QueryProfile],
    *,
    max_per_op: int | None = None,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    do_token_seed: int = 0,
) -> tuple[BaselineResult, BaselineResult, BaselineResult]:
    """Enumerate candidates once; return SC, LO, DO results."""
    cands = sky_mod.enumerate_candidates(max_per_op=max_per_op)
    sc = single_cloud_baseline(cands, weights=weights)
    lo = logical_optimal_baseline(cands, weights=weights)
    do = deterministic_optimal_baseline(
        queries, cands, weights=weights, token_seed=do_token_seed
    )
    return sc, lo, do


if __name__ == "__main__":  # pragma: no cover
    import utils as wf_utils

    from evaluation import evaluate_deployment_empirical, print_metrics_report

    n_q = 5
    qs = wf_utils.generate_realistic_queries(n_q, seed=7)
    cands = sky_mod.enumerate_candidates(max_per_op=4)
    WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    EVAL_SQ = 30
    EVAL_SEED = 2026

    print("Candidates per layer:", [len(c) for c in cands])
    lo = logical_optimal_baseline(cands, weights=WEIGHTS)
    sc = single_cloud_baseline(cands, weights=WEIGHTS)
    print(lo.name, "closed-form U=", lo.total_utility)
    print(sc.name, "providers", {n.provider for n in sc.nodes}, "closed-form U=", sc.total_utility)

    for name_prefix, bp in [(lo.name, lo), (sc.name, sc)]:
        m = evaluate_deployment_empirical(
            bp.nodes, qs, weights=WEIGHTS, samples_per_query=EVAL_SQ, eval_seed=EVAL_SEED
        )
        print_metrics_report(
            algorithm_label=f"{name_prefix}",
            metrics=m,
        )

    do = deterministic_optimal_baseline(qs, cands, weights=WEIGHTS)
    print(
        do.name,
        "MILP_status",
        do.pulp_status,
        "MILP_U",
        do.total_utility,
        "nodes",
        do.nodes,
    )
    md = evaluate_deployment_empirical(
        do.nodes, qs, weights=WEIGHTS, samples_per_query=EVAL_SQ, eval_seed=EVAL_SEED + 1
    )
    print_metrics_report(algorithm_label=f"{do.name} | MILP={do.pulp_status}", metrics=md)
