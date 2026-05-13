"""
Baseline deployment policies for comparison with Sky (CVaR–MILP).

* **SC (Single-Cloud):** all four logical ops use endpoints from one cloud provider;
  within that silo, each layer picks argmax μ。若在调用时传入 ``queries``（及 MC 参数），
  则在可行提供商之间 **优先最小化** 与 ``evaluation.evaluate_deployment_empirical``
  一致的蒙特卡洛 SLO 违规计数（费用违规 + 时延违规），再以 closed-form utility 为高者优先。
* **LO (Logical-Optimal):** each layer independently picks argmax μ (ignores cost,
  latency, cross-edge effects) — utility upper bound for the linear μ model.
* **DO (Deterministic-Optimal):** replace stochastic ξ with plug-in means of ρ via
  calibration sampling (not closed-form E[·]); midpoint segment/split execution from
  measured bounds, empirical mean LLM output tokens, empirical mean network delay from trace
  samples per link; maximise Σ w_i μ_i subject to deterministic **mean** cost &
  latency ≤ each query’s SLO (**MILP** via **gurobipy**, no CVaR slacks).

From ``simulation/``: ``python -m workflow1.baseline``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import random

import gurobipy as gp
from gurobipy import GRB

from sim_env import config as cfg
from sim_env.cost import egress_cost_usd, llm_token_cost_usd, split_cost_usd, storage_cost_usd, video_service_cost_usd
from sim_env.execution_latency import (
    llm_decode_duration_sec,
    node_segment_execute_bounds_at,
    node_split_execute_bounds_at,
    sample_execution_scale,
)
from sim_env.network import LinkCategory, classify_link, reset_link_counters, sample_link
from sim_env.utility import PhysicalNode, QueryProfile, physical_node_utility

from . import sky as sky_mod
from . import utils as wf_utils

OPS = sky_mod.OPS_ORDER
_OPS_MC = ("segment", "split", "caption", "query")
PROVIDERS_SINGLE_CLOUD = ("GCP", "AWS", "Aliyun")

# Empirical network mean: samples per directed edge (coefficient build).
_NET_MEAN_SAMPLES = 64
# Plug-in mean for LLM output tokens (caption / query), same spirit as ρ calibration.
_LLM_TOKEN_MEAN_SAMPLES = 512


class BaselineResult(NamedTuple):
    name: str
    nodes: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode]
    total_utility: float
    gurobi_status: str | None


def _mean_network_transfer_and_rtt(
    src: tuple[str, str],
    dst: tuple[str, str],
    xfer_gb: float,
    n: int = _NET_MEAN_SAMPLES,
    *,
    token_seed: int,
    q_idx: int,
    ei: int,
    ka: int,
    kb: int,
) -> tuple[float, float]:
    """
    Empirical mean of transfer time + RTT/2 for (src,dst) from trace pools.

    Each Monte Carlo draw uses a deterministic ``rng`` for ``sample_link`` rotation.
    """
    cat = classify_link(src, dst)
    if cat == LinkCategory.NONE:
        return 0.0, 0.0

    reset_link_counters([(src, dst)])
    tsf = 0.0
    rtt_h = 0.0
    for si in range(n):
        sm = sample_link(
            src,
            dst,
            rng=wf_utils.det_rng(
                token_seed,
                "do_net",
                q_idx,
                ei,
                ka,
                kb,
                src[0],
                src[1],
                dst[0],
                dst[1],
                si,
            ),
        )
        tsf += wf_utils._transfer_seconds(xfer_gb, sm)
        rtt_h += 0.5 * (sm.rtt_ms / 1000.0)
    return tsf / float(n), rtt_h / float(n)


def _edge_mean_cost_latency(
    na: PhysicalNode,
    nb: PhysicalNode,
    xfer_gb: float,
    *,
    token_seed: int,
    q_idx: int,
    ei: int,
    ka: int,
    kb: int,
) -> tuple[float, float]:
    ep_s, ep_d = sky_mod._endpoint(na), sky_mod._endpoint(nb)
    cents = egress_cost_usd(ep_s, ep_d, xfer_gb)
    t_tr, t_rtt = _mean_network_transfer_and_rtt(
        ep_s,
        ep_d,
        xfer_gb,
        token_seed=token_seed,
        q_idx=q_idx,
        ei=ei,
        ka=ka,
        kb=kb,
    )
    return cents, t_tr + t_rtt


def _local_edge_mean_cost_latency(
    cloud_ep: tuple[str, str],
    payload_gb: float,
    *,
    direction: str,
    token_seed: int,
    q_idx: int,
    ei: int,
    ka: int,
) -> tuple[float, float]:
    from sim_env.network import LOCAL_PROVIDER, LOCAL_REGION

    local_ep = (LOCAL_PROVIDER, LOCAL_REGION)
    if direction == "upload":
        ep_s, ep_d = local_ep, cloud_ep
    else:
        ep_s, ep_d = cloud_ep, local_ep
    cents = egress_cost_usd(ep_s, ep_d, payload_gb)
    t_tr, t_rtt = _mean_network_transfer_and_rtt(
        ep_s,
        ep_d,
        payload_gb,
        token_seed=token_seed,
        q_idx=q_idx,
        ei=ei,
        ka=ka,
        kb=0,
    )
    return cents, t_tr + t_rtt


def _deterministic_local_cost_latency(
    i: int,
    node: PhysicalNode,
    s_in: list[float],
    rho: tuple[float, float, float, float],
    seg_minutes: float,
    *,
    cap_pair: tuple[float, float],
    q_pair: tuple[float, float],
    exec_scale_rng: random.Random,
    llm_latency_rng: random.Random | None = None,
) -> tuple[float, float]:
    rho_i = float(rho[i])
    stor = storage_cost_usd(
        node.provider, node.region, float(s_in[i]) * (1.0 + rho_i), days=1.0
    )
    op = OPS[i]
    dur_sec = max(seg_minutes * 60.0, 1e-6)

    if op == "segment":
        exe = video_service_cost_usd(node.provider, node.region, "segment", seg_minutes)
        lo, hi = node_segment_execute_bounds_at(dur_sec, node.provider, node.region)
        k = sample_execution_scale(exec_scale_rng)
        t_exe = 0.5 * (lo * k + hi * k)
        return exe + stor, t_exe

    if op == "split":
        exe = split_cost_usd(node.provider, node.region, minutes=1.0)
        lo, hi = node_split_execute_bounds_at(dur_sec, node.provider, node.region)
        k = sample_execution_scale(exec_scale_rng)
        t_exe = 0.5 * (lo * k + hi * k)
        return exe + stor, t_exe

    cin, cout = cap_pair
    qin, qout = q_pair

    if op == "caption":
        mm = node.model or ""
        exe = llm_token_cost_usd(node.provider, node.region, mm, cin, cout)
        t_exe = llm_decode_duration_sec(mm, cout, rng=llm_latency_rng)
        return exe + stor, t_exe

    mm = node.model or ""
    exe = llm_token_cost_usd(node.provider, node.region, mm, qin, qout)
    t_exe = llm_decode_duration_sec(mm, qout, rng=llm_latency_rng)
    return exe + stor, t_exe


def _prepare_deterministic_coefficients(
    cands: tuple[tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...]],
    queries: list[QueryProfile],
    mean_rho: tuple[float, float, float, float],
    token_seed: int,
    *,
    llm_token_mean_samples: int = _LLM_TOKEN_MEAN_SAMPLES,
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
        cap_pair, q_pair = wf_utils._resolve_llm_tokens(
            seg_min,
            caption_tokens=None,
            query_tokens=None,
            rng_seed=seed,
            caption_output_payload_gb=float(sn[2]) * float(mean_rho[2]),
            query_output_payload_gb=float(sn[3]) * float(mean_rho[3]),
        )

        ac: list[list[float]] = []
        al: list[list[float]] = []
        for i in range(4):
            rc, rl = [], []
            for ki, node in enumerate(cands[i]):
                exec_scale_rng = wf_utils.det_rng(
                    token_seed, "do_exec_scale", q_idx, i, ki,
                    node.provider, node.region,
                )
                llm_latency_rng = None
                if i in (2, 3):
                    llm_latency_rng = wf_utils.det_rng(
                        token_seed,
                        "do_llm_jit",
                        q_idx,
                        i,
                        ki,
                        node.provider,
                        node.region,
                        node.model,
                    )
                c_ij, lat_ij = _deterministic_local_cost_latency(
                    i,
                    node,
                    sn,
                    mean_rho,
                    seg_min,
                    cap_pair=cap_pair,
                    q_pair=q_pair,
                    exec_scale_rng=exec_scale_rng,
                    llm_latency_rng=llm_latency_rng,
                )
                if i == 0:
                    lep = sky_mod._endpoint(node)
                    luc, lul = _local_edge_mean_cost_latency(
                        lep,
                        float(s_src),
                        direction="upload",
                        token_seed=token_seed,
                        q_idx=q_idx,
                        ei=40,
                        ka=ki,
                    )
                    c_ij += luc
                    lat_ij += lul
                elif i == 3:
                    lep = sky_mod._endpoint(node)
                    duc, dul = _local_edge_mean_cost_latency(
                        lep,
                        float(xfer[3]),
                        direction="download",
                        token_seed=token_seed,
                        q_idx=q_idx,
                        ei=43,
                        ka=ki,
                    )
                    c_ij += duc
                    lat_ij += dul
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
                    ec, el = _edge_mean_cost_latency(
                        na,
                        nb,
                        xg,
                        token_seed=token_seed,
                        q_idx=q_idx,
                        ei=ei,
                        ka=ka,
                        kb=kb,
                    )
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
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
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


def mc_violation_counts_wf1(
    nodes: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode],
    queries: Sequence[QueryProfile],
    *,
    samples_per_query: int,
    violation_eval_seed: int,
    cost_tol_abs: float = 1e-9,
    latency_tol_abs: float = 1e-9,
) -> tuple[int, int]:
    """与 ``evaluation.evaluate_deployment_empirical`` 相同 RNG 标签与判定，返回 (viol_cost, viol_lat)。"""
    viol_cost = 0
    viol_lat = 0
    ql = list(queries)
    for q_idx, qprof in enumerate(ql):
        for s_idx in range(samples_per_query):
            rho_rng = wf_utils.det_rng(violation_eval_seed, "eval_rho", q_idx, s_idx)
            wf_rng = wf_utils.det_rng(violation_eval_seed, "eval_mc", q_idx, s_idx)
            rho = tuple(cfg.sample_data_conversion_ratio(op, rho_rng) for op in _OPS_MC)
            c, ell = wf_utils.end_to_end_cost_and_latency(
                nodes, qprof.s_src_gb, rho, workflow_rng=wf_rng
            )
            if c > qprof.theta_cost + cost_tol_abs:
                viol_cost += 1
            if ell > qprof.theta_latency_sec + latency_tol_abs:
                viol_lat += 1
    return viol_cost, viol_lat


def single_cloud_baseline(
    cands: tuple[tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...]],
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    *,
    queries: Sequence[QueryProfile] | None = None,
    samples_per_query: int = 50,
    violation_eval_seed: int = 137,
) -> BaselineResult:
    """
    SC: choose cloud provider P such that all four layers admit candidates on P,
    then per-layer argmax μ within U_i ∩ {P}。

    若 ``queries`` 非空：在可行 P 上先最小化 MC 违规计数
    ``viol_cost + viol_latency``（与 empirical evaluation 一致），再最大化 closed-form utility。
    """
    ql = list(queries) if queries is not None else []
    use_viol = len(ql) > 0

    best: BaselineResult | None = None
    best_rank: tuple[int, float, str] | None = None  # (viol_sum, -utility, prov) 用于最小化

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
        if use_viol:
            vc, vl = mc_violation_counts_wf1(
                res.nodes,
                ql,
                samples_per_query=samples_per_query,
                violation_eval_seed=violation_eval_seed,
            )
            rank = (vc + vl, -res.total_utility, prov)
            if best is None or rank < best_rank:  # type: ignore[operator]
                best_rank = rank
                best = BaselineResult("Single-Cloud", res.nodes, res.total_utility, None)
        else:
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
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    token_seed: int = 0,
    time_limit_sec: int | None = None,
    rho_calibration_samples: int = 128,
    rho_calibration_seed: int = 4242,
    llm_token_mean_samples: int = _LLM_TOKEN_MEAN_SAMPLES,
) -> BaselineResult:
    """
    DO: MILP max Σ w_i μ_i x_i s.t. mean cost & mean latency ≤ SLO per query (gurobipy).

    Mean-field ``mean_rho`` is a plug-in average from ``rho_calibration_samples`` i.i.d.
    draws (not closed-form E[·] from hidden lognormal parameters).
    LLM output tokens use ``llm_token_mean_samples`` draws averaged per query (plug-in mean).
    """
    calib_rng = random.Random(rho_calibration_seed)
    mean_rho = cfg.plugin_mean_data_conversion_ratios(
        n_calibration_samples=rho_calibration_samples,
        rng=calib_rng,
        operations=OPS,
    )
    a_cost, b_cost, a_lat, b_lat = _prepare_deterministic_coefficients(
        cands,
        queries,
        mean_rho,
        token_seed,
        llm_token_mean_samples=llm_token_mean_samples,
    )

    dims = tuple(len(cands[i]) for i in range(4))
    m = gp.Model("baseline_do_lp")
    m.Params.OutputFlag = 0
    if time_limit_sec is not None:
        m.Params.TimeLimit = float(time_limit_sec)

    x = [
        [m.addVar(vtype=GRB.BINARY, name=f"do_x_{i}_{k}") for k in range(dims[i])]
        for i in range(4)
    ]
    y_edge: list[list[list[gp.Var]]] = []
    for ei in range(3):
        mat = [
            [
                m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"do_y_{ei}_{ka}_{kb}")
                for kb in range(dims[ei + 1])
            ]
            for ka in range(dims[ei])
        ]
        y_edge.append(mat)

    for i in range(4):
        m.addConstr(gp.quicksum(x[i][kk] for kk in range(dims[i])) == 1)

    for ei in range(3):
        for ka in range(dims[ei]):
            for kb in range(dims[ei + 1]):
                yv = y_edge[ei][ka][kb]
                m.addConstr(yv <= x[ei][ka])
                m.addConstr(yv <= x[ei + 1][kb])
                m.addConstr(yv >= x[ei][ka] + x[ei + 1][kb] - 1)

    util_expr = gp.quicksum(
        weights[i] * physical_node_utility(cands[i][k]) * x[i][k]
        for i in range(4)
        for k in range(dims[i])
    )

    nq = len(queries)
    for q_idx in range(nq):
        qprof = queries[q_idx]
        lc_expr = gp.quicksum(
            a_cost[q_idx][i][k] * x[i][k] for i in range(4) for k in range(dims[i])
        ) + gp.quicksum(
            b_cost[q_idx][ei][ka][kb] * y_edge[ei][ka][kb]
            for ei in range(3)
            for ka in range(dims[ei])
            for kb in range(dims[ei + 1])
        )
        lt_expr = gp.quicksum(
            a_lat[q_idx][i][k] * x[i][k] for i in range(4) for k in range(dims[i])
        ) + gp.quicksum(
            b_lat[q_idx][ei][ka][kb] * y_edge[ei][ka][kb]
            for ei in range(3)
            for ka in range(dims[ei])
            for kb in range(dims[ei + 1])
        )
        m.addConstr(lc_expr <= qprof.theta_cost + 1e-9)
        m.addConstr(lt_expr <= qprof.theta_latency_sec + 1e-9)

    m.setObjective(util_expr, GRB.MAXIMIZE)
    m.optimize()
    status_str = sky_mod._gurobi_status_str(m)

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
        vals = [(k, sky_mod._safe_var_x(v)) for k, v in enumerate(x[i])]
        ks = sorted(vals, key=lambda t: t[1], reverse=True)[0][0]
        picks.append(int(ks))

    chosen = tuple(cands[i][picks[i]] for i in range(4))
    util_val = float(m.ObjVal)

    return BaselineResult(
        "Deterministic-Optimal",
        chosen,
        util_val,
        status_str,
    )


def run_all_baselines(
    queries: list[QueryProfile],
    *,
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    do_token_seed: int = 0,
    sc_samples_per_query: int = 50,
    sc_violation_eval_seed: int = 137,
) -> tuple[BaselineResult, BaselineResult, BaselineResult]:
    """Enumerate candidates once; return SC, LO, DO results."""
    cands = sky_mod.enumerate_candidates()
    sc = single_cloud_baseline(
        cands,
        weights=weights,
        queries=queries,
        samples_per_query=sc_samples_per_query,
        violation_eval_seed=sc_violation_eval_seed,
    )
    lo = logical_optimal_baseline(cands, weights=weights)
    do = deterministic_optimal_baseline(
        queries, cands, weights=weights, token_seed=do_token_seed
    )
    return sc, lo, do


if __name__ == "__main__":  # pragma: no cover
    from . import utils as wf_utils
    from .evaluation import evaluate_deployment_empirical, print_metrics_report

    n_q = 5
    qs = wf_utils.generate_realistic_queries(n_q, seed=7)
    cands = sky_mod.enumerate_candidates()
    WEIGHTS = (0.25, 0.25, 0.25, 0.25)
    EVAL_SQ = 30
    EVAL_SEED = 2026

    print("Candidates per layer:", [len(c) for c in cands])
    lo = logical_optimal_baseline(cands, weights=WEIGHTS)
    sc = single_cloud_baseline(
        cands,
        weights=WEIGHTS,
        queries=qs,
        samples_per_query=EVAL_SQ,
        violation_eval_seed=EVAL_SEED,
    )
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
        "Gurobi_status",
        do.gurobi_status,
        "MILP_U",
        do.total_utility,
        "nodes",
        do.nodes,
    )
    md = evaluate_deployment_empirical(
        do.nodes, qs, weights=WEIGHTS, samples_per_query=EVAL_SQ, eval_seed=EVAL_SEED
    )
    print_metrics_report(algorithm_label=f"{do.name} | Gurobi={do.gurobi_status}", metrics=md)
