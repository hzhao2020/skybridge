"""
Workflow 2 baselines（对齐 workflow1）：SC / LO / DO。

SC：单提供商内各层在 ``provider`` 候选上独立 argmax μ（**不**强制同 region）；``queries`` 与 MC 参数必填；
在可行云之间 **优先最小化** ``evaluation.evaluate_deployment_empirical_wf2`` 的 MC SLO 违规，再以 utility 为高者优先。

DO：plug-in mean ρ（沿路径蒙特卡洛校准）、LLM token 经验均值、链路网络经验均值；
目标 max Σ w_i μ_i，约束为每条查询在均值场上的费用与时延不超过 Θ（**gurobipy MILP**）。

运行：``python -m workflow2.baseline``。
"""

from __future__ import annotations

import random
from typing import NamedTuple, Sequence

import gurobipy as gp
from gurobipy import GRB

from sim_env.cost import (
    database_instance_cost_usd,
    database_storage_cost_usd,
    egress_cost_usd,
    llm_token_cost_usd,
    split_cost_usd,
    storage_cost_usd,
    video_service_cost_usd,
)
from sim_env.execution_latency import (
    llm_decode_duration_sec,
    node_database_query_execute_bounds_at,
    node_label_detection_execute_bounds_at,
    node_ocr_execute_bounds_at,
    node_shot_detection_execute_bounds_at,
    node_speech_transcription_execute_bounds_at,
    node_video_split_execute_bounds_at,
    sample_execution_scale,
)
from sim_env.network import LinkCategory, classify_link, reset_link_counters, sample_link
from sim_env.utility import QueryProfile

from workflow1 import utils as wf1_utils

from . import sky as wf2_sky
from . import utils as wf2_utils
from .candidates import enumerate_candidates_wf2
from .utils import WF2PathId, WF2PhysicalNode, path_logical_ops, wf2_node_utility

PROVIDERS_SINGLE_CLOUD = ("GCP", "AWS", "Aliyun")

_NET_MEAN_SAMPLES = 64
_LLM_TOKEN_MEAN_SAMPLES = 512


class BaselineResultWf2(NamedTuple):
    name: str
    nodes: tuple[WF2PhysicalNode, ...]
    total_utility: float
    gurobi_status: str | None


def _mean_network_transfer_and_rtt_wf2(
    src: tuple[str, str],
    dst: tuple[str, str],
    xfer_gb: float,
    n: int,
    *,
    token_seed: int,
    q_idx: int,
    ei: int,
    ka: int,
    kb: int,
) -> tuple[float, float]:
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
            rng=wf1_utils.det_rng(
                token_seed,
                "wf2_do_net",
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
        tsf += wf1_utils._transfer_seconds(xfer_gb, sm)
        rtt_h += 0.5 * (sm.rtt_ms / 1000.0)
    return tsf / float(n), rtt_h / float(n)


def _edge_mean_cost_latency_wf2(
    na: WF2PhysicalNode,
    nb: WF2PhysicalNode,
    xfer_gb: float,
    *,
    token_seed: int,
    q_idx: int,
    ei: int,
    ka: int,
    kb: int,
) -> tuple[float, float]:
    ep_s = (na.provider, na.region)
    ep_d = (nb.provider, nb.region)
    cents = egress_cost_usd(ep_s, ep_d, xfer_gb)
    t_tr, t_rtt = _mean_network_transfer_and_rtt_wf2(
        ep_s,
        ep_d,
        xfer_gb,
        _NET_MEAN_SAMPLES,
        token_seed=token_seed,
        q_idx=q_idx,
        ei=ei,
        ka=ka,
        kb=kb,
    )
    return cents, t_tr + t_rtt


def _local_edge_mean_cost_latency_wf2(
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
    t_tr, t_rtt = _mean_network_transfer_and_rtt_wf2(
        ep_s,
        ep_d,
        payload_gb,
        _NET_MEAN_SAMPLES,
        token_seed=token_seed,
        q_idx=q_idx,
        ei=ei,
        ka=ka,
        kb=0,
    )
    return cents, t_tr + t_rtt


def _deterministic_local_cost_latency_wf2(
    logical_op: str,
    node: WF2PhysicalNode,
    s_in: list[float],
    rho: tuple[float, ...],
    idx: int,
    seg_minutes: float,
    *,
    cap_pair: tuple[float, float],
    q_pair: tuple[float, float],
    exec_scale_rng: random.Random,
    llm_latency_rng: random.Random | None = None,
    ops_full: tuple[str, ...],
) -> tuple[float, float]:
    rho_i = float(rho[idx])
    gb_local = float(s_in[idx]) * (1.0 + rho_i)
    if logical_op == "database":
        stor = database_storage_cost_usd(node.provider, node.region, gb_local, days=1.0)
    else:
        stor = storage_cost_usd(node.provider, node.region, gb_local, days=1.0)
    p, r = node.provider, node.region
    dur_sec = max(seg_minutes * 60.0, 1e-6)
    cin, cout = cap_pair
    qin, qout = q_pair
    k = sample_execution_scale(exec_scale_rng)

    if logical_op == "shot_detection":
        exe = video_service_cost_usd(p, r, "shot_detection", seg_minutes)
        lo, hi = node_shot_detection_execute_bounds_at(dur_sec, p, r)
        t_exe = 0.5 * (lo * k + hi * k)
        return exe + stor, t_exe

    if logical_op == "video_split":
        exe = split_cost_usd(p, r, minutes=1.0)
        lo, hi = node_video_split_execute_bounds_at(dur_sec, p, r)
        t_exe = 0.5 * (lo * k + hi * k)
        return exe + stor, t_exe

    if logical_op == "video_caption":
        mm = node.model or ""
        exe = llm_token_cost_usd(p, r, mm, cin, cout)
        t_exe = llm_decode_duration_sec(mm, cout, rng=llm_latency_rng)
        return exe + stor, t_exe

    if logical_op == "ocr":
        exe = video_service_cost_usd(p, r, "ocr", seg_minutes)
        lo, hi = node_ocr_execute_bounds_at(dur_sec, p, r)
        t_exe = 0.5 * (lo * k + hi * k)
        return exe + stor, t_exe

    if logical_op == "label_detection":
        exe = video_service_cost_usd(p, r, "label_detection", seg_minutes)
        lo, hi = node_label_detection_execute_bounds_at(dur_sec, p, r)
        t_exe = 0.5 * (lo * k + hi * k)
        return exe + stor, t_exe

    if logical_op == "speech_transcription":
        exe = video_service_cost_usd(p, r, "speech_transcription", seg_minutes)
        lo, hi = node_speech_transcription_execute_bounds_at(dur_sec, p, r)
        t_exe = 0.5 * (lo * k + hi * k)
        return exe + stor, t_exe

    if logical_op == "database":
        exe_inst = database_instance_cost_usd(p, r, days=1.0)
        lo, hi = node_database_query_execute_bounds_at(p, r)
        k = sample_execution_scale(exec_scale_rng)
        t_exe = 0.5 * (lo * k + hi * k)
        return exe_inst + stor, t_exe

    if logical_op == "qa":
        mm = node.model or ""
        exe = llm_token_cost_usd(p, r, mm, qin, qout)
        t_exe = llm_decode_duration_sec(mm, qout, rng=llm_latency_rng)
        return exe + stor, t_exe

    raise ValueError(logical_op)


def _prepare_deterministic_coefficients_wf2(
    path_id: WF2PathId,
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    queries: list[QueryProfile],
    mean_rho: tuple[float, ...],
    token_seed: int,
    *,
    llm_token_mean_samples: int = _LLM_TOKEN_MEAN_SAMPLES,
) -> tuple[
    list[list[list[float]]],
    list[list[list[list[float]]]],
    list[list[list[float]]],
    list[list[list[list[float]]]],
]:
    ops = path_logical_ops(path_id)
    L = len(ops)
    ops_t = tuple(ops)

    a_c: list[list[list[float]]] = []
    b_c: list[list[list[list[float]]]] = []
    a_l: list[list[list[float]]] = []
    b_l: list[list[list[list[float]]]] = []

    for q_idx, qprof in enumerate(queries):
        s_src = qprof.s_src_gb
        sn, xfer = wf2_utils.propagate_path_sizes(s_src, mean_rho)
        seg_min = wf2_utils._segment_video_minutes_from_source(s_src)
        seed = token_seed + q_idx * 17
        cap_pair, q_pair = wf2_utils.wf2_llm_token_bundle(path_id, s_src, mean_rho)

        ac: list[list[float]] = []
        al: list[list[float]] = []
        for i in range(L):
            op = ops[i]
            rc, rl = [], []
            for ki, node in enumerate(cands[i]):
                exec_scale_rng = wf1_utils.det_rng(
                    seed,
                    "wf2_do_exec_scale",
                    q_idx,
                    i,
                    ki,
                    node.provider,
                    node.region,
                )
                llm_latency_rng = None
                if op in ("video_caption", "qa"):
                    llm_latency_rng = wf1_utils.det_rng(
                        seed,
                        "wf2_do_llm_jit",
                        q_idx,
                        i,
                        ki,
                        node.provider,
                        node.region,
                        node.model,
                    )
                c_ij, lat_ij = _deterministic_local_cost_latency_wf2(
                    op,
                    node,
                    sn,
                    mean_rho,
                    i,
                    seg_min,
                    cap_pair=cap_pair,
                    q_pair=q_pair,
                    exec_scale_rng=exec_scale_rng,
                    llm_latency_rng=llm_latency_rng,
                    ops_full=ops_t,
                )
                if i == 0 and op in ("shot_detection", "speech_transcription"):
                    lep = (node.provider, node.region)
                    luc, lul = _local_edge_mean_cost_latency_wf2(
                        lep,
                        float(s_src),
                        direction="upload",
                        token_seed=token_seed,
                        q_idx=q_idx,
                        ei=140,
                        ka=ki,
                    )
                    c_ij += luc
                    lat_ij += lul
                elif i == L - 1 and op == "qa":
                    lep = (node.provider, node.region)
                    duc, dul = _local_edge_mean_cost_latency_wf2(
                        lep,
                        float(xfer[i]),
                        direction="download",
                        token_seed=token_seed,
                        q_idx=q_idx,
                        ei=143,
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
        for ei in range(L - 1):
            xg = xfer[ei]
            mat_c: list[list[float]] = []
            mat_l: list[list[float]] = []
            for ka, na in enumerate(cands[ei]):
                row_c = []
                row_l = []
                for kb, nb in enumerate(cands[ei + 1]):
                    ec, el = _edge_mean_cost_latency_wf2(
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


def logical_optimal_baseline_wf2(
    path_id: WF2PathId,
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    weights: Sequence[float],
) -> BaselineResultWf2:
    """LO：独立 argmax μ；均衡权重且写死 LO 在候选内时直接返回 ``WF2_LOGICAL_OPTIMAL_NODES``。"""
    L = len(cands)
    if len(weights) != L:
        raise ValueError("weights length must match layers")
    w_t = tuple(weights)
    inv_l = 1.0 / float(L)
    if all(abs(w_t[i] - inv_l) < 1e-15 for i in range(L)):
        fixed = wf2_utils.WF2_LOGICAL_OPTIMAL_NODES[path_id]
        if all(fixed[i] in cands[i] for i in range(L)):
            u_tot = sum(w_t[i] * wf2_node_utility(fixed[i]) for i in range(L))
            return BaselineResultWf2("Logical-Optimal", fixed, float(u_tot), None)
    picks: list[WF2PhysicalNode] = []
    u_tot = 0.0
    for i in range(L):
        best_j = max(
            range(len(cands[i])),
            key=lambda j: weights[i] * wf2_node_utility(cands[i][j]),
        )
        picks.append(cands[i][best_j])
        u_tot += weights[i] * wf2_node_utility(cands[i][best_j])
    return BaselineResultWf2("Logical-Optimal", tuple(picks), float(u_tot), None)


def mc_violation_counts_wf2(
    path_id: WF2PathId,
    nodes: tuple[WF2PhysicalNode, ...],
    queries: Sequence[QueryProfile],
    *,
    samples_per_query: int,
    violation_eval_seed: int,
    cost_tol_abs: float = 1e-9,
    latency_tol_abs: float = 1e-9,
) -> tuple[int, int]:
    """与 ``evaluation.evaluate_deployment_empirical_wf2`` 一致：时延违规按路径求和 ``ell_sum``。"""
    viol_cost = 0
    viol_lat = 0
    ql = list(queries)
    for q_idx, qprof in enumerate(ql):
        for s_idx in range(samples_per_query):
            wf_rng = wf1_utils.det_rng(violation_eval_seed, "wf2_eval_mc", q_idx, s_idx)
            c, ell_sum, _ell_disp = wf2_utils.exclusive_path_cost_and_latency_mc(
                path_id,
                nodes,
                qprof.s_src_gb,
                workflow_rng=wf_rng,
            )
            if c > qprof.theta_cost + cost_tol_abs:
                viol_cost += 1
            if ell_sum > qprof.theta_latency_sec + latency_tol_abs:
                viol_lat += 1
    return viol_cost, viol_lat


def single_cloud_baseline_wf2(
    path_id: WF2PathId,
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    weights: Sequence[float],
    *,
    queries: Sequence[QueryProfile],
    samples_per_query: int = 50,
    violation_eval_seed: int = 137,
) -> BaselineResultWf2:
    """SC：按云过滤后层内 argmax μ；云间比较 MC 违规再比效用。"""
    L = len(cands)
    ql = list(queries)
    if not ql:
        raise ValueError("single_cloud_baseline_wf2 requires non-empty queries")

    best: BaselineResultWf2 | None = None
    best_rank: tuple[int, float, str] | None = None

    for prov in PROVIDERS_SINGLE_CLOUD:
        filt_layers: list[tuple[WF2PhysicalNode, ...]] = []
        ok = True
        for i in range(L):
            layer = tuple(n for n in cands[i] if n.provider == prov)
            if not layer:
                ok = False
                break
            filt_layers.append(layer)
        if not ok:
            continue
        ft = tuple(filt_layers)
        res = logical_optimal_baseline_wf2(path_id, ft, weights=weights)
        vc, vl = mc_violation_counts_wf2(
            path_id,
            res.nodes,
            ql,
            samples_per_query=samples_per_query,
            violation_eval_seed=violation_eval_seed,
        )
        rank = (vc + vl, -res.total_utility, prov)
        if best is None or rank < best_rank:  # type: ignore[operator]
            best_rank = rank
            best = BaselineResultWf2("Single-Cloud", res.nodes, res.total_utility, None)

    if best is None:
        raise RuntimeError(
            "No single provider supports all layers for this path (check WORKFLOW_OPERATIONS / VIDEO_SERVICE)."
        )
    return best


def deterministic_optimal_baseline_wf2(
    path_id: WF2PathId,
    queries: list[QueryProfile],
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    *,
    weights: Sequence[float],
    token_seed: int = 0,
    time_limit_sec: int | None = None,
    rho_calibration_samples: int = 128,
    rho_calibration_seed: int = 4242,
    llm_token_mean_samples: int = _LLM_TOKEN_MEAN_SAMPLES,
) -> BaselineResultWf2:
    calib_rng = random.Random(rho_calibration_seed)
    mean_rho = wf2_utils.plugin_mean_data_conversion_ratios_wf2(
        path_id,
        n_calibration_samples=rho_calibration_samples,
        rng=calib_rng,
    )

    a_cost, b_cost, a_lat, b_lat = _prepare_deterministic_coefficients_wf2(
        path_id,
        cands,
        queries,
        mean_rho,
        token_seed,
        llm_token_mean_samples=llm_token_mean_samples,
    )

    L = len(cands)
    dims = tuple(len(cands[i]) for i in range(L))

    m = gp.Model("baseline_do_wf2_lp")
    m.Params.OutputFlag = 0
    if time_limit_sec is not None:
        m.Params.TimeLimit = float(time_limit_sec)

    x = [
        [m.addVar(vtype=GRB.BINARY, name=f"wf2_do_x_{i}_{k}") for k in range(dims[i])]
        for i in range(L)
    ]
    y_edge: list[list[list[gp.Var]]] = []
    for ei in range(L - 1):
        mat = [
            [
                m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"wf2_do_y_{ei}_{ka}_{kb}")
                for kb in range(dims[ei + 1])
            ]
            for ka in range(dims[ei])
        ]
        y_edge.append(mat)

    for i in range(L):
        m.addConstr(gp.quicksum(x[i][kk] for kk in range(dims[i])) == 1)

    for ei in range(L - 1):
        for ka in range(dims[ei]):
            for kb in range(dims[ei + 1]):
                yv = y_edge[ei][ka][kb]
                m.addConstr(yv <= x[ei][ka])
                m.addConstr(yv <= x[ei + 1][kb])
                m.addConstr(yv >= x[ei][ka] + x[ei + 1][kb] - 1)

    util_expr = gp.quicksum(
        weights[i] * wf2_node_utility(cands[i][k]) * x[i][k]
        for i in range(L)
        for k in range(dims[i])
    )

    nq = len(queries)
    for q_idx in range(nq):
        qprof = queries[q_idx]
        lc_expr = gp.quicksum(
            a_cost[q_idx][i][k] * x[i][k] for i in range(L) for k in range(dims[i])
        ) + gp.quicksum(
            b_cost[q_idx][ei][ka][kb] * y_edge[ei][ka][kb]
            for ei in range(L - 1)
            for ka in range(dims[ei])
            for kb in range(dims[ei + 1])
        )
        lt_expr = gp.quicksum(
            a_lat[q_idx][i][k] * x[i][k] for i in range(L) for k in range(dims[i])
        ) + gp.quicksum(
            b_lat[q_idx][ei][ka][kb] * y_edge[ei][ka][kb]
            for ei in range(L - 1)
            for ka in range(dims[ei])
            for kb in range(dims[ei + 1])
        )
        m.addConstr(lc_expr <= qprof.theta_cost + 1e-9)
        m.addConstr(lt_expr <= qprof.theta_latency_sec + 1e-9)

    m.setObjective(util_expr, GRB.MAXIMIZE)
    m.optimize()
    status_str = wf2_sky._gurobi_status_str(m)

    if status_str != "Optimal":
        placeholder = tuple(cands[i][0] for i in range(L))
        return BaselineResultWf2(
            "Deterministic-Optimal",
            placeholder,
            float("nan"),
            status_str,
        )

    picks: list[int] = []
    for i in range(L):
        vals = [(k, wf2_sky._safe_var_x(v)) for k, v in enumerate(x[i])]
        ks = sorted(vals, key=lambda t: t[1], reverse=True)[0][0]
        picks.append(int(ks))

    chosen = tuple(cands[i][picks[i]] for i in range(L))
    util_val = float(m.ObjVal)

    return BaselineResultWf2("Deterministic-Optimal", chosen, util_val, status_str)


def run_all_baselines_wf2(
    path_id: WF2PathId,
    queries: list[QueryProfile],
    *,
    weights: Sequence[float] | None = None,
    do_token_seed: int = 0,
    sc_samples_per_query: int = 50,
    sc_violation_eval_seed: int = 137,
) -> tuple[BaselineResultWf2, BaselineResultWf2, BaselineResultWf2]:
    w = weights if weights is not None else wf2_utils.default_weights_for_path(path_id)
    cands = enumerate_candidates_wf2(path_id)
    sc = single_cloud_baseline_wf2(
        path_id,
        cands,
        weights=w,
        queries=queries,
        samples_per_query=sc_samples_per_query,
        violation_eval_seed=sc_violation_eval_seed,
    )
    lo = logical_optimal_baseline_wf2(path_id, cands, weights=w)
    do = deterministic_optimal_baseline_wf2(
        path_id, queries, cands, weights=w, token_seed=do_token_seed
    )
    return sc, lo, do


if __name__ == "__main__":  # pragma: no cover
    from .evaluation import evaluate_deployment_empirical_wf2, print_metrics_report_wf2

    n_q = 5
    pid: WF2PathId = "video_caption"
    cands = enumerate_candidates_wf2(pid)
    WEIGHTS = wf2_utils.default_weights_for_path(pid)
    lo_ch = wf2_utils.wf2_logical_optimal_chain(pid, cands, WEIGHTS)
    qs = wf2_utils.generate_realistic_queries_wf2(
        n_q,
        pid,
        seed=7,
        budget_alpha=float(wf1_utils.BUDGET_ALPHA_SUITE_DEFAULT_WF1[-1]),
        lo_chain=lo_ch,
        weights=WEIGHTS,
        cands=cands,
    )

    print("Candidates per layer:", [len(c) for c in cands])
    lo = logical_optimal_baseline_wf2(pid, cands, weights=WEIGHTS)
    sc = single_cloud_baseline_wf2(
        pid,
        cands,
        weights=WEIGHTS,
        queries=qs,
        samples_per_query=30,
        violation_eval_seed=2026,
    )
    print(lo.name, "U=", lo.total_utility)
    print(sc.name, "U=", sc.total_utility)

    for name_prefix, bp in [(lo.name, lo), (sc.name, sc)]:
        m = evaluate_deployment_empirical_wf2(
            pid,
            bp.nodes,
            qs,
            weights=WEIGHTS,
            samples_per_query=30,
            eval_seed=2026,
        )
        print_metrics_report_wf2(algorithm_label=name_prefix, metrics=m)

    do = deterministic_optimal_baseline_wf2(pid, qs, cands, weights=WEIGHTS)
    print(do.name, do.gurobi_status, do.total_utility, do.nodes)
    md = evaluate_deployment_empirical_wf2(
        pid, do.nodes, qs, weights=WEIGHTS, samples_per_query=30, eval_seed=2026
    )
    print_metrics_report_wf2(algorithm_label=f"{do.name} | Gurobi={do.gurobi_status}", metrics=md)
