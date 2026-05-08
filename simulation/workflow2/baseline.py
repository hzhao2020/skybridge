"""
Workflow 2 baselines（对齐 workflow1）：SC / LO / DO。

SC：单提供商可行时层内 argmax μ；若传入 ``queries`` 与 MC 参数，则在提供商间
**优先最小化** 与 ``evaluation.evaluate_deployment_empirical_wf2`` 一致的 MC SLO 违规数（费用+时延），再以 utility 为高者优先。

DO：plug-in mean ρ（沿路径蒙特卡洛校准）、LLM token 经验均值、链路网络经验均值；
目标 max Σ w_i μ_i，约束为每条查询在均值场上的费用与时延不超过 Θ。

运行：``python -m workflow2.baseline``。
"""

from __future__ import annotations

import random
from typing import NamedTuple, Sequence

import pulp as pl

from sim_env.cost import egress_cost_usd, llm_token_cost_usd, split_cost_usd, storage_cost_usd, video_service_cost_usd
from sim_env.execution_latency import (
    llm_decode_duration_sec,
    node_label_detection_execute_bounds_at,
    node_ocr_execute_bounds_at,
    node_segment_execute_bounds_at,
    node_speech_transcription_execute_bounds_at,
    node_split_execute_bounds_at,
    sample_execution_scale,
)
from sim_env.llm import caption_visual_input_tokens, sample_caption_output_tokens, sample_query_output_tokens
from sim_env.network import LinkCategory, classify_link, reset_link_counters, sample_link
from sim_env.utility import QueryProfile

from workflow1 import utils as wf1_utils

from . import utils as wf2_utils
from .candidates import enumerate_candidates_wf2
from .sky import path_logical_ops
from .utils import WF2PathId, WF2PhysicalNode, wf2_node_utility

PROVIDERS_SINGLE_CLOUD = ("GCP", "AWS", "Aliyun")

_NET_MEAN_SAMPLES = 64
_LLM_TOKEN_MEAN_SAMPLES = 512


class BaselineResultWf2(NamedTuple):
    name: str
    nodes: tuple[WF2PhysicalNode, ...]
    total_utility: float
    pulp_status: str | None


def _deterministic_llm_token_means(
    seg_minutes: float,
    token_seed: int,
    n_samples: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    import numpy as np

    dur_sec = max(seg_minutes * 60.0, 1e-6)
    cin = caption_visual_input_tokens(dur_sec)
    cout_acc = 0.0
    qout_acc = 0.0
    for si in range(n_samples):
        cap_rng_seed = wf1_utils.det_rng(token_seed, "wf2_do_llm_cap", si).randrange(0, 2**31)
        cout_i = sample_caption_output_tokens(dur_sec, rng=cap_rng_seed)
        g = np.random.default_rng(
            wf1_utils.det_rng(token_seed, "wf2_do_llm_qry", si).randrange(0, 2**31)
        )
        qout_i = sample_query_output_tokens(rng=g)
        cout_acc += cout_i
        qout_acc += qout_i
    cout_m = cout_acc / float(n_samples)
    qout_m = qout_acc / float(n_samples)
    return (cin, cout_m), (cout_m, qout_m)


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
) -> tuple[float, float]:
    rho_i = float(rho[idx])
    stor = storage_cost_usd(
        node.provider, node.region, float(s_in[idx]) * (1.0 + rho_i), days=1.0
    )
    p, r = node.provider, node.region
    dur_sec = max(seg_minutes * 60.0, 1e-6)
    cin, cout = cap_pair
    _, qout = q_pair
    k = sample_execution_scale(exec_scale_rng)

    if logical_op == "video_segment":
        exe = video_service_cost_usd(p, r, "segment", seg_minutes)
        lo, hi = node_segment_execute_bounds_at(dur_sec, p, r)
        t_exe = 0.5 * (lo * k + hi * k)
        return exe + stor, t_exe

    if logical_op == "shot_detection":
        exe = split_cost_usd(p, r, minutes=1.0)
        lo, hi = node_split_execute_bounds_at(dur_sec, p, r)
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
        return wf2_utils.WF2_PLACEHOLDER_DB_FIXED_COST_USD + stor, wf2_utils.WF2_PLACEHOLDER_DB_LATENCY_SEC

    if logical_op == "qa":
        return wf2_utils.WF2_PLACEHOLDER_QA_FIXED_COST_USD + stor, wf2_utils.WF2_PLACEHOLDER_QA_LATENCY_SEC

    if logical_op == "answer":
        mm = node.model or ""
        exe = llm_token_cost_usd(
            p, r, mm, wf2_utils.WF2_PLACEHOLDER_ANSWER_INPUT_TOKENS, qout
        )
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

    a_c: list[list[list[float]]] = []
    b_c: list[list[list[list[float]]]] = []
    a_l: list[list[list[float]]] = []
    b_l: list[list[list[list[float]]]] = []

    for q_idx, qprof in enumerate(queries):
        s_src = qprof.s_src_gb
        sn, xfer = wf2_utils.propagate_path_sizes(s_src, mean_rho)
        seg_min = wf2_utils._segment_video_minutes_from_source(s_src)
        seed = token_seed + q_idx * 17
        cap_pair, q_pair = _deterministic_llm_token_means(
            seg_min,
            seed,
            llm_token_mean_samples,
        )

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
                if op in ("video_caption", "answer"):
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
                )
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
    L = len(cands)
    if len(weights) != L:
        raise ValueError("weights length must match layers")
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
    queries: Sequence[QueryProfile] | None = None,
    samples_per_query: int = 50,
    violation_eval_seed: int = 137,
) -> BaselineResultWf2:
    L = len(cands)
    ql = list(queries) if queries is not None else []
    use_viol = len(ql) > 0

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
        if use_viol:
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
        else:
            if best is None or res.total_utility > best.total_utility:
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

    prob = pl.LpProblem("baseline_do_wf2_lp", pl.LpMaximize)

    x = [
        [pl.LpVariable(f"wf2_do_x_{i}_{k}", cat=pl.LpBinary) for k in range(dims[i])]
        for i in range(L)
    ]
    y_edge: list[list[list[pl.LpVariable]]] = []
    for ei in range(L - 1):
        mat = [
            [
                pl.LpVariable(f"wf2_do_y_{ei}_{ka}_{kb}", lowBound=0, upBound=1)
                for kb in range(dims[ei + 1])
            ]
            for ka in range(dims[ei])
        ]
        y_edge.append(mat)

    for i in range(L):
        prob += pl.lpSum(x[i][kk] for kk in range(dims[i])) == 1

    for ei in range(L - 1):
        for ka in range(dims[ei]):
            for kb in range(dims[ei + 1]):
                yv = y_edge[ei][ka][kb]
                prob += yv <= x[ei][ka]
                prob += yv <= x[ei + 1][kb]
                prob += yv >= x[ei][ka] + x[ei + 1][kb] - 1

    util_terms: list = []
    for i in range(L):
        for k in range(dims[i]):
            util_terms.append(weights[i] * wf2_node_utility(cands[i][k]) * x[i][k])

    nq = len(queries)
    for q_idx in range(nq):
        qprof = queries[q_idx]
        lc_terms: list = []
        for i in range(L):
            for k in range(dims[i]):
                lc_terms.append(a_cost[q_idx][i][k] * x[i][k])
        for ei in range(L - 1):
            for ka in range(dims[ei]):
                for kb in range(dims[ei + 1]):
                    lc_terms.append(b_cost[q_idx][ei][ka][kb] * y_edge[ei][ka][kb])

        lt_terms: list = []
        for i in range(L):
            for k in range(dims[i]):
                lt_terms.append(a_lat[q_idx][i][k] * x[i][k])
        for ei in range(L - 1):
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
        placeholder = tuple(cands[i][0] for i in range(L))
        return BaselineResultWf2(
            "Deterministic-Optimal",
            placeholder,
            float("nan"),
            status_str,
        )

    picks: list[int] = []
    for i in range(L):
        vals = [(k, float(pl.value(v) or 0.0)) for k, v in enumerate(x[i])]
        ks = sorted(vals, key=lambda t: t[1], reverse=True)[0][0]
        picks.append(int(ks))

    chosen = tuple(cands[i][picks[i]] for i in range(L))
    util_obj = pl.value(prob.objective)
    util_val = float(util_obj if util_obj is not None else 0.0)

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
    pid: WF2PathId = "caption"
    qs = wf2_utils.generate_realistic_queries_wf2(n_q, pid, seed=7)
    cands = enumerate_candidates_wf2(pid)
    WEIGHTS = wf2_utils.default_weights_for_path(pid)

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
    print(do.name, do.pulp_status, do.total_utility, do.nodes)
    md = evaluate_deployment_empirical_wf2(
        pid, do.nodes, qs, weights=WEIGHTS, samples_per_query=30, eval_seed=2026
    )
    print_metrics_report_wf2(algorithm_label=f"{do.name} | MILP={do.pulp_status}", metrics=md)
