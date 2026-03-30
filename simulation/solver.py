#!/usr/bin/env python3
"""
基于论文的 CVaR-MILP 求解器：chance-constrained 节点选择与路径优化。

支持多 query（如 300 个）：约束为 P(所有 query 满足 cost/latency 阈值) >= 1-η，
即「至少有一个 query 不满足」的概率 <= η。目标：最大化 accuracy (utility)。

算法流程：
1. CVaR 近似：将 P(Z<=0)>=1-η 转为 F_{1-η}(α) = α + (1/η)E[(Z-α)^+] ≤ 0
2. SAA：用 K 个场景离散化期望，引入辅助变量 z_{q,k}^C, z_{q,k}^T（每 query、每场景）
3. MILP 求解：max U(x) s.t. 拓扑约束 + CVaR 约束
4. 拉格朗日对偶松弛：可选，将 CVaR 约束松弛到目标中迭代求解
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from distribution import (
    Node,
    compute_end_to_end_cost_usd,
    compute_end_to_end_latency_s,
    sample_data_conversion_ratios_all,
    compute_end_to_end_utility_score,
    get_cost_budget_usd,
    get_fixed_simulation_params,
    get_latency_budget_s,
)


def _find_node(nodes: Iterable[Node], name: str) -> Node:
    for n in nodes:
        if n.name == name:
            return n
    raise KeyError(f"未找到节点: {name!r}")


def _enumerate_paths(
    segment_names: list[str],
    split_names: list[str],
    caption_names: list[str],
    query_names: list[str],
    params=None,
) -> list[tuple[str, str, str, str]]:
    """枚举所有 (segment, split, caption, query) 路径。"""
    p = params or get_fixed_simulation_params()
    seg_set = {n.name for n in p.build_nodes("segment")}
    spl_set = {n.name for n in p.build_nodes("split")}
    cap_set = {n.name for n in p.build_nodes("caption")}
    qry_set = {n.name for n in p.build_nodes("query")}

    paths: list[tuple[str, str, str, str]] = []
    for sn in segment_names:
        if sn not in seg_set:
            continue
        for spn in split_names:
            if spn not in spl_set:
                continue
            for cn in caption_names:
                if cn not in cap_set:
                    continue
                for qn in query_names:
                    if qn in qry_set:
                        paths.append((sn, spn, cn, qn))
    return paths


def _sample_scenarios(
    video_sizes_mb: list[float],
    paths: list[tuple[str, str, str, str]],
    K: int,
    params=None,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """
    对 K 个场景、|Q| 个 query 采样 cost 和 latency 矩阵。
    cost[p][q][k] = 路径 p、query q、场景 k 下的 cost (USD)
    latency[p][q][k] = 路径 p、query q、场景 k 下的 latency (s)
    同一场景 k 下共享随机状态 ξ^k。
    """
    p = params or get_fixed_simulation_params()
    seg_nodes = {n.name: n for n in p.build_nodes("segment")}
    spl_nodes = {n.name: n for n in p.build_nodes("split")}
    cap_nodes = {n.name: n for n in p.build_nodes("caption")}
    qry_nodes = {n.name: n for n in p.build_nodes("query")}

    n_paths = len(paths)
    n_queries = len(video_sizes_mb)
    cost_mat: list[list[list[float]]] = [
        [[0.0] * K for _ in range(n_queries)] for _ in range(n_paths)
    ]
    latency_mat: list[list[list[float]]] = [
        [[0.0] * K for _ in range(n_queries)] for _ in range(n_paths)
    ]

    for k in range(K):
        for p, (sn, spn, cn, qn) in enumerate(paths):
            seg = seg_nodes[sn]
            spl = spl_nodes[spn]
            cap = cap_nodes[cn]
            qry = qry_nodes[qn]
            for q, video_size_mb in enumerate(video_sizes_mb):
                random.seed(42 + k)
                data_ratios = sample_data_conversion_ratios_all()
                cost_mat[p][q][k] = compute_end_to_end_cost_usd(
                    video_size_mb=video_size_mb,
                    segment_node=seg,
                    split_node=spl,
                    caption_node=cap,
                    query_node=qry,
                    data_conversion_ratios=data_ratios,
                )
                latency_mat[p][q][k] = compute_end_to_end_latency_s(
                    video_size_mb=video_size_mb,
                    segment_node=seg,
                    split_node=spl,
                    caption_node=cap,
                    query_node=qry,
                    data_conversion_ratios=data_ratios,
                )

    return cost_mat, latency_mat


def _compute_utility_per_path(paths: list[tuple[str, str, str, str]], params=None) -> list[float]:
    """计算每条路径的 utility（accuracy 加权和）。"""
    p = params or get_fixed_simulation_params()
    seg_nodes = {n.name: n for n in p.build_nodes("segment")}
    spl_nodes = {n.name: n for n in p.build_nodes("split")}
    cap_nodes = {n.name: n for n in p.build_nodes("caption")}
    qry_nodes = {n.name: n for n in p.build_nodes("query")}

    utils: list[float] = []
    for sn, spn, cn, qn in paths:
        seg = seg_nodes[sn]
        spl = spl_nodes[spn]
        cap = cap_nodes[cn]
        qry = qry_nodes[qn]
        u = compute_end_to_end_utility_score(
            segment_node=seg,
            split_node=spl,
            caption_node=cap,
            query_node=qry,
        )
        utils.append(u)
    return utils


@dataclass
class SolverConfig:
    """求解器配置。"""
    K: int = 20                    # SAA 场景数
    eta_cost: float = 0.1          # cost chance constraint: P(cost <= budget) >= 1 - eta_cost
    eta_latency: float = 0.1       # latency chance constraint
    segment_names: list[str] | None = None   # None = 使用全部
    split_names: list[str] | None = None
    caption_names: list[str] | None = None
    query_names: list[str] | None = None


def _default_node_names() -> tuple[list[str], list[str], list[str], list[str]]:
    """默认使用的节点子集（减少路径数，加速求解）。"""
    seg = [f"p{i}_r{j}_segment" for i in range(1, 4) for j in range(1, 3)][:6]
    spl = [f"p{i}_r{j}_split" for i in range(1, 4) for j in range(1, 3)][:6]
    cap = [f"p4_m{i}_caption" for i in range(1, 5)] + [f"p5_m{i}_caption" for i in range(3, 6)]
    qry = [f"p4_m{i}_query" for i in range(1, 5)] + [f"p5_m{i}_query" for i in range(3, 6)]
    return seg, spl, cap, qry


def solve_cvar_milp(
    video_sizes_mb: list[float],
    config: SolverConfig | None = None,
) -> tuple[tuple[str, str, str, str], float, float, float]:
    """
    求解 CVaR-MILP：在 chance 约束下最大化 accuracy。

    约束：P(所有 query 满足 cost/latency 阈值) >= 1-η
    即「没有 query 不满足阈值」的概率 >= 1-η（等价于「至少有一个不满足」的概率 <= η）

    video_sizes_mb: 每个 query 的视频大小 (MB)，如 300 个 query 则长度为 300
    返回：(最优路径, utility, 平均 cost, 平均 latency)
    """
    try:
        from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
    except ImportError:
        raise ImportError("需要安装 PuLP: pip install pulp")

    cfg = config or SolverConfig()
    seg_names, spl_names, cap_names, qry_names = _default_node_names()
    if cfg.segment_names is not None:
        seg_names = cfg.segment_names
    if cfg.split_names is not None:
        spl_names = cfg.split_names
    if cfg.caption_names is not None:
        cap_names = cfg.caption_names
    if cfg.query_names is not None:
        qry_names = cfg.query_names

    paths = _enumerate_paths(seg_names, spl_names, cap_names, qry_names)
    if not paths:
        raise ValueError("无有效路径")

    K = cfg.K
    eta_C = cfg.eta_cost
    eta_T = cfg.eta_latency
    n_queries = len(video_sizes_mb)
    theta_C = [get_cost_budget_usd(vs) for vs in video_sizes_mb]
    theta_T = [get_latency_budget_s(vs) for vs in video_sizes_mb]

    cost_mat, latency_mat = _sample_scenarios(video_sizes_mb, paths, K)
    utils = _compute_utility_per_path(paths)

    n_paths = len(paths)
    M_C = (
        max(
            max(cost_mat[p][q][k] for k in range(K) for q in range(n_queries))
            for p in range(n_paths)
        )
        + 1.0
    )
    M_T = (
        max(
            max(latency_mat[p][q][k] for k in range(K) for q in range(n_queries))
            for p in range(n_paths)
        )
        + 100.0
    )

    prob = LpProblem("CVaR_MILP", LpMaximize)

    x = [LpVariable(f"x_{p}", cat="Binary") for p in range(n_paths)]
    alpha_C = LpVariable("alpha_C", cat="Continuous")
    alpha_T = LpVariable("alpha_T", cat="Continuous")
    zC = [
        [LpVariable(f"zC_{q}_{k}", lowBound=0) for k in range(K)]
        for q in range(n_queries)
    ]
    zT = [
        [LpVariable(f"zT_{q}_{k}", lowBound=0) for k in range(K)]
        for q in range(n_queries)
    ]

    prob += lpSum(x) == 1
    prob += lpSum(utils[p] * x[p] for p in range(n_paths))

    for q in range(n_queries):
        for k in range(K):
            for p in range(n_paths):
                prob += (
                    zC[q][k]
                    >= cost_mat[p][q][k] - theta_C[q] - alpha_C - M_C * (1 - x[p])
                )
                prob += (
                    zT[q][k]
                    >= latency_mat[p][q][k] - theta_T[q] - alpha_T - M_T * (1 - x[p])
                )

    # CVaR: α + (1/(η|Q|K)) Σ_{q,k} z_{q,k} <= 0
    prob += (
        alpha_C
        + (1.0 / (eta_C * n_queries * K)) * lpSum(zC[q][k] for q in range(n_queries) for k in range(K))
        <= 0
    )
    prob += (
        alpha_T
        + (1.0 / (eta_T * n_queries * K)) * lpSum(zT[q][k] for q in range(n_queries) for k in range(K))
        <= 0
    )

    prob.solve()

    if prob.status != 1:
        raise RuntimeError(f"MILP 求解失败: status={prob.status}")

    best_p = next(p for p in range(n_paths) if value(x[p]) > 0.5)
    path = paths[best_p]
    u = value(prob.objective)
    c = sum(
        sum(cost_mat[best_p][q][k] for k in range(K)) / K
        for q in range(n_queries)
    ) / n_queries
    t = sum(
        sum(latency_mat[best_p][q][k] for k in range(K)) / K
        for q in range(n_queries)
    ) / n_queries

    return path, u, c, t


def solve_lagrangian_relaxation(
    video_sizes_mb: list[float],
    config: SolverConfig | None = None,
    max_iter: int = 50,
    step_size: float = 0.1,
) -> tuple[tuple[str, str, str, str], float, float, float]:
    """
    拉格朗日对偶松弛：将 CVaR 约束松弛到目标中，迭代更新乘子。
    """
    try:
        from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
    except ImportError:
        raise ImportError("需要安装 PuLP: pip install pulp")

    cfg = config or SolverConfig()
    seg_names, spl_names, cap_names, qry_names = _default_node_names()
    if cfg.segment_names is not None:
        seg_names = cfg.segment_names
    if cfg.split_names is not None:
        spl_names = cfg.split_names
    if cfg.caption_names is not None:
        cap_names = cfg.caption_names
    if cfg.query_names is not None:
        qry_names = cfg.query_names

    paths = _enumerate_paths(seg_names, spl_names, cap_names, qry_names)
    if not paths:
        raise ValueError("无有效路径")

    K = cfg.K
    n_queries = len(video_sizes_mb)
    theta_C = [get_cost_budget_usd(vs) for vs in video_sizes_mb]
    theta_T = [get_latency_budget_s(vs) for vs in video_sizes_mb]
    eta_C = cfg.eta_cost
    eta_T = cfg.eta_latency

    cost_mat, latency_mat = _sample_scenarios(video_sizes_mb, paths, K)
    utils = _compute_utility_per_path(paths)

    n_paths = len(paths)
    M_C = (
        max(
            max(cost_mat[p][q][k] for k in range(K) for q in range(n_queries))
            for p in range(n_paths)
        )
        + 1.0
    )
    M_T = (
        max(
            max(latency_mat[p][q][k] for k in range(K) for q in range(n_queries))
            for p in range(n_paths)
        )
        + 100.0
    )

    lam_C = 1.0
    lam_T = 1.0
    best_p = 0

    for it in range(max_iter):
        prob = LpProblem("Lagrangian", LpMaximize)
        x = [LpVariable(f"x_{p}", cat="Binary") for p in range(n_paths)]
        alpha_C = LpVariable("alpha_C", cat="Continuous")
        alpha_T = LpVariable("alpha_T", cat="Continuous")
        zC = [
            [LpVariable(f"zC_{q}_{k}", lowBound=0) for k in range(K)]
            for q in range(n_queries)
        ]
        zT = [
            [LpVariable(f"zT_{q}_{k}", lowBound=0) for k in range(K)]
            for q in range(n_queries)
        ]

        prob += lpSum(x) == 1
        obj = lpSum(utils[p] * x[p] for p in range(n_paths))
        obj -= lam_C * (
            alpha_C
            + (1.0 / (eta_C * n_queries * K))
            * lpSum(zC[q][k] for q in range(n_queries) for k in range(K))
        )
        obj -= lam_T * (
            alpha_T
            + (1.0 / (eta_T * n_queries * K))
            * lpSum(zT[q][k] for q in range(n_queries) for k in range(K))
        )
        prob += obj

        for q in range(n_queries):
            for k in range(K):
                for p in range(n_paths):
                    prob += (
                        zC[q][k]
                        >= cost_mat[p][q][k] - theta_C[q] - alpha_C - M_C * (1 - x[p])
                    )
                    prob += (
                        zT[q][k]
                        >= latency_mat[p][q][k] - theta_T[q] - alpha_T - M_T * (1 - x[p])
                    )

        prob.solve()
        if prob.status != 1:
            break

        best_p = next(p for p in range(n_paths) if value(x[p]) > 0.5)

        g_C = value(alpha_C) + sum(
            value(zC[q][k]) for q in range(n_queries) for k in range(K)
        ) / (eta_C * n_queries * K)
        g_T = value(alpha_T) + sum(
            value(zT[q][k]) for q in range(n_queries) for k in range(K)
        ) / (eta_T * n_queries * K)

        lam_C = max(0.0, lam_C + step_size * g_C)
        lam_T = max(0.0, lam_T + step_size * g_T)

        if g_C <= 1e-4 and g_T <= 1e-4:
            break

    path = paths[best_p]
    u = sum(utils[p] * value(x[p]) for p in range(n_paths))
    c = sum(
        sum(cost_mat[best_p][q][k] for k in range(K)) / K
        for q in range(n_queries)
    ) / n_queries
    t = sum(
        sum(latency_mat[best_p][q][k] for k in range(K)) / K
        for q in range(n_queries)
    ) / n_queries

    return path, u, c, t


def main() -> None:
    import argparse

    from distribution import get_data_size_mb

    parser = argparse.ArgumentParser(description="CVaR-MILP 求解器（多 query）")
    parser.add_argument("--n-queries", type=int, default=300, help="query 数量")
    parser.add_argument("--video-size", type=float, default=100.0, help="单 query 时使用；多 query 时作为均值")
    parser.add_argument("--video-size-min", type=float, default=5.0)
    parser.add_argument("--video-size-max", type=float, default=500.0)
    parser.add_argument("--K", type=int, default=20, help="SAA 场景数")
    parser.add_argument("--eta", type=float, default=0.1, help="chance 约束: P(所有 query 满足) >= 1-eta")
    parser.add_argument("--lagrangian", action="store_true", help="使用拉格朗日松弛")
    args = parser.parse_args()

    if args.n_queries == 1:
        video_sizes_mb = [args.video_size]
    else:
        random.seed(42)
        video_sizes_mb = [
            get_data_size_mb(args.video_size_min, args.video_size_max)
            for _ in range(args.n_queries)
        ]

    cfg = SolverConfig(K=args.K, eta_cost=args.eta, eta_latency=args.eta)
    if args.lagrangian:
        path, u, c, t = solve_lagrangian_relaxation(video_sizes_mb, cfg)
        print("方法: 拉格朗日对偶松弛")
    else:
        path, u, c, t = solve_cvar_milp(video_sizes_mb, cfg)
        print("方法: CVaR-MILP 直接求解")

    print(f"n_queries = {len(video_sizes_mb)}")
    print(f"video_size_mb 范围: [{min(video_sizes_mb):.1f}, {max(video_sizes_mb):.1f}]")
    print(f"最优路径: segment={path[0]}, split={path[1]}, caption={path[2]}, query={path[3]}")
    print(f"accuracy (utility) = {u:.6f}")
    print(f"cost_usd (跨 query 场景均值) = {c:.6f}")
    print(f"latency_s (跨 query 场景均值) = {t:.6f}")
    avg_budget_c = sum(get_cost_budget_usd(vs) for vs in video_sizes_mb) / len(video_sizes_mb)
    avg_budget_t = sum(get_latency_budget_s(vs) for vs in video_sizes_mb) / len(video_sizes_mb)
    print(f"平均 cost_budget = {avg_budget_c:.6f}, 平均 latency_budget_s = {avg_budget_t:.6f}")


if __name__ == "__main__":
    main()
