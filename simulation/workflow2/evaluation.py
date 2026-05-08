"""
Workflow 2 部署方案的蒙特卡洛 KPI。

- **mean_latency_sec**：与 MILP / 算法一致，为所选互斥路径上的时延 **求和**（不做 path 间 max）。
- **mean_workflow_display_latency_sec**：结果展示用，DAG 岔路上 ``max(T_shot_parallel_modalities, T_speech)``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from sim_env.utility import QueryProfile

from workflow1 import utils as wf1_utils

from .utils import (
    WF2PathId,
    WF2PhysicalNode,
    end_to_end_utility_exclusive_path,
    exclusive_path_cost_and_latency_mc,
    validate_exclusive_path_nodes,
)


@dataclass(frozen=True)
class EmpiricalDeploymentMetricsWf2:
    aggregate_utility_u: float
    mean_cost_usd: float
    mean_latency_sec: float
    std_latency_sec: float
    mean_workflow_display_latency_sec: float
    std_workflow_display_latency_sec: float
    slo_cost_violation_rate: float
    slo_latency_violation_rate: float
    n_draws_total: int
    std_cost_usd: float


def format_metrics_lines_wf2(
    *,
    algorithm_label: str,
    metrics: EmpiricalDeploymentMetricsWf2,
) -> list[str]:
    hdr = f"===== Evaluation KPIs ({algorithm_label}) ====="
    sep = "=" * max(len(hdr), 50)
    return [
        sep,
        hdr,
        sep,
        f"(1) Aggregate Utility (U):        {metrics.aggregate_utility_u:.6g}",
        f"(2) Total monetary cost - mean (USD over draws): "
        f"{metrics.mean_cost_usd:.6g}  [std~ {metrics.std_cost_usd:.4g}]",
        f"(3) Latency — optimized path sum (algorithm): "
        f"{metrics.mean_latency_sec:.6g} s  [std~ {metrics.std_latency_sec:.4g}]",
        f"(3b) Latency — workflow display max(forks): "
        f"{metrics.mean_workflow_display_latency_sec:.6g} s  "
        f"[std~ {metrics.std_workflow_display_latency_sec:.4g}]",
        f"(4a) Cost SLO Violation Rate:     {metrics.slo_cost_violation_rate:.6g} "
        f"(C > Theta_C^q, {metrics.n_draws_total} draws)",
        f"(4b) Latency SLO Violation Rate: {metrics.slo_latency_violation_rate:.6g} "
        f"(T_sum > Theta_T^q, same draws; compares algorithm path sum)",
        sep,
    ]


def print_metrics_report_wf2(*, algorithm_label: str, metrics: EmpiricalDeploymentMetricsWf2) -> None:
    for line in format_metrics_lines_wf2(algorithm_label=algorithm_label, metrics=metrics):
        print(line)


def evaluate_deployment_empirical_wf2(
    path_id: WF2PathId,
    nodes: Sequence[WF2PhysicalNode],
    queries: Iterable[QueryProfile],
    *,
    weights: Sequence[float],
    samples_per_query: int = 50,
    eval_seed: int = 137,
    cost_tol_abs: float = 1e-9,
    latency_tol_abs: float = 1e-9,
) -> EmpiricalDeploymentMetricsWf2:
    validate_exclusive_path_nodes(path_id, nodes)
    tup = tuple(nodes)
    if len(weights) != len(tup):
        raise ValueError("weights length must match deployment chain")

    ql = list(queries)
    if len(ql) == 0:
        raise ValueError("queries must be non-empty")

    u_val = end_to_end_utility_exclusive_path(path_id, tup, weights=tuple(weights))

    costs: list[float] = []
    lats_sum: list[float] = []
    lats_disp: list[float] = []
    viol_cost = 0
    viol_latency = 0
    draws = 0

    for q_idx, qprof in enumerate(ql):
        for s_idx in range(samples_per_query):
            wf_rng = wf1_utils.det_rng(eval_seed, "wf2_eval_mc", q_idx, s_idx)
            c, ell_sum, ell_disp = exclusive_path_cost_and_latency_mc(
                path_id,
                tup,
                qprof.s_src_gb,
                workflow_rng=wf_rng,
            )
            costs.append(c)
            lats_sum.append(ell_sum)
            lats_disp.append(ell_disp)
            draws += 1
            if c > qprof.theta_cost + cost_tol_abs:
                viol_cost += 1
            if ell_sum > qprof.theta_latency_sec + latency_tol_abs:
                viol_latency += 1

    mean_c = sum(costs) / float(draws)
    mean_ts = sum(lats_sum) / float(draws)
    mean_td = sum(lats_disp) / float(draws)
    if draws > 1:
        var_c = sum((x - mean_c) ** 2 for x in costs) / float(draws - 1)
        var_ts = sum((x - mean_ts) ** 2 for x in lats_sum) / float(draws - 1)
        var_td = sum((x - mean_td) ** 2 for x in lats_disp) / float(draws - 1)
        std_c = math.sqrt(max(var_c, 0.0))
        std_ts = math.sqrt(max(var_ts, 0.0))
        std_td = math.sqrt(max(var_td, 0.0))
    else:
        std_c = 0.0
        std_ts = 0.0
        std_td = 0.0

    denom = float(draws) if draws > 0 else 1.0
    vr_c = float(viol_cost) / denom
    vr_l = float(viol_latency) / denom

    return EmpiricalDeploymentMetricsWf2(
        aggregate_utility_u=u_val,
        mean_cost_usd=mean_c,
        mean_latency_sec=mean_ts,
        std_latency_sec=std_ts,
        mean_workflow_display_latency_sec=mean_td,
        std_workflow_display_latency_sec=std_td,
        slo_cost_violation_rate=vr_c,
        slo_latency_violation_rate=vr_l,
        n_draws_total=draws,
        std_cost_usd=std_c,
    )
