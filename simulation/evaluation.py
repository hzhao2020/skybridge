"""
Empirical KPIs after a deployment plan is fixed:

(1) Aggregate Utility U — weighted Σ μ_i(w) over logical nodes.
(2) Mean total monetary cost C — Monte Carlo sample mean per request × query set.
(3) Mean end-to-end latency T — same Monte Carlo draws.
(4) SLO violation rates (same Monte Carlo draws; counts are not mutually exclusive):
    - Cost VR: fraction where C > Θ_C^q
    - Latency VR: fraction where T > Θ_T^q

Runs from ``simulation/`` alongside ``sky`` / ``baseline``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from sim_env import config as cfg
from sim_env.utility import PhysicalNode, QueryProfile

import utils as wf_utils

_OPS = ("segment", "split", "caption", "query")


class EmpiricalDeploymentMetrics:
    __slots__ = (
        "aggregate_utility_u",
        "mean_cost_usd",
        "mean_latency_sec",
        "slo_cost_violation_rate",
        "slo_latency_violation_rate",
        "n_draws_total",
        "std_cost_usd",
        "std_latency_sec",
    )

    def __init__(
        self,
        *,
        aggregate_utility_u: float,
        mean_cost_usd: float,
        mean_latency_sec: float,
        slo_cost_violation_rate: float,
        slo_latency_violation_rate: float,
        n_draws_total: int,
        std_cost_usd: float,
        std_latency_sec: float,
    ) -> None:
        self.aggregate_utility_u = aggregate_utility_u
        self.mean_cost_usd = mean_cost_usd
        self.mean_latency_sec = mean_latency_sec
        self.slo_cost_violation_rate = slo_cost_violation_rate
        self.slo_latency_violation_rate = slo_latency_violation_rate
        self.n_draws_total = n_draws_total
        self.std_cost_usd = std_cost_usd
        self.std_latency_sec = std_latency_sec


def evaluate_deployment_empirical(
    nodes: Sequence[PhysicalNode],
    queries: Iterable[QueryProfile],
    *,
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    samples_per_query: int = 50,
    eval_seed: int = 137,
    cost_tol_abs: float = 1e-9,
    latency_tol_abs: float = 1e-9,
) -> EmpiricalDeploymentMetrics:
    """
    Monte Carlo evaluation over independent draws of rho and network/exec noise (via utils).

    Denominator for both marginal rates: ``len(queries) x samples_per_query``.
    """
    ql = list(queries)
    if len(ql) == 0:
        raise ValueError("queries must be non-empty")

    tup = tuple(nodes)
    u_val = wf_utils.end_to_end_utility(tup, weights=weights)

    costs: list[float] = []
    lats: list[float] = []
    viol_cost = 0
    viol_latency = 0
    draws = 0

    for q_idx, qprof in enumerate(ql):
        for s_idx in range(samples_per_query):
            rho_rng = wf_utils.det_rng(eval_seed, "eval_rho", q_idx, s_idx)
            wf_rng = wf_utils.det_rng(eval_seed, "eval_mc", q_idx, s_idx)
            rho = tuple(cfg.sample_data_conversion_ratio(op, rho_rng) for op in _OPS)
            c, ell = wf_utils.end_to_end_cost_and_latency(
                tup, qprof.s_src_gb, rho, workflow_rng=wf_rng
            )
            costs.append(c)
            lats.append(ell)
            draws += 1
            if c > qprof.theta_cost + cost_tol_abs:
                viol_cost += 1
            if ell > qprof.theta_latency_sec + latency_tol_abs:
                viol_latency += 1

    mean_c = sum(costs) / float(draws)
    mean_t = sum(lats) / float(draws)
    if draws > 1:
        var_c = sum((x - mean_c) ** 2 for x in costs) / float(draws - 1)
        var_t = sum((x - mean_t) ** 2 for x in lats) / float(draws - 1)
        std_c = math.sqrt(max(var_c, 0.0))
        std_l = math.sqrt(max(var_t, 0.0))
    else:
        std_c = 0.0
        std_l = 0.0

    denom = float(draws) if draws > 0 else 1.0
    vr_c = float(viol_cost) / denom
    vr_l = float(viol_latency) / denom

    return EmpiricalDeploymentMetrics(
        aggregate_utility_u=u_val,
        mean_cost_usd=mean_c,
        mean_latency_sec=mean_t,
        slo_cost_violation_rate=vr_c,
        slo_latency_violation_rate=vr_l,
        n_draws_total=draws,
        std_cost_usd=std_c,
        std_latency_sec=std_l,
    )


def format_metrics_lines(
    *,
    algorithm_label: str,
    metrics: EmpiricalDeploymentMetrics,
) -> list[str]:
    """Renderable lines matching the KPI definitions in the paper."""
    hdr = f"===== Evaluation KPIs ({algorithm_label}) ====="
    sep = "=" * max(len(hdr), 50)
    return [
        sep,
        hdr,
        sep,
        f"(1) Aggregate Utility (U):        {metrics.aggregate_utility_u:.6g}",
        f"(2) Total monetary cost - mean (USD over draws): "
        f"{metrics.mean_cost_usd:.6g}  [std~ {metrics.std_cost_usd:.4g}]",
        f"(3) End-to-end latency - mean (seconds over draws): "
        f"{metrics.mean_latency_sec:.6g}  [std~ {metrics.std_latency_sec:.4g}]",
        f"(4a) Cost SLO Violation Rate:     {metrics.slo_cost_violation_rate:.6g} "
        f"(C > Theta_C^q, {metrics.n_draws_total} draws)",
        f"(4b) Latency SLO Violation Rate: {metrics.slo_latency_violation_rate:.6g} "
        f"(T > Theta_T^q, same draws)",
        sep,
    ]


def print_metrics_report(*, algorithm_label: str, metrics: EmpiricalDeploymentMetrics) -> None:
    """Print KPI block to stdout."""
    for line in format_metrics_lines(algorithm_label=algorithm_label, metrics=metrics):
        print(line)
