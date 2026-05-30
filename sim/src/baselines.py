"""Baseline deployment policies: Single-Cloud, Logical-Optimal, and Greedy."""

from __future__ import annotations

import itertools
import logging
import time
from typing import Callable

from src.capability import endpoint_capability_mu
from src.cost_latency import endpoints_by_operation, filter_endpoints
from src.evaluator import evaluate_deployment
from src.schemas import (
    DeploymentAssignment,
    Endpoint,
    OptimizationResult,
    Query,
    Scenario,
    SolverConfig,
    WorkflowDAG,
)

logger = logging.getLogger(__name__)

# Cap exhaustive single-cloud search (product of per-layer candidates).
_MAX_SC_ASSIGNMENTS = 128

SolveFn = Callable[
    [
        WorkflowDAG,
        list[Endpoint],
        list[Query],
        list[Scenario],
        str,
        SolverConfig,
    ],
    OptimizationResult,
]


def _node_candidates(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    quality_level: str,
    config: SolverConfig,
) -> dict[str, list[Endpoint]]:
    filtered = filter_endpoints(endpoints, quality_level, config.ablation)
    ops_map = endpoints_by_operation(filtered)
    out: dict[str, list[Endpoint]] = {}
    for node in workflow.compute_nodes():
        cands = ops_map.get(node, [])
        if not cands:
            raise ValueError(f"No candidates for {node} at quality {quality_level}")
        out[node] = cands
    return out


def _topological_compute_nodes(workflow: WorkflowDAG) -> list[str]:
    compute = set(workflow.compute_nodes())
    indegree = {n: len([p for p in workflow.predecessors(n) if p in compute]) for n in compute}
    ready = [n for n in compute if indegree[n] == 0]
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for succ in workflow.successors(n):
            if succ not in compute:
                continue
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
    if len(order) != len(compute):
        raise ValueError("Workflow compute nodes are not acyclic")
    return order


def _placeholder_endpoint(candidates: list[Endpoint]) -> Endpoint:
    """Lowest estimated per-query cost endpoint (tie-breaker for unassigned nodes in Greedy)."""

    def key(ep: Endpoint) -> tuple[float, str]:
        est = ep.fixed_cost + ep.cost_per_mb * 100.0
        return (est, ep.endpoint_id)

    return min(candidates, key=key)


def _full_assignment(
    workflow: WorkflowDAG,
    node_candidates: dict[str, list[Endpoint]],
    partial: dict[str, Endpoint],
) -> dict[str, Endpoint]:
    """Complete assignment using placeholders for nodes not yet chosen."""
    full = dict(partial)
    for node in workflow.compute_nodes():
        if node not in full:
            full[node] = _placeholder_endpoint(node_candidates[node])
    return full


def _metrics_for_assignment(
    workflow: WorkflowDAG,
    assignment: dict[str, Endpoint],
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> dict:
    return evaluate_deployment(
        workflow=workflow,
        assignment=assignment,
        endpoints=endpoints,
        queries=queries,
        scenarios=scenarios,
        quality_level=quality_level,
        config=config,
    )


def _rank_violation_cost(violation_rate: float, expected_cost: float) -> tuple[float, float]:
    return (violation_rate, expected_cost)


def _to_result(
    *,
    workflow: WorkflowDAG,
    assignment: dict[str, Endpoint],
    quality_level: str,
    method: str,
    metrics: dict,
    runtime_sec: float,
) -> OptimizationResult:
    deployments = [
        DeploymentAssignment(
            logical_node=node,
            endpoint_id=ep.endpoint_id,
            provider=ep.provider,
            region=ep.region,
            model_name=ep.model_name,
        )
        for node, ep in assignment.items()
        if not workflow.is_virtual(node)
    ]
    return OptimizationResult(
        workflow=workflow.name,
        quality_level=quality_level,
        method=method,
        assignments=deployments,
        objective_value=float(metrics["violation_rate"]),
        expected_cost=float(metrics["expected_cost"]),
        avg_latency=float(metrics["avg_latency"]),
        p95_latency=float(metrics["p95_latency"]),
        p99_latency=float(metrics["p99_latency"]),
        violation_rate=float(metrics["violation_rate"]),
        cvar_value=float(metrics["cvar_value"]),
        solver_runtime_sec=runtime_sec,
        status="optimal",
        per_query_scenario_metrics=metrics.get("per_qs", []),
    )


def solve_logical_optimal(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    LO: per logical node, pick the endpoint with highest μ_k (ignores cost/latency coupling).
    """
    t0 = time.perf_counter()
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)
    assignment: dict[str, Endpoint] = {}
    for node, cands in node_candidates.items():
        assignment[node] = max(cands, key=lambda ep: (endpoint_capability_mu(ep), ep.endpoint_id))

    metrics = _metrics_for_assignment(
        workflow, assignment, endpoints, queries, scenarios, quality_level, config
    )
    return _to_result(
        workflow=workflow,
        assignment=assignment,
        quality_level=quality_level,
        method="logical_optimal",
        metrics=metrics,
        runtime_sec=time.perf_counter() - t0,
    )


def _best_assignment_single_provider(
    workflow: WorkflowDAG,
    node_candidates: dict[str, list[Endpoint]],
    provider: str,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> dict[str, Endpoint] | None:
    """Minimize empirical SLA violation rate within one cloud; ties by expected cost."""
    order = _topological_compute_nodes(workflow)
    layers: list[list[Endpoint]] = []
    for node in order:
        layer = [ep for ep in node_candidates[node] if ep.provider == provider]
        if not layer:
            return None
        layers.append(layer)

    n_combos = 1
    for layer in layers:
        n_combos *= len(layer)
    if n_combos > _MAX_SC_ASSIGNMENTS:
        assignment = {
            node: max(layer, key=lambda ep: (endpoint_capability_mu(ep), ep.endpoint_id))
            for node, layer in zip(order, layers)
        }
        return assignment

    best_assign: dict[str, Endpoint] | None = None
    best_rank: tuple[float, float] | None = None
    for combo in itertools.product(*layers):
        assignment = dict(zip(order, combo))
        metrics = _metrics_for_assignment(
            workflow, assignment, endpoints, queries, scenarios, quality_level, config
        )
        rank = _rank_violation_cost(metrics["violation_rate"], metrics["expected_cost"])
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_assign = assignment
    return best_assign


def solve_single_cloud(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    SC: all nodes on one provider; pick the provider with lowest empirical violation rate.
    """
    t0 = time.perf_counter()
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)
    providers = sorted({ep.provider for eps in node_candidates.values() for ep in eps})

    best_assign: dict[str, Endpoint] | None = None
    best_rank: tuple[float, float, str] | None = None
    best_metrics: dict | None = None

    for prov in providers:
        assign = _best_assignment_single_provider(
            workflow,
            node_candidates,
            prov,
            endpoints,
            queries,
            scenarios,
            quality_level,
            config,
        )
        if assign is None:
            continue
        metrics = _metrics_for_assignment(
            workflow, assign, endpoints, queries, scenarios, quality_level, config
        )
        rank = (*_rank_violation_cost(metrics["violation_rate"], metrics["expected_cost"]), prov)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_assign = assign
            best_metrics = metrics

    if best_assign is None or best_metrics is None:
        raise RuntimeError(
            "Single-Cloud: no provider supports all logical nodes at this quality level"
        )

    logger.info("Single-Cloud selected provider=%s", best_rank[2])
    return _to_result(
        workflow=workflow,
        assignment=best_assign,
        quality_level=quality_level,
        method="single_cloud",
        metrics=best_metrics,
        runtime_sec=time.perf_counter() - t0,
    )


def solve_greedy(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    Greedy: topological order; each step minimizes empirical SLA violation rate given
    fixed predecessors (unassigned nodes use low-cost placeholders for evaluation).
    """
    t0 = time.perf_counter()
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)
    partial: dict[str, Endpoint] = {}

    for node in _topological_compute_nodes(workflow):
        best_ep: Endpoint | None = None
        best_rank: tuple[float, float] | None = None
        for ep in node_candidates[node]:
            trial = dict(partial)
            trial[node] = ep
            full = _full_assignment(workflow, node_candidates, trial)
            metrics = _metrics_for_assignment(
                workflow, full, endpoints, queries, scenarios, quality_level, config
            )
            rank = _rank_violation_cost(metrics["violation_rate"], metrics["expected_cost"])
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_ep = ep
        if best_ep is None:
            raise RuntimeError(f"Greedy: no candidate for {node}")
        partial[node] = best_ep

    metrics = _metrics_for_assignment(
        workflow, partial, endpoints, queries, scenarios, quality_level, config
    )
    return _to_result(
        workflow=workflow,
        assignment=partial,
        quality_level=quality_level,
        method="greedy",
        metrics=metrics,
        runtime_sec=time.perf_counter() - t0,
    )


def solve_murakkab_profile(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    Murakkab-style profile-guided optimizer.

    This approximates Murakkab's profile-guided SLO-aware configuration:
    it starts from the cheapest profiled deployment, then greedily reconfigures
    one logical executor at a time using only calibration samples until the
    empirical latency SLO target is met or no improving reconfiguration remains.
    """
    t0 = time.perf_counter()
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)

    def cheapest_assignment() -> dict[str, Endpoint]:
        return {
            node: min(cands, key=lambda ep: (ep.fixed_cost + ep.cost_per_mb * 100.0, ep.endpoint_id))
            for node, cands in node_candidates.items()
        }

    assignment = cheapest_assignment()
    metrics = _metrics_for_assignment(
        workflow, assignment, endpoints, queries, scenarios, quality_level, config
    )
    target_violation = config.eta
    seen: set[tuple[tuple[str, str], ...]] = set()

    for _ in range(64):
        key = tuple(sorted((node, ep.endpoint_id) for node, ep in assignment.items()))
        if key in seen:
            break
        seen.add(key)
        if metrics["violation_rate"] <= target_violation:
            break

        best_trial: dict[str, Endpoint] | None = None
        best_metrics: dict | None = None
        best_rank: tuple[float, float, float] | None = None
        current_violation = float(metrics["violation_rate"])
        current_cost = float(metrics["expected_cost"])

        for node, cands in node_candidates.items():
            current_ep = assignment[node]
            for ep in cands:
                if ep.endpoint_id == current_ep.endpoint_id:
                    continue
                trial = dict(assignment)
                trial[node] = ep
                trial_metrics = _metrics_for_assignment(
                    workflow, trial, endpoints, queries, scenarios, quality_level, config
                )
                violation = float(trial_metrics["violation_rate"])
                cost = float(trial_metrics["expected_cost"])
                improvement = current_violation - violation
                if improvement <= 1e-12:
                    continue
                # Prefer feasible profiles; otherwise maximize violation reduction with low cost growth.
                feasible_penalty = 0.0 if violation <= target_violation else 1.0
                cost_growth = max(cost - current_cost, 0.0)
                rank = (feasible_penalty, violation, cost_growth / max(improvement, 1e-9))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_trial = trial
                    best_metrics = trial_metrics

        if best_trial is None or best_metrics is None:
            break
        assignment = best_trial
        metrics = best_metrics

    return _to_result(
        workflow=workflow,
        assignment=assignment,
        quality_level=quality_level,
        method="murakkab_profile",
        metrics=metrics,
        runtime_sec=time.perf_counter() - t0,
    )


BASELINE_SOLVERS: dict[str, SolveFn] = {
    "single_cloud": solve_single_cloud,
    "logical_optimal": solve_logical_optimal,
    "greedy": solve_greedy,
    "murakkab_profile": solve_murakkab_profile,
}


def solve_baseline(
    method: str,
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    try:
        solver = BASELINE_SOLVERS[method]
    except KeyError as e:
        raise ValueError(
            f"Unknown baseline {method!r}; choose from {sorted(BASELINE_SOLVERS)}"
        ) from e
    return solver(workflow, endpoints, queries, scenarios, quality_level, config)
