"""Full SAA-CVaR MILP solver."""

from __future__ import annotations

import logging

from src.evaluator import evaluate_deployment
from src.milp_model import build_milp, extract_deployment, solve_model
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


def solve_full_milp(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """Solve the full SkyFlow SAA-CVaR MILP over all scenarios."""
    logger.info(
        "Building full MILP: workflow=%s quality=%s queries=%d",
        workflow.name,
        quality_level,
        len(queries),
    )

    artifacts = build_milp(
        workflow=workflow,
        all_endpoints=endpoints,
        queries=queries,
        scenarios=scenarios,
        quality_level=quality_level,
        config=config,
        active_scenario_keys=None,
    )

    x_sol, y_sol, alpha_val, z_sol, status, runtime, obj = solve_model(artifacts)
    if not x_sol:
        raise RuntimeError(f"Full MILP failed to find a solution (status={status})")
    assignment = extract_deployment(artifacts, x_sol)

    metrics = evaluate_deployment(
        workflow=workflow,
        assignment=assignment,
        endpoints=endpoints,
        queries=queries,
        scenarios=scenarios,
        quality_level=quality_level,
        config=config,
        alpha=alpha_val,
    )

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
        method="full_milp",
        assignments=deployments,
        objective_value=obj,
        expected_cost=metrics["expected_cost"],
        avg_latency=metrics["avg_latency"],
        p95_latency=metrics["p95_latency"],
        p99_latency=metrics["p99_latency"],
        violation_rate=metrics["violation_rate"],
        cvar_value=metrics["cvar_value"],
        solver_runtime_sec=runtime,
        status=status,
        num_iterations=1,
        active_scenario_count=len(artifacts.qs_pairs),
        active_path_cut_count=len(artifacts.active_path_cuts),
        per_query_scenario_metrics=metrics["per_qs"],
    )
