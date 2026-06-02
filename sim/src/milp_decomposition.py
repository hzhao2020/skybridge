"""Scenario-adaptive decomposition with constraint generation."""

from __future__ import annotations

import gc
import logging
import math

from src.cost_latency import critical_path_latency
from src.data_loader import load_network_links
from src.evaluator import evaluate_deployment
from src.milp_model import build_milp, extract_deployment, solve_model
from src.schemas import (
    ConvergenceRecord,
    DeploymentAssignment,
    Endpoint,
    OptimizationResult,
    Query,
    Scenario,
    SolverConfig,
    WorkflowDAG,
)

logger = logging.getLogger(__name__)


def solve_decomposition(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
    *,
    stop_on_infeasible: bool = False,
) -> OptimizationResult:
    """
    Scenario-adaptive decomposition: iteratively add violated scenarios
    to the latency excess constraint set.
    """
    qs_pairs: list[tuple[Query, Scenario]] = []
    scenario_by_q: dict[str, list[Scenario]] = {}
    for s in scenarios:
        scenario_by_q.setdefault(s.query_id, []).append(s)
    for q in queries:
        for s in scenario_by_q.get(q.query_id, []):
            qs_pairs.append((q, s))

    all_keys = {(q.query_id, s.scenario_id) for q, s in qs_pairs}
    batch = max(1, math.ceil(len(qs_pairs) * config.active_batch_fraction))
    tol = config.violation_tolerance
    max_iter = config.max_iterations

    active_keys: set[tuple[str, str]] = set()
    if qs_pairs:
        init_count = min(
            max(1, math.ceil(len(qs_pairs) * config.initial_active_fraction)),
            len(qs_pairs),
        )
        active_keys = {
            (qs_pairs[i][0].query_id, qs_pairs[i][1].scenario_id) for i in range(init_count)
        }

    convergence: list[ConvergenceRecord] = []
    assignment: dict = {}
    last_feasible_assignment: dict | None = None
    total_runtime = 0.0
    obj = float("inf")
    status = "UNKNOWN"
    alpha_val = 0.0
    completed_iterations = 0

    network_links = load_network_links()
    network_index = {(l.src_endpoint_id, l.dst_endpoint_id): l for l in network_links}
    endpoint_map = {e.endpoint_id: e for e in endpoints}
    query_map = {q.query_id: q for q in queries}
    scenario_map = {s.scenario_id: s for s in scenarios}
    virtual_map = {
        "ClientSource": endpoint_map["client_source"],
        "ClientSink": endpoint_map["client_sink"],
    }

    for iteration in range(1, max_iter + 1):
        logger.info(
            "Decomposition iter %d: active scenarios=%d",
            iteration,
            len(active_keys),
        )

        artifacts = build_milp(
            workflow=workflow,
            all_endpoints=endpoints,
            queries=queries,
            scenarios=scenarios,
            quality_level=quality_level,
            config=config,
            active_scenario_keys=active_keys,
        )

        x_sol, y_sol, alpha_val, z_sol, status, runtime, obj = solve_model(artifacts)
        total_runtime += runtime
        completed_iterations = iteration
        if x_sol:
            assignment = extract_deployment(artifacts, x_sol)
            last_feasible_assignment = assignment
        elif last_feasible_assignment is not None:
            logger.warning(
                "Iteration %d infeasible (status=%s); reusing last feasible plan",
                iteration,
                status,
            )
            assignment = last_feasible_assignment
            if stop_on_infeasible:
                status = f"{status}_REUSED_LAST_FEASIBLE"
                artifacts.model.dispose()
                gc.collect()
                break
        else:
            artifacts.model.dispose()
            gc.collect()
            raise RuntimeError(f"Decomposition infeasible at iteration {iteration} (status={status})")

        inactive = all_keys - active_keys
        violations: list[tuple[tuple[str, str], float]] = []

        for key in inactive:
            q = query_map[key[0]]
            s = scenario_map[key[1]]
            assign_full = {**assignment, **virtual_map}
            t_val = critical_path_latency(
                workflow,
                assign_full,
                endpoint_map,
                network_index,
                q,
                s,
                config.ablation,
            )
            z_hat = z_sol.get(key, 0.0)
            delta = t_val - q.sla_sec - alpha_val - z_hat
            if delta > tol:
                violations.append((key, delta))

        max_viol = max((v[1] for v in violations), default=0.0)
        convergence.append(
            ConvergenceRecord(
                workflow=workflow.name,
                quality_level=quality_level,
                iteration=iteration,
                active_scenario_count=len(active_keys),
                objective_value=obj,
                max_violation=max_viol,
                num_violated_scenarios=len(violations),
                runtime_sec=runtime,
            )
        )

        if not violations:
            logger.info("Decomposition converged at iteration %d", iteration)
            artifacts.model.dispose()
            gc.collect()
            break

        violations.sort(key=lambda x: x[1], reverse=True)
        for key, _ in violations[:batch]:
            active_keys.add(key)
        artifacts.model.dispose()
        gc.collect()

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
        method="decomposition",
        assignments=deployments,
        objective_value=obj,
        expected_cost=metrics["expected_cost"],
        avg_latency=metrics["avg_latency"],
        p95_latency=metrics["p95_latency"],
        p99_latency=metrics["p99_latency"],
        violation_rate=metrics["violation_rate"],
        cvar_value=metrics["cvar_value"],
        solver_runtime_sec=total_runtime,
        status=status,
        num_iterations=completed_iterations,
        active_scenario_count=len(active_keys),
        convergence_history=[c.model_dump() for c in convergence],
        per_query_scenario_metrics=metrics["per_qs"],
    )
