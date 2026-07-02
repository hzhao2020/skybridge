"""Critical-path-aware scenario-path cut generation for SkyFlow."""

from __future__ import annotations

import gc
import logging
import math

from src.cost_latency import critical_path
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


def solve_decomposition_with_initializer_selection(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    calibration_scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    Backward-compatible wrapper for the paper algorithm.

    SkyFlow now starts from an empty scenario-path cut set and lets the
    critical-path separator add violated cuts. The previous train/validation
    initializer selection is intentionally bypassed so callers using the old
    entry point still execute the current algorithm.
    """
    logger.info(
        "Initializer selection is deprecated; running critical-path cut generation "
        "with an empty initial cut set."
    )
    return solve_decomposition(
        workflow,
        endpoints,
        queries,
        calibration_scenarios,
        quality_level,
        config,
    )


def solve_decomposition(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
    *,
    stop_on_infeasible: bool = False,
    initial_active_keys: set[tuple[str, str]] | None = None,
) -> OptimizationResult:
    """
    Critical-path-aware scenario-path cutting plane: iteratively add violated
    scenario-path latency excess constraints to the restricted master MILP.
    """
    qs_pairs: list[tuple[Query, Scenario]] = []
    scenario_by_q: dict[str, list[Scenario]] = {}
    for s in scenarios:
        scenario_by_q.setdefault(s.query_id, []).append(s)
    for q in queries:
        for s in scenario_by_q.get(q.query_id, []):
            qs_pairs.append((q, s))

    if config.top_k > 0:
        batch = config.top_k
    else:
        if config.active_batch_fraction <= 0:
            raise ValueError("active_batch_fraction must be positive when top_k <= 0")
        batch = max(1, math.ceil(len(qs_pairs) * config.active_batch_fraction))
    max_iter = config.max_iterations

    convergence: list[ConvergenceRecord] = []
    assignment: dict = {}
    last_feasible_assignment: dict | None = None
    total_runtime = 0.0
    obj = float("inf")
    status = "UNKNOWN"
    alpha_val = 0.0
    completed_iterations = 0
    converged = False

    network_links = load_network_links()
    network_index = {(l.src_endpoint_id, l.dst_endpoint_id): l for l in network_links}
    endpoint_map = {e.endpoint_id: e for e in endpoints}
    virtual_map = {
        "ClientSource": endpoint_map["client_source"],
        "ClientSink": endpoint_map["client_sink"],
    }

    active_path_cuts: set[tuple[str, str, tuple[str, ...]]] = set()
    active_keys: set[tuple[str, str]] = set()
    if initial_active_keys is not None:
        logger.info(
            "Ignoring deprecated initial_active_keys; SkyFlow initializes "
            "C^(0)=empty and separates violated critical-path cuts."
        )

    for iteration in range(1, max_iter + 1):
        logger.info(
            "Decomposition iter %d: active cuts=%d active scenarios=%d",
            iteration,
            len(active_path_cuts),
            len(active_keys),
        )

        artifacts = build_milp(
            workflow=workflow,
            all_endpoints=endpoints,
            queries=queries,
            scenarios=scenarios,
            quality_level=quality_level,
            config=config,
            active_path_cuts=active_path_cuts,
        )

        x_sol, y_sol, alpha_val, z_sol, status, runtime, obj = solve_model(artifacts)
        total_runtime += runtime
        completed_iterations = iteration
        if x_sol:
            assignment = extract_deployment(artifacts, x_sol)
            last_feasible_assignment = assignment
        elif last_feasible_assignment is not None:
            logger.warning(
                "Iteration %d failed (status=%s); keeping last feasible restricted plan "
                "only for diagnostics",
                iteration,
                status,
            )
            assignment = last_feasible_assignment
            status = f"{status}_AFTER_{len(active_path_cuts)}_ACTIVE_CUTS"
            artifacts.model.dispose()
            gc.collect()
            if stop_on_infeasible:
                break
            raise RuntimeError(
                f"Decomposition master failed at iteration {iteration} "
                f"with {len(active_path_cuts)} active cuts (status={status})"
            )
        else:
            artifacts.model.dispose()
            gc.collect()
            raise RuntimeError(f"Decomposition infeasible at iteration {iteration} (status={status})")

        new_violated_cuts: list[tuple[tuple[str, str, tuple[str, ...]], float]] = []
        max_critical_path_violation = 0.0
        max_new_cut_violation = 0.0
        active_cut_violation_count = 0

        assign_full = {**assignment, **virtual_map}
        for q, s in qs_pairs:
            key = (q.query_id, s.scenario_id)
            path, t_val = critical_path(
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
            max_critical_path_violation = max(max_critical_path_violation, delta)
            if delta <= 0:
                continue
            cut = (q.query_id, s.scenario_id, tuple(path))
            if cut in active_path_cuts:
                active_cut_violation_count += 1
                continue
            if cut[2]:
                new_violated_cuts.append((cut, delta))
                max_new_cut_violation = max(max_new_cut_violation, delta)

        active_keys = {(qid, sid) for qid, sid, _ in active_path_cuts}
        num_new_violated_scenarios = len(
            {(qid, sid) for qid, sid, _ in (cut for cut, _ in new_violated_cuts)}
        )
        convergence.append(
            ConvergenceRecord(
                workflow=workflow.name,
                quality_level=quality_level,
                iteration=iteration,
                active_scenario_count=len(active_keys),
                active_path_cut_count=len(active_path_cuts),
                objective_value=obj,
                max_violation=max(max_critical_path_violation, 0.0),
                max_new_cut_violation=max(max_new_cut_violation, 0.0),
                num_violated_scenarios=num_new_violated_scenarios,
                num_violated_cuts=len(new_violated_cuts),
                active_cut_violation_count=active_cut_violation_count,
                active_cut_batch_size=batch,
                runtime_sec=runtime,
            )
        )

        if not new_violated_cuts:
            logger.info("Decomposition converged at iteration %d", iteration)
            converged = True
            artifacts.model.dispose()
            gc.collect()
            break

        new_violated_cuts.sort(key=lambda x: x[1], reverse=True)
        new_cuts = [cut for cut, _ in new_violated_cuts[:batch]]
        for cut in new_cuts[:batch]:
            active_path_cuts.add(cut)
        active_keys = {(qid, sid) for qid, sid, _ in active_path_cuts}
        artifacts.model.dispose()
        gc.collect()

    if not converged:
        status = f"{status}_MAX_ITERATIONS_REACHED"
        logger.warning(
            "Decomposition reached max_iterations=%d before convergence",
            max_iter,
        )

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
        active_path_cut_count=len(active_path_cuts),
        convergence_history=[c.model_dump() for c in convergence],
        per_query_scenario_metrics=metrics["per_qs"],
    )
