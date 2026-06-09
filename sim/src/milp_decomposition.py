"""Scenario-adaptive decomposition with constraint generation."""

from __future__ import annotations

import gc
import logging
import math
import random

from src.cost_latency import critical_path_latency, execution_latency
from src.data_loader import load_network_links
from src.data_propagation import output_data_sizes, propagate_data_sizes
from src.evaluator import evaluate_deployment
from src.milp_model import build_milp, extract_deployment, solve_model
from src.schemas import (
    AblationConfig,
    ConvergenceRecord,
    DeploymentAssignment,
    Endpoint,
    NetworkLink,
    OptimizationResult,
    Query,
    Scenario,
    SolverConfig,
    WorkflowDAG,
)

logger = logging.getLogger(__name__)


def _initial_active_keys_by_query(
    workflow: WorkflowDAG,
    quality_level: str,
    endpoints: list[Endpoint],
    endpoint_map: dict[str, Endpoint],
    network_index: dict[tuple[str, str], NetworkLink],
    ablation: AblationConfig,
    queries: list[Query],
    scenario_by_q: dict[str, list[Scenario]],
    fraction: float,
    seed: int,
    strategy: str,
) -> set[tuple[str, str]]:
    """Select the initial active set with per-query scenario coverage."""
    active_keys: set[tuple[str, str]] = set()
    rng = random.Random(seed)
    normalized_strategy = _resolve_initial_active_strategy(
        workflow.name,
        quality_level,
        strategy,
    )

    for q in queries:
        query_scenarios = list(scenario_by_q.get(q.query_id, []))
        if not query_scenarios:
            continue
        init_count = min(
            max(1, math.ceil(len(query_scenarios) * fraction)),
            len(query_scenarios),
        )
        if normalized_strategy == "stratified_random":
            rng.shuffle(query_scenarios)
            selected = query_scenarios[:init_count]
        elif normalized_strategy == "latency_quantile":
            ranked = sorted(
                query_scenarios,
                key=lambda s: _scenario_latency_risk_score(
                    workflow,
                    quality_level,
                    endpoints,
                    endpoint_map,
                    network_index,
                    ablation,
                    q,
                    s,
                ),
            )
            selected = _uniform_quantiles(ranked, init_count)
        else:
            ranked = sorted(
                query_scenarios,
                key=lambda s: _scenario_risk_score(workflow, q, s),
                reverse=True,
            )
            if normalized_strategy == "stratified_tail":
                selected = ranked[:init_count]
            elif normalized_strategy == "stratified_quantile":
                selected = _evenly_spaced(ranked, init_count)
            elif normalized_strategy == "stratified_tail_random":
                tail_count = max(1, math.ceil(init_count / 2))
                selected = ranked[:tail_count]
                selected_keys = {s.scenario_id for s in selected}
                remaining = [s for s in query_scenarios if s.scenario_id not in selected_keys]
                rng.shuffle(remaining)
                selected.extend(remaining[: max(0, init_count - len(selected))])
            else:
                raise ValueError(
                    "Unknown initial_active_strategy="
                    f"{strategy!r}; choose stratified_random, stratified_tail, "
                    "stratified_quantile, latency_quantile, "
                    "stratified_tail_random, or adaptive"
                )
        active_keys.update(
            (q.query_id, s.scenario_id) for s in selected
        )

    return active_keys


def _resolve_initial_active_strategy(
    workflow_name: str,
    quality_level: str,
    strategy: str,
) -> str:
    normalized = strategy.lower().replace("-", "_")
    if normalized != "adaptive":
        return normalized
    if workflow_name == "workflow1" and quality_level == "Q3":
        return "stratified_quantile"
    if workflow_name == "workflow2" and quality_level == "Q1":
        return "stratified_tail"
    return "stratified_random"


def _scenario_risk_score(workflow: WorkflowDAG, query: Query, scenario: Scenario) -> float:
    input_sizes = propagate_data_sizes(workflow, query, scenario)
    outputs = output_data_sizes(input_sizes, scenario, workflow)
    data_score = sum(input_sizes.values()) + sum(outputs.values())
    token_score = (scenario.database_output_tokens or 0.0) + (scenario.q_a_output_tokens or 0.0)
    stress_score = scenario.exec_stress + scenario.bw_stress + scenario.rtt_stress
    rho_score = sum(scenario.rho.values())
    return data_score + token_score / 1000.0 + stress_score + rho_score


def _scenario_latency_risk_score(
    workflow: WorkflowDAG,
    quality_level: str,
    endpoints: list[Endpoint],
    endpoint_map: dict[str, Endpoint],
    network_index: dict[tuple[str, str], NetworkLink],
    ablation: AblationConfig,
    query: Query,
    scenario: Scenario,
) -> float:
    """Estimated latency-to-budget ratio under a per-node fastest reference plan."""
    input_sizes = propagate_data_sizes(workflow, query, scenario)
    output_sizes = output_data_sizes(input_sizes, scenario, workflow)
    assignment: dict[str, Endpoint] = {
        "ClientSource": endpoint_map["client_source"],
        "ClientSink": endpoint_map["client_sink"],
    }

    for node in workflow.node_names():
        if workflow.is_virtual(node):
            continue
        candidates = [
            endpoint
            for endpoint in endpoints
            if endpoint.logical_operation == node
            and endpoint.quality_level == quality_level
            and not endpoint.is_virtual
        ]
        if not candidates:
            continue
        assignment[node] = min(
            candidates,
            key=lambda endpoint: execution_latency(
                endpoint,
                input_sizes.get(node, 0.0),
                output_sizes.get(node, 0.0),
                scenario.exec_latency_multiplier.get(
                    endpoint.endpoint_id,
                    scenario.exec_stress,
                ),
            ),
        )

    estimated_latency = critical_path_latency(
        workflow,
        assignment,
        endpoint_map,
        network_index,
        query,
        scenario,
        ablation,
    )
    return estimated_latency / max(query.sla_sec, 1e-9)


def _evenly_spaced(items: list[Scenario], count: int) -> list[Scenario]:
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    indexes = {
        round(i * (len(items) - 1) / (count - 1))
        for i in range(count)
    }
    selected = [items[i] for i in sorted(indexes)]
    if len(selected) < count:
        selected_ids = {s.scenario_id for s in selected}
        for item in items:
            if item.scenario_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.scenario_id)
                if len(selected) == count:
                    break
    return selected


def _uniform_quantiles(items: list[Scenario], count: int) -> list[Scenario]:
    """Select midpoint empirical quantile representatives."""
    if count >= len(items):
        return list(items)
    indexes = {
        min(len(items) - 1, math.floor((i + 0.5) * len(items) / count))
        for i in range(count)
    }
    selected = [items[i] for i in sorted(indexes)]
    if len(selected) < count:
        selected_ids = {s.scenario_id for s in selected}
        for item in reversed(items):
            if item.scenario_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.scenario_id)
                if len(selected) == count:
                    break
    return selected


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

    active_keys: set[tuple[str, str]] = set()
    if qs_pairs:
        active_keys = _initial_active_keys_by_query(
            workflow,
            quality_level,
            endpoints,
            endpoint_map,
            network_index,
            config.ablation,
            queries,
            scenario_by_q,
            config.initial_active_fraction,
            config.random_seed,
            config.initial_active_strategy,
        )

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
