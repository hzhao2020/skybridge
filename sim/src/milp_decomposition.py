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
from src.experiment_protocol import split_calibration_train_validation
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


def _deployment_assignment_map(
    result: OptimizationResult,
    endpoints: list[Endpoint],
) -> dict[str, Endpoint]:
    endpoint_by_id = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
    return {
        assignment.logical_node: endpoint_by_id[assignment.endpoint_id]
        for assignment in result.assignments
    }


def _initializer_validation_rank(metrics: dict, eta: float) -> tuple[float, float, float]:
    violation_rate = float(metrics["violation_rate"])
    expected_cost = float(metrics["expected_cost"])
    if violation_rate <= eta:
        return (0.0, expected_cost, violation_rate)
    return (1.0, violation_rate, expected_cost)


def _initial_active_key_sets_by_query(
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
    strategies: list[str],
    config: SolverConfig,
) -> dict[str, set[tuple[str, str]]]:
    """Build initial active sets for multiple strategies, sharing Greedy scores."""
    normalized = [
        _resolve_initial_active_strategy(workflow.name, quality_level, strategy)
        for strategy in strategies
    ]
    greedy_scores: dict[tuple[str, str], float] | None = None
    if any(strategy in {"qbw", "qbb", "qbq"} for strategy in normalized):
        greedy_scores = _greedy_latency_excess_scores(
            workflow=workflow,
            quality_level=quality_level,
            endpoints=endpoints,
            endpoint_map=endpoint_map,
            network_index=network_index,
            ablation=ablation,
            queries=queries,
            scenario_by_q=scenario_by_q,
            config=config,
        )

    out: dict[str, set[tuple[str, str]]] = {}
    for raw, strategy in zip(strategies, normalized):
        out[raw] = _initial_active_keys_by_query(
            workflow,
            quality_level,
            endpoints,
            endpoint_map,
            network_index,
            ablation,
            queries,
            scenario_by_q,
            fraction,
            seed,
            strategy,
            greedy_scores=greedy_scores,
        )
    return out


def _greedy_latency_excess_scores(
    *,
    workflow: WorkflowDAG,
    quality_level: str,
    endpoints: list[Endpoint],
    endpoint_map: dict[str, Endpoint],
    network_index: dict[tuple[str, str], NetworkLink],
    ablation: AblationConfig,
    queries: list[Query],
    scenario_by_q: dict[str, list[Scenario]],
    config: SolverConfig,
) -> dict[tuple[str, str], float]:
    """Signed normalized latency excess g[q,s] under one Greedy reference plan."""
    from src.baselines import solve_greedy

    scenarios = [
        scenario
        for query in queries
        for scenario in scenario_by_q.get(query.query_id, [])
    ]
    greedy_result = solve_greedy(
        workflow,
        endpoints,
        queries,
        scenarios,
        quality_level,
        config,
    )
    assignment = _deployment_assignment_map(greedy_result, endpoints)
    scores: dict[tuple[str, str], float] = {}
    for query in queries:
        for scenario in scenario_by_q.get(query.query_id, []):
            latency = critical_path_latency(
                workflow,
                assignment,
                endpoint_map,
                network_index,
                query,
                scenario,
                ablation,
            )
            scores[query.query_id, scenario.scenario_id] = (
                latency - query.sla_sec
            ) / max(query.sla_sec, 1e-9)
    return scores


def solve_decomposition_with_initializer_selection(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    calibration_scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    Select SkyFlow's initializer on a calibration train/validation split, then
    re-run on the full calibration set with the selected initializer.
    """
    train_scenarios, validation_scenarios = split_calibration_train_validation(
        calibration_scenarios,
        validation_fraction=config.initializer_validation_fraction,
    )
    candidates = [
        candidate
        for candidate in dict.fromkeys(config.initializer_selection_candidates)
        if candidate
    ]
    if not validation_scenarios or not candidates:
        return solve_decomposition(
            workflow,
            endpoints,
            queries,
            calibration_scenarios,
            quality_level,
            config,
        )

    network_links = load_network_links()
    network_index = {(link.src_endpoint_id, link.dst_endpoint_id): link for link in network_links}
    endpoint_map = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
    train_scenario_by_q: dict[str, list[Scenario]] = {}
    for scenario in train_scenarios:
        train_scenario_by_q.setdefault(scenario.query_id, []).append(scenario)
    train_initial_active_sets = _initial_active_key_sets_by_query(
        workflow=workflow,
        quality_level=quality_level,
        endpoints=endpoints,
        endpoint_map=endpoint_map,
        network_index=network_index,
        ablation=config.ablation,
        queries=queries,
        scenario_by_q=train_scenario_by_q,
        fraction=config.initial_active_fraction,
        seed=config.random_seed,
        strategies=candidates,
        config=config,
    )

    selection_history: list[dict] = []
    best_strategy: str | None = None
    best_rank: tuple[float, float, float, str] | None = None

    for strategy in candidates:
        candidate_config = config.model_copy(
            update={"initial_active_strategy": strategy}
        )
        try:
            candidate_result = solve_decomposition(
                workflow,
                endpoints,
                queries,
                train_scenarios,
                quality_level,
                candidate_config,
                stop_on_infeasible=True,
                initial_active_keys=train_initial_active_sets[strategy],
            )
            assignment = _deployment_assignment_map(candidate_result, endpoints)
            metrics = evaluate_deployment(
                workflow=workflow,
                assignment=assignment,
                endpoints=endpoints,
                queries=queries,
                scenarios=validation_scenarios,
                quality_level=quality_level,
                config=candidate_config,
            )
            rank_base = _initializer_validation_rank(metrics, config.eta)
            rank = (*rank_base, strategy)
            row = {
                "initializer": strategy,
                "status": candidate_result.status,
                "train_scenario_count": len(train_scenarios),
                "validation_scenario_count": len(validation_scenarios),
                "expected_cost": float(metrics["expected_cost"]),
                "avg_latency": float(metrics["avg_latency"]),
                "p95_latency": float(metrics["p95_latency"]),
                "p99_latency": float(metrics["p99_latency"]),
                "violation_rate": float(metrics["violation_rate"]),
                "cvar_value": float(metrics["cvar_value"]),
                "solver_runtime_sec": candidate_result.solver_runtime_sec,
                "num_iterations": candidate_result.num_iterations,
                "active_scenario_count": candidate_result.active_scenario_count,
                "selection_feasible": float(metrics["violation_rate"]) <= config.eta,
                "selection_rank": list(rank_base),
            }
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_strategy = strategy
        except Exception as exc:  # noqa: BLE001 - keep selecting other initializers
            row = {
                "initializer": strategy,
                "status": "ERROR",
                "train_scenario_count": len(train_scenarios),
                "validation_scenario_count": len(validation_scenarios),
                "error": f"{type(exc).__name__}: {exc}",
            }
        selection_history.append(row)

    if best_strategy is None:
        errors = "; ".join(
            f"{row['initializer']}: {row.get('error', row.get('status', 'ERROR'))}"
            for row in selection_history
        )
        raise RuntimeError(f"No SkyFlow initializer could be selected: {errors}")

    final_config = config.model_copy(
        update={"initial_active_strategy": best_strategy}
    )
    calibration_scenario_by_q: dict[str, list[Scenario]] = {}
    for scenario in calibration_scenarios:
        calibration_scenario_by_q.setdefault(scenario.query_id, []).append(scenario)
    final_initial_active_sets = _initial_active_key_sets_by_query(
        workflow=workflow,
        quality_level=quality_level,
        endpoints=endpoints,
        endpoint_map=endpoint_map,
        network_index=network_index,
        ablation=config.ablation,
        queries=queries,
        scenario_by_q=calibration_scenario_by_q,
        fraction=config.initial_active_fraction,
        seed=config.random_seed,
        strategies=[best_strategy],
        config=final_config,
    )
    final_result = solve_decomposition(
        workflow,
        endpoints,
        queries,
        calibration_scenarios,
        quality_level,
        final_config,
        initial_active_keys=final_initial_active_sets[best_strategy],
    )
    final_result.selected_initializer = best_strategy
    final_result.initializer_selection_history = selection_history
    return final_result


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
    *,
    greedy_scores: dict[tuple[str, str], float] | None = None,
) -> set[tuple[str, str]]:
    """Select the initial active set with per-query scenario coverage."""
    active_keys: set[tuple[str, str]] = set()
    rng = random.Random(seed)
    normalized_strategy = _resolve_initial_active_strategy(
        workflow.name,
        quality_level,
        strategy,
    )

    if normalized_strategy == "data_budget_tertile":
        return _initial_active_keys_data_budget_tertile(
            workflow,
            queries,
            scenario_by_q,
            seed,
        )

    for q in queries:
        query_scenarios = list(scenario_by_q.get(q.query_id, []))
        if not query_scenarios:
            continue
        init_count = min(
            max(1, math.ceil(len(query_scenarios) * fraction)),
            len(query_scenarios),
        )
        if normalized_strategy == "qbr":
            rng.shuffle(query_scenarios)
            selected = query_scenarios[:init_count]
        elif normalized_strategy in {"qbw", "qbb", "qbq"}:
            if greedy_scores is None:
                greedy_scores = _greedy_latency_excess_scores(
                    workflow=workflow,
                    quality_level=quality_level,
                    endpoints=endpoints,
                    endpoint_map=endpoint_map,
                    network_index=network_index,
                    ablation=ablation,
                    queries=queries,
                    scenario_by_q=scenario_by_q,
                    config=SolverConfig(
                        random_seed=seed,
                        ablation=ablation,
                    ),
                )
            if normalized_strategy == "qbw":
                ranked = sorted(
                    query_scenarios,
                    key=lambda s: (
                        -greedy_scores.get((q.query_id, s.scenario_id), 0.0),
                        s.scenario_id,
                    ),
                )
                selected = ranked[:init_count]
            elif normalized_strategy == "qbb":
                ranked = sorted(
                    query_scenarios,
                    key=lambda s: (
                        greedy_scores.get((q.query_id, s.scenario_id), 0.0),
                        s.scenario_id,
                    ),
                )
                selected = ranked[:init_count]
            else:
                ranked = sorted(
                    query_scenarios,
                    key=lambda s: (
                        greedy_scores.get((q.query_id, s.scenario_id), 0.0),
                        s.scenario_id,
                    ),
                )
                selected = [
                    ranked[i]
                    for i in _unique_midpoint_quantile_indices(
                        len(ranked),
                        init_count,
                    )
                ]
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
                    f"{strategy!r}; choose qbr, qbw, qbb, qbq, "
                    "stratified_random, stratified_tail, stratified_quantile, latency_quantile, "
                    "stratified_tail_random, data_budget_tertile, or adaptive"
                )
        active_keys.update(
            (q.query_id, s.scenario_id) for s in selected
        )

    return active_keys


def _query_data_budget_ratio(workflow: WorkflowDAG, query: Query) -> float:
    """Ratio of nominal workflow data volume to latency budget (SLA)."""
    scenario = Scenario(query_id=query.query_id, scenario_id="_nominal")
    sizes = propagate_data_sizes(workflow, query, scenario)
    outputs = output_data_sizes(sizes, scenario, workflow)
    data_amount = sum(sizes.values()) + sum(outputs.values())
    return data_amount / max(query.sla_sec, 1e-9)


def _tertile_split_bounds(n: int) -> tuple[int, int]:
    """Return slice bounds (top_end, mid_end) for equal-ish tertiles."""
    if n <= 0:
        return 0, 0
    base = n // 3
    remainder = n % 3
    top_end = base + (1 if remainder >= 1 else 0)
    mid_end = top_end + base + (1 if remainder >= 2 else 0)
    return top_end, mid_end


def _initial_active_keys_data_budget_tertile(
    workflow: WorkflowDAG,
    queries: list[Query],
    scenario_by_q: dict[str, list[Scenario]],
    seed: int,
) -> set[tuple[str, str]]:
    """
    Rank queries by data_amount / budget, split into tertiles, then randomly
    sample 15 / 10 / 5 calibration scenarios for high / mid / low tiers.
    """
    sorted_queries = sorted(
        queries,
        key=lambda q: _query_data_budget_ratio(workflow, q),
        reverse=True,
    )
    top_end, mid_end = _tertile_split_bounds(len(sorted_queries))
    tier_specs: list[tuple[list[Query], int]] = [
        (sorted_queries[:top_end], 15),
        (sorted_queries[top_end:mid_end], 10),
        (sorted_queries[mid_end:], 5),
    ]

    active_keys: set[tuple[str, str]] = set()
    rng = random.Random(seed)
    for tier_queries, init_count in tier_specs:
        for q in tier_queries:
            query_scenarios = list(scenario_by_q.get(q.query_id, []))
            if not query_scenarios:
                continue
            count = min(init_count, len(query_scenarios))
            rng.shuffle(query_scenarios)
            active_keys.update(
                (q.query_id, s.scenario_id) for s in query_scenarios[:count]
            )
    return active_keys


def _resolve_initial_active_strategy(
    workflow_name: str,
    quality_level: str,
    strategy: str,
) -> str:
    normalized = strategy.lower().replace("-", "_")
    aliases = {
        "stratified_random": "qbr",
        "query_balanced_random": "qbr",
        "query_balanced_worst": "qbw",
        "query_balanced_worst_violation": "qbw",
        "query_balanced_best": "qbb",
        "query_balanced_best_slack": "qbb",
        "query_balanced_quantile": "qbq",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized != "adaptive":
        return normalized
    if workflow_name == "workflow1" and quality_level == "Q3":
        return "stratified_quantile"
    if workflow_name == "workflow2" and quality_level == "Q1":
        return "stratified_tail"
    return "qbr"


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


def _unique_midpoint_quantile_indices(n: int, count: int) -> list[int]:
    """Midpoint empirical-quantile indices with deterministic duplicate repair."""
    if n <= 0 or count <= 0:
        return []
    count = min(count, n)
    selected: list[int] = []
    used: set[int] = set()
    for j in range(count):
        idx = min(n - 1, max(0, math.floor(((j + 0.5) / count) * n)))
        if idx in used:
            idx = _nearest_unused_index(idx, n, used)
        selected.append(idx)
        used.add(idx)
    return selected


def _nearest_unused_index(idx: int, n: int, used: set[int]) -> int:
    for delta in range(1, n):
        left = idx - delta
        if left >= 0 and left not in used:
            return left
        right = idx + delta
        if right < n and right not in used:
            return right
    raise ValueError("No unused quantile index available")


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
    if initial_active_keys is not None:
        active_keys = set(initial_active_keys) & all_keys
    elif qs_pairs:
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
