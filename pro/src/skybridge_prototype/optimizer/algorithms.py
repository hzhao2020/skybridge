from __future__ import annotations

import math
import random
import time
from typing import Any

from .evaluation import (
    AssignmentMap,
    critical_path_latency,
    endpoints_by_role,
    evaluate_assignment,
    input_output_sizes,
    node_cost,
    node_latency,
    relay_download_cost,
    relay_download_latency,
    relay_upload_cost,
    relay_upload_latency,
    result_from_assignment,
    source_to_sink_paths,
    topological_nodes,
    total_cost,
)
from .models import PlannerConfig, PlanningQuery, PlanningResult, PlanningScenario, RuntimeEndpoint, RuntimeProfile


def plan_workflow(method: str, profile: RuntimeProfile, config: PlannerConfig) -> PlanningResult:
    normalized = method.lower().replace("-", "_")
    solvers = {
        "logical_optimal": solve_logical_optimal,
        "single_cloud": solve_single_cloud,
        "greedy": solve_greedy,
        "dpgm": solve_dpgm,
        "murakkab_profile": solve_dpgm,
        "mtgp": solve_mtgp,
        "mtgp_3d": solve_mtgp,
        "skyflow": solve_decomposition,
        "decomposition": solve_decomposition,
    }
    try:
        solver = solvers[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown planner {method!r}; choose from {sorted(solvers)}") from exc
    return solver(profile, config)


def solve_logical_optimal(profile: RuntimeProfile, config: PlannerConfig) -> PlanningResult:
    """LO from sim: independently pick the highest-capability endpoint per node."""
    t0 = time.perf_counter()
    candidates = endpoints_by_role(profile)
    assignment = {
        node: max(candidates[node], key=lambda endpoint: (endpoint.capability, endpoint.endpoint_id))
        for node in profile.workflow.nodes
    }
    metrics = evaluate_assignment(profile, assignment, config, include_per_qs=True)
    return result_from_assignment("logical_optimal", "optimal", profile, assignment, metrics, time.perf_counter() - t0)


def solve_single_cloud(profile: RuntimeProfile, config: PlannerConfig) -> PlanningResult:
    """SC from sim: restrict all logical nodes to one provider, then choose the best provider."""
    t0 = time.perf_counter()
    candidates = endpoints_by_role(profile)
    provider_keys = sorted({endpoint.provider_key for endpoint in profile.endpoints})
    covering = [
        provider
        for provider in provider_keys
        if all(any(endpoint.provider_key == provider for endpoint in candidates[node]) for node in profile.workflow.nodes)
    ]
    if not covering:
        raise RuntimeError("single_cloud cannot find one provider covering all workflow nodes")

    best_assignment: AssignmentMap | None = None
    best_rank: tuple[float, float, float, str] | None = None
    best_metrics: dict[str, Any] | None = None
    for provider in covering:
        assignment: AssignmentMap = {}
        for node in topological_nodes(profile):
            role_candidates = [endpoint for endpoint in candidates[node] if endpoint.provider_key == provider]
            assignment[node] = min(
                role_candidates,
                key=lambda endpoint: (_expected_endpoint_cost(profile, node, endpoint), endpoint.endpoint_id),
            )
        metrics = evaluate_assignment(profile, assignment, config)
        violation = float(metrics["violation_rate"])
        cost = float(metrics["expected_cost"])
        rank = (0.0, cost, violation, provider) if violation <= config.eta else (1.0, violation, cost, provider)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_assignment = assignment
            best_metrics = metrics

    if best_assignment is None or best_metrics is None:
        raise RuntimeError("single_cloud failed to produce an assignment")
    final_metrics = evaluate_assignment(profile, best_assignment, config, include_per_qs=True)
    return result_from_assignment("single_cloud", "optimal", profile, best_assignment, final_metrics, time.perf_counter() - t0)


def solve_greedy(profile: RuntimeProfile, config: PlannerConfig) -> PlanningResult:
    """Greedy from sim: topological one-pass, local expected node cost with latency tie-breaker."""
    t0 = time.perf_counter()
    candidates = endpoints_by_role(profile)
    assignment: AssignmentMap = {}
    for node in topological_nodes(profile):
        assignment[node] = min(
            candidates[node],
            key=lambda endpoint: (
                _expected_endpoint_cost(profile, node, endpoint),
                _expected_endpoint_latency(profile, node, endpoint),
                endpoint.endpoint_id,
            ),
        )
    metrics = evaluate_assignment(profile, assignment, config, include_per_qs=True)
    return result_from_assignment("greedy", "optimal", profile, assignment, metrics, time.perf_counter() - t0)


def solve_dpgm(profile: RuntimeProfile, config: PlannerConfig) -> PlanningResult:
    """DPGM from sim: deterministic profile MILP, then fall back to latency slack if infeasible."""
    gp, GRB = _import_gurobi()
    t0 = time.perf_counter()
    profiles = _query_profiles(profile)
    model_data = _build_dpgm_milp(profile, config, profiles, gp, GRB, allow_slack=False)
    model_data["model"].optimize()
    used_slack = False
    if model_data["model"].SolCount == 0:
        model_data = _build_dpgm_milp(profile, config, profiles, gp, GRB, allow_slack=True)
        model_data["model"].optimize()
        used_slack = True
    if model_data["model"].SolCount == 0:
        raise RuntimeError(f"DPGM failed to find a solution; Gurobi status={model_data['model'].Status}")

    assignment = _extract_assignment(profile, model_data["x"])
    metrics = evaluate_assignment(profile, assignment, config, include_per_qs=True)
    status = _gurobi_status(model_data["model"].Status, GRB)
    if used_slack:
        status = f"profile_slack_{status}"
    return result_from_assignment(
        "dpgm",
        status,
        profile,
        assignment,
        metrics,
        time.perf_counter() - t0,
        objective_value=float(model_data["model"].ObjVal),
        active_scenario_count=len(profiles),
    )


def solve_mtgp(profile: RuntimeProfile, config: PlannerConfig) -> PlanningResult:
    """MTGP-style hyper-heuristic: evolve priority rules over runtime profile features."""
    t0 = time.perf_counter()
    rng = random.Random(config.random_seed)
    population_size = max(2, config.mtgp_population_size)
    elite_count = min(max(1, config.mtgp_elite_count), population_size)
    generations = max(1, config.mtgp_generations)
    population = [_random_rule(rng) for _ in range(population_size)]
    best_rule = population[0]
    best_fitness = float("inf")
    history: list[dict[str, Any]] = []

    for generation in range(1, generations + 1):
        scored = sorted(
            ((_mtgp_fitness(profile, rule, config), rule) for rule in population),
            key=lambda item: item[0],
        )
        if scored[0][0] < best_fitness:
            best_fitness = scored[0][0]
            best_rule = scored[0][1]
        history.append(
            {
                "iteration": generation,
                "active_scenario_count": len(profile.queries),
                "objective_value": float(scored[0][0]),
                "max_violation": 0.0,
                "num_violated_scenarios": 0,
            }
        )
        next_population = [rule for _, rule in scored[:elite_count]]
        while len(next_population) < population_size:
            parent_a = rng.choice(scored[: max(elite_count, population_size // 2)])[1]
            parent_b = rng.choice(scored[: max(elite_count, population_size // 2)])[1]
            child = _crossover_rule(parent_a, parent_b, rng)
            if rng.random() < config.mtgp_mutation_rate:
                child = _mutate_rule(child, rng)
            next_population.append(child)
        population = next_population

    assignment = _assignment_from_rule(profile, best_rule)
    metrics = evaluate_assignment(profile, assignment, config, include_per_qs=True)
    return result_from_assignment(
        "mtgp",
        "optimal",
        profile,
        assignment,
        metrics,
        time.perf_counter() - t0,
        objective_value=best_fitness,
        num_iterations=generations,
        active_scenario_count=len(profile.queries),
        convergence_history=history,
    )


def solve_decomposition(profile: RuntimeProfile, config: PlannerConfig) -> PlanningResult:
    """SkyFlow-style active-scenario CVaR MILP with constraint generation."""
    gp, GRB = _import_gurobi()
    t0 = time.perf_counter()
    pairs = _query_scenario_pairs(profile)
    all_keys = {(query.query_id, scenario.scenario_id) for query, scenario in pairs}
    active_keys = _initial_active_keys(profile, config)
    batch = max(1, math.ceil(len(pairs) * config.active_batch_fraction))
    convergence: list[dict[str, Any]] = []

    best_assignment: AssignmentMap | None = None
    status = "UNKNOWN"
    objective = float("inf")
    completed = 0
    for iteration in range(1, config.max_iterations + 1):
        model_data = _build_cvar_milp(profile, config, active_keys, gp, GRB)
        model_data["model"].optimize()
        status = _gurobi_status(model_data["model"].Status, GRB)
        if model_data["model"].SolCount == 0:
            raise RuntimeError(f"decomposition failed to find a solution; Gurobi status={status}")
        assignment = _extract_assignment(profile, model_data["x"])
        best_assignment = assignment
        objective = float(model_data["model"].ObjVal)

        violations: list[tuple[float, tuple[str, str]]] = []
        for query, scenario in pairs:
            key = (query.query_id, scenario.scenario_id)
            latency = critical_path_latency(profile, assignment, query, scenario)
            excess = latency - query.sla_sec
            if excess > config.violation_tolerance and key not in active_keys:
                violations.append((excess, key))

        convergence.append(
            {
                "iteration": iteration,
                "active_scenario_count": len(active_keys),
                "objective_value": objective,
                "max_violation": max((item[0] for item in violations), default=0.0),
                "num_violated_scenarios": len(violations),
            }
        )
        completed = iteration
        if not violations or active_keys == all_keys:
            break
        violations.sort(reverse=True)
        active_keys.update(key for _, key in violations[:batch])

    if best_assignment is None:
        raise RuntimeError("decomposition did not produce an assignment")
    metrics = evaluate_assignment(profile, best_assignment, config, include_per_qs=True)
    return result_from_assignment(
        "decomposition",
        status,
        profile,
        best_assignment,
        metrics,
        time.perf_counter() - t0,
        objective_value=objective,
        num_iterations=completed,
        active_scenario_count=len(active_keys),
        convergence_history=convergence,
    )


def _expected_endpoint_cost(profile: RuntimeProfile, node: str, endpoint: RuntimeEndpoint) -> float:
    values = []
    for query, scenario in _query_scenario_pairs(profile):
        inputs, outputs = input_output_sizes(profile, query, scenario)
        values.append(node_cost(endpoint, inputs[node], outputs[node]))
    return sum(values) / max(len(values), 1)


def _expected_endpoint_latency(profile: RuntimeProfile, node: str, endpoint: RuntimeEndpoint) -> float:
    values = []
    for query, scenario in _query_scenario_pairs(profile):
        inputs, outputs = input_output_sizes(profile, query, scenario)
        values.append(node_latency(endpoint, scenario, inputs[node], outputs[node]))
    return sum(values) / max(len(values), 1)


def _query_scenario_pairs(profile: RuntimeProfile) -> list[tuple[PlanningQuery, PlanningScenario]]:
    scenarios_by_query: dict[str, list[PlanningScenario]] = {}
    for scenario in profile.scenarios:
        scenarios_by_query.setdefault(scenario.query_id, []).append(scenario)
    return [
        (query, scenario)
        for query in profile.queries
        for scenario in scenarios_by_query.get(query.query_id, [])
    ]


def _query_profiles(profile: RuntimeProfile) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[PlanningQuery, PlanningScenario]]] = {}
    for query, scenario in _query_scenario_pairs(profile):
        grouped.setdefault(query.query_id, []).append((query, scenario))
    profiles = []
    candidates = endpoints_by_role(profile)
    for rows in grouped.values():
        query = rows[0][0]
        node_costs: dict[tuple[str, str], float] = {}
        node_latencies: dict[tuple[str, str], float] = {}
        edge_costs: dict[tuple[str, str, str, str], float] = {}
        edge_latencies: dict[tuple[str, str, str, str], float] = {}
        for node in profile.workflow.nodes:
            for endpoint in candidates[node]:
                costs = []
                latencies = []
                for q, scenario in rows:
                    inputs, outputs = input_output_sizes(profile, q, scenario)
                    costs.append(node_cost(endpoint, inputs[node], outputs[node]))
                    latencies.append(node_latency(endpoint, scenario, inputs[node], outputs[node]))
                node_costs[node, endpoint.endpoint_id] = sum(costs) / max(len(costs), 1)
                node_latencies[node, endpoint.endpoint_id] = sum(latencies) / max(len(latencies), 1)
        for src, dst in profile.workflow.edges:
            for src_ep in candidates[src]:
                for dst_ep in candidates[dst]:
                    costs = []
                    latencies = []
                    for q, scenario in rows:
                        _, outputs = input_output_sizes(profile, q, scenario)
                        mb = outputs[src]
                        costs.append(relay_download_cost(profile, src_ep, mb) + relay_upload_cost(profile, dst_ep, mb))
                        latencies.append(
                            relay_download_latency(profile, src_ep, scenario, mb)
                            + relay_upload_latency(profile, dst_ep, scenario, mb)
                        )
                    edge_costs[src, dst, src_ep.endpoint_id, dst_ep.endpoint_id] = sum(costs) / max(len(costs), 1)
                    edge_latencies[src, dst, src_ep.endpoint_id, dst_ep.endpoint_id] = sum(latencies) / max(len(latencies), 1)
        profiles.append(
            {
                "query": query,
                "node_cost": node_costs,
                "node_latency": node_latencies,
                "edge_cost": edge_costs,
                "edge_latency": edge_latencies,
            }
        )
    return profiles


def _initial_active_keys(profile: RuntimeProfile, config: PlannerConfig) -> set[tuple[str, str]]:
    pairs_by_query: dict[str, list[tuple[PlanningQuery, PlanningScenario]]] = {}
    for query, scenario in _query_scenario_pairs(profile):
        pairs_by_query.setdefault(query.query_id, []).append((query, scenario))
    greedy = solve_greedy(profile, config)
    endpoint_by_id = {endpoint.endpoint_id: endpoint for endpoint in profile.endpoints}
    greedy_assignment = {item.role: endpoint_by_id[item.endpoint_id] for item in greedy.assignments}
    rng = random.Random(config.random_seed)
    active: set[tuple[str, str]] = set()
    strategy = config.initial_active_strategy.lower().replace("-", "_")
    for query_id, rows in pairs_by_query.items():
        count = min(max(1, math.ceil(len(rows) * config.initial_active_fraction)), len(rows))
        scored = []
        for query, scenario in rows:
            latency = critical_path_latency(profile, greedy_assignment, query, scenario)
            scored.append(((latency - query.sla_sec) / max(query.sla_sec, 1e-9), query, scenario))
        scored.sort(key=lambda item: (item[0], item[2].scenario_id))
        if strategy in {"qbr", "query_balanced_random"}:
            selected = list(scored)
            rng.shuffle(selected)
            selected = selected[:count]
        elif strategy in {"qbw", "query_balanced_worst"}:
            selected = list(reversed(scored))[:count]
        elif strategy in {"qbb", "query_balanced_best"}:
            selected = scored[:count]
        elif strategy in {"qbq", "query_balanced_quantile"}:
            selected = [scored[idx] for idx in _quantile_indices(len(scored), count)]
        else:
            start = max(0, len(scored) // 2)
            upper = scored[start:] or scored
            selected = [upper[min(len(upper) - 1, idx)] for idx in _quantile_indices(len(upper), count)]
        active.update((query.query_id, scenario.scenario_id) for _, query, scenario in selected)
    return active


def _random_rule(rng: random.Random) -> dict[str, float]:
    return {
        "cost": rng.uniform(-1.0, 1.0),
        "latency": rng.uniform(-1.0, 1.0),
        "relay": rng.uniform(-1.0, 1.0),
        "capability": rng.uniform(-1.0, 1.0),
    }


def _crossover_rule(a: dict[str, float], b: dict[str, float], rng: random.Random) -> dict[str, float]:
    return {key: (a[key] if rng.random() < 0.5 else b[key]) for key in a}


def _mutate_rule(rule: dict[str, float], rng: random.Random) -> dict[str, float]:
    mutated = dict(rule)
    key = rng.choice(list(mutated))
    mutated[key] += rng.gauss(0.0, 0.35)
    return mutated


def _assignment_from_rule(profile: RuntimeProfile, rule: dict[str, float]) -> AssignmentMap:
    candidates = endpoints_by_role(profile)
    assignment: AssignmentMap = {}
    for node in topological_nodes(profile):
        assignment[node] = min(
            candidates[node],
            key=lambda endpoint: (
                _mtgp_endpoint_priority(profile, node, endpoint, rule),
                endpoint.endpoint_id,
            ),
        )
    return assignment


def _mtgp_endpoint_priority(profile: RuntimeProfile, node: str, endpoint: RuntimeEndpoint, rule: dict[str, float]) -> float:
    expected_cost = _expected_endpoint_cost(profile, node, endpoint)
    expected_latency = _expected_endpoint_latency(profile, node, endpoint)
    relay = _expected_relay_latency_for_endpoint(profile, endpoint)
    return (
        rule["cost"] * expected_cost
        + rule["latency"] * expected_latency
        + rule["relay"] * relay
        - rule["capability"] * endpoint.capability
    )


def _expected_relay_latency_for_endpoint(profile: RuntimeProfile, endpoint: RuntimeEndpoint) -> float:
    values = []
    for query, scenario in _query_scenario_pairs(profile):
        values.append(relay_upload_latency(profile, endpoint, scenario, query.video_size_mb))
        values.append(relay_download_latency(profile, endpoint, scenario, query.video_size_mb * 0.01))
    return sum(values) / max(len(values), 1)


def _mtgp_fitness(profile: RuntimeProfile, rule: dict[str, float], config: PlannerConfig) -> float:
    assignment = _assignment_from_rule(profile, rule)
    metrics = evaluate_assignment(profile, assignment, config)
    latency_penalty = max(0.0, metrics["cvar_value"]) * 25.0
    violation_penalty = metrics["violation_rate"] * 1_000.0
    return float(metrics["expected_cost"] + latency_penalty + violation_penalty)


def _build_dpgm_milp(profile: RuntimeProfile, config: PlannerConfig, profiles: list[dict[str, Any]], gp, GRB, *, allow_slack: bool) -> dict[str, Any]:
    candidates = endpoints_by_role(profile)
    model = _new_model("prototype_dpgm", config, gp)
    x = _assignment_vars(model, candidates, GRB)
    y = _edge_vars(model, profile, candidates)
    _add_assignment_constraints(model, x, candidates, gp)
    _add_mccormick_constraints(model, profile, x, y, gp)

    cost_expr = gp.LinExpr()
    for query_profile in profiles:
        for node in profile.workflow.nodes:
            for endpoint in candidates[node]:
                cost_expr += query_profile["node_cost"][node, endpoint.endpoint_id] * x[node, endpoint.endpoint_id]
        for key, var in y.items():
            cost_expr += query_profile["edge_cost"].get(key, 0.0) * var
    cost_expr = cost_expr / max(len(profiles), 1)

    slack_vars = []
    paths = source_to_sink_paths(profile)
    for query_profile in profiles:
        query = query_profile["query"]
        for path_index, path in enumerate(paths):
            expr = gp.LinExpr()
            if path:
                for endpoint in candidates[path[0]]:
                    expr += _profile_avg_source_upload(profile, endpoint, query_profile, path[0]) * x[path[0], endpoint.endpoint_id]
            for idx, node in enumerate(path):
                for endpoint in candidates[node]:
                    expr += query_profile["node_latency"][node, endpoint.endpoint_id] * x[node, endpoint.endpoint_id]
                if idx + 1 < len(path):
                    src = node
                    dst = path[idx + 1]
                    for src_ep in candidates[src]:
                        for dst_ep in candidates[dst]:
                            key = (src, dst, src_ep.endpoint_id, dst_ep.endpoint_id)
                            expr += query_profile["edge_latency"].get(key, 0.0) * y[key]
                else:
                    for endpoint in candidates[node]:
                        expr += _profile_avg_sink_download(profile, endpoint, query_profile, node) * x[node, endpoint.endpoint_id]
            if allow_slack:
                slack = model.addVar(lb=0.0, name=f"slack_{query.query_id}_{path_index}")
                slack_vars.append(slack)
                model.addConstr(expr <= query.sla_sec + slack)
            else:
                model.addConstr(expr <= query.sla_sec)
    if allow_slack:
        model.setObjective(1_000_000.0 * gp.quicksum(slack_vars) + cost_expr, GRB.MINIMIZE)
    else:
        model.setObjective(cost_expr, GRB.MINIMIZE)
    return {"model": model, "x": x, "y": y}


def _build_cvar_milp(profile: RuntimeProfile, config: PlannerConfig, active_keys: set[tuple[str, str]], gp, GRB) -> dict[str, Any]:
    candidates = endpoints_by_role(profile)
    pairs = _query_scenario_pairs(profile)
    active_pairs = [(query, scenario) for query, scenario in pairs if (query.query_id, scenario.scenario_id) in active_keys]
    model = _new_model("prototype_skyflow", config, gp)
    x = _assignment_vars(model, candidates, GRB)
    y = _edge_vars(model, profile, candidates)
    _add_assignment_constraints(model, x, candidates, gp)
    _add_mccormick_constraints(model, profile, x, y, gp)

    objective = _expected_cost_expr(profile, pairs, candidates, x, y, gp)
    if config.latency_tiebreaker_weight > 0:
        objective += config.latency_tiebreaker_weight * _expected_latency_expr(profile, active_pairs, candidates, x, y, gp)

    alpha = model.addVar(lb=-GRB.INFINITY, name="alpha")
    z = {(query.query_id, scenario.scenario_id): model.addVar(lb=0.0, name=f"z_{query.query_id}_{scenario.scenario_id}") for query, scenario in active_pairs}
    if active_pairs:
        model.addConstr(alpha + (1.0 / (max(config.eta, 1e-9) * max(len(pairs), 1))) * gp.quicksum(z.values()) <= 0.0)
        for query, scenario in active_pairs:
            key = (query.query_id, scenario.scenario_id)
            for path in source_to_sink_paths(profile):
                model.addConstr(z[key] >= _path_latency_expr(profile, query, scenario, path, candidates, x, y, gp) - query.sla_sec - alpha)
    model.setObjective(objective, GRB.MINIMIZE)
    return {"model": model, "x": x, "y": y}


def _new_model(name: str, config: PlannerConfig, gp):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = config.gurobi_time_limit_sec
    model.Params.MIPGap = config.gurobi_mip_gap
    model.Params.Seed = config.random_seed
    model.Params.Threads = 1
    return model


def _assignment_vars(model, candidates: dict[str, list[RuntimeEndpoint]], GRB):
    return {
        (node, endpoint.endpoint_id): model.addVar(vtype=GRB.BINARY, name=f"x_{node}_{endpoint.endpoint_id}")
        for node, role_candidates in candidates.items()
        for endpoint in role_candidates
    }


def _edge_vars(model, profile: RuntimeProfile, candidates: dict[str, list[RuntimeEndpoint]]):
    return {
        (src, dst, src_ep.endpoint_id, dst_ep.endpoint_id): model.addVar(lb=0.0, ub=1.0, name=f"y_{src}_{dst}_{src_ep.endpoint_id}_{dst_ep.endpoint_id}")
        for src, dst in profile.workflow.edges
        for src_ep in candidates[src]
        for dst_ep in candidates[dst]
    }


def _add_assignment_constraints(model, x, candidates: dict[str, list[RuntimeEndpoint]], gp) -> None:
    for node, role_candidates in candidates.items():
        model.addConstr(gp.quicksum(x[node, endpoint.endpoint_id] for endpoint in role_candidates) == 1.0)


def _add_mccormick_constraints(model, profile: RuntimeProfile, x, y, gp) -> None:
    for src, dst in profile.workflow.edges:
        for key, var in [(key, var) for key, var in y.items() if key[0] == src and key[1] == dst]:
            _, _, src_eid, dst_eid = key
            xs = x[src, src_eid]
            xd = x[dst, dst_eid]
            model.addConstr(var <= xs)
            model.addConstr(var <= xd)
            model.addConstr(var >= xs + xd - 1.0)


def _expected_cost_expr(profile: RuntimeProfile, pairs, candidates, x, y, gp):
    expr = gp.LinExpr()
    for query, scenario in pairs:
        inputs, outputs = input_output_sizes(profile, query, scenario)
        for node in profile.workflow.nodes:
            for endpoint in candidates[node]:
                expr += node_cost(endpoint, inputs[node], outputs[node]) * x[node, endpoint.endpoint_id]
        for src in _sources(profile):
            for endpoint in candidates[src]:
                expr += relay_upload_cost(profile, endpoint, inputs[src]) * x[src, endpoint.endpoint_id]
        for src, dst in profile.workflow.edges:
            mb = outputs[src]
            for src_ep in candidates[src]:
                for dst_ep in candidates[dst]:
                    key = (src, dst, src_ep.endpoint_id, dst_ep.endpoint_id)
                    expr += (relay_download_cost(profile, src_ep, mb) + relay_upload_cost(profile, dst_ep, mb)) * y[key]
        for sink in _sinks(profile):
            for endpoint in candidates[sink]:
                expr += relay_download_cost(profile, endpoint, outputs[sink]) * x[sink, endpoint.endpoint_id]
    return expr / max(len(pairs), 1)


def _expected_latency_expr(profile: RuntimeProfile, pairs, candidates, x, y, gp):
    expr = gp.LinExpr()
    paths = source_to_sink_paths(profile)
    for query, scenario in pairs:
        for path in paths:
            expr += _path_latency_expr(profile, query, scenario, path, candidates, x, y, gp)
    return expr / max(len(pairs) * max(len(paths), 1), 1)


def _path_latency_expr(profile: RuntimeProfile, query: PlanningQuery, scenario: PlanningScenario, path: list[str], candidates, x, y, gp):
    inputs, outputs = input_output_sizes(profile, query, scenario)
    expr = gp.LinExpr()
    if path:
        for endpoint in candidates[path[0]]:
            expr += relay_upload_latency(profile, endpoint, scenario, inputs[path[0]]) * x[path[0], endpoint.endpoint_id]
    for idx, node in enumerate(path):
        for endpoint in candidates[node]:
            expr += node_latency(endpoint, scenario, inputs[node], outputs[node]) * x[node, endpoint.endpoint_id]
        if idx + 1 < len(path):
            dst = path[idx + 1]
            for src_ep in candidates[node]:
                for dst_ep in candidates[dst]:
                    key = (node, dst, src_ep.endpoint_id, dst_ep.endpoint_id)
                    mb = outputs[node]
                    expr += (
                        relay_download_latency(profile, src_ep, scenario, mb)
                        + relay_upload_latency(profile, dst_ep, scenario, mb)
                    ) * y[key]
        else:
            for endpoint in candidates[node]:
                expr += relay_download_latency(profile, endpoint, scenario, outputs[node]) * x[node, endpoint.endpoint_id]
    return expr


def _extract_assignment(profile: RuntimeProfile, x) -> AssignmentMap:
    endpoint_by_id = {endpoint.endpoint_id: endpoint for endpoint in profile.endpoints}
    assignment: AssignmentMap = {}
    for node in profile.workflow.nodes:
        selected = [
            endpoint_id
            for (role, endpoint_id), var in x.items()
            if role == node and float(var.X) > 0.5
        ]
        if not selected:
            selected = [
                max(
                    ((float(var.X), endpoint_id) for (role, endpoint_id), var in x.items() if role == node),
                    key=lambda item: item[0],
                )[1]
            ]
        assignment[node] = endpoint_by_id[selected[0]]
    return assignment


def _profile_avg_source_upload(profile: RuntimeProfile, endpoint: RuntimeEndpoint, query_profile: dict[str, Any], node: str) -> float:
    query = query_profile["query"]
    scenarios = [scenario for q, scenario in _query_scenario_pairs(profile) if q.query_id == query.query_id]
    values = []
    for scenario in scenarios:
        inputs, _ = input_output_sizes(profile, query, scenario)
        values.append(relay_upload_latency(profile, endpoint, scenario, inputs[node]))
    return sum(values) / max(len(values), 1)


def _profile_avg_sink_download(profile: RuntimeProfile, endpoint: RuntimeEndpoint, query_profile: dict[str, Any], node: str) -> float:
    query = query_profile["query"]
    scenarios = [scenario for q, scenario in _query_scenario_pairs(profile) if q.query_id == query.query_id]
    values = []
    for scenario in scenarios:
        _, outputs = input_output_sizes(profile, query, scenario)
        values.append(relay_download_latency(profile, endpoint, scenario, outputs[node]))
    return sum(values) / max(len(values), 1)


def _quantile_indices(n: int, count: int) -> list[int]:
    if n <= 0 or count <= 0:
        return []
    count = min(count, n)
    out = []
    used: set[int] = set()
    for idx in range(count):
        candidate = min(n - 1, math.floor(((idx + 0.5) / count) * n))
        while candidate in used and candidate + 1 < n:
            candidate += 1
        while candidate in used and candidate - 1 >= 0:
            candidate -= 1
        out.append(candidate)
        used.add(candidate)
    return out


def _sources(profile: RuntimeProfile) -> list[str]:
    dsts = {dst for _, dst in profile.workflow.edges}
    return [node for node in profile.workflow.nodes if node not in dsts]


def _sinks(profile: RuntimeProfile) -> list[str]:
    srcs = {src for src, _ in profile.workflow.edges}
    return [node for node in profile.workflow.nodes if node not in srcs]


def _import_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("This planner requires gurobipy in the active conda environment") from exc
    return gp, GRB


def _gurobi_status(status: int, GRB) -> str:
    mapping = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbd",
        GRB.UNBOUNDED: "unbounded",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
    }
    return mapping.get(status, f"status_{status}")
