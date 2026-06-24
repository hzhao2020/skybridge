from __future__ import annotations

import math
from statistics import mean

from .models import (
    PlannerConfig,
    PlanningAssignment,
    PlanningQuery,
    PlanningResult,
    PlanningScenario,
    RuntimeEndpoint,
    RuntimeProfile,
)


AssignmentMap = dict[str, RuntimeEndpoint]


def endpoints_by_role(profile: RuntimeProfile) -> dict[str, list[RuntimeEndpoint]]:
    out: dict[str, list[RuntimeEndpoint]] = {}
    for endpoint in profile.endpoints:
        out.setdefault(endpoint.role, []).append(endpoint)
    return out


def topological_nodes(profile: RuntimeProfile) -> list[str]:
    nodes = profile.workflow.nodes
    indegree = {node: 0 for node in nodes}
    successors: dict[str, list[str]] = {node: [] for node in nodes}
    for src, dst in profile.workflow.edges:
        indegree[dst] += 1
        successors[src].append(dst)
    ready = [node for node in nodes if indegree[node] == 0]
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for succ in successors[node]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
    if len(order) != len(nodes):
        raise ValueError("Workflow contains a cycle")
    return order


def source_to_sink_paths(profile: RuntimeProfile) -> list[list[str]]:
    predecessors = {node: [] for node in profile.workflow.nodes}
    successors = {node: [] for node in profile.workflow.nodes}
    for src, dst in profile.workflow.edges:
        successors[src].append(dst)
        predecessors[dst].append(src)
    sources = [node for node in profile.workflow.nodes if not predecessors[node]]
    sinks = [node for node in profile.workflow.nodes if not successors[node]]
    paths: list[list[str]] = []

    def visit(node: str, path: list[str]) -> None:
        if node in sinks:
            paths.append(path + [node])
            return
        for succ in successors[node]:
            visit(succ, path + [node])

    for source in sources:
        visit(source, [])
    return paths or [profile.workflow.nodes]


def input_output_sizes(
    profile: RuntimeProfile,
    query: PlanningQuery,
    scenario: PlanningScenario,
) -> tuple[dict[str, float], dict[str, float]]:
    inputs: dict[str, float] = {}
    outputs: dict[str, float] = {}
    predecessors = {node: [] for node in profile.workflow.nodes}
    for src, dst in profile.workflow.edges:
        predecessors[dst].append(src)
    for node in topological_nodes(profile):
        if predecessors[node]:
            inputs[node] = sum(outputs[pred] for pred in predecessors[node])
        else:
            inputs[node] = query.video_size_mb
        ratio = scenario.output_ratio.get(node, 1.0)
        outputs[node] = max(0.0, inputs[node] * ratio)
    return inputs, outputs


def node_cost(endpoint: RuntimeEndpoint, input_mb: float, output_mb: float) -> float:
    return (
        endpoint.fixed_cost
        + endpoint.cost_per_mb * (input_mb + output_mb)
        + endpoint.storage_cost_per_mb * (input_mb + output_mb)
    )


def node_latency(
    endpoint: RuntimeEndpoint,
    scenario: PlanningScenario,
    input_mb: float,
    output_mb: float,
) -> float:
    multiplier = scenario.node_latency_multiplier.get(endpoint.endpoint_id, 1.0)
    return max(0.0, (endpoint.base_latency_sec + endpoint.latency_per_mb * (input_mb + output_mb)) * multiplier)


def relay_upload_latency(profile: RuntimeProfile, endpoint: RuntimeEndpoint, scenario: PlanningScenario, mb: float) -> float:
    link = profile.relay_links[endpoint.endpoint_id]
    multiplier = scenario.upload_bandwidth_multiplier.get(endpoint.endpoint_id, 1.0)
    bandwidth = max(link.upload_bandwidth_mb_per_sec * multiplier, 1e-9)
    return mb / bandwidth + link.upload_rtt_sec / 2.0


def relay_download_latency(profile: RuntimeProfile, endpoint: RuntimeEndpoint, scenario: PlanningScenario, mb: float) -> float:
    link = profile.relay_links[endpoint.endpoint_id]
    multiplier = scenario.download_bandwidth_multiplier.get(endpoint.endpoint_id, 1.0)
    bandwidth = max(link.download_bandwidth_mb_per_sec * multiplier, 1e-9)
    return mb / bandwidth + link.download_rtt_sec / 2.0


def relay_upload_cost(profile: RuntimeProfile, endpoint: RuntimeEndpoint, mb: float) -> float:
    return (mb / 1024.0) * profile.relay_links[endpoint.endpoint_id].upload_cost_per_gb


def relay_download_cost(profile: RuntimeProfile, endpoint: RuntimeEndpoint, mb: float) -> float:
    return (mb / 1024.0) * profile.relay_links[endpoint.endpoint_id].download_cost_per_gb


def total_cost(
    profile: RuntimeProfile,
    assignment: AssignmentMap,
    query: PlanningQuery,
    scenario: PlanningScenario,
) -> float:
    inputs, outputs = input_output_sizes(profile, query, scenario)
    cost = 0.0
    for node, endpoint in assignment.items():
        cost += node_cost(endpoint, inputs[node], outputs[node])
    first_nodes = _sources(profile)
    sink_nodes = _sinks(profile)
    for node in first_nodes:
        cost += relay_upload_cost(profile, assignment[node], inputs[node])
    for src, dst in profile.workflow.edges:
        transferred = outputs[src]
        cost += relay_download_cost(profile, assignment[src], transferred)
        cost += relay_upload_cost(profile, assignment[dst], transferred)
    for node in sink_nodes:
        cost += relay_download_cost(profile, assignment[node], outputs[node])
    return cost


def critical_path_latency(
    profile: RuntimeProfile,
    assignment: AssignmentMap,
    query: PlanningQuery,
    scenario: PlanningScenario,
) -> float:
    inputs, outputs = input_output_sizes(profile, query, scenario)
    paths = source_to_sink_paths(profile)
    latencies = []
    for path in paths:
        total = 0.0
        if path:
            total += relay_upload_latency(profile, assignment[path[0]], scenario, inputs[path[0]])
        for idx, node in enumerate(path):
            endpoint = assignment[node]
            total += node_latency(endpoint, scenario, inputs[node], outputs[node])
            if idx + 1 < len(path):
                next_node = path[idx + 1]
                transferred = outputs[node]
                total += relay_download_latency(profile, endpoint, scenario, transferred)
                total += relay_upload_latency(profile, assignment[next_node], scenario, transferred)
            else:
                total += relay_download_latency(profile, endpoint, scenario, outputs[node])
        latencies.append(total)
    return max(latencies) if latencies else 0.0


def evaluate_assignment(
    profile: RuntimeProfile,
    assignment: AssignmentMap,
    config: PlannerConfig,
    *,
    include_per_qs: bool = False,
) -> dict:
    scenario_by_query: dict[str, list[PlanningScenario]] = {}
    for scenario in profile.scenarios:
        scenario_by_query.setdefault(scenario.query_id, []).append(scenario)

    rows: list[dict] = []
    costs: list[float] = []
    latencies: list[float] = []
    slas: list[float] = []
    violations = 0
    total = 0
    for query in profile.queries:
        for scenario in scenario_by_query.get(query.query_id, []):
            cost = total_cost(profile, assignment, query, scenario)
            latency = critical_path_latency(profile, assignment, query, scenario)
            violated = latency > query.sla_sec + config.violation_tolerance
            costs.append(cost)
            latencies.append(latency)
            slas.append(query.sla_sec)
            violations += int(violated)
            total += 1
            if include_per_qs:
                rows.append(
                    {
                        "query_id": query.query_id,
                        "scenario_id": scenario.scenario_id,
                        "cost": cost,
                        "latency": latency,
                        "sla_sec": query.sla_sec,
                        "violated": violated,
                    }
                )
    return {
        "expected_cost": mean(costs) if costs else 0.0,
        "avg_latency": mean(latencies) if latencies else 0.0,
        "p95_latency": percentile(latencies, 95),
        "p99_latency": percentile(latencies, 99),
        "violation_rate": violations / max(total, 1),
        "cvar_value": cvar_excess(latencies, slas, config.eta),
        "per_qs": rows,
    }


def cvar_excess(latencies: list[float], slas: list[float], eta: float) -> float:
    if not latencies:
        return 0.0
    excesses = sorted((latency - sla for latency, sla in zip(latencies, slas)), reverse=True)
    count = max(1, math.ceil(max(min(eta, 1.0), 1e-9) * len(excesses)))
    return mean(excesses[:count])


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ranked = sorted(values)
    if len(ranked) == 1:
        return ranked[0]
    pos = (len(ranked) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ranked[lo]
    return ranked[lo] * (hi - pos) + ranked[hi] * (pos - lo)


def result_from_assignment(
    method: str,
    status: str,
    profile: RuntimeProfile,
    assignment: AssignmentMap,
    metrics: dict,
    runtime_sec: float,
    *,
    objective_value: float | None = None,
    num_iterations: int = 1,
    active_scenario_count: int = 0,
    convergence_history: list[dict] | None = None,
) -> PlanningResult:
    assignments = [
        PlanningAssignment(
            role=node,
            endpoint_id=assignment[node].endpoint_id,
            provider_key=assignment[node].provider_key,
            provider=assignment[node].provider,
            region=assignment[node].region,
            model_name=assignment[node].model_name,
        )
        for node in profile.workflow.nodes
    ]
    selected = {item.role: item.provider_key for item in assignments}
    return PlanningResult(
        method=method,
        status=status,
        assignments=assignments,
        selected_providers=selected,
        objective_value=metrics["expected_cost"] if objective_value is None else objective_value,
        expected_cost=metrics["expected_cost"],
        avg_latency=metrics["avg_latency"],
        p95_latency=metrics["p95_latency"],
        p99_latency=metrics["p99_latency"],
        violation_rate=metrics["violation_rate"],
        cvar_value=metrics["cvar_value"],
        solver_runtime_sec=runtime_sec,
        num_iterations=num_iterations,
        active_scenario_count=active_scenario_count,
        convergence_history=convergence_history or [],
        per_query_scenario_metrics=metrics.get("per_qs", []),
    )


def _sources(profile: RuntimeProfile) -> list[str]:
    dsts = {dst for _, dst in profile.workflow.edges}
    return [node for node in profile.workflow.nodes if node not in dsts]


def _sinks(profile: RuntimeProfile) -> list[str]:
    srcs = {src for src, _ in profile.workflow.edges}
    return [node for node in profile.workflow.nodes if node not in srcs]
