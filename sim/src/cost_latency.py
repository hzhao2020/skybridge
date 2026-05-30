"""Cost and latency computation for deployment plans."""

from __future__ import annotations

from src.data_propagation import output_data_sizes, propagate_data_sizes
from src.path_utils import enumerate_source_to_sink_paths, path_edges
from src.schemas import (
    AblationConfig,
    Endpoint,
    NetworkLink,
    Query,
    Scenario,
    WorkflowDAG,
)


def filter_endpoints(
    endpoints: list[Endpoint],
    quality_level: str,
    ablation: AblationConfig,
) -> list[Endpoint]:
    filtered = [e for e in endpoints if e.quality_level == quality_level]
    if ablation.fixed_provider:
        filtered = [e for e in filtered if e.provider == ablation.fixed_provider]
    if ablation.fixed_region:
        filtered = [e for e in filtered if e.region == ablation.fixed_region]
    return filtered


def endpoints_by_operation(
    endpoints: list[Endpoint],
) -> dict[str, list[Endpoint]]:
    result: dict[str, list[Endpoint]] = {}
    for ep in endpoints:
        result.setdefault(ep.logical_operation, []).append(ep)
    return result


def build_network_index(links: list[NetworkLink]) -> dict[tuple[str, str], NetworkLink]:
    return {(l.src_endpoint_id, l.dst_endpoint_id): l for l in links}


def execution_cost(
    endpoint: Endpoint,
    input_size_mb: float,
) -> float:
    return endpoint.fixed_cost + input_size_mb * endpoint.cost_per_mb


def storage_cost(
    endpoint: Endpoint,
    input_size_mb: float,
    output_size_mb: float,
    enabled: bool,
) -> float:
    if not enabled:
        return 0.0
    return endpoint.storage_cost_per_mb * (input_size_mb + output_size_mb)


def network_transfer_cost(
    link: NetworkLink,
    output_size_mb: float,
    enabled: bool,
) -> float:
    if not enabled:
        return 0.0
    output_gb = output_size_mb / 1024.0
    return output_gb * link.egress_cost_per_gb


def execution_latency(
    endpoint: Endpoint,
    input_size_mb: float,
    multiplier: float = 1.0,
) -> float:
    return (endpoint.base_latency_sec + input_size_mb * endpoint.latency_per_mb) * multiplier


def network_latency(
    link: NetworkLink,
    output_size_mb: float,
    bandwidth_mult: float = 1.0,
    rtt_mult: float = 1.0,
    enabled: bool = True,
) -> float:
    if not enabled:
        return 0.0
    bw = max(link.bandwidth_mb_per_sec * bandwidth_mult, 1e-6)
    rtt = link.rtt_sec * rtt_mult
    return output_size_mb / bw + rtt / 2.0


def path_latency(
    path: list[str],
    workflow: WorkflowDAG,
    assignment: dict[str, Endpoint],
    endpoint_map: dict[str, Endpoint],
    network_index: dict[tuple[str, str], NetworkLink],
    input_sizes: dict[str, float],
    output_sizes: dict[str, float],
    scenario: Scenario,
    ablation: AblationConfig,
) -> float:
    total = 0.0
    for node in path:
        if workflow.is_virtual(node):
            continue
        ep = assignment.get(node)
        if ep is None:
            continue
        mult = scenario.exec_latency_multiplier.get(ep.endpoint_id, scenario.exec_stress)
        total += execution_latency(ep, input_sizes.get(node, 0.0), mult)

    for src, dst in path_edges(path):
        if not ablation.enable_client_upload_download:
            if src == "ClientSource" or dst == "ClientSink":
                continue
        src_ep = _resolve_endpoint(src, assignment, endpoint_map, workflow)
        dst_ep = _resolve_endpoint(dst, assignment, endpoint_map, workflow)
        if src_ep is None or dst_ep is None:
            continue
        link = network_index.get((src_ep.endpoint_id, dst_ep.endpoint_id))
        if link is None:
            continue
        pair_key = f"{src_ep.endpoint_id}->{dst_ep.endpoint_id}"
        bw_mult = scenario.bandwidth_multiplier.get(pair_key, scenario.bw_stress)
        rtt_mult = scenario.rtt_multiplier.get(pair_key, scenario.rtt_stress)
        total += network_latency(
            link,
            output_sizes.get(src, 0.0),
            bw_mult,
            rtt_mult,
            ablation.enable_network_latency,
        )
    return total


def critical_path_latency(
    workflow: WorkflowDAG,
    assignment: dict[str, Endpoint],
    endpoint_map: dict[str, Endpoint],
    network_index: dict[tuple[str, str], NetworkLink],
    query: Query,
    scenario: Scenario,
    ablation: AblationConfig,
) -> float:
    input_sizes = propagate_data_sizes(workflow, query, scenario)
    output_sizes = output_data_sizes(input_sizes, scenario, workflow)
    paths = enumerate_source_to_sink_paths(workflow)
    latencies = [
        path_latency(
            p,
            workflow,
            assignment,
            endpoint_map,
            network_index,
            input_sizes,
            output_sizes,
            scenario,
            ablation,
        )
        for p in paths
    ]
    return max(latencies) if latencies else 0.0


def total_cost(
    workflow: WorkflowDAG,
    assignment: dict[str, Endpoint],
    endpoint_map: dict[str, Endpoint],
    network_index: dict[tuple[str, str], NetworkLink],
    query: Query,
    scenario: Scenario,
    ablation: AblationConfig,
) -> float:
    input_sizes = propagate_data_sizes(workflow, query, scenario)
    output_sizes = output_data_sizes(input_sizes, scenario, workflow)
    cost = 0.0

    for node, ep in assignment.items():
        if workflow.is_virtual(node):
            continue
        inp = input_sizes.get(node, 0.0)
        out = output_sizes.get(node, 0.0)
        cost += execution_cost(ep, inp)
        cost += storage_cost(ep, inp, out, ablation.enable_storage_cost)

    for edge in workflow.edges:
        src, dst = edge.src, edge.dst
        if not ablation.enable_client_upload_download:
            if src == "ClientSource" or dst == "ClientSink":
                continue
        src_ep = _resolve_endpoint(src, assignment, endpoint_map, workflow)
        dst_ep = _resolve_endpoint(dst, assignment, endpoint_map, workflow)
        if src_ep is None or dst_ep is None:
            continue
        link = network_index.get((src_ep.endpoint_id, dst_ep.endpoint_id))
        if link is None:
            continue
        cost += network_transfer_cost(
            link, output_sizes.get(src, 0.0), ablation.enable_network_cost
        )

    return cost


def _resolve_endpoint(
    node: str,
    assignment: dict[str, Endpoint],
    endpoint_map: dict[str, Endpoint],
    workflow: WorkflowDAG,
) -> Endpoint | None:
    if node in assignment:
        return assignment[node]
    if workflow.is_virtual(node):
        if node == "ClientSource":
            return endpoint_map.get("client_source")
        if node == "ClientSink":
            return endpoint_map.get("client_sink")
        return endpoint_map.get(f"virtual_{node}")
    return None


def compute_cvar_value(
    latencies: list[float],
    slas: list[float],
    eta: float,
) -> float:
    """CVaR_alpha of (T - SLA) at confidence eta."""
    excesses = [max(0.0, t - s) for t, s in zip(latencies, slas)]
    if not excesses:
        return 0.0
    sorted_ex = sorted(excesses, reverse=True)
    n = len(sorted_ex)
    tail_count = max(1, int(n * eta))
    tail_mean = sum(sorted_ex[:tail_count]) / tail_count
    return tail_mean
