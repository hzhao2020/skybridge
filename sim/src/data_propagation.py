"""Data size propagation through workflow DAG."""

from __future__ import annotations

from src.schemas import Query, Scenario, WorkflowDAG
from src.pricing import caption_output_size_mb, tokens_per_mb, video_to_audio_ratio


def propagate_data_sizes(
    workflow: WorkflowDAG,
    query: Query,
    scenario: Scenario,
    default_rho: float = 1.0,
) -> dict[str, float]:
    """
    Compute input data size S_i for each logical node.

    Source nodes receive video_size_mb.
    Downstream: S_i = sum_{j in pred(i)} S_j * rho_j
    """
    sizes: dict[str, float] = {}
    topo_order = _topological_sort(workflow)

    for node in topo_order:
        if node == "ClientSource":
            sizes[node] = query.video_size_mb
        elif workflow.is_virtual(node):
            preds = workflow.predecessors(node)
            sizes[node] = sum(sizes.get(p, 0.0) for p in preds) if preds else 0.0
        elif node == "Speech Transcription":
            sizes[node] = query.video_size_mb * video_to_audio_ratio()
        else:
            preds = workflow.predecessors(node)
            if not preds:
                sizes[node] = query.video_size_mb
            else:
                total = 0.0
                for pred in preds:
                    total += _node_output_size(
                        pred,
                        sizes,
                        scenario,
                        workflow,
                        default_rho,
                        query,
                    )
                sizes[node] = total

    return sizes


def output_data_sizes(
    sizes: dict[str, float],
    scenario: Scenario,
    workflow: WorkflowDAG,
    query: Query | None = None,
    default_rho: float = 1.0,
) -> dict[str, float]:
    """Output size at node i = S_i * rho_i."""
    outputs: dict[str, float] = {}
    for node in workflow.node_names():
        if workflow.is_virtual(node):
            outputs[node] = sizes.get(node, 0.0)
        else:
            outputs[node] = _node_output_size(
                node,
                sizes,
                scenario,
                workflow,
                default_rho,
                query,
            )
    return outputs


def edge_transfer_size(
    src: str,
    dst: str,
    output_sizes: dict[str, float],
    query: Query,
) -> float:
    """Data size transferred on a DAG edge."""
    if src == "ClientSource" and dst == "Speech Transcription":
        return query.video_size_mb * video_to_audio_ratio()
    return output_sizes.get(src, 0.0)


def _node_output_size(
    node: str,
    sizes: dict[str, float],
    scenario: Scenario,
    workflow: WorkflowDAG,
    default_rho: float,
    query: Query | None = None,
) -> float:
    if workflow.is_virtual(node):
        return sizes.get(node, 0.0)
    if (
        node == "Video Caption"
        and scenario.caption_output_tokens_per_frame is not None
        and query is not None
    ):
        return caption_output_size_mb(
            query.video_duration_sec,
            query.quality_level,
            scenario.caption_output_tokens_per_frame,
        )
    if node == "Database" and scenario.database_output_tokens is not None:
        return scenario.database_output_tokens / tokens_per_mb()
    if node == "Q/A" and scenario.q_a_output_tokens is not None:
        return scenario.q_a_output_tokens / tokens_per_mb()
    rho = scenario.rho.get(node, default_rho)
    return sizes.get(node, 0.0) * rho


def _topological_sort(workflow: WorkflowDAG) -> list[str]:
    in_degree = {n: 0 for n in workflow.node_names()}
    for e in workflow.edges:
        in_degree[e.dst] = in_degree.get(e.dst, 0) + 1

    queue = [n for n, d in in_degree.items() if d == 0]
    order: list[str] = []

    adj: dict[str, list[str]] = {n: [] for n in workflow.node_names()}
    for e in workflow.edges:
        adj[e.src].append(e.dst)

    while queue:
        node = queue.pop(0)
        order.append(node)
        for succ in adj.get(node, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(order) != len(workflow.node_names()):
        raise ValueError("Workflow contains a cycle")
    return order
