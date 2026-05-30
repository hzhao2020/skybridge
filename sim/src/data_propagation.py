"""Data size propagation through workflow DAG."""

from __future__ import annotations

from src.schemas import Query, Scenario, WorkflowDAG


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
        else:
            preds = workflow.predecessors(node)
            if not preds:
                sizes[node] = query.video_size_mb
            else:
                total = 0.0
                for pred in preds:
                    rho = scenario.rho.get(pred, default_rho)
                    total += sizes.get(pred, 0.0) * rho
                sizes[node] = total

    return sizes


def output_data_sizes(
    sizes: dict[str, float],
    scenario: Scenario,
    workflow: WorkflowDAG,
    default_rho: float = 1.0,
) -> dict[str, float]:
    """Output size at node i = S_i * rho_i."""
    outputs: dict[str, float] = {}
    for node in workflow.node_names():
        if workflow.is_virtual(node):
            outputs[node] = sizes.get(node, 0.0)
        else:
            rho = scenario.rho.get(node, default_rho)
            outputs[node] = sizes.get(node, 0.0) * rho
    return outputs


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
