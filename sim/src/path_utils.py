"""DAG path enumeration utilities."""

from __future__ import annotations

from src.schemas import WorkflowDAG


def enumerate_source_to_sink_paths(workflow: WorkflowDAG) -> list[list[str]]:
    """Enumerate all paths from ClientSource to ClientSink."""
    source = "ClientSource"
    sink = "ClientSink"

    if source not in workflow.node_names() or sink not in workflow.node_names():
        raise ValueError("Workflow must contain ClientSource and ClientSink")

    paths: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> None:
        if node == sink:
            paths.append(path.copy())
            return
        for succ in workflow.successors(node):
            if succ in path:
                continue
            path.append(succ)
            dfs(succ, path)
            path.pop()

    dfs(source, [source])
    return paths


def path_edges(path: list[str]) -> list[tuple[str, str]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def path_nodes_excluding_virtual(path: list[str], workflow: WorkflowDAG) -> list[str]:
    return [n for n in path if not workflow.is_virtual(n)]
