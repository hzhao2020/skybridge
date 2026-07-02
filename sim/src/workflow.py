"""Workflow DAG construction from YAML configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.schemas import LogicalEdge, LogicalNode, WorkflowDAG

LOGICAL_OPERATIONS = [
    "Sample",
    "Shot Detection",
    "Split & Sample",
    "Temporal Grounding",
    "Frame Caption",
    "OCR",
    "Label Detection",
    "Speech Transcription",
    "Database",
    "Reason",
]

WORKFLOW_OPERATIONS = {
    "workflow1": {
        "Sample",
        "Reason",
    },
    "workflow2": {
        "Shot Detection",
        "Split & Sample",
        "Frame Caption",
        "Reason",
    },
    "workflow3": {
        "Shot Detection",
        "Split & Sample",
        "Temporal Grounding",
        "Frame Caption",
        "Reason",
    },
    "workflow4": {
        "Shot Detection",
        "Split & Sample",
        "Frame Caption",
        "OCR",
        "Label Detection",
        "Speech Transcription",
        "Database",
        "Reason",
    },
}


def build_workflow_from_yaml(path: Path) -> WorkflowDAG:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    nodes = [LogicalNode(name=n, is_virtual=n in data.get("virtual_nodes", [])) for n in data["nodes"]]
    edges = [LogicalEdge(src=e[0], dst=e[1]) for e in data["edges"]]

    return WorkflowDAG(
        name=data["name"],
        description=data.get("description", ""),
        nodes=nodes,
        edges=edges,
        virtual_nodes=data.get("virtual_nodes", []),
        query_metadata_targets=data.get("query_metadata_targets", []),
    )


def get_workflow(workflow_name: str, config_dir: Path | None = None) -> WorkflowDAG:
    from src.config import CONFIG_DIR, workflow_config_path

    path = workflow_config_path(workflow_name)
    return build_workflow_from_yaml(path)
