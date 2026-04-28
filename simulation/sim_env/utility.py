"""
utility.py
Task utility (quality) coefficients for segment / split / caption / query nodes.

Workflow order: segment -> split -> caption -> query. Split utility is always 1.0.
Values follow the user's benchmark table; segment entries marked * in the table
are still stored as 0.90 / 0.85 here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OperationName = Literal["segment", "split", "caption", "query"]

# ---------------------------------------------------------------------------
# Tables (provider for segment; model name for caption / query — matches config.py)
# ---------------------------------------------------------------------------
SEGMENT_UTILITY_BY_PROVIDER: dict[str, float] = {
    "GCP": 0.90,
    "AWS": 0.90,
    "Aliyun": 0.85,
}

# Shared Gemini / Claude / Qwen utilities for vision (caption) and text (query) where applicable.
LLM_UTILITY_BY_MODEL: dict[str, float] = {
    "Gemini 2.5 Pro": 0.78,
    "Gemini 2.5 Flash": 0.72,
    "Claude 3.5 Sonnet": 0.60,
    "Claude 3.5 Haiku": 0.55,
    "Qwen3-VL-Plus": 0.76,
    "Qwen3-VL-Flash": 0.70,
    "Qwen3.5-Plus": 0.87,
    "Qwen3.5-Flash": 0.83,
}

SPLIT_UTILITY = 1.0


@dataclass(frozen=True)
class QueryProfile:
    """One calibration request q: source workload size and chance-constraint budgets."""

    s_src_gb: float
    theta_cost: float
    theta_latency_sec: float


@dataclass(frozen=True)
class PhysicalNode:
    """One deployed step: operation, cloud, region, and LLM model when applicable."""

    operation: OperationName
    provider: str
    region: str
    model: str | None = None


def physical_node_utility(node: PhysicalNode) -> float:
    """Lookup utility for one node; raises if provider/model is unknown for that operation."""
    op = node.operation
    if op == "split":
        return SPLIT_UTILITY
    if op == "segment":
        try:
            return SEGMENT_UTILITY_BY_PROVIDER[node.provider]
        except KeyError as e:
            raise KeyError(
                f"No segment utility for provider={node.provider!r}"
            ) from e
    if op in ("caption", "query"):
        if not node.model:
            raise ValueError(f"{op} node requires a model name")
        try:
            return LLM_UTILITY_BY_MODEL[node.model]
        except KeyError as e:
            raise KeyError(
                f"No LLM utility for model={node.model!r} (operation={op!r})"
            ) from e
    raise ValueError(f"Unknown operation: {op!r}")


def workflow_utility(*nodes: PhysicalNode) -> float:
    """
    Mean utility over exactly four physical nodes (typically segment, split, caption, query).

    Call as ``workflow_utility(seg, spl, cap, qry)`` or ``workflow_utility(*pipeline)``.

    Example::

        seg = PhysicalNode("segment", "GCP", "us-west1")
        spl = PhysicalNode("split", "GCP", "us-west1")
        cap = PhysicalNode("caption", "GCP", "us-west1", "Gemini 2.5 Pro")
        qry = PhysicalNode("query", "GCP", "us-east1", "Gemini 2.5 Flash")
        u = workflow_utility(seg, spl, cap, qry)
    """
    if len(nodes) != 4:
        raise ValueError(f"workflow_utility expects 4 PhysicalNode arguments, got {len(nodes)}")
    return sum(physical_node_utility(n) for n in nodes) / 4.0


if __name__ == "__main__":
    # Example: segment & split on GCP us-west1; caption Gemini Pro; query Gemini Flash on another region.
    node_segment = PhysicalNode("segment", "GCP", "us-west1")
    node_split = PhysicalNode("split", "GCP", "us-west1")
    node_caption = PhysicalNode("caption", "GCP", "us-west1", "Gemini 2.5 Pro")
    node_query = PhysicalNode("query", "GCP", "us-east1", "Gemini 2.5 Flash")

    pipeline = (node_segment, node_split, node_caption, node_query)
    print("mean utility:", workflow_utility(*pipeline))
