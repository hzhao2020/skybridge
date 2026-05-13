"""
utility.py
Task utility (quality) coefficients for segment / split / caption / query nodes,
and for workflow2 visual-intelligence steps (ocr / label_detection / speech_transcription).

Workflow order: segment -> split -> caption -> query. Split utility is always 1.0.
Benchmark values from SkyAPI Evaluation.md (table percentages scaled to [0, 1]).
Caption vs query LLM utilities differ where the doc gives separate columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OperationName = Literal["segment", "split", "caption", "query"]

VisualIntelligenceOperation = Literal["ocr", "label_detection", "speech_transcription"]

# ---------------------------------------------------------------------------
# Tables (provider for segment; model name for caption / query — matches config.py)
# ---------------------------------------------------------------------------
SEGMENT_UTILITY_BY_PROVIDER: dict[str, float] = {
    "GCP": 0.95,
    "AWS": 0.95,
    "Aliyun": 0.95,
}

LLM_CAPTION_UTILITY_BY_MODEL: dict[str, float] = {
    "Gemini 2.5 Pro": 0.713,
    "Gemini 2.5 Flash": 0.652,
    "Amazon Nova Pro": 0.778,
    "Amazon Nova Lite": 0.778,
    "Qwen3-VL-Plus": 0.713,
    "Qwen3-VL-Flash": 0.652,
}

LLM_QUERY_UTILITY_BY_MODEL: dict[str, float] = {
    "Gemini 2.5 Pro": 0.690,
    "Gemini 2.5 Flash": 0.622,
    "Amazon Nova Pro": 0.416,
    "Amazon Nova Lite": 0.404,
    "Qwen3-VL-Plus": 0.677,
    "Qwen3-VL-Flash": 0.625,
}

SPLIT_UTILITY = 1.0

# Workflow 2: Video Intelligence (ocr / label / speech) — provider column in SkyAPI Evaluation.md.
OCR_UTILITY_BY_PROVIDER: dict[str, float] = {
    "GCP": 0.958,
    "AWS": 0.942,
    "Aliyun": 0.948,
}
LABEL_DETECTION_UTILITY_BY_PROVIDER: dict[str, float] = {
    "GCP": 0.93,
    "AWS": 0.89,
    "Aliyun": 0.9316,
}
SPEECH_TRANSCRIPTION_UTILITY_BY_PROVIDER: dict[str, float] = {
    "GCP": 0.973,
    "AWS": 0.958,
    "Aliyun": 0.963,
}

_VI_UTILITY_TABLES: dict[VisualIntelligenceOperation, dict[str, float]] = {
    "ocr": OCR_UTILITY_BY_PROVIDER,
    "label_detection": LABEL_DETECTION_UTILITY_BY_PROVIDER,
    "speech_transcription": SPEECH_TRANSCRIPTION_UTILITY_BY_PROVIDER,
}


def visual_intelligence_utility(
    provider: str,
    vi_operation: VisualIntelligenceOperation,
) -> float:
    """Quality coefficient for workflow2 VI APIs; keyed by cloud (GCP / AWS / Aliyun)."""
    try:
        return _VI_UTILITY_TABLES[vi_operation][provider]
    except KeyError as e:
        raise KeyError(
            f"No VI utility for provider={provider!r} vi_operation={vi_operation!r}"
        ) from e


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
    if op == "caption":
        if not node.model:
            raise ValueError("caption node requires a model name")
        try:
            return LLM_CAPTION_UTILITY_BY_MODEL[node.model]
        except KeyError as e:
            raise KeyError(
                f"No caption utility for model={node.model!r}"
            ) from e
    if op == "query":
        if not node.model:
            raise ValueError("query node requires a model name")
        try:
            return LLM_QUERY_UTILITY_BY_MODEL[node.model]
        except KeyError as e:
            raise KeyError(
                f"No query utility for model={node.model!r}"
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
