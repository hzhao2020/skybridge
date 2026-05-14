"""
utility.py
Task utility (quality) coefficients for shot_detection / video_split / video_caption / query nodes,
and for workflow2 visual-intelligence steps (ocr / label_detection / speech_transcription).

Workflow order (paper): shot_detection -> video_split -> video_caption -> query. Video-split raw quality is constant 1.0.

Returned node utilities are **normalized per operation**: each lookup is divided by the
maximum value in that operation's table (shot_detection / video_caption / query / VI sub-operation),
so coefficients lie in ``[0, 1]`` relative to the best tabulated option.

Benchmark raw values from SkyAPI Evaluation.md (table percentages scaled to [0, 1]).
Caption vs query LLM utilities differ where the doc gives separate columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OperationName = Literal["shot_detection", "video_split", "video_caption", "query"]

VisualIntelligenceOperation = Literal["ocr", "label_detection", "speech_transcription"]

# ---------------------------------------------------------------------------
# Tables (provider for shot_detection; model name for video_caption / query — matches config.py)
# ---------------------------------------------------------------------------
SHOT_DETECTION_UTILITY_BY_PROVIDER: dict[str, float] = {
    # SkyAPI Evaluation.md：三云列表值均为 95%（缩放后 0.95）；max 相同 ⇒ 归一化后恒为 1。
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

VIDEO_SPLIT_UTILITY = 1.0  # Evaluation.md Utility 表无单独 video_split 行，默认可视为 1

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

_MAX_SHOT_DETECTION_UTILITY = max(SHOT_DETECTION_UTILITY_BY_PROVIDER.values())
_MAX_CAPTION_UTILITY = max(LLM_CAPTION_UTILITY_BY_MODEL.values())
_MAX_QUERY_UTILITY = max(LLM_QUERY_UTILITY_BY_MODEL.values())
_MAX_VIDEO_SPLIT_UTILITY = VIDEO_SPLIT_UTILITY
_MAX_VI_UTILITY_BY_OP: dict[VisualIntelligenceOperation, float] = {
    op: max(tbl.values()) for op, tbl in _VI_UTILITY_TABLES.items()
}


def visual_intelligence_utility(
    provider: str,
    vi_operation: VisualIntelligenceOperation,
) -> float:
    """Normalized quality in ``[0, 1]`` for workflow2 VI APIs (divide by max in that VI table)."""
    try:
        raw = _VI_UTILITY_TABLES[vi_operation][provider]
        return raw / _MAX_VI_UTILITY_BY_OP[vi_operation]
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
    """
    Normalized utility in ``[0, 1]`` for one node (raw table value divided by max in that operation).
    Raises if provider/model is unknown for that operation.
    """
    op = node.operation
    if op == "video_split":
        return VIDEO_SPLIT_UTILITY / _MAX_VIDEO_SPLIT_UTILITY
    if op == "shot_detection":
        try:
            return SHOT_DETECTION_UTILITY_BY_PROVIDER[node.provider] / _MAX_SHOT_DETECTION_UTILITY
        except KeyError as e:
            raise KeyError(
                f"No shot_detection utility for provider={node.provider!r}"
            ) from e
    if op == "video_caption":
        if not node.model:
            raise ValueError("video_caption node requires a model name")
        try:
            return LLM_CAPTION_UTILITY_BY_MODEL[node.model] / _MAX_CAPTION_UTILITY
        except KeyError as e:
            raise KeyError(
                f"No caption utility for model={node.model!r}"
            ) from e
    if op == "query":
        if not node.model:
            raise ValueError("query node requires a model name")
        try:
            return LLM_QUERY_UTILITY_BY_MODEL[node.model] / _MAX_QUERY_UTILITY
        except KeyError as e:
            raise KeyError(
                f"No query utility for model={node.model!r}"
            ) from e
    raise ValueError(f"Unknown operation: {op!r}")


def workflow_utility(*nodes: PhysicalNode) -> float:
    """
    Mean utility over exactly four physical nodes (typically shot_detection, video_split, video_caption, query).

    Call as ``workflow_utility(sd, vs, vc, qry)`` or ``workflow_utility(*pipeline)``.

    Example::

        sd = PhysicalNode("shot_detection", "GCP", "us-west1")
        vs = PhysicalNode("video_split", "GCP", "us-west1")
        vc = PhysicalNode("video_caption", "GCP", "us-west1", "Gemini 2.5 Pro")
        qry = PhysicalNode("query", "GCP", "us-east1", "Gemini 2.5 Flash")
        u = workflow_utility(sd, vs, vc, qry)
    """
    if len(nodes) != 4:
        raise ValueError(f"workflow_utility expects 4 PhysicalNode arguments, got {len(nodes)}")
    return sum(physical_node_utility(n) for n in nodes) / 4.0


if __name__ == "__main__":
    # Example: shot_detection & video_split on GCP us-west1; video_caption Gemini Pro; query Gemini Flash on another region.
    node_shot_detection = PhysicalNode("shot_detection", "GCP", "us-west1")
    node_video_split = PhysicalNode("video_split", "GCP", "us-west1")
    node_video_caption = PhysicalNode("video_caption", "GCP", "us-west1", "Gemini 2.5 Pro")
    node_query = PhysicalNode("query", "GCP", "us-east1", "Gemini 2.5 Flash")

    pipeline = (node_shot_detection, node_video_split, node_video_caption, node_query)
    print("mean utility:", workflow_utility(*pipeline))
