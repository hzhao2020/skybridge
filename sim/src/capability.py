"""Capability scores μ_k for logical-optimal and related baselines (SkyAPI Evaluation.md)."""

from __future__ import annotations

from src.schemas import Endpoint

# Raw benchmark scores (scaled to [0, 1] in SkyAPI Evaluation.md).
_SHOT_BY_PROVIDER: dict[str, float] = {
    "GCP": 0.95,
    "AWS": 0.95,
    "Aliyun": 0.95,
}
_MAX_SHOT = max(_SHOT_BY_PROVIDER.values())

_SAMPLE = 1.0
_SPLIT_SAMPLE = 1.0

# Claude 4.5 tiers mapped to caption / Q&A columns (same ranking as Nova Pro / Flash in the doc).
_CLAUDE_CAPTION_RAW: dict[str, float] = {
    "Claude Haiku 4.5": 0.652,
    "Claude Sonnet 4.5": 0.778,
    "Claude Opus 4.5": 0.778,
}
_CLAUDE_QUERY_RAW: dict[str, float] = {
    "Claude Haiku 4.5": 0.622,
    "Claude Sonnet 4.5": 0.416,
    "Claude Opus 4.5": 0.690,
}
_MAX_CAPTION = max(_CLAUDE_CAPTION_RAW.values())
_MAX_QUERY = max(_CLAUDE_QUERY_RAW.values())

_OCR_BY_PROVIDER: dict[str, float] = {
    "GCP": 0.958,
    "AWS": 0.942,
    "Aliyun": 0.948,
    "Azure": 0.958,
}
_LABEL_BY_PROVIDER: dict[str, float] = {
    "GCP": 0.93,
    "AWS": 0.89,
    "Aliyun": 0.9316,
    "Azure": 0.93,
}
_SPEECH_BY_PROVIDER: dict[str, float] = {
    "GCP": 0.973,
    "AWS": 0.958,
    "Aliyun": 0.963,
    "Azure": 0.973,
}
_MAX_OCR = max(_OCR_BY_PROVIDER.values())
_MAX_LABEL = max(_LABEL_BY_PROVIDER.values())
_MAX_SPEECH = max(_SPEECH_BY_PROVIDER.values())

# Database: no separate utility row in the evaluation table — neutral capability.
_DATABASE = 1.0


def endpoint_capability_mu(endpoint: Endpoint) -> float:
    """
    Normalized capability μ_k in [0, 1] for one endpoint at the required quality level.
    """
    op = endpoint.logical_operation
    if op == "Shot Detection":
        raw = _SHOT_BY_PROVIDER.get(endpoint.provider)
        if raw is None:
            raise KeyError(f"No shot_detection μ for provider={endpoint.provider!r}")
        return raw / _MAX_SHOT
    if op == "Sample":
        return _SAMPLE
    if op == "Split & Sample":
        return _SPLIT_SAMPLE
    if op == "Frame Caption":
        if not endpoint.model_name:
            raise ValueError("Frame Caption endpoint requires model_name")
        raw = _CLAUDE_CAPTION_RAW.get(endpoint.model_name)
        if raw is None:
            raise KeyError(f"No caption μ for model={endpoint.model_name!r}")
        return raw / _MAX_CAPTION
    if op == "Reason":
        if not endpoint.model_name:
            raise ValueError("Reason endpoint requires model_name")
        raw = _CLAUDE_QUERY_RAW.get(endpoint.model_name)
        if raw is None:
            raise KeyError(f"No Reason μ for model={endpoint.model_name!r}")
        return raw / _MAX_QUERY
    if op == "OCR":
        raw = _OCR_BY_PROVIDER.get(endpoint.provider)
        if raw is None:
            raise KeyError(f"No OCR μ for provider={endpoint.provider!r}")
        return raw / _MAX_OCR
    if op == "Label Detection":
        raw = _LABEL_BY_PROVIDER.get(endpoint.provider)
        if raw is None:
            raise KeyError(f"No label_detection μ for provider={endpoint.provider!r}")
        return raw / _MAX_LABEL
    if op == "Speech Transcription":
        raw = _SPEECH_BY_PROVIDER.get(endpoint.provider)
        if raw is None:
            raise KeyError(f"No speech_transcription μ for provider={endpoint.provider!r}")
        return raw / _MAX_SPEECH
    if op in ("Database", "Temporal Grounding"):
        return _DATABASE
    raise ValueError(f"Unknown logical operation for capability: {op!r}")
